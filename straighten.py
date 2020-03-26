#!/usr/bin/env python
#coding: utf-8

import os
import optparse
import copy
from EMAN2 import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve, correlate2d
from scipy.spatial.transform import Rotation
from scipy.ndimage import *
from scipy.interpolate import *
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import hessian, laplace
from scipy.misc import *
from pyfftw.builders import *
import pyfftw
import multiprocessing

pyfftw.config.NUM_THREADS = 4
pyfftw.interfaces.cache.enable()
ALLOWED_ANGLE = 45
BSIZE = 1.5

def setupParserOptions():
    parser = optparse.OptionParser(usage="Usage: %prog <filtered volume> <raw volume> [options]")
    parser.add_option('--thres', dest='thres', action='store', type='float', default=0.05, help='thresholding value to detect filaments')
    parser.add_option('--radius', dest='radius', action='store', type='int', default=39, help='radius of a filament')
    parser.add_option('--largeness', dest='largeness', action='store', type='int', default=15, help='how large a filament should be?')
    parser.add_option('--average', dest='average', action='store', type='int', default=40, help='how many images are averaged to calculate the center of a filament cross section?')
    options, args = parser.parse_args()
    if len(args) > 2: parser.error("Unknown command-line options: %s"%str(args))
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit()
    params = {}
    for i in parser.option_list:
        if isinstance(i.dest, str): params[i.dest] = getattr(options, i.dest)
    return params

def convolve2d(A, B, mode="same"):
        # determine output array shape (Y, X) and calculation array shape (L_calc, M_calc)
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        LK, MK = B.shape
        L, M = A.shape
        if mode == "same": Y, X = L, M
        elif mode == "valid": Y, X = L-LK+1, M-MK+1
        elif mode == "full": Y, X = L+LK-1, M+MK-1
        L_calc = L+LK-1
        M_calc = M+MK-1
        offset_Y = int(np.floor((L_calc-Y)/2))
        offset_X = int(np.floor((M_calc-X)/2))
        # pad with zeros
        a = np.zeros((L_calc, M_calc))
        b = np.zeros((L_calc, M_calc))
        a[:L, :M] = A
        b[:LK, :MK] = B
        # calculate
        fft_A_obj = rfftn(a, s=(L_calc, M_calc))
        fft_B_obj = rfftn(b, s=(L_calc, M_calc))
        ifft_obj = irfftn(fft_A_obj.output_array, s=(L_calc, M_calc))
        # calculate
        fft_mult = fft_A_obj(a) * fft_B_obj(b)
        ifft = ifft_obj(fft_mult)
        ifft = ifft[offset_Y:offset_Y+Y, offset_X:offset_X+X]
        return ifft

def load_volume(filename):
        # input volume should have white objects with black background
        print "Loading the volume: %s ..."%filename
        e = EMData(filename)
        data = EMNumPy.em2numpy(e)
        data = data.astype(np.float32)
        data *= -1
        data -= data.mean()
        print " -> complete!"
        return data

def cutoff_noise(array, c_value=3, verbose=True):
        if verbose: print "Cutting off noise ..."
        ave = array.mean()
        std = array.std()
        high = ave + std*c_value
        low = ave - std*c_value
        high_values = array > high
        low_values = array < low
        array[high_values] = high
        array[low_values] = low
        if verbose: print " -> complete!"
        return array


def save(filename, data):
        EMNumPy.numpy2em(data).write_image(filename)

def meijering(image, sigma, black_ridges=False):
        # sigma equals the radius of the ridge being emphasized
        print "Applying meijering filter ..."
        image = image.astype(np.float64)
        if black_ridges: image = -image
        value = np.zeros(image.shape)
        Hxx, Hxy, Hyy = hessian_matrix(image, sigma=sigma, order="rc")
        b1 = Hxx + Hyy
        b2 = Hxx - Hyy
        d = np.sqrt(4*Hxy*Hxy + b2*b2)
        L1 = (b1 + 2*d) / 3.0
        L2 = (b1 - 2*d) / 3.0
        vect1x = b2 + d
        vect2x = b2 - d
        vecty = 2 * Hxy
        vectx = np.array([vect1x, vect2x])
        sortedL, sortedvectx = sortbyabs(np.array([L1, L2]), auxiliary=vectx)
        L = sortedL[1]
        vectx = sortedvectx[0]
        vectlen = np.sqrt(vectx**2 + vecty**2)
        vectx /= vectlen
        vecty /= vectlen
        valL = np.where(L > 0, 0, L)
        mask = np.abs(vecty) < np.cos(np.deg2rad(ALLOWED_ANGLE)) # make sure to remove unnecessary signals oriented close to the z-axis
        valL[mask] = 0
        valL = divide_nonzero(valL, np.min(valL))
        vect = np.array([vectx, vecty])
        print " -> complete!"
        return valL, vect

def sortbyabs(array, axis=0, auxiliary=None):
        index = list(np.ix_(*[np.arange(i) for i in array.shape]))
        index[axis] = np.abs(array).argsort(axis)
        if auxiliary is None:
                return array[tuple(index)]
        else:
                return array[tuple(index)], auxiliary[tuple(index)]

def divide_nonzero(array1, array2, cval=1e-10):
        denom = np.copy(array2)
        denom[denom == 0] = cval
        return np.divide(array1, denom)

def create_projection(data, axis=1):
        print "Creating a projection image ..."
        ret = np.sum(data, axis=axis)
        print " -> complete!"
        return ret

def normalize(data):
        print "Normalizing the image ..."
        ret = data - data.mean()
        ret /= data.std()
        print " -> complete!"
        return ret

def remove_small_objects(meij, outer_radius, largeness=10, thres=0.05):
        print "Removing small objects ..."
        print " -> labelling the filtered image"
        meij[meij < thres] = 0
        labels, n = label(meij)
        object_ids = range(n+1)
        counts = np.bincount(labels.flatten())
        to_be_removed = counts < outer_radius * outer_radius * largeness
        for obj_id in object_ids:
                if to_be_removed[obj_id]:
                        labels[labels == obj_id] = 0
        meij[labels == 0] = 0
        print " -> complete!"
        return meij

def remove_near_edge_objects(meij, outer_radius):
        print "Removing near-edge objects ..."
        labels, n = label(meij)
        centers = center_of_mass(meij, labels, range(1, n+1))
        m = 3
        for i, center in enumerate(centers):
                if outer_radius*m < center[0] and center[0] < meij.shape[0]-outer_radius*m and outer_radius*m < center[1] and center[1] <  meij.shape[1]-outer_radius*n:
                        pass
                else:
                        labels[labels == i+1] = 0
        meij[labels == 0] = 0
        print " -> complete!"
        return meij

def FDAGK(sigma, theta, rho):
        height = int(4 * sigma * rho)
        width = int(4 * sigma)
        theta = np.deg2rad(theta)
        x, y = np.meshgrid(np.arange(-(width/2), width/2+1), np.arange(-(height/2), height/2+1))
        t1 = 1 * np.cos(theta) * (x*np.cos(theta) + y*np.sin(theta)) - rho**-2 * np.sin(theta) * (-x*np.sin(theta) + y*np.cos(theta))
        t2 = 1 * np.sin(theta) * (x*np.cos(theta) + y*np.sin(theta)) + rho**-2 * np.cos(theta) * (-x*np.sin(theta) + y*np.cos(theta))
        phi = x*t1 + y*t2
        AGK = 1./(2 * np.pi * rho * sigma**2) * np.exp(-phi/(2*sigma**2))
        FDAGK = -1 * ((x*np.cos(theta)+y*np.sin(theta)) / sigma**2) * AGK
        return FDAGK

def find_filament_axis(proj, meij, vect, outer_radius):
        print "Finding the axis for each object in the projection image ..."
        labels, n = label(meij)
        axes = [[] for i in range(n)]
        print " -> roughly estimating the central points of the object along the x-axis of the image ..."
        print " -> working on %d objects ..."%n
        deltas = [0] * n
        for x in range(meij.shape[0]):
                for obj_id in range(1, n+1):
                        # find the center of the object (obj_id) at this x-coordinate level
                        labels_x = labels[x]
                        meij_x = meij[x]
                        positions_x = np.arange(meij_x.size)
                        # pick up a region of the filtered image containing the object
                        labels_x_equals_obj_id = labels_x == obj_id
                        meij_x_id = meij_x[labels_x_equals_obj_id]
                        positions_x_id = positions_x[labels_x_equals_obj_id]
                        if len(meij_x_id) == 0: continue # if this level doesn't contain the object, move to the next object
                        filt = maximum_filter(meij_x_id, size=outer_radius)
                        peaks = np.where(filt == meij_x_id)[0]
                        peaks = positions_x_id[peaks]
                        # several peaks may be found, so the peak closest to the previous one is chosen
                        if len(axes[obj_id-1]) != 0:
                                prev_peak_pos = axes[obj_id-1][-1]
                                prev_peak_x = prev_peak_pos[0]
                                prev_peak_y = prev_peak_pos[1]
                                peaks_diff = np.abs(peaks - prev_peak_y)
                                peak_index = np.where(peaks_diff == np.min(peaks_diff))[0][0]
                                peak = peaks[peak_index]
                        else:
                                peak = peaks[0]
                        axes[obj_id-1].append((x, peak))
        for i in range(len(axes)):
                axes[i] = np.array(axes[i]).transpose()
        print " -> complete!"
        return axes

def find_3D_axis(data, twod_axes, outer_radius, ave=10):
        print "Finding the axis for each object in the volume data ..."
        rad = int(outer_radius*2)
        axes = []
        for i, twod_axis in enumerate(twod_axes):
                print " -> calculating for object #%d ..."%(i+1)
                centers = []
                y_positions, z_positions = twod_axis
                #f = open("test.txt", "w")
                for j in range(len(y_positions)/ave):
                        ys = y_positions[j*ave:(j+1)*ave]
                        zs = z_positions[j*ave:(j+1)*ave]
                        z = int(zs.mean())
                        if z-rad < 0 or z+rad < data.shape[1]: continue
                        patch = data[ys[0]:ys[-1]+1, :, z-rad:z+rad].sum(axis=0)
                        patch = cutoff_noise(patch, c_value=5, verbose=False)
                        EMNumPy.numpy2em(patch).write_image("test.mrcs", -1)
                        center_x, center_z = find_center(patch, outer_radius/2)
                        center_z = z-rad+center_z
                        center_y = ys.mean()
                        if outer_radius*BSIZE < center_x and center_x < data.shape[1] - outer_radius*BSIZE:
                                centers.append((center_x, center_y, center_z))
                                #f.write("%d %d %d\n"%(center_z, center_x, center_y))
                #f.close()
                centers = np.array(centers).transpose()
                axes.append(centers)
        print " -> complete!"
        return axes

def find_center(data, limit):
        data -= data.min()
        thres = data.mean() - data.std()
        data[data < thres] = thres
        data -= data.mean()
        data_inv = data[::-1, ::-1]
        cor = correlate2d(data, data_inv, mode="same")
        cor = cor[cor.shape[0]/2-limit*2:cor.shape[0]/2+limit*2, cor.shape[1]/2-limit*2:cor.shape[1]/2+limit*2]
        max_pos = np.argmax(cor)
        cor_Y, cor_X = cor.shape
        dy = (max_pos/cor_X - cor_Y/2)/2.
        dx = (max_pos%cor_X - cor_X/2)/2.
        center_y = data.shape[0]/2 + dy
        center_x = data.shape[1]/2 + dx
        plt.imshow(data)
        return center_y, center_x

def create_straight_volume(data, axes, outer_radius):
        vols = []
        od = int(outer_radius*BSIZE)
        for axis in axes:
                x, y, z = axis
                fx = interp1d(y, x)
                fz = interp1d(y, z)
                newy = np.arange(y[0], y[-1])
                newx = fx(newy)
                newz = fz(newy)
                vol = np.zeros((len(newy), od*2, od*2))
                for i, y in enumerate(newy):
                        y = int(y)
                        newx_i = int(newx[i])
                        newz_i = int(newz[i])
                        vol[i] = data[y, newx_i-od:newx_i+od, newz_i-od:newz_i+od]
                vols.append(vol)
        return vols

def remove_near_edge_signals(meij, outer_radius):
        print "Removing near-edge signals ..."
        y_size, x_size = meij.shape
        mask = np.ones((y_size, x_size)).astype(np.bool)
        x_offset = outer_radius*2
        y_offset = outer_radius*2
        x_size -= outer_radius * 4
        y_size -= outer_radius * 4
        mask[y_offset:y_offset+y_size, x_offset:x_offset+x_size] = False
        
        meij[mask] = 0
        print " -> complete!"
        return meij

def save_volumes(vols):
        print "Saving the volumes ..."
        for i, vol in enumerate(vols):
                save("%s.vol%d.mrc"%(sys.argv[2], i), vol)
        print " -> complete!"

def main():
        # params
        params = setupParserOptions()
        # constants
        outer_radius = params["radius"]
        # load the volume
        data = load_volume(sys.argv[1])
        # create a projection image
        proj = create_projection(data, axis=1) # along x-axis
        save("proj.mrc", proj)
        # cut-off noise and normalize
        proj = cutoff_noise(proj, c_value=3)
        proj = normalize(proj)
        # meijering filter
        meij, vect = meijering(proj, outer_radius)
        # objects must not include near-edge regions
        meij = remove_near_edge_signals(meij, outer_radius)
        save("meij.mrc", meij)
        # remove small objects
        meij = remove_small_objects(meij, outer_radius, largeness=params["largeness"], thres=params["thres"])
        # remove islands too close to edges
        meij = remove_near_edge_objects(meij, outer_radius)
        # finding the filament axis for each object in the projection image
        twod_axes = find_filament_axis(proj, meij, vect, outer_radius)
        plt.imshow(proj)
        for obj_axis in twod_axes:
                x, y = obj_axis
                plt.scatter(y, x)
        plt.savefig("meij.png")
        # determine three-dimensioal axis for each object
        axes = find_3D_axis(data, twod_axes, outer_radius, ave=params["average"])
        # create straight volumes
        data = load_volume(sys.argv[2])
        vols = create_straight_volume(data, axes, outer_radius)
        # save!
        save_volumes(vols)

if __name__ == "__main__":
        main()
