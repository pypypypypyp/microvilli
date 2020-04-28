#!/usr/bin/env python
#coding: utf-8

import warnings

with warnings.catch_warnings():
        warnings.simplefilter("ignore")
warnings.simplefilter("ignore")

import os
import optparse
import copy
from EMAN2 import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve, correlate2d
from scipy.spatial.transform import Rotation
from scipy.ndimage import *
from scipy.ndimage import gaussian_laplace as LOG
from scipy.interpolate import *
from scipy.spatial import distance_matrix
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import hessian, laplace
from sklearn.cluster import KMeans
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
        parser.add_option('--apix', dest='apix', action='store', type='float', default=4.2625, help='pixel size of an UNBINNED tomogram')
        parser.add_option('--bin', dest='bin', action='store', type='int', default=4, help='binning factor')
        parser.add_option('--rise', dest='rise', action='store', type='int', default=27.8, help='rise of the filament in Angstroms')
        parser.add_option('--thres', dest='thres', action='store', type='float', default=0.05, help='thresholding value to detect filaments')
        parser.add_option('--r_mem', dest='r_mem', action='store', type='int', default=27, help='radius of membrane')
        parser.add_option('--largeness', dest='largeness', action='store', type='int', default=10, help='the minimal value of the area of projected filaments')
        parser.add_option('--average', dest='average', action='store', type='int', default=40, help='how many images are averaged to calculate the center of a filament cross section?')
        parser.add_option('--thickness', dest='thickness', action='store', type='int', default=25, help='thickness of y-slices')
        parser.add_option('--overlap', dest='overlap', action='store', type='int', default=10, help='overlap of averaged slices')
        parser.add_option('--cutoff_vol_top', dest='cutoff_vol_top', action='store', type='int', default=0, help='length of volume cut-off from the top')
        parser.add_option('--cutoff_vol_bottom', dest='cutoff_vol_bottom', action='store', type='int', default=0, help='length of volume cut-off from the bottom')
        parser.add_option('--r_actin', dest='r_actin',action='store', type='float', default=1.8, help='diameter of actin')
        parser.add_option('--thres_cutoff_mask', dest='thres_cutoff_mask', action='store', type='float', default=3.8, help='threshold of cut-off mask')
        parser.add_option('--thres_dis', dest='thres_dis', action='store', type='float', default=3, help='separated blobs within in-plane distance r_actin*thres_dis be connected as one actin')
        parser.add_option('--patch_size', dest='patch_size', action='store', type='int', default=5, help='patch size')
        parser.add_option('--asym_unit_step', dest='asym_unit_step', action='store', type='int', default=6, help='the interval to segment particles (expresed in the number of asymmetrical units)')
        parser.add_option('--classify', dest='classify', action='store', type='string', default='all', help='expressed in the format of "K-k", where K is the number of class and k is the class-id to be saved')
        parser.add_option('--polarity', dest='polarity', action='store', type='int', default=1, help='RotAxes will be saved as the difference vector from the present to the next particle multiplied by this polarity value')
        parser.add_option('--continue', dest='continue', action='store', type='string', default=None, help='continue from an already straightened volume')
        parser.add_option('--debug', dest='debug', action='store_true', default=False, help='debug mode')
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
        # input volume should have black objects with a white background
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
        mask = np.abs(vecty) < np.cos(np.deg2rad(ALLOWED_ANGLE)) # make sure to remove unnecessary signals oriented close to the y-axis
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

def normalize(data, verbose=True):
        if verbose: print "Normalizing the image ..."
        ret = data - data.mean()
        ret /= data.std()
        if verbose: print " -> complete!"
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
        rad = int(outer_radius)+15
        axes = []
        x_size = data.shape[1]
        try: os.remove("patch.mrcs")
        except: pass
        try: os.remove("cor.mrcs")
        except: pass
        for i, twod_axis in enumerate(twod_axes):
                print " -> calculating for object #%d ..."%(i+1)
                centers = []
                y_positions, z_positions = twod_axis
                for j in range(len(y_positions)/ave):
                        ys = y_positions[j*ave:(j+1)*ave]
                        zs = z_positions[j*ave:(j+1)*ave]
                        z = int(zs.mean())
                        if z-rad < 0 or z+rad < data.shape[1]: continue
                        patch = data[ys[0]:ys[-1]+1, :, z-rad:z+rad]
                        patch = cutoff_noise(patch, c_value=3, verbose=False).sum(axis=0)
                        patch = patch - patch.mean()
                        patch[patch < 0] = 0
                        EMNumPy.numpy2em(patch).write_image("patch.mrcs", -1)
                        center_x, center_z, ccc = find_center(patch, outer_radius/2)
                        center_z = z-rad+center_z
                        center_y = ys.mean()
                        if outer_radius*BSIZE < center_x and center_x < data.shape[1] - outer_radius*BSIZE:
                                centers.append((center_x, center_y, center_z))
                if len(centers) < 3: 
                        print " -> discarding this object (#%d), because the majority of points are too close to the edges!"%(i+1)
                        continue
                centers = np.array(centers)
                centers = np.array(centers).transpose()
                axes.append(centers)
        print " -> complete!"
        return axes

def find_center(data, limit):
        data -= data.mean()
        data_inv = data[::-1, ::-1]
        cor = correlate2d(data, data_inv, mode="same")
        EMNumPy.numpy2em(cor).write_image("cor.mrcs", -1)
        cor = cor[int(cor.shape[0]/2-limit*2):int(cor.shape[0]/2+limit*2), int(cor.shape[1]/2-limit*2):int(cor.shape[1]/2+limit*2)]
        max_pos = np.argmax(cor)
        cor_Y, cor_X = cor.shape
        dy = (max_pos/cor_X - cor_Y/2)/2.
        dx = (max_pos%cor_X - cor_X/2)/2.
        center_y = data.shape[0]/2 + dy
        center_x = data.shape[1]/2 + dx
        plt.imshow(data)
        return center_y, center_x, cor.max()

def create_straight_volume(data, axes, outer_radius):
        vols = []
        centers = []
        od = int(outer_radius*BSIZE)
        for axis in axes:
                x, y, z = axis
                fx = interp1d(y, x)
                fz = interp1d(y, z)
                newy = np.arange(y[0], y[-1])
                newx = fx(newy)
                newz = fz(newy)
                vol = np.zeros((len(newy), od*2, od*2))
                center = []
                for i, y in enumerate(newy):
                        y = int(y)
                        newx_i = int(newx[i])
                        newz_i = int(newz[i])
                        vol[i] = data[y, newx_i-od:newx_i+od, newz_i-od:newz_i+od]
                        center.append([y, newx_i, newz_i])
                vols.append(vol)
                centers.append(center)
        return vols, centers

def remove_near_edge_signals(meij, outer_radius):
        print "Removing near-edge signals ..."
        y_size, x_size = meij.shape
        mask = np.ones((y_size, x_size)).astype(np.bool)
        x_offset = outer_radius*2
        y_offset = outer_radius*2
        x_size -= outer_radius * 4
        y_size -= outer_radius * 4
        mask[int(y_offset):int(y_offset+y_size), int(x_offset):int(x_offset+x_size)] = False
        
        meij[mask] = 0
        print " -> complete!"
        return meij

def save_volumes_and_centers(vols, centers):
        print "Saving the volumes ..."
        volnames = []
        for i, vol in enumerate(vols):
                volname = "%s.vol%d.mrc"%(sys.argv[2], i)
                volnames.append(volname)
                save(volname, vol)
                cent = "\n".join(["%f %f %f"%(pos[2], pos[1], pos[0]) for pos in centers[i]])
                open(volname+".center.txt", "w").write(cent)
        print " -> complete!"
        return volnames

def make_slices(data, thickness, overlap, cutoff_vol_top, cutoff_vol_back):
        z = data.shape[0]
        slices = (z-overlap-cutoff_vol_top-cutoff_vol_back) / (thickness-overlap)
        stack = []
        z_values = []
        for i in range(slices):
                zz = thickness*i + thickness/2 - overlap*i + cutoff_vol_top
                sl = np.array(np.sum(data[int(zz-thickness/2):int(zz+thickness/2), :, :], axis=0))
                stack.append(sl)
                z_values.append(zz)
        return stack, z_values

def detect_blobs(data, r_actin, r_mem, thres_cutoff, patch_size):
        # apply LoG filter and roughly detect blobs
        data_filtered = -LOG(data, r_actin)
        ave = data_filtered.mean()
        std = data_filtered.std()
        thres = ave + std * thres_cutoff
        mask_cutoff = data_filtered < thres
        data_masked = copy.deepcopy(data_filtered)
        data_masked[mask_cutoff] = 0
        # pick up all the blobs
        labels, n = label(data_masked)
        # calculate center_of_mass and size for each blob
        centers = center_of_mass(data_masked, labels, range(1,n+1))
        size_of_blobs = np.bincount(labels.flatten())[1:]
        size_of_blobs = [[size] for size in size_of_blobs]
        # if a blob is too large, raise the threshold and try to separate it into two or three blobs
        blob_size_thres = 10
        centers_new = []
        for i in range(n):
                patch = data_filtered[int(centers[i][0]-patch_size):int(centers[i][0]+patch_size), int(centers[i][1]-patch_size):int(centers[i][1]+patch_size)]
                new_centers = [centers[i]]
                while np.min(size_of_blobs[i]) > blob_size_thres:
                        thres += std/5
                        mask_cutoff_patch = patch < thres
                        patch[mask_cutoff_patch] = 0
                        labels_patch, n_patch = label(patch)
                        if n_patch == 0:
                                break
                        else:
                                new_centers = center_of_mass(patch, labels_patch, range(1,n_patch+1))
                                for j, pos in enumerate(new_centers):
                                        x = centers[i][0] + new_centers[j][0] - patch_size
                                        y = centers[i][1] + new_centers[j][1] - patch_size
                                        new_centers[j] = (x, y)
                        size_of_blobs[i] = np.bincount(labels_patch.flatten())[1:]
                centers_new += new_centers
        ##if a pair of blobs are too close, replace with midpoint
        centers = np.array(centers_new)
        dist = distance_matrix(centers, centers)
        dist[dist == 0] = np.inf
        while np.min(dist) < r_actin*2:
                too_close = np.where(dist < r_actin*2)
                too_close_pairs = []
                for i in range(too_close[0].size):
                        p1 = too_close[0][i]
                        p2 = too_close[1][i]
                        if p1 < p2: too_close_pairs.append([p1, p2, dist[p1, p2]])
                too_close_pairs = np.array(too_close_pairs)
                too_close_pairs = sorted(too_close_pairs, key=lambda x:x[2])
                centers = list(centers)
                for pair in too_close_pairs:
                        if centers[p1][0] > 0 and centers[p2][0] > 0:
                                new_x = (centers[p1][0] + centers[p2][0])/2
                                new_y = (centers[p1][1] + centers[p2][1])/2
                                centers.append((new_x, new_y))
                                centers[p1] = (-1, -1)
                                centers[p2] = (-1, -1)
                centers = np.array(centers)
                centers = centers[np.sum(centers, axis=1) > 0]
                dist = distance_matrix(centers, centers)
                dist[dist == 0] = np.inf
        return data_masked, centers

def connect_blobs(centers, r_actin, thres_dis, slices_z, size):
        print "Connecting the blobs ..."
        actin_censl = [[[i, 0]] for i in range(len(centers[0]))]
        actin_xysl = {}
        actin_xyz = {}
        # first of all, roughly connect
        print " -> roughly connecting the blobs to make up filaments ..."
        for i in range(len(centers)-1):
                dist = distance_matrix(centers[i], centers[i+1])
                connect_to = np.argsort(dist, axis=1)[:, 0] # [0, 2, 1, 3] means #0->#0, #1->2, #2->1, #3->#3
                connection_distance = np.sort(dist, axis=1)[:, 0]
                connect_from = [-1]*len(centers[i+1]) # [0, 2, 1, 3] means #0->#0, #2->#1, #1->#2, and #3->#3
                for connect_from_ind, connect_to_ind in enumerate(connect_to):
                        if connection_distance[connect_from_ind] > r_actin * thres_dis:
                                # if the closest blob is too far
                                connect_to[connect_from_ind] = -1
                                continue
                        if connect_from[connect_to_ind] != -1 and connection_distance[connect_from_ind] < connection_distance[connect_from[connect_to_ind]]:
                                # if already connected but closer than the previous one
                                connect_to[connect_from[connect_to_ind]] = -1
                                connect_from[connect_to_ind] = connect_from_ind
                        elif connect_from[connect_to_ind] == -1:
                                connect_from[connect_to_ind] = connect_from_ind
                for j, line in enumerate(actin_censl):
                        if line[-1][1] == i:
                                if connect_to[line[-1][0]] != -1: # if a blob that can be connected to exists
                                        actin_censl[j].append([connect_to[line[-1][0]], i+1])
                # for the blobs to which no blob could be connected
                for connect_to_ind, connect_from_ind in enumerate(connect_from):
                        if connect_from_ind == -1: actin_censl.append([[connect_to_ind, i+1]])

        # if a filament appeas in 2 or 3 slices ahead and it seems that both constitute the same filament, connect
        print " -> connecting separate short filaments into a long filament ..."
        actin_start_xy = np.array([centers[line[0][1]][line[0][0]] for line in actin_censl])
        actin_start_sl = np.array([[0, line[0][1]] for line in actin_censl])
        actin_stop_xy = np.array([centers[line[-1][1]][line[-1][0]] for line in actin_censl])
        actin_stop_sl = np.array([[0, line[-1][1]] for line in actin_censl])
        dist_stop_start_sl = distance_matrix(actin_stop_sl, actin_start_sl)
        dist_stop_start_xy = distance_matrix(actin_stop_xy, actin_start_xy)
        mask_sl = (2 <= dist_stop_start_sl) * (dist_stop_start_sl <= 4)
        mask_xy = dist_stop_start_xy <= r_actin * thres_dis
        y, x = np.meshgrid(np.arange(mask_sl.shape[0]), np.arange(mask_sl.shape[1]))
        mask_avoid_doublecount = y > x
        mask = mask_sl * mask_xy * mask_avoid_doublecount
        connect_from, connect_to = np.where(mask)
        for connect_from_ind, connect_to_ind in zip(connect_from, connect_to):
                if actin_censl[connect_from_ind] is not None and actin_censl[connect_to_ind] is not None:
                        actin_censl.append(actin_censl[connect_from_ind]+actin_censl[connect_to_ind])
                        actin_censl[connect_from_ind] = None
                        actin_censl[connect_to_ind] = None
        actin_censl_new = []
        for line in actin_censl:
                if line is not None: actin_censl_new.append(line)
        actin_censl = actin_censl_new
        
        ## make actin_xysl
        for l in range(len(actin_censl)):
                actin_xysl[l] = []
                for position in actin_censl[l]:
                        actin_xysl[l].append((centers[position[1]][position[0]][0], centers[position[1]][position[0]][1], position[1]))
        ##make actin_xyz
        for m in range(len(actin_censl)):
                actin_xyz[m] = []
                for position in actin_censl[m]:
                        actin_xyz[m].append((centers[position[1]][position[0]][0]-size/2, centers[position[1]][position[0]][1]-size/2, slices_z[position[1]]))
        print " -> complete!"
        return actin_censl, actin_xysl, actin_xyz

def track(filename, params):
        e = EMData(filename)
        data = EMNumPy.em2numpy(e)
        # make averaged slices
        slices, slice_z_coords = make_slices(data, params['thickness'], params['overlap'], params['cutoff_vol_top'], params['cutoff_vol_bottom'])
        slices_log = []
        centers_log = []
        ns = []
        os.remove("slices.mrcs")
        # for each slice, 
        print "Detecting blobs in each slice ..."
        for slice in slices:
                # cut-off noise and normalize
                slice = cutoff_noise(slice, verbose=False)
                slice = normalize(slice, verbose=False)
                size = slice.shape[0]
                # apply circular mask to remove the signals outside of the membrane
                x, y = np.meshgrid(np.arange(-size/2, size/2), np.arange(-size/2, size/2))
                out_of_membrane = np.sqrt(x**2 + y**2) >= params['r_mem']
                slice[out_of_membrane] = 0
                EMNumPy.numpy2em(copy.deepcopy(slice)).write_image("slices.mrcs", -1)
                slice_log, centers = detect_blobs(slice, params['r_actin'], params['r_mem'], params['thres_cutoff_mask'], params['patch_size'])
                slices_log.append(slice_log)
                centers_log.append(centers)
                ns.append(len(centers))
        print " -> complete!"

        actin_censl, actin_xysl, actin_xyz = connect_blobs(centers_log, params['r_actin'], params['thres_dis'], slice_z_coords, size)
        if params["debug"]:
                fall = open("all.txt", "w")
                for key in actin_xysl.keys():
                        f = open('xysl_vol%d.txt'%key, 'w')
                        for position in actin_xysl[key]:
                                f.write('%f %f %d\n'%(position[1], position[0], position[2]))
                                fall.write('%f %f %d\n'%(position[1], position[0], position[2]))
                        f.close()
                fall.close()
        return actin_xyz

def classify_filaments(xyz, K, k, params):
        print "Classifying the filaments according to the distance from the center ..."
        filament_offaxis_distance = []
        for filament_id, filament_coords in xyz.items():
                dist_from_center = np.array([np.sqrt(np.sum(np.array(coord[:2])**2)) for coord in filament_coords])
                dist_from_center = np.sum(dist_from_center)/dist_from_center.size
                filament_offaxis_distance.append(dist_from_center)
        filament_offaxis_distance = np.array(filament_offaxis_distance)
        npdata = np.array([[0]*len(filament_offaxis_distance), filament_offaxis_distance])
        npdata = npdata.transpose()
        pred = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=0).fit_predict(npdata)
        for cls, point in zip(pred, npdata):
                if cls == 1:
                        plt.scatter(0,point[1],color='r')
                elif cls == 2:
                        plt.scatter(0,point[1],color='g')
                else:
                        plt.scatter(0,point[1],color='b')
        plt.savefig("cls.png")
        res = []
        for i in range(K): res.append([])
        i = 0
        for cls, point in zip(pred, npdata):
                res[cls].append([point[1], i])
                i += 1
        res = sorted(res, key=lambda x:x[0][0])
        res = np.array(res[k])
        save_indices = res[:, 1]
        save_indices = map(int, save_indices)
        print " -> filament %s will be saved ..."%(",".join(map(str, save_indices)))
        saved = {}
        for save_index in save_indices:
                saved[save_index] = xyz[save_index]
        print " -> complete!"
        return saved

def save_coordinates(xyz, centers, volname, params):
        # xyz -> {0: [[x00, y00, z00], [x01, y01, z01], ...]], 1:[[x10, y10, z10], [x11, y11, z11], ...]}
        # centers: [[X0, Y0, Z0], [X1, Y1, Z1], ...]
        print "Saving the results ..."
        APIX = params["apix"]
        BIN = params["bin"]
        APIX *= BIN
        step = params["rise"] * params["asym_unit_step"] / APIX
        coords_all = []
        yaxes_all = []
        motl_all = []
        xyz_global = copy.deepcopy(xyz)
        for filament_id, filament_coords in xyz.items():
                dist_from_top = [0]
                # estimate global coordinates in the tomogram volume
                for i, coord in enumerate(filament_coords):
                        y, x, z = coord
                        X, Y, Z = centers[int(z)]
                        x_new, y_new, z_new = x+X, y+Y, Z
                        xyz_global[filament_id][i] = [x_new, y_new, z_new]
                        if i > 0: 
                                x_prev, y_prev, z_prev = xyz_global[filament_id][i-1]
                                dist_add = np.sqrt((x_new-x_prev)**2+(y_new-y_prev)**2+(z_new-z_prev)**2)
                                dist_from_top.append(dist_from_top[-1]+dist_add)
                total_length = dist_from_top[-1]
                n_points = int(total_length / step)
                if n_points <= 2: continue
                points_xyz = []
                slice_i = 0
                # determine the output coordinates at regular intervals (=step), y-axes of the particles, and the MOTL angles
                for i in range(n_points):
                        # output coordinates
                        while not (dist_from_top[slice_i] <= i*step < dist_from_top[slice_i+1]):
                                slice_i += 1
                        delta = dist_from_top[slice_i+1] - dist_from_top[slice_i]
                        prev_global_coord = np.array(xyz_global[filament_id][slice_i])
                        next_global_coord = np.array(xyz_global[filament_id][slice_i+1])
                        l = i*step - dist_from_top[slice_i]
                        global_coord = prev_global_coord + (next_global_coord-prev_global_coord) * l/delta
                        global_coord *= BIN # make sure to multiply binning factor to make the coordinates global
                        coords_all.append(global_coord)
                        #f.write(" ".join(map(str, global_coord))+"\n")
                        if i > 0:
                                # y-axes (Using polarity=1 here. Polarity is applied later!)
                                yaxis = global_coord - coords_all[-2]
                                yaxes_all.append(yaxis)
                                # initial motive list
                                prev_local_coord = np.array(xyz[filament_id][slice_i])
                                next_local_coord = np.array(xyz[filament_id][slice_i+1])
                                local_coord = prev_local_coord + (next_local_coord-prev_local_coord) * l/delta
                                y, x, z = local_coord
                                xaxis_inplane = -np.array([x*100, y*100, 0])
                                yaxis = np.array(yaxis)
                                xaxis = xaxis_inplane - np.dot(xaxis_inplane, yaxis)/np.linalg.norm(yaxis)**2 * yaxis
                                zaxis = np.cross(xaxis, yaxis)
                                xaxis = xaxis / np.linalg.norm(xaxis)
                                yaxis = yaxis / np.linalg.norm(yaxis)
                                zaxis = zaxis / np.linalg.norm(zaxis)
                                mat = np.array([xaxis, yaxis, zaxis])
                                rot = Rotation.from_dcm(mat)
                                motl = rot.as_euler("ZXZ")/np.pi*180
                                motl_all.append(motl)
                yaxes_all.append(yaxis)
                motl_all.append(motl)
        yaxes_all = np.array(yaxes_all) * params["polarity"] # apply polarity
        f_name = "%s.filament_%s.coords.txt"%(volname, params["classify"])
        fy_name = "%s.filament_%s.RotAxes.txt"%(volname, params["classify"])
        fm_name = "%s.filament_%s.MOTL.csv"%(volname, params["classify"])
        print " -> coordinates are written in %s"%f_name
        print " -> RotAxes are written in %s (using polarity=%d)"%(fy_name, params["polarity"])
        print " -> initial MOTL angles (ZXZ rotation) are written in %s"%fm_name
        f = open(f_name, "w")
        fy = open(fy_name, "w")
        fm = open(fm_name, "w")
        fm.write("CCC,reserved,reserved,pIndex,wedgeWT,NA,NA,NA,NA,NA,xOffset,yOffset,zOffset,NA,NA,reserved,EulerZ(1),EulerZ(3),EulerX(2),reserved,CREATED WITH %s\n"%sys.argv[0])
        print "Total number of points: %d"%(len(coords_all))
        for i in range(len(coords_all)):
                f.write(" ".join(map(str, coords_all[i]))+"\n")
                fy.write(",".join(map(str, yaxes_all[i]))+"\n")
                angles = motl_all[i]
                motl = "1,0,0,%d,0,0,0,0,0,0,0,0,0,0,0,0,%f,%f,%f,0\n"%(i+1,angles[0],angles[1],angles[2])
                fm.write(motl)
        f.close()
        fy.close()
        fm.close()
        print " -> complete!"

def straighten(lp_map_name, nlp_map_name, params):
        # constants
        outer_radius = params["r_mem"]*sqrt(2)
        # load the volume
        data = load_volume(lp_map_name)
        # create a projection image
        proj = create_projection(data, axis=1) # along x-axis
        # cut-off noise and normalize
        proj = cutoff_noise(proj, c_value=3)
        proj = normalize(proj)
        save("proj.mrc", proj)
        # meijering filter
        meij, vect = meijering(proj, outer_radius)
        # objects must not include near-edge regions
        meij = remove_near_edge_signals(meij, outer_radius)
        # remove small objects
        meij = remove_small_objects(meij, outer_radius, largeness=params["largeness"], thres=params["thres"])
        # remove objects too close to edges
        meij = remove_near_edge_objects(meij, outer_radius)
        # find the filament axis of each object in the projection image
        twod_axes = find_filament_axis(proj, meij, vect, outer_radius)
        plt.imshow(proj)
        for obj_axis in twod_axes:
                x, y = obj_axis
                plt.scatter(y, x)
        plt.savefig("meij.png")
        # determine the  axis of each object in 3D
        axes = find_3D_axis(data, twod_axes, outer_radius, ave=params["average"])
        # create straight volumes
        data = load_volume(nlp_map_name)
        vols, centers = create_straight_volume(data, axes, outer_radius)
        # save!
        volnames = save_volumes_and_centers(vols, centers)
        return volnames

def main():
        # setup
        params = setupParserOptions()
        lp_map_name, nlp_map_name = sys.argv[1:3]
        # from the beginning
        if params["continue"] is None:
                # straighten
                volnames = straighten(lp_map_name, nlp_map_name, params)
                # process which volume?
                print "The volumes are saved in"
                for i in range(len(volnames)):
                        print " vol #%d: %s"%(i, volnames[i])
                print "Which volumes do you want to process? Make sure to check the output volumes using 3dmod!"
                print " (you shold type the id number of the volumes splitting by whitespace, e.g., '0 1 3'.)"
                vols = map(int, raw_input(" > ").split())
                for i in vols:
                        print " -> %s is selected."%volnames[i]
        else:
                vols = [0]
                volnames = [params["continue"]]
        # trace actins for each straightened volume
        for i in vols:
                print "Tracking actins on vol #%d: %s"%(i, volnames[i])
                xyz = track(volnames[i], params)
                centers = [map(float, line.split()) for line in open(volnames[i]+".center.txt").readlines()]
                if params["classify"] == "all": saved = xyz
                else:
                        K, k = map(int, params["classify"].split("-"))
                        saved = classify_filaments(xyz, K, k, params)
                save_coordinates(saved, centers, volnames[i], params)
                print

if __name__ == "__main__":
        main()
