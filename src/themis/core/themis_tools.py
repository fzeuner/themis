#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 05 2025

@author: franziskaz

This toolbox contains useful functions

Version: 2025-3-27
"""


import matplotlib.pyplot as plt
from astropy.io import fits 
import numpy as np
from themis.ui.style_themis import style_aa
from matplotlib.ticker import (NullFormatter)

from scipy.signal import convolve2d

from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy.signal import windows
from scipy import ndimage
from scipy.ndimage import shift

from skimage.registration import phase_cross_correlation

from matplotlib import gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import gc
import warnings

from pathlib import Path

from themis.core import themis_data_reduction as tdr
from themis.core import data_classes as dct
#from scipy.ndimage import distance_transform_edt
from scipy.interpolate import griddata


from themis.datasets import themis_datasets_2025 as dst
from themis.core import cam_config as cc


#%%

class l1_data:
    def __init__(self, upper = '-V', lower = '+V'):   

        self.wvl = ''         # wavelength array
        self.ui = ''          # upper image
        self.li = ''          # lower image 
        self.upper_state = upper
        self.lower_state = lower
        
class v_data: #
    def __init__(self):
    
        self.wavelength = ''  # name
        self.method = ''
        self.file = ''
        self.wvl = ''         # wavelength array
        self.i = ''          # i image
        self.v = ''          # v image  
        self.date = ''


def calc_v_beamexchange(data):
    v = 0.25*(1.- data.li[0,:]/data.li[1,:]*data.ui[1,:]/data.ui[0,:])
    
    return(v)

def calc_i_beamexchange(data):
    i = 0.25*(data.li[0,:]+data.li[1,:]+data.ui[1,:]+data.ui[0,:])  
    return(i)

def beam_exchange(data):
    out_data = v_data()
    out_data.wavelength = data.wavelength
    out_data.file = data.file
    out_data.wvl = data.wvl
    out_data.date = data.date
    out_data.i = calc_i_beamexchange(data)
    out_data.v = calc_v_beamexchange(data)
    out_data.method = 'beam exchange'
    return(out_data)



def clean_bad_pixels(data, header):
    """
    Clean bad pixels in raw data frames.
    
    Identifies bad pixels as:
    - Pixels with value == header['BZERO'] (saturated/maximum value)
    - Pixels with negative values (< 0)
    
    Processing:
    - If a bad pixel is isolated (no bad neighbors within radius 4), interpolate it.
    - If any bad pixel has at least one neighboring bad pixel, reject the entire frame.
    
    Args:
        data: FramesSet or CycleSet containing raw frames (unscaled FITS data)
        header: FITS header containing BZERO keyword
        
    Returns:
        tuple: (cleaned_data, additional_keywords)
            - cleaned_data: Data with interpolated pixels and bad frames removed
            - additional_keywords: dict with statistics for header
    """

    
    # Get bad pixel values from header
    bzero = header.get('BZERO', None)
    
    if bzero is None:
        # No bad pixel value defined, return data unchanged
        return data, {}
    
    total_interpolated = 0
    total_rejected = 0
    frames_to_remove = []
    
    # Determine if we have FramesSet or CycleSet
    if isinstance(data, dct.FramesSet):
        frame_keys = list(data.keys())
    elif isinstance(data, dct.CycleSet):
        frame_keys = list(data.frames.keys())
    else:
        # Unknown type, return unchanged
        return data, {}
    
    # Process each frame
    for frame_key in frame_keys:
        if isinstance(data, dct.FramesSet):
            frame = data.get(frame_key)
        else:
            frame = data.frames[frame_key]
        
        frame_rejected = False
        frame_interpolated = 0
        
        # Process each half (upper, lower)
        for position in ['upper', 'lower']:
            half = frame.get_half(position)
            if half is None:
                continue
                
            image = half.data
            
            # Identify bad pixels: pixels at bzero (saturation) or negative values
            bad_mask = np.logical_or(image == bzero, image < 0)
            
            if not np.any(bad_mask):
                # No bad pixels in this half
                continue
            
            # Check for neighboring bad pixels using morphological dilation
            # Create a structure element for radius 4 neighborhood
            y, x = np.ogrid[-4:5, -4:5]
            structure = (x**2 + y**2 <= 4**2)
            
            # Dilate the bad pixel mask
            from scipy.ndimage import binary_dilation
            dilated_mask = binary_dilation(bad_mask, structure=structure)
            
            # Find bad pixels that have neighboring bad pixels
            # A pixel has a neighbor if (dilated_mask - original_mask) overlaps with original_mask
            neighbor_mask = np.logical_and(dilated_mask, bad_mask)
            # Check if any bad pixel touches another bad pixel
            for i in range(bad_mask.shape[0]):
                for j in range(bad_mask.shape[1]):
                    if bad_mask[i, j]:
                        # Check neighborhood (radius 4)
                        y_min = max(0, i - 4)
                        y_max = min(bad_mask.shape[0], i + 5)
                        x_min = max(0, j - 4)
                        x_max = min(bad_mask.shape[1], j + 5)
                        
                        neighborhood = bad_mask[y_min:y_max, x_min:x_max]
                        # If there's more than one bad pixel in neighborhood, reject frame
                        if np.sum(neighborhood) > 1:
                            frame_rejected = True
                            break
                if frame_rejected:
                    break
            
            if frame_rejected:
                break
            
            # If not rejected, interpolate isolated bad pixels
            if np.any(bad_mask):
                # Get coordinates of good and bad pixels
                good_mask = ~bad_mask
                y_coords, x_coords = np.mgrid[0:image.shape[0], 0:image.shape[1]]
                
                good_points = np.array([y_coords[good_mask], x_coords[good_mask]]).T
                good_values = image[good_mask]
                bad_points = np.array([y_coords[bad_mask], x_coords[bad_mask]]).T
                
                # Interpolate bad pixels using nearest neighbor (fast and simple)
                interpolated_values = griddata(good_points, good_values, bad_points, method='nearest')
                
                # Update the image
                image[bad_mask] = interpolated_values
                frame_interpolated += np.sum(bad_mask)
        
        if frame_rejected:
            frames_to_remove.append(frame_key)
            total_rejected += 1
        else:
            total_interpolated += frame_interpolated
    
    # Remove rejected frames
    if isinstance(data, dct.FramesSet):
        for frame_key in frames_to_remove:
            del data.frames[frame_key]
    else:  # CycleSet
        for frame_key in frames_to_remove:
            del data.frames[frame_key]
    
    # Prepare keywords for header
    additional_keywords = {
        'NBADPXIN': (total_interpolated, 'Number of bad pixels interpolated'),
        'NBADFRM': (total_rejected, 'Number of frames rejected (bad pixels)')
    }
    
    if total_rejected > 0:
        print(f"\033[91mWARNING\033[0m: {total_rejected} frame(s) rejected due to clustered bad pixels")
    if total_interpolated > 0:
        print(f"INFO: {total_interpolated} isolated bad pixel(s) interpolated")
    
    return data, additional_keywords


def z3denoise(image, method='poly_fit', rand=None, verbose=False):
    # Get image dimensions
    sz = image.shape
    xd = sz[0]
    yd = sz[1] 
 
    at = image.dtype  # Type of the image array

    # Handle the rand array
    rd = np.zeros(4, dtype=int)
    if rand is not None:
        rd += rand
    rdnull = np.all(rd == 0)

    # Handle method input
    method = method.lower().split() if isinstance(method, str) else method
    res = np.zeros((xd, yd), dtype=at)

    ok = False

    if method[0] == 'poly_fit':
        if verbose:
         print('Method: poly_fit')
        ok = True
        degx = int(method[1]) if len(method) > 1 else 7
        degy = int(method[2]) if len(method) > 2 else degx
        ix = np.arange(xd)
        iy = np.arange(yd)

   
        if xd > degx:
                    for y in range(yd):
                        ce = np.polyfit(ix[rd[0]:xd-rd[1]], image[rd[0]:xd-rd[1], y], degx)
                        if rdnull:
                            res[:, y] = np.polyval(ce, ix)
                        else:
                            for n in range(degx + 1):
                                res[:, y] += ce[n] * ix ** n

        if yd > degy:
                    for x in range(xd):
                        ce = np.polyfit(iy[rd[2]:yd-rd[3]], res[x, rd[2]:yd-rd[3]], degy)
                        if rdnull:
                            res[x, :] = np.polyval(ce, iy)
                        else:
                            res[x, :] = 0
                            for n in range(degy + 1):
                                res[x, :] += ce[n] * iy ** n

    elif method[0] == 'smooth':
        if verbose:
         print('Method: smooth')
        ok = True
        width = int(method[1]) if len(method) > 1 else 5


        if rdnull:
                    res[:, :] = ndimage.gaussian_filter(image[:, :], width)
        else:
                    sim = ndimage.gaussian_filter(image[rd[0]:xd-rd[1], rd[2]:yd-rd[3]], width)
                    res[:, :,] = ndimage.zoom(sim, (xd / sim.shape[0], yd / sim.shape[1]))
                    res[rd[0]:xd-rd[1], rd[2]:yd-rd[3]] = sim


    if ok:
        return(res)

    print(f'z3denoise: Method {method[0]} unknown!')
    return(None)




  
def get_pol_half_names(frame_name: str):
    """
    Given a polarization frame name like 'pQ' or 'mU', return the names for
    the upper and lower halves based on the sign.

    Args:
        frame_name (str): Name starting with 'p' or 'm', e.g., 'pQ', 'mU'

    Returns:
        tuple: (upper_name, lower_name)
    """
    if not frame_name or len(frame_name) < 2:
        raise ValueError("Frame name must be at least two characters long (e.g., 'pQ').")

    sign = frame_name[0]
    pol_axis = frame_name[1:]

    if sign == 'p':
        return (f"p{pol_axis}", f"m{pol_axis}")
    elif sign == 'm':
        return (f"m{pol_axis}", f"p{pol_axis}")
    else:
        raise ValueError("Frame name must start with 'p' or 'm'.")
  


def z3ccspectrum(yin, xatlas, yatlas, FACL=0.8, FACH=1.5, FACS=0.01, CUT=None, DERIV=None, CONT=None, SHOW=2):
    
    """


    ;+----------------------------------------------------------------------------
    ;
    ; F: Cross-correlates an observed spectrum with the fts atlas - translated to python
    ; 
    ; Calculates the least square difference in each pixel of the spectrum
    ; and scales in the wavelength direction. The returned vector contains the 
    ; wavelength at each observed data point.
    ; If no solution can be found, then try to change the scaling factors facL and facH
    ;
    ;
    ; Parameter        yin : Array  containing observed data (spectrum in x direction)
    ;
    ; Optional parameters and KEYWORDs
    ;                  FACL   : Lower limit of scaling factor (default 0.8)
    ;                  FACH   : Upper limit of scaling factor (default 1.5)
    ;                  FACS   : Scaling factor step between facL and facH  (default 0.01)
    ;                  CUT    : Number of wavelength points to be discarded at the ([xl,xr]), default [10,10]
    ;                  SHOW   : Shows plots 0: no plot, 1: plot final result, 2: plot all scaling steps (default 2)
    ;                  /DERIV : Caluculates derivative of the spectrum before least square difference
    ;                  CONT   : For broad lines such as Ca K, give the value of continuum
    ;
    ;-
    ; Version  v09.02.2024
    ;-----------------------------------------------------------------------------
    example use in example_wavelength_calibration

    """

    
    if len(xatlas) < 30:
        print("Provide longer spectrum!")
        return(0)
    
    yin/=yin.max()

    yobs = np.array(yin)
    if yobs.ndim > 1:
        yobs = np.sum(yobs[:, :, 0, 0], axis=1)
        


    cut = [10, 10] if CUT is None else CUT
    facs = 0.01 if FACS is None else FACS
    facL = 0.8 if FACL is None else FACL
    facH = 1.5 if FACH is None else FACH

    if not isinstance(SHOW, int):
        show = 2
    else:
        show = SHOW

    # READ atlas data and scale them to 1
    # data = Z3atlas(wlin, dwl, NOPLOT=True)
    # xatlas = np.array(data[0, :])
    # yatlas = np.array(data[1, :])
    # yatlas /= np.max(yatlas)  # normalization atlas

    sfts = len(xatlas)
    sobs = len(yobs)
    
    xapix = np.arange(sfts)

    yobs /= np.max(yobs[cut[0]:sobs - cut[1] +1 ])
    if CONT is not None:
        yobs = CONT * yobs  # normalization data

    newyobs = yobs[cut[0]:sobs - cut[1] + 1]
    snewyobs = len(newyobs)


    # FITTING SPECTRUM, with variable scaling
    idealshift=0
    npts=0

    correlold = 1000.  # high value

    if DERIV:
            dyatlas = np.gradient(yatlas)

    for fac in np.arange(facL, facH, facs):
          
            # if SAMPLING > 0:   # if correct sampling is already provided - does not work
            #     fac = (xatlas[1]-xatlas[0])/SAMPLING
                
            ipobs = np.arange(fac * (snewyobs ), dtype=int)
            #ippts = 1.*ipobs
            nipobs = len(ipobs)
            for j in range(nipobs):
                           ipobs[j] = j/fac
            ipnewyobs = interp1d(np.arange(snewyobs), newyobs, kind='cubic')(ipobs)   
            nipnewyobs = len(ipnewyobs)
            if DERIV:
                dipnewyobs = np.gradient(ipnewyobs)
            for s in range(sfts - nipnewyobs):
                correl = 0.
                if DERIV:
                    correl = np.sum((dyatlas[s:(nipnewyobs + s )] - dipnewyobs)**2) / (nipnewyobs - 1)
                else:
                    correl = np.sum((yatlas[s:(nipnewyobs + s )] - ipnewyobs)**2) / (nipnewyobs - 1)
                if correl < correlold:
                    idealshift = 1*s
                    correlold = 1.*correl
                    idealfac = 1.*fac
                    npts = 1*nipnewyobs
                    ippts = 1.*ipobs

            # if show > 1:
            #     plt.figure(num=1)
            #     plt.plot(xapix, yatlas, color='red')
            #     plt.plot(np.arange(nipnewyobs) + idealshift, ipnewyobs)
            #     plt.title('Z3ccspectrum - shift ')
            #     plt.xlabel('Wavelength [px]')
            #     plt.ylabel('I!DNormalized')
            #     plt.show()



    # CALCULATE OBSERVED WAVELENGTH RANGE
    print(f'  z3ccspectrum: idealfac = {idealfac:.6f}')
    xobs0 = xatlas[idealshift]
    xobs1 = xatlas[idealshift + npts - 1]
    lambdaperpx = (xobs1 - xobs0) / (ippts[npts - 1])
    lambda0 = xobs0 - cut[0] * lambdaperpx
    xobs = lambda0 + np.arange(sobs) * lambdaperpx
    
    
    start_index = np.argmin(abs(xobs[0]-xatlas))
    end_index = np.argmin(abs(xobs[-1]-xatlas))

    xatlas = xatlas[start_index :end_index]
    yatlas = yatlas[start_index :end_index]

    # PLOT
    if show > 0:
        plt.figure(num=2)
        plt.plot(xobs, yobs, label='Observation')
        plt.plot(xatlas, yatlas, color='red', linestyle='--',label='Reference')
        plt.title('Z3ccspectrum - result')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Normalized')
        plt.legend()
        plt.show()

    return(xobs, idealfac)



def find_shift(reference, target):
    # Use phase cross-correlation to find shifts
    shifts, _, _ = phase_cross_correlation(reference, target, upsample_factor=20)
    return(shifts[0], shifts[1])

def shift_image(image, y_shift, x_shift):
    # Use scipy.ndimage.shift for subpixel accuracy
    return(shift(image, shift=(-y_shift, -x_shift), mode='constant', cval=0))




def run_pca(data, n_components):
    X = data
    sc = StandardScaler()
    X = X.reshape((X.shape[0]*X.shape[1], X.shape[2],X.shape[3]))

    dummy=np.zeros((X.shape[1],X.shape[0],X.shape[2]))

    for i in range(dummy.shape[1]):
        dummy[:,i,:]=1.*X[i,:,:]

    X=1.*dummy.reshape(dummy.shape[0], dummy.shape[1]*dummy.shape[2])
    for i in range(X.shape[0]):
        X[i,:]=X[i,:]-np.mean(X[i,:])
    
    
    #X = sc.fit_transform(X)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    pca.transform(X)
    
    transformed = pca.fit_transform(X)
    inverted = pca.inverse_transform(transformed)
    print(pca.explained_variance_ratio_)
    
    return(pca,transformed, inverted)

def run_pca_s(data, n_components): # data is already wvl,observations
    X = data
    #sc = StandardScaler()
    
    #X = sc.fit_transform(X)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    pca.transform(X)
    
    transformed = pca.fit_transform(X)
    inverted = pca.inverse_transform(transformed)
    print('Explained variances by PCA components:')
    print(pca.explained_variance_ratio_)
    
    return(pca,transformed, inverted)






def align_scans(data): # with continuum images

    reference_v = 1.*data.v[:,0,:, 163:166].mean(axis=2)#dst.continuum[0]:dst.continuum[1]].mean(axis=2)
    reference_i = 1.*data.i[:,0,:, dst.continuum[0]:dst.continuum[1]].mean(axis=2)
    # go through all scans

    mask=np.ma.make_mask(data.i)
    current_image_i = 1.*reference_i
    current_image_v = 1.*reference_v
    for i in range(1, data.i.shape[1]):
       target_v = 1.*data.v[:,i,:,  163:166].mean(axis=2)#dst.continuum[0]:dst.continuum[1]].mean(axis=2)
       target_i = 1.*data.i[:,i,:,  dst.continuum[0]:dst.continuum[1]].mean(axis=2)
       y_shift_i, x_shift_i = find_shift(current_image_i, target_i)
       y_shift_v, x_shift_v = find_shift(current_image_v, target_v)
       print(i)
       print(y_shift_i, x_shift_i)
       print(y_shift_v, x_shift_v)
       
       # decide how to apply shift
       if abs((abs(y_shift_i)-abs(y_shift_v)) < 2): 
          if abs((y_shift_i-y_shift_v) > 2):
              reference = 1.*data.v[:,i-2,:, 163:166].mean(axis=2)
              y_shift, x_shift_d = find_shift(reference, target_v)
              print('Exception 1')
              print(y_shift, x_shift_d)
          else:
            y_shift = 0.5*(y_shift_i+y_shift_v)
       else:
           if abs(y_shift_i)>5 or abs(y_shift_v)>5:
             if abs(y_shift_i-y_shift_v)>5:
               reference = 1.*data.v[:,i-2,:, 163:166].mean(axis=2)
               y_shift, x_shift_d = find_shift(reference, target_v)
               print('Exception 1')
               print(y_shift, x_shift_d)
             else:
               if abs(y_shift_i) < abs(y_shift_v):      
                 y_shift = y_shift_i
               else:
                 y_shift = y_shift_v  
           else:
              reference = 1.*data.v[:,i-2,:, 163:166].mean(axis=2)
              y_shift, x_shift_d = find_shift(reference, target_v)
              print('Exception 1')
              print(y_shift, x_shift_d)
       
       if abs((abs(x_shift_i)-abs(x_shift_v)) < 2): 
           if abs((x_shift_i-x_shift_v) > 2):
               reference = 1.*data.v[:,i-2,:, 163:166].mean(axis=2)
               y_shift, x_shift_d = find_shift(reference, target_v)
               print('Exception 2')
               print(y_shift, x_shift_d)
           else:
            x_shift = 0.5*(x_shift_i+x_shift_v)
       else:
            if abs(x_shift_i)>5 or abs(x_shift_v)>5:
             if abs(x_shift_i-x_shift_v)>5:
                reference = 1.*data.v[:,i-2,:, 163:166].mean(axis=2)
                y_shift_d, x_shift = find_shift(reference, target_v)
                print('Exception 2')
                print(y_shift_d, x_shift)
             else:
              if abs(x_shift_i) < abs(x_shift_v):      
                x_shift = x_shift_i
              else:
                x_shift = x_shift_v  
              
            else:
               reference = 1.*data.v[:,i-2,:, 163:166].mean(axis=2)
               y_shift_d, x_shift = find_shift(reference, target_v)
               print('Exception 2')
               print(y_shift_d, x_shift)
           
       aligned_image_i = shift_image(target_i, -y_shift, -x_shift)
       
       aligned_image_v = shift_image(target_v, -y_shift, -x_shift)
       
       
       for w in range(data.i.shape[3]):
           data.i[:,i,:,w]=shift_image(data.i[:,i,:,w], -y_shift, -x_shift)
           data.v[:,i,:,w]=shift_image(data.v[:,i,:,w], -y_shift, -x_shift)
  
    current_image_i = 1.*aligned_image_i  # Update reference to the latest aligned image
    current_image_v = 1.*aligned_image_v  # Update reference to the latest aligned image
    # second iteration:
    # reference = 1.*data.i[:,0,:, dst.continuum[0]:dst.continuum[1]].mean(axis=2)
    # current_image = 1.*reference
    # for i in range(1, data.i.shape[1]):
    #    target = 1.*data.i[:,i,:, dst.continuum[0]:dst.continuum[1]].mean(axis=2)
    #    y_shift, x_shift = find_shift(current_image, target)
    #    print(y_shift, x_shift)
       
    #    if (abs(y_shift) > 5) or (abs(x_shift) > 5): 
    #        reference = 1.*data.i[:,i-2,:, dst.continuum[0]:dst.continuum[1]].mean(axis=2)
    #        y_shift, x_shift = find_shift(reference, target)
    #        print(y_shift, x_shift)
           
    #    aligned_image = shift_image(target, -y_shift, -x_shift)
       
       
    #    for w in range(data.i.shape[3]):
    #        data.i[:,i,:,w]=shift_image(data.i[:,i,:,w], -y_shift, -x_shift)
    #        data.v[:,i,:,w]=shift_image(data.v[:,i,:,w], -y_shift, -x_shift)
       
    return(data)

def binning(data, binning = [1,1,1,1]):
    
    if binning[0] > 1:
        print('not implemented yet')
    if binning[3] > 1:
        print('not implemented yet')
    if binning[1] > 1 or binning[2] > 1:
        
      kernel = np.ones((binning[1],binning[2]))/(binning[1]*binning[2])
        
      for w in range(data.i.shape[3]):
          for s in range(data.i.shape[0]):
            data.v[s,:data.i.shape[1]-binning[1]+1,:data.i.shape[2]-binning[2]+1,w] = convolve2d(data.v[s,:,:,w], kernel, 'valid')
            data.i[s,:data.i.shape[1]-binning[1]+1,:data.i.shape[2]-binning[2]+1,w] = convolve2d(data.i[s,:,:,w], kernel, 'valid')
    
    return(data)

    

def dummy_binning(data, binning=[1,1]):
    """Perform binning by convolution with averaging kernel.
    
    Works for both 2D and 3D data.
    For 3D data: applies binning to each wavelength slice.
    For 2D data: applies binning to the single image.
    """
    from scipy.signal import convolve2d
    kernel = np.ones((binning[0],binning[1]))/(binning[0]*binning[1])
    
    if data.ndim == 3:
        # 3D data: apply to each wavelength slice
        data_dummy = np.zeros_like(data)
        for i in range(data.shape[2]):
            conv_data = convolve2d(data[:,:,i], kernel, 'valid')
            data_dummy[:conv_data.shape[0],:conv_data.shape[1],i] = conv_data
    elif data.ndim == 2:
        # 2D data: apply to single image
        conv_data = convolve2d(data, kernel, 'valid')
        data_dummy = conv_data
    else:
        raise ValueError(f"dummy_binning expects 2D or 3D data, got {data.ndim}D")
    
    return(data_dummy)


def check_continuum_noise(data): # stokes V in continuum only
    
    zoom=2
    my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
    plt.style.use('default')
    plt.rcParams.update(params)
    nullfmt=NullFormatter()

    fig=plt.figure(num=89) 
    fig.clf()
    fig.set_size_inches([fig_size_single[0], fig_size_single[1]],forward=True)

    gs = gridspec.GridSpec(1,1, )#width_ratios=[0.25,0.25,0.25,0.25] , height_ratios=[1,1,1,1] )
    gs.update(top=0.96, bottom=0.2,left=0.15, right=0.99, wspace=0.05, hspace=0.26)
         
    ax=fig.add_subplot(gs[0,0]) 
    
    #use standard deviation along slit 
    initial_noise = np.std(data)
    
    
    ideal_binned_noise = []
    
    binned_noise = []
    
    binning_x = []
    binning_x.append(1)
    
    ideal_binned_noise.append(initial_noise)
    binned_noise.append(initial_noise)
    
    
    for i in range(1,data.shape[0]-2):
        for p in range(1,data.shape[1]-2):
            binning_x.append((i)*(p))
            ideal_binned_noise.append(initial_noise/np.sqrt((i)*(p)))
            __dummy = np.std(dummy_binning(data, binning=[i,p]))
            binned_noise.append(__dummy)
            
    
    binning_x = np.array(binning_x)
    binned_noise = np.array(binned_noise)
    ideal_binned_noise = np.array(ideal_binned_noise)
    sort = np.argsort(binning_x)
    
    # print(binning_x.shape, binned_noise.shape, ideal_binned_noise.shape)
    ax.plot(binning_x[sort],100*ideal_binned_noise[sort], label = 'theoretical noise')
    ax.plot(binning_x[sort],100*binned_noise[sort], label = 'continuum noise')
    ax.set_xlabel('# pixel binned')
    ax.set_ylabel('noise [%]')
    ax.legend()
    return(0)




class ff:
    # Class object to store folder and file names
 def __init__(self):
         
   self.directory = dst.directory
   self.inversion_dir = dst.directory_inversion
   self.inversion_inp_ext = '_inv_inp.fits'
   self.inv_inp_data_file = 'obs_prof_'+ dst.data_files[dst.line] # compliant with SIRExplorer
   self.red_data_file = dst.directory + dst.data_files[dst.line]+dst.data_files['r']
   self.figure_dir = dst.directory_figures
   self.v_file= dst.directory + dst.data_files[dst.line]+dst.data_files['v_i']
   self.i_file= dst.directory + dst.data_files[dst.line]+dst.data_files['i']
   

class init:
    """Legacy init class for backward compatibility with old display code.
    
    Note: For new code, use get_config() from themis.datasets.themis_datasets_2025 instead.
    """
    def __init__(self):
        self.directories = dst.directories
        self.dataset = dst.dataset
        self.cam = cc.cam[self.dataset['line']]
        self.data_types = dst.data_types
        self.reduction_levels = tdr.reduction_levels
        self.slit_width = dst.slit_width
        self.polarization_states = dst.states
        
        # Note: File discovery is now handled by get_config() in themis_datasets_2025.py
        # This legacy class does not populate file sets
    
    def __repr__(self):
     lines = [
        f"<Init Configuration>",
        f"  Line:       {self.dataset['line']}",
        f"  Camera:     {self.cam}",
        f"  Slit Width: {self.slit_width} arcsec",
        f"  Polarization States: {', '.join(self.polarization_states)}",
        f"  Directories:",
        f"    Raw:      {self.directories.raw}",
        f"    Reduced:  {self.directories.reduced}",
        f"    Figures:  {self.directories.figures}",
        f"    Inversion:{self.directories.inversion}",
        f"  Files:"
     ]

     for key in dst.file_types:
        entry = self.dataset[key]
        file_set = entry.get('files')
        if file_set:
            for level in self.reduction_levels.list_levels():
                path = file_set.get(level)
                exists = "✓" if path and path.exists() else "✗"
                lines.append(f"    {key} ({level}): {path.name if path else '—'} {exists}")
        else:
            lines.append(f"    {key}: —")

     return "\n".join(lines)



    
class StokesResult:
    """Container for Stokes polarimetry results.
    
    Access results via method ('difference' or 'ratio') and Stokes parameter.
    Arrays are shaped (n_slit, n_map, ny, nx), so numpy slicing works:
        result.difference.I[0, :]       -> all maps at slit 0
        result.ratio.V[0, 0]            -> single 2D frame
        result.difference.Q[:, 0, :, :] -> all slits at map 0
    """
    class _MethodResult:
        def __init__(self):
            self.I = None  # np.ndarray (n_slit, n_map, ny, nx)
            self.Q = None
            self.U = None
            self.V = None
        
        def __repr__(self):
            params = [s for s in ['I', 'Q', 'U', 'V'] if getattr(self, s) is not None]
            shape = self.I.shape if self.I is not None else None
            return f"StokesMethod(params={params}, shape={shape})"
    
    def __init__(self):
        self.difference = self._MethodResult()
        self.ratio = self._MethodResult()
    
    def __repr__(self):
        return f"StokesResult(difference={self.difference}, ratio={self.ratio})"


def compute_polarimetry(cycle_set):
    """Compute Stokes I, Q, U, V from a CycleSet using difference and ratio methods.
    
    For each Stokes parameter S in {Q, U, V} at each (slit_idx, map_idx):
    
    Difference method:
        S/I = 0.25 * (pS_upper - pS_lower + mS_lower - mS_upper) / I
        I   = 0.25 * (pS_upper + pS_lower + mS_lower + mS_upper)
    
    Ratio method:
        I = same as difference
        S = pS_upper / mS_upper * mS_lower / pS_lower - 1
    
    Parameters
    ----------
    cycle_set : CycleSet
        L4 scan data with keys (frame_state, slit_idx, map_idx).
        frame_state is e.g. 'pQ', 'mQ', 'pU', 'mU', 'pV', 'mV'.
    
    Returns
    -------
    StokesResult
        Access via .difference or .ratio, then .I, .Q, .U, .V
        Each is a numpy array with shape (n_slit, n_map, ny, nx).
    """
    # Collect grid dimensions and available Stokes parameters
    slit_indices = set()
    map_indices = set()
    stokes_params = set()
    for key in cycle_set.keys():
        frame_state, slit_idx, map_idx = key
        slit_indices.add(slit_idx)
        map_indices.add(map_idx)
        stokes_params.add(frame_state[1:])
    
    slit_indices = sorted(slit_indices)
    map_indices = sorted(map_indices)
    stokes_params = sorted(stokes_params)
    n_slit = len(slit_indices)
    n_map = len(map_indices)
    slit_to_idx = {s: i for i, s in enumerate(slit_indices)}
    map_to_idx = {m: i for i, m in enumerate(map_indices)}
    
    # Get frame shape from first available frame
    first_key = next(iter(cycle_set.keys()))
    first_frame = cycle_set[first_key]
    ny, nx = first_frame.get_half('upper').data.shape
    
    print(f'  Stokes parameters found: {stokes_params}')
    print(f'  Grid: {n_slit} slit x {n_map} map, frame shape: ({ny}, {nx})')
    
    # Pre-allocate arrays
    result = StokesResult()
    for method in [result.difference, result.ratio]:
        method.I = np.zeros((n_slit, n_map, ny, nx), dtype='float32')
        for s in stokes_params:
            setattr(method, s, np.zeros((n_slit, n_map, ny, nx), dtype='float32'))
    
    for slit_idx, map_idx in tqdm(
            [(s, m) for s in slit_indices for m in map_indices],
            desc='Computing polarimetry'):
        si = slit_to_idx[slit_idx]
        mi = map_to_idx[map_idx]
        
        for stokes in stokes_params:
            p_frame = cycle_set.get_state_slit_map(f'p{stokes}', slit_idx, map_idx)
            m_frame = cycle_set.get_state_slit_map(f'm{stokes}', slit_idx, map_idx)
            
            if p_frame is None or m_frame is None:
                print(f'    ⚠ Missing p{stokes}/m{stokes} at slit={slit_idx}, map={map_idx}')
                continue
            
            pS_upper = p_frame.get_half('upper').data.astype('float64')
            pS_lower = p_frame.get_half('lower').data.astype('float64')
            mS_upper = m_frame.get_half('upper').data.astype('float64')
            mS_lower = m_frame.get_half('lower').data.astype('float64')
            
            # --- Difference method ---
            I_diff = 0.25 * (pS_upper + pS_lower + mS_lower + mS_upper)
            with np.errstate(divide='ignore', invalid='ignore'):
                S_diff = 0.25 * (pS_upper - pS_lower + mS_lower - mS_upper) / I_diff
            S_diff = np.nan_to_num(S_diff, nan=0.0, posinf=0.0, neginf=0.0)
            
            # --- Ratio method ---
            with np.errstate(divide='ignore', invalid='ignore'):
                S_ratio = (pS_upper / mS_upper * mS_lower / pS_lower - 1)
            S_ratio = np.nan_to_num(S_ratio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Store
            result.difference.I[si, mi] = I_diff.astype('float32')
            result.ratio.I[si, mi] = I_diff.astype('float32')
            getattr(result.difference, stokes)[si, mi] = S_diff.astype('float32')
            getattr(result.ratio, stokes)[si, mi] = S_ratio.astype('float32')
    
    print(f'  ✓ Polarimetry computed: {n_slit} slit x {n_map} map, {len(stokes_params)} Stokes parameters')
    return result


def prep_data_for_pca(data):
    # prepare data for PCA - first dimension is wavelength << other dimension
    # print(data.shape)
    if len(data.shape) == 3:
        data_prep =data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
    else:
        print('Data for PCA has the wrong dimensions')
        data_prep = 0
    mean = np.mean(data_prep)
    data_prep-=mean
    return(data_prep, mean)


    
