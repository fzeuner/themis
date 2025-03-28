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
from style_themis import style_aa
from matplotlib.ticker import (NullFormatter)

from scipy.signal import convolve2d
from scipy.io import readsav
import themis_datasets as dst
from scipy.signal import medfilt
from tqdm import tqdm
import matplotlib.colors as mplcolor
from scipy.interpolate import interp1d
from scipy.signal import windows

from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

from matplotlib import gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# GLOBAL VARIABLES

directory_atlas = '/home/zeuner/data/atlas' # on x1


#%%

class l1_data:
    def __init__(self, upper = '-V', lower = '+V'):
    
        self.wavelength = ''  # name
        self.file = ''
        self.wvl = ''         # wavelength array
        self.ui = ''          # upper image
        self.li = ''          # lower image 
        self.upper_state = upper
        self.lower_state = lower

        self.date = ''
        
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

def read_images_file(file_name,original=False, verbose=False):  # not needed really
    
    sif_sp = fits.open(file_name, memmap=True)
    header=sif_sp[0].header
    steps=int(header['NBSTEP'])+1
    pol_states = int(len(header['POL_VAL'].split())/2) # by two because there is the retarder and polarizer given
    
    dummy=np.array(sif_sp[0].data)
    if verbose:
     print(dummy.shape)
     print(header)
    # xmin=float(header['WAVEMIN']) #nm
    # xmax=float(header['WAVEMAX']) #nm
    # xlam=10.*np.arange(xmin,xmax,(xmax-xmin)/data[0,:].shape[1]) # A
    sif_sp.close()
    
    data = l1_data()
    data.ui =  np.zeros((pol_states,steps,int(dummy.shape[0]/pol_states/steps), int(dummy.shape[1]/2), dummy.shape[2]))
    data.li =  1.*data.ui
    for i in range(pol_states):
        
     data_dummy = dummy[i::pol_states]
     
     for s in range(steps):
         dummy_dummy = data_dummy[s::steps]
         data.li[i,s,:] = dummy_dummy[:,0:int(dummy.shape[1]/2),:]
         data.ui[i,s,:] = dummy_dummy[:,int(dummy.shape[1]/2):,:]
    print('Read '+file_name)
    if original:
        return(dummy, header) # pol_states+scan pos, camera y, camera x (wvl)
    else:
      return(data, header)

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
        plt.xlabel('Wavelength [A]')
        plt.ylabel('Normalized')
        plt.legend()
        plt.show()

    return(xobs)

def read_v_file(file_name,file_i_name,original=False, i_ct = 0, verbose=False):  # not needed really
    
    sif_sp = fits.open(file_name, memmap=True)
    header=sif_sp[0].header
    steps=int(header['NBSTEP'])+1
    
    dummy=np.array(sif_sp[0].data)
    if verbose:
     print(dummy.shape)
     print(header)
    # xmin=float(header['WAVEMIN']) #nm
    # xmax=float(header['WAVEMAX']) #nm
    # xlam=10.*np.arange(xmin,xmax,(xmax-xmin)/data[0,:].shape[1]) # A
    sif_sp.close()
    
    data = v_data()
    data.v =  np.zeros((steps,int(dummy.shape[0]/steps), int(dummy.shape[1]), dummy.shape[2]))
    data.i =  0.*data.v
    
    sif_sp = fits.open(file_i_name, memmap=True)
    # header=sif_sp[0].header
    
    dummy_i=np.array(sif_sp[0].data)
 
    sif_sp.close()

    for s in range(steps):
         dummy_dummy = dummy[s::steps]
         data.v[s,:] = dummy_dummy[:,0:int(dummy.shape[1]),:]
         dummy_dummy_i = dummy_i[s::steps]
         data.i[s,:] = dummy_dummy_i[:,0:int(dummy_i.shape[1]),:]
        
    print('Read '+file_name + ' and '+file_i_name)
    
    #data.v = data.v + i_ct*data.i

    if original:
        return(dummy, header) # pol_states+scan pos, camera y, camera x (wvl)
    else:
      return(data, header)


def find_shift(reference, target):
    # Use phase cross-correlation to find shifts
    shifts, _, _ = phase_cross_correlation(reference, target)
    return(shifts[0], shifts[1])

def shift_image(image, y_shift, x_shift):
    # Use scipy.ndimage.shift for subpixel accuracy
    return(shift(image, shift=(-y_shift, -x_shift), mode='constant', cval=0))


def plot_power_spectrum(data, zoom=1): # data: y, x

    print(r'Power spectrum')
    image=1.*data
    # create figure
    intmeth = 'none' # imshow integration method
    my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
    plt.style.use('default')
    plt.rcParams.update(params)
    nullfmt=NullFormatter()
    
    fig=plt.figure(num=32) 
    fig.clf()
    fig.set_size_inches([fig_size_single[0], 1.5*fig_size_single[1]],forward=True)

    gs = gridspec.GridSpec(2,1, height_ratios=[0.38,1] )#width_ratios=[0.25,0.25,0.25,0.25] , height_ratios=[1,1,1,1] )
    gs.update(top=0.96, bottom=0.1,left=0.13, right=0.99, wspace=0.05, hspace=0.26)
         
         
    axs=fig.add_subplot(gs[1,0]) 
    ax0=fig.add_subplot(gs[0,0]) 
    
    window_x=windows.hann(image.shape[1]) 
    window_y=windows.hann(image.shape[0]) 
    for x in range(image.shape[0]):
        image[x,:]*=window_x
    for y in range(image.shape[1]):
        image[:,y]*=window_y
        

    fs = np.fft.fftn(image)
    fs_abs=(np.abs(np.fft.fftshift(fs))**2)
    # fs_sz=np.shape(fs_abs)
    # axs[0].imshow(np.log(fs_abs/np.amax(fs_abs)), cmap='jet', interpolation=intmeth, aspect='auto')
   
    y,x = np.indices((fs_abs.shape)) # first determine radii of all pixels
    center=[int(np.shape(fs_abs)[0]/2),int(np.shape(fs_abs)[1]/2)]
    r = np.sqrt((x-center[1])**2+(y-center[0])**2)    

    # radius of the image.
    r_max = int(np.floor(np.max(r)))

    # ring_brightness, radius = np.histogram(r, weights=fs_abs, bins=r_max*2)
    # ring_size, radius_1 = np.histogram(r, bins=r_max*2)
    # #plt.plot(radius[1:], ring_brightness/ring_size)
    # liney,= axs[1].plot(radius[1:]/np.max(radius[1:])*1./(2.*dst.pixel[dst.line]), 100*ring_brightness/ring_brightness.max(), 
    #                     color='black', marker='.', linestyle='None') #   
    # axs[1].set_xlabel("1/arcsec")
    # axs[1].set_ylim(0.0000001,0.1)
    # axs[1].set_xlim(0.2,10.1)
    # axs[1].set_ylabel('Power [a.u]')
    # axs[1].set_yscale('log')
    
    image_2=1.*data
    fs_ = np.zeros((image_2.shape[0],int(image_2.shape[1]/2)))
    
    freqs = (np.fft.fftfreq(image_2.shape[1])* 1./(dst.pixel[dst.line]))[:int(image_2.shape[1]/2)]

    contrast=np.zeros(image_2.shape[0])
    for i in range(image_2.shape[0]):
        i_m=np.mean(image_2[i,:])
      #  max_da=np.mean(data[0,i,550,np.argsort(data[0,i,550,:])][-100:-5])
        
        contrast[i]=100.*np.std(image_2[i,:])/i_m
        
    print(r'Mean rms contrast [%]: '+str(np.mean(contrast)))
    ax0.plot(contrast, color='black')
    ax0.set_ylabel(r'RMS contrast [%]', labelpad=0)
    ax0.set_xlabel("Scan position", labelpad=-0.8)
    ax0.axhline(y=contrast.mean(), color='black',
               linestyle="dotted", alpha=0.8, linewidth=0.8*zoom)
    for te in range(image_2.shape[0]):
     # if contrast[te] > contrast.mean():  
        image_2[te,:]*=window_x
        fs=np.fft.fftn(image_2[te,:])
        fs_abs=np.roll((np.abs(np.fft.fftshift(fs))**2)[int(image_2.shape[1]/2):],-int(image_2.shape[1]/2))
        fs_[te,:]=fs_abs/fs_abs.max()
    
    fs_=np.mean(fs_, axis=0)
    # use only non-zero parts    
    fs_/=fs_.mean()
    #idx_nonzero=np.where(fs > 10e-11)
    
   # freq_axis=(radius[1:]/np.max(radius[1:])*1./(2.*dst.pixel[dst.line]))[idx_nonzero]
    freq_axis=freqs#[idx_nonzero]
  
    liney,= axs.plot(freq_axis,fs_,  color='black', marker='.', linestyle='None') #   
    
    n_filtered = 111 # needs to be odd!!!
    medi_filtered = medfilt(fs_,n_filtered)
    
    
    axs.plot(freq_axis[int((n_filtered+1)):-int((n_filtered+1))],
                medi_filtered[int((n_filtered+1)):-int((n_filtered+1))],
                        color='blue',linestyle='solid', linewidth=2*zoom ) #   
    
    # gradient=np.gradient(medi_filtered[int((n_filtered+1)):-int((n_filtered+1))])
    # axs[2].plot((radius[1:]/np.max(radius[1:])*1./(2.*dst.pixel[dst.line]))[int((n_filtered+1)):-int((n_filtered+1))],
    #             gradient,
    #                     color='red',linestyle='solid' )
    
    axs.axvline(x=1/0.58, color='red',
               linestyle="dashed", alpha=0.8, linewidth=0.8*zoom)
    axs.axhline(y=2e-7, color='black',
               linestyle="dashed", alpha=0.8, linewidth=0.8*zoom)
    axs.set_ylabel('Power [a.u]', labelpad=-1)
    axs.set_yscale('log')
    axs.set_ylim(0.00000001,0.01)
    axs.set_xlim(0.1,2.1)
    axs.set_xlabel(r"arcsec$^{-1}$",labelpad=-1)
    #fig.show()

    return(fig, fs_)

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


def readpro(filename):

    """ 
    Reads a line profile from a .per file
    Call:
    line_ind, wvlen, StkI, StkQ, StkU, StkV = st.readpro(filename)
    """
    
    from numpy import array

    f = open(filename, 'r')

    line_ind = []
    wvlen = []
    StkI = []
    StkQ = []
    StkU = []
    StkV = []
    
    for line in f:
        data = line.split()
        line_ind.append(float(data[0]))
        wvlen.append(float(data[1]))
        StkI.append(float(data[2]))
        StkQ.append(float(data[3]))
        StkU.append(float(data[4]))
        StkV.append(float(data[5]))

    f.close()

    line_ind = array(line_ind)
    wvlen = array(wvlen)
    StkI = array(StkI)
    StkQ = array(StkQ)
    StkU = array(StkU)
    StkV = array(StkV)

    return(line_ind, wvlen, StkI, StkQ, StkU, StkV)



def writepro(filename, line_ind, wvlen, StkI, StkQ, StkU, StkV):
    """ 
    Routine that writes the Stokes profiles into a SIR formatted Stokes file.
    Call:
    writepro(filename, line_ind, wvlen, StkI, StkQ, StkU, StkV)
    """

    f = open(filename, "w+")

    for k in range(0, len(line_ind)):
        f.write('     {0}   {1:> 10.4f}  {2:> 8.6e} {3:> 8.6e} {4:> 8.6e} {5:> 8.6e} \n'.format(line_ind[k], wvlen[k], StkI[k], StkQ[k], StkU[k], StkV[k]))

    f.close()

    return()


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



def add_axis(fig,ax,data, wl,label, pixel, xvis='no', yvis='no',title=None):
    
   ax.cla()
   intmeth = 'none' # imshow integration method
   dmin,dmax=np.percentile(data.flatten(),(0.5,99.5))
   mn=np.mean(data)
   fct=np.mean(data)/mn
   
   image_sz=np.array(data.shape)
   
   extent=[0,image_sz[1]*pixel,0,image_sz[0]*pixel]

   if wl.shape[0] == image_sz[0]:
       extent[2]=wl[0]
       extent[3]=wl[-1]
   if title != None:
       ax.set_title(title)
   
   im=ax.imshow(data, cmap='gray',origin='lower',\
   interpolation=intmeth,clim=[fct*dmin,fct*dmax], 
   extent=extent, aspect='auto')#
       
   #Plot color bar   
   cb=fig.colorbar(im,ax=ax, orientation="vertical")
   cb.ax.minorticks_on()
   cb.set_label(label)
   
   if xvis == 'no':
       ax.get_xaxis().set_visible(False)
   else:
       ax.get_xaxis().set_visible(True)
       if xvis != 'yes':         
           ax.set_xlabel(xvis)  
   if yvis == 'no':
       ax.get_yaxis().set_visible(False)
   else:
       ax.get_yaxis().set_visible(True)
       if yvis != 'yes':         
           ax.set_ylabel(yvis)  
       
   return(ax)


def read_fts5(wl_start, wl_end): # wavelenght start and end in Angstroem

     # -- fts data
    file_fts5 = directory_atlas+'/fts5.sav'
        
    s =  readsav(file_fts5 , verbose=False,python_dict=True)

    wavelength=s['w']      # wavelength (Å)
    fts_i=s['b'] 	       # I/Ic
    fts_v=s['vi'] 	       # Stokes V/Ic (unsmoothed, recommended)

    s_idx =  np.argmin(abs(wavelength-wl_start))  # in Angstrom
    e_idx = np.argmin(abs(wavelength-wl_end))     # in Angstrom

    wavelength = wavelength[s_idx-1:e_idx]
    fts_i = fts_i[s_idx-1:e_idx]
    fts_v = fts_v[s_idx-1:e_idx]
    
    print("read_fts: Only working for start wavelength below 6906 and above 5253.8")
    return(wavelength, fts_i, fts_v)

def dummy_binning(data, binning=[1,1]):
    kernel = np.ones((binning[0],binning[1]))/(binning[0]*binning[1])
    data_dummy = 0.*data
    for i in range(data.shape[2]):
     conv_data = 1.*convolve2d(data[:,:,i], kernel, 'valid')
     data_dummy[:conv_data.shape[0],:conv_data.shape[1],i] = conv_data
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
    initial_noise = np.mean(np.std(data, axis=2))
    
    
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
            __dummy = np.mean(np.std(dummy_binning(data, binning=[i,p]), axis=2))
            binned_noise.append(__dummy)
            
    
    binning_x = np.array(binning_x)
    binned_noise = np.array(binned_noise)
    ideal_binned_noise = np.array(ideal_binned_noise)
    sort = np.argsort(binning_x)
    
    # print(binning_x.shape, binned_noise.shape, ideal_binned_noise.shape)
    ax.plot(binning_x[sort],100*ideal_binned_noise[sort], label = 'theoretical V noise')
    ax.plot(binning_x[sort],100*binned_noise[sort], label = 'V continuum noise')
    ax.set_xlabel('# pixel binned')
    ax.set_ylabel('noise [%]')
    ax.legend()
    return(0)
def add_scan_axis(fig,ax,data, label, stokes =False, clim=[0,0], xvis='no', yvis='no', x_offset=0, title=None, total=False):
    
   if stokes:  
    if total:
        cdict = {'red':   ((0.0, 0.0, 0.0), (0.25, 1.0, 1.0), (0.75,1.0,1.0), (1.0,1.0,1.0)),
                'green': ((0.0, 0.0, 0.0),  (0.25, 1.0, 1.0), (0.75,0.0,0.0), (1.0,1.0,1.0)),
                'blue':  ((0.0, 1.0, 1.0),  (0.25, 0.0, 0.0), (0.75,0.0,0.0), (1.0,1.0,1.0))}
    else:

     cdict = {'red':  ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 1.0, 1.0), (0.75,1.0,1.0)  , (1.0,1.0,1.0)),
             'green': ((0.0, 1.0, 1.0), (0.25, 0.0, 0.0), (0.5, 1.0, 1.0), (0.75,0.75,0.75), (1.0,0.0,1.0)),
             'blue':  ((0.0, 0.0, 0.0), (0.25, 1.0, 1.0), (0.5, 1.0, 1.0), (0.75,0.0,0.0)  , (1.0,0.0,0.0))}

    p_cmap = mplcolor.LinearSegmentedColormap('p_colormap',cdict,256)    
    
   else: 
       p_cmap='gray'
    
   ax.cla()
   intmeth = 'none' # imshow integration method
   dmin,dmax=np.percentile(data.flatten(),(0.5,99.5))
   mn=np.mean(data)
   fct=np.mean(data)/mn
   
   image_sz=np.array(data.shape)
   
   extent=[x_offset*dst.pixel[dst.line],(x_offset+image_sz[1])*dst.pixel[dst.line],0,image_sz[0]*dst.slitwidth]

   if title != None:
       ax.set_title(title)
   
   clim_plot=[fct*dmin,fct*dmax]
   if abs(clim[0] - clim[1]) > 10e-8:
       clim_plot=clim
   im=ax.imshow(data, cmap=p_cmap,origin='lower',\
                    interpolation=intmeth,clim=clim_plot, 
                    extent=extent, aspect='auto')#
       
   #Plot color bar   
   cb=fig.colorbar(im,ax=ax, orientation="vertical")
   cb.ax.minorticks_on()
   cb.set_label(label)
   
   if xvis == 'no':
       ax.get_xaxis().set_visible(False)
   else:
       ax.get_xaxis().set_visible(True)
       if xvis != 'yes':         
           ax.set_xlabel(xvis, labelpad=-0.8)  
   if yvis == 'no':
       ax.get_yaxis().set_visible(False)
   else:
       ax.get_yaxis().set_visible(True)
       if yvis != 'yes':         
           ax.set_ylabel(yvis, labelpad=-0.8)  
       
   return(ax)

def add_axis_contour(fig, ax, data, data2, threshold, zoom, wl,label, pixel, xvis='no', yvis='no',title=None):
    
   ax.cla()
   intmeth = 'none' # imshow integration method
   dmin,dmax=np.percentile(data.flatten(),(3.,97.5))
   mn=np.mean(data)
   fct=np.mean(data)/mn

   
   image_sz=np.array(data.shape)
   linewidth_contour=2.*zoom
   alpha_value=0.7
   
   extent=[0,image_sz[1]*pixel,0,image_sz[0]*pixel]

   if wl.shape[0] == image_sz[0]:
       extent[2]=wl[0]
       extent[3]=wl[-1]
   if title != None:
       ax.set_title(title)
   
   im=ax.imshow(data, cmap='gray',origin='lower',\
   interpolation=intmeth,clim=[fct*dmin,fct*dmax], 
   extent=extent, aspect='auto')#
       
   #Plot color bar   
   cb=fig.colorbar(im,ax=ax, orientation="vertical")
   cb.ax.minorticks_on()
   cb.set_label(label)
   
   if xvis == 'no':
       ax.get_xaxis().set_visible(False)
   else:
       ax.get_xaxis().set_visible(True)
       if xvis != 'yes':         
           ax.set_xlabel(xvis)  
   if yvis == 'no':
       ax.get_yaxis().set_visible(False)
   else:
       ax.get_yaxis().set_visible(True)
       if yvis != 'yes':         
           ax.set_ylabel(yvis)  
            
   
   # for u, contour in enumerate(contours):
   #      #ax.plot(contour[:,1],contour[:,0],linewidth=linewidth_contour, color='k', alpha=alpha_value)
   #      ax.plot([0,2000],[0,2000],linewidth=linewidth_contour, color='k', alpha=alpha_value)
   #      #mask=sm.grid_points_in_poly([subfield[1]-subfield[0],subfield[3]-subfield[2]], contour)
   ax.contour(data2, threshold, colors='magenta', origin='lower', extent=extent, linewidths= linewidth_contour, alpha=alpha_value)
   return(ax)


 

def plot_stokes_simple(data, wl, pixel=1,zoom=1, fnum=0, label_up = False, title=None, save_figure=[False,'None',__file__]):
    
    plt.close()
    my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
    plt.style.use('default')
    plt.rcParams.update(params)
    nullfmt=NullFormatter()
    
    fig, axs = plt.subplots(4,constrained_layout=True, num=3)
    fig.set_size_inches(fig_size,forward=True)
    
    if label_up:
        stokes_str=[r'$I$',r'$Q$ [$I_c$]',r'$U$ [$I_c$]',r'$V$ [$I_c$]']
    else:
        stokes_str=[r'$I$',r'$Q/I$ [$\%$]',r'$U/I$ [$\%$]',r'$V/I$ [$\%$]']
    titles=[title, None, None, None]
    yvis=[r'Wavelength $[\mathrm{\AA}]$', r'Wavelength $[\mathrm{\AA}]$', r'Wavelength $[\mathrm{\AA}]$', r'Wavelength $[\mathrm{\AA}]$']
    
    if pixel == 1:
     xvis=[r'no', r'no',r'no', r'pixel']
    else:
     xvis=[r'no', r'no',r'no', r'arcsec'] 
     
    
    for i,s in enumerate(stokes_str):
        ax=add_axis(fig,axs[i],data[i,:], wl,s, pixel,xvis=xvis[i], yvis=yvis[i], title=titles[i])
        
    fig.show()
    
    if save_figure[0]:
        
       fig.savefig(save_figure[1], 
              dpi=my_dpi, metadata={'Software': save_figure[2]})
    return(fig, axs)

def plot_scan_images(data, wl_idx, zoom=2, fnum=0, binning= [1,1,1], clim=[0,0], vlim=[0,0], title=None, save_figure=[False,'None',__file__]):
    # binning is in spatially in scan direction, spatially along slit, spectrally
    
    # print('plotted data is binned to' )
    my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
    plt.style.use('default')
    plt.rcParams.update(params)
    nullfmt=NullFormatter()
    
    fig = plt.figure(num=fnum, layout="constrained")
    fig.set_size_inches([fig_size_single[0],1.3*fig_size_single[1]],forward=True)
    fig.clf()
    
    gs = gridspec.GridSpec(4,1, figure=fig)#width_ratios=[1,0.25,1,0.25])# , height_ratios=[1,1,1,1] )
    gs.update(top=0.99, bottom=0.14,left=0.11, right=0.95, wspace=0.17, hspace=0.12)
    
    
    stokes_str=[r'$I$',r'$Q/I$ [$\%$]',r'$U/I$ [$\%$]',r'$V/I$ [$\%$]']
    titles=[title, None, None, None]
    yvis=[r'arcsec', r'arcsec', r'arcsec', r'arcsec']
    
    xvis=[r'no', r'no',r'no', r'arcsec'] 
    
    data_plot= np.copy(data)
    
    # wavelength binning
    shifts=[0,1,-1,2,-2,3,-3,4,-4,5,-5]
    
    if binning[2]>1:
       data_plot=0.* np.copy(data)
       for wv in range(binning[2]):
          data_plot += np.roll(data, shifts[wv], axis=2) 
       data_plot/=binning[2]
    data_plot=data_plot[:,:,wl_idx,:]
    kernel = np.ones((binning[0],binning[1]))/(binning[0]*binning[1])
    
    
    for i,s in enumerate(stokes_str):
        
        axs = fig.add_subplot(gs[i, 0])
        
        if i > 0:
            stokes=True
        else:
            stokes=False
            
        # spatial binning
        if binning[0] > 1 or binning[1]> 1:
               data_plot_dummy = convolve2d(data_plot[i,:], kernel, 'valid')
        else:
            data_plot_dummy = data_plot[i,:]
        if i == 1 or i == 2:
            clim_plot = clim
        else:
            clim_plot=[0,0]
            if i == 3:
                clim_plot=vlim
        ax=add_scan_axis(fig,axs,data_plot_dummy,s, xvis=xvis[i], yvis=yvis[i],  clim=clim_plot,title=titles[i], stokes=stokes)
        ax.set_yticks([0,0.5,1])
    
    fig.show()
    
    if save_figure[0]:
        
       fig.savefig(save_figure[1], 
              dpi=my_dpi, metadata={'Software': save_figure[2]})
       
      # plot intensity spectrum with vertical line indicating the wavelength point plotted  
       
    fig2 = plt.figure(num=44,  layout="constrained")
    fig2.set_size_inches(fig_size_single,forward=True)
    fig2.clf()

    gs2 = gridspec.GridSpec(1,1, figure=fig2 )#width_ratios=[1,0.25,1,0.25])# , height_ratios=[1,1,1,1] )
    gs2.update(top=0.95, bottom=0.12,left=0.08, right=0.99, wspace=0.17, hspace=0.05)
    
    ax2 = fig2.add_subplot(gs2[0, 0]) # 
    ax2.cla()
    
    intensity_spectrum = data[0,0,:,0]
    ax2.plot(intensity_spectrum)
    ax2.vlines(wl_idx, ymin=intensity_spectrum.min(), ymax=intensity_spectrum.max(), color='red', linewidth=2)
    fig2.show() 
    return(fig)

def add_noise_to_profiles(file, noise): # gaussian noise
    
    line_ind, wvlen, StkI, StkQ, StkU, StkV = readpro(file)
    noise_v = np.random.normal(0.0, noise, StkV.shape[0])
    noise_q = np.random.normal(0.0, noise, StkQ.shape[0])
    noise_u = np.random.normal(0.0, noise, StkU.shape[0])

    print('max SNR Q: '+str(max(StkQ)/noise))
    print('max SNR U: '+str(max(StkU)/noise))
    print('max SNR V: '+str(max(StkV)/noise))
    
    filename_without_ext = os.path.splitext(os.path.basename(file))[0]
    dirname, fname = os.path.split(file)
    
    writepro(dirname+'/'+filename_without_ext+'_n.per',line_ind, wvlen, StkI, StkQ+noise_q, StkU+noise_u, StkV+noise_v)
        
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
       
# def process_data(data, xlam, pca=False,  #data is a scan_pos, Y, wavelength
#                              continuum=False):
         
    
    
#     if background: 
#          data = denoise(data) 
         
#     if continuum:
#          data = continuum_correction(data, xlam, dst.continuum[dst.line])
         
#     return(data)
    

def save_inv_input_data(data, xlam,ff,scan, overwrite = True):
    
    hdr = fits.Header()
    hdr.set('RED_ROUT',  'themis/save_reduced_data', comment='Code used for reduction')
    
    primary_hdu = fits.PrimaryHDU(data, header=hdr)
    hdul = fits.HDUList(
        [primary_hdu])
    
    inv_inp_data_file= ff.inversion_dir + ff.inv_inp_data_file +'_{:.0f}'.format(scan) + '.fits'
    
    if os.path.basename(inv_inp_data_file) in os.listdir(ff.inversion_dir):
        if overwrite:
          hdul.writeto( inv_inp_data_file, overwrite=True)
    else:
        hdul.writeto(inv_inp_data_file, overwrite=True)
    print('Inverison input data file created: \n '+inv_inp_data_file)
    return()


def write_wht_file(xlam, directory):
    file=open(directory+'profiles.wht','w')
    for x in xlam:
       file.write(str(1)+'   {:>4.3f}'.format(x)+'    {:>1.3f}'.format(1)+'   {:>1.3f}'.format(0)+'    {:>1.3f}'.format(0)+'   {:>1.3f}'.format(1)+'\n')
   # correction_second_line = (dst.line_core[1]-dst.line_core[0])*dst.spectral_sampling
   # for x in xlam:
   #    file.write(str(2)+'   {:>4.3f}'.format(x+correction_second_line)+'    {:>1.3f}'.format(1)+'   {:>1.3f}'.format(0)+'    {:>1.3f}'.format(0)+'   {:>1.3f}'.format(1)+'\n')

    file.close()
    
    print('.wht file written')
    
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

def process_data_for_inversion(data,  scan=0, ff_p =ff,  
                                             wl_calibration = True,
                                             binning =  1, # scan pos, scan, x along slit, wavelength
                                             mask = True,
                                             pca = 0,  # define number of pca components
                                             continuum = True,  # correct continuum: intensity/I_c, V - V_c
                                             cut=True, 
                                             test = False, debug = False, save = True): #only one scan at a time
    

    if binning == 1:
       print("process_dataset: Processing data \n"+ff_p.v_file)
    else:
       print("process_dataset: Processing data \n"+ff_p.v_file+' \n with binning '+ str(binning)+'scans')
    print('----------------------')
    
    # figure out if it is a single scan
    single_scan=False
    if len(data.i.shape) == 3: 
       single_scan = True
       print('process_dataset: There is only one scan in the data. \n')
    if not single_scan:   
       # mask tellurics
        for tel in range(2):
         data.i[:,:,:,dst.tellurics[tel][0]:dst.tellurics[tel][1]] = -1
         data.v[:,:,:,dst.tellurics[tel][0]:dst.tellurics[tel][1]] = 0.0 # 

        
        i_c = data.i[:,:,20:-20,dst.continuum[0]:dst.continuum[1]].mean(axis=3)
        i_c = i_c.max()
        
        if continuum:
            data.i /= i_c
            for i in range(data.v.shape[0]):
               for m in range(data.v.shape[1]):
                 for n in range(data.v.shape[2]):
                     data.v[i,m,:,n] -= data.v[i,m,n,dst.continuum[0]:dst.continuum[1]].mean()
        
        if cut:
               data.i = data.i[dst.yroi[0]:dst.yroi[1],:,dst.roi[0]:dst.roi[1],dst.sroi[0]:dst.sroi[1]]
               data.v = data.v[dst.yroi[0]:dst.yroi[1],:,dst.roi[0]:dst.roi[1],dst.sroi[0]:dst.sroi[1]]

               if continuum:
                    data.i /= data.i.max()    
                            
        reduced_data = np.zeros((data.i.shape[3], 4, data.i.shape[0], data.i.shape[2] ) ) 
        # stokes, wvl, x, y
        xlam = 1000*(np.arange( data.i.shape[3]) - dst.line_core[0]+dst.sroi[0])*dst.spectral_sampling # to be used in the wavelength grid for SIR
        
        print('According to the parameters in themis_datasets:'+'\n'+\
              'start wavelength [mA]: '+str(xlam[0]) +'\n' + \
              'wl step [mA]: '+str(1000*dst.spectral_sampling)+'\n' + \
               'end wavelength [mA]: '+str(xlam[-1]) +'\n'  )
        if wl_calibration:
            wl_reference, ref_spectrum, _ = read_fts5(6299, 6305)
            spectrum = data.i.mean(axis=(0,1,2))
            spectrum/=spectrum.max()
            xlam = z3ccspectrum(spectrum,  wl_reference, ref_spectrum, FACL=0.8, FACH=1.5, FACS=0.005, 
                                   CUT=[10,20], DERIV=None, CONT=1.01, SHOW=2)
            
        for i in range(reduced_data.shape[0]):
            for m in range(reduced_data.shape[2]):
              reduced_data[i,0, m,:] = data.i[m,scan,:,i]
              
              reduced_data[i,3,m,:] = data.v[m,scan,:,i]
              
        if pca > 0:
            data_line_1, mean_1=prep_data_for_pca(reduced_data[int(reduced_data.shape[0]/2):,3, :])
            data_line_2, mean_2=prep_data_for_pca(reduced_data[:int(reduced_data.shape[0]/2),3, :])
            
            pca_line_1, transformed_line_1, inverted_line_1 = run_pca_s(data_line_1, pca)
            pca_line_2, transformed_line_2, inverted_line_2 = run_pca_s(data_line_2, pca)
            
            X_pca_1 = pca_line_1.transform(data_line_1)
            X_reconstructed_1 = pca_line_1.inverse_transform(X_pca_1) + mean_1
            
            X_pca_2 = pca_line_2.transform(data_line_2)
            X_reconstructed_2 = pca_line_2.inverse_transform(X_pca_2) + mean_2

            reduced_data[int(reduced_data.shape[0]/2):,3,:] = X_reconstructed_1.reshape((int(reduced_data.shape[0]/2),reduced_data.shape[2],reduced_data.shape[3]))
            reduced_data[:int(reduced_data.shape[0]/2),3,:] = X_reconstructed_2.reshape((int(reduced_data.shape[0]/2),reduced_data.shape[2],reduced_data.shape[3]))
  
        if save:
            
            save_inv_input_data(reduced_data, xlam,ff_p, scan, overwrite = True)
        
    return(reduced_data, xlam, ff_p)    
    
    
    