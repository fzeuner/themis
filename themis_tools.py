#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 05 2025

@author: franziskaz

This toolbox contains useful functions

Version: 2025-3-05
"""

from astropy.time import Time

#from sunpy.net import Fido, attrs as a
import matplotlib.pyplot as plt
from astropy.io import fits 
import numpy as np
from style_themis import style_aa
from matplotlib.ticker import (NullFormatter)
# from mpl_point_clicker import clicker
# from mpl_interactions import zoom_factory, panhandler
from scipy.signal import convolve2d
from scipy.signal import windows
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import datasets as dst
from tqdm import tqdm
import matplotlib.colors as mplcolor
#import astropy.units as u
from scipy import signal
import themis_datasets as dst
from skimage.registration import phase_cross_correlation

from matplotlib import gridspec

import os

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

def find_all_files(dir_name, stokes='I'):
    list_of_files = os.listdir(dir_name) #list of files in the current directory
    num=0
    filename=[]

    for file in list_of_files:
     
       if "_"+stokes+"_" in file:  
     
            filename.append(file)
           
            num+=1
            
    return(filename, num)
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

    correlold = 100.  # high value

    if DERIV:
            dyatlas = np.gradient(yatlas)

    for fac in np.arange(facL, facH, facs):
          
          
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

def find_all_files(dir_name, stokes='I'):
    list_of_files = os.listdir(dir_name) #list of files in the current directory
    num=0
    filename=[]

    for file in list_of_files:
     
       if "_"+stokes+"_" in file:  
     
            filename.append(file)
           
            num+=1
            
    return(filename, num)

def align_scans(data): # with continuum images
    
    template = data.i[:,0,:, dst.continuum[0]:dst.continuum[1]].mean(axis=2)
    # go through all scans
    max_x_shift=0
    max_y_shift=0
    mask=
    for i in range(data.i.shape[1]):
       img = data.i[:,i,:, dst.continuum[0]:dst.continuum[1]].mean(axis=2)
       detected_shift, _, _ = phase_cross_correlation( img, template )
       
       max_x_shift += abs(detected_shift[0])
       if abs(detected_shift[1]) > max_y_shift:
           max_y_shift = abs(detected_shift[0])  
      
       data.i[:,i,:,:]=np.roll(np.roll(data.i[:,i,:,:], detected_shift[0], axis=0), detected_shift[1], axis=1)
       data.v[:,i,:,:] = np.roll(np.roll(data.v[:,i,:,:], detected_shift[0], axis=0), detected_shift[1], axis=1)
       data.i[0:htemplate,i,0:wtemplate,:] = dummy_data
       data.v[0:htemplate,i,0:wtemplate,:] = dummy_data_v
       template = data.i[:,i,:, dst.continuum[0]:dst.continuum[1]].mean(axis=2)
       
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

def add_axis_scan_contour(fig, ax, data, data2, threshold,label, stokes=False, zoom=1,clim=[0,0], xvis='no', yvis='no',x_offset=0, title=None, total=False):
    
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
    linewidth_contour=1.2*zoom
    alpha_value=0.7
    
    clim_plot=[fct*dmin,fct*dmax]
    if abs(clim[0] - clim[1]) > 10e-8:
        clim_plot=clim
    
    extent=[0,image_sz[1]*dst.pixel[dst.line],0,image_sz[0]*dst.slitwidth]

    if title != None:
        ax.set_title(title)
    
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
            ax.set_xlabel(xvis, labelpad=-0.1)  
    if yvis == 'no':
        ax.get_yaxis().set_visible(False)
    else:
        ax.get_yaxis().set_visible(True)
        if yvis != 'yes':         
            ax.set_ylabel(yvis, labelpad=-0.1)  
   
    
    ax.contour(data2, threshold, colors=('black','green','darkviolet','green', 'cyan' ), origin='lower', extent=extent, linewidths= linewidth_contour, alpha=alpha_value)

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







def plot_stokes_contours(data, threshold, wl, pixel=1,zoom=1, fnum=0, title=None, save_figure=[False,'None',__file__]):
    
    plt.close(fnum)
    my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
    plt.style.use('default')
    plt.rcParams.update(params)
    nullfmt=NullFormatter()
    
    fig, axs = plt.subplots(4,constrained_layout=True, num=3)
    fig.set_size_inches(fig_size,forward=True)
    
    stokes_str=[r'$I$',r'$Q/I$ [$\%$]',r'$U/I$ [$\%$]',r'$V/I$ [$\%$]']
    titles=[title, None, None, None]
    yvis=[r'Wavelength $[\mathrm{\AA}]$', r'Wavelength $[\mathrm{\AA}]$', r'Wavelength $[\mathrm{\AA}]$', r'Wavelength $[\mathrm{\AA}]$']
    #yvis=[r'arcsec', r'arcsec',r'arcsec', r'arcsec']
    xvis=[r'no', r'no',r'no', r'pixel']
    data2=data[0,:] # reference    
    for i,s in enumerate(stokes_str):
        ax=add_axis_contour(fig,axs[i],data[i,:], data2, threshold, zoom, wl,s, pixel,xvis=xvis[i], yvis=yvis[i], title=titles[i])
        # if i ==2:
        #     ax.contour(data[2,:], np.arange(0.2,0.35, 0.1), colors='red', origin='upper', alpha=0.5)
    fig.show()
    
    if save_figure[0]:
        
       fig.savefig(save_figure[1], 
              dpi=my_dpi, metadata={'Software': save_figure[2]})
    return(fig)



class ff:
    # Class object to store folder and file names
 def __init__(self):
         
   self.directory = ''
   self.vbi_directory = ''
   self.co_file = ''
   self.red_data_file = ''
   self.date = ''
   self.line_file = ''  # line processing
   self.xlam_file = 'xlam'
   self.figure_dir = ''
   self.v_file= ''
       
    
    
    
    