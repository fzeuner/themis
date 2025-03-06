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
import astropy.units as u

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
    data = l1_data()
    data.ui =  np.zeros((pol_states,steps,int(dummy.shape[0]/pol_states/steps), int(dummy.shape[1]/2), dummy.shape[2]))
    data.li =  1.*data.ui
    for i in range(pol_states):
        
     data_dummy = dummy[i::pol_states]
     
     for s in range(steps):
         dummy_dummy = data_dummy[s::steps]
         data.li[i,s,:] = dummy_dummy[:,0:int(dummy.shape[1]/2),:]
         data.ui[i,s,:] = dummy_dummy[:,int(dummy.shape[1]/2):,:]
    
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


def read_single_stokes(dataset, scan_position=0, scan=0 ):
    
    if dataset.array_axis_physical_types[0][0] != 'phys.polarization.stokes':
           print('read_single_stokes: dataset has the wrong order!')
           
    if dataset.array_axis_physical_types[1][2] != 'time':
           print('read_single_stokes: dataset has the wrong order!')
           
    if dataset.array_axis_physical_types[2][2] != 'time':
               print('read_single_stokes: dataset has the wrong order!')
    
    if dataset.array_axis_physical_types[3][0] != 'em.wl':
           print('read_single_stokes: dataset has the wrong order!')
    
    xmin=dataset[:,scan,scan_position,:].headers[0]['WAVEMIN'] #nm
    xmax=dataset[:,scan,scan_position,:].headers[0]['WAVEMAX'] #nm 
    
    stokes=np.array(dataset[:,scan,scan_position,:].data)
    # stokes[1,:]=100.*stokes[1,:]
    # stokes[2,:]=100.*stokes[2,:]
    # stokes[3,:]=100.*stokes[3,:]
    xlam=10.*np.arange(xmin,xmax,(xmax-xmin)/stokes[0,:].shape[0])
    
    if xlam.shape[0] > stokes[0,:,0].shape[0]:
        xlam = xlam[:-1]
    return(stokes, xlam, dataset[:,scan,scan_position,:].headers[0]) # ordering: stokes, wvl, spatial

def read_single_stokes_single_scan(dataset, scan_position=0 ): # if there is only one scan
    
    if dataset.array_axis_physical_types[0][0] != 'phys.polarization.stokes':
           print('read_single_stokes: dataset has the wrong order!')
           
    if dataset.array_axis_physical_types[1][2] != 'time':
           print('read_single_stokes: dataset has the wrong order!')
    
    if dataset.array_axis_physical_types[2][0] != 'em.wl':
           print('read_single_stokes: dataset has the wrong order!')
    
    xmin=dataset[:,scan_position,:].headers[0]['WAVEMIN'] #nm
    xmax=dataset[:,scan_position,:].headers[0]['WAVEMAX'] #nm 
    
    stokes=np.array(dataset[:,scan_position,:].data)
    # stokes[1,:]=100.*stokes[1,:]
    # stokes[2,:]=100.*stokes[2,:]
    # stokes[3,:]=100.*stokes[3,:]
    xlam=10.*np.arange(xmin,xmax,(xmax-xmin)/stokes[0,:].shape[0])
    
    if xlam.shape[0] > stokes[0,:,0].shape[0]:
        xlam = xlam[:-1]
    return(stokes, xlam, dataset[:,scan_position,:].headers[0]) # ordering: stokes, wvl, spatial


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
       
    
    
    
    