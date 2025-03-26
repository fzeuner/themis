#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:24:27 2025

@author: zeuner
"""

import themis_tools as tt
import themis_display_scan as display
import themis_display_v as display_v
import themis_datasets as dst
from lmfit import Model #conda install -c conda-forge lmfit

from style_themis import style_aa
import matplotlib.pyplot as plt
from matplotlib.ticker import (NullFormatter)
from matplotlib import gridspec
import numpy as np

from tqdm import tqdm
import gc


def plot_minima(data):
    zoom=1
    plt.close()
    my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
    plt.style.use('default')
    plt.rcParams.update(params)
    nullfmt=NullFormatter()
    
    fig = plt.figure(num=2, layout="constrained")
    fig.set_size_inches([fig_size_single[0],fig_size_single[1]],forward=True)
    fig.clf()
    
    gs = gridspec.GridSpec(0,1, figure=fig)#width_ratios=[1,0.25,1,0.25])# , height_ratios=[1,1,1,1] )
    gs.update(top=0.99, bottom=0.14,left=0.11, right=0.95, wspace=0.17, hspace=0.12)
    
    ax = fig.add_subplot(gs[0, 0])
    
    ax.plot()
        
    fig.show()

    return(fig)

def gaus(x, amp, mean, sigma, offset, gradient):
    return(offset+gradient*x+amp*np.exp(-(x-mean)**2/(2*sigma**2)))

class line():
    
    def __init__(self, name, spectrum, idx, wvl, fit = True):
    # data: scan positions, scans, slit, wavelength

        self.name = name

        self.wvl = wvl[idx[0]:idx[1]]
        
        self.spectrum = spectrum[:,:,:,idx[0]:idx[1]]
 
        self.fitted = False
            
        self.fit = 0.*self.spectrum
        
        # fit Gaussians at each position
        
        self.center    = np.zeros((self.spectrum.shape[0],self.spectrum.shape[1],self.spectrum.shape[2] ))
      
        self.width     = 0.*self.center 
        
        self.amplitude = 0.*self.center
       
        if fit:
        
            # fit Gaussians  
            self.fitted = True
            
            model=Model(gaus)       
            
            mean = self.wvl[np.where(self.spectrum.mean(axis=(0,1,2)) == self.spectrum.mean(axis=(0,1,2)).min())]
            
            params=model.make_params(amp=0.7,
                                     mean=mean, # center of line should coincide with middle of wavelength scale
                                     sigma=3*(self.wvl[1]-self.wvl[0]),  # 5 pixel is a typical width
                                     offset=self.spectrum.max(),
                                     gradient=0)
            params['amp'].vary=True
            params['mean'].vary=True
            params['sigma'].vary=True
            params['offset'].vary=True
            params['gradient'].vary=True
            
            params['sigma'].min=0.5*1*(wvl[1]-wvl[0])
            params['sigma'].max=1.5*4*(wvl[1]-wvl[0])
           
            for i in tqdm(range(self.spectrum.shape[0])):
             for s in range(self.spectrum.shape[1]):
               for p in range(self.spectrum.shape[2]):
                
                
                result=model.fit(self.spectrum[i,s,p,:],params,x=self.wvl)
                
                self.fit[i,s,p,:] = result.best_fit
                self.center[i,s,p] = result.params['mean']
                self.amplitude[i,s,p] = result.params['amp']
                self.width[i,s,p] = result.params['sigma']
              
        
#%%
#----------------------------------------------------------------------
"""
MAIN: 
   
"""
#---------------------------------------------------------------

# -----------------------------------------------------
# initialize files and folder object

ff = tt.ff()

#wvl = tt.z3ccspectrum(spectrum,  wavelengthscale, ref_spectrum, FACL=0.8, FACH=1.5, FACS=0.01, CUT=None, DERIV=None, CONT=None, SHOW=2)

#%%
# read original data

gc.collect()
data, header = tt.read_images_file(ff.red_data_file)

data.wvl = np.arange(data.ui.shape[4])
data.ui /=data.ui.max()
#%%
x_offset = [25, 11]

tl1_idx = [int(202),int(220)]
sl1_idx = [int(145),int(195)]
tl2_idx = [int(258),int(275)]
sl2_idx = [int(234),int(257)]
#%%

tl1_s0_ui=line('TL1, upper image, state 0', data.ui[0,:,:,x_offset[0]:-x_offset[1],:], tl1_idx, data.wvl, fit=True)
tl1_s0_ui_center = tl1_s0_ui.center

del tl1_s0_ui
gc.collect()

tl1_s0_li=line('TL1, lower image, state 0', data.li[0,:,:,x_offset[0]:-x_offset[1],:], tl1_idx, data.wvl, fit=True)
tl1_s0_li_center = tl1_s0_li.center

del tl1_s0_li
gc.collect()


sl1_s0_ui=line('SL1, upper image, state 0', data.ui[0,:,:,x_offset[0]:-x_offset[1],:], sl1_idx, data.wvl, fit=True)
sl1_s0_ui_center = sl1_s0_ui.center

del sl1_s0_ui
gc.collect()

sl1_s0_li=line('SL1, upper image, state 0', data.li[0,:,:,x_offset[0]:-x_offset[1],:], sl1_idx, data.wvl, fit=True)
sl1_s0_li_center = sl1_s0_li.center

del sl1_s0_li
gc.collect()


tl2_s0_ui=line('TL2, upper image, state 0', data.ui[0,:,:,x_offset[0]:-x_offset[1],:], tl2_idx, data.wvl, fit=True)
tl2_s0_ui_center = tl2_s0_ui.center

del tl2_s0_ui
gc.collect()

tl2_s0_li=line('TL2, upper image, state 0', data.li[0,:,:,x_offset[0]:-x_offset[1],:], tl2_idx, data.wvl, fit=True)
tl2_s0_li_center = tl2_s0_li.center

del tl2_s0_li
gc.collect()


tl2_s0_ui=line('SL2, upper image, state 0', data.ui[0,:,:,x_offset[0]:-x_offset[1],:], sl2_idx, data.wvl, fit=True)

tl2_s0_li=line('TL2, upper image, state 0', data.li[0,:,:,x_offset[0]:-x_offset[1],:], tl2_idx, data.wvl, fit=True)
tl2_s0_li_center = tl2_s0_li.center

del tl2_s0_li
gc.collect()

#%%
for i in range(tl1_s0_ui.spectrum.shape[0]):
 plt.plot(tl1_s0_ui.spectrum[i,0,0,:])
 plt.plot(tl1_s0_ui.fit[i,0,0,:])
 

#%%
for i in range(tl1_s0_ui.spectrum.shape[2]):
 plt.plot(tl1_s0_ui.center[:,0,i])
#%%
plt.imshow(tl2_s0_ui_center[:,0,:])


#%%
zoom=2
my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
plt.style.use('default')
plt.rcParams.update(params)
nullfmt=NullFormatter()

fig=plt.figure(num=34) 
fig.clf()
fig.set_size_inches([fig_size_single[0], 1.*fig_size_single[1]],forward=True)

gs = gridspec.GridSpec(1,2, )#width_ratios=[0.25,0.25,0.25,0.25] , height_ratios=[1,1,1,1] )
gs.update(top=0.96, bottom=0.1,left=0.1, right=0.55, wspace=0.05, hspace=0.26)
     
     
ax1=fig.add_subplot(gs[0,1]) 
ax0=fig.add_subplot(gs[0,0]) 

x = 5
y = 70
z = 10

ax0.plot(data.ui[0,x,z,y,tl1_idx[0]:tl1_idx[1]].T/data.ui[0,x,z,y,tl1_idx[0]:tl1_idx[1]].max(), linestyle='solid',label='upper image state 0')
ax0.plot(data.ui[1,x,z,y,tl1_idx[0]:tl1_idx[1]].T/data.ui[1,x,z,y,tl1_idx[0]:tl1_idx[1]].max(), linestyle='dashed', label='upper image state 1')

ax0.plot(data.li[0,x,z,y,tl1_idx[0]:tl1_idx[1]].T/data.li[0,x,z,y,tl1_idx[0]:tl1_idx[1]].max(), linestyle='dotted', label='lower image state 0')
ax0.plot(data.li[1,x,z,y,tl1_idx[0]:tl1_idx[1]].T/data.li[1,x,z,y,tl1_idx[0]:tl1_idx[1]].max(), linestyle='-.', label='lower image state 1')

ax1.plot(data.ui[0,x,z,y,tl2_idx[0]:tl2_idx[1]].T/data.ui[0,x,z,y,tl2_idx[0]:tl2_idx[1]].max(), linestyle='solid', label='upper image state 0')
ax1.plot(data.ui[1,x,z,y,tl2_idx[0]:tl2_idx[1]].T/data.ui[1,x,z,y,tl2_idx[0]:tl2_idx[1]].max(), linestyle='dashed', label='upper image state 1')

ax1.plot(data.li[0,x,z,y,tl2_idx[0]:tl2_idx[1]].T/data.li[0,x,z,y,tl2_idx[0]:tl2_idx[1]].max(), linestyle='dotted', label='lower image state 0')
ax1.plot(data.li[1,x,z,y,tl2_idx[0]:tl2_idx[1]].T/data.li[1,x,z,y,tl2_idx[0]:tl2_idx[1]].max(), linestyle='-.', label='lower image state 1')

ax0.legend(loc='best', bbox_to_anchor=(3.2, 0., 1, 1))