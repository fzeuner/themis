#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:24:27 2025

@author: zeuner
"""

import themis_tools as tt

from lmfit import Model #conda install -c conda-forge lmfit

from style_themis import style_aa
import matplotlib.pyplot as plt
from matplotlib.ticker import (NullFormatter)
from matplotlib import gridspec
import numpy as np
import scipy.ndimage






#%%
#----------------------------------------------------------------------
"""
MAIN: 
    Test wavelength shifts for THEMIS
   
"""
#---------------------------------------------------------------

# -----------------------------------------------------
# initialize files and folder object

def gaus(x, amp, mean, sigma, offset, gradient):
    return(offset+gradient*x+amp*np.exp(-(x-mean)**2/(2*sigma**2)))

shift = 0.001

wvl_hres = np.arange(6300,6304,0.001)
line_fe1_hres=gaus(wvl_hres,-0.8,6301.5008, 0.07,1,0)
line_fe2_hres=gaus(wvl_hres,-0.7,6302.4932, 0.071,1,0)

line_t1_hres=gaus(wvl_hres,-0.5,6301.95, 0.02,1,0)
line_t2_hres=gaus(wvl_hres,-0.5,6302.7, 0.02,1,0)

spectra_hres = line_fe1_hres + line_fe2_hres +  line_t1_hres + line_t2_hres
spectra_hres /=spectra_hres.max()


wvl_lres = np.arange(6300,6304,0.01)
line_fe1_lres=gaus(wvl_lres,-0.8,6301.5008, 0.07,1,0)
line_fe2_lres=gaus(wvl_lres,-0.7,6302.4932, 0.071,1,0)

line_t1_lres=gaus(wvl_lres,-0.5,6301.95, 0.02,1,0)
line_t2_lres=gaus(wvl_lres,-0.5,6302.7, 0.02,1,0)

spectra_lres = line_fe1_lres + line_fe2_lres +  line_t1_lres + line_t2_lres
spectra_lres /=spectra_lres.max()

zoom=2
plt.close()
my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
plt.style.use('default')
plt.rcParams.update(params)
nullfmt=NullFormatter()
        
fig = plt.figure(num=2)
fig.set_size_inches([fig_size_single[0],fig_size_single[1]],forward=True)
fig.clf()
        
gs = gridspec.GridSpec(1,1, figure=fig)#width_ratios=[1,0.25,1,0.25])# , height_ratios=[1,1,1,1] )
gs.update(top=0.95, bottom=0.18,left=0.15, right=0.95, wspace=0.17, hspace=0.12)
        
ax = fig.add_subplot(gs[0, 0])
        


ax.plot(wvl_hres, spectra_hres)

ax.legend()
ax.set_ylabel('Intensity', labelpad=4*zoom)
ax.set_xlabel(r'Wavelength [\AA]', labelpad=4*zoom)
            
fig.show()
 

#%%

zoom=2
plt.close()
my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
plt.style.use('default')
plt.rcParams.update(params)
nullfmt=NullFormatter()
        
fig = plt.figure(num=3)
fig.set_size_inches([fig_size_single[0],fig_size_single[1]],forward=True)
fig.clf()
        
gs = gridspec.GridSpec(1,1, figure=fig)#width_ratios=[1,0.25,1,0.25])# , height_ratios=[1,1,1,1] )
gs.update(top=0.95, bottom=0.18,left=0.15, right=0.95, wspace=0.17, hspace=0.12)
        
ax = fig.add_subplot(gs[0, 0])

ax.plot(wvl_hres, spectra_hres-scipy.ndimage.shift(spectra_hres,shift/(wvl_hres[1]-wvl_hres[0])), label='hres')

ax.plot(wvl_lres, spectra_lres-scipy.ndimage.shift(spectra_lres,shift/(wvl_lres[1]-wvl_lres[0])), label='lres')
ax.set_xlim((6300.5,6303))
ax.set_ylim((-0.15,0.15))
ax.legend()
ax.set_ylabel('Intensity diff', labelpad=4*zoom)
ax.set_xlabel(r'Wavelength [\AA]', labelpad=4*zoom)
            
fig.show()
