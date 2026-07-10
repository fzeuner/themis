#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: franziskaz

plots the ZIMPOL I, Q,U V image from GREGOR campaign

ver. 06.01.2025

"""


import numpy as np
import sys
import os

# Add scripts directory to path for importing SpectrumContainer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from astropy.io import fits
from matplotlib import gridspec
import matplotlib.pyplot as plt
from style_all import style_aa

from matplotlib.ticker import (NullFormatter,
                               MaxNLocator,
                               MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)

import line_tools as lt

from scipy.special import voigt_profile
from lmfit.models import VoigtModel, ConstantModel , SkewedVoigtModel

from themis.core import themis_tools as tt
from themis.core import themis_data_reduction as tdr
from themis.core import themis_io as tio
from themis.datasets.themis_datasets_2025 import get_config
from themis.plots import plot_power_spectrum

from process_formation_height_line_levels import SpectrumContainer

###################
# Definitions
###################

def initialize_parameter(params, ti_center_wl, continuum):
   
    params['A_amplitude'].value= -0.09
    params['A_center'].value = ti_center_wl
    params['A_sigma'].value= 0.03
    params['A_gamma'].value= 1
    #params['A_skew'].value= 1
    # Apply parameter constraints before fitting
    # Constrain the Ti line center to the given center wavelength
    # params['A_center'].min = ti_center_wl - 0.01
    # params['A_center'].max = ti_center_wl + 0.01
    # params['A_center'].vary = True
    
    # params['A_parasite_amplitude'].value= -0.07
    # params['A_parasite_center'].value = 4536.60
    # params['A_parasite_sigma'].value= 0.03
    # params['A_parasite_gamma'].value= 1
    
    params['B_amplitude'].value= -0.09
    params['B_center'].value = 4536.268
    params['B_sigma'].value= 0.03
    params['B_gamma'].value= 1
    
    params['C_amplitude'].value= -0.09
    params['C_center'].value = 4536.050
    params['C_sigma'].value= 0.03
    params['C_gamma'].value= 1
    
    params['D_amplitude'].value= -0.05
    params['D_center'].value = 4536.09
    params['D_sigma'].value= 0.03
    params['D_gamma'].value= 1
    
    params['c_c'].value= continuum
    params['c_c'].vary = True
    params['c_c'].min = continuum*0.9
    params['c_c'].max = continuum*1.1

    return(params)

class line():
    
    def __init__(self, wvl, profile, center, width, continuum=None):
        
        self.wvl = wvl
        
        self.profile = profile
        
        if continuum is None:
            continuum = profile[-1]  # fallback to last point of profile
        if not np.isfinite(continuum):
            continuum = np.nanmean(profile)  # fallback if continuum is NaN/inf
        self.continuum = continuum

# fit blue and red part of profile with two different models
# voigt model has four Parameters: amplitude, center, sigma, and gamma, skewed:  skew
# skew is not helping too much..
        voigtA = VoigtModel(prefix='A_')  # Ti I
        pars = voigtA.make_params()
        # voigtAParasite = VoigtModel(prefix='A_parasite_')  # Ti I
        # pars.update(voigtAParasite.make_params())

        voigtB = VoigtModel(prefix='B_')
        pars.update(voigtB.make_params())
        voigtC = VoigtModel(prefix='C_')
        pars.update(voigtC.make_params())
        voigtD = VoigtModel(prefix='D_')
        pars.update(voigtD.make_params())

        const = ConstantModel(prefix='c_')
        pars.update(const.make_params())

        pars = initialize_parameter(pars, center, self.continuum)

        # build model, but only fit it for specific wavelength without red wing of T
        model_blue = voigtA  + voigtB + voigtC +  voigtD + const #+ voigtAParasite

        # build model, but only fit it for specific wavelength in red wing of T
        model_red = voigtA + const #+ voigtAParasite

        # find wavelength points which belong to the line
        idx_left = np.argmin(abs(self.wvl - (center-width/2.)))
        idx_right = np.argmin(abs(self.wvl - (center+width/2.)))

        # find approximate center of the line to cut into blue and red part
        idx_center_approx = np.argmin(self.profile[idx_left:idx_right])  # should be improved by fitting
        
        # Refine the minimum by fitting a parabola in a small region
        # around the approximate minimum (within idx_left and idx_right)
        parabola_half_width = 5  # number of points on each side
        idx_parabola_left = max(idx_left, idx_left + idx_center_approx - parabola_half_width)
        idx_parabola_right = min(idx_right, idx_left + idx_center_approx + parabola_half_width)
        
        wvl_parabola = self.wvl[idx_parabola_left:idx_parabola_right]
        profile_parabola = self.profile[idx_parabola_left:idx_parabola_right]
        
        # Fit 2nd degree polynomial: p(wvl) = a*wvl^2 + b*wvl + c
        # The minimum is at wvl = -b/(2*a)
        poly_coefs = np.polyfit(wvl_parabola, profile_parabola, 2)
        a, b, c = poly_coefs
        if a > 0:  # parabola opens upward, minimum exists
            wvl_min_parabola = -b / (2 * a)
        else:
            # Fallback to approximate minimum if parabola doesn't have a minimum
            wvl_min_parabola = self.wvl[idx_left + idx_center_approx]
        
        # Find the index of the parabola minimum in the full wavelength array
        self.idx_center = np.argmin(abs(self.wvl - wvl_min_parabola)) - idx_left
        
        # adding  more pixel to get the minimum right
        pixel_overlap = 2
        self.profile_red = self.profile[idx_left+self.idx_center-pixel_overlap:]
        self.wvl_red = self.wvl[idx_left+self.idx_center-pixel_overlap:]

        self.profile_blue = profile[:idx_left+self.idx_center+pixel_overlap]
        self.wvl_blue = self.wvl[:idx_left+self.idx_center+pixel_overlap]


        self.out_blue = model_blue.fit(self.profile_blue, pars, x=self.wvl_blue)
        
        # Print final parameters of blue fit
        print("--- Blue fit parameters ---")
        print(self.out_blue.fit_report())
        
       
        components_blue = self.out_blue.eval_components(x=self.wvl_blue)
        
        self.component_ti_blue =  components_blue['A_']+components_blue['c_']    
        self.out_red = model_red.fit(self.profile_red, pars, x=self.wvl_red )
        
        # for the red part of the profile to extract Ti, I need to construct a new profile
        
        components_blue_full = self.out_blue.eval_components(x=self.wvl)
        self.new_profile = self.profile - (components_blue_full['B_']+components_blue_full['C_']  +components_blue_full['D_'])
        self.out_red_ti = model_red.fit(self.new_profile[idx_left+self.idx_center-pixel_overlap:], pars, x=self.wvl_red ) # only red part of new profile
        
        # Print final parameters of red fit
        print("--- Red fit parameters ---")
        print(self.out_red_ti.fit_report())
        
        components_red = self.out_red_ti.eval_components(x=self.wvl_red)
        self.component_ti_red =  components_red['A_']+components_red['c_']
        
        

###########################
#
# Font

zoom=1 # zoom factor
my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
plt.style.use('default')
plt.rcParams.update(params)
nullfmt=NullFormatter()

intmeth = 'none' # imshow integration method

# -----------------------------
spectra = SpectrumContainer.load_all()
res = spectra['ti']['disk_center'].residual.data[70:,20:200,0]
r_w = spectra['ti']['disk_center'].residual.wvl[70:]
li = spectra['ti']['disk_center'].line.data[:,20:200,0]
l_w =  spectra['ti']['disk_center'].line.wvl

continuum = np.nanmean(spectra['ti']['disk_center'].continuum.data[:,20:200,0], axis=0)
si=np.concatenate((res,li))

# -----------------------------
si = np.transpose(si)
image_sz=np.array(si.shape)

wavelengthscale=10*np.concatenate((r_w,l_w)) # in A!

# -----------------------------
 
ti_center_wl = 4536.39 # Ti
cont_center_wl = wavelengthscale[-1]

spec_region_width=0.16 # /2 +/- center_wl

####################
# plotting parameter
##################

i_gscale=[0.2,0.95]# greyscale for si

line_color1 = 'red'
line_color2 = 'lightseagreen'
line_color3 = 'blue'
line_color4 = 'indigo'

# data processing

intergranule = line( wavelengthscale, si[175,:], ti_center_wl,  spec_region_width, continuum=continuum[175])
granule = line( wavelengthscale, si[60,:], ti_center_wl,  spec_region_width, continuum=continuum[60])


# components_intergranule = out_intergranule.eval_components(x=wavelengthscale)
# components_granule = out_granule.eval_components(x=wavelengthscale)

#%%
  # --------------------------------------------------------
  # PLOT STARTS
  # -------------------------------------------------------
f=plt.figure(num=1) 
f.set_size_inches(fig_size,forward=True)
plt.clf()

gs = gridspec.GridSpec(2,1)#, width_ratios=[1,1,1,1] , height_ratios=[1,1,1,1] )
gs.update(top=0.94, bottom=0.11,left=0.09, right=0.9, wspace=0.15, hspace=0.2)

  #--------------------------------------------------------

ax=f.add_subplot(gs[0,0])

ax.get_yaxis().set_visible(True)
 

  # ------------- Stokes I image
  
im=ax.imshow(si, cmap='gray',origin='lower',\
              interpolation=intmeth,vmin=i_gscale[0], vmax=i_gscale[1], 
              extent=[wavelengthscale[0],wavelengthscale[-1],0,image_sz[1]], aspect='auto')#  
pos=ax.get_position()
cbaxes = f.add_axes([pos.x1, pos.y0, zoom*0.01*(pos.x1-pos.x0), (pos.y1-pos.y0)])
cb = plt.colorbar(im,orientation='vertical',cax=cbaxes,ticks=[i_gscale[0],i_gscale[1]])
cb.set_label('$I/I_c$ [a.u.]')


ax.axvline(x=ti_center_wl-spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted')
ax.axvline(x=ti_center_wl+spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted')


# ax.axvline(x=cont_center_wl-spec_region_width/2., color=line_color3, linewidth=1, linestyle='dotted')
# ax.axvline(x=cont_center_wl+spec_region_width/2., color=line_color3, linewidth=1, linestyle='dotted')
ax.set_title('MTR@THEMIS, 2025, disk center')
ax.set_ylabel(r'X along slit [pixel]')
ax.get_xaxis().set_visible(False)

  # ------------- Stokes I spectrum
  
ax1=f.add_subplot(gs[1,0])  

ax1.plot( granule.wvl, granule.profile, color=line_color4, marker='+',linestyle='None', label = 'Granule') 
ax1.plot( granule.wvl_blue[:-2], granule.out_blue.best_fit[:-2], '-', color='blue', label='best fit granule blue')
ax1.plot( granule.wvl_red[2:], granule.out_red.best_fit[2:], '-', color='red', label='best fit granule red')
ax1.plot( granule.wvl_blue[:-2], granule.component_ti_blue[:-2], '-', marker='.', color='blue', label='best fit granule, Ti')
ax1.plot( granule.wvl_red[2:], granule.component_ti_red[2:],'-',  marker='.', color='red')
# ax1.plot(wavelengthscale, components_granule_red['A_']+components_granule['c_'], '-', color='blue', label='best fit granule, Ti')
# ax1.plot(wavelengthscale, components_granule_blue['A_']+components_granule['c_'], '-', color='magenta', label='best fit granule, Ti')



ax1.plot(intergranule.wvl, intergranule.profile, color=line_color2, marker='+',linestyle='None', label = 'Intergranule') 
ax1.plot( intergranule.wvl_blue[:-2], intergranule.out_blue.best_fit[:-2], '-', color='blue', label='best fit intergranule blue')
ax1.plot( intergranule.wvl_red[2:], intergranule.out_red.best_fit[2:], '-', color='red', label='best fit intergranule red')
ax1.plot( intergranule.wvl_blue[:-2], intergranule.component_ti_blue[:-2], '-', marker='.', color='green', label='best fit intergranule, Ti')
ax1.plot( intergranule.wvl_red[2:], intergranule.component_ti_red[2:],'-',  marker='.', color='magenta')
# ax1.plot(wavelengthscale, si[0,:].mean(axis=0), color='black', label = 'Spatial mean') 

ax.axhline(y=175, color=line_color2, linewidth=1, linestyle='dotted')
ax.axhline(y=60, color=line_color4, linewidth=1, linestyle='dotted')

ax1.axvline(x=ti_center_wl-spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted')
ax1.axvline(x=ti_center_wl+spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted', label='Ti I')
ax1.legend()


ax1.set_xlabel(r'Wavelength $[\AA]$')  
ax1.set_ylabel(r'Counts [a.u.]')


plt.savefig('/home/franziskaz/figures/themis/test_voigt_themis.png'
          , dpi=my_dpi, metadata={'Software': __file__})




