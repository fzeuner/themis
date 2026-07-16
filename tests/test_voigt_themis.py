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
from lmfit import Model
from lmfit.lineshapes import voigt as voigt_lineshape

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
    
    params['D_amplitude'].value= -0.09
    params['D_center'].value = 4535.9
    params['D_sigma'].value= 0.03
    params['D_gamma'].value= 1
    
    params['c_c'].value= continuum
    params['c_c'].vary = True
    params['c_c'].min = continuum*0.9
    params['c_c'].max = continuum*1.1

    return(params)

def initialize_piecewise_parameter(params, center, continuum):
   
    params['ti_r_amplitude'].value= -0.08
    params['ti_b_amplitude'].value= -0.08
    params['ti_r_center'].value = center
    params['ti_r_center'].vary = True
    params['ti_b_center'].expr = 'ti_r_center'
    params['ti_r_sigma'].value= 0.03
    params['ti_b_sigma'].value= 0.03
    params['ti_b_gamma'].value= 0.03
    params['ti_r_gamma'].value= 0.03
    params['ti_r_gamma'].min = 0.001
    params['ti_r_gamma'].max = 0.09
    params['ti_b_gamma'].min = 0.001
    params['ti_b_gamma'].max = 0.09
    # Initialize y to the center for each component
    params['ti_y'].expr = 'ti_r_center'

  
    params['B_r_amplitude'].value= -0.09
    params['B_b_amplitude'].value= -0.09
    params['B_r_center'].value = 4536.268
    params['B_r_center'].vary = True
    params['B_b_center'].expr = 'B_r_center'
    params['B_r_sigma'].value= 0.03
    params['B_b_sigma'].value= 0.03
    params['B_r_gamma'].value= 0.03
    params['B_b_gamma'].value= 0.03
    params['B_r_gamma'].min = 0.001
    params['B_r_gamma'].max = 0.09
    params['B_b_gamma'].min = 0.001
    params['B_b_gamma'].max = 0.09
    params['B_y'].expr = 'B_r_center'
    
    params['C_r_amplitude'].value= -0.09
    params['C_b_amplitude'].value= -0.09
    params['C_r_center'].value = 4536.050
    params['C_r_center'].vary = True
    params['C_b_center'].expr = 'C_r_center'
    params['C_r_sigma'].value= 0.03
    params['C_b_sigma'].value= 0.03
    params['C_r_gamma'].value= 0.03
    params['C_b_gamma'].value= 0.03
    params['C_y'].expr = 'C_r_center'
    
    params['D_r_amplitude'].value= -0.09
    params['D_b_amplitude'].value= -0.09
    params['D_r_center'].value = 4535.9
    params['D_r_center'].vary = True
    params['D_b_center'].expr = 'D_r_center'
    params['D_r_sigma'].value= 0.03
    params['D_b_sigma'].value= 0.03
    params['D_r_gamma'].value= 0.03
    params['D_b_gamma'].value= 0.03
    params['D_y'].expr = 'D_r_center'
    
    params['c_c'].value= continuum
    params['c_c'].vary = False
    params['c_c'].min = continuum*0.9
    params['c_c'].max = continuum*1.1

    return(params)


def _piecewise_voigt_func(x, y, r_amplitude, r_center, r_sigma, r_gamma,
                           b_amplitude, b_center, b_sigma, b_gamma,
                           transition_width=0.005):
    # Two independent Voigt profiles, selected pointwise by wavelength x
    # relative to the split point y (blue side: x < y, red side: x >= y)
    voigt_r = voigt_lineshape(x, r_amplitude, r_center, r_sigma, r_gamma)
    voigt_b = voigt_lineshape(x, b_amplitude, b_center, b_sigma, b_gamma)
    return np.where(x < y, voigt_b, voigt_r)

    # Smooth (sigmoid) blend alternative, kept for reference:
    # A tanh blend (instead of a hard np.where cutoff) avoids introducing
    # a discontinuity when the blue/red amplitude, sigma, or gamma differ,
    # which otherwise shows up as visible "kinks" when summing components.
    # frac_red = 0.5 * (1 + np.tanh((x - y) / transition_width))
    # return (1 - frac_red) * voigt_b + frac_red * voigt_r

def piecewise_voigt(prefix='line'): # fit y, the point where the split should appear
    model = Model(_piecewise_voigt_func, independent_vars=['x'], prefix=prefix+'_',
                  param_names=['y', 'r_amplitude', 'r_center', 'r_sigma', 'r_gamma',
                               'b_amplitude', 'b_center', 'b_sigma', 'b_gamma'],
                  )
    return model


def piecewise_model():
    
    voigt_ti = piecewise_voigt(prefix='ti')  # Ti I 
    pars = voigt_ti.make_params()
    voigtB = piecewise_voigt(prefix='B')
    pars.update(voigtB.make_params())
    voigtC = piecewise_voigt(prefix='C')
    pars.update(voigtC.make_params())
    voigtD = piecewise_voigt(prefix='D')
    pars.update(voigtD.make_params())
    
    const = ConstantModel(prefix='c_')
    pars.update(const.make_params())
    
    model = voigt_ti + voigtB + voigtC + voigtD + const
    
    return model, pars

def get_initial_piecewise_fit(wvl, center, continuum):
    """Generate the initial piecewise model fit (before optimization)."""
    model, pars = piecewise_model()
    pars = initialize_piecewise_parameter(pars, center, continuum)
    initial_fit = model.eval(pars, x=wvl)
    return initial_fit

def add_overlap_weights(wvl, params, base_weight=1.0, boost=3, boost_width=0.008): # keep boost_width small
    """Build a weights array that upweights the region around each component's
    split point y, where blue/red wings and neighboring components overlap
    the most. Use as `weights=` in model.fit().
    """
    weights = np.full_like(wvl, base_weight, dtype=float)
    for prefix in ['ti', 'B', 'C', 'D']:
        y_split = params[prefix+'_y'].value
        mask = np.abs(wvl - y_split) <= boost_width
        weights[mask] = np.maximum(weights[mask], boost)
    return weights

def two_models():
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


            # build model, but only fit it for specific wavelength without red wing of T
            model_blue = voigtA  + voigtB + voigtC +  voigtD + const #+ voigtAParasite

            # build model, but only fit it for specific wavelength in red wing of T
            model_red = voigtA + const #+ voigtAParasite
            
            return model_blue, model_red, pars

class line():
    
    def __init__(self, wvl, profile, center, width, continuum=None):
        
        self.wvl = wvl
        
        self.profile = profile
        
        if continuum is None:
            continuum = profile[-1]  # fallback to last point of profile
        if not np.isfinite(continuum):
            continuum = np.nanmean(profile)  # fallback if continuum is NaN/inf
        self.continuum = continuum

        model_blue, model_red, pars = two_models()
        
        pars = initialize_parameter(pars, center, self.continuum)

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
        pixel_overlap = 5
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
        
        
class line_piecewise():
    
    def __init__(self, wvl, profile, center, width, continuum=None):
        
        self.wvl = wvl
        
        self.profile = profile
        
        if continuum is None:
            continuum = profile[-1]  # fallback to last point of profile
        if not np.isfinite(continuum):
            continuum = np.nanmean(profile)  # fallback if continuum is NaN/inf
        self.continuum = continuum

        # Initialize y parameter close to center
        model, pars = piecewise_model()
        
        pars = initialize_piecewise_parameter(pars, center, self.continuum)

        # Upweight the region around each component's split point y, where
        # blue/red wings and neighboring components overlap the most.
        weights = add_overlap_weights(self.wvl, pars)
        self.out = model.fit(self.profile, pars, x=self.wvl, weights=weights)
        print(self.out.fit_report())
        
        # Evaluate the best fit and components on the original wavelength array
        self.best_fit = model.eval(self.out.params, x=self.wvl)
        components = model.eval_components(params=self.out.params, x=self.wvl)
        
        # Ti component
        self.component_ti = components['ti_']+ components['c_']
        
        # Parasitic components
        self.component_residual = components['B_'] + components['C_'] + components['D_']+ components['c_']
        
        

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
wvl_nm, si_full = spectra['ti']['disk_center'].reconstruct(('residual', 'line'))
si = si_full[:, 300, 20:100]

continuum = np.nanmean(spectra['ti']['disk_center'].continuum.data[:,300,20:100], axis=0)

# -----------------------------
si = np.transpose(si)
image_sz=np.array(si.shape)

wavelengthscale=10*wvl_nm # in A!

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

intergranule_idx = 39
granule_idx = 17
#%%
intergranule = line( wavelengthscale, si[intergranule_idx,:], ti_center_wl,  spec_region_width, continuum=continuum[intergranule_idx])
granule = line( wavelengthscale, si[granule_idx,:], ti_center_wl,  spec_region_width, continuum=continuum[granule_idx])
#%%
intergranule =  line_piecewise( wavelengthscale, si[intergranule_idx,:], ti_center_wl,  spec_region_width, continuum=continuum[intergranule_idx])
granule =  line_piecewise( wavelengthscale, si[granule_idx,:], ti_center_wl,  spec_region_width, continuum=continuum[granule_idx])
# components_intergranule = out_intergranule.eval_components(x=wavelengthscale)
# components_granule = out_granule.eval_components(x=wavelengthscale)
#%%
# Evaluate the initial model guess (before fitting) for diagnostic plotting
initial_guess = get_initial_piecewise_fit(wavelengthscale, ti_center_wl, continuum[intergranule_idx])
plt.plot(wavelengthscale,initial_guess)
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
              extent=[wavelengthscale[0],wavelengthscale[-1],0,image_sz[0]], aspect='auto')#  
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

ax1.plot( granule.wvl, granule.profile, color=line_color4, marker='+',linestyle='None', label = 'granule data') 
# ax1.plot( granule.wvl_blue[:-2], granule.out_blue.best_fit[:-2], '-', color='blue', label='best fit granule blue')
# ax1.plot( granule.wvl_red[2:], granule.out_red.best_fit[2:], '-', color='red', label='best fit granule red')
ax1.plot( granule.wvl, granule.best_fit, '-', color=line_color4, label='best fit granule')
ax1.plot( granule.wvl, initial_guess, '--', color='gray', label='initial guess')

# ax1.plot(wavelengthscale, components_granule_red['A_']+components_granule['c_'], '-', color='blue', label='best fit granule, Ti')
# ax1.plot(wavelengthscale, components_granule_blue['A_']+components_granule['c_'], '-', color='magenta', label='best fit granule, Ti')

ax1.plot( granule.wvl, granule.component_ti, '-', marker='.', color='red', label='best fit granule, Ti')
ax1.plot( granule.wvl, granule.component_residual,'-', color='red', alpha=0.2,label='best fit granule, residuals')

ax1.plot(intergranule.wvl, intergranule.profile, color=line_color2, marker='+',linestyle='None', label = 'intergranule, data') 
ax1.plot( intergranule.wvl, intergranule.best_fit, '-', color=line_color2, label='best fit intergranule')

ax1.plot( intergranule.wvl, intergranule.component_ti, '-', marker='.', color='magenta', label='best fit intergranule, Ti')
ax1.plot( intergranule.wvl, intergranule.component_residual,'-', color='magenta', alpha=0.2,label='best fit intergranule, residuals')


# ax1.plot( intergranule.wvl_blue[:-2], intergranule.out_blue.best_fit[:-2], '-', color='blue', label='best fit intergranule blue')
# ax1.plot( intergranule.wvl_red[2:], intergranule.out_red.best_fit[2:], '-', color='red', label='best fit intergranule red')
# ax1.plot( intergranule.wvl_blue[:-2], intergranule.component_ti_blue[:-2], '-', marker='.', color='green', label='best fit intergranule, Ti')
# ax1.plot( intergranule.wvl_red[2:], intergranule.component_ti_red[2:],'-',  marker='.', color='magenta')
# ax1.plot(wavelengthscale, si[0,:].mean(axis=0), color='black', label = 'Spatial mean') 

ax.axhline(y=intergranule_idx, color=line_color2, linewidth=1, linestyle='dotted')
ax.axhline(y=granule_idx, color=line_color4, linewidth=1, linestyle='dotted')

ax1.axvline(x=ti_center_wl-spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted')
ax1.axvline(x=ti_center_wl+spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted', label='Ti I')
ax1.legend()


ax1.set_xlabel(r'Wavelength $[\AA]$')  
ax1.set_ylabel(r'Counts [a.u.]')


plt.savefig('/home/franziskaz/figures/themis/test_voigt_themis.png'
          , dpi=my_dpi, metadata={'Software': __file__})


#%%
  # --------------------------------------------------------
  # PLOT STARTS: is there a blend in Ti in intergranules?
  # -------------------------------------------------------
f=plt.figure(num=2) 
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
              extent=[wavelengthscale[0],wavelengthscale[-1],0,image_sz[0]], aspect='auto')#  
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

ax1.plot( granule.wvl, granule.profile, color='magenta', marker='+',linestyle='solid', label = 'Granule') 
# ax1.plot(wavelengthscale, components_granule_red['A_']+components_granule['c_'], '-', color='blue', label='best fit granule, Ti')
# ax1.plot(wavelengthscale, components_granule_blue['A_']+components_granule['c_'], '-', color='magenta', label='best fit granule, Ti')



ax1.plot(intergranule.wvl, intergranule.profile, color='black', marker='+',linestyle='solid', label = 'Intergranule') 

# ax1.plot(wavelengthscale, si[0,:].mean(axis=0), color='black', label = 'Spatial mean') 

ax.axhline(y=granule_idx, color='magenta', linewidth=1, linestyle='dotted')
ax.axhline(y=intergranule_idx, color='black', linewidth=1, linestyle='dotted')

ax1.axvline(x=ti_center_wl-spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted')
ax1.axvline(x=ti_center_wl+spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted', label='Ti I')
ax1.legend()


ax1.set_xlabel(r'Wavelength $[\AA]$')  
ax1.set_ylabel(r'Counts [a.u.]')


plt.savefig('/home/franziskaz/figures/themis/test_voigt_themis-blend.png'
          , dpi=my_dpi, metadata={'Software': __file__})




