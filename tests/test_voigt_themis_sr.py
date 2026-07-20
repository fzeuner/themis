#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: franziskaz

test piecewise voigt fitting for sr

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


from lmfit.models import ConstantModel
from lmfit import Model
from lmfit.lineshapes import voigt as voigt_lineshape

from process_formation_height_line_levels import SpectrumContainer, refine_center_with_parabola

###################
# Definitions
###################

def initialize_piecewise_parameter(params, center, continuum, wvl=None, profile=None):

    params['sr_r_amplitude'].value= -0.08
    params['sr_r_amplitude'].min= -0.1
    params['sr_r_amplitude'].max= -0.03

    params['sr_b_amplitude'].value= -0.08
    params['sr_b_amplitude'].min= -0.1
    params['sr_b_amplitude'].max= -0.03

    params['sr_r_center'].value = center
    params['sr_r_center'].vary = True
    params['sr_r_center'].min = center - 0.005
    params['sr_r_center'].max = center + 0.005
    params['sr_b_center'].expr = 'sr_r_center'

    params['sr_r_sigma'].value= 0.03
    params['sr_b_sigma'].value= 0.03
    params['sr_r_sigma'].max= 0.05
    params['sr_b_sigma'].max= 0.05
    params['sr_r_sigma'].min= 0.01
    params['sr_b_sigma'].min= 0.01

    params['sr_b_gamma'].value= 0.03
    params['sr_r_gamma'].value= 0.03
    params['sr_r_gamma'].min = 0.001
    params['sr_r_gamma'].max = 0.09
    params['sr_b_gamma'].min = 0.001
    params['sr_b_gamma'].max = 0.09
    # Initialize y to the center for each component
    params['sr_y'].expr = 'sr_r_center'


    params['B_r_amplitude'].value= -0.09
    params['B_b_amplitude'].value= -0.09
    params['B_r_amplitude'].max= -0.03
    params['B_r_amplitude'].min= -0.1
    params['B_b_amplitude'].max= -0.03
    params['B_b_amplitude'].min= -0.1

    # Use parabola-based initialization for B component center if wvl and profile are provided
    b_center_guess = 4607.65
    if wvl is not None and profile is not None:
        b_center_refined = refine_center_with_parabola(wvl, profile, b_center_guess, search_width=0.07)
        params['B_r_center'].value = b_center_refined
    else:
        params['B_r_center'].value = b_center_guess

    params['B_r_center'].vary = True
    params['B_b_center'].expr = 'B_r_center'

    params['B_r_sigma'].value= 0.03
    params['B_b_sigma'].value= 0.03
    params['B_r_sigma'].max= 0.05
    params['B_b_sigma'].max= 0.05
    params['B_r_sigma'].min= 0.01
    params['B_b_sigma'].min= 0.01

    params['B_r_gamma'].value= 0.03
    params['B_b_gamma'].value= 0.03
    params['B_r_gamma'].min = 0.001
    params['B_r_gamma'].max = 0.09
    params['B_b_gamma'].min = 0.001
    params['B_b_gamma'].max = 0.09

    params['B_y'].expr = 'B_r_center'


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
    
    voigt_sr = piecewise_voigt(prefix='sr')  # Sr I 
    pars = voigt_sr.make_params()
    voigtB = piecewise_voigt(prefix='B')
    pars.update(voigtB.make_params())
    
    const = ConstantModel(prefix='c_')
    pars.update(const.make_params())
    
    model = voigt_sr + voigtB + const
    
    return model, pars

def add_overlap_weights(wvl, params, base_weight=1.0, boost=3, boost_width=0.008): # keep boost_width small
    """Build a weights array that upweights the region around each component's
    split point y, where blue/red wings and neighboring components overlap
    the most. Use as `weights=` in model.fit().
    """
    weights = np.full_like(wvl, base_weight, dtype=float)
    for prefix in ['sr', 'B']:
        y_split = params[prefix+'_y'].value
        mask = np.abs(wvl - y_split) <= boost_width
        weights[mask] = np.maximum(weights[mask], boost)
    return weights


class line_piecewise():

    def __init__(self, wvl, profile, center, width, continuum=None):

        self.wvl = wvl

        self.profile = profile

        if continuum is None:
            continuum = profile[-1]  # fallback to last point of profile
        if not np.isfinite(continuum):
            continuum = np.nanmean(profile)  # fallback if continuum is NaN/inf
        self.continuum = continuum

        # Refine center using parabola fit (like in line class)
        idx_left = np.argmin(abs(self.wvl - (center-width/2.)))
        idx_right = np.argmin(abs(self.wvl - (center+width/2.)))

        # find approximate center of the line
        idx_center_approx = np.argmin(self.profile[idx_left:idx_right])

        # Refine the minimum by fitting a parabola
        parabola_half_width = 5
        idx_parabola_left = max(idx_left, idx_left + idx_center_approx - parabola_half_width)
        idx_parabola_right = min(idx_right, idx_left + idx_center_approx + parabola_half_width)

        wvl_parabola = self.wvl[idx_parabola_left:idx_parabola_right]
        profile_parabola = self.profile[idx_parabola_left:idx_parabola_right]

        poly_coefs = np.polyfit(wvl_parabola, profile_parabola, 2)
        a, b, c = poly_coefs
        if a > 0:
            center_refined = -b / (2 * a)
        else:
            center_refined = self.wvl[idx_left + idx_center_approx]

        # Initialize y parameter close to center
        model, pars = piecewise_model()


        pars = initialize_piecewise_parameter(pars, center_refined, self.continuum, wvl=self.wvl, profile=self.profile)

        # Upweight the region around each component's split point y, where
        # blue/red wings and neighboring components overlap the most.
        weights = add_overlap_weights(self.wvl, pars)
        self.out = model.fit(self.profile, pars, x=self.wvl, weights=weights)
        print(self.out.fit_report(show_correl=False))
        
        # Evaluate the best fit and components on the original wavelength array
        self.best_fit = model.eval(self.out.params, x=self.wvl)
        components = model.eval_components(params=self.out.params, x=self.wvl)
        
        # Ti component
        self.component_ti = components['sr_']+ components['c_']
        
        # Parasitic components
        self.component_residual = components['B_'] + components['c_']
        
        

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
wavelengthscale, si_full = spectra['sr']['disk_center'].reconstruct(('residual', 'line'))
si = si_full[:, 300, 20:100]

continuum = np.nanmean(spectra['sr']['disk_center'].continuum.data[:,300,20:100], axis=0)

# -----------------------------
si = np.transpose(si)
image_sz=np.array(si.shape)

# -----------------------------
 
sr_center_wl = 4607.34 # Sr
cont_center_wl = wavelengthscale[-1]

spec_region_width=0.16 # /2 +/- center_wl

####################
# plotting parameter
##################

i_gscale=[0.2,1.05]# greyscale for si

line_color1 = 'red'
line_color2 = 'lightseagreen'
line_color3 = 'blue'
line_color4 = 'indigo'

# data processing

intergranule_idx = 40
granule_idx = 16
#%%
intergranule =  line_piecewise( wavelengthscale, si[intergranule_idx,:], sr_center_wl,  spec_region_width, continuum=continuum[intergranule_idx])
granule =  line_piecewise( wavelengthscale, si[granule_idx,:], sr_center_wl,  spec_region_width, continuum=continuum[granule_idx])
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
              extent=[wavelengthscale[0],wavelengthscale[-1],0,image_sz[0]], aspect='auto')#  
pos=ax.get_position()
cbaxes = f.add_axes([pos.x1, pos.y0, zoom*0.01*(pos.x1-pos.x0), (pos.y1-pos.y0)])
cb = plt.colorbar(im,orientation='vertical',cax=cbaxes,ticks=[i_gscale[0],i_gscale[1]])
cb.set_label('$I/I_c$ [a.u.]')


ax.axvline(x=sr_center_wl-spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted')
ax.axvline(x=sr_center_wl+spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted')

ax.set_title('MTR@THEMIS, 2025, disk center')
ax.set_ylabel(r'X along slit [pixel]')
ax.get_xaxis().set_visible(False)

  # ------------- Stokes I spectrum
  
ax1=f.add_subplot(gs[1,0])  

ax1.plot( granule.wvl, granule.profile, color=line_color4, marker='+',linestyle='None', label = 'granule data') 
ax1.plot( granule.wvl, granule.best_fit, '-', color=line_color4, label='best fit granule')
#ax1.plot( granule.wvl, initial_guess, '--', color='gray', label='initial guess')

ax1.plot( granule.wvl, granule.component_ti, '-', marker='.', color='red', label='best fit granule, Sr')
ax1.plot( granule.wvl, granule.component_residual,'-', color='red', alpha=0.2,label='best fit granule, residuals')

ax1.plot(intergranule.wvl, intergranule.profile, color=line_color2, marker='+',linestyle='None', label = 'intergranule, data') 
ax1.plot( intergranule.wvl, intergranule.best_fit, '-', color=line_color2, label='best fit intergranule')

ax1.plot( intergranule.wvl, intergranule.component_ti, '-', marker='.', color='magenta', label='best fit intergranule, Sr')
ax1.plot( intergranule.wvl, intergranule.component_residual,'-', color='magenta', alpha=0.2,label='best fit intergranule, residuals')


ax.axhline(y=intergranule_idx, color=line_color2, linewidth=1, linestyle='dotted')
ax.axhline(y=granule_idx, color=line_color4, linewidth=1, linestyle='dotted')

ax1.axvline(x=sr_center_wl-spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted')
ax1.axvline(x=sr_center_wl+spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted', label='Sr I')
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

ax1.plot(intergranule.wvl, intergranule.profile, color='black', marker='+',linestyle='solid', label = 'Intergranule') 

ax.axhline(y=granule_idx, color='magenta', linewidth=1, linestyle='dotted')
ax.axhline(y=intergranule_idx, color='black', linewidth=1, linestyle='dotted')

ax1.axvline(x=ti_center_wl-spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted')
ax1.axvline(x=ti_center_wl+spec_region_width/2., color=line_color1, linewidth=1, linestyle='dotted', label='Ti I')
ax1.legend()


ax1.set_xlabel(r'Wavelength $[\AA]$')  
ax1.set_ylabel(r'Counts [a.u.]')


plt.savefig('/home/franziskaz/figures/themis/test_voigt_themis-blend.png'
          , dpi=my_dpi, metadata={'Software': __file__})



#%%
# try to understand how sensitive the fitting is to the noise
#noise level from continuum:
spectra['ti']['30'].continuum.data.std(axis=0).mean()

# Use the best_fit from line_piecewise as reference (includes full pipeline: weights + parabola)
reference = intergranule.best_fit 
reference_ti = intergranule.component_ti

# add noise 100 different realizations per noise level (Gaussian) and fit each realization.
# compare  component_ti from reference and calculate absolute difference of profiles
# plot abs difference of profiles versus noise
noise_levels = [0.001, 0.01,  0.1]
n_realizations = 30

# Store results
mean_diffs = []
std_diffs = []

# Get reference wavelength and continuum
wvl_ref = intergranule.wvl
continuum_ref = continuum[intergranule_idx]

for noise_std in noise_levels:
    diffs = []
    for i in range(n_realizations):
        # Add Gaussian noise to reference
        noisy_profile = reference + np.random.normal(0, noise_std, size=reference.shape)

        # Fit the noisy profile using full line_piecewise pipeline
        try:
            noisy_fit = line_piecewise(wvl_ref, noisy_profile, ti_center_wl, spec_region_width, continuum=continuum_ref)

            # Calculate absolute difference of Ti component
            diff = np.abs(noisy_fit.component_ti - reference_ti)
            diffs.append(np.mean(diff))
        except Exception as e:
            print(f"  Realization {i} failed for noise {noise_std}: {e}")
            continue

    if diffs:
        mean_diffs.append(np.mean(diffs))
        std_diffs.append(np.std(diffs))
        print(f"Noise {noise_std}: mean diff = {mean_diffs[-1]:.6f}, std = {std_diffs[-1]:.6f}")
    else:
        mean_diffs.append(np.nan)
        std_diffs.append(np.nan)
#%%
# Plot results
f = plt.figure(num=4)
f.set_size_inches(fig_size, forward=True)
plt.clf()

ax = f.add_subplot(1, 1, 1)
ax.errorbar(noise_levels, mean_diffs, yerr=std_diffs, fmt='o-', capsize=5, label='Mean absolute difference')
ax.set_xlabel('Noise standard deviation')
ax.set_ylabel('Mean absolute difference between fitted and reference')
ax.set_title('Fitting sensitivity to Gaussian noise')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/franziskaz/figures/themis/test_voigt_themis-noise-sensitivity.png', dpi=my_dpi, metadata={'Software': __file__})

#%%
plt.plot(noisy_fit.best_fit)
plt.plot(noisy_profile)
plt.plot(reference)

#%%
plt.plot(noisy_fit.component_ti)
plt.plot(intergranule.component_ti)


