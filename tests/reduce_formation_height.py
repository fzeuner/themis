#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 4 15:24:27 2025

@author: zeuner

Testing the reduction of raw to l0

"""

from themis.core import themis_tools as tt
from themis.core import themis_data_reduction as tdr
from themis.core import themis_io as tio
from themis.datasets import themis_datasets_2025 as dst
from themis.datasets.themis_datasets_2025 import get_config
from themis.plots import plot_power_spectrum

from spectator.controllers.app_controller import display_data # from spectator

import matplotlib.pyplot as plt
import numpy as np
import gc

#%%
#----------------------------------------------------------------------
"""
MAIN: 
   
"""

if __name__ == '__main__':

# -----------------------------------------------------
    # initialize configuration from datasets
    config = get_config(config_path='configs/formation_dataset_sr_2025-07-05.toml', # omit to use defaults
    auto_discover_files=True,           # pre-fill matching files per level
    auto_create_dirs=False              # create directories if missing
    )
    
    lv0 = tdr.reduction_levels["l0"]
    lv1 = tdr.reduction_levels["l1"]
    lv2 = tdr.reduction_levels["l2"]
    lv3 = tdr.reduction_levels["l3"]
    lv4 = tdr.reduction_levels["l4"]
    
    #result = lv0.reduce(config, data_type='dark', return_reduced = False)
    #result = lv1.reduce(config, data_type='dark', return_reduced = False)
    
    #result = lv0.reduce(config, data_type='flat_center', return_reduced = False)
    #result = lv1.reduce(config, data_type='flat_center', return_reduced = False)
    #result = lv2.reduce(config, data_type='flat_center', return_reduced = False)
    #result = lv3.reduce(config, data_type='flat_center', return_reduced = False)
    #result = lv4.reduce(config, data_type='flat_center', return_reduced = False)
    
#    result = lv0.reduce(config, data_type='scan', return_reduced = False)
#    result = lv1.reduce(config, data_type='scan', return_reduced = False)
 #   result = lv2.reduce(config, data_type='scan', return_reduced = False)
#    result = lv3.reduce(config, data_type='scan', return_reduced = False)
 #   result = lv4.reduce(config, data_type='scan', return_reduced = False)
  
    
    #print(lv1.get_description(data_type='flat'))
    #%%
  
#%%
scan, _ = tio.read_any_file(config, 'scan', status='l4')
state='Q'
I = 0.5*(np.array((scan.get_state('p'+state).stack_all('upper')))+
                np.array((scan.get_state('p'+state).stack_all('lower'))))

Q = 0.5*(np.array((scan.get_state('p'+state).stack_all('upper')))-
                np.array((scan.get_state('p'+state).stack_all('lower'))))
 
#%%
   
   
data_plot = np.array([I[:,30:-30,30:-30],Q[:,30:-30,30:-30]])
   
viewer = display_data( data_plot, ['states','spatial_y',  'spatial_x', 'spectral'],
                      title='scan' 
                      )


#%%
cont_image = I[:,30:-30,800]

from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as nd_shift

# correlate each row against the previous one and accumulate shifts
MAX_SHIFT = 10.0
cumulative_shift = 0.0
shifts = [0.0]
for i in range(1, cont_image.shape[0]):
    s, _, _ = phase_cross_correlation(cont_image[i-1], cont_image[i], upsample_factor=10)
    delta = s[0]
    if abs(delta) > MAX_SHIFT:
        # try second previous slice
        if i >= 2:
            s2, _, _ = phase_cross_correlation(cont_image[i-2], cont_image[i], upsample_factor=10)
            delta2 = s2[0] - shifts[-1] + shifts[-2]  # relative to previous
            if abs(delta2) <= MAX_SHIFT:
                delta = delta2
                print(f"  step {i}: used 2nd previous (shift={delta:.2f})")
            else:
                # try mean of last two slices
                ref_mean = 0.5 * (cont_image[i-1] + cont_image[i-2])
                sm, _, _ = phase_cross_correlation(ref_mean, cont_image[i], upsample_factor=10)
                deltam = sm[0]
                if abs(deltam) <= MAX_SHIFT:
                    delta = deltam
                    print(f"  step {i}: used mean of last 2 (shift={delta:.2f})")
                else:
                    delta = 0.0
                    print(f"  step {i}: all fallbacks exceeded {MAX_SHIFT}px, shift set to 0")
        else:
            delta = 0.0
            print(f"  step {i}: shift {s[0]:.2f}px > {MAX_SHIFT}px, set to 0")
    cumulative_shift += delta
    shifts.append(cumulative_shift)

# apply shifts to each row
cont_image_aligned = np.zeros_like(cont_image)
for i, s in enumerate(shifts):
    cont_image_aligned[i] = nd_shift(cont_image[i], shift=-s, mode='nearest')

print(f"Shifts range: {min(shifts):.2f} to {max(shifts):.2f} pixels")

#%%
from themis.plots.plot_power_spectrum import plot_power_spectrum
    
fig, power = plot_power_spectrum(
    I[:,30:-30,780:810].mean(axis=2),
    pixel_scale_x=config.cam.pixel_scale,  # arcsec/pixel along slit (from cam_config)
    pixel_scale_y=config.slit_width,        # arcsec/step along scan (from config params)
    reference_freq_line=1/0.18, reference_power_line=1.2e-6
)    
    