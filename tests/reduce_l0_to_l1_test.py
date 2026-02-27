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
from themis.plots.plot_frame_statistics import plot_frame_statistics, plot_state_comparison

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
    config = get_config(config_path='configs/sample_dataset_sr_2025-07-07.toml', # omit to use defaults
    auto_discover_files=True,           # pre-fill matching files per level
    auto_create_dirs=False              # create directories if missing
    )
    
    lv1 = tdr.reduction_levels["l1"]
    #print(lv1.get_description(data_type='flat'))
    #%%
    #data, header = tio.read_any_file(config, 'flat_center', verbose=False, status='l0')
    
    result = lv1.reduce(config, data_type='flat_center', return_reduced = False)
    result = lv1.reduce(config, data_type='flat', return_reduced = False)
    
   # data_plot = np.array([result.get(0).get_half('upper').data,result.get(0).get_half('lower').data ])
    
#%%
    data, header = tio.read_any_file(config, 'flat_center', verbose=False, status='l1')
    dust, _ = tio.read_any_file(config, 'flat_center', verbose=False, status='dust')
    
    data_plot = np.array([data.get(0).get_half('upper').data,data.get(0).get_half('lower').data,
                          dust.get(0).get_half('upper').data,dust.get(0).get_half('lower').data])
    
    viewer = display_data( data_plot,  ['states',  'spatial_x', 'spectral'],
                       title='flat center l1', 
                       state_names=['upper dust corrected', 'lower dust corrected',
                                    'dust upper', 'dust lower'])
    
#%%
# test flat center on flat

    data, header = tio.read_any_file(config, 'flat', verbose=False, status='l0')
    dust, _ = tio.read_any_file(config, 'flat_center', verbose=False, status='dust')
    l1, header = tio.read_any_file(config, 'flat', verbose=False, status='l1')
    
    data_plot = np.array([data.get(0).get_half('upper').data/dust.get(0).get_half('upper').data,
                          data.get(0).get_half('lower').data/dust.get(0).get_half('lower').data,
                          l1.get(0).get_half('upper').data, l1.get(0).get_half('lower').data])
    
    viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
                       title='flat center l1', 
                       state_names=['upper different dust corrected', 'lower different dust corrected',
                                    'self dust corrected upper', 'self dust corrected lower'])
    
    
#%%
# test polarimetry    
    
    scan_data, _ = tio.read_any_file(config, 'scan', status='l4')
    result = tt.compute_polarimetry(scan_data)
 
    # Access Stokes V at slit 0, map 0 via difference method
    I_diff = result.difference.I.mean(axis=(0,1))
    V_diff = result.difference.V.mean(axis=(0,1))
    Q_diff = result.difference.Q.mean(axis=(0,1))
    U_diff = result.difference.U.mean(axis=(0,1))
    
    
    data_plot = np.array([I_diff[30:-30,30:-30],Q_diff[30:-30,30:-30], U_diff[30:-30,30:-30], V_diff[30:-30,30:-30]])
    
    viewer = display_data( data_plot, ['states',  'spatial_x', 'spectral'],
                       title='scan', 
                       state_names=['I', 'Q',
                                    'U', 'V'])
    
#%%
# test polarimetry    

 
    # Access Stokes V at slit 0, map 0 via difference method
    I_diff = result.ratio.I[1, 0]
    V_diff = result.ratio.V[1, 0]
    Q_diff = result.ratio.Q[1, 0]
    U_diff = result.ratio.U[1, 0]
    
    
    data_plot = np.array([I_diff[30:-30,30:-30],Q_diff[30:-30,30:-30], U_diff[30:-30,30:-30], V_diff[30:-30,30:-30]])
    
    viewer = display_data( data_plot, ['states',  'spatial_x', 'spectral'],
                       title='scan', 
                       state_names=['I', 'Q',
                                   'U', 'V'])
 #%%    
# test polarimetry    

 
    # Access Stokes V at slit 0, map 0 via difference method
    I_diff = result.ratio.I.mean(axis=1)
    V_diff = result.ratio.V.mean(axis=1)
    Q_diff = result.ratio.Q.mean(axis=1)
    U_diff = result.ratio.U.mean(axis=1)
    
    
    data_plot = np.array([I_diff[:,30:-30,30:-30],Q_diff[:,30:-30,30:-30], U_diff[:,30:-30,30:-30], V_diff[:,30:-30,30:-30]])
    
    viewer = display_data( data_plot, ['states','spatial_y',  'spatial_x', 'spectral'],
                       title='scan', 
                       state_names=['I', 'Q',
                                    'U', 'V'])
    
    
    
    
    