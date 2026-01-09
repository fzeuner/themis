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
    
    lv2 = tdr.reduction_levels["l2"]
    
    result = lv2.reduce(config, data_type='flat_center', return_reduced = False)
    result = lv2.reduce(config, data_type='flat', return_reduced = False)
    result = lv2.reduce(config, data_type='scan', return_reduced = False)
#%%    
    data_old, header = tio.read_any_file(config, 'flat_center', verbose=False, status='l1')
    data_plot_old = np.array([data_old.get(0).get_half('upper').data,data_old.get(0).get_half('lower').data ])
    
#%%
    scan, header = tio.read_any_file(config, 'scan', verbose=False, status='l2')
    scan_old, header = tio.read_any_file(config, 'scan', verbose=False, status='l1')
    
    data_plot = np.array([np.array((scan.get_state('pQ').stack_all('upper')[0]).data)-
                         np.array((scan.get_state('pQ').stack_all('lower')[0]).data),
                         np.array((scan_old.get_state('pQ').stack_all('upper')[0]).data)-
                         np.array((scan_old.get_state('pQ').stack_all('lower')[0]).data)])
    
    viewer = display_data( data_plot, ['states',  'spatial_x', 'spectral'],
                       title='scan', 
                       state_names=['y shifted',
                                    'l1'])
    
#%%
# test y overlap in scan

    data, header = tio.read_any_file(config, 'scan', verbose=False, status='l2')
    data_old, header = tio.read_any_file(config, 'scan', verbose=False, status='l1')
    
    data_plot = np.array([data.get(0).get_half('upper').data,data.get(0).get_half('lower').data,
                          data_old.get(0).get_half('upper').data,data_old.get(0).get_half('lower').data])
  
    l1, header = tio.read_any_file(config, 'flat', verbose=False, status='l1')
    
    data_plot = np.array([data.get(0).get_half('upper').data/dust.get(0).get_half('upper').data,
                          data.get(0).get_half('lower').data/dust.get(0).get_half('lower').data,
                          l1.get(0).get_half('upper').data, l1.get(0).get_half('lower').data])
    
    viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
                       title='flat center l1', 
                       state_names=['upper different dust corrected', 'lower different dust corrected',
                                    'self dust corrected upper', 'self dust corrected lower'])
    
    
    
    
    
    
    
    
    
    
    
    
    
    