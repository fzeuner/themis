#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 4 15:24:27 2025

@author: zeuner
"""

from themis.core import themis_tools as tt
from themis.core import themis_data_reduction as tdr
from themis.datasets import themis_datasets_2025 as dst
from themis.datasets.themis_datasets_2025 import get_config
from spectator.controllers.app_controller import display_data # from spectator

import matplotlib.pyplot as plt
import numpy as np
import imreg_dft as ird
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
    # 
    lv0 = tdr.reduction_levels["l0"]
    
    data, header = tt.read_any_file(config, 'dark', verbose=False, status='raw')
    
    upper = data.stack_all('upper')
    lower = data.stack_all('lower')
    
  
    
    data_plot = np.array([upper.mean(axis=0),lower.mean(axis=0)])
    # viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
    #                  title='dark raw', 
    #                  state_names=['upper', 'lower'])
    
    
    result = lv0.reduce(config, data_type='dark', return_reduced = True)
    
    
    data_plot = np.array([result.get(0).get_half('upper').data-result.get(1).get_half('upper').data,result.get(0).get_half('lower').data -result.get(1).get_half('lower').data ])
    viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
                     title='dark l0', 
                     state_names=['upper', 'lower'])
    viewer = display_data( np.array([result.get(1).get_half('upper').data,result.get(1).get_half('lower').data ]), 'states',  'spatial', 'spectral',
                     title='dark l0', 
                     state_names=['upper', 'lower'])
    
    viewer = display_data( np.array([upper[0]-result.get(0).get_half('upper').data,lower[0]-result.get(0).get_half('lower').data ]), 'states',  'spatial', 'spectral',
                     title='dark raw single - l0', 
                     state_names=['upper', 'lower'])
    
    # print(lvl.file_ext)  # .lf.fits
    # print(lvl.get_description("scan"))  # Gaussian fitting applied