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
    config = get_config(config_path='configs/spot_dataset_sr_2025-07-07.toml', # omit to use defaults
    auto_discover_files=True,           # pre-fill matching files per level
    auto_create_dirs=False              # create directories if missing
    )
    
    lv0 = tdr.reduction_levels["l0"]
    lv1 = tdr.reduction_levels["l1"]
    lv2 = tdr.reduction_levels["l2"]
    lv3 = tdr.reduction_levels["l3"]
    lv4 = tdr.reduction_levels["l4"]
    
    result = lv0.reduce(config, data_type='scan', return_reduced = False)
    result = lv1.reduce(config, data_type='scan', return_reduced = False)
    result = lv2.reduce(config, data_type='scan', return_reduced = False)
    result = lv3.reduce(config, data_type='scan', return_reduced = False)
    result = lv4.reduce(config, data_type='scan', return_reduced = False)
    
    #print(lv1.get_description(data_type='flat'))
    #%%
  
#%%
    scan, _ = tio.read_any_file(config, 'scan', status='l4')
    result = tt.compute_polarimetry(scan)
 
   # Access Stokes V at slit 0, map 0 via difference method
    I = result.ratio.I.mean(axis=1)
    V = result.ratio.V.mean(axis=1)
    Q = result.ratio.Q.mean(axis=1)
    U = result.ratio.U.mean(axis=1)
   
   
    data_plot = np.array([I[:,30:-30,30:-30],Q[:,30:-30,30:-30], U[:,30:-30,30:-30], V[:,30:-30,30:-30]])
   
    viewer = display_data( data_plot, ['states','spatial_y',  'spatial_x', 'spectral'],
                      title='scan', 
                      state_names=['I', 'Q',
                                   'U', 'V'])


    
    
    