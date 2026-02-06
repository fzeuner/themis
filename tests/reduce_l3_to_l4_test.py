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
    config = get_config(config_path='configs/sample_dataset_sr_2025-07-07.toml')
    
    lv4 = tdr.reduction_levels["l4"]
    
    result = lv4.reduce(config, data_type='flat_center', return_reduced = False)
    #result = lv3.reduce(config, data_type='flat', return_reduced = False)
    #result = lv3.reduce(config, data_type='scan', return_reduced = False)
#%%    
    data, _ = tio.read_any_file(config, 'flat_center', verbose=False, status='l4')
    data_old, _ = tio.read_any_file(config, 'flat_center', verbose=False, status='l3')
    
    data_plot = np.array([(data_old.get(0).get_half('lower').data-
     data_old.get(0).get_half('upper').data ),
                          (data.get(0).get_half('lower').data-
                           data.get(0).get_half('upper').data )])
    
    viewer = display_data( data_plot[:,20:-20,20:-20], ['states',  'spatial_x', 'spectral'],
                       title='flat center', 
                       state_names=['l3 lower-upper',
                                    'l4 lower-upper'])
#%%

    
#%%
 
    
    
    
    
    
    
    
    
    
    
    