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
    
    lv1 = tdr.reduction_levels["l1"]
    #print(lv1.get_description(data_type='flat'))
    #%%
    #data, header = tio.read_any_file(config, 'flat_center', verbose=False, status='l0')
    
    result = lv1.reduce(config, data_type='flat_center', return_reduced = False)
    
   # data_plot = np.array([result.get(0).get_half('upper').data,result.get(0).get_half('lower').data ])
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    