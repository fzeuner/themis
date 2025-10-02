#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 4 15:24:27 2025

@author: zeuner
"""

import os, sys
from themis.core import themis_tools as tt
from themis.display import themis_display_scan as display
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
    # initialize files and folder object
    config = get_config(config_path='configs/sample_dataset_sr_2025-07-07.toml', # omit to use defaults
    auto_discover_files=True,           # pre-fill matching files per level
    auto_create_dirs=False              # create directories if missing
    )
    # read data
    data, header = tt.read_any_file(config, 'scan', verbose=False, status='raw')
    
    data_plot = np.array([data.get_state_slit_map('mQ',0,0).get_half('upper').data,data.get_state_slit_map('mQ',0,0).get_half('lower').data])
    
    viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
                     title='mQ', 
                     state_names=['upper', 'lower'])


    #result = ird.similarity(data.ui[0]/data.ui[0].max(), data.li[0]/data.li[0].max(), numiter=3)
    #ird.imshow(data.ui[0]/data.ui[0].max(), data.li[0]/data.li[0].max(), resulti)
    #plt.imshow(data.li[0]/data.li[0].max()-data.ui[0]/data.ui[0].max(), origin='lower')
    # data.wvl = np.arange(data.ui.shape[4])
    # display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line]) # because of a bug which I do not understand - close this one
    # display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line])