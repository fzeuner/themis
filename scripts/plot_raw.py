#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 4 15:24:27 2025

@author: zeuner
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from themis.core import themis_tools as tt
from themis.display import themis_display_scan as display
from themis.datasets import themis_datasets_2025 as dst
from themis.datasets.themis_datasets_2025 import get_config

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
    config = get_config()

    # read data
    data, header = tt.read_any_file(config, 'scan', verbose=False, status='raw')
    
    plt.imshow(data.get_state_slit_map('mQ',0,0).get_half('lower').data)
    #result = ird.similarity(data.ui[0]/data.ui[0].max(), data.li[0]/data.li[0].max(), numiter=3)
    #ird.imshow(data.ui[0]/data.ui[0].max(), data.li[0]/data.li[0].max(), resulti)
    #plt.imshow(data.li[0]/data.li[0].max()-data.ui[0]/data.ui[0].max(), origin='lower')
    # data.wvl = np.arange(data.ui.shape[4])
    # display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line]) # because of a bug which I do not understand - close this one
    # display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line])