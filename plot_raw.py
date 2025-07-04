#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:24:27 2025

@author: zeuner
"""

import themis_tools as tt
import themis_display_scan as display
import themis_datasets_2025 as dst

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
    # initialize files and folder object
    ff = tt.init() 

    # read data
    data, header = tt.read_any_file(ff, verbose=False)
    # data.wvl = np.arange(data.ui.shape[4])
    # display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line]) # because of a bug which I do not understand - close this one
    # display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line])