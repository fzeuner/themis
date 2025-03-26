#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:24:27 2025

@author: zeuner
"""

import themis_tools as tt
import themis_display_scan as display
import themis_display_v as display_v
import themis_datasets as dst

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
    ff = tt.ff() 

    # read data
    gc.collect()
    data_v, header = tt.read_v_file(ff.v_file, ff.i_file)
    data_v.wvl = np.arange(data_v.v.shape[3])

    display_v.display_scan_data(data_v, data_v.wvl)

    display_v.display_scan_data(data_v, data_v.wvl)