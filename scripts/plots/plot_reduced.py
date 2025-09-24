#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:24:27 2025

@author: zeuner
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from themis.core import themis_tools as tt
from themis.display import themis_display_scan as display
from themis.display import themis_display_v as display_v
from themis.datasets import themis_datasets as dst

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

    data, header = tt.read_images_file(ff.red_data_file)
    data.wvl = np.arange(data.ui.shape[4])
    display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line]) # because of a bug which I do not understand - close this one
    display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line])