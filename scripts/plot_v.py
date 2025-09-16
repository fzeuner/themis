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
    # comment: change the file which is shown to v_diff defined in themis_datasets
    # ff.v_file= dst.directory + dst.data_files[dst.line]+dst.data_files['v_diff']
    # read data
    gc.collect()
    data_v, header = tt.read_v_file(ff.v_file, ff.i_file)
    data_v.wvl = np.arange(data_v.v.shape[3])

    display_v.display_scan_data(data_v, data_v.wvl) # because of a bug which I do not understand - close this one

    display_v.display_scan_data(data_v, data_v.wvl)