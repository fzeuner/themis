#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 4 15:24:27 2025

@author: zeuner
"""

import themis_tools as tt
import themis_data_reduction as tdr
import themis_display_scan as display
import themis_datasets_2025 as dst

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
    ff = tt.init() 

    # 
    lv0 = tdr.reduction_levels["lv0"]
    result = lv0.reduce(ff)
    # print(lvl.file_ext)  # .lf.fits
    # print(lvl.get_description("scan"))  # Gaussian fitting applied