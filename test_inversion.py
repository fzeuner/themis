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
   - convert data for inversion
"""
#---------------------------------------------------------------

# -----------------------------------------------------
# initialize files and folder object

ff = tt.ff()


#%%
# read v data

gc.collect()
data, header = tt.read_v_file(ff.v_file, ff.i_file)
#%%

reduced_data, xlam, ff_p =  tt.process_data_for_inversion(data,  scan=0, ff_p =ff,  
                                             wl_calibration = False,
                                             binning =  1, # scan binning
                                             mask = False,
                                             pca = False,
                                             continuum = True,  # correct continuum: intensity/I_c, V - V_c
                                             cut=True, 
                                             test = False, debug = False, save = True) #only one scan at a time

#tt.write_wht_file(xlam, '/home/zeuner/data/themis/inversion/test_parallel/run_files/')
#%%
# test adding noise to profile for SIR inversion

_ = tt.add_noise_to_profiles(ff.inversion_dir+'test/bincl1.per', 0.05)