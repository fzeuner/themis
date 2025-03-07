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
    read file
   
"""
#---------------------------------------------------------------

# -----------------------------------------------------
# initialize files and folder object

ff = tt.ff()
ff.directory = dst.directory
ff.red_data_file = dst.data_files[dst.line]
ff.figure_odir = dst.directory_figures
data, header = tt.read_images_file(ff.directory+ff.red_data_file)
data_v = tt.calc_v_beamexchange(data)

data.wvl = np.arange(data.ui.shape[4])
#%%
  # -----------------------------------------------------
  # Read dataset



display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line])

###
#%%
data_v = tt.beam_exchange(data)

#%%
display_v.display_scan_data(data_v, data_v.wvl)
#%%
del data
gc.collect()
#%%
  
