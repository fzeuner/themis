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
ff.red_data_file = dst.directory + dst.data_files[dst.line]+dst.data_files['o']
ff.v_file =  dst.directory + dst.data_files[dst.line]+dst.data_files['v_ratio']
ff.i_file = dst.directory + dst.data_files[dst.line]+dst.data_files['i']
ff.figure_odir = dst.directory_figures

#wvl = tt.z3ccspectrum(spectrum,  wavelengthscale, ref_spectrum, FACL=0.8, FACH=1.5, FACS=0.01, CUT=None, DERIV=None, CONT=None, SHOW=2)

#%%
# read original data
data, header = tt.read_images_file(ff.red_data_file)

data.wvl = np.arange(data.ui.shape[4])
#%%
# display original data
display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line])

###
#%%
# calculate beam exchange
#data_v = tt.beam_exchange(data)

#%%
# read v data
del data_v
gc.collect()
data_v, header = tt.read_v_file(ff.v_file, ff.i_file)
data_v.wvl = np.arange(data_v.v.shape[3])
display_v.display_scan_data(data_v, data_v.wvl)
#%%
# del data
# gc.collect()
#%%

