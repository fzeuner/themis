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
#---------------------------------------------------------------

# -----------------------------------------------------
# initialize files and folder object

ff = tt.ff() 
ff.v_file =  dst.directory + dst.data_files[dst.line]+dst.data_files['v_diff']

#wvl = tt.z3ccspectrum(spectrum,  wavelengthscale, ref_spectrum, FACL=0.8, FACH=1.5, FACS=0.01, CUT=None, DERIV=None, CONT=None, SHOW=2)

#%%
# read original data
# data, header = tt.read_images_file(ff.red_data_file)

# data.wvl = np.arange(data.ui.shape[4])
#%%
# display original data
# display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line])

###
#%%
# calculate beam exchange
#data_v = tt.beam_exchange(data)

#%%
# read v data
# 
# del data_v
gc.collect()
data_v, header = tt.read_v_file(ff.v_file, ff.i_file)
data_v.wvl = np.arange(data_v.v.shape[3])
#%%
# import themis_tools as tt
# data_v = tt.align_scans(data_v)
#%%

data_avg = tt.binning(data_v, binning= [1,3,1,1])
display_v.display_scan_data(data_avg, data_v.wvl)
#%%


display_v.display_scan_data(data_v, data_v.wvl)
#%%
# del data
gc.collect()
f, _ = tt.plot_power_spectrum(data_v.i[:,0,:226,dst.continuum[0]:dst.continuum[1]].mean(axis=2), zoom=1)
#%%
import themis_tools as tt
gc.collect()

data_v_dummy = data_v.v[:,:,:,dst.continuum[0]:dst.continuum[1]].mean(axis=3)
m = tt.check_continuum_noise(data_v_dummy)