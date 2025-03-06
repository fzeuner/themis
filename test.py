#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:24:27 2025

@author: zeuner
"""

import themis_tools as tt
import display_scan as display
import datasets as dst

import matplotlib.pyplot as plt
import numpy as np

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
  
  # -----------------------------------------------------
  # Read dataset
data, header = tt.read_images_file(ff.directory+ff.red_data_file)

xlam = np.arange(data.ui.shape[4])
display.display_scan_data(data, xlam,  title = dst.data_files[dst.line])
#%%
  
