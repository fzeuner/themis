#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 4 15:24:27 2025

@author: zeuner
"""

import themis_tools as tt
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

    # read data
    data, header = tt.read_any_file(ff, verbose=False)
    #result = ird.similarity(data.ui[0]/data.ui[0].max(), data.li[0]/data.li[0].max(), numiter=3)
    #ird.imshow(data.ui[0]/data.ui[0].max(), data.li[0]/data.li[0].max(), result['timg'])
    #plt.imshow(data.li[0]/data.li[0].max()-data.ui[0]/data.ui[0].max(), origin='lower')
    # data.wvl = np.arange(data.ui.shape[4])
    # display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line]) # because of a bug which I do not understand - close this one
    # display.display_scan_data(data, data.wvl, title = dst.data_files[dst.line])