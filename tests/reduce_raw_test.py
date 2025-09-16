#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 4 15:24:27 2025

@author: zeuner
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from themis.core import themis_tools as tt
from themis.core import themis_data_reduction as tdr
from themis.display import themis_display_scan as display
from themis.datasets import themis_datasets_2025 as dst
from themis.datasets.themis_datasets_2025 import get_config

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
    # initialize configuration from datasets
    config = get_config()

    # 
    lv0 = tdr.reduction_levels["l0"]
    result = lv0.reduce(config)
    # print(lvl.file_ext)  # .lf.fits
    # print(lvl.get_description("scan"))  # Gaussian fitting applied