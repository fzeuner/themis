#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  28 10:41:33 2025

@author: zeuner
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from themis.core import themis_tools as tt
from themis.datasets import themis_datasets as dst

import matplotlib.pyplot as plt
import numpy as np

import gc
import sys
import argparse
#%%
#----------------------------------------------------------------------
"""
MAIN: 
   - convert data for inversion
"""
#---------------------------------------------------------------

if __name__ == '__main__':
# initialize files and folder object

    ff = tt.ff()

# read v data

    gc.collect()
    data, header = tt.read_v_file(ff.v_file, ff.i_file)
    

    parser = argparse.ArgumentParser("python create_inversion_file.py")
    parser.add_argument("scan", help="Which scan you want to process.", type=int)
    parser.add_argument("pca", help="How many pca components you want to use. 0 for no PCA.", type=int)
    args = parser.parse_args()


#%%

    reduced_data, xlam, ff_p =  tt.process_data_for_inversion(data,  scan=args.scan, ff_p =ff,  
                                             binning =  1, # scan binning
                                             pca = args.pca,
                                             continuum = True,  # correct continuum: intensity/I_c, V - V_c
                                             cut=True, 
                                             test = False,  save = True) #only one scan at a time

