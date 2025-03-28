#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  28 14:10:21 2025

@author: zeuner
"""

import themis_tools as tt
import themis_datasets as dst

import matplotlib.pyplot as plt
import numpy as np

import gc
import sys
import argparse
#%%
#----------------------------------------------------------------------
"""
MAIN: 
   - convert data for inversion - for a single pixel
"""
#---------------------------------------------------------------

if __name__ == '__main__':
# initialize files and folder object

    ff = tt.ff()

# read v data

    gc.collect()
    data, header = tt.read_v_file(ff.v_file, ff.i_file)
    
    parser = argparse.ArgumentParser("python create_inversion_file_single.py")
    parser.add_argument("scan", help="Which scan you want to process.", type=int)
    parser.add_argument("x", help="Which x pixel you want to process.", type=int)
    parser.add_argument("y", help="Which y pixel you want to process.", type=int)
    parser.add_argument("pca", help="How many pca components you want to use. 0 for no PCA.", type=int)
    parser.add_argument('dir', type=str, default='', help='Optional output file folder (relative to inversion folder)')
    parser.add_argument('ext', type=str, default='', help='Optional output file extension')
    
    args = parser.parse_args()


#%%

    reduced_data, xlam, ff_p = tt.process_pixel_for_inversion(data,  x=args.x, y=args.y, scan=args.scan, ff_p =ff,  
                                                 pca = args.pca,  # define number of pca components
                                                 continuum = True,  # correct continuum: intensity/I_c, V - V_c
                                                 cut=True, 
                                                 test = False, save = True, dir_ = args.dir, ext = args.ext) #only one scan at a time

