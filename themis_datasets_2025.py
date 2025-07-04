#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fr Jul 4 11:52:55 2025

@author: franziskaz

Configuration file
"""

# --- parameters for the loading

line = 'ti'
date = '2025-07-04'
sequence = 6
data_t = 'flat'
status = 'raw'

# --- parameters to be changed only once  (in principle)

directory = "/home/franziskaz/data/themis/"+date
directory_figures = "/home/franziskaz/figures/themis/"
directory_inversion = "/home/franziskaz/data/themis/inversion/"

slit_width=0.33 #/ [arcsec] SlitWidth

#%%

reduction_levels={'raw':'', 'l0': '_shifted'}