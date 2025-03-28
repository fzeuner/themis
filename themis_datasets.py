#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 3 11:52:55 2025

@author: franziskaz

Configuration file
"""

directory = "/home/franziskaz/data/themis/"
directory_figures = "/home/franziskaz/figures/themis/"
directory_inversion = "/home/franziskaz/data/themis/inversion/"

line='fe' # 'fe'

pixel={'fe': 0.235} # pixel scale arcsec/pixel CCD camera, THEMIS website
slitwidth=0.5 #/ [arcsec] SlitWidth  from Marianne 

spectral_sampling = 0.01348  # in \AA

# INVERSION parameters

continuum=[96,101] # continuum region - original data pixel scale 

tellurics = [[200,215],[261,273]] # masking areas for tellurics - original data x pixel scale 

line_core= [172,246]  # pixel for line cores - original data x pixel scale 

line_idx = 1 # left Fe line index, this is needed to create the correct input file, same as ine LINE and wavelength.grid

yroi = [0,40]  # scan position
roi = [20,220] # spatial - 227 max
sroi = [130,280] # spectral - 512 max - not too large!!


#%%

data_files={ 'fe': "t002_b0202_sp_20240723_081105",  # name of data set and camera 
            'r': "_tc3_p045p124_shifted.fts",        # reduced data extension
            'v_diff':"_Vdiff.fts",                   # V by difference method
            'v_ratio':"_Vratio.fts",                 # V by beam exchange method
            'v_i':"_VoverI_noflat.fts",              # V/I with beam exchange
            'i':"_I.fts",}                           # I by beam combination


#%%