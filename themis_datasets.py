#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:52:55 2024

@author: franziskaz

Configuration file
"""

directory = "/home/franziskaz/data/themis/"
directory_figures = "/home/franziskaz/figures/themis/"
directory_inversion = "inversion/"

line='fe' # 'fe'

pixel={'fe': 0.235} # pixel scale arcsec/pixel CCD camera, THEMIS website
slitwidth=0.5 #/ [arcsec] SlitWidth  from Marianne 

spectral_sampling = 0.01348  # in \AA

continuum=[96,101] # continuum region - original data pixel scale 

tellurics = [[200,215],[261,273]] # masking areas for tellurics - original data pixel scale 

line_core= 172  # pixel for line cores - original data pixel scale 

yroi = [0,1]# scan position
roi = [20,220] # spatial - 227 max
sroi = [130,300] # spectral - 512 max  
# ONLY UNCOMMENT ONE BEFORE RUNNING 

#%%

data_files={ 'fe': "t002_b0202_sp_20240723_081105",  # name of data set and camera 
            'r': "_tc3_p045p124_shifted.fts",        # reduced data extension
            'v_diff':"_Vdiff.fts",                   # V by difference method
            'v_ratio':"_Vratio.fts",                 # V by beam exchange method
            'v_i':"_VoverI_noflat.fts",              # V/I with beam exchange
            'i':"_I.fts",}                           # I by beam combination


#%%