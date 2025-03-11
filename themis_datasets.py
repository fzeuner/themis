#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:52:55 2024

@author: franziskaz

Configuration file
"""

directory = "/home/franziskaz/data/themis/"
directory_figures = "/home/franziskaz/figures/themis/"
line='fe' # 'fe'

pixel={'fe': 0.235} # pixel scale arcsec/pixel CCD camera, THEMIS website
slitwidth=0.5 #/ [arcsec] SlitWidth  from Marianne  

continuum=[96,101]
# ONLY UNCOMMENT ONE BEFORE RUNNING 

#%%

data_files={ 'fe': "t002_b0202_sp_20240723_081105", 
            'o': "_tc3_p045p124_shifted.fts",
            'v_diff':"_Vdiff.fts",
            'v_ratio':"_Vratio.fts",
            'i':"_I.fts",}


#%%