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
from lmfit import Model #conda install -c conda-forge lmfit

from style_themis import style_aa
import matplotlib.pyplot as plt
from matplotlib.ticker import (NullFormatter)
from matplotlib import gridspec
import numpy as np

from tqdm import tqdm
import gc
import sys
sys.path.append('/home/zeuner/code/irsol/sr_hanle_interp/python/sir_stuff/sir_gui')
import sirtools2 as st



#%%
#----------------------------------------------------------------------
"""
MAIN: 
    Test the B_los of different SIR models
   
"""
#---------------------------------------------------------------

# -----------------------------------------------------
# initialize files and folder object
directory = '/home/zeuner/data/themis/inversion/test/'
models =['blos', 'bincl1', 'bincl2', 'blos_inv', 'bincl1_inv', 'bincl2_inv']

zoom=2
plt.close()
my_dpi, fig_size_single, fig_size, params=style_aa(zoom)
plt.style.use('default')
plt.rcParams.update(params)
nullfmt=NullFormatter()
        
fig = plt.figure(num=2)
fig.set_size_inches([fig_size_single[0],fig_size_single[1]],forward=True)
fig.clf()
        
gs = gridspec.GridSpec(1,1, figure=fig)#width_ratios=[1,0.25,1,0.25])# , height_ratios=[1,1,1,1] )
gs.update(top=0.95, bottom=0.18,left=0.15, right=0.95, wspace=0.17, hspace=0.12)
        
ax = fig.add_subplot(gs[0, 0])
        

# read original data

for m in models:

 read_file_name=directory+m+'.mod'
            
 tau, temp, Pe, vmic, B, vlos, gamma, phi, vmac, ff, stray, z, rho, Pg  = st.readmod(read_file_name)

 ax.plot(tau, B*np.cos(gamma/180.*np.pi), label = m, linewidth=zoom*1, marker='.')

 ax.legend()
 ax.set_ylabel('B_los [G]', labelpad=4*zoom)
 ax.set_xlabel(r'log($\tau$)', labelpad=4*zoom)
            
fig.show()
 
