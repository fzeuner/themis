#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 4 15:24:27 2025

@author: zeuner

Testing the reduction of raw to l0

"""

from themis.core import themis_tools as tt
from themis.core import themis_data_reduction as tdr
from themis.core import themis_io as tio
from themis.datasets import themis_datasets_2025 as dst
from themis.datasets.themis_datasets_2025 import get_config
from themis.plots.plot_frame_statistics import plot_frame_statistics, plot_state_comparison

from spectator.controllers.app_controller import display_data # from spectator

import matplotlib.pyplot as plt
import numpy as np
import gc

#%%
#----------------------------------------------------------------------
"""
MAIN: 
   
"""

if __name__ == '__main__':

# -----------------------------------------------------
    # initialize configuration from datasets
    config = get_config(config_path='configs/sample_dataset_sr_2025-07-07.toml')
    
    lv4 = tdr.reduction_levels["l4"]
    
   # result = lv4.reduce(config, data_type='flat_center', return_reduced = False)
  #  result = lv4.reduce(config, data_type='flat', return_reduced = False)
  #  result = lv4.reduce(config, data_type='scan', return_reduced = False)
#%%    
    data, _ = tio.read_any_file(config, 'flat_center', verbose=False, status='l4')
    data_old, _ = tio.read_any_file(config, 'flat_center', verbose=False, status='l3')
  #%%    
    data_plot = np.array([(data_old.get(0).get_half('lower').data-
     data_old.get(0).get_half('upper').data ),
                          (data.get(0).get_half('lower').data-
                           data.get(0).get_half('upper').data )])
    
    viewer = display_data( data_plot[:,20:-20,20:-20], ['states',  'spatial_x', 'spectral'],
                       title='flat center', 
                       state_names=['l3 lower-upper',
                                    'l4 lower-upper'])
#%%
    data_plot = np.array([(data_old.get(0).get_half('upper').data-
    data.get(0).get_half('upper').data ),
                        (data.get(0).get_half('upper').data )])
  
    viewer = display_data( data_plot[:,20:-20,20:-20], ['states',  'spatial_x', 'spectral'],
                     title='flat center', 
                     state_names=['l3 lower-upper',
                                  'l4 lower-upper'])
    
#%%
 
     #%%

        scan, header = tio.read_any_file(config, 'scan', verbose=False, status='l4')
        state='U'
        I = 0.25*(np.array((scan.get_state('p'+state).stack_all('upper'))[0][30:-30,30:-30])+
                np.array((scan.get_state('m'+state).stack_all('upper'))[0][30:-30,30:-30])+
                np.array((scan.get_state('m'+state).stack_all('lower'))[0][30:-30,30:-30])+
                np.array((scan.get_state('p'+state).stack_all('lower')[0][30:-30,30:-30])))
        Q = 1/I*100*(np.array((scan.get_state('p'+state).stack_all('upper')[0][30:-30,30:-30]))/np.array((scan.get_state('m'+state).stack_all('upper')[0][30:-30,30:-30]))*
               np.array((scan.get_state('m'+state).stack_all('lower')[0][30:-30,30:-30]))/np.array((scan.get_state('p'+state).stack_all('lower')[0][30:-30,30:-30]))-1)
        
        # Q = 100*(np.array((scan.get_state('p'+state).stack_all('upper')[0]))-np.array((scan.get_state('m'+state).stack_all('upper')[0]))+
        #         -np.array((scan.get_state('m'+state).stack_all('lower')[0]))+np.array((scan.get_state('p'+state).stack_all('lower')[0])))
        data_plot = np.array([I,
                            Q
                             ])
        
        viewer = display_data( data_plot, ['states',  'spatial_x', 'spectral'],
                           title='scan', 
                           state_names=['I',
                                        state])    
    
#%%
    scan, header = tio.read_any_file(config, 'scan', verbose=False, status='l4')
    scan_old, header = tio.read_any_file(config, 'scan', verbose=False, status='l1')
    
    data_plot = np.array([np.array((-scan.get_state('mU').stack_all('upper')[0][30:-30,30:-30]).data)+
                         np.array((scan.get_state('mU').stack_all('lower')[0][30:-30,30:-30]).data)-
                         np.array((scan.get_state('pU').stack_all('lower')[0][30:-30,30:-30]).data)+
                         np.array((scan.get_state('pU').stack_all('upper')[0][30:-30,30:-30]).data),
                         np.array((scan_old.get_state('mU').stack_all('upper')[0][30:-30,30:-30]).data)-
                         np.array((scan_old.get_state('mU').stack_all('lower')[0][30:-30,30:-30]).data)])
    
    viewer = display_data( data_plot, ['states',  'spatial_x', 'spectral'],
                       title='scan', 
                       state_names=['l4',
                                    'l1']) 
    
    
    
    
    
    
    
    