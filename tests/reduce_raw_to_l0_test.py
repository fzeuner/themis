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
    config = get_config(config_path='configs/spot_dataset_sr_2025-07-07.toml', # omit to use defaults
    auto_discover_files=True,           # pre-fill matching files per level
    auto_create_dirs=False              # create directories if missing
    )
    
    # config = get_config(config_path='configs/sample_dataset_sr_2025-07-08.toml', # omit to use defaults
    # auto_discover_files=True,           # pre-fill matching files per level
    # auto_create_dirs=False              # create directories if missing
    # )
    # 
    lv0 = tdr.reduction_levels["l0"]
    #print(lv0.get_description(data_type='flat'))
    
    #data, header = tio.read_any_file(config, 'dark', verbose=False, status='raw')
    # print(data[0]['lower'].data.shape)
    # print(data[0]['upper'].data.shape)
    
    # upper = data.stack_all('upper')
    # lower = data.stack_all('lower')
      
    # data_plot = np.array([upper.mean(axis=0),lower.mean(axis=0)])
    # viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
    #                  title='dark raw', 
    #                  state_names=['upper', 'lower'])
#%%     
    
   # result = lv0.reduce(config, data_type='dark', return_reduced = False)
   
    # data_plot = np.array([result.get(0).get_half('upper').data-result.get(1).get_half('upper').data,result.get(0).get_half('lower').data -result.get(1).get_half('lower').data ])
    # viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
    #                  title='dark l0', 
    #                  state_names=['upper', 'lower'])
    # viewer = display_data( np.array([result.get(1).get_half('upper').data,result.get(1).get_half('lower').data ]), 'states',  'spatial', 'spectral',
    #                  title='dark l0', 
    #                  state_names=['upper', 'lower'])
    
    # viewer = display_data( np.array([upper[0]-result.get(0).get_half('upper').data,lower[0]-result.get(0).get_half('lower').data ]), 'states',  'spatial', 'spectral',
    #                  title='dark raw single - l0', 
    #                  state_names=['upper', 'lower'])
    
    # raw_frame, header = tio.read_any_file(config, 'dark', verbose=False, status='raw')
    # ax=plot_frame_statistics(raw_frame, stat='std')
   
#%%      
    
    #frame_l0, header_l0 = tio.read_any_file(config, 'dark', verbose=False, status='l0')
    
    # plt.hist((data[0]['lower'].data-frame_l0[0]['lower'].data).flatten(), bins=100, color='blue', label='lower')
    # plt.hist((data[0]['upper'].data-frame_l0[0]['upper'].data).flatten(), bins=100, color='red', alpha=0.6, label='upper')
    # plt.hist(((data[0]['lower'].data-frame_l0[0]['upper'].data+data[0]['upper'].data-frame_l0[0]['lower'].data)*0.5).flatten(), bins=100, color='green', alpha=0.6, label='combined')
    
#%% 

    raw_frame, header = tio.read_any_file(config, 'flat', verbose=False, status='raw')
    ax=plot_frame_statistics(raw_frame, stat='mean')
    ax=plot_frame_statistics(raw_frame, stat='max')
    ax=plot_frame_statistics(raw_frame, ax=ax,stat='min')
    #result = lv0.reduce(config, data_type='flat', return_reduced = True)
    #data_plot = np.array([result.get(0).get_half('upper').data,result.get(0).get_half('lower').data ])
    # viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
    #                   title='flat l0', 
    #                   state_names=['upper', 'lower'])
    # viewer = display_data( np.array([raw_frame[25]['upper'].data,raw_frame[25]['lower'].data ]), 'states',  'spatial', 'spectral',
    #                   title='flat raw', 
    #                   state_names=['upper', 'lower'])
    
#%%

    # raw_frame, header = tio.read_any_file(config, 'scan', verbose=False, status='raw')
    # data_plot = np.array([
    # raw_frame.get_data('mQ', 0, 0, 'mQ'),  # frame_state='mQ', pol_state='mQ'
    # raw_frame.get_data('mQ', 0, 0, 'pQ'),   # frame_state='mQ', pol_state='pQ'
    # raw_frame.get_data('pQ', 0, 0, 'mQ'),  # frame_state='pQ', pol_state='mQ'
    # raw_frame.get_data('pQ', 0, 0, 'pQ')   # frame_state='pQ', pol_state='pQ'
    # ])
    # viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
    #                    title='scan raw ', 
    #                    state_names=['mQ,mQ', 'mQ,pQ','pQ,mQ', 'pQ,pQ'])
    
    #result = lv0.reduce(config, data_type='scan', return_reduced = False)
    result, header = tio.read_any_file(config, 'scan', verbose=False, status='l0')
    print(header['HISTORY'])
    data_plot = np.array([
    result.get_data('mQ', 0, 0, 'mQ'),  # frame_state='mQ', pol_state='mQ'
    result.get_data('mQ', 0, 0, 'pQ'),   # frame_state='mQ', pol_state='pQ'
    result.get_data('pQ', 0, 0, 'mQ'),  # frame_state='pQ', pol_state='mQ'
    result.get_data('pQ', 0, 0, 'pQ')   # frame_state='pQ', pol_state='pQ'
    ])
    viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
                       title='scan l0', 
                       state_names=['mQ,mQ', 'mQ,pQ','pQ,mQ', 'pQ,pQ'])
#%%    
