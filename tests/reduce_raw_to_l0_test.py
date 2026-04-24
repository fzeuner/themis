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

def _build_scan_plot_data(cycle_set, slit_idx=0, map_idx=0):
    keys = sorted(cycle_set.keys())
    frame_states = sorted({k[0] for k in keys if k[1] == slit_idx and k[2] == map_idx})

    arrays = []
    labels = []
    for frame_state in frame_states:
        frame = cycle_set.get_state_slit_map(frame_state, slit_idx, map_idx)
        if frame is None:
            continue
        for half_name in ['upper', 'lower']:
            half = frame.get_half(half_name)
            if half is None:
                continue
            arrays.append(half.data)
            pol = half.pol_state if half.pol_state is not None else half_name
            labels.append(f'{frame_state},{pol}')

    if not arrays:
        raise ValueError(f'No scan data found for slit={slit_idx}, map={map_idx}')

    return np.stack(arrays, axis=0), labels


#%%
#----------------------------------------------------------------------
"""
MAIN: 
   
"""

if __name__ == '__main__':

# -----------------------------------------------------
    # initialize configuration from datasets
    config = get_config(config_path='configs/formation_dataset_ti_2025-07-05.toml', # omit to use defaults
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
    
    # data_plot = np.array([upper.mean(axis=0), lower.mean(axis=0)])
    
    data, header = tio.read_any_file(config, 'flat_center', verbose=False, status='l0')
    data_plot=np.array([data[0]['lower'].data,data[0]['upper'].data])
    viewer = display_data(data_plot, ['states', 'spatial_x', 'spectral'],
                         title='flat center l0',
                         state_names=['upper', 'lower'])
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
    
    raw_frame, header = tio.read_any_file(config, 'dark', verbose=False, status='raw')
    ax = plot_frame_statistics(raw_frame, stat='std')
   
#%%      
    
    #frame_l0, header_l0 = tio.read_any_file(config, 'dark', verbose=False, status='l0')
    
    # plt.hist((data[0]['lower'].data-frame_l0[0]['lower'].data).flatten(), bins=100, color='blue', label='lower')
    # plt.hist((data[0]['upper'].data-frame_l0[0]['upper'].data).flatten(), bins=100, color='red', alpha=0.6, label='upper')
    # plt.hist(((data[0]['lower'].data-frame_l0[0]['upper'].data+data[0]['upper'].data-frame_l0[0]['lower'].data)*0.5).flatten(), bins=100, color='green', alpha=0.6, label='combined')
    
#%% 

    raw_frame, header = tio.read_any_file(config, 'flat', verbose=False, status='raw')
    ax=plot_frame_statistics(raw_frame, stat='mean')
    # ax=plot_frame_statistics(raw_frame, stat='max')
    # ax=plot_frame_statistics(raw_frame, ax=ax,stat='min')
    #result = lv0.reduce(config, data_type='flat', return_reduced = True)
    #data_plot = np.array([result.get(0).get_half('upper').data,result.get(0).get_half('lower').data ])
    # viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
    #                   title='flat l0', 
    #                   state_names=['upper', 'lower'])
    # viewer = display_data( np.array([raw_frame[25]['upper'].data,raw_frame[25]['lower'].data ]), 'states',  'spatial', 'spectral',
    #                   title='flat raw', 
    #                   state_names=['upper', 'lower'])
    
#%%

    #result = lv0.reduce(config, data_type='scan', return_reduced = False)
    result, header = tio.read_any_file(config, 'scan', verbose=False, status='l0')
    print(header['HISTORY'])
    data_plot, state_names = _build_scan_plot_data(result, slit_idx=0, map_idx=0)
    viewer = display_data(data_plot, 'states', 'spatial', 'spectral',
                       title='flat l0', 
                       state_names=state_names)
#%%    
    raw_frame, header = tio.read_any_file(config, 'scan', verbose=False, status='raw')
    data_plot, state_names = _build_scan_plot_data(raw_frame, slit_idx=0, map_idx=0)
    viewer = display_data(data_plot, ['states', 'spatial_x', 'spectral'],
                          title='scan raw',
                          state_names=state_names)