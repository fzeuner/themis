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
    config = get_config(line='ti', config_path='configs/sample_dataset_2025-07-07.toml')
    
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
    
#%%
    scan_data, _ = tio.read_any_file(config, 'scan', status='l4')
    result = tt.compute_polarimetry(scan_data)
    
    I = result.ratio.I.mean(axis=(0,1))
    U_d = result.difference.U.mean(axis=(0,1))
    U_r = result.ratio.U.mean(axis=(0,1))
    data_plot=np.array([I[100:-10,10:-150], U_d[100:-10,10:-150], U_r[100:-10,10:-150]])
    viewer = display_data( data_plot, ['states',  'spatial_x', 'spectral'],
                       title='scan', 
                       state_names=['I',
                                    'U_diff', 'U_ratio']) 
    
    #%%
    plt.plot(U_d[660,30:-30], label=['difference'])
    plt.plot(U_r[660,30:-30], label=['ratio'])
    plt.ylabel('U/I')
    # plt.plot(U_d[460,30:-30], label=['difference'])
    # plt.plot(U_r[460,30:-30], label=['ratio'])
    plt.legend()
    
    #%%
    _ = tt.check_continuum_noise( result.ratio.U[:,0,20:600,811])    
    
    
    #%%
    # improve ti lower to upper level alignment
    config = get_config(line='ti', config_path='configs/sample_dataset_2025-07-07.toml')
    data_type='flat_center'
    
    # upper L4 frame (corrected)
    data4, _ = tio.read_any_file(config, data_type, verbose=False, status='l4')
    upper = data4.get(0).get_half('upper').data
    lower = data4.get(0).get_half('lower').data
    data_plot=np.array([upper[100:-100,100:-100], lower[100:-100,100:-100], upper[100:-100,100:-100]- lower[100:-100,100:-100]])
    viewer = display_data( data_plot, ['states',  'spatial_x', 'spectral'],
                       title='flat center', 
                       state_names=['upper',
                                    'lower','u-l']) 
    

    #%%
   
    #%%
    # Wavelength shift refinement using parabolic line fitting
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    
    # Load wavelength grid
    file_set = config.dataset[data_type]['files']
    delta_offsets_upper_file = file_set.auxiliary.get('delta_offsets_upper')
    with fits.open(delta_offsets_upper_file) as hdul:
        wl_upper = (np.array(hdul['WAVELENGTH'].data))[100:-100]  # in nm
    
    # Extract row 400 spectrum
    row = 400
    upper_spec = upper[row, 100:-100]
    lower_spec = lower[row, 100:-100]
    
    # Wavelengths for line fitting
    wl_line_targets = [453.51, 453.57, 453.59, 453.60, 453.65]  # nm
    wl_window = 0.02  # nm window for line fitting
    
    upper_line_centers = []
    lower_line_centers = []
    shifts_pixels = []
    
    for wl_target in wl_line_targets:
        # Find pixels within wavelength window
        mask = np.abs(wl_upper - wl_target) < wl_window
        if np.sum(mask) > 5:  # Need enough pixels for parabolic fit
            wl_window_vals = wl_upper[mask]
            upper_window_vals = upper_spec[mask]
            lower_window_vals = lower_spec[mask]
            
            # Fit parabola to upper: I = a*x^2 + b*x + c
            # Use pixel indices as x
            pixel_indices = np.where(mask)[0]
            try:
                coeffs_upper = np.polyfit(pixel_indices, upper_window_vals, 2)
                coeffs_lower = np.polyfit(pixel_indices, lower_window_vals, 2)
                
                # Find peak: derivative = 2*a*x + b = 0 => x = -b/(2*a)
                pixel_peak_upper = -coeffs_upper[1] / (2 * coeffs_upper[0])
                pixel_peak_lower = -coeffs_lower[1] / (2 * coeffs_lower[0])
                
                upper_line_centers.append(pixel_peak_upper)
                lower_line_centers.append(pixel_peak_lower)
                shift = pixel_peak_lower - pixel_peak_upper
                shifts_pixels.append(shift)
                
                print(f'Line at {wl_target:.2f} nm: upper={pixel_peak_upper:.2f} px, lower={pixel_peak_lower:.2f} px, shift={shift:.4f} px')
            except Exception as e:
                print(f'⚠ Could not fit parabola at {wl_target:.2f} nm: {e}')
        else:
            print(f'⚠ Not enough pixels at {wl_target:.2f} nm')
    
    # Fit parabolic function to shifts vs wavelength
    if len(shifts_pixels) >= 3:
        shift_wls = np.array([wl_line_targets[i] for i in range(len(shifts_pixels))])
        shift_vals = np.array(shifts_pixels)
        
        # Use parabolic (quadratic) interpolation
        f_shift = interp1d(shift_wls, shift_vals, kind='quadratic',
                          bounds_error=False, fill_value='extrapolate')
        shift_correction = f_shift(wl_upper)
        
        print(f'Parabolic interpolation shift fit')
        print(f'Shift correction range: {shift_correction.min():.4f} – {shift_correction.max():.4f} px')
        
        # Apply shift correction to lower spectrum
        # Shift by subpixels using interpolation
        pixel_indices_shifted = np.arange(len(wl_upper)) - shift_correction
        f_interp = interp1d(pixel_indices_shifted, lower_spec, kind='cubic',
                          bounds_error=False, fill_value='extrapolate')
        lower_spec_shifted = f_interp(np.arange(len(wl_upper)))
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Top panel: original spectra with line centers marked
        ax1 = axes[0]
        ax1.plot(wl_upper, upper_spec, label='Upper L4', alpha=0.8, color='blue')
        ax1.plot(wl_upper, lower_spec, label='Lower L4 (original)', alpha=0.8, color='orange', ls='--')
        
        # Mark line centers
        for i, wl_target in enumerate(wl_line_targets):
            if i < len(upper_line_centers):
                ax1.axvline(wl_upper[int(upper_line_centers[i])], color='blue', alpha=0.3, linestyle=':')
                ax1.axvline(wl_upper[int(lower_line_centers[i])], color='orange', alpha=0.3, linestyle=':')
                ax1.plot(wl_target, upper_spec[int(upper_line_centers[i])], 'bo', markersize=8)
                ax1.plot(wl_target, lower_spec[int(lower_line_centers[i])], 'o', color='orange', markersize=8)
        
        ax1.set_ylabel('Intensity')
        ax1.set_title(f'Row {row}: Original Spectra with Line Centers Marked')
        ax1.legend()
        
        # Bottom panel: shifted lower spectrum
        ax2 = axes[1]
        ax2.plot(wl_upper, upper_spec, label='Upper L4', alpha=0.8, color='blue')
        ax2.plot(wl_upper, lower_spec_shifted, label='Lower L4 (shifted)', alpha=0.8, color='green', ls='--')
        
        ax2.set_xlabel('Wavelength [nm]')
        ax2.set_ylabel('Intensity')
        ax2.set_title(f'Row {row}: Lower Spectrum Shifted by Parabolic Function')
        ax2.legend()
        
        plt.tight_layout()
        plot_path = config.directories.reduced / 'wavelength_shift_refinement_test.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'✓ Saved plot: {plot_path.name}')
    else:
        print('⚠ Could not fit enough lines for shift correction')
    
    #%%
