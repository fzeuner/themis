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
    #%%
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
    from themis.core.line_fit import LineFit
    
    # Load wavelength grid
    file_set = config.dataset[data_type]['files']
    delta_offsets_upper_file = file_set.auxiliary.get('delta_offsets_upper')
    with fits.open(delta_offsets_upper_file) as hdul:
        wl_upper = (np.array(hdul['WAVELENGTH'].data))[100:-100]  # in nm
    
    # Extract row 400 spectrum
    row = 400
    upper_spec = upper[row, 100:-100]/upper[row, 100:-100].mean()
    lower_spec = lower[row, 100:-100]/lower[row, 100:-100].mean()
    #%%
    # Wavelengths for line fitting
    wl_line_targets = [453.478, 453.5575, 453.593, 453.605, 453.648]  # nm
    #wl_line_targets = [453.515, 453.5575, 453.593, 453.605, 453.65]  # nm
    wl_window = 0.002  # nm window for line fitting
    
    upper_line_centers = []
    lower_line_centers = []
    shifts_pixels = []
    fit_results = []  # Store fit objects for plotting
    
    for wl_target in wl_line_targets:
        # Find pixels within wavelength window
        mask = np.abs(wl_upper - wl_target) < wl_window
        if np.sum(mask) > 3:  # Need enough pixels for parabolic fit
            wl_window_vals = wl_upper[mask]
            upper_window_vals = upper_spec[mask]
            lower_window_vals = lower_spec[mask]
            
            # Use LineFit from themis.core for upper spectrum
            try:
                lf_upper = LineFit(wl_window_vals, upper_window_vals, center_guess=wl_target, error_threshold=0.5)
                lf_upper.run()
                
                # Use LineFit for lower spectrum
                lf_lower = LineFit(wl_window_vals, lower_window_vals, center_guess=wl_target, error_threshold=0.5)
                lf_lower.run()
                
                # Create fine wavelength grid for sub-pixel precision
                wl_fine = np.linspace(wl_window_vals.min(), wl_window_vals.max(), 1000)
                
                # Evaluate fitted models on fine grid
                upper_fit_fine = lf_upper.eval(wl_fine)
                lower_fit_fine = lf_lower.eval(wl_fine)
                
                # Find minima on fine grid (absorption line cores)
                idx_upper_fine = np.argmin(upper_fit_fine)
                idx_lower_fine = np.argmin(lower_fit_fine)
                wl_peak_upper = wl_fine[idx_upper_fine]
                wl_peak_lower = wl_fine[idx_lower_fine]
                
                # Convert wavelength positions to sub-pixel indices in the full spectrum
                # Use linear interpolation to get fractional pixel positions
                pixel_peak_upper = np.interp(wl_peak_upper, wl_upper, np.arange(len(wl_upper)))
                pixel_peak_lower = np.interp(wl_peak_lower, wl_upper, np.arange(len(wl_upper)))
                
                upper_line_centers.append(wl_peak_upper)
                lower_line_centers.append(wl_peak_lower)
                
                # Shift in pixels: how many pixels to shift lower to match upper (sub-pixel precision)
                shift_pixels = pixel_peak_lower - pixel_peak_upper
                shifts_pixels.append(shift_pixels)
                
                # Store fit results for plotting
                fit_results.append({
                    'wl_target': wl_target,
                    'wl_window': wl_window_vals,
                    'upper_data': upper_window_vals,
                    'lower_data': lower_window_vals,
                    'upper_fit': lf_upper,
                    'lower_fit': lf_lower,
                    'wl_fine': wl_fine,
                    'upper_fit_fine': upper_fit_fine,
                    'lower_fit_fine': lower_fit_fine
                })
                
                print(f'Line at {wl_target:.3f} nm: upper={wl_peak_upper:.4f} nm (px {pixel_peak_upper:.2f}), lower={wl_peak_lower:.4f} nm (px {pixel_peak_lower:.2f}), shift={shift_pixels:.4f} px')
            except RuntimeError as e:
                print(f'⚠ LineFit failed at {wl_target:.2f} nm: {e}')
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
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        
        # Top panel: original spectra with line centers marked
        ax1 = axes[0]
        ax1.plot(wl_upper, upper_spec, label='Upper L4', alpha=0.8, color='blue')
        ax1.plot(wl_upper, lower_spec, label='Lower L4 (original)', alpha=0.8, color='orange', ls='--')
        
        # Mark line centers
        for i, wl_target in enumerate(wl_line_targets):
            if i < len(upper_line_centers):
                wl_upper_center = upper_line_centers[i]
                wl_lower_center = lower_line_centers[i]
                ax1.axvline(wl_upper_center, color='blue', alpha=0.3, linestyle=':')
                ax1.axvline(wl_lower_center, color='orange', alpha=0.3, linestyle=':')
                # Plot markers at the fitted wavelength positions with actual intensity at those wavelengths
                # Use interpolation to get intensity at exact wavelength
                f_upper = interp1d(wl_upper, upper_spec, kind='cubic', bounds_error=False, fill_value='extrapolate')
                f_lower = interp1d(wl_upper, lower_spec, kind='cubic', bounds_error=False, fill_value='extrapolate')
                ax1.plot(wl_upper_center, f_upper(wl_upper_center), 'bo', markersize=8)
                ax1.plot(wl_lower_center, f_lower(wl_lower_center), 'o', color='orange', markersize=8)
        
        ax1.set_ylabel('Intensity')
        ax1.set_title(f'Row {row}: Original Spectra with Line Centers Marked (using lmfit VoigtModel)')
        ax1.legend()
        
        # Middle panel: fitted Voigt profiles
        ax2 = axes[1]
        for fit_data in fit_results:
            wl_win = fit_data['wl_window']
            # Plot data points
            ax2.plot(wl_win, fit_data['upper_data'], 'o', color='blue', alpha=0.5, markersize=4)
            ax2.plot(wl_win, fit_data['lower_data'], 'o', color='orange', alpha=0.5, markersize=4)
            # Plot fitted profiles
            ax2.plot(wl_win, fit_data['upper_fit'].fitted_profile, '-', color='blue', linewidth=2, alpha=0.8)
            ax2.plot(wl_win, fit_data['lower_fit'].fitted_profile, '-', color='orange', linewidth=2, alpha=0.8)
            # Mark fitted centers
            ax2.axvline(fit_data['upper_fit'].max_location, color='blue', alpha=0.3, linestyle=':')
            ax2.axvline(fit_data['lower_fit'].max_location, color='orange', alpha=0.3, linestyle=':')
        
        ax2.set_ylabel('Intensity')
        ax2.set_title('Voigt Profile Fits (data points + fitted curves)')
        ax2.legend(['Upper data', 'Lower data', 'Upper fit', 'Lower fit'])
        
        # Bottom panel: shifted lower spectrum
        ax3 = axes[2]
        ax3.plot(wl_upper, upper_spec, label='Upper L4', alpha=0.8, color='blue')
        ax3.plot(wl_upper, lower_spec_shifted, label='Lower L4 (shifted)', alpha=0.8, color='green', ls='--')
        
        ax3.set_xlabel('Wavelength [nm]')
        ax3.set_ylabel('Intensity')
        ax3.set_title(f'Row {row}: Lower Spectrum Shifted by Parabolic Function')
        ax3.legend()
        
        plt.tight_layout()
        plot_path = config.directories.figures / 'wavelength_shift_refinement_test.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'✓ Saved plot: {plot_path.name}')
    else:
        print('⚠ Could not fit enough lines for shift correction')
    
    #%%
