#!/usr/bin/env python3
"""
Test script to verify amend-spectroflat functionality.
This script tests if the amend-spectroflat script runs correctly
with a sample configuration and line file.
"""
import subprocess
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

def plot_amend_spectroflat_results(config, data_type='flat_center'):
    """
    Plot comparison of original and amended offset maps and soft flats.
    
    Parameters
    ----------
    config : Config
        Configuration object containing dataset and directory information
    data_type : str
        Type of data being processed ('flat_center' or 'flat')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits
    from pathlib import Path
    
    output_dir = Path(config.directories.reduced)
    
    # File paths
    original_offset_map = config.dataset[data_type]['files'].auxiliary.get('offset_map_outdated')
    wl_calibrated_offsets = config.dataset[data_type]['files'].auxiliary.get('offset_map')
    original_soft_flat = config.dataset[data_type]['files'].auxiliary.get('illumination_pattern_outdated')
    amended_soft_flat = config.dataset[data_type]['files'].auxiliary.get('illumination_pattern')
    
    # Check if all files exist
    if not all([original_offset_map.exists(), wl_calibrated_offsets.exists(),
                original_soft_flat.exists(), amended_soft_flat.exists()]):
        print("Error: Not all files exist for plotting")
        return
    
    # Read the files
    with fits.open(original_offset_map) as hdul:
        orig_offset_data = hdul[0].data
        orig_offset_header = hdul[0].header
    
    with fits.open(wl_calibrated_offsets) as hdul:
        wl_offset_data = hdul[0].data
        wl_offset_header = hdul[0].header
    
    with fits.open(original_soft_flat) as hdul:
        orig_flat_data = hdul[0].data
        orig_flat_header = hdul[0].header
    
    with fits.open(amended_soft_flat) as hdul:
        amended_flat_data = hdul[0].data
        amended_flat_header = hdul[0].header
    
    # Extract wavelength calibration info from headers
    # For offset maps (should have WL info after amend)
    if 'HIERARCH MIN_WL_NM' in wl_offset_header:
        min_wl = wl_offset_header['HIERARCH MIN_WL_NM']
        max_wl = wl_offset_header['HIERARCH MAX_WL_NM']
        dispersion = wl_offset_header.get('HIERARCH DISPERSION', (max_wl - min_wl) / wl_offset_data.shape[-1])
    else:
        # Fallback to pixel indices
        min_wl = 0
        max_wl = orig_offset_data.shape[-1]
        dispersion = 1
    
    # Create wavelength axis
    n_wavelength = orig_offset_data.shape[-1]
    wavelength_axis = np.linspace(min_wl, max_wl, n_wavelength)
    
    # Create figure with 4 rows × 2 columns (state 0 = upper, state 1 = lower)
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('Amend Spectroflat Results Comparison (State 0=Upper, State 1=Lower)', 
                 fontsize=16, fontweight='bold')
    
    # Plot both states
    for col, state in enumerate([0, 1]):
        state_label = 'Upper' if state == 0 else 'Lower'
        
        # Row 0: Original offset map
        ax = axes[0, col]
        im0 = ax.imshow(orig_offset_data[state], aspect='auto', cmap='RdBu_r', 
                         extent=[min_wl, max_wl, orig_offset_data.shape[1], 0])
        ax.set_title(f'Original Offset Map ({state_label})')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Spatial [px]')
        plt.colorbar(im0, ax=ax, label='Offset [px]')
        
        # Row 1: WL-calibrated offset map
        ax = axes[1, col]
        im1 = ax.imshow(wl_offset_data[state], aspect='auto', cmap='RdBu_r',
                         extent=[min_wl, max_wl, wl_offset_data.shape[1], 0])
        ax.set_title(f'WL-Calibrated Offset Map ({state_label})')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Spatial [px]')
        plt.colorbar(im1, ax=ax, label='Offset [px]')
        
        # Row 2: Original soft flat (illumination pattern)
        ax = axes[2, col]
        im2 = ax.imshow(orig_flat_data[state], aspect='auto', cmap='viridis',
                         extent=[min_wl, max_wl, orig_flat_data.shape[1], 0])
        ax.set_title(f'Original Illumination Pattern ({state_label})')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Spatial [px]')
        plt.colorbar(im2, ax=ax, label='Relative intensity')
        
        # Row 3: Amended soft flat
        ax = axes[3, col]
        im3 = ax.imshow(amended_flat_data[state], aspect='auto', cmap='viridis',
                         extent=[min_wl, max_wl, amended_flat_data.shape[1], 0])
        ax.set_title(f'Amended Illumination Pattern ({state_label})')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Spatial [px]')
        plt.colorbar(im3, ax=ax, label='Relative intensity')
    
    plt.tight_layout()
    
    # Save figure in the figures directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figures_dir = Path(config.directories.figures)
    figures_dir.mkdir(exist_ok=True, parents=True)
    output_plot = figures_dir / f'amend_spectroflat_comparison_{timestamp}.png'
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_plot}")
    
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Wavelength range: {min_wl:.3f} - {max_wl:.3f} nm")
    print(f"Dispersion: {dispersion:.6f} nm/px")
    
    print("\nOffset map differences (RMS):")
    for state in [0, 1]:
        state_label = 'Upper' if state == 0 else 'Lower'
        rms_offset = np.sqrt(np.mean((wl_offset_data[state] - orig_offset_data[state])**2))
        print(f"  State {state} ({state_label}): {rms_offset:.4f} px")
    
    print("\nSoft flat differences (RMS):")
    for state in [0, 1]:
        state_label = 'Upper' if state == 0 else 'Lower'
        rms_flat = np.sqrt(np.mean((amended_flat_data[state] - orig_flat_data[state])**2))
        print(f"  State {state} ({state_label}): {rms_flat:.6f}")


if __name__ == "__main__":
        
        # Optionally plot results
        try:
            from themis.datasets.themis_datasets_2025 import get_config
            config = get_config(config_path='configs/sample_dataset_sr_2025-07-07.toml')
            
            user_input = input("\nWould you like to plot the results? (y/n): ").strip().lower()
            if user_input == 'y':
                plot_amend_spectroflat_results(config, data_type='flat_center')
        except Exception as e:
            print(f"Could not plot results: {e}")

  
