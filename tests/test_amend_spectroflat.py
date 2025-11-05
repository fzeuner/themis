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

def test_amend_spectroflat():
    """Test if amend-spectroflat runs successfully with test data."""
    
    # Get the project root and amend-spectroflat script
    project_root = Path(__file__).resolve().parents[1]
    amend_script = project_root / 'atlas-fit' / 'bin' / 'amend_spectroflat'
    
    if not amend_script.exists():
        print(f"Error: Amend-spectroflat script not found at {amend_script}")
        return False
    
    print(f"Testing amend-spectroflat...")
    
    # Load configuration
    from themis.datasets.themis_datasets_2025 import get_config
    from themis.core import themis_io as tio
    
    config = get_config(config_path='configs/sample_dataset_sr_2025-07-07.toml')
    data_type = 'flat_center'
    
    # Get the atlas config path
    atlas_config = config.cam.atlas_fit_config
    
    if not atlas_config or not Path(atlas_config).exists():
        print(f"Error: Atlas config file not found at {atlas_config}")
        return False
    
    print(f"Using atlas config: {atlas_config}")
    
    # Access auxiliary files
    atlas_lines_upper = config.dataset[data_type]['files'].auxiliary.get('atlas_lines_upper')
    offset_map_file = config.dataset[data_type]['files'].auxiliary.get('offset_map')
    illumination_pattern_file = config.dataset[data_type]['files'].auxiliary.get('illumination_pattern')
    
    if not atlas_lines_upper or not atlas_lines_upper.exists():
        print(f"Error: Atlas lines file (upper) not found")
        print("Please run atlas-fit prepare first")
        return False
        
    if not offset_map_file or not offset_map_file.exists():
        print(f"Error: Offset map not found")
        print("Please run spectroflat processing first")
        return False
        
    if not illumination_pattern_file or not illumination_pattern_file.exists():
        print(f"Error: Illumination pattern not found")
        print("Please run spectroflat processing first")
        return False
    
    print(f"Using atlas lines (upper): {atlas_lines_upper}")
    print(f"Using offset map: {offset_map_file}")
    print(f"Using illumination pattern: {illumination_pattern_file}")
    
    # Read L0 data and create temporary FITS file for corrected_frame (like test_atlas_fit_window)
    l0_file = config.dataset[data_type]['files'].get('l0')
    if not l0_file or not l0_file.exists():
        print(f"Error: L0 file not found")
        return False
    
    print(f"Reading L0 data from: {l0_file}")
    
    # Extract upper and lower frame data to create FITS file like spectroflat
    try:
        from astropy.io import fits
        import numpy as np
        
        # Read the L0 data properly using themis_io
        flat_data, flat_header = tio.read_any_file(config, data_type, verbose=False, status='l0')
        
        # Get the upper and lower frame data
        if len(flat_data) > 0:
            upper_frame_data = flat_data[0]['upper'].data
            lower_frame_data = flat_data[0]['lower'].data
            
            # Stack as 4 states to match the illumination_pattern (soft_flat) from spectroflat
            # Spectroflat was run with 4 states, so illumination_pattern has 4 states
            # We duplicate lower frames to match: [upper, lower, lower, lower]
            stacked_data = np.stack([upper_frame_data, lower_frame_data, lower_frame_data, lower_frame_data], axis=0)
            
            # Create a temporary FITS file in the reduced data directory
            temp_fits_path = Path(config.directories.reduced) / 'temp_flat_center_stacked_for_amend_test.fits'
            hdu = fits.PrimaryHDU(data=stacked_data)
            hdu.writeto(temp_fits_path, overwrite=True)
            
            print(f"Created temporary FITS file: {temp_fits_path}")
            print(f"Stacked shape: {stacked_data.shape} [state=4, spatial, wavelength]")
        else:
            print("Error: No frames found in L0 data")
            return False
            
    except Exception as e:
        import traceback
        print(f"Error extracting frame data: {e}")
        print(f"Full traceback:")
        traceback.print_exc()
        return False
    
    # Modify the atlas config by updating corrected_frame, offset_map, and soft_flat
    import re
    
    with open(atlas_config, 'r') as f:
        original_config_text = f.read()
    
    # Create modified config
    temp_config_text = original_config_text
    
    # Update corrected_frame path
    temp_config_text = re.sub(
        r'corrected_frame:\s*(.+)',
        f'corrected_frame: {temp_fits_path}',
        temp_config_text
    )
    
    # Update ROI to include state dimension for 3D array: [s, spatial, wavelength]
    # The 's' placeholder gets replaced with state numbers (0,1,2,3) by amend_spectroflat
    temp_config_text = re.sub(
        r'roi:\s*"?\[([^\]]+)\]"?',
        r'roi: "[s, \1]"',
        temp_config_text
    )
    
    # Add or update mod_states parameter (4 states to match illumination_pattern)
    if 'mod_states:' in temp_config_text:
        temp_config_text = re.sub(
            r'mod_states:\s*\d+',
            'mod_states: 4',
            temp_config_text
        )
    else:
        # Add after corrected_frame
        temp_config_text = re.sub(
            r'(corrected_frame:\s*.+)',
            r'\1\n  mod_states: 4',
            temp_config_text
        )
    
    # Add or update offset_map in input section
    if 'offset_map:' in temp_config_text:
        temp_config_text = re.sub(
            r'offset_map:\s*(.+)',
            f'offset_map: {offset_map_file}',
            temp_config_text
        )
    else:
        # Add after corrected_frame
        temp_config_text = re.sub(
            r'(corrected_frame:\s*.+)',
            rf'\1\n  offset_map: {offset_map_file}',
            temp_config_text
        )
    
    # Add or update soft_flat in input section (atlas-fit expects 'soft_flat' not 'illumination_pattern')
    if 'soft_flat:' in temp_config_text:
        temp_config_text = re.sub(
            r'soft_flat:\s*(.+)',
            f'soft_flat: {illumination_pattern_file}',
            temp_config_text
        )
    else:
        # Add after offset_map
        temp_config_text = re.sub(
            r'(offset_map:\s*.+)',
            rf'\1\n  soft_flat: {illumination_pattern_file}',
            temp_config_text
        )
    
    # Write temporary config file
    temp_config_path = project_root / 'configs' / 'temp_amend_test_config.yml'
    with open(temp_config_path, 'w') as f:
        f.write(temp_config_text)
    
    print(f"Created temporary config: {temp_config_path}")
    
    # Provide command for external terminal execution
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    print("\nPlease run the following command in an EXTERNAL terminal:")
    print("\n" + "-"*70)
    print(f"cd {config.directories.reduced}")
    print(f"{amend_script} {temp_config_path} {atlas_lines_upper}")
    print("-"*70)
    
    print("\nThis will:")
    print("  - Apply wavelength calibration to the offset map")
    print("  - Amend the soft flat (illumination pattern)")
    print("  - Create output files:")
    print("    • wl_calibrated_offsets.fits")
    print("    • amended_soft_flat.fits")
    
    # Wait for user confirmation
    while True:
        user_input = input("\nDid amend-spectroflat complete successfully? (y/n): ").strip().lower()
        if user_input == 'y':
            print("Checking for output files...")
            break
        elif user_input == 'n':
            print("Amend-spectroflat did not complete successfully.")
            return False
        else:
            print("Please enter 'y' or 'n'")
    
    # Check if output files exist in the data directory
    output_dir = Path(config.directories.reduced)
    wl_calibrated_offsets = output_dir / 'wl_calibrated_offsets.fits'
    amended_soft_flat = output_dir / 'amended_soft_flat.fits'
    
    success = True
    
    if wl_calibrated_offsets.exists():
        print(f"✓ Found: {wl_calibrated_offsets}")
    else:
        print(f"✗ Not found: {wl_calibrated_offsets}")
        success = False
    
    if amended_soft_flat.exists():
        print(f"✓ Found: {amended_soft_flat}")
    else:
        print(f"✗ Not found: {amended_soft_flat}")
        success = False
    
    if success:
        print("\nAmend-spectroflat completed successfully!")
    else:
        print("\nAmend-spectroflat did not create all expected output files.")
    
    # Cleanup: Revert config file and remove temporary files
    print("\nCleaning up...")
    
    # Restore original atlas config
    try:
        with open(atlas_config, 'w') as f:
            f.write(original_config_text)
        print(f"✓ Restored original config: {atlas_config}")
    except Exception as e:
        print(f"✗ Failed to restore config: {e}")
    
    # Remove temporary FITS file
    if temp_fits_path.exists():
        temp_fits_path.unlink()
        print(f"✓ Removed temporary FITS: {temp_fits_path}")
    
    # Remove temporary config file
    if temp_config_path.exists():
        temp_config_path.unlink()
        print(f"✓ Removed temporary config: {temp_config_path}")
    
    print("Cleanup complete.")
    
    return success

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
    original_offset_map = config.dataset[data_type]['files'].auxiliary.get('offset_map')
    wl_calibrated_offsets = output_dir / 'wl_calibrated_offsets.fits'
    original_soft_flat = config.dataset[data_type]['files'].auxiliary.get('illumination_pattern')
    amended_soft_flat = output_dir / 'amended_soft_flat.fits'
    
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
    success = test_amend_spectroflat()
    if success:
        print("\nTest completed successfully!")
        
        # Optionally plot results
        try:
            from themis.datasets.themis_datasets_2025 import get_config
            config = get_config(config_path='configs/sample_dataset_sr_2025-07-07.toml')
            
            user_input = input("\nWould you like to plot the results? (y/n): ").strip().lower()
            if user_input == 'y':
                plot_amend_spectroflat_results(config, data_type='flat_center')
        except Exception as e:
            print(f"Could not plot results: {e}")
        
        sys.exit(0)
    else:
        print("\nTest failed!")
        sys.exit(1)
