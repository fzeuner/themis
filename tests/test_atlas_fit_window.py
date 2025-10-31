#!/usr/bin/env python3
"""
Test script to verify atlas-fit window opening.
This script tests if the interactive matplotlib window opens correctly
when running atlas-fit prepare.
"""
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

def test_atlas_fit_window():
    """Test if atlas-fit prepare opens the interactive matplotlib window."""
    
    # Get the project root and atlas-fit prepare script
    project_root = Path(__file__).resolve().parents[1]
    prepare_script = project_root / 'atlas-fit' / 'bin' / 'prepare'
    
    if not prepare_script.exists():
        print(f"Error: Atlas-fit prepare script not found at {prepare_script}")
        return False
    
    # Get the atlas config file
    atlas_config = project_root / 'configs' / 'atlas_fit_config_cam1.yml'
    
    if not atlas_config.exists():
        print(f"Error: Atlas config file not found at {atlas_config}")
        return False
    
    # Find an existing flat_center L0 file to use for testing
    from themis.datasets.themis_datasets_2025 import get_config
    from themis.core import themis_io as tio
    import tempfile
    
    config = get_config()
    flat_center_l0_file = config.dataset['flat_center']['files'].get('l0')
    
    if not flat_center_l0_file or not flat_center_l0_file.exists():
        print("Error: No flat_center L0 file found for testing")
        return False
    
    print(f"Using existing flat_center L0 file: {flat_center_l0_file}")
    
    # Extract the upper frame data to create a simple FITS file for atlas-fit
    try:
        from astropy.io import fits
        
        # Read the flat_center data properly using themis_io
        flat_data, flat_header = tio.read_any_file(config, 'flat_center', verbose=False, status='l0')
        
        # Get the upper frame data
        if len(flat_data) > 0:
            upper_frame_data = flat_data[0]['upper'].data
            
            # Create a temporary FITS file with the 2D frame data
            # The ROI in atlas-fit config will extract 1D spectral data
            temp_fits_path = project_root / 'temp_flat_center_upper_for_test.fits'
            hdu = fits.PrimaryHDU(data=upper_frame_data)
            hdu.writeto(temp_fits_path, overwrite=True)
            
            print(f"Created temporary 2D frame FITS file: {temp_fits_path}")
            print(f"Frame shape: {upper_frame_data.shape}")
        else:
            print("Error: No frames found in flat_center data")
            return False
            
    except Exception as e:
        print(f"Error extracting frame data: {e}")
        return False
    
    # Create a temporary config file that points to the existing flat_center file
    import re
    
    with open(atlas_config, 'r') as f:
        config_text = f.read()
    
    # Replace the corrected_frame path with our temporary FITS file
    temp_config_text = re.sub(
        r'corrected_frame:\s*(.+)',
        f'corrected_frame: {temp_fits_path}',
        config_text
    )
    
    # Don't override the ROI - use the original configuration that extracts 1D data
    # The original ROI "[25,10:]" extracts row 25 from column 10 onwards, creating 1D data
    
    # Write temporary config file
    temp_config_path = project_root / 'configs' / 'temp_atlas_test_config.yml'
    with open(temp_config_path, 'w') as f:
        f.write(temp_config_text)
    
    print("Testing atlas-fit prepare window opening...")
    print(f"Script: {prepare_script}")
    print(f"Temp Config: {temp_config_path}")
    print("\nNote: An interactive matplotlib window should open for line selection.")
    print("If the window opens, close it after testing to continue.")
    print("\nRunning atlas-fit prepare...")
    
    try:
        # Run atlas-fit prepare without capturing output to allow interactive window
        result = subprocess.run(
            [str(prepare_script), str(temp_config_path)],
            timeout=60  # 1 minute timeout for testing
        )
        
        if result.returncode == 0:
            print("Atlas-fit prepare completed successfully!")
            return True
        else:
            print(f"Atlas-fit prepare failed with return code: {result.returncode}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Atlas-fit prepare timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"Error running atlas-fit prepare: {e}")
        return False
    
    finally:
        # Clean up temporary files
        if temp_config_path.exists():
            temp_config_path.unlink()
        if 'temp_fits_path' in locals() and temp_fits_path.exists():
            temp_fits_path.unlink()

if __name__ == "__main__":
    success = test_atlas_fit_window()
    if success:
        print("\nTest completed successfully!")
        sys.exit(0)
    else:
        print("\nTest failed!")
        sys.exit(1)
