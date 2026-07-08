#!/usr/bin/env python3
"""
Test script to read L4 file and plot mean intensity over wavelength.
Uses SR line data as a test case.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.themis.datasets.themis_datasets_2025 import get_config
from src.themis.core import themis_io as tio
from src.themis.core import themis_tools as tt

def test_l4_intensity_plot():
    """
    Read SR L4 data and plot mean intensity over wavelength.
    """
    
    line = 'ti'
    # Load config for SR line
    config = get_config(
        line=line,
        config_path='configs/formation_dataset_2025-07-05.toml',
        auto_discover_files=True,
        auto_create_dirs=False
    )

    # Load L4 data for disk_center position
    print("Loading L4 data for disk_center...")
    data, header = tio.read_any_file(config, data_type='scan', status='l4', verbose=False)

    # Compute polarimetry to get intensity
    print("Computing polarimetry...")
    result = tt.compute_polarimetry(data)
    intensity = result.uml.I[5:-5,0,5:-10,100:-50]

    print(f"Intensity shape: {intensity.shape}")

    # Get wavelength from delta_offsets_upper.fits
    print("Loading wavelength calibration...")
    wavelength, _ = tio.read_any_file(config, data_type='scan', status='wl', verbose=True)
   

    print(f"Wavelength shape: {wavelength.shape}")
    print(f"Wavelength range: {wavelength.min():.4f} - {wavelength.max():.4f} nm")

    # Calculate spatial average (mean over spatial axes)
    # intensity shape is (n_slit, n_map, n_spatial, n_wavelength)
    # We want to average over slit and spatial dimensions

    spatial_avg = np.mean(intensity, axis=(0, 1))  # Average over slit and spatial
  

    print(f"Spatially averaged intensity shape: {spatial_avg.shape}")

    # Plot mean intensity over wavelength
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength[100:-50], spatial_avg, label='disk_center')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity')
    plt.title(line+' L4 - Spatially Averaged Spectrum (disk_center)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Test completed successfully!")

if __name__ == '__main__':
    test_l4_intensity_plot()
