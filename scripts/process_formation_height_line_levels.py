#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 4 15:24:27 2025

@author: zeuner


    # Example usage: load L4 data for different positions and lines

    # Option 1: Load specific positions
    ti_disk_center_l4, ti_disk_center_header = load_l4_for_position('ti', 'disk_center')
    print(f"Loaded TI disk center L4 data: {ti_disk_center_l4.keys() if hasattr(ti_disk_center_l4, 'keys') else 'scalar'}")

    # Option 2: Iterate over all positions for a line
    print("\nIterating over all TI positions:")
    for position, sequence in iterate_positions('ti'):
        print(f"  Position: {position}, Sequence: {sequence}")

    # Option 3: Load all L4 data for a line at once
    print("\nLoading all TI L4 data:")
    ti_all_l4 = load_all_l4_for_line('ti')

    # Option 4: Get all available lines and positions
    print(f"\nAvailable lines: {get_all_lines()}")
    print(f"TI positions: {get_all_positions('ti')}")
    print(f"SR positions: {get_all_positions('sr')}")

    # Option 5: Loop over all lines and positions
    print("\nLooping over all lines and positions:")
    for line in get_all_lines():
        print(f"  Line: {line}")
        for position in get_all_positions(line):
            sequence = get_formation_height_config(line, position)
            print(f"    {position}: sequence {sequence}")


"""

from themis.core import themis_tools as tt
from themis.core import themis_data_reduction as tdr
from themis.core import themis_io as tio
from themis.datasets.themis_datasets_2025 import get_config
from themis.plots import plot_power_spectrum

from spectator.controllers.app_controller import display_data # from spectator

import matplotlib.pyplot as plt
import numpy as np
import gc

#----------------------------------------------------------------------
"""
MAIN: 
   
"""

# Configuration file path
CONFIG_PATH = 'configs/formation_dataset_2025-07-05.toml'

# Configuration for formation height analysis: maps disk positions to sequences
# Based on comments in formation_dataset_2025-07-05.toml
# m := minus (e.g., m40 = -40)
# Both TI and SR lines share the same positions and sequences
POSITION_TO_SEQUENCE = {
    'disk_center': 27,
    'm40': 24,
    'm30': 18,
    '30': 50,
    '40': 36,
    '45': 38,
    'm45': 48,
    'm55': 92,
    '55': 86,
}

# Available spectral lines
AVAILABLE_LINES = ['ti', 'sr']

# Helper: convert position name to numeric value (for plotting labels, etc.)
POSITION_TO_NUMERIC = {
    'disk_center': 0,
    'm40': -40,
    'm30': -30,
    '30': 30,
    '40': 40,
    '45': 45,
    'm45': -45,
    'm55': -55,
    '55': 55,
}

# Predefined crop configurations for each line
CROP_CONFIGS = {
    'sr': {'slit': (0, -1), 'spatial': (2, -55), 'wavelength': (276, 886)},
    'ti': {'slit': (0, -1), 'spatial': (2, -55), 'wavelength': (406, 1016)},
}

def crop_to_array(crop_dict):
    """
    Convert crop dictionary to array format for internal use.

    Args:
        crop_dict: Dictionary with keys 'slit', 'spatial', 'wavelength',
                   each containing a tuple (start, end).

    Returns:
        list: Array [slit_start, slit_end, spatial_start, spatial_end, wl_start, wl_end]
    """
    return [
        crop_dict['slit'][0], crop_dict['slit'][1],
        crop_dict['spatial'][0], crop_dict['spatial'][1],
        crop_dict['wavelength'][0], crop_dict['wavelength'][1]
    ]

def get_formation_height_config(line: str, position: str):
    """
    Get configuration for a specific line and disk position.

    Args:
        line: Spectral line ('ti' or 'sr')
        position: Disk position ('disk_center', 'm40', etc.)

    Returns:
        int: Sequence number for the given line and position
    """
    return POSITION_TO_SEQUENCE[position]

def get_all_lines():
    """
    Get all available spectral lines.

    Returns:
        list: List of line identifiers (e.g., ['ti', 'sr'])
    """
    return AVAILABLE_LINES

def get_all_positions(line: str = None):
    """
    Get all available disk positions.

    Args:
        line: Spectral line (optional, kept for backward compatibility)
              Since positions are shared, this parameter is not used.

    Returns:
        list: List of position strings (e.g., ['disk_center', 'm40', 'm30', ...])
    """
    return list(POSITION_TO_SEQUENCE.keys())

def iterate_positions(line: str = None):
    """
    Iterate over all positions, yielding (position, sequence) pairs.

    Args:
        line: Spectral line (optional, kept for backward compatibility)
              Since positions are shared, this parameter is not used.

    Yields:
        tuple: (position, sequence) for each position
    """
    for position, sequence in POSITION_TO_SEQUENCE.items():
        yield position, sequence

def load_all_l4_for_line(line: str):
    """
    Load L4 data for all positions of a given line.

    Args:
        line: Spectral line ('ti' or 'sr')

    Returns:
        dict: Dictionary mapping position -> (l4_data, header)
    """
    results = {}
    for position, sequence in iterate_positions(line):
        l4_data, header = load_l4_for_position(line, position)
        results[position] = (l4_data, header)
        print(f"Loaded {line.upper()} at {position} (sequence {sequence})")
    return results

def load_l4_for_position(line: str, position: str):
    """
    Load L4 data for a specific line and disk position.

    Args:
        line: Spectral line ('ti' or 'sr')
        position: Disk position ('disk_center', '-40', etc.)

    Returns:
        tuple: (l4_data, header) from tio.read_any_file
    """
    sequence = get_formation_height_config(line, position)
    config = get_config(
        line=line,
        config_path=CONFIG_PATH,
        auto_discover_files=True,
        auto_create_dirs=False
    )
    # Override sequence for this position
    config.dataset['scan']['sequence'] = sequence
    return tio.read_any_file(config, data_type='flat_center', status='l4')


def calculate_intensity(line: str, position: str, crop=None):
    """
    Calculate intensity for a specific line and position.

    Args:
        line: Spectral line ('ti' or 'sr')
        position: Disk position ('disk_center', '-40', etc.)
        crop: Optional crop specification. Can be:
              - None: no cropping
              - Array: [slit_start, slit_end, spatial_start, spatial_end, wl_start, wl_end]
              - Dict: {'slit': (start, end), 'spatial': (start, end), 'wavelength': (start, end)}

    Returns:
        numpy array: Intensity values
    """
    data, header = load_l4_for_position(line, position)

    # Check that only one map exists in the cycle set
    map_indices = sorted(set(k[2] for k in data.keys()))
    if len(map_indices) > 1:
        raise ValueError(f"Multiple maps found in cycle set for {line} at {position}: {map_indices}. "
                         f"Expected exactly one map.")
    map_idx = map_indices[0]

    if line == 'sr':
        result = tt.compute_polarimetry(data)
        # Collapse the map dimension (axis 1) to match TI shape
        sr_intensity = result.uml.I[:, map_idx, :, :]
        intensity = sr_intensity.squeeze(axis=1) if sr_intensity.shape[1] == 1 else sr_intensity
    elif line == 'ti': # use only lower state, as argued in the data reduction documentation
        # Since we verified only one map exists, stacking all is fine
        intensity = data.get_state('pQ').stack_all('lower')

    # Apply cropping if specified
    if crop is not None:
        # Convert dict to array if needed
        if isinstance(crop, dict):
            crop = crop_to_array(crop)
        slit_start, slit_end, spatial_start, spatial_end, wl_start, wl_end = crop
        intensity = intensity[slit_start:slit_end, spatial_start:spatial_end, wl_start:wl_end]

    return intensity
            
    
    

def calculate_intensity_for_all(crop=False):
    """
    Calculate intensity for all lines and positions and plot spatially averaged spectra.

    Args:
        crop: Boolean to control cropping. If True, uses predefined crop configurations
              from CROP_CONFIGS (line-dependent). If False, no cropping is applied.

    Returns:
        tuple: (intensity, wavelength) where intensity is a dictionary mapping
               line -> position -> full intensity array, and wavelength is a
               dictionary mapping line -> wavelength array
    """
    intensity = {}
    wavelength = {}
    spatial_avg_intensity = {}  # For plotting only

    for line in get_all_lines():
        print(f"  Line: {line}")
        intensity[line] = {}
        wavelength[line] = {}
        spatial_avg_intensity[line] = {}

        # Load wavelength once per line
        config = get_config(
            line=line,
            config_path=CONFIG_PATH,
            auto_discover_files=True,
            auto_create_dirs=False
        )
        wavelength[line] = tio.read_any_file(config, data_type='scan', status='wl')

        # Determine crop configuration for this line
        line_crop = CROP_CONFIGS[line] if crop else None

        # Apply wavelength cropping if crop is enabled
        if crop and line_crop is not None:
            wl_crop = line_crop['wavelength']
            wavelength[line] = wavelength[line][wl_crop[0]:wl_crop[1]]

        for position in get_all_positions(line):
            print(f"    Position: {position}")
            intensity_data = calculate_intensity(line, position, line_crop)

            # Store full intensity data
            intensity[line][position] = intensity_data

            # Calculate spatial average (mean over spatial axis) for plotting
            # Assuming shape is (n_spatial, n_wavelength) or (n_slit, n_spatial, n_wavelength)
            if intensity_data.ndim == 2:
                spatial_avg = np.mean(intensity_data, axis=0)
            elif intensity_data.ndim == 3:
                spatial_avg = np.mean(intensity_data, axis=(0, 1))
            else:
                spatial_avg = intensity_data

            spatial_avg_intensity[line][position] = spatial_avg

    # Create figure with 2 subplots (one for each line)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for idx, line in enumerate(get_all_lines()):
        ax = axes[idx]
        wl = wavelength[line]
        for position in get_all_positions(line):
            spec = spatial_avg_intensity[line][position]
            ax.plot(wl, spec, label=position)

        ax.set_title(f"{line.upper()} Spatially Averaged Spectrum")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return intensity, wavelength

def display_one_position(position: str, crop_sr=None, crop_ti=None):
    """
    Display intensity data for a single position for both lines.

    Args:
        position: Disk position ('disk_center', '-40', etc.)
        crop_sr: Optional crop specification for SR data (dict or array format)
        crop_ti: Optional crop specification for TI data (dict or array format)
    """
    sr = calculate_intensity('sr', position, crop_sr)
    ti = calculate_intensity('ti', position, crop_ti)
    data_plot = np.array([sr, ti])
    viewer = display_data( data_plot, ['states', 'spatial_y', 'spatial_x', 'spectral'],
                          title=position,
                          state_names=['Sr', 'Ti']
                          )
          
#%%
if __name__ == '__main__':

    # Use predefined crop configurations (easier to read and modify)

    # check out one position
    #display_one_position('m40', crop_sr=CROP_CONFIGS['sr'], crop_ti=CROP_CONFIGS['ti'])
    
    
    
    # Example: calculate and plot intensity for all lines and positions
    intensity, wavelength = calculate_intensity_for_all(crop=True)  # Use line-dependent crop configs

# -----------------------------------------------------

