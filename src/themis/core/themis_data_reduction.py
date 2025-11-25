#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 12:43:33 2025

@author: zeuner
"""
import textwrap
from themis.core import themis_tools as tt
from themis.core import data_classes as dct
from themis.core import themis_io as tio

class ReductionLevel:
    def __init__(self, name, file_ext, func, per_type_meta=None):
        self.name = name
        self.file_ext = file_ext
        self.func = func  # Callable
        self.per_type_meta = per_type_meta or {}  # Optional: interpretation per data type

    def reduce(self, *args, **kwargs):
        """
        Forward all arguments to the underlying reduction function.
        This allows level-specific functions to define their own signatures
        (e.g., reduce_raw_to_l0(config, data_type=None, return_reduced=False)).
        """
        return self.func(*args, **kwargs)

    def get_description(self, data_type=None, width=80, indent="  "):
        """Get formatted description for this reduction level.
        
        Args:
            data_type: Specific data type ('dark', 'scan', 'flat', etc.)
            width: Maximum line width for text wrapping
            indent: Indentation string for wrapped lines
            
        Returns:
            Formatted description string
        """
        if data_type and data_type in self.per_type_meta:
            desc = self.per_type_meta[data_type]
            # Clean up the description: remove line continuation backslashes and extra whitespace
            desc = desc.replace('\\\n', ' ').replace('\n', ' ')
            desc = ' '.join(desc.split())  # Normalize whitespace
            # Wrap text nicely
            wrapped = textwrap.fill(desc, width=width, initial_indent=indent, 
                                   subsequent_indent=indent)
            return f"{self.name} level for '{data_type}':\n{wrapped}"
        return f"{self.name} level. For full description provide optional keyword data_type='data_type'"

    def __repr__(self):
        """Concise, informative summary of this reduction level.

        Example:
            ReductionLevel(name='l0', file_extension='_l0.fits', func='reduce_raw_to_l0', supports=['dark','scan','flat'])
        """
        func_name = getattr(self.func, "__name__", str(self.func))
        supports = list(self.per_type_meta.keys()) if self.per_type_meta else []
        return (
            "ReductionLevel("
            f"name='{self.name}', "
            f"file_extension='{self.file_ext}', "
            f"func='{func_name}', "
            f"supports={supports}"
            ")"
        )
    
    
class ReductionRegistry:
    def __init__(self):
        self._levels = {}

    def add(self, level):
        self._levels[level.name] = level

    def __getitem__(self, name):
        if name not in self._levels:
            available = list(self._levels.keys())
            raise KeyError(
                f"Reduction level '{name}' not found. "
                f"Available levels: {available}"
            )
        return self._levels[name]

    def __getattr__(self, name):
        if name.startswith('_'):
            # Allow normal attribute access for private attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if name in self._levels:
            return self._levels[name]
        available = list(self._levels.keys())
        raise AttributeError(
            f"No reduction level named '{name}'. "
            f"Available levels: {available}"
        )

    def __iter__(self):
        return iter(self._levels)

    def items(self):
        return self._levels.items()

    def keys(self):
        return self._levels.keys()

    def values(self):
        return self._levels.values()

    def list_levels(self):
        return list(self._levels.keys())

    def __repr__(self):
        if not self._levels:
            return "ReductionRegistry(n=0, levels=[])"

        # Render per level with explicit fields
        parts = []
        for name, lvl in self._levels.items():
            func_name = getattr(lvl.func, "__name__", str(lvl.func))
            parts.append(
                f"{{name='{name}', file_extension='{lvl.file_ext}', func='{func_name}'}}"
            )

        # Compact if many
        if len(parts) > 6:
            shown = parts[:3] + ["..."] + parts[-2:]
        else:
            shown = parts

        return (
            "ReductionRegistry("
            f"n={len(self._levels)}, "
            "levels=[" + ", ".join(shown) + "]"
            ")"
        )


def process_wavelength_calibration_with_atlas_fit(config, data_type):
    """
    Process wavelength calibration using atlas-fit for flat_center or flat data.
    
    This function extracts upper and lower frames, runs atlas-fit prepare on each,
    and stores the resulting atlas lines files (containing line info and dispersion)
    in the FileSet auxiliary entry.
    
    Parameters
    ----------
    config : Config
        Configuration object containing dataset and directory information
    data_type : str
        Type of data to process ('flat_center' or 'flat')
    
    Returns
    -------
    dict
        Dictionary with 'upper' and 'lower' keys mapping to Path objects of atlas lines files
    """
    import shutil
    import re
    from pathlib import Path
    from astropy.io import fits
    
    print(f'Performing wavelength calibration on {data_type} using atlas-fit...')
    
    # Check if atlas-fit config is available
    if not hasattr(config.cam, 'atlas_fit_config') or config.cam.atlas_fit_config is None:
        print('Error: No atlas_fit_config found for this camera.')
        return {}
    
    atlas_config_path = config.cam.atlas_fit_config
    if not Path(atlas_config_path).exists():
        print(f'Error: Atlas fit config file not found: {atlas_config_path}')
        return {}
    
    # Get the L0 file path
    l0_file = config.dataset[data_type]['files'].get('l0')
    if l0_file is None or not l0_file.exists():
        print(f'Error: L0 {data_type} file not found.')
        return {}
    
    # Find atlas-fit prepare script
    project_root = Path(__file__).resolve().parents[3]
    prepare_script = project_root / 'atlas-fit' / 'bin' / 'prepare'
    if not prepare_script.exists():
        print(f'Error: Atlas-fit prepare script not found: {prepare_script}')
        return {}
    
    # Read the original config as text to preserve formatting and comments
    with open(atlas_config_path, 'r') as f:
        original_config_text = f.read()
    
    # Extract the original corrected_frame path
    match = re.search(r'corrected_frame:\s*(.+)', original_config_text)
    if match:
        original_input_file = match.group(1).strip()
    else:
        print('Error: Could not find corrected_frame in config file')
        return {}
    
    # Get base filename from L0 file (without level extension)
    base_filename = l0_file.stem.replace('_l0', '')
    
    # Dictionary to store generated atlas lines files
    atlas_lines_files = {}
    
    # Process both frames (upper and lower)
    for frame_name in ['upper', 'lower']:
        print(f'Processing {frame_name} frame...')
        
        # Extract individual frame to temporary file
        temp_frame_path = config.directories.reduced / f'temp_{data_type}_{frame_name}_for_atlas.fits'
        
        try:
            # Read the data properly using themis_io
            data, header = tio.read_any_file(config, data_type, verbose=False, status='l0')
            
            # Get the frame data from FramesSet structure
            # L0 has one frame (index 0) with upper and lower halves
            if len(data) > 0:
                if frame_name == 'upper':
                    frame_data = data[0]['upper'].data
                elif frame_name == 'lower':
                    frame_data = data[0]['lower'].data
                else:
                    print(f'Unknown frame name: {frame_name}')
                    continue
                
                # Create a simple FITS file with the 2D frame data
                # The ROI in atlas-fit config will extract 1D spectral data
                hdu = fits.PrimaryHDU(data=frame_data)
                hdu.writeto(temp_frame_path, overwrite=True)
            else:
                print(f'No frames found in {data_type} data')
                continue
                
        except Exception as e:
            print(f'Error extracting {frame_name} frame: {e}')
            continue
        
        # Only modify the corrected_frame path using regex to preserve formatting
        modified_config_text = re.sub(
            r'corrected_frame:\s*(.+)',
            f'corrected_frame: {temp_frame_path}',
            original_config_text
        )
        
        # Check if output file already exists
        lines_file = config.directories.reduced / f'{base_filename}_{frame_name}_atlas_lines.yaml'
        
        if lines_file.exists():
            # File exists - ask user what to do
            print(f'\n{"="*70}')
            print(f'{frame_name.upper()} FRAME: Output file already exists')
            print(f'{"="*70}')
            print(f'File: {lines_file.name}')
            
            while True:
                user_choice = input(f'\nOptions:\n  [o] Overwrite (run atlas-fit again)\n  [k] Keep existing file (skip)\nChoice: ').strip().lower()
                if user_choice in ['o', 'k']:
                    break
                else:
                    print('Please enter "o" to overwrite or "k" to keep existing')
            
            if user_choice == 'k':
                # Keep existing file
                atlas_lines_files[frame_name] = lines_file
                print(f'✓ Using existing file: {lines_file.name}')
                # Clean up temporary frame file
                if temp_frame_path.exists():
                    temp_frame_path.unlink()
                continue  # Skip to next frame
        
        # Write the modified config (only change the corrected_frame line)
        with open(atlas_config_path, 'w') as f:
            f.write(modified_config_text)
        
        try:
            # Display command to run in external terminal
            print(f'\n{"="*70}')
            print(f'READY TO PROCESS {frame_name.upper()} FRAME')
            print(f'{"="*70}')
            print(f'\nPlease run the following command in an EXTERNAL terminal:')
            print(f'\n{"-"*70}')
            print(f'cd {config.directories.reduced}')
            print(f'{prepare_script} {atlas_config_path}')
            print(f'{"-"*70}')
            
            print('\nNote: An interactive matplotlib window should open for line selection.')
            print('When complete, the output will be automatically saved.')
            input('\nPress ENTER after atlas-fit completes...')
            
            # Check if output file was created and rename it
            temp_lines = config.directories.reduced / 'atlas_fit_lines.yaml'
            
            if temp_lines.exists():
                shutil.move(str(temp_lines), str(lines_file))
                atlas_lines_files[frame_name] = lines_file
                print(f'✓ Atlas lines file saved to: {lines_file}')
                print('   (This file contains both line information and dispersion solution)')
            else:
                print(f'✗ Warning: Atlas lines file not found: {temp_lines}')
                print(f'   Expected output file was not created. Skipping {frame_name} frame.')
        
        finally:
            # Clean up temporary frame file
            if temp_frame_path.exists():
                temp_frame_path.unlink()
    
    # Restore original config file (restore the original corrected_frame line)
    restored_config_text = re.sub(
        r'corrected_frame:\s*(.+)',
        f'corrected_frame: {original_input_file}',
        modified_config_text if 'modified_config_text' in locals() else original_config_text
    )
    with open(atlas_config_path, 'w') as f:
        f.write(restored_config_text)
    
    print(f'\nWavelength calibration complete for {data_type}.')
    print(f'Generated {len(atlas_lines_files)} atlas lines files.')
    
    # Add the atlas lines files to the FileSet's auxiliary entry
    if atlas_lines_files:
        file_set = config.dataset[data_type]['files']
        if not hasattr(file_set, 'auxiliary'):
            file_set.auxiliary = {}
        
        for frame_name, file_path in atlas_lines_files.items():
            file_set.auxiliary[f'atlas_lines_{frame_name}'] = file_path
        
        print(f'Added {len(atlas_lines_files)} atlas lines files to FileSet auxiliary entry.')
    
    return atlas_lines_files

def process_spectroflat(config, data_type):
    """
    Process spectroflat for flat_center or flat data.
    
    This function extracts upper and lower frames, runs spectroflat on both together
    (spectroflat needs at least two "states"), and returns the dust_flat and offset_map
    in an L1 FramesSet structure.
    
    Parameters
    ----------
    config : Config
        Configuration object containing dataset and directory information
    data_type : str
        Type of data to process ('flat_center' or 'flat')
    
    Returns
    -------
    FramesSet
        L1 reduced frames containing dust_flat (upper/lower) with offset_map in metadata
    """
    from spectroflat import Analyser, Config as SpectroflatConfig, SmileConfig, SensorFlatConfig
    from qollib.strings import parse_shape
    import numpy as np
    from pathlib import Path
    
    print(f'Processing spectroflat for {data_type}...')
    
    # Clean up outdated files if they exist (re-running spectroflat from scratch)
    outdated_offset = config.dataset[data_type]['files'].auxiliary.get('offset_map_outdated')
    outdated_illumination = config.dataset[data_type]['files'].auxiliary.get('illumination_pattern_outdated')
    
    if outdated_offset or outdated_illumination:
        print(f'  Found outdated files from previous amend run. Cleaning up...')
        
        # Remove outdated files
        if outdated_offset and outdated_offset.exists():
            outdated_offset.unlink()
            print(f'    ✓ Removed: {outdated_offset.name}')
            del config.dataset[data_type]['files'].auxiliary['offset_map_outdated']
        
        if outdated_illumination and outdated_illumination.exists():
            outdated_illumination.unlink()
            print(f'    ✓ Removed: {outdated_illumination.name}')
            del config.dataset[data_type]['files'].auxiliary['illumination_pattern_outdated']
        
        # Remove current offset_map and illumination_pattern (amended versions)
        current_offset = config.dataset[data_type]['files'].auxiliary.get('offset_map')
        current_illumination = config.dataset[data_type]['files'].auxiliary.get('illumination_pattern')
        
        if current_offset and current_offset.exists():
            current_offset.unlink()
            print(f'    ✓ Removed: {current_offset.name}')
        
        if current_illumination and current_illumination.exists():
            current_illumination.unlink()
            print(f'    ✓ Removed: {current_illumination.name}')
        
        print(f'  Spectroflat will generate fresh offset_map and illumination_pattern files.')
    
    # Read the L0 data
    data, header = tio.read_any_file(config, data_type, verbose=False, status='l0')
    
    # L0 data has one frame (index 0) with upper and lower halves
    if len(data) == 0:
        print(f'Error: No frames found in {data_type} L0 data')
        return None
    
    # Extract upper and lower as 2D arrays [spatial, wavelength]
    upper_2d = data[0]['upper'].data  # shape: (spatial, wavelength)
    lower_2d = data[0]['lower'].data  # shape: (spatial, wavelength)
    
    # Spectroflat needs input as [state, spatial, wavelength]
    # Stack upper and lower as two "states"
    dirty_flat = np.stack([upper_2d, lower_2d, lower_2d, lower_2d], axis=0)  # shape: (4, spatial, wavelength)
    
    print(f'  Input shape for spectroflat: {dirty_flat.shape} [state, spatial, wavelength]')
    
    # Define ROI (avoiding edges)
    roi = parse_shape(f'[2:{dirty_flat.shape[1]-2},2:{dirty_flat.shape[2]-2}]')
    
    # Configure spectroflat
    sf_config = SpectroflatConfig(roi=roi, iterations=2)
    sf_config.sensor_flat = SensorFlatConfig(
        spacial_degree=4,
        sigma_mask=4.5,
        fit_border=1,
        average_column_response_map=False,
        ignore_gradient=True,
        roi=roi
    )
    sf_config.smile = SmileConfig(
        line_distance=11,
        strong_smile_deg=8,
        max_dispersion_deg=4,
        line_prominence=0.1,
        height_sigma=0.04,
        smooth=True,
        emission_spectrum=False,
        state_aware=False,
        align_states=True,
        smile_deg=3,
        rotation_correction=0,
        detrend=False,
        roi=roi
    )
    
    # Run spectroflat analysis with report
    report_dir = Path(config.directories.figures) / 'spectroflat_report'
    report_dir.mkdir(exist_ok=True, parents=True)
    print('  Running spectroflat analysis...')
    analyser = Analyser(dirty_flat, sf_config, report_dir)
    analyser.run()
    
    print(f'  Spectroflat analysis complete.')
    
    # Create L1 FramesSet with dust_flat
    reduced_frames = dct.FramesSet()
    
    # The analyser.dust_flat has shape [state, spatial, wavelength]
    # Extract upper (state 0) and lower (state 1)
    dust_flat_upper = analyser.dust_flat[0]  # shape: (spatial, wavelength)
    dust_flat_lower = analyser.dust_flat[1]  # shape: (spatial, wavelength)
    
    # Create a single L1 frame with upper and lower halves
    frame_name_str = f"{data_type}_l1_frame{0:04d}"
    l1_frame = dct.Frame(frame_name_str)
    l1_frame.set_half("upper", dust_flat_upper.astype('float32'))
    l1_frame.set_half("lower", dust_flat_lower.astype('float32'))
    
    # Add frame to FramesSet
    reduced_frames.add_frame(l1_frame, frame_idx=0)
    
    # Also save offset_map and illumination_pattern as separate FITS files for later use
    line = config.dataset['line']
    seq = config.dataset[data_type]['sequence']
    seq_str = f"t{seq:03d}"
    
    # Save offset map
    offset_map_path = Path(config.directories.reduced) / f'{line}_{data_type}_{seq_str}_offset_map.fits'
    analyser.offset_map.dump(str(offset_map_path))
    print(f'  Offset map saved to: {offset_map_path}')
    
    # Save illumination pattern (soft flat)
    illumination_path = Path(config.directories.reduced) / f'{line}_{data_type}_{seq_str}_illumination_pattern.fits'
    from astropy.io import fits
    hdu = fits.PrimaryHDU(data=analyser.illumination_pattern)
    hdu.header['COMMENT'] = 'Illumination pattern (soft flat) from spectroflat'
    hdu.writeto(str(illumination_path), overwrite=True)
    print(f'  Illumination pattern saved to: {illumination_path}')
    
    # Add to auxiliary files
    file_set = config.dataset[data_type]['files']
    if not hasattr(file_set, 'auxiliary'):
        file_set.auxiliary = {}
    file_set.auxiliary['offset_map'] = offset_map_path
    file_set.auxiliary['illumination_pattern'] = illumination_path
    print(f'  ✓ Added offset map and illumination pattern to FileSet auxiliary entry')
    
    return reduced_frames


def amend_spectroflat_with_atlas_lines(config, data_type, atlas_lines_files):
    """
    Amend spectroflat results with atlas lines wavelength calibration.
    
    This function runs the external amend_spectroflat command to apply
    wavelength calibration from atlas lines to the offset map.
    
    Parameters
    ----------
    config : Config
        Configuration object containing dataset and directory information
    data_type : str
        Type of data being processed ('flat_center' or 'flat')
    atlas_lines_files : dict
        Dictionary mapping frame names to atlas lines YAML files
    
    Returns
    -------
    Path
        Path to the updated L1 FITS file
    """
    import re
    from pathlib import Path
    
    print(f'\n{"="*70}')
    print('STEP 3: AMEND SPECTROFLAT WITH ATLAS LINES')
    print(f'{"="*70}')
    
    # Check if spectroflat config exists
    if not hasattr(config.cam, 'atlas_fit_config') or config.cam.atlas_fit_config is None:
        print('Error: No atlas_fit_config found for this camera.')
        return None
    
    config_path = config.cam.atlas_fit_config
    
    # Get the L1 file path
    l1_file = config.dataset[data_type]['files'].get('l1')
    if not l1_file or not l1_file.exists():
        print(f'Error: L1 {data_type} file not found.')
        return None
    
    # Get the offset map path from auxiliary files
    offset_map_file = config.dataset[data_type]['files'].auxiliary.get('offset_map')
    if not offset_map_file or not offset_map_file.exists():
        print(f'Error: Offset map file not found in auxiliary files.')
        print('  Run spectroflat step first to generate offset map.')
        return None
    
    print(f'  Using offset map: {offset_map_file.name}')
    
    # Check if outdated files exist (re-running amend step)
    # These are auto-discovered by _build_file_set in themis_datasets_2025.py
    outdated_offset = config.dataset[data_type]['files'].auxiliary.get('offset_map_outdated')
    outdated_illumination = config.dataset[data_type]['files'].auxiliary.get('illumination_pattern_outdated')
    illumination_pattern_file = config.dataset[data_type]['files'].auxiliary.get('illumination_pattern')
    
    if outdated_offset or outdated_illumination:
        print(f'\n{"="*70}')
        print(f'WAVELENGTH-CALIBRATED FILES ALREADY EXIST')
        print(f'{"="*70}')
        print(f'The offset map and illumination pattern have already been amended')
        print(f'with wavelength calibration from atlas lines.')
        print(f'\nCurrent auxiliary files:')
        print(f'  offset_map: {offset_map_file.name} (wavelength calibrated)')
        if illumination_pattern_file:
            print(f'  illumination_pattern: {illumination_pattern_file.name} (wavelength calibrated)')
        print(f'\nOriginal spectroflat outputs (backed up):')
        if outdated_offset:
            print(f'  offset_map_outdated: {outdated_offset.name}')
        if outdated_illumination:
            print(f'  illumination_pattern_outdated: {outdated_illumination.name}')
        
        # Ask user if they want to re-run
        print(f'\n{"="*70}')
        while True:
            user_choice = input(f'Re-run amend_spectroflat with new atlas lines? [y/n]: ').strip().lower()
            if user_choice in ['y', 'n']:
                break
            else:
                print('Please enter "y" to re-run or "n" to skip')
        
        if user_choice == 'n':
            print(f'Skipping amend step. Using existing wavelength-calibrated files.')
            return l1_file
        
        # User chose to re-run: restore original spectroflat outputs
        print(f'\n  Restoring original spectroflat outputs for re-amending...')
        
        # Delete current wavelength-calibrated offset_map
        if offset_map_file and offset_map_file.exists():
            offset_map_file.unlink()
            print(f'    ✓ Deleted wavelength-calibrated: {offset_map_file.name}')
        
        # Move offset_map_outdated → offset_map
        if outdated_offset and outdated_offset.exists():
            outdated_offset.rename(offset_map_file)
            del config.dataset[data_type]['files'].auxiliary['offset_map_outdated']
            print(f'    ✓ Restored original offset map: {offset_map_file.name}')
        
        # Delete current wavelength-calibrated illumination_pattern
        if illumination_pattern_file and illumination_pattern_file.exists():
            illumination_pattern_file.unlink()
            print(f'    ✓ Deleted wavelength-calibrated: {illumination_pattern_file.name}')
        
        # Move illumination_pattern_outdated → illumination_pattern
        if outdated_illumination and outdated_illumination.exists():
            outdated_illumination.rename(illumination_pattern_file)
            del config.dataset[data_type]['files'].auxiliary['illumination_pattern_outdated']
            print(f'    ✓ Restored original illumination pattern: {illumination_pattern_file.name}')
        
        print(f'  Now proceeding with fresh amend step on original spectroflat outputs.\n')
    
    # Create temporary stacked FITS file for amend_spectroflat (like the test does)
    # amend_spectroflat needs a simple 4D array [state, spatial, wavelength]
    import numpy as np
    from astropy.io import fits
    
    print(f'  Creating temporary stacked FITS file for amend_spectroflat...')
    
    # Read L0 data (same as test - we need the original data, not the L1 which has spectroflat corrections)
    l0_data, l0_header = tio.read_any_file(config, data_type, verbose=False, status='l0')
    
    if len(l0_data) == 0:
        print(f'Error: No frames found in L0 data')
        return None
    
    # Extract upper and lower frames
    upper_frame_data = l0_data[0]['upper'].data
    lower_frame_data = l0_data[0]['lower'].data
    
    # Stack as 4 states to match illumination_pattern from spectroflat
    stacked_data = np.stack([upper_frame_data, lower_frame_data, lower_frame_data, lower_frame_data], axis=0)
    
    # Create temporary FITS file
    temp_fits_path = config.directories.reduced / f'temp_{data_type}_stacked_for_amend.fits'
    hdu = fits.PrimaryHDU(data=stacked_data)
    hdu.writeto(temp_fits_path, overwrite=True)
    
    print(f'  ✓ Created temporary FITS: {temp_fits_path.name}')
    print(f'    Shape: {stacked_data.shape} [state=4, spatial, wavelength]')
    
    # Check illumination pattern file exists (already retrieved earlier)
    if not illumination_pattern_file or not illumination_pattern_file.exists():
        print(f'Warning: Illumination pattern file not found in auxiliary files.')
        illumination_pattern_file = None
    
    # Read the original config as text to preserve formatting and comments
    with open(config_path, 'r') as f:
        original_config_text = f.read()
    
    temp_config_text = original_config_text
    
    # Update corrected_frame to point to the temporary stacked FITS
    temp_config_text = re.sub(
        r'corrected_frame:\s*(.+)',
        f'corrected_frame: {temp_fits_path}',
        temp_config_text
    )
    
    # Update ROI to include state dimension for 3D array: [s, spatial, wavelength]
    # The 's' placeholder gets replaced with state numbers by amend_spectroflat
    temp_config_text = re.sub(
        r'roi:\s*"?\[([^\]]+)\]"?',
        r'roi: "[s, \1]"',
        temp_config_text
    )
    
    # Add or update mod_states parameter (spectroflat uses 4 states)
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
    
    # Add offset_map (same simple approach as test)
    temp_config_text = re.sub(
        r'(mod_states:\s*\d+)',
        rf'\1\n  offset_map: {offset_map_file}',
        temp_config_text
    )
    
    # Add soft_flat after offset_map
    if illumination_pattern_file:
        temp_config_text = re.sub(
            r'(offset_map:\s*.+)',
            rf'\1\n  soft_flat: {illumination_pattern_file}',
            temp_config_text
        )
    
    # Create temporary config file (don't modify original)
    project_root = Path(__file__).resolve().parents[3]
    temp_config_path = project_root / 'configs' / f'temp_amend_{data_type}_config.yml'
    with open(temp_config_path, 'w') as f:
        f.write(temp_config_text)
    
    print(f'  ✓ Created temporary config: {temp_config_path}')
    print(f'    Original config file remains unchanged.')
    
    # Run amend_spectroflat once with upper atlas lines (state 0)
    # The rest is calculated relatively
    if 'upper' not in atlas_lines_files:
        print(f'Error: No atlas lines file for upper frame')
        # Clean up temp files
        if temp_config_path.exists():
            temp_config_path.unlink()
        if temp_fits_path.exists():
            temp_fits_path.unlink()
        return None
    
    upper_lines_file = atlas_lines_files['upper']
    
    print(f'\nRunning amend_spectroflat with UPPER frame atlas lines (state 0)...')
    print(f'  Config: {temp_config_path.name}')
    print(f'  Lines file: {upper_lines_file.name}')
    print(f'  Note: Lower frame (state 1) corrections are computed relatively')
    
    # Find amend_spectroflat script
    amend_script = project_root / 'atlas-fit' / 'bin' / 'amend_spectroflat'
    
    if not amend_script.exists():
        print(f'Error: amend_spectroflat script not found: {amend_script}')
        print('  This script should be available in the atlas-fit repository')
        # Clean up temp files
        if temp_config_path.exists():
            temp_config_path.unlink()
        if temp_fits_path.exists():
            temp_fits_path.unlink()
        return None
    
    # Display command to run in external terminal
    print(f'\nPlease run the following command in an EXTERNAL terminal:')
    print(f'{"-"*70}')
    print(f'cd {config.directories.reduced}')
    print(f'{amend_script} {temp_config_path} {upper_lines_file}')
    print(f'{"-"*70}')
    print('\nThis will update the offset map with wavelength calibration.')
    print('Both upper and lower frame corrections will be computed.')
    
    # Wait for user confirmation
    while True:
        user_input = input(f'\nDid amend_spectroflat complete successfully? (y/n): ').strip().lower()
        if user_input == 'y':
            print(f'✓ Spectroflat amendment completed successfully')
            break
        elif user_input == 'n':
            print(f'✗ Skipping spectroflat amendment')
            # Clean up temp files
            if temp_config_path.exists():
                temp_config_path.unlink()
            if temp_fits_path.exists():
                temp_fits_path.unlink()
            return None
        else:
            print('Please enter "y" or "n"')
    
    # Process amended files: rename and update auxiliary file tracking
    print(f'\n  Processing amended spectroflat outputs...')
    
    reduced_dir = config.directories.reduced
    
    # Handle amended_soft_flat.fits -> rename to illumination_pattern pattern
    amended_soft_flat = reduced_dir / 'amended_soft_flat.fits'
    if amended_soft_flat.exists():
        # Backup old illumination_pattern as _outdated (original spectroflat output)
        if illumination_pattern_file and illumination_pattern_file.exists():
            outdated_illumination = illumination_pattern_file.parent / f'{illumination_pattern_file.stem}_outdated{illumination_pattern_file.suffix}'
            # The outdated file should not exist at this point (we restored it earlier if it did)
            illumination_pattern_file.rename(outdated_illumination)
            config.dataset[data_type]['files'].auxiliary['illumination_pattern_outdated'] = outdated_illumination
            print(f'    ✓ Backed up original illumination pattern: {outdated_illumination.name}')
        
        # Rename amended_soft_flat to illumination_pattern filename
        new_illumination = illumination_pattern_file  # Use original filename
        amended_soft_flat.rename(new_illumination)
        config.dataset[data_type]['files'].auxiliary['illumination_pattern'] = new_illumination
        print(f'    ✓ Updated illumination pattern (wavelength calibrated): {new_illumination.name}')
    else:
        print(f'    Warning: amended_soft_flat.fits not found')
    
    # Handle wl_calibrated_offsets.fits -> rename to offset_map pattern
    wl_calibrated_offsets = reduced_dir / 'wl_calibrated_offsets.fits'
    if wl_calibrated_offsets.exists():
        # Backup old offset_map as _outdated (original spectroflat output)
        if offset_map_file and offset_map_file.exists():
            outdated_offset = offset_map_file.parent / f'{offset_map_file.stem}_outdated{offset_map_file.suffix}'
            # The outdated file should not exist at this point (we restored it earlier if it did)
            offset_map_file.rename(outdated_offset)
            config.dataset[data_type]['files'].auxiliary['offset_map_outdated'] = outdated_offset
            print(f'    ✓ Backed up original offset map: {outdated_offset.name}')
        
        # Rename wl_calibrated_offsets to offset_map filename
        new_offset_map = offset_map_file  # Use original filename
        wl_calibrated_offsets.rename(new_offset_map)
        config.dataset[data_type]['files'].auxiliary['offset_map'] = new_offset_map
        print(f'    ✓ Updated offset map (wavelength calibrated): {new_offset_map.name}')
    else:
        print(f'    Warning: wl_calibrated_offsets.fits not found')
    
    # Clean up temporary files
    if temp_config_path.exists():
        temp_config_path.unlink()
    if temp_fits_path.exists():
        temp_fits_path.unlink()
    print(f'\n✓ Cleaned up temporary files')
    
    print(f'\n✓ Spectroflat amendment complete for {data_type}')
    print(f'  L1 file: {l1_file}')
    
    return l1_file


def reduce_raw(config):
    print('No processing for reduction level raw')
    return None

def load_dark(config, auto_reduce_dark: bool = False):
    """Load L0 dark frame, optionally auto-reducing if not found.
    
    Args:
        config: Configuration object
        auto_reduce_dark: If True, automatically reduce dark to L0 if not found
        
    Returns:
        tuple: (dark_frame, header_dark) or None if failed
        
    Raises:
        Prints error message and returns None if dark cannot be loaded
    """
    try:
        dark_frame, header_dark = tio.read_any_file(config, 'dark', verbose=False, status='l0')
        return dark_frame, header_dark
    except FileNotFoundError as e:
        if auto_reduce_dark:
            print("LV0 dark not found. Auto-reducing 'dark' to LV0...")
            # Trigger dark reduction and try again
            out_path = reduce_raw_to_l0(config, data_type='dark', return_reduced=False)
            if out_path is None:
                # Upstream failed gracefully
                print("Automatic dark reduction did not produce an output. Aborting reduction.")
                return None
            # Retry reading LV0 dark
            dark_frame, header_dark = tio.read_any_file(config, 'dark', verbose=False, status='l0')
            return dark_frame, header_dark
        else:
            print(
                "LV0 dark file is required for reduction but was not found. "
                "Run LV0 dark reduction first or pass auto_reduce_dark=True.\n"
                f"Reason: {e}"
            )
            return None

def reduce_raw_to_l0(config, data_type=None, return_reduced=False, auto_reduce_dark: bool = False):
    # No processing
    
    if data_type==None:
        print('No processing - provide a specific data type.')
        return None
    
    else:
      data, header = tio.read_any_file(config, data_type, verbose=False, status='raw')
      data, bad_pixel_keywords = tt.clean_bad_pixels(data, header)
      if data_type == 'dark':
         
        upper = data.stack_all('upper') # should always return one extra dimension that we can "average"
        lower = data.stack_all('lower') # should always return one extra dimension that we can "average"
        
        reduced_frames = dct.FramesSet()
        
        frame_name_str = f"{data_type}_l0_frame{0:04d}"
        single_frame = dct.Frame(frame_name_str)
        # Explicitly convert to float32 for consistency
        single_frame.set_half("upper", upper.mean(axis=0).astype('float32')) 
        single_frame.set_half("lower", lower.mean(axis=0).astype('float32'))  
        reduced_frames.add_frame(single_frame, 0)
        
        frame_name_str = f"{data_type}_l0_frame{1:04d}"
        single_frame = dct.Frame(frame_name_str)
        # z3denoise already returns float32, but be explicit
        single_frame.set_half("upper", tt.z3denoise(upper.mean(axis=0).astype('float32')) )
        single_frame.set_half("lower", tt.z3denoise(lower.mean(axis=0).astype('float32'))   )
        reduced_frames.add_frame(single_frame, 1)
        
        # Store number of averaged frames
        n_frames_averaged = upper.shape[0]
        
      elif data_type == 'flat':
        upper = data.stack_all('upper') # should always return one extra dimension that we can "average"
        lower = data.stack_all('lower') # should always return one extra dimension that we can "average"
        
        n_frames = upper.shape[0]
        
        # Load L0 dark frame
        result = load_dark(config, auto_reduce_dark)
        if result is None:
            return None
        dark_frame, header_dark = result
          
        reduced_frames = dct.FramesSet()
          
        frame_name_str = f"{data_type}_l0_frame{0:04d}"
        single_frame = dct.Frame(frame_name_str)
        # Convert to float32 to match dark data type and avoid precision issues
        upper_mean = upper.mean(axis=0).astype('float32')
        lower_mean = lower.mean(axis=0).astype('float32')
        single_frame.set_half("upper", upper_mean - dark_frame[0]['upper'].data) 
        single_frame.set_half("lower", lower_mean - dark_frame[0]['lower'].data)  
        reduced_frames.add_frame(single_frame, 0)
        
        # Store number of averaged frames for later use
        n_frames_averaged = n_frames
        
      elif data_type == 'flat_center':
        upper = data.stack_all('upper') # should always return one extra dimension that we can "average"
        lower = data.stack_all('lower') # should always return one extra dimension that we can "average"
        
        n_frames = upper.shape[0]
        
        # Load L0 dark frame
        result = load_dark(config, auto_reduce_dark)
        if result is None:
            return None
        dark_frame, header_dark = result
          
        reduced_frames = dct.FramesSet()
          
        frame_name_str = f"{data_type}_l0_frame{0:04d}"
        single_frame = dct.Frame(frame_name_str)
        # Convert to float32 to match dark data type and avoid precision issues
        upper_mean = upper.mean(axis=0).astype('float32')
        lower_mean = lower.mean(axis=0).astype('float32')
        single_frame.set_half("upper", upper_mean - dark_frame[0]['upper'].data) 
        single_frame.set_half("lower", lower_mean - dark_frame[0]['lower'].data)  
        reduced_frames.add_frame(single_frame, 0)
        
        # Store number of averaged frames for later use
        n_frames_averaged = n_frames
        
      elif data_type == 'scan':
        # Scan: subtract dark from all frames without averaging
        # data is a CycleSet with keys (frame_state, slit_idx, map_idx)
        
        # Load L0 dark frame
        result = load_dark(config, auto_reduce_dark)
        if result is None:
            return None
        dark_frame, header_dark = result
        
        reduced_frames = dct.CycleSet()
        
        # Iterate through all frames in the CycleSet and subtract dark
        for key, frame in data.items():
            # key is (frame_state, slit_idx, map_idx)
            frame_name_str = f"{data_type}_l0_{frame.name}"
            single_frame = dct.Frame(frame_name_str)
            
            # Subtract dark from upper and lower halves
            upper_data = frame.get_half('upper').data.astype('float32')
            lower_data = frame.get_half('lower').data.astype('float32')
            
            upper_dark_subtracted = upper_data - dark_frame[0]['upper'].data
            lower_dark_subtracted = lower_data - dark_frame[0]['lower'].data
            
            # Preserve polarization state of each half if present
            upper_half = frame.get_half('upper')
            lower_half = frame.get_half('lower')
            
            if upper_half.pol_state:
                single_frame.set_half("upper", upper_dark_subtracted, upper_half.pol_state)
            else:
                single_frame.set_half("upper", upper_dark_subtracted)
                
            if lower_half.pol_state:
                single_frame.set_half("lower", lower_dark_subtracted, lower_half.pol_state)
            else:
                single_frame.set_half("lower", lower_dark_subtracted)
            
            reduced_frames.add_frame(single_frame, key)
     
      else:
            print('Unknown data_type.')
            return None
    if return_reduced:
            return reduced_frames
        
    else:
            # Prepare additional header keywords
            extra_keywords = {}
            if 'n_frames_averaged' in locals():
                extra_keywords['NFRAMAVG'] = (n_frames_averaged, 'Number of averaged raw frames')
            
            # Merge bad pixel keywords
            extra_keywords.update(bad_pixel_keywords)
            
            out_path = tio.save_reduction(
                config,
                data_type=data_type,
                level='l0',
                frames=reduced_frames,
                source_header=header,
                verbose=True,
                overwrite=True,  # set True if you want to allow replacing an existing file
                extra_keywords=extra_keywords,
                )
            return out_path

def reduce_l0_to_l1(config, data_type=None, return_reduced=False, auto_reduce_dark: bool = False):
    """
    Reduce L0 data to L1 level.
    
    For flat and flat_center: performs wavelength calibration using atlas-fit bin/prepare.
    """
    
    if data_type is None:
        print('No processing - provide a specific data type.')
        return None
    
    if data_type == 'dark':
        print('No reduction procedure defined for dark l0->l1.')
        return None
    
    elif data_type == 'scan':
        print('Scan l0->l1 not yet implemented.')
        return None
    
    elif data_type in ['flat', 'flat_center']:
        # Step 1: Process wavelength calibration using atlas-fit
        atlas_lines_files = process_wavelength_calibration_with_atlas_fit(config, data_type)
        
        if not atlas_lines_files:
            print(f'\nL1 reduction failed for {data_type} (atlas-fit step).')
            return None
        
        print(f'\n✓ Generated atlas lines files for {len(atlas_lines_files)} frames.')
        
        # Check if L1 file already exists
        l1_file = config.dataset[data_type]['files'].get('l1')
        if l1_file and l1_file.exists():
            print(f'\nL1 file already exists: {l1_file.name}')
            while True:
                user_choice = input(f'Re-run spectroflat? [y/n]: ').strip().lower()
                if user_choice in ['y', 'n']:
                    break
                else:
                    print('Please enter "y" to re-run or "n" to skip')
            
            if user_choice == 'n':
                print('Skipping spectroflat step.')
                # Still proceed to amend step if atlas lines exist
                return amend_spectroflat_with_atlas_lines(config, data_type, atlas_lines_files)
        
        # Step 2: Process with spectroflat to get corrected flat field
        reduced_frames = process_spectroflat(config, data_type)
        
        if reduced_frames is None:
            print(f'\nL1 reduction failed for {data_type} (spectroflat step).')
            return None
        
        print(f'✓ Generated dust flat for {len(reduced_frames)} frames.')
        
        # If return_reduced is True, return the FramesSet
        if return_reduced:
            return reduced_frames
        
        # Otherwise, save to L1 FITS file
        # Read the L0 header for metadata
        _, header = tio.read_any_file(config, data_type, verbose=False, status='l0')
        
        # Prepare extra keywords for FITS header
        seq = config.dataset[data_type]['sequence']
        seq_str = f"t{seq:03d}"
        extra_keywords = {
            'ATLSUPR': (str(atlas_lines_files.get('upper', '')), 'Atlas lines file for upper frame'),
            'ATLSLWR': (str(atlas_lines_files.get('lower', '')), 'Atlas lines file for lower frame'),
            'SPCTRFLT': ('TRUE', 'Spectroflat correction applied'),
            'OFSTMAP': (f'{config.dataset["line"]}_{data_type}_{seq_str}_offset_map.fits', 'Offset map file')
        }
        
        # Save L1 FITS file
        out_path = tio.save_reduction(
            config,
            data_type=data_type,
            level='l1',
            frames=reduced_frames,
            source_header=header,
            verbose=True,
            overwrite=True,
            extra_keywords=extra_keywords,
        )
        
        print(f'✓ L1 file saved to: {out_path}')
        
        # Step 3: Amend spectroflat with atlas lines
        return amend_spectroflat_with_atlas_lines(config, data_type, atlas_lines_files)
    
    else:
        print(f'Unknown data_type: {data_type}')
        return None



reduction_levels = ReductionRegistry()

reduction_levels.add(ReductionLevel("raw", "fts", reduce_raw, {
    "dark": "Nothing.",
    "scan": "Nothing.",
    "flat": "Nothing.",
    "flat_center": "Nothing."
}))
reduction_levels.add(ReductionLevel("l0", "_l0.fits", reduce_raw_to_l0, {
    "dark": ("Splitting original camera image into upper/lower frames. Flip x axis (blue is left). "
             "Clean bad pixels (interpolate isolated, reject frames with clustered bad pixels). "
             "Averaging raw frames."),
    "scan": ("Splitting original camera image into upper/lower frames. Flip x axis (blue is left). "
             "Clean bad pixels (interpolate isolated, reject frames with clustered bad pixels). "
             "Subtract l0 dark from all frames (no averaging)."),
    "flat": ("Splitting original camera image into upper/lower frames. Flip x axis (blue is left). "
             "Clean bad pixels (interpolate isolated, reject frames with clustered bad pixels). "
             "Averaging raw frames. Subtract l0 dark."),
    "flat_center": ("Splitting original camera image into upper/lower frames. Flip x axis (blue is left). "
             "Clean bad pixels (interpolate isolated, reject frames with clustered bad pixels). "
             "Averaging raw frames. Subtract l0 dark.")
}))

reduction_levels.add(ReductionLevel("l1", "_l1.fits", reduce_l0_to_l1, {
    "dark": "Nothing.",
    "scan": "Nothing at the moment.",
    "flat": "Wavelength calibration with atlas-fit (generates atlas_lines.yaml per frame). Flat-field correction with spectroflat (generates dust_flat and offset_map). Saves L1 FITS with corrected frames.",
    "flat_center": "Wavelength calibration with atlas-fit (generates atlas_lines.yaml per frame). Flat-field correction with spectroflat (generates dust_flat and offset_map). Saves L1 FITS with corrected frames."
}))

