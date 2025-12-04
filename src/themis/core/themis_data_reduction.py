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
from spectroflat import Analyser, Config as SpectroflatConfig, OffsetMap, SmileConfig, SensorFlatConfig
from qollib.strings import parse_shape
import numpy as np
from pathlib import Path
import shutil
import re
from astropy.io import fits

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


def _process_single_frame_spectroflat(frame_2d, frame_name, config, data_type):
    """
    Helper function to process a single frame (upper or lower) with spectroflat.
    
    Parameters
    ----------
    frame_2d : ndarray
        2D array of frame data (spatial, wavelength)
    frame_name : str
        'upper' or 'lower'
    config : Config
        Configuration object
    data_type : str
        Type of data being processed
        
    Returns
    -------
    dict
        Dictionary with keys: 'dust_flat', 'offset_map', 'illumination_pattern', 'analyser'
    """
    
    print(f"  Processing {frame_name.upper()} frame...")
    
    # Create 4 states by duplicating the frame
    dirty_flat = np.stack([frame_2d, frame_2d, frame_2d, frame_2d], axis=0)
    print(f"    Input shape: {dirty_flat.shape} [state, spatial, wavelength]")
    
    # Define ROI
    roi = parse_shape(f'[20:{dirty_flat.shape[1]-20},20:{dirty_flat.shape[2]-20}]')
    
    # Configure spectroflat
    sf_config = SpectroflatConfig(roi=roi, iterations=2) # iterations 2 is considered enough
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
        align_states=False,
        smile_deg=3,
        rotation_correction=0,
        detrend=False,
        roi=roi
    )
    
    # Create report directory
    report_dir = Path(config.directories.figures) / 'spectroflat_report' / frame_name
    report_dir.mkdir(exist_ok=True, parents=True)
    
    # Run spectroflat
    analyser = Analyser(dirty_flat, sf_config, str(report_dir))
    analyser.run()
    print(f"    ✓ {frame_name.capitalize()} analysis complete")
    
    # Extract results (state 0 since all 4 states are identical)
    return {
        'dust_flat': analyser.dust_flat[0],
        'offset_map': analyser.offset_map.get_map()[0],
        'illumination_pattern': analyser.illumination_pattern[0],
        'analyser': analyser
    }


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


def _process_single_frame_atlas_fit(config, data_type, frame_name):
    """
    Helper function to run atlas-fit for a single frame (upper or lower).
    
    Parameters
    ----------
    config : Config
        Configuration object
    data_type : str
        'flat' or 'flat_center'
    frame_name : str
        'upper' or 'lower'
        
    Returns
    -------
    Path
        Path to the generated atlas_lines YAML file, or None on failure
    """
    import re
    from astropy.io import fits
    
    print(f'  Running atlas-fit for {frame_name} frame...')
    
    # Check if atlas-fit config is available
    if not hasattr(config.cam, 'atlas_fit_config') or config.cam.atlas_fit_config is None:
        print('  Error: No atlas_fit_config found for this camera.')
        return None
    
    atlas_config_path = config.cam.atlas_fit_config
    if not Path(atlas_config_path).exists():
        print(f'  Error: Atlas fit config file not found: {atlas_config_path}')
        return None
    
    # Get the L0 file path
    l0_file = config.dataset[data_type]['files'].get('l0')
    if l0_file is None or not l0_file.exists():
        print(f'  Error: L0 {data_type} file not found.')
        return None
    
    # Find atlas-fit prepare script
    project_root = Path(__file__).resolve().parents[3]
    prepare_script = project_root / 'atlas-fit' / 'bin' / 'prepare'
    if not prepare_script.exists():
        print(f'  Error: Atlas-fit prepare script not found: {prepare_script}')
        return None
    
    # Extract individual frame to temporary file
    temp_frame_path = config.directories.reduced / f'temp_{data_type}_{frame_name}_for_atlas.fits'
    
    try:
        # Read the data using themis_io
        data, header = tio.read_any_file(config, data_type, verbose=False, status='l0')
        
        if len(data) > 0:
            if frame_name == 'upper':
                frame_data = data[0]['upper'].data
            elif frame_name == 'lower':
                frame_data = data[0]['lower'].data
            else:
                print(f'  Unknown frame name: {frame_name}')
                return None
            
            # Create a simple FITS file with the 2D frame data
            hdu = fits.PrimaryHDU(data=frame_data)
            hdu.writeto(temp_frame_path, overwrite=True)
            print(f'  ✓ Extracted {frame_name} frame to temporary file')
        else:
            print(f'  Error: No frames found in {data_type} data')
            return None
    except Exception as e:
        print(f'  Error extracting {frame_name} frame: {e}')
        return None
    
    # Read and modify config
    with open(atlas_config_path, 'r') as f:
        original_config_text = f.read()
    
    modified_config_text = re.sub(
        r'corrected_frame:\s*(.+)',
        f'corrected_frame: {temp_frame_path}',
        original_config_text
    )
    
    # For lower frame: use stray_light_lower if available
    if frame_name == 'lower':
        stray_light_lower_match = re.search(r'stray_light_lower:\s*([\d.]+)', modified_config_text)
        if stray_light_lower_match:
            stray_light_lower_value = stray_light_lower_match.group(1)
            # Replace stray_light value with stray_light_lower value
            modified_config_text = re.sub(
                r'stray_light:\s*[\d.]+',
                f'stray_light: {stray_light_lower_value}',
                modified_config_text
            )
            print(f'  ✓ Using stray_light_lower: {stray_light_lower_value}% for lower frame')
    
    # Remove stray_light_lower entry (atlas-fit doesn't recognize it)
    modified_config_text = re.sub(
        r'^\s*#.*stray.light.*lower.*\n',  # Remove comment line
        '',
        modified_config_text,
        flags=re.MULTILINE | re.IGNORECASE
    )
    modified_config_text = re.sub(
        r'^\s*stray_light_lower:\s*[\d.]+\s*\n',  # Remove the actual entry
        '',
        modified_config_text,
        flags=re.MULTILINE
    )
    
    # Determine output filename
    line = config.dataset['line']
    seq = config.dataset[data_type]['sequence']
    base_filename = f'{line}_{data_type}_t{seq:03d}'
    lines_file = config.directories.reduced / f'{base_filename}_{frame_name}_atlas_lines.yaml'
    
    # Create temporary config file
    temp_config_path = project_root / 'configs' / f'temp_atlas_fit_{data_type}_{frame_name}_config.yml'
    with open(temp_config_path, 'w') as f:
        f.write(modified_config_text)
    
    print(f'  ✓ Created temporary config: {temp_config_path.name}')
    
    # The prepare script outputs to 'atlas_fit_lines.yaml' in the working directory
    temp_output_file = config.directories.reduced / 'atlas_fit_lines.yaml'
    
    # Display command to run
    print(f'\n  Please run the following command in an EXTERNAL terminal:')
    print(f'  {"-"*68}')
    print(f'  cd {config.directories.reduced}')
    print(f'  {prepare_script} {temp_config_path}')
    print(f'  {"-"*68}')
    print(f'  Note: Output will be saved as atlas_fit_lines.yaml and renamed automatically.')
    if frame_name == 'upper':
        print(f'\n  IMPORTANT: After atlas-fit completes, note the fitted stray-light value')
        print(f'  and FWHM displayed in the output (e.g., "stray-light: X.XX %") and update')
        print(f'  "stray_light" and "fwhm" in the config file before running amend_spectroflat:')
        print(f'  {project_root / "configs" / "atlas_fit_config_cam1.yml"}')
    elif frame_name == 'lower':
        print(f'\n  IMPORTANT: After atlas-fit completes, note the fitted stray-light value')
        print(f'  and update "stray_light_lower" in the config file (FWHM is shared):')
        print(f'  {project_root / "configs" / "atlas_fit_config_cam1.yml"}')
    
    # Wait for user confirmation
    while True:
        user_input = input(f'\n  Did atlas-fit complete successfully for {frame_name}? (y/n): ').strip().lower()
        if user_input == 'y':
            print(f'  ✓ Atlas-fit completed for {frame_name}')
            break
        elif user_input == 'n':
            print(f'  ✗ Skipping atlas-fit for {frame_name}')
            # Clean up
            if temp_config_path.exists():
                temp_config_path.unlink()
            if temp_frame_path.exists():
                temp_frame_path.unlink()
            # Clean up any generated atlas_fit_lines.yaml
            if temp_output_file.exists():
                temp_output_file.unlink()
            return None
        else:
            print('  Please enter "y" or "n"')
    
    # Clean up temporary files
    if temp_config_path.exists():
        temp_config_path.unlink()
    if temp_frame_path.exists():
        temp_frame_path.unlink()
    
    # Rename the output file to the correct name
    if temp_output_file.exists():
        # Delete old file if it exists (shouldn't, but just in case)
        if lines_file.exists():
            lines_file.unlink()
        temp_output_file.rename(lines_file)
        print(f'  ✓ Renamed atlas_fit_lines.yaml → {lines_file.name}')
    else:
        print(f'  Error: Output file not found: {temp_output_file}')
        return None
    
    # Verify output file exists
    if not lines_file.exists():
        print(f'  Error: Output file not found: {lines_file}')
        return None
    
    # Store in auxiliary files
    file_set = config.dataset[data_type]['files']
    if not hasattr(file_set, 'auxiliary'):
        file_set.auxiliary = {}
    file_set.auxiliary[f'atlas_lines_{frame_name}'] = lines_file
    
    return lines_file


def _process_single_frame_spectroflat_wrapper(config, data_type, frame_name):
    """
    Wrapper to run spectroflat for a single frame and save outputs.
    
    Parameters
    ----------
    config : Config
        Configuration object
    data_type : str
        'flat' or 'flat_center'
    frame_name : str
        'upper' or 'lower'
        
    Returns
    -------
    ndarray
        The dust_flat for this frame, or None on failure
    """
    print(f'  Running spectroflat for {frame_name} frame...')
    
    # Read L0 data
    data, header = tio.read_any_file(config, data_type, verbose=False, status='l0')
    
    if frame_name == 'upper':
        frame_2d = data[0]['upper'].data
    else:
        frame_2d = data[0]['lower'].data
    
    # Run spectroflat using existing helper
    result = _process_single_frame_spectroflat(frame_2d, frame_name, config, data_type)
    
    if not result:
        return None
    
    # Save offset map and illumination pattern
    line = config.dataset['line']
    seq = config.dataset[data_type]['sequence']
    seq_str = f"t{seq:03d}"
    
    file_set = config.dataset[data_type]['files']
    if not hasattr(file_set, 'auxiliary'):
        file_set.auxiliary = {}
    
    # Save offset map
    offset_map_path = Path(config.directories.reduced) / f'{line}_{data_type}_{seq_str}_offset_map_{frame_name}.fits'
    result['analyser'].offset_map.dump(str(offset_map_path))
    file_set.auxiliary[f'offset_map_{frame_name}'] = offset_map_path
    print(f'  ✓ Saved offset map: {offset_map_path.name}')
    
    # Save illumination pattern
    illumination_path = Path(config.directories.reduced) / f'{line}_{data_type}_{seq_str}_illumination_pattern_{frame_name}.fits'
    hdu = fits.PrimaryHDU(data=result['analyser'].illumination_pattern)
    hdu.header['COMMENT'] = f'Illumination pattern from spectroflat - {frame_name} frame'
    hdu.header['FRAME'] = frame_name
    hdu.writeto(str(illumination_path), overwrite=True)
    file_set.auxiliary[f'illumination_pattern_{frame_name}'] = illumination_path
    print(f'  ✓ Saved illumination pattern: {illumination_path.name}')
    
    # Save dust flat as auxiliary file
    dust_flat_path = Path(config.directories.reduced) / f'{line}_{data_type}_{seq_str}_dust_flat_{frame_name}.fits'
    dust_hdu = fits.PrimaryHDU(data=result['dust_flat'])
    dust_hdu.header['COMMENT'] = f'Dust flat from spectroflat - {frame_name} frame'
    dust_hdu.header['FRAME'] = frame_name
    dust_hdu.writeto(str(dust_flat_path), overwrite=True)
    file_set.auxiliary[f'dust_flat_{frame_name}'] = dust_flat_path
    print(f'  ✓ Saved dust flat: {dust_flat_path.name}')
    
    return result['dust_flat']


def _process_single_frame_amend(config, data_type, frame_name, atlas_lines_path):
    """
    Run amend_spectroflat for a single frame.
    
    Parameters
    ----------
    config : Config
        Configuration object
    data_type : str
        'flat' or 'flat_center'
    frame_name : str
        'upper' or 'lower'
    atlas_lines_path : Path
        Path to atlas lines YAML file for this frame
        
    Returns
    -------
    bool
        True on success, False on failure
    """
    import numpy as np
    import re
    from astropy.io import fits
    
    print(f'  Running amend_spectroflat for {frame_name} frame...')
    
    # Get offset map and illumination pattern paths
    offset_map_path = config.dataset[data_type]['files'].auxiliary.get(f'offset_map_{frame_name}')
    illum_pattern_path = config.dataset[data_type]['files'].auxiliary.get(f'illumination_pattern_{frame_name}')
    
    if not offset_map_path or not offset_map_path.exists():
        print(f'  Error: Offset map not found for {frame_name}')
        return False
    
    if not atlas_lines_path or not atlas_lines_path.exists():
        print(f'  Error: Atlas lines file not found for {frame_name}')
        return False
    
    # Read L0 data
    l0_data, l0_header = tio.read_any_file(config, data_type, verbose=False, status='l0')
    if frame_name == 'upper':
        frame_data = l0_data[0]['upper'].data
    else:
        frame_data = l0_data[0]['lower'].data
    
    # Create temporary FITS with 4 identical states
    stacked_data = np.stack([frame_data, frame_data, frame_data, frame_data], axis=0)
    temp_fits_path = config.directories.reduced / f'temp_{data_type}_{frame_name}_for_amend.fits'
    hdu = fits.PrimaryHDU(data=stacked_data)
    hdu.writeto(temp_fits_path, overwrite=True)
    print(f'  ✓ Created temporary FITS: {temp_fits_path.name}')
    
    # Create temporary config
    atlas_config_path = config.cam.atlas_fit_config
    with open(atlas_config_path, 'r') as f:
        original_config_text = f.read()
    
    temp_config_text = original_config_text
    temp_config_text = re.sub(r'corrected_frame:\s*(.+)', f'corrected_frame: {temp_fits_path}', temp_config_text)
    temp_config_text = re.sub(r'roi:\s*"?\[([^\]]+)\]"?', r'roi: "[s, \1]"', temp_config_text)
    
    if 'mod_states:' in temp_config_text:
        temp_config_text = re.sub(r'mod_states:\s*\d+', 'mod_states: 4', temp_config_text)
    else:
        temp_config_text = re.sub(r'(corrected_frame:\s*.+)', r'\1\n  mod_states: 4', temp_config_text)
    
    temp_config_text = re.sub(r'(mod_states:\s*\d+)', rf'\1\n  offset_map: {offset_map_path}', temp_config_text)
    
    if illum_pattern_path and illum_pattern_path.exists():
        temp_config_text = re.sub(r'(offset_map:\s*.+)', rf'\1\n  soft_flat: {illum_pattern_path}', temp_config_text)
    
    # For lower frame: use stray_light_lower if available
    if frame_name == 'lower':
        stray_light_lower_match = re.search(r'stray_light_lower:\s*([\d.]+)', temp_config_text)
        if stray_light_lower_match:
            stray_light_lower_value = stray_light_lower_match.group(1)
            temp_config_text = re.sub(
                r'stray_light:\s*[\d.]+',
                f'stray_light: {stray_light_lower_value}',
                temp_config_text
            )
            print(f'  ✓ Using stray_light_lower: {stray_light_lower_value}% for lower frame')
    
    # Remove stray_light_lower entry (atlas-fit doesn't recognize it)
    temp_config_text = re.sub(
        r'^\s*#.*stray.light.*lower.*\n',  # Remove comment line
        '',
        temp_config_text,
        flags=re.MULTILINE | re.IGNORECASE
    )
    temp_config_text = re.sub(
        r'^\s*stray_light_lower:\s*[\d.]+\s*\n',  # Remove the actual entry
        '',
        temp_config_text,
        flags=re.MULTILINE
    )
    
    project_root = Path(__file__).resolve().parents[3]
    temp_config_path = project_root / 'configs' / f'temp_amend_{data_type}_{frame_name}_config.yml'
    with open(temp_config_path, 'w') as f:
        f.write(temp_config_text)
    
    print(f'  ✓ Created temporary config: {temp_config_path.name}')
    
    # Find amend script
    amend_script = project_root / 'atlas-fit' / 'bin' / 'amend_spectroflat'
    if not amend_script.exists():
        print(f'  Error: amend_spectroflat script not found')
        return False
    
    # Display command
    print(f'\n  Please run the following command in an EXTERNAL terminal:')
    print(f'  {"-"*68}')
    print(f'  cd {config.directories.reduced}')
    print(f'  {amend_script} {temp_config_path} {atlas_lines_path}')
    print(f'  {"-"*68}')
    
    # Wait for user confirmation
    while True:
        user_input = input(f'\n  Did amend_spectroflat complete successfully for {frame_name}? (y/n): ').strip().lower()
        if user_input == 'y':
            break
        elif user_input == 'n':
            print(f'  ✗ Skipping amend for {frame_name}')
            if temp_config_path.exists():
                temp_config_path.unlink()
            if temp_fits_path.exists():
                temp_fits_path.unlink()
            return False
        else:
            print('  Please enter "y" or "n"')
    
    # Process amended files
    reduced_dir = config.directories.reduced
    
    # Handle amended_soft_flat.fits
    amended_soft_flat = reduced_dir / 'amended_soft_flat.fits'
    if amended_soft_flat.exists():
        if illum_pattern_path and illum_pattern_path.exists():
            outdated_illumination = illum_pattern_path.parent / f'{illum_pattern_path.stem}_outdated{illum_pattern_path.suffix}'
            illum_pattern_path.rename(outdated_illumination)
            config.dataset[data_type]['files'].auxiliary[f'illumination_pattern_{frame_name}_outdated'] = outdated_illumination
        
        amended_soft_flat.rename(illum_pattern_path)
        config.dataset[data_type]['files'].auxiliary[f'illumination_pattern_{frame_name}'] = illum_pattern_path
        print(f'  ✓ Updated illumination pattern (wavelength calibrated)')
    
    # Handle wl_calibrated_offsets.fits
    wl_calibrated_offsets = reduced_dir / 'wl_calibrated_offsets.fits'
    if wl_calibrated_offsets.exists():
        if offset_map_path and offset_map_path.exists():
            outdated_offset = offset_map_path.parent / f'{offset_map_path.stem}_outdated{offset_map_path.suffix}'
            offset_map_path.rename(outdated_offset)
            config.dataset[data_type]['files'].auxiliary[f'offset_map_{frame_name}_outdated'] = outdated_offset
        
        wl_calibrated_offsets.rename(offset_map_path)
        config.dataset[data_type]['files'].auxiliary[f'offset_map_{frame_name}'] = offset_map_path
        print(f'  ✓ Updated offset map (wavelength calibrated)')
    
    # Clean up temporary files and unused amend outputs
    if temp_config_path.exists():
        temp_config_path.unlink()
    if temp_fits_path.exists():
        temp_fits_path.unlink()
    
    # Clean up amended_corrected_frame outputs (not needed for pipeline)
    amended_corrected_frame_fits = reduced_dir / 'amended_corrected_frame.fits'
    if amended_corrected_frame_fits.exists():
        amended_corrected_frame_fits.unlink()
        print(f'  ✓ Removed temporary amended_corrected_frame.fits')
    
    amended_corrected_frame_txt = reduced_dir / 'amended_corrected_frame.txt'
    if amended_corrected_frame_txt.exists():
        amended_corrected_frame_txt.unlink()
        print(f'  ✓ Removed temporary amended_corrected_frame.txt')
    
    return True


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
        print(f'\n{"="*70}')
        print(f'L0 → L1 REDUCTION FOR {data_type.upper()}')
        print(f'{"="*70}')
        print(f'Processing sequence: Complete all 3 steps for each frame half')
        print(f'  1. Atlas-fit wavelength calibration')
        print(f'  2. Spectroflat smile correction')
        print(f'  3. Amend spectroflat with wavelength calibration')
        print(f'{"="*70}\n')
        
        # Storage for results from each step
        atlas_lines_files = {}
        dust_flats = {}
        
        # Process each frame half completely before moving to the next
        for frame_name in ['upper', 'lower']:
            print(f'\n{"#"*70}')
            print(f'# PROCESSING {frame_name.upper()} FRAME')
            print(f'{"#"*70}\n')
            
            # ============================================================
            # STEP 1: ATLAS-FIT WAVELENGTH CALIBRATION FOR THIS FRAME
            # ============================================================
            print(f'\n{"-"*70}')
            print(f'STEP 1/{frame_name.upper()}: Atlas-fit wavelength calibration')
            print(f'{"-"*70}')
            
            # Check if atlas lines file already exists
            line = config.dataset['line']
            seq = config.dataset[data_type]['sequence']
            
            atlas_lines_file = config.dataset[data_type]['files'].auxiliary.get(f'atlas_lines_{frame_name}')
            
            skip_atlas_fit = False
            if atlas_lines_file and atlas_lines_file.exists():
                print(f'  Atlas lines file already exists: {atlas_lines_file.name}')
                while True:
                    user_choice = input(f'  Re-run atlas-fit for {frame_name}? [y/n]: ').strip().lower()
                    if user_choice in ['y', 'n']:
                        break
                    else:
                        print('  Please enter "y" or "n"')
                
                if user_choice == 'n':
                    print(f'  ✓ Using existing atlas lines file')
                    atlas_lines_files[frame_name] = atlas_lines_file
                    skip_atlas_fit = True
                else:
                    # User wants to re-run: delete the old file
                    print(f'  Deleting old atlas lines file: {atlas_lines_file.name}')
                    atlas_lines_file.unlink()
            
            if not skip_atlas_fit:
                # Run atlas-fit for this frame
                atlas_result = _process_single_frame_atlas_fit(config, data_type, frame_name)
                if not atlas_result:
                    print(f'\n✗ L1 reduction failed for {frame_name} (atlas-fit step).')
                    return None
                atlas_lines_files[frame_name] = atlas_result
                print(f'  ✓ Generated atlas lines file for {frame_name}')
            
            # ============================================================
            # STEP 2: SPECTROFLAT FOR THIS FRAME
            # ============================================================
            print(f'\n{"-"*70}')
            print(f'STEP 2/{frame_name.upper()}: Spectroflat smile correction')
            print(f'{"-"*70}')
            
            # Check if spectroflat outputs already exist (offset map and illumination pattern for this frame)
            offset_map_file = config.dataset[data_type]['files'].auxiliary.get(f'offset_map_{frame_name}')
            illum_pattern_file = config.dataset[data_type]['files'].auxiliary.get(f'illumination_pattern_{frame_name}')
            
            skip_spectroflat = False
            if offset_map_file and offset_map_file.exists() and illum_pattern_file and illum_pattern_file.exists():
                # Check if dust flat auxiliary file exists for this frame
                dust_flat_file = config.dataset[data_type]['files'].auxiliary.get(f'dust_flat_{frame_name}')
                
                if dust_flat_file and dust_flat_file.exists():
                    print(f'  Spectroflat outputs already exist:')
                    print(f'    Offset map: {offset_map_file.name}')
                    print(f'    Illumination pattern: {illum_pattern_file.name}')
                    print(f'    Dust flat: {dust_flat_file.name}')
                    while True:
                        user_choice = input(f'  Re-run spectroflat for {frame_name}? [y/n]: ').strip().lower()
                        if user_choice in ['y', 'n']:
                            break
                        else:
                            print('  Please enter "y" or "n"')
                    
                    if user_choice == 'n':
                        print(f'  ✓ Using existing spectroflat outputs')
                        skip_spectroflat = True
                else:
                    print(f'  Offset map and illumination pattern exist, but dust flat file is missing.')
                    print(f'  Will re-run spectroflat to regenerate dust flat.')
            
            if not skip_spectroflat:
                # Clean up _outdated files from previous amend step if they exist
                outdated_offset = config.dataset[data_type]['files'].auxiliary.get(f'offset_map_{frame_name}_outdated')
                outdated_illum = config.dataset[data_type]['files'].auxiliary.get(f'illumination_pattern_{frame_name}_outdated')
                
                if outdated_offset and outdated_offset.exists():
                    outdated_offset.unlink()
                    print(f'  ✓ Removed outdated offset map from previous amend: {outdated_offset.name}')
                    del config.dataset[data_type]['files'].auxiliary[f'offset_map_{frame_name}_outdated']
                
                if outdated_illum and outdated_illum.exists():
                    outdated_illum.unlink()
                    print(f'  ✓ Removed outdated illumination pattern from previous amend: {outdated_illum.name}')
                    del config.dataset[data_type]['files'].auxiliary[f'illumination_pattern_{frame_name}_outdated']
                
                # Run spectroflat for this frame
                dust_flat_result = _process_single_frame_spectroflat_wrapper(config, data_type, frame_name)
                if dust_flat_result is None:
                    print(f'\n✗ L1 reduction failed for {frame_name} (spectroflat step).')
                    return None
                dust_flats[frame_name] = dust_flat_result
                print(f'  ✓ Generated dust flat and offset map for {frame_name}')
            else:
                # If skipping, load the dust flat from auxiliary file
                dust_flat_file = config.dataset[data_type]['files'].auxiliary.get(f'dust_flat_{frame_name}')
                if dust_flat_file and dust_flat_file.exists():
                    print(f'  Loading dust flat from auxiliary file...')
                    with fits.open(dust_flat_file) as hdu:
                        dust_flats[frame_name] = hdu[0].data
                        print(f'  ✓ Loaded existing dust flat for {frame_name}')
                else:
                    print(f'  ✗ Error: Dust flat file not found for {frame_name}')
                    print(f'  This should not happen - spectroflat check passed but file is missing')
                    return None
            
            # ============================================================
            # STEP 3: AMEND SPECTROFLAT FOR THIS FRAME
            # ============================================================
            print(f'\n{"-"*70}')
            print(f'STEP 3/{frame_name.upper()}: Amend spectroflat with wavelength calibration')
            print(f'{"-"*70}')
            
            # Check if wavelength-calibrated files already exist
            # The presence of _outdated file indicates wavelength calibration has been applied
            outdated_offset = config.dataset[data_type]['files'].auxiliary.get(f'offset_map_{frame_name}_outdated')
            
            skip_amend = False
            if outdated_offset and outdated_offset.exists():
                print(f'  Wavelength-calibrated offset map already exists for {frame_name}')
                while True:
                    user_choice = input(f'  Re-run amend_spectroflat for {frame_name}? [y/n]: ').strip().lower()
                    if user_choice in ['y', 'n']:
                        break
                    else:
                        print('  Please enter "y" or "n"')
                
                if user_choice == 'n':
                    print(f'  ✓ Using existing wavelength-calibrated files')
                    skip_amend = True
            
            if not skip_amend:
                # Run amend for this frame
                amend_result = _process_single_frame_amend(config, data_type, frame_name, atlas_lines_files[frame_name])
                if not amend_result:
                    print(f'\n✗ L1 reduction failed for {frame_name} (amend step).')
                    return None
                print(f'  ✓ Wavelength calibration applied to {frame_name}')
        
        # ============================================================
        # SAVE L1 FILE WITH BOTH FRAMES
        # ============================================================
        print(f'\n{"="*70}')
        print(f'SAVING L1 FILE')
        print(f'{"="*70}')
        
        # Create L1 FramesSet with dust flats from both frames
        if all(df is not None for df in dust_flats.values()):
            reduced_frames = dct.FramesSet()
            frame_name_str = f"{data_type}_l1_frame{0:04d}"
            l1_frame = dct.Frame(frame_name_str)
            l1_frame.set_half("upper", dust_flats['upper'].astype('float32'))
            l1_frame.set_half("lower", dust_flats['lower'].astype('float32'))
            reduced_frames.add_frame(l1_frame, frame_idx=0)
            
            if return_reduced:
                return reduced_frames
            
            # Save to L1 FITS file
            _, header = tio.read_any_file(config, data_type, verbose=False, status='l0')
            seq_str = f"t{seq:03d}"
            extra_keywords = {
                'ATLSUPR': (str(atlas_lines_files.get('upper', '')), 'Atlas lines file for upper frame'),
                'ATLSLWR': (str(atlas_lines_files.get('lower', '')), 'Atlas lines file for lower frame'),
                'SPCTRFLT': ('TRUE', 'Spectroflat correction applied'),
            }
            
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
            return out_path
        else:
            # If we skipped spectroflat, L1 file should already exist
            l1_file = config.dataset[data_type]['files'].get('l1')
            if l1_file and l1_file.exists():
                print(f'✓ Using existing L1 file: {l1_file}')
                return l1_file
            else:
                print(f'✗ Cannot create L1 file - dust flats not available')
                print(f'  Please re-run spectroflat step')
                return None
    
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

