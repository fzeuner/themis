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
from scipy.ndimage import shift as scipy_shift
from tqdm import tqdm  


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
        sigma_mask=4.5,  # masks pixels which deviate from the mean by more than 4.5 sigma
        fit_border=1,
        average_column_response_map=False,
        ignore_gradient=True,
        roi=roi
    )
    sf_config.smile = SmileConfig(
        line_distance=11,
        strong_smile_deg=8, #8 - 20 not much effect
        max_dispersion_deg=4,
        line_prominence=0.1,
        height_sigma=0.04, # 0.04 - 0.2 not much effect
        smooth=True,
        emission_spectrum=False,
        state_aware=False,
        align_states=False,
        smile_deg=3, # 3  - 8 not much effect
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
        print('Subtracting dark from all scan frames (raw -> L0)...')
        items_iter = data.items()
        items_iter = tqdm(items_iter, desc='RAW→L0 scan')

        for key, frame in items_iter:
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
    
    # Save dust flat as auxiliary file
    dust_flat_path = Path(config.directories.reduced) / f'{line}_{data_type}_{seq_str}_dust_flat_{frame_name}.fits'
    dust_hdu = fits.PrimaryHDU(data=result['dust_flat'])
    dust_hdu.header['COMMENT'] = f'Dust flat from spectroflat - {frame_name} frame'
    dust_hdu.header['FRAME'] = frame_name
    dust_hdu.writeto(str(dust_flat_path), overwrite=True)
    file_set.auxiliary[f'dust_flat_{frame_name}'] = dust_flat_path
    print(f'  ✓ Saved dust flat: {dust_flat_path.name}')
    
    return result['dust_flat']


def reduce_l0_to_l1(config, data_type=None, return_reduced=False, auto_reduce_dark: bool = False):
    """
    Reduce L0 data to L1 level.
    
    For flat and flat_center: performs wavelength calibration using atlas-fit bin/prepare.
    """
    
    def _load_flat_center_dust_flats(cfg):
        """Load flat_center dust flats (status='dust') and return dict per half."""
        print('Loading flat_center dust flats (status="dust")...')
        try:
            dust_collection, _ = tio.read_any_file(cfg, 'flat_center', status='dust', verbose=False)
        except FileNotFoundError as e:
            print('  Error: flat_center dust flats not found. Run flat_center l0->l1 first.')
            print(f'  Reason: {e}')
            return None

        dust_frame = dust_collection.get(0)
        if dust_frame is None:
            print('  Error: No dust frame found in flat_center dust collection')
            return None

        dust_by_half = {}
        for half in ['upper', 'lower']:
            dust_by_half[half] = dust_frame.get_half(half).data.astype('float32')
        return dust_by_half

    def _dust_correct_frame(src_frame, frame_name_str, dust_by_half, propagate_pol_state=False):
        """Create a new Frame with L1 = L0 / dust per half.

        If propagate_pol_state is True, copy pol_state from src_frame halves.
        """
        dest = dct.Frame(frame_name_str)
        for half in ['upper', 'lower']:
            half_obj = src_frame.get_half(half)
            data_l0 = half_obj.data.astype('float32')
            data_l1 = data_l0 / dust_by_half[half]
            if propagate_pol_state and half_obj.pol_state:
                dest.set_half(half, data_l1, half_obj.pol_state)
            else:
                dest.set_half(half, data_l1)
        return dest

    if data_type is None:
        print('No processing - provide a specific data type.')
        return None

    # Common containers filled by each branch, saved once at the end
    reduced_frames = None
    header = None
    extra_keywords = {}
    target_data_type = data_type

    if data_type == 'dark':
        print('No reduction procedure defined for dark l0->l1.')
        return None
    
    elif data_type == 'scan':
        print(f"\n{'='*70}")
        print('L0 -> L1 REDUCTION FOR SCAN')
        print(f"{'='*70}")

        # ------------------------------------------------------------
        # 1) Load dust flats from flat_center (status='dust')
        # ------------------------------------------------------------
        dust_flats = _load_flat_center_dust_flats(config)
        if dust_flats is None:
            return None

        # ------------------------------------------------------------
        # 2) Load L0 scan and apply per-half dust correction
        # ------------------------------------------------------------
        print('Loading L0 scan data...')
        l0_scan, header = tio.read_any_file(config, 'scan', status='l0', verbose=False)

        if not isinstance(l0_scan, dct.CycleSet):
            print('  Error: Expected L0 scan data to be a CycleSet')
            return None

        reduced_frames = dct.CycleSet()

        print('Applying dust flat correction to all scan frames (L1 = L0 / dust_flat)...')

        items_iter = l0_scan.items()
        items_iter = tqdm(items_iter, desc='L0→L1 scan')

        for key, frame in items_iter:
            # key is (frame_state, slit_idx, map_idx)
            frame_name_str = f"scan_l1_{frame.name}"
            single_frame = _dust_correct_frame(frame, frame_name_str, dust_flats, propagate_pol_state=True)
            reduced_frames.add_frame(single_frame, key)

        extra_keywords = {
            'SPCTRFLT': ('TRUE', 'Scan dust correction applied using flat_center dust flats (L0/dust)'),
        }

    elif data_type == 'flat':
        # Treat flat like scan: use flat_center dust flats, do NOT run spectroflat on flat
        print(f"\n{'='*70}")
        print('L0 -> L1 REDUCTION FOR FLAT (USING FLAT_CENTER DUST)')
        print(f"{'='*70}")

        # 1) Load dust flats from flat_center
        dust_flats = _load_flat_center_dust_flats(config)
        if dust_flats is None:
            return None

        # 2) Load L0 flat and apply per-half dust correction
        print('Loading L0 flat data...')
        l0_frames, header = tio.read_any_file(config, 'flat', status='l0', verbose=False)
        l0_frame = l0_frames.get(0)
        if l0_frame is None:
            print('  Error: No L0 frame found for flat')
            return None

        reduced_frames = dct.FramesSet()
        frame_name_str = f"flat_l1_frame{0:04d}"
        print('Applying dust flat correction to flat (L1 = L0 / dust_flat)...')
        l1_frame = _dust_correct_frame(l0_frame, frame_name_str, dust_flats, propagate_pol_state=False)
        reduced_frames.add_frame(l1_frame, 0)

        extra_keywords = {
            'SPCTRFLT': ('TRUE', 'Flat dust correction applied using flat_center dust flats (L0/dust)'),
        }

    elif data_type == 'flat_center':
        print(f'\n{"="*70}')
        print(f'L0 → L1 REDUCTION FOR FLAT_CENTER')
        print(f'{"="*70}')
        
        dust_flats = {}
        
        # Process each frame half completely before moving to the next
        for frame_name in ['upper', 'lower']:
            print(f'\n{"#"*70}')
            print(f'# PROCESSING {frame_name.upper()} FRAME')
            print(f'{"#"*70}\n')
            
            # ============================================================
            # SPECTROFLAT DUST FOR THIS FRAME
            # ============================================================
        
                # Check if dust flat auxiliary file exists for this frame
            skip_spectroflat = False
            dust_flat_file = config.dataset[data_type]['files'].auxiliary.get(f'dust_flat_{frame_name}')
            
            if dust_flat_file and dust_flat_file.exists():
                print(f'  Spectroflat outputs already exist:')
                print(f'    Dust flat: {dust_flat_file.name}')
                while True:
                    user_choice = input(f'  Re-run spectroflat for {frame_name}? [y/n]: ').strip().lower()
                    if user_choice in ['y', 'n']:
                        break
                    else:
                        print('  Please enter "y" or "n"')
                
                if user_choice == 'n':
                    print(f'  Using existing spectroflat outputs')
                    skip_spectroflat = True
            
            if not skip_spectroflat:
                # Run spectroflat for this frame
                dust_flat_result = _process_single_frame_spectroflat_wrapper(config, data_type, frame_name)
                if dust_flat_result is None:
                    print(f'\n L1 reduction failed for {frame_name} (spectroflat step).')
                    return None
                dust_flats[frame_name] = dust_flat_result
                print(f'  Generated dust flat for {frame_name}')
            else:
                # If skipping, load dust flats using the standard IO helper (status='dust')
                print(f'  Loading dust flats from auxiliary files via read_any_file(status="dust")...')
                try:
                    dust_collection, _ = tio.read_any_file(config, data_type, verbose=False, status='dust')
                except FileNotFoundError as e:
                    print(f'  Error: Could not load dust flats from auxiliary files: {e}')
                    return None

                dust_frame = dust_collection.get(0)
                if dust_frame is None:
                    print('  Error: No dust frame found in dust collection')
                    return None

                half_data = dust_frame.get_half(frame_name).data
                dust_flats[frame_name] = half_data
                print(f'  Loaded existing dust flat for {frame_name} from status="dust"')

        # Read L0 averaged frame (index 0) for division by dust flats
        l0_frames, header = tio.read_any_file(config, data_type, verbose=False, status='l0')
        l0_frame = l0_frames.get(0)
        if l0_frame is None:
            print('  Error: No L0 frame found for flat/flat_center')
            return None
        
        # Create L1 FramesSet with dust flats from both frames
        if all(df is not None for df in dust_flats.values()):
            reduced_frames = dct.FramesSet()
            frame_name_str = f"{data_type}_l1_frame{0:04d}"
            # L1 corrected flat_center = L0 / dust, per half
            l1_frame = _dust_correct_frame(l0_frame, frame_name_str, dust_flats, propagate_pol_state=False)

            reduced_frames.add_frame(l1_frame, frame_idx=0)

            extra_keywords = {
                'SPCTRFLT': ('TRUE', 'Spectroflat dust correction applied (L0/dust)'),
            }
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

    # -----------------------------------------------------------------
    # Finalize: either return FramesSet/CycleSet or save to disk once
    # -----------------------------------------------------------------
    if reduced_frames is None:
        print(f'Unknown data_type: {data_type}')
        return None

    if return_reduced:
        return reduced_frames

    out_path = tio.save_reduction(
        config,
        data_type=target_data_type,
        level='l1',
        frames=reduced_frames,
        source_header=header,
        verbose=True,
        overwrite=True,
        extra_keywords=extra_keywords,
    )

    print(f'✓ L1 {target_data_type} file saved')
    return out_path


def _load_yshift_arrays(config):
    """Load y-shift arrays from auxiliary files.
    
    Returns
    -------
    dict or None
        Dictionary with 'upper' and 'lower' keys containing the y-shift arrays,
        or None if files are not found.
    """
    yshifts = {}
    
    # Y-shifts are stored in flat_center auxiliary files
    files_fc = config.dataset.get('flat_center', {}).get('files')
    if files_fc is None:
        print('  Error: flat_center files not found in config')
        return None
    
    aux = getattr(files_fc, 'auxiliary', {})
    
    for frame in ['upper', 'lower']:
        key = f'yshift_{frame}'
        path = aux.get(key)
        if path is None or not Path(path).exists():
            print(f'  Error: y-shift auxiliary file not found for {frame}')
            print(f'    Expected key: {key}')
            print(f'    Run determine_image_shifts.py first to generate y-shift files.')
            return None
        yshifts[frame] = np.load(path)
        print(f'  Loaded y-shift for {frame}: {Path(path).name} (shape: {yshifts[frame].shape})')
    
    return yshifts


def _apply_yshift_to_half(data_2d, yshift_array):
    """Apply per-column y-shifts to a 2D image.
    
    Parameters
    ----------
    data_2d : 2D ndarray
        Input image (ny, nx).
    yshift_array : 1D ndarray
        Y-shift values for each column (length must equal nx).
    
    Returns
    -------
    shifted : 2D ndarray
        Image with per-column y-shifts applied.
    """
    ny, nx = data_2d.shape
    if yshift_array.size != nx:
        raise ValueError(
            f"yshift_array length {yshift_array.size} does not match image width {nx}"
        )
    
    shifted = np.empty_like(data_2d, dtype=float)
    
    for j in range(nx):
        col = data_2d[:, j].astype(float)
        dy = yshift_array[j]
        if np.isfinite(dy) and dy != 0:
            # Shift along y (axis=0), negative because we want to correct the shift
            shifted[:, j] = scipy_shift(col, -dy, order=3, mode='constant', cval=np.nan)
        else:
            shifted[:, j] = col
    
    return shifted


def reduce_l1_to_l2(config, data_type=None, return_reduced=False):
    """L1 → L2 reduction: apply y-shifts from calibration target analysis.
    
    For scan, flat, and flat_center data types, each frame (upper/lower) is
    shifted using the y-shift arrays determined from the calibration target
    and saved as auxiliary numpy files.
    
    Parameters
    ----------
    config : Config
        Configuration object with dataset and file paths.
    data_type : str
        One of 'scan', 'flat', 'flat_center'.
    return_reduced : bool
        If True, return the reduced frames instead of saving to disk.
    
    Returns
    -------
    Path or FramesSet/CycleSet or None
        Path to saved L2 file, or reduced frames if return_reduced=True,
        or None on failure.
    """
    if data_type is None:
        print('No processing - provide a specific data type.')
        return None
    
    if data_type not in ['scan', 'flat', 'flat_center']:
        print(f'L2 reduction not defined for data_type: {data_type}')
        return None
    
    print(f"\n{'='*70}")
    print(f'L1 → L2 REDUCTION FOR {data_type.upper()} (Y-SHIFT CORRECTION)')
    print(f"{'='*70}")
    
    # Load y-shift arrays from auxiliary files
    print('\nLoading y-shift arrays from auxiliary files...')
    yshifts = _load_yshift_arrays(config)
    if yshifts is None:
        return None
    
    # Load L1 data
    print(f'\nLoading L1 {data_type} data...')
    l1_data, header = tio.read_any_file(config, data_type, status='l1', verbose=False)
    
    reduced_frames = None
    extra_keywords = {}
    
    if data_type == 'scan':
        if not isinstance(l1_data, dct.CycleSet):
            print('  Error: Expected L1 scan data to be a CycleSet')
            return None
        
        reduced_frames = dct.CycleSet()
        
        print('Applying y-shift correction to all scan frames...')

        items_iter = l1_data.items()
        
        items_iter = tqdm(items_iter, desc='L1→L2 scan')

        for key, frame in items_iter:
            frame_name_str = f"scan_l2_{frame.name}"
            dest = dct.Frame(frame_name_str)
            
            for half in ['upper', 'lower']:
                half_obj = frame.get_half(half)
                data_l1 = half_obj.data.astype('float32')
                data_l2 = _apply_yshift_to_half(data_l1, yshifts[half])
                
                if half_obj.pol_state:
                    dest.set_half(half, data_l2.astype('float32'), half_obj.pol_state)
                else:
                    dest.set_half(half, data_l2.astype('float32'))
            
            reduced_frames.add_frame(dest, key)
        
        extra_keywords = {
            'YSHIFTCR': ('TRUE', 'Y-shift correction applied from calibration target analysis'),
        }
    
    elif data_type in ['flat', 'flat_center']:
        if not isinstance(l1_data, dct.FramesSet):
            print(f'  Error: Expected L1 {data_type} data to be a FramesSet')
            return None
        
        l1_frame = l1_data.get(0)
        if l1_frame is None:
            print(f'  Error: No L1 frame found for {data_type}')
            return None
        
        reduced_frames = dct.FramesSet()
        frame_name_str = f"{data_type}_l2_frame{0:04d}"
        dest = dct.Frame(frame_name_str)
        
        print(f'Applying y-shift correction to {data_type}...')
        for half in ['upper', 'lower']:
            half_obj = l1_frame.get_half(half)
            data_l1 = half_obj.data.astype('float32')
            data_l2 = _apply_yshift_to_half(data_l1, yshifts[half])
            dest.set_half(half, data_l2.astype('float32'))
        
        reduced_frames.add_frame(dest, 0)
        
        extra_keywords = {
            'YSHIFTCR': ('TRUE', 'Y-shift correction applied from calibration target analysis'),
        }
    
    # Finalize: either return or save
    if reduced_frames is None:
        print(f'Unknown data_type: {data_type}')
        return None
    
    if return_reduced:
        return reduced_frames
    
    out_path = tio.save_reduction(
        config,
        data_type=data_type,
        level='l2',
        frames=reduced_frames,
        source_header=header,
        verbose=True,
        overwrite=True,
        extra_keywords=extra_keywords,
    )
    
    print(f'✓ L2 {data_type} file saved')
    return out_path


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
    "scan": ("Dust flat correction using flat_center dust flats. "
              "Each L0 scan frame (upper/lower) is divided by the corresponding flat_center dust flat."),
    "flat": ("Dust flat correction using flat_center dust flats. "
              "Each L0 scan frame (upper/lower) is divided by the corresponding flat_center dust flat."),
    "flat_center": "Flat-field correction with spectroflat generated dust_flat. Saves L1 FITS with dust corrected frames and auxilary file with dust flat."
}))

reduction_levels.add(ReductionLevel("l2", "_l2.fits", reduce_l1_to_l2, {
    "dark": "Nothing.",
    "scan": ("Y-shift correction using calibration target analysis. "
             "Each L1 scan frame (upper/lower) is shifted per-column using y-shift arrays from auxiliary files."),
    "flat": ("Y-shift correction using calibration target analysis. "
             "Each L1 flat frame (upper/lower) is shifted per-column using y-shift arrays from auxiliary files."),
    "flat_center": ("Y-shift correction using calibration target analysis. "
                    "Each L1 flat_center frame (upper/lower) is shifted per-column using y-shift arrays from auxiliary files.")
}))

