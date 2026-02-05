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
from scipy.ndimage import map_coordinates

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

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
            # NOTE: Using cval=np.nan with spline interpolation (order>1) can
            # spread NaNs through the entire output column due to spline
            # prefiltering. Use a NaN-safe boundary mode instead.
            shifted[:, j] = scipy_shift(col, -dy, order=1, mode='nearest')
        else:
            shifted[:, j] = col
    
    return shifted


def _apply_yshift_map_to_half(data_2d, yshift_map):
    yshift_map = np.asarray(yshift_map, dtype=float)
    if yshift_map.shape != data_2d.shape:
        raise ValueError(f"yshift_map shape {yshift_map.shape} does not match image shape {data_2d.shape}")

    ny, nx = data_2d.shape
    yy, xx = np.meshgrid(np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing='ij')

    # Match the sign convention of _apply_yshift_to_half:
    # scipy_shift(col, -dy) implies output[y] = input[y + dy].
    coords = [yy + yshift_map, xx]

    shifted = map_coordinates(
        data_2d.astype(float),
        coords,
        order=1,
        mode='nearest',
    )
    return shifted


def _apply_yshift_correction_to_half(data_2d, yshift):
    yshift = np.asarray(yshift)
    if yshift.ndim == 1:
        return _apply_yshift_to_half(data_2d, yshift)
    if yshift.ndim == 2:
        return _apply_yshift_map_to_half(data_2d, yshift)
    raise ValueError(f"Unsupported yshift ndim={yshift.ndim}; expected 1D (nx,) or 2D (ny,nx).")


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
                data_l2 = _apply_yshift_correction_to_half(data_l1, yshifts[half])
                
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
            data_l2 = _apply_yshift_correction_to_half(data_l1, yshifts[half])
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


def _apply_desmiling(image, offset_map_2d):
    """
    Apply desmiling row-by-row using spectroflat's SmileInterpolator.desmile_row.
    This matches how atlas-fit's amend_spectroflat applies corrections.
    
    Parameters
    ----------
    image : np.ndarray
        2D image [spatial, wavelength]
    offset_map_2d : np.ndarray
        2D offset map array [spatial, wavelength]
    
    Returns
    -------
    np.ndarray
        Desmiled image [spatial, wavelength]
    """
    from spectroflat.smile import SmileInterpolator
    from spectroflat.utils.processing import MP
    
    ny, nx = image.shape
    rows = np.arange(ny)
    xes = np.arange(nx)
    
    # Apply desmiling row by row, same as atlas-fit does
    # Arguments: (row_index, x_coordinates, offsets_for_this_row, data_for_this_row)
    arguments = [(r, xes, offset_map_2d[r], image[r]) for r in rows]
    res = dict(MP.simultaneous(SmileInterpolator.desmile_row, arguments))
    
    # Reconstruct the desmiled image from row results
    desmiled = np.array([res[row] for row in rows])
    
    return desmiled


def _create_temp_atlas_config(atlas_config_path, output_name, remove_stray_light_key=True):
    """
    Create a temporary atlas-fit config file with unsupported keys removed.
    
    Always removes stray_light_lower which is a custom key not recognized by 
    atlas-fit's Config class. Optionally removes stray_light as well.
    
    Parameters
    ----------
    atlas_config_path : Path or str
        Path to the original atlas-fit config file
    output_name : str
        Name for the temporary config file (without path)
    remove_stray_light_key : bool
        If True (default), also remove stray_light key (for external atlas-fit calls).
        If False, keep stray_light (for internal Comperator usage).
        
    Returns
    -------
    tuple
        (temp_config_path, config_text)
    """
    with open(atlas_config_path, 'r') as f:
        config_text = f.read()
    
    # Always remove stray_light_lower (custom key not recognized by atlas-fit)
    config_text = re.sub(
        r'^\s*#.*stray.light.*lower.*\n',
        '',
        config_text,
        flags=re.MULTILINE | re.IGNORECASE
    )
    config_text = re.sub(
        r'^\s*stray_light_lower:\s*[\d.]+\s*\n',
        '',
        config_text,
        flags=re.MULTILINE
    )
    
    # Optionally remove stray_light (only for external atlas-fit prepare calls)
    if remove_stray_light_key:
        config_text = re.sub(
            r'^\s*stray_light:\s*[\d.]+\s*\n',
            '',
            config_text,
            flags=re.MULTILINE
        )
    
    # Write temporary config file
    project_root = Path(__file__).resolve().parents[3]
    temp_config_path = project_root / 'configs' / output_name
    with open(temp_config_path, 'w') as f:
        f.write(config_text)
    
    return temp_config_path, config_text


def _load_wavelength_calibration(config, data_type='flat_center'):
    """
    Load wavelength calibration from auxiliary file.
    
    Parameters
    ----------
    config : Config
        Configuration object
    data_type : str
        Data type to load wavelength calibration for. For 'scan' and 'flat',
        the calibration is loaded from 'flat_center' auxiliary files.
        
    Returns
    -------
    dict or None
        Dictionary with wavelength calibration data for 'upper' and 'lower' frames:
        {
            'upper': {'wavelength': np.array, 'min_wl': float, 'max_wl': float, 'dispersion': float, 'n_pixels': int},
            'lower': {'wavelength': np.array, 'min_wl': float, 'max_wl': float, 'dispersion': float, 'n_pixels': int}
        }
        Returns None if file not found.
    """
    # Wavelength calibration is always from flat_center
    source_type = 'flat_center'
    
    files_src = config.dataset.get(source_type, {}).get('files')
    if files_src is None:
        return None
    
    aux = getattr(files_src, 'auxiliary', {})
    wl_calib_file = aux.get('wavelength_calibration')
    
    if wl_calib_file is None or not Path(wl_calib_file).exists():
        return None
    
    # Load wavelength calibration FITS file
    wl_calibrations = {}
    
    with fits.open(wl_calib_file) as hdul:
        for frame_name in ['upper', 'lower']:
            ext_name = f'WL_{frame_name.upper()}'
            if ext_name in hdul:
                wl_hdu = hdul[ext_name]
                wl_calibrations[frame_name] = {
                    'wavelength': np.array(wl_hdu.data),
                    'min_wl': wl_hdu.header.get('MIN_WL'),
                    'max_wl': wl_hdu.header.get('MAX_WL'),
                    'dispersion': wl_hdu.header.get('DISPERS'),
                    'n_pixels': wl_hdu.header.get('NPIXELS'),
                }
    
    if len(wl_calibrations) != 2:
        return None
    
    return wl_calibrations


def _generate_wavelength_calibration(config, data_type='flat_center'):
    """
    Generate a combined wavelength calibration file for upper and lower frames
    using existing atlas_fit_lines files and the external generate_wl_calibration script.
    
    Parameters
    ----------
    config : Config
        Configuration object
    data_type : str
        Data type (default 'flat_center')
        
    Returns
    -------
    Path or None
        Path to the generated wavelength calibration file, or None on failure
    """
    print(f'\n{"="*70}')
    print(f'GENERATING WAVELENGTH CALIBRATION FILE')
    print(f'{"="*70}')
    
    # Check if wavelength calibration file already exists (auto-detection)
    file_set = config.dataset[data_type]['files']
    aux = getattr(file_set, 'auxiliary', {})
    wl_calib_file = aux.get('wavelength_calibration')
    
    if wl_calib_file and Path(wl_calib_file).exists():
        print(f'  Wavelength calibration file already exists: {Path(wl_calib_file).name}')
        while True:
            user_choice = input(f'  Re-generate wavelength calibration? [y/n]: ').strip().lower()
            if user_choice in ['y', 'n']:
                break
            else:
                print('  Please enter "y" or "n"')
        
        if user_choice == 'n':
            print(f'  ✓ Using existing wavelength calibration file')
            return wl_calib_file
        else:
            print(f'  Deleting old wavelength calibration file: {Path(wl_calib_file).name}')
            Path(wl_calib_file).unlink()
    
    # Check atlas-fit config
    if not hasattr(config.cam, 'atlas_fit_config') or config.cam.atlas_fit_config is None:
        print('  Error: No atlas_fit_config found for this camera.')
        return None
    
    atlas_config_path = config.cam.atlas_fit_config
    if not Path(atlas_config_path).exists():
        print(f'  Error: Atlas fit config not found: {atlas_config_path}')
        return None
    
    # Check L3 file exists
    l3_file = config.dataset[data_type]['files'].get('l3')
    if l3_file is None or not l3_file.exists():
        print(f'  Error: L3 {data_type} file not found. Run L3 reduction first.')
        return None
    
    # Check atlas_lines files exist for both frames
    aux = getattr(file_set, 'auxiliary', {})
    upper_lines = aux.get('atlas_lines_upper')
    lower_lines = aux.get('atlas_lines_lower')
    
    if upper_lines is None or not Path(upper_lines).exists():
        print(f'  Error: atlas_lines file not found for upper frame')
        print(f'  Run atlas-fit wavelength calibration first.')
        return None
    if lower_lines is None or not Path(lower_lines).exists():
        print(f'  Error: atlas_lines file not found for lower frame')
        print(f'  Run atlas-fit wavelength calibration first.')
        return None
    
    # Extract individual frames to temporary files
    print(f'\n  Extracting frames for wavelength calibration...')
    temp_upper_path = config.directories.reduced / f'temp_{data_type}_upper_for_wl_calib.fits'
    temp_lower_path = config.directories.reduced / f'temp_{data_type}_lower_for_wl_calib.fits'
    
    try:
        l3_data, _ = tio.read_any_file(config, data_type, verbose=False, status='l3')
        
        for frame_name, temp_path in [('upper', temp_upper_path), ('lower', temp_lower_path)]:
            frame_data = l3_data[0][frame_name].data
            hdu = fits.PrimaryHDU(data=frame_data)
            hdu.writeto(temp_path, overwrite=True)
        print(f'  ✓ Extracted upper and lower frames to temporary files')
    except Exception as e:
        print(f'  Error extracting frames: {e}')
        return None
    
    # Create temporary config file (keep stray_light for Comperator)
    temp_config_path, _ = _create_temp_atlas_config(
        atlas_config_path, 
        f'temp_wl_calib_{data_type}_config.yml',
        remove_stray_light_key=False
    )
    
    # Determine output path
    line = config.dataset['line']
    seq = config.dataset[data_type]['sequence']
    seq_str = f"t{seq:03d}"
    wl_calib_path = Path(config.directories.reduced) / f'{line}_{data_type}_{seq_str}_wavelength_calibration.fits'
    
    # Find script
    project_root = Path(__file__).resolve().parents[3]
    wl_script = project_root / 'scripts' / 'generate_wl_calibration'
    
    # Display command to run
    print(f'\n  Please run the following command in an EXTERNAL terminal:')
    print(f'  {"-"*68}')
    print(f'  conda activate atlas-fit')
    print(f'  cd {config.directories.reduced}')
    print(f'  {wl_script} \\')
    print(f'      {temp_config_path} \\')
    print(f'      {upper_lines} \\')
    print(f'      {lower_lines} \\')
    print(f'      {temp_upper_path} \\')
    print(f'      {temp_lower_path} \\')
    print(f'      {wl_calib_path}')
    print(f'  {"-"*68}')
    
    # Wait for user confirmation
    while True:
        user_input = input(f'\n  Did wavelength calibration complete successfully? (y/n): ').strip().lower()
        if user_input == 'y':
            print(f'  ✓ Wavelength calibration completed')
            break
        elif user_input == 'n':
            print(f'  ✗ Skipping wavelength calibration')
            # Clean up temp files
            for temp_file in [temp_upper_path, temp_lower_path, temp_config_path]:
                if temp_file.exists():
                    temp_file.unlink()
            return None
        else:
            print('  Please enter "y" or "n"')
    
    # Verify output file exists
    if not wl_calib_path.exists():
        print(f'  Error: Wavelength calibration file not found: {wl_calib_path}')
        # Clean up temp files before returning
        for temp_file in [temp_upper_path, temp_lower_path, temp_config_path]:
            if temp_file.exists():
                temp_file.unlink()
        return None
    
    # Clean up temp files
    for temp_file in [temp_upper_path, temp_lower_path, temp_config_path]:
        if temp_file.exists():
            temp_file.unlink()
    print(f'  ✓ Cleaned up temporary files')
    
    # Store in auxiliary files
    if not hasattr(file_set, 'auxiliary'):
        file_set.auxiliary = {}
    file_set.auxiliary['wavelength_calibration'] = wl_calib_path
    
    print(f'  ✓ Saved wavelength calibration: {wl_calib_path.name}')
    return wl_calib_path


def _process_single_frame_atlas_fit(config, data_type, frame_name):
    """
    Helper function to run atlas-fit for a single frame (upper or lower).
    Uses L3 (desmiled) data as input.
    
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
    
    print(f'  Running atlas-fit for {frame_name} frame...')
    
    # Check if atlas-fit config is available
    if not hasattr(config.cam, 'atlas_fit_config') or config.cam.atlas_fit_config is None:
        print('  Error: No atlas_fit_config found for this camera.')
        return None
    
    atlas_config_path = config.cam.atlas_fit_config
    if not Path(atlas_config_path).exists():
        print(f'  Error: Atlas fit config file not found: {atlas_config_path}')
        return None
    
    # Get the L3 file path (desmiled data)
    l3_file = config.dataset[data_type]['files'].get('l3')
    if l3_file is None or not l3_file.exists():
        print(f'  Error: L3 {data_type} file not found.')
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
        # Read the L3 data (desmiled)
        data, header = tio.read_any_file(config, data_type, verbose=False, status='l3')
        
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
    
    # Create temporary config file with unsupported keys removed
    temp_config_path, config_text = _create_temp_atlas_config(
        atlas_config_path,
        f'temp_atlas_fit_{data_type}_{frame_name}_config.yml'
    )
    
    # Also update corrected_frame to point to extracted frame
    config_text = re.sub(
        r'corrected_frame:\s*(.+)',
        f'corrected_frame: {temp_frame_path}',
        config_text
    )
    with open(temp_config_path, 'w') as f:
        f.write(config_text)
    
    # Determine output filename
    line = config.dataset['line']
    seq = config.dataset[data_type]['sequence']
    base_filename = f'{line}_{data_type}_t{seq:03d}'
    lines_file = config.directories.reduced / f'{base_filename}_{frame_name}_atlas_lines.yaml'
    
    print(f'  ✓ Created temporary config: {temp_config_path.name}')
    
    # The prepare script outputs to 'atlas_fit_lines.yaml' in the working directory
    temp_output_file = config.directories.reduced / 'atlas_fit_lines.yaml'
    
    # Display command to run
    print(f'\n  Please run the following command in an EXTERNAL terminal:')
    print(f'  {"-"*68}')
    print(f'  conda activate atlas-fit')
    print(f'  cd {config.directories.reduced}')
    print(f'  {prepare_script} {temp_config_path}')
    print(f'  {"-"*68}')
    print(f'  Note: Output will be saved as atlas_fit_lines.yaml and renamed automatically.')
    if frame_name == 'upper':
        print(f'\n  IMPORTANT: After atlas-fit completes, note the fitted stray-light value')
        print(f'  and FWHM displayed in the output (e.g., "stray-light: X.XX %") and update')
        print(f'  "stray_light" and "fwhm" in the config file before running amend_spectroflat:')
        print(f'  {atlas_config_path}')
    elif frame_name == 'lower':
        print(f'\n  IMPORTANT: After atlas-fit completes, note the fitted stray-light value')
        print(f'  and update "stray_light_lower" in the config file (FWHM is shared):')
        print(f'  {atlas_config_path}')
    
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


def _remove_outdated_auxiliary_files(config, data_type):
    """
    Remove outdated auxiliary files (offset_map and illumination_pattern) from 
    filesystem and config when creating new L3 products.
    
    Parameters
    ----------
    config : Config
        Configuration object
    data_type : str
        Data type (e.g., 'flat_center')
    """
    file_set = config.dataset[data_type]['files']
    aux = getattr(file_set, 'auxiliary', {})
    
    outdated_keys = []
    for key in list(aux.keys()):
        if '_outdated' in key:
            path = aux[key]
            if path and Path(path).exists():
                print(f'  Removing outdated file: {Path(path).name}')
                Path(path).unlink()
            outdated_keys.append(key)
    
    # Remove from config
    for key in outdated_keys:
        del aux[key]
        print(f'  Removed auxiliary key: {key}')


def reduce_l2_to_l3(config, data_type=None, return_reduced=False):
    """L2 -> L3 reduction: smile correction via offset maps.
    
    For flat_center: runs spectroflat on L2 (y-shift corrected) data to produce
    offset map and illumination pattern. Removes outdated auxiliary files.
    
    For flat: applies offset map from flat_center to L2 flat data to produce
    desmiled L3 flat.
    
    Parameters
    ----------
    config : Config
        Configuration object with dataset and file paths.
    data_type : str
        One of 'flat', 'flat_center'.
    return_reduced : bool
        If True, return the reduced frames instead of saving to disk.
    
    Returns
    -------
    Path or FramesSet or None
        Path to saved L3 file, or reduced frames if return_reduced=True,
        or None on failure.
    """
    if data_type is None:
        print('No processing - provide a specific data type.')
        return None
    
    if data_type not in ['flat_center', 'flat', 'scan']:
        print(f'L3 reduction not yet defined for data_type: {data_type}')
        return None
    
    print(f"\n{'='*70}")
    print(f'L2 → L3 REDUCTION FOR {data_type.upper()} (DESMILING)')
    print(f"{'='*70}")
    
    # For flat_center: optionally run spectroflat first to produce offset maps
    if data_type == 'flat_center':
        print('\nRemoving outdated auxiliary files...')
        _remove_outdated_auxiliary_files(config, data_type)
        
        file_set = config.dataset[data_type]['files']
        if not hasattr(file_set, 'auxiliary'):
            file_set.auxiliary = {}
        
        line = config.dataset['line']
        seq = config.dataset[data_type]['sequence']
        seq_str = f"t{seq:03d}"
        
        # Check/run spectroflat for each half
        for frame_name in ['upper', 'lower']:
            offset_map_file = file_set.auxiliary.get(f'offset_map_{frame_name}')
            skip_spectroflat = False
            
            if offset_map_file and Path(offset_map_file).exists():
                print(f'\n  Offset map exists for {frame_name}: {Path(offset_map_file).name}')
                while True:
                    user_choice = input(f'  Re-run spectroflat for {frame_name}? [y/n]: ').strip().lower()
                    if user_choice in ['y', 'n']:
                        break
                    print('  Please enter "y" or "n"')
                if user_choice == 'n':
                    skip_spectroflat = True
            
            if not skip_spectroflat:
                # Load L2 data for spectroflat
                l2_data_sf, _ = tio.read_any_file(config, data_type, status='l2', verbose=False)
                frame_2d = l2_data_sf.get(0).get_half(frame_name).data.astype('float32')
                
                result = _process_single_frame_spectroflat(frame_2d, frame_name, config, data_type)
                if result is None:
                    print(f'\n  L3 reduction failed for {frame_name} (spectroflat step).')
                    return None
                
                # Save offset map
                offset_map_path = Path(config.directories.reduced) / f'{line}_{data_type}_{seq_str}_offset_map_{frame_name}.fits'
                offset_hdu = fits.PrimaryHDU(data=result['offset_map'])
                offset_hdu.header['FRAME'] = frame_name
                offset_hdu.header['REDLEV'] = 'L3'
                offset_hdu.writeto(str(offset_map_path), overwrite=True)
                file_set.auxiliary[f'offset_map_{frame_name}'] = offset_map_path
                print(f'  ✓ Saved offset map: {offset_map_path.name}')
                
                # Save illumination pattern
                illum_path = Path(config.directories.reduced) / f'{line}_{data_type}_{seq_str}_illumination_pattern_{frame_name}.fits'
                illum_hdu = fits.PrimaryHDU(data=result['illumination_pattern'])
                illum_hdu.header['FRAME'] = frame_name
                illum_hdu.header['REDLEV'] = 'L3'
                illum_hdu.writeto(str(illum_path), overwrite=True)
                file_set.auxiliary[f'illumination_pattern_{frame_name}'] = illum_path
                print(f'  ✓ Saved illumination pattern: {illum_path.name}')
    
    # Load offset maps (from flat_center for both flat and flat_center)
    print('\nLoading offset maps...')
    offset_source = 'flat_center'
    files_src = config.dataset.get(offset_source, {}).get('files')
    if files_src is None:
        print(f'  Error: {offset_source} files not found in config')
        return None
    
    aux_src = getattr(files_src, 'auxiliary', {})
    offset_maps = {}
    for frame_name in ['upper', 'lower']:
        path = aux_src.get(f'offset_map_{frame_name}')
        if path is None or not Path(path).exists():
            print(f'  ✗ WARNING: No offset map found for {frame_name}.')
            print(f'    Run flat_center L2->L3 first to generate offset maps.')
            return None
        with fits.open(path) as hdul:
            offset_maps[frame_name] = np.array(hdul[0].data)
        print(f'  Loaded: {Path(path).name}')
    
    # Load L2 data and apply desmiling
    print(f'\nLoading L2 {data_type} data and applying desmiling...')
    l2_data, header = tio.read_any_file(config, data_type, status='l2', verbose=False)
    
    reduced_frames = None
    
    if data_type == 'scan':
        # Scan is a CycleSet with multiple frames
        if not isinstance(l2_data, dct.CycleSet):
            print(f'  Error: Expected L2 scan data to be a CycleSet')
            return None
        
        reduced_frames = dct.CycleSet()
        items_iter = tqdm(l2_data.items(), desc='L2→L3 scan desmiling')
        
        for key, frame in items_iter:
            dest = dct.Frame(f"scan_l3_{frame.name}")
            for half in ['upper', 'lower']:
                half_obj = frame.get_half(half)
                frame_2d = half_obj.data.astype('float32')
                desmiled = _apply_desmiling(frame_2d, offset_maps[half])
                if half_obj.pol_state:
                    dest.set_half(half, desmiled.astype('float32'), half_obj.pol_state)
                else:
                    dest.set_half(half, desmiled.astype('float32'))
            reduced_frames.add_frame(dest, key)
        
        print(f'  ✓ Applied desmiling to scan frames')
    
    else:
        # flat/flat_center is a FramesSet with single frame
        if not isinstance(l2_data, dct.FramesSet):
            print(f'  Error: Expected L2 {data_type} data to be a FramesSet')
            return None
        
        l2_frame = l2_data.get(0)
        if l2_frame is None:
            print(f'  Error: No L2 frame found for {data_type}')
            return None
        
        reduced_frames = dct.FramesSet()
        dest = dct.Frame(f"{data_type}_l3_frame0000")
        
        for frame_name in ['upper', 'lower']:
            frame_2d = l2_frame.get_half(frame_name).data.astype('float32')
            desmiled = _apply_desmiling(frame_2d, offset_maps[frame_name])
            dest.set_half(frame_name, desmiled.astype('float32'))
            print(f'  ✓ Applied desmiling to {frame_name}')
        
        reduced_frames.add_frame(dest, 0)
    
    extra_keywords = {'DESMILED': ('TRUE', 'L3 desmiling applied using offset maps')}
    
    if return_reduced:
        return reduced_frames
    
    out_path = tio.save_reduction(
        config,
        data_type=data_type,
        level='l3',
        frames=reduced_frames,
        source_header=header,
        verbose=True,
        overwrite=True,
        extra_keywords=extra_keywords,
    )
    
    print(f'✓ L3 {data_type} file saved')
    
    # ============================================================
    # ATLAS-FIT WAVELENGTH CALIBRATION (flat_center only, after desmiling)
    # ============================================================
    if data_type == 'flat_center':
        print(f'\n{"="*70}')
        print(f'ATLAS-FIT WAVELENGTH CALIBRATION FOR FLAT_CENTER')
        print(f'{"="*70}')
        
        file_set = config.dataset[data_type]['files']
        if not hasattr(file_set, 'auxiliary'):
            file_set.auxiliary = {}
        
        line = config.dataset['line']
        seq = config.dataset[data_type]['sequence']
        
        atlas_lines_files = {}
        
        for frame_name in ['upper', 'lower']:
            print(f'\n{"-"*70}')
            print(f'ATLAS-FIT: {frame_name.upper()} FRAME')
            print(f'{"-"*70}')
            
            # Check if atlas lines file already exists
            atlas_lines_file = file_set.auxiliary.get(f'atlas_lines_{frame_name}')
            
            skip_atlas_fit = False
            if atlas_lines_file and Path(atlas_lines_file).exists():
                print(f'  Atlas lines file already exists: {Path(atlas_lines_file).name}')
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
                    print(f'  Deleting old atlas lines file: {Path(atlas_lines_file).name}')
                    Path(atlas_lines_file).unlink()
            
            if not skip_atlas_fit:
                # Run atlas-fit for this frame
                atlas_result = _process_single_frame_atlas_fit(config, data_type, frame_name)
                if not atlas_result:
                    print(f'\n✗ Atlas-fit failed for {frame_name}.')
                    print(f'  L3 file was saved successfully, but wavelength calibration was not completed.')
                    # Don't return None here - L3 was already saved successfully
                else:
                    atlas_lines_files[frame_name] = atlas_result
                    print(f'  ✓ Generated atlas lines file for {frame_name}')
        
        if len(atlas_lines_files) == 2:
            print(f'\n✓ Atlas-fit wavelength calibration complete for flat_center')
            
            # Generate combined wavelength calibration file
            print('\nGenerating combined wavelength calibration file...')
            wl_calib_path = _generate_wavelength_calibration(config, data_type)
            if wl_calib_path:
                print(f'✓ Wavelength calibration file generated')
            else:
                print('✗ Failed to generate wavelength calibration file')
        elif atlas_lines_files:
            print(f'\n⚠ Atlas-fit completed for {list(atlas_lines_files.keys())} only')
            print('  Cannot generate wavelength calibration without both upper and lower atlas lines')
    
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

reduction_levels.add(ReductionLevel("l3", "_l3.fits", reduce_l2_to_l3, {
    "dark": "Nothing.",
    "scan": "Applies offset map (desmiling) from flat_center to produce smile-corrected scan.",
    "flat": "Applies offset map (desmiling) from flat_center to produce smile-corrected flat field.",
    "flat_center": "Produces offset map and illumination pattern using spectroflat. Applies desmiling to image data. \
    Runs atlas-fit wavelength calibration on desmiled L3 data. Saves the wavelength calibration file."
}))

