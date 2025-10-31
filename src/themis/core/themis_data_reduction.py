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
    
    For flat_center: performs wavelength calibration using atlas-fit bin/prepare.
    """
    import subprocess
    import tempfile
    import os
    import shutil
    from pathlib import Path
    
    if data_type==None:
        print('No processing - provide a specific data type.')
        return None
    
    else:
        data, header = tio.read_any_file(config, data_type, verbose=False, status='l0')
        
        if data_type == 'dark':
            print('No reduction procedure defined for dark l0->l1.')
            return None
        
        elif data_type == 'flat':
            print('Flat l0->l1 not yet implemented.')
            return None
        
        elif data_type == 'scan':
            print('Scan l0->l1 not yet implemented.')
            return None
        
        elif data_type == 'flat_center':
            print('Performing wavelength calibration on flat_center using atlas-fit...')
            
            # Check if atlas-fit config is available
            if not hasattr(config.cam, 'atlas_fit_config') or config.cam.atlas_fit_config is None:
                print('Error: No atlas_fit_config found for this camera.')
                return None
            
            atlas_config_path = config.cam.atlas_fit_config
            if not Path(atlas_config_path).exists():
                print(f'Error: Atlas fit config file not found: {atlas_config_path}')
                return None
            
            # Get the l0 flat_center file path
            flat_center_l0_file = config.dataset['flat_center']['files'].get('l0')
            if flat_center_l0_file is None or not flat_center_l0_file.exists():
                print('Error: L0 flat_center file not found.')
                return None
            
            # Find atlas-fit prepare script
            project_root = Path(__file__).resolve().parents[3]
            prepare_script = project_root / 'atlas-fit' / 'bin' / 'prepare'
            if not prepare_script.exists():
                print(f'Error: Atlas-fit prepare script not found: {prepare_script}')
                return None
            
            # Read the original config as text to preserve formatting and comments
            import yaml
            import re
            
            with open(atlas_config_path, 'r') as f:
                original_config_text = f.read()
            
            # Extract the original corrected_frame path
            match = re.search(r'corrected_frame:\s*(.+)', original_config_text)
            if match:
                original_input_file = match.group(1).strip()
            else:
                print('Error: Could not find corrected_frame in config file')
                return None
            
            # Process both frames (upper and lower)
            reduced_frames = dct.FramesSet()
            dispersion_files = {}
            lines_files = {}
            
            # Get base filename from flat_center L0 file (without level extension)
            base_filename = flat_center_l0_file.stem.replace('_l0', '')
            
            for frame_idx, frame_name in enumerate(['upper', 'lower']):
                print(f'Processing {frame_name} frame...')
                
                # Extract individual frame to temporary file
                temp_frame_path = config.directories.reduced / f'temp_flat_center_{frame_name}_for_atlas.fits'
                
                # Extract the specific frame from the L0 flat_center file
                try:
                    from astropy.io import fits
                    from themis.core import themis_io as tio
                    
                    # Read the flat_center data properly using themis_io
                    flat_data, flat_header = tio.read_any_file(config, 'flat_center', verbose=False, status='l0')
                    
                    # Get the frame data from FramesSet structure
                    # flat_center L0 has one frame (index 0) with upper and lower halves
                    if len(flat_data) > 0:
                        if frame_name == 'upper':
                            frame_data = flat_data[0]['upper'].data
                        elif frame_name == 'lower':
                            frame_data = flat_data[0]['lower'].data
                        else:
                            print(f'Unknown frame name: {frame_name}')
                            continue
                        
                        # Create a simple FITS file with the 2D frame data
                        # The ROI in atlas-fit config will extract 1D spectral data
                        hdu = fits.PrimaryHDU(data=frame_data)
                        hdu.writeto(temp_frame_path, overwrite=True)
                    else:
                        print(f'No frames found in flat_center data')
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
                
                # Write the modified config (only change the corrected_frame line)
                with open(atlas_config_path, 'w') as f:
                    f.write(modified_config_text)
                
                # Run atlas-fit prepare with interactive mode (no capture_output)
                print(f'Running atlas-fit prepare on {frame_name} frame...')
                print('Note: An interactive matplotlib window should open for line selection.')
                try:
                    result = subprocess.run(
                        [str(prepare_script), str(atlas_config_path)],
                        cwd=config.directories.reduced,
                        timeout=300  # 5 minute timeout
                    )
                    
                    if result.returncode != 0:
                        print(f'Error running atlas-fit prepare for {frame_name}:')
                        if result.stderr:
                            print(f'STDERR: {result.stderr}')
                        continue
                    
                    print(f'Atlas-fit prepare completed for {frame_name} frame.')
                    
                    # Create meaningful output filenames using base filename
                    dispersion_file = config.directories.reduced / f'{base_filename}_{frame_name}_dispersion.txt'
                    lines_file = config.directories.reduced / f'{base_filename}_{frame_name}_atlas_lines.yaml'
                    
                    # Move/rename output files if they exist
                    temp_dispersion = config.directories.reduced / 'dispersion.txt'
                    temp_lines = config.directories.reduced / 'atlas_fit_lines.yaml'
                    
                    if temp_dispersion.exists():
                        shutil.move(str(temp_dispersion), str(dispersion_file))
                        dispersion_files[frame_name] = dispersion_file.name
                        print(f'Dispersion function saved to: {dispersion_file}')
                    
                    if temp_lines.exists():
                        shutil.move(str(temp_lines), str(lines_file))
                        lines_files[frame_name] = lines_file.name
                        print(f'Atlas lines saved to: {lines_file}')
                    
                except subprocess.TimeoutExpired:
                    print(f'Error: Atlas-fit prepare timed out for {frame_name} frame.')
                    continue
                except Exception as e:
                    print(f'Error running atlas-fit prepare for {frame_name}: {e}')
                    continue
                
                finally:
                    # Clean up temporary frame file
                    if temp_frame_path.exists():
                        temp_frame_path.unlink()
            
            # Restore original config file (restore the original corrected_frame line)
            restored_config_text = re.sub(
                r'corrected_frame:\s*(.+)',
                f'corrected_frame: {original_input_file}',
                modified_config_text
            )
            with open(atlas_config_path, 'w') as f:
                f.write(restored_config_text)
            
            # Store the original L0 data as L1 (could be enhanced later with calibration data)
            reduced_frames = data
            
            # Create auxiliary files keyword section
            extra_keywords = {}
            aux_files_list = []
            
            for frame_name in ['upper', 'lower']:
                if frame_name in dispersion_files:
                    aux_files_list.append(f'DISP{frame_name.upper()}: {dispersion_files[frame_name]}')
                if frame_name in lines_files:
                    aux_files_list.append(f'ATLAS{frame_name.upper()}: {lines_files[frame_name]}')
            
            # Store as a single keyword with all auxiliary files
            if aux_files_list:
                extra_keywords['AUXL1FILES'] = (
                    '; '.join(aux_files_list),
                    'Auxiliary L1 files: dispersion and atlas lines for upper/lower frames'
                )
        
        else:
            print('Unknown data_type.')
            return None
    
    if return_reduced:
        return reduced_frames
    
    else:
        # Prepare additional header keywords
        extra_keywords = extra_keywords if 'extra_keywords' in locals() else {}
        
        out_path = tio.save_reduction(
            config,
            data_type=data_type,
            level='l1',
            frames=reduced_frames,
            source_header=header,
            verbose=True,
            overwrite=True,  # set True if you want to allow replacing an existing file
            extra_keywords=extra_keywords,
        )
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
    "scan": "Nothing at the moment.",
    "flat": "Nothing at the moment.",
    "flat_center": "Calculate wavelength calibration using atlas-fit bin/prepare on upper and lower frames."
}))

