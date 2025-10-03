from astropy.io import fits 
import numpy as np

import gc
import warnings

from themis.core import themis_data_reduction as tdr
from themis.core import data_classes as dct
from themis.core import themis_tools as tt
from datetime import datetime

from pathlib import Path

from astropy.utils.exceptions import AstropyWarning
import warnings
warnings.simplefilter('ignore', AstropyWarning)

def _build_output_path(config, *, data_type: str, level: str) -> Path:
    """Construct the output file path following discovery conventions.

    Name must contain camera extension, data type extension, sequence, and level suffix
    so that datasets._build_file_set can discover it later.
    """
    cam_ext = config.cam.file_ext
    data_ext = config.data_types[data_type].file_ext
    seq = int(config.dataset[data_type]['sequence'])
    seq_str = f"t{seq:03d}"
    level_ext = config.reduction_levels[level].file_ext

    # Compose filename: 
    fname = f"{config.dataset['line']}_{data_type}_{seq_str}{level_ext}"
    out_dir = config.directories.reduced
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / fname

def _sanitize_primary_header(hdr: fits.Header) -> fits.Header:
    """Remove extension-related keywords that may cause the primary HDU to be
    interpreted as a named extension (e.g., SPECTRA).

    This keeps only appropriate primary keywords and drops extension descriptors
    like EXTNAME, XTENSION, HDUCLAS*, HDUVERS, EXTVER, etc., if present
    in the source header.
    """
    to_drop = [
        'EXTNAME', 'XTENSION', 'EXTVER', 'HDUCLAS1', 'HDUCLAS2', 'HDUCLAS3', 'HDUCLAS4', 'HDUVERS',
        'TFIELDS', 'PCOUNT', 'GCOUNT', 'FILTERCH', 'FILTERFE', 'TIMESTEP', 'WAVEUNIT', 'ATMOS_R0', 'AO_LOCK',
        'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z', 'TEXPOSUR', 'NSUMEXP', 'TCYCLE'
    ]
    for key in to_drop:
        if key in hdr:
            del hdr[key]

    # Drop common table column descriptors if present (TTYPE*, TUNIT*, TFORM*, TDIM*)
    for k in list(hdr.keys()):
        if k.startswith('TTYPE') or k.startswith('TUNIT') or k.startswith('TFORM') or k.startswith('TDIM'):
            try:
                del hdr[k]
            except Exception:
                pass

    # Ensure an empty primary HDU: reset NAXIS and remove NAXISn cards
    if 'NAXIS' in hdr and hdr['NAXIS'] != 0:
        hdr['NAXIS'] = 0
    # Remove any dimension descriptors
    for n in range(1, 10):
        key = f'NAXIS{n}'
        if key in hdr:
            del hdr[key]

    # BITPIX is fine to keep; if missing, set to 8 (character data) as safe default for empty primary
    if 'BITPIX' not in hdr:
        hdr['BITPIX'] = 8
    return hdr


def save_reduction(config, *, data_type: str, level: str, frames: dct.FramesSet, source_header, verbose: bool = False, overwrite: bool = False, extra_keywords: dict = None):
    """Save a reduced product to FITS, update config, and mirror read_any_file conventions.

    Args:
        config: Current configuration object
        data_type: e.g., 'dark', 'flat', 'scan'
        level: e.g., 'l0'
        frames: A FramesSet containing the reduced output frames
        source_header: FITS header to copy/augment
        verbose: Print informative messages
        overwrite: Allow overwriting existing files (emits a warning)
        extra_keywords: Optional dict of additional FITS header keywords to add.
                       Format: {key: (value, comment)} or {key: value}

    Returns:
        The updated config (same object, modified in place)
    """
    gc.collect()

    out_path = _build_output_path(config, data_type=data_type, level=level)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out_path}. Pass overwrite=True to replace.")
    if out_path.exists() and overwrite:
        print(f"\033[91mWARNING\033[0m: Overwriting {out_path.name}")

    # Copy header, sanitize for primary, and add minimal provenance
    hdr = fits.Header(source_header)
    hdr = _sanitize_primary_header(hdr)
    now = datetime.utcnow().isoformat()
    hdr.add_history(f"{now} - Reduced to level {level} for data_type '{data_type}'")
    hdr['REDLEV'] = (level, 'Reduction level')
    hdr['DATATYPE'] = (data_type, 'Data type reduced')
    
    # Add extra keywords if provided
    if extra_keywords:
        for key, value in extra_keywords.items():
            if isinstance(value, tuple) and len(value) == 2:
                # Format: (value, comment)
                hdr[key] = value
            else:
                # Just a value, no comment
                hdr[key] = value

    # Write data_type + level specific HDUs
    hdus = [fits.PrimaryHDU(header=hdr)]

    if data_type == 'dark' and level == 'l0':
        # Expect FramesSet with indices {0: averaged, 1: denoised}
        avg = frames.get(0)

        if avg is None:
            raise ValueError("Expected reduced dark frames to contain index 0 (averaged)")

        upper_avg = avg.get_half('upper').data.astype('float32', copy=False)
        lower_avg = avg.get_half('lower').data.astype('float32', copy=False)
        hdus.append(fits.ImageHDU(upper_avg, name='UPPER_AVG'))
        hdus.append(fits.ImageHDU(lower_avg, name='LOWER_AVG'))

    elif data_type == 'flat' and level == 'l0':
        # Expect FramesSet with index {0: averaged flat}
        avg = frames.get(0)

        if avg is None:
            raise ValueError("Expected reduced flat frames to contain index 0 (averaged)")

        upper_avg = avg.get_half('upper').data.astype('float32', copy=False)
        lower_avg = avg.get_half('lower').data.astype('float32', copy=False)
        hdus.append(fits.ImageHDU(upper_avg, name='UPPER_FLAT'))
        hdus.append(fits.ImageHDU(lower_avg, name='LOWER_FLAT'))

    else:
        # Placeholder for other data types/levels
        raise NotImplementedError(f"Saving not implemented for data_type='{data_type}', level='{level}'")

    # Write FITS
    fits.HDUList(hdus).writeto(out_path, overwrite=True)

    # Update config: add/replace the file reference for this level
    fileset = config.dataset[data_type]['files']
    # 'files' is a FileSet with add(level_name, file_path)
    fileset.add(level, out_path)

    if verbose:
        print(f"Saved {data_type} at level '{level}' to: {out_path}")
        print("HDUList content:")
        for h in hdus:
            print(f"  - {h.name if hasattr(h, 'name') else 'PRIMARY'}: shape={getattr(h, 'data', None).shape if hasattr(h, 'data') and h.data is not None else None}")

    del hdus
    gc.collect()
    return config




def read_any_file(config, data_type, status='raw', verbose=False):  
    """
    Reads a FITS file and populates a CycleSet with Frames for all defined
    polarization states.

    Args:
        config (init): An initialized configuration object.
        data_type (str): The key for the dataset entry (e.g., 'science_data').
        status (str): The reduction level (e.g., 'raw', 'reduced').
        verbose (bool): If True, print verbose output.

    Returns:
        tuple: (collection, header) - A CycleSet or FramesSet containing frames,
               and the FITS header.
    """
    gc.collect()
    
    file_path = config.dataset[data_type]['files'].get(status)
    if not file_path or not file_path.exists():
        raise FileNotFoundError(f"File for data_type '{data_type}' at status '{status}' not found: {file_path}")

    hdu = fits.open(file_path, memmap=True, do_not_scale_image_data=True) # use original so you can identify bad pixels!
    
    header = hdu[0].header
    
    num_slit_positions = header.get('NBSTEP', 0) + 1 if 'NBSTEP' in header else 1
    data = np.array(hdu[0].data)
    
    collection = None
    
    if status == 'raw':  
        if data_type == 'scan':  
            # Original scan handling: build CycleSet keyed by (pol_state, slit_idx, map_idx)
            collection = dct.CycleSet()
            num_pol_states = len(config.polarization_states)
            total_frames_in_file = data.shape[0]
            if total_frames_in_file % (num_pol_states * num_slit_positions) != 0:
                raise ValueError(
                    f"Raw data dimension mismatch. Expected total frames to be a multiple of "
                    f"({config.polarization_states} states * {num_slit_positions} slit_positions). "
                    f"Got {total_frames_in_file} frames."
                )
            
            num_maps = total_frames_in_file // (num_pol_states * num_slit_positions)

            if verbose:
                print(f"Detected {config.polarization_states} polarization states, {num_slit_positions} slit positions, {num_maps} maps.")

            for map_idx in range(num_maps):
                for s_idx in range(num_slit_positions): # Iterate by index instead of name
                    for p_idx, pol_state in enumerate(config.polarization_states):
                        frame_index_in_data_d = (map_idx * num_slit_positions * num_pol_states) + \
                                                (s_idx * num_pol_states) + \
                                                 p_idx
                        current_frame_data_3d = data[frame_index_in_data_d]
                        r1_data, r2_data = config.cam.roi.extract(current_frame_data_3d)

                        frame_name_str = f"{pol_state}_slit{s_idx:02d}_map{map_idx:02d}"  # Name using index
                        single_frame = dct.Frame(frame_name_str)
                        upper_name, lower_name = tt.get_pol_half_names(pol_state)

                        # Keep current orientation choice (adjust if needed)
                        r1_data_flipped = np.flip(r1_data, axis=(-1)) # flip x axis if desired
                        r2_data_flipped = np.flip(r2_data, axis=(-1)) # flip x axis if desired

                        single_frame.set_half("upper", r1_data_flipped, upper_name)
                        single_frame.set_half("lower", r2_data_flipped, lower_name)
                        collection.add_frame(single_frame, (pol_state, s_idx, map_idx))

        elif data_type in ('flat', 'dark'):
            # Generic frames (no states/slit/map semantics): use FramesSet keyed by frame_idx
            total_frames_in_file = data.shape[0]
            collection = dct.FramesSet()
            if verbose:
                print(f"Detected {total_frames_in_file} frames in file for '{data_type}'.")

            for frame_idx in range(total_frames_in_file):
                current_frame_data_3d = data[frame_idx]
                # Apply camera ROI to split into upper/lower halves
                r1_data, r2_data = config.cam.roi.extract(current_frame_data_3d)
                # Orientation: mirror choice from scan branch unless you decide to flip here
                r1_data_flipped = np.flip(r1_data, axis=(-1)) # flip x axis if desired
                r2_data_flipped = np.flip(r2_data, axis=(-1)) # flip x axis if desired
                frame_name_str = f"{data_type}_frame{frame_idx:04d}"
                single_frame = dct.Frame(frame_name_str)
                single_frame.set_half("upper", r1_data_flipped)  # no pol_state
                single_frame.set_half("lower", r2_data_flipped)  # no pol_state
                collection.add_frame(single_frame, frame_idx)
        else:
            # Fallback: treat as single full-data frame in a FramesSet
            collection = dct.FramesSet()
            single_frame = dct.Frame(f"{data_type}_full")
            single_frame.set_half("full_data", np.array(hdu[0].data))
            collection.add_frame(single_frame, 0)
    elif status == 'l0' and data_type == 'dark':
        # Read reduced dark created by save_reduction: primary header only, image HDUs for data
        collection = dct.FramesSet()

        # Find expected HDUs by name
        hdu_names = {h.name.upper(): idx for idx, h in enumerate(hdu)}

        # Averaged frames (required)
        try:
            upper_avg = np.array(hdu[hdu_names['UPPER_AVG']].data)
            lower_avg = np.array(hdu[hdu_names['LOWER_AVG']].data)
        except KeyError as e:
            raise ValueError("Missing UPPER_AVG/LOWER_AVG HDUs in l0 dark file") from e

        frame_name_str = f"{data_type}_l0_frame{0:04d}"
        single_frame = dct.Frame(frame_name_str)
        single_frame.set_half("upper", upper_avg)
        single_frame.set_half("lower", lower_avg)
        collection.add_frame(single_frame, 0)

    else:
        # Non-raw status fallback
        collection = dct.FramesSet()
        single_frame = dct.Frame(f"{data_type}_full")
        single_frame.set_half("full_data", np.array(hdu[0].data))
        collection.add_frame(single_frame, 0)


    if verbose:
        print(f'Reading a {config.data_types[data_type].name} file with reduction status {status}')
        print(collection) 
        print("Header:")
        print(header)

    hdu.close()
    del data
    gc.collect()
    return collection, header


# NON THEMIS DATA IO
# GLOBAL VARIABLES

directory_atlas = '/home/zeuner/data/atlas' # on x1

def read_fts5(wl_start, wl_end): # wavelenght start and end in Angstroem

     # -- fts data
    file_fts5 = directory_atlas+'/fts5.sav'
        
    s =  readsav(file_fts5 , verbose=False,python_dict=True)

    wavelength=s['w']      # wavelength (Å)
    fts_i=s['b'] 	       # I/Ic
    fts_v=s['vi'] 	       # Stokes V/Ic (unsmoothed, recommended)

    s_idx =  np.argmin(abs(wavelength-wl_start))  # in Angstrom
    e_idx = np.argmin(abs(wavelength-wl_end))     # in Angstrom

    wavelength = wavelength[s_idx-1:e_idx]
    fts_i = fts_i[s_idx-1:e_idx]
    fts_v = fts_v[s_idx-1:e_idx]
    
    print("read_fts: Only working for start wavelength below 6906 and above 5253.8")
    return(wavelength, fts_i, fts_v)

def dummy_binning(data, binning=[1,1]):
    kernel = np.ones((binning[0],binning[1]))/(binning[0]*binning[1])
    data_dummy = 0.*data
    for i in range(data.shape[2]):
     conv_data = 1.*convolve2d(data[:,:,i], kernel, 'valid')
     data_dummy[:conv_data.shape[0],:conv_data.shape[1],i] = conv_data
    return(data_dummy)

# INVERSION

def readpro(filename):

    """ 
    Reads a line profile from a .per file
    Call:
    line_ind, wvlen, StkI, StkQ, StkU, StkV = st.readpro(filename)
    """
    
    from numpy import array

    f = open(filename, 'r')

    line_ind = []
    wvlen = []
    StkI = []
    StkQ = []
    StkU = []
    StkV = []
    
    for line in f:
        data = line.split()
        line_ind.append(float(data[0]))
        wvlen.append(float(data[1]))
        StkI.append(float(data[2]))
        StkQ.append(float(data[3]))
        StkU.append(float(data[4]))
        StkV.append(float(data[5]))

    f.close()

    line_ind = array(line_ind)
    wvlen = array(wvlen)
    StkI = array(StkI)
    StkQ = array(StkQ)
    StkU = array(StkU)
    StkV = array(StkV)

    return(line_ind, wvlen, StkI, StkQ, StkU, StkV)



def writepro(filename, line_ind, wvlen, StkI, StkQ, StkU, StkV):
    """ 
    Routine that writes the Stokes profiles into a SIR formatted Stokes file.
    Call:
    writepro(filename, line_ind, wvlen, StkI, StkQ, StkU, StkV)
    """

    f = open(filename, "w+")

    for k in range(0, len(line_ind)):
     
         f.write('     {0}   {1:> .4f}  {2:> 8.6e} {3:> 8.6e} {4:> 8.6e} {5:> 8.6e} \n'.format(line_ind[k], wvlen[k], StkI[k], StkQ[k], StkU[k], StkV[k]))

    f.close()

    return()