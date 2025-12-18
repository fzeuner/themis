#!/usr/bin/env python3
"""
Test script to calculate offset between upper and lower frame halves.

This script applies all flat-field corrections and then computes the spatial/spectral
shift between upper and lower frames.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.ndimage import shift as scipy_shift
from scipy.signal import correlate2d

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
# Add atlas-fit to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'atlas-fit' / 'src'))

from themis.datasets.themis_datasets_2025 import get_config
from themis.core import themis_io as tio
from spectroflat.smile import SmileInterpolator, OffsetMap
from spectroflat.utils.processing import MP

from spectator.controllers.app_controller import display_data # from spectator

import imreg_dft as ird

#%%
def apply_desmiling_spectroflat(image, offset_map_array, roi):
    """
    Apply desmiling row-by-row using spectroflat's SmileInterpolator.desmile_row.
    Uses multiprocessing for speed. Only applies desmiling within the ROI region.
    
    Parameters
    ----------
    image : np.ndarray
        2D image [spatial, wavelength] - full frame
    offset_map_array : np.ndarray
        2D offset map array [spatial, wavelength] for this specific state.
    roi : tuple of slices
        Region of interest from offset_map.header['ROI'], e.g.
        (slice(20, 858, None), slice(20, 1230, None))
    
    Returns
    -------
    np.ndarray
        Desmiled image [spatial, wavelength] - full frame with ROI region desmiled
    """
    # Extract ROI from image and offset map
    image_roi = image[roi]
    offset_roi = offset_map_array[roi]
    ny_roi, nx_roi = image_roi.shape
    
    rows = np.arange(ny_roi)
    xes = np.arange(nx_roi)
    
    # Apply desmiling row by row with multiprocessing (only within ROI)
    # Arguments: (row_index, x_coordinates, offsets_for_this_row, data_for_this_row)
    arguments = [(r, xes, offset_roi[r], image_roi[r]) for r in rows]
    res = dict(MP.simultaneous(SmileInterpolator.desmile_row, arguments))
    
    # Reconstruct the desmiled ROI from row results
    desmiled_roi = np.array([res[row] for row in rows])
    
    # Create output array (copy of input) and insert desmiled ROI
    desmiled = image.copy()
    desmiled[roi] = desmiled_roi
    
    return desmiled


def _load_dust_flats(config, data_type):
    with fits.open(config.dataset[data_type]['files'].auxiliary.get('dust_flat_upper')) as hdu:
        dust_flat_upper = hdu[0].data
    with fits.open(config.dataset[data_type]['files'].auxiliary.get('dust_flat_lower')) as hdu:
        dust_flat_lower = hdu[0].data
    return dust_flat_upper, dust_flat_lower


def _load_outdated_offset_maps(config, data_type):
    offset_map_upper_outdated_file = config.dataset[data_type]['files'].auxiliary.get('offset_map_upper_outdated')
    offset_map_lower_outdated_file = config.dataset[data_type]['files'].auxiliary.get('offset_map_lower_outdated')

    offset_map_upper_outdated = OffsetMap.from_file(offset_map_upper_outdated_file)
    offset_map_lower_outdated = OffsetMap.from_file(offset_map_lower_outdated_file)

    offset_upper_outdated_2d = offset_map_upper_outdated.map[0].copy()
    offset_lower_outdated_2d = offset_map_lower_outdated.map[0].copy()
    roi_upper = eval(offset_map_upper_outdated.header['ROI'])
    roi_lower = eval(offset_map_lower_outdated.header['ROI'])

    del offset_map_upper_outdated, offset_map_lower_outdated

    return {
        'offset_upper_outdated_2d': offset_upper_outdated_2d,
        'offset_lower_outdated_2d': offset_lower_outdated_2d,
        'roi_upper': roi_upper,
        'roi_lower': roi_lower,
    }


def _normalize_for_alignment(img, margin):
    if margin <= 0:
        return img / np.mean(img)
    return (img / np.mean(img))[margin:-margin, margin:-margin]


def _compute_alignment_transform(upper_img, lower_img, alignment_cfg):
    margin = int(alignment_cfg.get('margin', 50))
    upper_norm = _normalize_for_alignment(upper_img, margin)
    lower_norm = _normalize_for_alignment(lower_img, margin)

    constraints = alignment_cfg.get('constraints')
    numiter = int(alignment_cfg.get('numiter', 5))

    kwargs = {'numiter': numiter}
    if constraints is not None:
        kwargs['constraints'] = constraints

    result = ird.similarity(upper_norm, lower_norm, **kwargs)
    return result['tvec'], result['angle'], result['scale']


def _apply_alignment(lower_img, tvec, angle, scale):
    return ird.transform_img(lower_img, tvec=tvec, angle=angle, scale=scale)


def _processing_title_suffix(processing):
    steps = []
    if processing.get('final_use_dust_flat', True):
        steps.append('dust')
    desmile_mode = processing.get('final_desmile')
    if desmile_mode is not None:
        steps.append(f'desmile({desmile_mode})')
    if processing.get('final_align_lower_to_upper', False):
        steps.append('align')
    if not steps:
        return 'none'
    return ', '.join(steps)


def process_flat(config, data_type='flat_center', flat_index=0, processing=None):
    if processing is None:
        processing = {}

    l0_data, _ = tio.read_any_file(config, data_type, verbose=False, status='l0')
    upper_raw = l0_data[flat_index]['upper'].data
    lower_raw = l0_data[flat_index]['lower'].data

    dust_flat_upper, dust_flat_lower = _load_dust_flats(config, data_type)
    upper_dust = upper_raw / dust_flat_upper
    lower_dust = lower_raw / dust_flat_lower

    upper_final = upper_dust if processing.get('final_use_dust_flat', True) else upper_raw
    lower_final = lower_dust if processing.get('final_use_dust_flat', True) else lower_raw

    desmile_mode = processing.get('final_desmile')
    if desmile_mode:
        if desmile_mode is True:
            desmile_mode = 'outdated'
        if desmile_mode != 'outdated':
            raise ValueError(f"Unsupported desmile mode: {desmile_mode!r}")
        om = _load_outdated_offset_maps(config, data_type)
        upper_final = apply_desmiling_spectroflat(upper_final, om['offset_upper_outdated_2d'], om['roi_upper'])
        lower_final = apply_desmiling_spectroflat(lower_final, om['offset_lower_outdated_2d'], om['roi_lower'])

    tvec = (0.0, 0.0)
    angle = 0.0
    scale = 1.0
    if processing.get('final_align_lower_to_upper', False):
        alignment_cfg = processing.get('alignment', {})
        transform = alignment_cfg.get('transform')
        if transform is not None:
            tvec = tuple(transform.get('tvec', tvec))
            angle = float(transform.get('angle', angle))
            scale = float(transform.get('scale', scale))
        else:
            tvec, angle, scale = _compute_alignment_transform(upper_dust, lower_dust, alignment_cfg)

        lower_dust = _apply_alignment(lower_dust, tvec, angle, scale)
        lower_final = _apply_alignment(lower_final, tvec, angle, scale)

    return {
        'upper_dust': upper_dust,
        'lower_dust': lower_dust,
        'upper_final': upper_final,
        'lower_final': lower_final,
        'tvec': tvec,
        'angle': angle,
        'scale': scale,
        'title_suffix': _processing_title_suffix(processing),
    }


def process_scan(config, aux_data_type='flat_center', pol_state='pQ', slit_idx=0, map_idx=0, processing=None):
    if processing is None:
        processing = {}

    l0_scan, _ = tio.read_any_file(config, 'scan', verbose=False, status='l0')
    upper_raw = l0_scan[pol_state, slit_idx, map_idx]['upper'].data
    lower_raw = l0_scan[pol_state, slit_idx, map_idx]['lower'].data

    dust_flat_upper, dust_flat_lower = _load_dust_flats(config, aux_data_type)
    upper_dust = upper_raw / dust_flat_upper
    lower_dust = lower_raw / dust_flat_lower

    upper_final = upper_dust if processing.get('final_use_dust_flat', True) else upper_raw
    lower_final = lower_dust if processing.get('final_use_dust_flat', True) else lower_raw

    desmile_mode = processing.get('final_desmile')
    if desmile_mode:
        if desmile_mode is True:
            desmile_mode = 'outdated'
        if desmile_mode != 'outdated':
            raise ValueError(f"Unsupported desmile mode: {desmile_mode!r}")
        om = _load_outdated_offset_maps(config, aux_data_type)
        upper_final = apply_desmiling_spectroflat(upper_final, om['offset_upper_outdated_2d'], om['roi_upper'])
        lower_final = apply_desmiling_spectroflat(lower_final, om['offset_lower_outdated_2d'], om['roi_lower'])

    tvec = (0.0, 0.0)
    angle = 0.0
    scale = 1.0
    if processing.get('final_align_lower_to_upper', False):
        alignment_cfg = processing.get('alignment', {})
        transform = alignment_cfg.get('transform')
        if transform is not None:
            tvec = tuple(transform.get('tvec', tvec))
            angle = float(transform.get('angle', angle))
            scale = float(transform.get('scale', scale))
        else:
            tvec, angle, scale = _compute_alignment_transform(upper_dust, lower_dust, alignment_cfg)

        lower_dust = _apply_alignment(lower_dust, tvec, angle, scale)
        lower_final = _apply_alignment(lower_final, tvec, angle, scale)

    return {
        'upper_dust': upper_dust,
        'lower_dust': lower_dust,
        'upper_final': upper_final,
        'lower_final': lower_final,
        'tvec': tvec,
        'angle': angle,
        'scale': scale,
        'title_suffix': _processing_title_suffix(processing),
    }


def _format_processing_for_title(processing):
    if not processing:
        return ''
    items = []
    for k in sorted(processing.keys()):
        items.append(f"{k}={processing[k]!r}")
    return ', '.join(items)


def plot_alignment_diff(data, corrected, 
                        tvec, angle, scale, config, margin=50, title_suffix='', processing=None):
    """
    Plot differences between upper and aligned lower frames.
    
    Parameters
    ----------
    upper_data : ndarray
        L0 upper frame
    lower_data_aligned : ndarray
        L0 lower frame after alignment
    corrected_upper : ndarray
        Corrected upper frame
    corrected_lower_aligned : ndarray
        Corrected lower frame after alignment
    tvec : tuple
        Translation vector (y, x)
    angle : float
        Rotation angle in degrees
    scale : float
        Scale factor
    config : Config
        Configuration object
    margin : int
        Margin for percentile calculation
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    from datetime import datetime
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    suffix = f"\nApplied: {title_suffix}" if title_suffix else ''
    processing_str = _format_processing_for_title(processing)
    processing_suffix = f"\nProcessing: {processing_str}" if processing_str else ''
    fig.suptitle(
        f'Translation: ({tvec[0]:.2f}, {tvec[1]:.2f}) px, Rotation: {angle:.3f}°, Scale: {scale:.5f}'
        f'{suffix}{processing_suffix}',
        fontsize=14,
        fontweight='bold'
    )
    
    # Left: L0 difference (upper - aligned lower)
    diff_l0 = data 
    if margin > 0:
        vmax_l0 = np.percentile(np.abs(diff_l0[margin:-margin, margin:-margin]), 99)
    else:
        vmax_l0 = np.percentile(np.abs(diff_l0), 99)
    
    ax = axes[0]
    im = ax.imshow(diff_l0, aspect='auto', cmap='RdBu_r', vmin=-vmax_l0, vmax=vmax_l0)
    ax.set_title('L0: Upper - Lower')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im, ax=ax, label='Difference')
    
    # Right: Corrected difference (corrected_upper - aligned corrected_lower)
    diff_corrected = corrected
    # if margin > 0:
    #     vmax_corr = np.percentile(np.abs(diff_corrected[margin:-margin, margin:-margin]), 99)
    # else:
    #     vmax_corr = np.percentile(np.abs(diff_corrected), 99)
    
    ax = axes[1]
    im = ax.imshow(diff_corrected, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_title('Corrected: Upper - Lower')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im, ax=ax, label='Difference')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figures_dir = Path(config.directories.figures)
    figures_dir.mkdir(exist_ok=True, parents=True)
    output_plot = figures_dir / f'frame_alignment_diff_{timestamp}.png'
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved alignment difference plot: {output_plot}")
    
    plt.show()
    
    return fig

   
#%%

if __name__ == "__main__":
    """Main test function."""
    print("="*70)
    print("Frame Alignment Test: Upper vs Lower")
    print("="*70)
    
    # Load configuration
    config = get_config(
        config_path='configs/sample_dataset_sr_2025-07-07.toml',
        auto_discover_files=True
    )
    
    data_type = 'flat_center'

    processing_flat = {
        'final_use_dust_flat': False,
        'final_desmile': False,
        'final_align_lower_to_upper': False,
    }
    processing_scan = {
        'final_use_dust_flat': True,
        'final_desmile': False,
        'final_align_lower_to_upper': False,
        # "alignment": {
        #     "transform": {
        #  "tvec": (2.7, -8.0),   # (y, x) pixels
        #  "angle": -0.09,        # degrees
        #  "scale": 1.0,}}
    }

    print("\n0. Processing flat...")
    flat_res = process_flat(config, data_type=data_type, flat_index=0, processing=processing_flat)

    print("\n0. Processing scan...")
    scan_res_p = process_scan(config, aux_data_type=data_type, pol_state='pQ', slit_idx=0, map_idx=0, processing=processing_scan)
    
    scan_res_m = process_scan(config, aux_data_type=data_type, pol_state='mQ', slit_idx=0, map_idx=0, processing=processing_scan)
    
    I = 0.5*(scan_res_p['upper_final']+scan_res_m['upper_final'])#+scan_res_p['lower_final']+scan_res_m['lower_final'])
    Q = 100*(scan_res_p['upper_final']-scan_res_m['upper_final'])/I#*scan_res_m['lower_final']/scan_res_p['lower_final']-1)
#%%
    # Step 5: Calculate alignment using imreg_dft on L0 data
    # print("\n5. Calculating alignment transformation using L0 data...")
    
    # # Normalize images to maximum before alignment
    # # Crop edges to avoid border effects
    # margin = 50
    # upper_data_norm = _normalize_for_alignment(flat_res['upper_dust'], margin)
    # lower_data_norm = _normalize_for_alignment(flat_res['lower_dust'], margin)

    # corrected_upper_norm = _normalize_for_alignment(flat_res['upper_final'], margin)
    # corrected_lower_norm = _normalize_for_alignment(flat_res['lower_final'], margin)

    # upper_scan_norm = _normalize_for_alignment(scan_res['upper_final'], margin)
    # lower_scan_norm = _normalize_for_alignment(scan_res['lower_final'], margin)

    # upper_scan_nondesmiled_norm = _normalize_for_alignment(scan_res['upper_dust'], margin)
    # lower_scan_nondesmiled_norm = _normalize_for_alignment(scan_res['lower_dust'], margin)
    
    # # Step 5a: Determine initial large shift in spectral direction (axis 1)
    # # by cross-correlating averaged spectra over central 400 spatial pixels
    # print("   Finding initial spectral shift from averaged spectra...")
    # ny, nx = upper_scan_norm.shape
    # spatial_center = ny // 2
    # spatial_half_width = 200  # 400 pixels total
    
    # # Average over central spatial pixels to get 1D spectra
    # upper_spectrum_avg = np.mean(upper_scan_norm[spatial_center-spatial_half_width:spatial_center+spatial_half_width, 250:-450], axis=0)
    # lower_spectrum_avg = np.mean(lower_scan_norm[spatial_center-spatial_half_width:spatial_center+spatial_half_width,250 :-450], axis=0)
    
    # # Cross-correlate to find spectral shift - does work very well
    # #correlation = np.correlate(upper_spectrum_avg, lower_spectrum_avg, mode='full')
    # #lag = np.arange(-len(lower_spectrum_avg) + 1, len(upper_spectrum_avg))
    # # hardcoded!!!
    # initial_shift_x = -8#lag[np.argmax(correlation)] #-8#
    # print(f"   ✓ Initial spectral shift (x): {initial_shift_x} px - HARDCODED")
    
    # # Pre-shift lower images by initial spectral shift
    # from scipy.ndimage import shift as scipy_shift
    # lower_scan_norm_preshifted = scipy_shift(lower_scan_norm, (0, initial_shift_x), order=3, mode='constant')
    
    # lower_data_norm_preshifted = scipy_shift(lower_data_norm, (0, initial_shift_x), order=3, mode='constant')
    # corrected_lower_norm_preshifted = scipy_shift(corrected_lower_norm, (0, initial_shift_x), order=3, mode='constant')
    
    # # Use imreg_dft to find residual transformation (on pre-shifted data)
    # print("   Finding residual transformation with imreg_dft...")
    # result = ird.similarity(
    #     upper_scan_nondesmiled_norm, 
    #     lower_scan_nondesmiled_norm, # preshifted for better results
    #     numiter=5,
    #     constraints={
    #     'tx': (-1,1),      # x translation bounds
    #     'ty': (-2, 2),      # y translation bounds  
    #     'angle': (-0.0001,0.0001),     # rotation bounds in degrees
    #     'scale': (0.99999, 1.00001),  # scale bounds
    #     }
    # )
    
    # # Extract transformation parameters and add initial shift
    # tvec = result['tvec']  # Translation vector (y, x)
    # angle = result['angle']  # Rotation angle
    # scale = result['scale']  # Scale factor
    
    # # Total shift = initial + residual
    # total_tvec = (tvec[0]+2.7, tvec[1] + initial_shift_x)
    # print(f"   ✓ Residual shift: ({tvec[0]:.3f}, {tvec[1]:.3f}) px")
    # print(f"   ✓ Total transformation:")
    # print(f"     Translation: ({total_tvec[0]:.3f}, {total_tvec[1]:.3f}) px (y, x)")
    # print(f"     Rotation: {angle:.4f}°")
    # print(f"     Scale: {scale:.6f}")
    
    # # Apply full transformation (total_tvec) to original normalized data
    # lower_data_norm_aligned = ird.transform_img(lower_data_norm, tvec=total_tvec, angle=angle, scale=scale)
    
    # # Apply the SAME transformation to normalized corrected_lower
    # corrected_lower_norm_aligned = ird.transform_img(corrected_lower_norm, tvec=total_tvec, angle=angle, scale=scale)
    
    # # Apply the SAME transformation to normalized corrected_lower_scan
    # corrected_lower_norm_scan_aligned = ird.transform_img(lower_scan_norm, tvec=total_tvec, angle=angle, scale=scale)
    
    
    # total_tvec = (1.4,  -8)
    # lower_scan_nondesmiled_aligned = ird.transform_img(lower_scan_nondesmiled_norm,tvec=total_tvec, angle=0, scale=1)
    # print(f"   ✓ Applied transformation to both L0 and corrected lower frames")
  #%%  
    # Step 6: Plot differences
    print("\n6. Plotting alignment results...")
    # flat_plot=plot_alignment_diff(
    #     flat_res['upper_dust']-flat_res['lower_dust'],
    #     flat_res['upper_final']-flat_res['lower_final'],
    #     flat_res['tvec'], flat_res['angle'],flat_res['scale'], config, margin=0, title_suffix=flat_res['title_suffix'], processing=processing_flat
    # )
    
    scan_plot=plot_alignment_diff(
        scan_res_p['upper_dust']- scan_res_m['upper_dust'],
        scan_res_p['upper_final']-scan_res_m['upper_final'],
        scan_res_p['tvec'], scan_res_p['angle'],scan_res_p['scale'], config, margin=0, title_suffix=scan_res_p['title_suffix'], processing=processing_scan
    )
    
    scan_plot_i=plot_alignment_diff(
        I,
        Q,
        scan_res_p['tvec'], scan_res_p['angle'],scan_res_p['scale'], config, margin=0, title_suffix=scan_res_p['title_suffix'], processing=processing_scan
    )
    
    print("\n" + "="*70)
    print("Frame alignment test complete!")
    print("="*70)
    
    
#%%

    # total_tvec = (1,  -8)
    # lower_scan_nondesmiled_aligned = ird.transform_img(lower_scan_nondesmiled_norm,tvec=total_tvec, angle=-0.09, scale=1)
    # t=plot_alignment_diff(
    #     upper_scan_nondesmiled_norm, lower_scan_nondesmiled_aligned,
    #     upper_scan_norm, corrected_lower_norm_scan_aligned,
    #     total_tvec, angle, scale, config, margin=0, title_suffix=scan_res['title_suffix']
    # )
    
    
    from scipy.ndimage import gaussian_filter
    
    test_pu = scan_res_p['upper_final'][55:175,380:512]
    test_mu = scan_res_m['upper_final'][55:175,380:512]
    
    sigma =2.5
    print(np.std(gaussian_filter(test_pu,sigma)-0.998*gaussian_filter(_apply_alignment(test_mu, [0.1,0.3], -0.1, 1),sigma)))
    print(np.std(test_mu))
    #lt.imshow(test_pu-0.998*_apply_alignment(test_mu, [-0.1,0.4], 0, 1))
    plt.imshow( gaussian_filter(test_pu,sigma)-gaussian_filter(0.99*_apply_alignment(test_mu, [0.1,0.3], -0.1, 1),sigma), vmin=-100, vmax=100)