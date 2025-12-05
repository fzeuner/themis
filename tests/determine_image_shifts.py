#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:46:34 2025

@author: franziskaz
"""

#!/usr/bin/env python3
"""
Determine image transformation between upper and lower image.

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.ndimage import shift as scipy_shift
from scipy.signal import correlate2d

from themis.datasets.themis_datasets_2025 import get_config
from themis.core import themis_io as tio
from spectroflat.smile import SmileInterpolator, OffsetMap
from spectroflat.utils.processing import MP

from spectator.controllers.app_controller import display_data # from spectator

import imreg_dft as ird

from themis.core import themis_data_reduction as tdr

#%%

def plot_alignment_diff(upper_data, lower_data_aligned, 
                        tvec, angle, scale, config, margin=50):
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
    
    fig, axes = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle(f'Upper vs Lower Frame Alignment\nTranslation: ({tvec[0]:.2f}, {tvec[1]:.2f}) px, Rotation: {angle:.3f}°, Scale: {scale:.5f}', 
                 fontsize=14, fontweight='bold')
    
    # Left: L0 difference (upper - aligned lower)
    diff_l0 = upper_data - lower_data_aligned
    if margin > 0:
        vmax_l0 = np.percentile(np.abs(diff_l0[margin:-margin, margin:-margin]), 99)
    else:
        vmax_l0 = np.percentile(np.abs(diff_l0), 99)
    
    ax = axes[0]
    im = ax.imshow(diff_l0, aspect='auto', cmap='RdBu_r', vmin=-vmax_l0, vmax=vmax_l0)
    ax.set_title('L0: Upper - Aligned Lower')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im, ax=ax, label='Difference')
    

    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figures_dir = Path(config.directories.figures)
    figures_dir.mkdir(exist_ok=True, parents=True)
    #output_plot = figures_dir / f'frame_alignment_diff_{timestamp}.png'
    #plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    #print(f"\n✓ Saved alignment difference plot: {output_plot}")
    
    plt.show()
    
    return fig



def reduce_all(config):
    lv0 = tdr.reduction_levels["l0"]
    
    data_types = ['dark', 'flat', 'flat_center', 'scan']
    for data_type in data_types:
       result = lv0.reduce(config, data_type=data_type, return_reduced = False)

    # lv1 = tdr.reduction_levels["l1"]
    # result = lv1.reduce(config, data_type='flat_center', return_reduced = False)
    return(0)
   
#%%

if __name__ == "__main__":
    """Main test function."""
    print("="*70)
    print("Calibration data: determine image shifts, rotation, scale")
    print("="*70)
    
    # Load configuration
    config = get_config(
        config_path='configs/calibration_data_sr_2025-07-05.toml',
        auto_discover_files=True
    )
    
    # reduce (only once)
    #result = reduce_all(config)
  #%%  
    scan, header = tio.read_any_file(config, 'scan', verbose=False, status='l0')
    pQ_scan = scan.get_state('pQ')
    upper_scan = pQ_scan.stack_all('upper')
    lower_scan = pQ_scan.stack_all('lower')
    
    with fits.open(config.dataset['flat_center']['files'].auxiliary.get('dust_flat_upper')) as hdu:
        dust_flat_upper = hdu[0].data
    with fits.open(config.dataset['flat_center']['files'].auxiliary.get('dust_flat_lower')) as hdu:
        dust_flat_lower = hdu[0].data
    
#%%
    data_plot = np.array([upper_scan[0],lower_scan[0] ])
    viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
                       title='target scan', 
                      state_names=['upper', 'lower'])
    
#%%
    # Crop edges to avoid border effects
    margin = 50
    upper_scan_norm = (upper_scan[0]/dust_flat_upper / np.mean(upper_scan[0]/dust_flat_lower))[margin:-margin, margin:-margin]
    lower_scan_norm = (lower_scan[0]/dust_flat_lower / np.mean(lower_scan[0]/dust_flat_lower))[margin:-margin, margin:-margin]
    
    # result = ird.similarity(
    #     upper_scan_norm, 
    #     lower_scan_norm, # preshifted for better results
    #     numiter=5,
    #     constraints={
    #     'tx': (-1,1),      # x translation bounds
    #     'ty': (-2, 2),      # y translation bounds  
    #     'angle': (-0.5,0.5),     # rotation bounds in degrees
    #     'scale': (0.9, 1.1),  # scale bounds
    #     }
    # )
    lower_scan_aligned = scipy_shift(lower_scan_norm, (0, -1), order=3, mode='constant')
    lower_scan_aligned_rotated = ird.transform_img(lower_scan_aligned,tvec=[0,0], angle=-0.5, scale=1)
    upper_scan_rotated = ird.transform_img(lower_scan_norm,tvec=[0,0], angle=-0.3, scale=1)
    # print(f"   ✓ Shift: ({result['tvec'][0]:.3f}, {result['tvec'][1]:.3f}) px")
    # print(f"     Rotation: {result['angle']:.4f}°")
    # print(f"     Scale: {result['scale']:.6f}")
   
    
    viewer = display_data( np.array([upper_scan_norm,lower_scan_aligned_rotated, upper_scan_norm-lower_scan_aligned_rotated]), 'states',  'spatial', 'spectral',
                       title='target scan', 
                      state_names=['upper', 'lower aligned', 'upper-lower aligned'])
    