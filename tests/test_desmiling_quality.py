"""
Test the quality of spectroflat desmiling by comparing upper and lower frames
before and after correction.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from spectroflat import OffsetMap
from spectroflat.smile import SmileInterpolator
from themis.datasets.themis_datasets_2025 import get_config
from themis.core import themis_io as tio
from themis.core.themis_data_reduction import _apply_desmiling as apply_desmiling


def test_desmiling_quality(config_path='configs/sample_dataset_sr_2025-07-07.toml'):
    """
    Compare upper and lower flat_center frames before and after desmiling.
    
    Compares desmiling quality using:
    - OUTDATED: Original spectroflat offset maps (pixel-based)
    - AMENDED: Refined offset maps from amend-spectroflat (pixel-based + wavelength metadata)
    
    Both maps use pixel-based offsets and work with SmileInterpolator.
    The amended maps have refined offsets from atlas line fitting.
    """
    # Load configuration
    config = get_config(
        config_path=config_path,
        auto_discover_files=True,
        auto_create_dirs=False
    )
    
    # Read L0 flat_center data
    print("Loading L0 flat_center data...")
    data, header = tio.read_any_file(config, 'flat_center', verbose=False, status='l0')
    
    # Extract upper and lower frames
    upper_original = data[0]['upper'].data  # shape: (spatial, wavelength)
    lower_original = data[0]['lower'].data  # shape: (spatial, wavelength)
    
    # Load BOTH offset maps (outdated and amended)
    print("\nLoading offset maps...")
    offset_map_outdated_file = config.dataset['flat_center']['files'].auxiliary.get('offset_map_outdated')
    offset_map_amended_file = config.dataset['flat_center']['files'].auxiliary.get('offset_map')
    
    has_outdated = offset_map_outdated_file and offset_map_outdated_file.exists()
    has_amended = offset_map_amended_file and offset_map_amended_file.exists()
    
    if not has_amended:
        print("Error: Amended offset map not found. Run spectroflat first!")
        return
    
    # Always process with amended offset map
    from astropy.io import fits
    print(f"  Loading amended (WL-calibrated) offset map: {offset_map_amended_file.name}")
    offset_map_amended = OffsetMap.from_file(offset_map_amended_file)
    offset_map_amended_array = offset_map_amended.get_map()  # shape: (n_states, spatial, wavelength)
    
    # Get header for wavelength calibration info
    with fits.open(offset_map_amended_file) as hdul:
        offset_map_amended_header = hdul[0].header
    
    # Check if amended map has wavelength calibration
    if 'MIN_WL_NM' in offset_map_amended_header:
        print(f"  ✓ Wavelength calibration detected:")
        print(f"    Range: {offset_map_amended_header['MIN_WL_NM']:.3f} - {offset_map_amended_header['MAX_WL_NM']:.3f} nm")
        print(f"    Dispersion: {offset_map_amended_header['DISPERSION']:.6f} nm/px")
    
    # Optionally process with outdated offset map if it exists
    offset_map_outdated_array = None
    offset_map_outdated_header = None
    if has_outdated:
        print(f"  Loading outdated (original) offset map: {offset_map_outdated_file.name}")
        offset_map_outdated = OffsetMap.from_file(offset_map_outdated_file)
        
        # Outdated map may have 4 states - extract states 0 and 1 (upper and lower)
        outdated_full = offset_map_outdated.get_map()
        print(f"  Outdated offset map full shape: {outdated_full.shape}")
        
        if outdated_full.shape[0] > 2:
            # Extract first 2 states to match amended
            offset_map_outdated_array = outdated_full[0:2]
            print(f"  Extracting states 0-1 from outdated map: {offset_map_outdated_array.shape}")
        else:
            offset_map_outdated_array = outdated_full
        
        with fits.open(offset_map_outdated_file) as hdul:
            offset_map_outdated_header = hdul[0].header
    else:
        print("  No outdated offset map found. Will only plot amended results.")
    
    print(f"Amended offset map shape: {offset_map_amended_array.shape}")
    
    # Check if offset map has state dimension
    if offset_map_amended_array.ndim == 2:
        print("Warning: Offset map is 2D (no state dimension). Using same map for both frames.")
        # Duplicate the 2D map for both states
        offset_map_amended_array = np.stack([offset_map_amended_array, offset_map_amended_array], axis=0)
    
    if has_outdated and offset_map_outdated_array.ndim == 2:
        print("Warning: Outdated offset map is 2D (no state dimension). Using same map for both frames.")
        offset_map_outdated_array = np.stack([offset_map_outdated_array, offset_map_outdated_array], axis=0)
    
    # Compare offset maps if both exist
    if has_outdated:
        print("\n" + "="*70)
        print("COMPARING OFFSET MAPS: OUTDATED vs AMENDED")
        print("="*70)
        
        # Compute difference
        offset_diff = offset_map_amended_array - offset_map_outdated_array
        
        for state in range(min(2, offset_diff.shape[0])):
            state_label = 'Upper' if state == 0 else 'Lower'
            rms_diff = np.sqrt(np.mean(offset_diff[state]**2))
            max_diff = np.max(np.abs(offset_diff[state]))
            mean_diff = np.mean(offset_diff[state])
            
            print(f"\nState {state} ({state_label}):")
            print(f"  Mean difference:   {mean_diff:.4f} px")
            print(f"  RMS difference:    {rms_diff:.4f} px")
            print(f"  Max abs difference: {max_diff:.4f} px")
        
        # Check if wavelength calibration info exists
        if 'MIN_WL_NM' in offset_map_amended_header and 'MIN_WL_NM' in offset_map_outdated_header:
            print(f"\nWavelength calibration changes:")
            print(f"  Outdated: {offset_map_outdated_header.get('MIN_WL_NM', 'N/A'):.3f} - {offset_map_outdated_header.get('MAX_WL_NM', 'N/A'):.3f} nm")
            print(f"  Amended:  {offset_map_amended_header['MIN_WL_NM']:.3f} - {offset_map_amended_header['MAX_WL_NM']:.3f} nm")
    
    # Calculate DELTA offsets (like atlas-fit does)
    print("\n" + "="*70)
    print("PROCESSING WITH DELTA OFFSETS (AMENDED - OUTDATED)")
    print("="*70)
    print("NOTE: Atlas-fit applies INCREMENTAL corrections (delta_offsets), not the full")
    print("      amended offset map. We compute delta = amended - outdated and apply it.")
    
    offset_upper_amended = offset_map_amended_array[0]
    offset_lower_amended = offset_map_amended_array[1]
    
    diff_before = upper_original - lower_original
    rms_before = np.sqrt(np.nanmean(diff_before**2))
    
    # Calculate delta offsets if we have outdated maps
    if has_outdated:
        offset_upper_outdated = offset_map_outdated_array[0]
        offset_lower_outdated = offset_map_outdated_array[1]
        
        # Compute incremental corrections (like atlas-fit does)
        delta_upper = offset_upper_amended - offset_upper_outdated
        delta_lower = offset_lower_amended - offset_lower_outdated
        
        print(f"\nDelta offset statistics:")
        print(f"  Upper: mean={np.mean(delta_upper):.4f}, std={np.std(delta_upper):.4f}, max={np.max(np.abs(delta_upper)):.4f} px")
        print(f"  Lower: mean={np.mean(delta_lower):.4f}, std={np.std(delta_lower):.4f}, max={np.max(np.abs(delta_lower)):.4f} px")
        
        # Apply corrections like atlas-fit does:
        # 1. First desmile with outdated offset map
        # 2. Then apply small delta corrections to the already-desmiled data
        print("\nApplying corrections (outdated + delta on desmiled data)...")
        print("  Step 1: Desmile with OUTDATED offset map...")
        upper_desmiled_outdated_first = apply_desmiling(upper_original, offset_upper_outdated, mod_state=0)
        lower_desmiled_outdated_first = apply_desmiling(lower_original, offset_lower_outdated, mod_state=1)
        
        print("  Step 2: Apply small DELTA corrections to desmiled data...")
        upper_desmiled_amended = apply_desmiling(upper_desmiled_outdated_first, delta_upper, mod_state=0)
        lower_desmiled_amended = apply_desmiling(lower_desmiled_outdated_first, delta_lower, mod_state=1)
        diff_after_amended = upper_desmiled_amended - lower_desmiled_amended
        
        rms_after_amended = np.sqrt(np.nanmean(diff_after_amended**2))
        
        print(f"\nRMS difference (upper - lower) with DELTA corrections:")
        print(f"  Before correction: {rms_before:.4f}")
        print(f"  After correction:  {rms_after_amended:.4f}")
        print(f"  Improvement:       {rms_before/rms_after_amended:.2f}x")
    else:
        print("\nNo outdated offset map found - cannot compute delta offsets.")
        upper_desmiled_amended = None
        lower_desmiled_amended = None
        diff_after_amended = None
        rms_after_amended = None
    
    # Process with OUTDATED offset map if available
    upper_desmiled_outdated = None
    lower_desmiled_outdated = None
    diff_after_outdated = None
    rms_after_outdated = None
    
    if has_outdated:
        print("\n" + "="*70)
        print("PROCESSING WITH OUTDATED (ORIGINAL) OFFSET MAP")
        print("="*70)
        
        offset_upper_outdated = offset_map_outdated_array[0]
        offset_lower_outdated = offset_map_outdated_array[1]
        
        print("Applying desmiling with OUTDATED offset map (row-by-row, like atlas-fit)...")
        print(f"  Note: Outdated map has {outdated_full.shape[0]} states, using states 0 and 1")
        upper_desmiled_outdated = apply_desmiling(upper_original, offset_upper_outdated, mod_state=0)
        lower_desmiled_outdated = apply_desmiling(lower_original, offset_lower_outdated, mod_state=1)
        diff_after_outdated = upper_desmiled_outdated - lower_desmiled_outdated
        
        rms_after_outdated = np.sqrt(np.nanmean(diff_after_outdated**2))
        
        print(f"\nRMS difference (upper - lower) with OUTDATED offset map:")
        print(f"  Before desmiling: {rms_before:.4f}")
        print(f"  After desmiling:  {rms_after_outdated:.4f}")
        print(f"  Improvement:      {rms_before/rms_after_outdated:.2f}x")
        
        if rms_after_amended is not None:
            print("\n" + "="*70)
            print("COMPARISON: AMENDED vs OUTDATED")
            print("="*70)
            print(f"RMS after with AMENDED:  {rms_after_amended:.4f}")
            print(f"RMS after with OUTDATED: {rms_after_outdated:.4f}")
            improvement_ratio = rms_after_outdated / rms_after_amended
            print(f"Amended is {improvement_ratio:.2f}x better than outdated")
        else:
            print("\nNote: Amended offset map cannot be used for desmiling.")
    
    # Plot comparison
    print("\nCreating comparison plot(s)...")
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figures_dir = Path(config.directories.figures)
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Define common color limits for differences
    vmin_diff = np.nanpercentile(diff_before, 1)
    vmax_diff = np.nanpercentile(diff_before, 99)
    
    # Helper function to create a plot
    def create_plot(upper_orig, lower_orig, offset_up, offset_low, diff_bef, diff_aft, 
                    rms_bef, rms_aft, title_suffix, filename_suffix):
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(f'Desmiling Quality Test: {title_suffix}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Row 0: Original frames
        im0 = axes[0, 0].imshow(upper_orig, aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Upper Frame (Original)')
        axes[0, 0].set_xlabel('Wavelength [px]')
        axes[0, 0].set_ylabel('Spatial [px]')
        plt.colorbar(im0, ax=axes[0, 0], label='Intensity')
        
        im1 = axes[0, 1].imshow(lower_orig, aspect='auto', cmap='viridis')
        axes[0, 1].set_title('Lower Frame (Original)')
        axes[0, 1].set_xlabel('Wavelength [px]')
        axes[0, 1].set_ylabel('Spatial [px]')
        plt.colorbar(im1, ax=axes[0, 1], label='Intensity')
        
        # Row 1: Offset maps
        im2 = axes[1, 0].imshow(offset_up, aspect='auto', cmap='RdBu_r')
        axes[1, 0].set_title('Offset Map (Upper, State 0)')
        axes[1, 0].set_xlabel('Wavelength [px]')
        axes[1, 0].set_ylabel('Spatial [px]')
        plt.colorbar(im2, ax=axes[1, 0], label='Offset [px]')
        
        im3 = axes[1, 1].imshow(offset_low, aspect='auto', cmap='RdBu_r')
        axes[1, 1].set_title('Offset Map (Lower, State 1)')
        axes[1, 1].set_xlabel('Wavelength [px]')
        axes[1, 1].set_ylabel('Spatial [px]')
        plt.colorbar(im3, ax=axes[1, 1], label='Offset [px]')
        
        # Row 2: Difference before desmiling
        im4 = axes[2, 0].imshow(diff_bef, aspect='auto', cmap='RdBu_r',
                                vmin=vmin_diff, vmax=vmax_diff)
        axes[2, 0].set_title(f'Upper - Lower (BEFORE desmiling)\nRMS: {rms_bef:.4f}')
        axes[2, 0].set_xlabel('Wavelength [px]')
        axes[2, 0].set_ylabel('Spatial [px]')
        plt.colorbar(im4, ax=axes[2, 0], label='Difference')
        
        # Row 2: Difference after desmiling
        im5 = axes[2, 1].imshow(diff_aft, aspect='auto', cmap='RdBu_r',
                                vmin=vmin_diff, vmax=vmax_diff)
        axes[2, 1].set_title(f'Upper - Lower (AFTER desmiling)\nRMS: {rms_aft:.4f}')
        axes[2, 1].set_xlabel('Wavelength [px]')
        axes[2, 1].set_ylabel('Spatial [px]')
        plt.colorbar(im5, ax=axes[2, 1], label='Difference')
        
        # Row 3: Improvement factor and statistics
        improvement = rms_bef / rms_aft
        axes[3, 0].text(0.5, 0.5, f'Improvement Factor:\n{improvement:.2f}x\n\n'
                                   f'RMS before: {rms_bef:.4f}\n'
                                   f'RMS after: {rms_aft:.4f}',
                        ha='center', va='center', fontsize=14,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[3, 0].axis('off')
        axes[3, 0].set_title('Quality Improvement')
        
        # Row 3: Additional info
        axes[3, 1].text(0.5, 0.5, f'{title_suffix}\n\nOffset Map Shape:\n{offset_up.shape}\n\n'
                                   f'Spatial: {offset_up.shape[0]}\n'
                                   f'Wavelength: {offset_up.shape[1]}',
                        ha='center', va='center', fontsize=14,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[3, 1].axis('off')
        axes[3, 1].set_title('Configuration Info')
        
        plt.tight_layout(h_pad=3.0, w_pad=2.0)
        
        # Save figure
        output_plot = figures_dir / f'desmiling_quality_{filename_suffix}_{timestamp}.png'
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot: {output_plot}")
        plt.close()
        return output_plot
    
    # Create comparison plot of offset maps if both exist
    offset_comparison_plot = None
    if has_outdated:
        print("\nCreating offset map comparison plot...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Offset Map Comparison: Outdated vs Amended', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        for row, state in enumerate([0, 1]):
            state_label = 'Upper' if state == 0 else 'Lower'
            
            # Column 0: Outdated offset map
            im0 = axes[row, 0].imshow(offset_map_outdated_array[state], aspect='auto', cmap='RdBu_r')
            axes[row, 0].set_title(f'Outdated Offset Map ({state_label}, State {state})')
            axes[row, 0].set_xlabel('Wavelength [px]')
            axes[row, 0].set_ylabel('Spatial [px]')
            plt.colorbar(im0, ax=axes[row, 0], label='Offset [px]')
            
            # Column 1: Amended offset map
            im1 = axes[row, 1].imshow(offset_map_amended_array[state], aspect='auto', cmap='RdBu_r')
            axes[row, 1].set_title(f'Amended Offset Map ({state_label}, State {state})')
            axes[row, 1].set_xlabel('Wavelength [px]')
            axes[row, 1].set_ylabel('Spatial [px]')
            plt.colorbar(im1, ax=axes[row, 1], label='Offset [px]')
            
            # Column 2: Difference (Amended - Outdated)
            diff = offset_map_amended_array[state] - offset_map_outdated_array[state]
            rms_diff = np.sqrt(np.mean(diff**2))
            im2 = axes[row, 2].imshow(diff, aspect='auto', cmap='seismic', 
                                      vmin=-np.percentile(np.abs(diff), 99),
                                      vmax=np.percentile(np.abs(diff), 99))
            axes[row, 2].set_title(f'Difference ({state_label}, State {state})\\nRMS: {rms_diff:.4f} px')
            axes[row, 2].set_xlabel('Wavelength [px]')
            axes[row, 2].set_ylabel('Spatial [px]')
            plt.colorbar(im2, ax=axes[row, 2], label='Difference [px]')
        
        plt.tight_layout()
        
        offset_comparison_plot = figures_dir / f'offset_map_comparison_{timestamp}.png'
        plt.savefig(offset_comparison_plot, dpi=150, bbox_inches='tight')
        print(f"✓ Saved offset map comparison: {offset_comparison_plot}")
        plt.close()
    
    # Create plot for DELTA corrections if available
    amended_plot = None
    if diff_after_amended is not None:
        print("\nCreating plot with DELTA corrections (amended - outdated)...")
        amended_plot = create_plot(
            upper_original, lower_original,
            delta_upper, delta_lower,
            diff_before, diff_after_amended,
            rms_before, rms_after_amended,
            "Delta Corrections (Amended - Outdated)",
            "delta"
        )
    else:
        print("\nSkipping delta corrections plot (no outdated map available)")
    
    # Create plot for OUTDATED offset map if available
    outdated_plot = None
    if has_outdated:
        print("Creating plot with OUTDATED (original) offset map...")
        outdated_plot = create_plot(
            upper_original, lower_original,
            offset_upper_outdated, offset_lower_outdated,
            diff_before, diff_after_outdated,
            rms_before, rms_after_outdated,
            "Outdated (Original) Offset Map",
            "outdated"
        )
    
    return {
        'upper_original': upper_original,
        'lower_original': lower_original,
        'upper_desmiled_amended': upper_desmiled_amended,
        'lower_desmiled_amended': lower_desmiled_amended,
        'upper_desmiled_outdated': upper_desmiled_outdated,
        'lower_desmiled_outdated': lower_desmiled_outdated,
        'diff_before': diff_before,
        'diff_after_amended': diff_after_amended,
        'diff_after_outdated': diff_after_outdated,
        'rms_before': rms_before,
        'rms_after_amended': rms_after_amended,
        'rms_after_outdated': rms_after_outdated,
        'has_outdated': has_outdated,
        'amended_plot': amended_plot,
        'outdated_plot': outdated_plot if has_outdated else None,
        'offset_comparison_plot': offset_comparison_plot,
        'offset_map_amended_array': offset_map_amended_array,
        'offset_map_outdated_array': offset_map_outdated_array if has_outdated else None
    }


if __name__ == "__main__":
    results = test_desmiling_quality()
    
    if results:
        print("\n" + "="*70)
        print("TEST COMPLETE - SUMMARY")
        print("="*70)
        
        # Offset map comparison
        if results['offset_comparison_plot'] is not None:
            print(f"\n✓ Generated offset map comparison: {results['offset_comparison_plot'].name}")
            
            # Show offset map difference statistics
            if results['offset_map_outdated_array'] is not None:
                offset_diff = results['offset_map_amended_array'] - results['offset_map_outdated_array']
                print(f"\nOffset map differences (Amended - Outdated):")
                for state in range(min(2, offset_diff.shape[0])):
                    state_label = 'Upper' if state == 0 else 'Lower'
                    rms_diff = np.sqrt(np.mean(offset_diff[state]**2))
                    max_diff = np.max(np.abs(offset_diff[state]))
                    print(f"  State {state} ({state_label}): RMS={rms_diff:.4f} px, Max={max_diff:.4f} px")
        
        if results['rms_after_amended'] is not None and results['amended_plot'] is not None:
            print(f"\n✓ Generated plot with DELTA corrections: {results['amended_plot'].name}")
        
        if results['has_outdated']:
            print(f"✓ Generated plot with OUTDATED offset map: {results['outdated_plot'].name}")
            
            if results['rms_after_amended'] is not None:
                print(f"\nDesmiling Quality Comparison:")
                print(f"  RMS before correction:        {results['rms_before']:.4f}")
                print(f"  RMS after (OUTDATED):         {results['rms_after_outdated']:.4f}")
                print(f"  RMS after (DELTA):            {results['rms_after_amended']:.4f}")
                print(f"  Improvement (OUTDATED):       {results['rms_before']/results['rms_after_outdated']:.2f}x")
                print(f"  Improvement (DELTA):          {results['rms_before']/results['rms_after_amended']:.2f}x")
                
                if results['rms_after_amended'] < results['rms_after_outdated']:
                    improvement = results['rms_after_outdated'] / results['rms_after_amended']
                    print(f"  ✓ Delta corrections are {improvement:.2f}x better than outdated alone")
                elif results['rms_after_amended'] > results['rms_after_outdated']:
                    print(f"  ⚠ Outdated performed better (delta corrections may be too small to see effect)")
                else:
                    print(f"  = Similar performance")
            else:
                print(f"\nDesmiling Quality (using OUTDATED offset map):")
                print(f"  RMS before desmiling:  {results['rms_before']:.4f}")
                print(f"  RMS after desmiling:   {results['rms_after_outdated']:.4f}")
                print(f"  Improvement:           {results['rms_before']/results['rms_after_outdated']:.2f}x")
                
        print("\nTest complete!")
