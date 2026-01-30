"""
Test and verify that desmiling is working correctly.
Visualizes the offset map, compares original vs desmiled images,
and verifies that _apply_desmiling actually applies the offset correction.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from themis.datasets.themis_datasets_2025 import get_config
from themis.core import themis_io as tio
from themis.core.themis_data_reduction import _apply_desmiling


def test_desmiling(data_type='flat', config_path='configs/sample_dataset_sr_2025-07-07.toml'):
    """
    Test that desmiling is working correctly.
    
    Parameters
    ----------
    data_type : str
        Data type to test: 'flat' or 'flat_center'
    config_path : str
        Path to configuration file
    """
    
    print("="*70)
    print("DESMILING VERIFICATION TEST")
    print("="*70)
    
    # Load configuration
    config = get_config(config_path=config_path)
    
    figures_dir = Path(config.directories.figures) / 'desmiling_test'
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # =========================================================================
    # PART 1: Load and visualize offset maps
    # =========================================================================
    print("\n" + "="*70)
    print("PART 1: OFFSET MAP VISUALIZATION")
    print("="*70)
    
    # Get offset maps from flat_center auxiliary files
    files_fc = config.dataset.get('flat_center', {}).get('files')
    if files_fc is None:
        print("ERROR: flat_center files not found in config")
        return
    
    aux_fc = getattr(files_fc, 'auxiliary', {})
    
    offset_maps = {}
    for frame_name in ['upper', 'lower']:
        key = f'offset_map_{frame_name}'
        path = aux_fc.get(key)
        if path is None or not Path(path).exists():
            print(f"  WARNING: offset map not found for {frame_name}")
            print(f"    Looking for key: {key}")
            print(f"    Available keys: {list(aux_fc.keys())}")
            continue
        
        with fits.open(path) as hdul:
            offset_maps[frame_name] = np.array(hdul[0].data)
        print(f"  Loaded offset map for {frame_name}: {Path(path).name}")
        print(f"    Shape: {offset_maps[frame_name].shape}")
        print(f"    Min: {offset_maps[frame_name].min():.4f}, Max: {offset_maps[frame_name].max():.4f}")
        print(f"    Mean: {offset_maps[frame_name].mean():.4f}, Std: {offset_maps[frame_name].std():.4f}")
    
    if not offset_maps:
        print("\nNo offset maps found. Cannot continue test.")
        return
    
    # Plot offset maps
    fig, axes = plt.subplots(1, len(offset_maps), figsize=(6*len(offset_maps), 5))
    if len(offset_maps) == 1:
        axes = [axes]
    
    for ax, (frame_name, omap) in zip(axes, offset_maps.items()):
        vmax = np.percentile(np.abs(omap), 99)
        im = ax.imshow(omap, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Offset Map - {frame_name.upper()}\nRange: [{omap.min():.2f}, {omap.max():.2f}] px')
        ax.set_xlabel('Wavelength [px]')
        ax.set_ylabel('Spatial [px]')
        plt.colorbar(im, ax=ax, label='Offset [px]')
    
    plt.tight_layout()
    offset_plot = figures_dir / f'offset_maps_{data_type}.png'
    plt.savefig(offset_plot, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved offset map plot: {offset_plot}")
    plt.close()
    
    # =========================================================================
    # PART 2: Test with synthetic data
    # =========================================================================
    print("\n" + "="*70)
    print("PART 2: SYNTHETIC DATA TEST")
    print("="*70)
    
    # Create a simple synthetic test: vertical lines that should become straight after desmiling
    ny, nx = 100, 200
    
    # Create synthetic offset map: sinusoidal smile pattern
    # The offset map tells us where each pixel SHOULD be (the correction needed)
    y_coords = np.arange(ny)
    synthetic_offset = 3.0 * np.sin(2 * np.pi * y_coords / ny)[:, np.newaxis] * np.ones((ny, nx))
    print(f"  Synthetic offset map: sinusoidal with amplitude ±3 pixels")
    print(f"    Shape: {synthetic_offset.shape}")
    
    # Create synthetic image: vertical lines (the "true" undistorted image)
    synthetic_image = np.zeros((ny, nx))
    for x in range(20, nx, 40):  # Vertical lines every 40 pixels
        synthetic_image[:, x] = 1.0
    
    # Create "smiled" version (what we'd observe due to optical distortion)
    # The smile distortion shifts pixels in the OPPOSITE direction of the offset map
    # If offset says +3, the observed image has features shifted by -3
    smiled_image = np.zeros_like(synthetic_image)
    x_indices = np.arange(nx)
    for y in range(ny):
        # Distortion shifts in opposite direction of the correction
        distorted_x = x_indices - synthetic_offset[y]
        smiled_image[y] = np.interp(x_indices, distorted_x, synthetic_image[y])
    
    # Now apply desmiling - this should recover the original vertical lines
    desmiled_synthetic = _apply_desmiling(smiled_image, synthetic_offset)
    
    # Plot synthetic test
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Synthetic test
    axes[0, 0].imshow(synthetic_offset, aspect='auto', cmap='RdBu_r')
    axes[0, 0].set_title('Synthetic Offset Map')
    axes[0, 0].set_xlabel('Wavelength [px]')
    axes[0, 0].set_ylabel('Spatial [px]')
    
    axes[0, 1].imshow(smiled_image, aspect='auto', cmap='gray')
    axes[0, 1].set_title('Smiled Image (what we observe)')
    axes[0, 1].set_xlabel('Wavelength [px]')
    axes[0, 1].set_ylabel('Spatial [px]')
    
    axes[0, 2].imshow(desmiled_synthetic, aspect='auto', cmap='gray')
    axes[0, 2].set_title('After Desmiling (should be straight lines)')
    axes[0, 2].set_xlabel('Wavelength [px]')
    axes[0, 2].set_ylabel('Spatial [px]')
    
    # Row 2: Difference
    diff_smiled_original = smiled_image - synthetic_image
    diff_desmiled_original = desmiled_synthetic - synthetic_image
    
    axes[1, 0].imshow(synthetic_image, aspect='auto', cmap='gray')
    axes[1, 0].set_title('Original (straight lines)')
    
    axes[1, 1].imshow(diff_smiled_original, aspect='auto', cmap='RdBu_r', 
                      vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title(f'Smiled - Original\nRMS: {np.sqrt(np.mean(diff_smiled_original**2)):.4f}')
    
    axes[1, 2].imshow(diff_desmiled_original, aspect='auto', cmap='RdBu_r',
                      vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title(f'Desmiled - Original\nRMS: {np.sqrt(np.mean(diff_desmiled_original**2)):.4f}')
    
    plt.tight_layout()
    synthetic_plot = figures_dir / f'synthetic_desmiling_test_{data_type}.png'
    plt.savefig(synthetic_plot, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved synthetic test plot: {synthetic_plot}")
    plt.close()
    
    # Check if desmiling worked
    rms_before = np.sqrt(np.mean(diff_smiled_original**2))
    rms_after = np.sqrt(np.mean(diff_desmiled_original**2))
    print(f"\n  Synthetic test results:")
    print(f"    RMS (smiled vs original): {rms_before:.6f}")
    print(f"    RMS (desmiled vs original): {rms_after:.6f}")
    
    if rms_after < rms_before:
        print(f"    ✓ Desmiling REDUCED the error (good!)")
    else:
        print(f"    ✗ Desmiling did NOT reduce the error (problem!)")
    
    # =========================================================================
    # PART 3: Real data test
    # =========================================================================
    print("\n" + "="*70)
    print("PART 3: REAL DATA TEST")
    print("="*70)
    
    # Load L2 data
    print(f"  Loading L2 {data_type} data...")
    try:
        l2_data, header = tio.read_any_file(config, data_type, status='l2', verbose=False)
    except Exception as e:
        print(f"  ERROR loading L2 {data_type}: {e}")
        return
    
    l2_frame = l2_data.get(0)
    
    # Process each half
    for frame_name in ['upper', 'lower']:
        if frame_name not in offset_maps:
            print(f"  Skipping {frame_name} - no offset map")
            continue
        
        print(f"\n  Processing {frame_name.upper()} frame...")
        
        half_obj = l2_frame.get_half(frame_name)
        original = half_obj.data.astype('float32')
        omap = offset_maps[frame_name]
        
        print(f"    Original shape: {original.shape}")
        print(f"    Offset map shape: {omap.shape}")
        
        # Apply desmiling
        desmiled = _apply_desmiling(original, omap)
        
        # Compute difference
        diff = desmiled - original
        
        print(f"    Difference stats:")
        print(f"      Min: {diff.min():.4f}, Max: {diff.max():.4f}")
        print(f"      Mean: {diff.mean():.6f}, Std: {diff.std():.4f}")
        print(f"      RMS: {np.sqrt(np.mean(diff**2)):.4f}")
        
        # Check if difference is significant
        if np.abs(diff).max() < 1e-6:
            print(f"    ✗ WARNING: No change detected! Desmiling may not be working.")
        else:
            print(f"    ✓ Difference detected - desmiling is doing something.")
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Row 1: Full images
        vmin, vmax = np.percentile(original, [1, 99])
        
        im0 = axes[0, 0].imshow(original, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f'Original L2 - {frame_name.upper()}')
        plt.colorbar(im0, ax=axes[0, 0])
        
        im1 = axes[0, 1].imshow(desmiled, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f'Desmiled - {frame_name.upper()}')
        plt.colorbar(im1, ax=axes[0, 1])
        
        diff_vmax = np.percentile(np.abs(diff), 99)
        im2 = axes[0, 2].imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
        axes[0, 2].set_title(f'Difference (Desmiled - Original)\nRMS: {np.sqrt(np.mean(diff**2)):.4f}')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Row 2: Zoom on a region + offset map
        omap_vmax = np.percentile(np.abs(omap), 99)
        im3 = axes[1, 0].imshow(omap, aspect='auto', cmap='RdBu_r', vmin=-omap_vmax, vmax=omap_vmax)
        axes[1, 0].set_title(f'Offset Map - {frame_name.upper()}\nRange: [{omap.min():.2f}, {omap.max():.2f}] px')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Show horizontal slices at different y positions
        y_positions = [original.shape[0]//4, original.shape[0]//2, 3*original.shape[0]//4]
        colors = ['r', 'g', 'b']
        
        for y_pos, color in zip(y_positions, colors):
            axes[1, 1].plot(original[y_pos, :], color=color, alpha=0.7, label=f'Original y={y_pos}')
            axes[1, 1].plot(desmiled[y_pos, :], color=color, linestyle='--', alpha=0.7, label=f'Desmiled y={y_pos}')
        axes[1, 1].set_title('Horizontal Slices Comparison')
        axes[1, 1].set_xlabel('Wavelength [px]')
        axes[1, 1].set_ylabel('Intensity')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].set_xlim(0, 100)  # Zoom to first 100 pixels
        
        # Show offset values along a column
        x_pos = original.shape[1] // 2
        axes[1, 2].plot(omap[:, x_pos], label=f'Offset at x={x_pos}')
        axes[1, 2].set_title(f'Offset Profile at x={x_pos}')
        axes[1, 2].set_xlabel('Spatial [px]')
        axes[1, 2].set_ylabel('Offset [px]')
        axes[1, 2].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[1, 2].legend()
        
        plt.tight_layout()
        real_plot = figures_dir / f'real_data_desmiling_{data_type}_{frame_name}.png'
        plt.savefig(real_plot, dpi=150, bbox_inches='tight')
        print(f"    ✓ Saved plot: {real_plot}")
        plt.close()
    
    # =========================================================================
    # PART 4: Compare L3 saved data with desmiled L2
    # =========================================================================
    print("\n" + "="*70)
    print("PART 4: VERIFY L3 DATA MATCHES DESMILED L2")
    print("="*70)
    
    # Load L3 data
    print(f"  Loading L3 {data_type} data...")
    try:
        l3_data, l3_header = tio.read_any_file(config, data_type, status='l3', verbose=False)
        l3_frame = l3_data.get(0)
        
        # Compare with freshly desmiled L2
        for frame_name in ['upper', 'lower']:
            if frame_name not in offset_maps:
                print(f"  Skipping {frame_name} - no offset map")
                continue
            
            print(f"\n  Comparing {frame_name.upper()} frame...")
            
            # Get L2 and L3 data
            l2_half = l2_frame.get_half(frame_name).data.astype('float32')
            l3_half = l3_frame.get_half(frame_name).data.astype('float32')
            
            # Apply desmiling to L2
            omap = offset_maps[frame_name]
            l2_desmiled = _apply_desmiling(l2_half, omap)
            
            # Compare L3 with desmiled L2
            diff_l3_desmiled = l3_half - l2_desmiled
            diff_l3_l2 = l3_half - l2_half
            
            print(f"    L3 vs L2 (should be different if desmiling applied):")
            print(f"      RMS: {np.sqrt(np.mean(diff_l3_l2**2)):.6f}")
            print(f"      Max abs diff: {np.abs(diff_l3_l2).max():.6f}")
            
            print(f"    L3 vs freshly desmiled L2 (should be ~0 if L3 is correct):")
            print(f"      RMS: {np.sqrt(np.mean(diff_l3_desmiled**2)):.6f}")
            print(f"      Max abs diff: {np.abs(diff_l3_desmiled).max():.6f}")
            
            if np.abs(diff_l3_desmiled).max() < 1e-4:
                print(f"    ✓ L3 matches desmiled L2 - saving is correct!")
            elif np.abs(diff_l3_l2).max() < 1e-4:
                print(f"    ✗ L3 is identical to L2 - desmiling NOT saved!")
            else:
                print(f"    ? L3 differs from both L2 and desmiled L2 - investigate")
            
            # Plot comparison
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            vmin, vmax = np.percentile(l2_half, [1, 99])
            
            axes[0, 0].imshow(l2_half, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0, 0].set_title(f'L2 Original - {frame_name.upper()}')
            
            axes[0, 1].imshow(l2_desmiled, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0, 1].set_title(f'L2 Desmiled (fresh) - {frame_name.upper()}')
            
            axes[0, 2].imshow(l3_half, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0, 2].set_title(f'L3 Loaded - {frame_name.upper()}')
            
            diff_vmax = max(np.percentile(np.abs(diff_l3_l2), 99), 1e-6)
            axes[1, 0].imshow(diff_l3_l2, aspect='auto', cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
            axes[1, 0].set_title(f'L3 - L2\nRMS: {np.sqrt(np.mean(diff_l3_l2**2)):.6f}')
            
            diff2_vmax = max(np.percentile(np.abs(diff_l3_desmiled), 99), 1e-6)
            axes[1, 1].imshow(diff_l3_desmiled, aspect='auto', cmap='RdBu_r', vmin=-diff2_vmax, vmax=diff2_vmax)
            axes[1, 1].set_title(f'L3 - Desmiled L2\nRMS: {np.sqrt(np.mean(diff_l3_desmiled**2)):.6f}')
            
            # Difference between L2 original and desmiled
            diff_desmiled_l2 = l2_desmiled - l2_half
            diff3_vmax = max(np.percentile(np.abs(diff_desmiled_l2), 99), 1e-6)
            axes[1, 2].imshow(diff_desmiled_l2, aspect='auto', cmap='RdBu_r', vmin=-diff3_vmax, vmax=diff3_vmax)
            axes[1, 2].set_title(f'Desmiled L2 - L2\nRMS: {np.sqrt(np.mean(diff_desmiled_l2**2)):.6f}')
            
            plt.tight_layout()
            l3_plot = figures_dir / f'l3_vs_desmiled_l2_{data_type}_{frame_name}.png'
            plt.savefig(l3_plot, dpi=150, bbox_inches='tight')
            print(f"    ✓ Saved plot: {l3_plot}")
            plt.close()
            
    except Exception as e:
        print(f"  Could not load L3 {data_type} data: {e}")
        print(f"  Run L2->L3 reduction first: reduce_l2_to_l3(config, '{data_type}')")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print(f"All plots saved to: {figures_dir}")
    print("="*70)


if __name__ == '__main__':
    test_desmiling(data_type='flat_center')
