#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test spectroflat processing separately to diagnose report generation issues.

This script tests spectroflat with 2-state input (upper and lower frames) and
attempts to generate the report to identify any plotting issues.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from themis.datasets.themis_datasets_2025 import get_config
from themis.core import themis_io as tio
from spectroflat import Analyser, Config as SpectroflatConfig, SmileConfig, SensorFlatConfig
from qollib.strings import parse_shape

import matplotlib.pyplot as plt

def test_spectroflat_with_report():
    """Test spectroflat with report generation enabled."""
    print("="*70)
    print("Testing spectroflat with 2-state input and report generation")
    print("="*70)
    
    # Load configuration
    config = get_config(
        config_path='configs/sample_dataset_sr_2025-07-07.toml',
        auto_discover_files=True,
        auto_create_dirs=False
    )
    
    data_type = 'flat_center'
    
    # Read L0 data
    print(f"\n1. Loading {data_type} L0 data...")
    data, header = tio.read_any_file(config, data_type, verbose=False, status='l0')
    
    if len(data) == 0:
        print(f"Error: No frames found in {data_type} L0 data")
        return
    
    print(f"   Loaded {len(data)} frame(s)")
    
    # Extract upper and lower as 2D arrays
    upper_2d = data[0]['upper'].data
    lower_2d = data[0]['lower'].data
    
    print(f"   Upper shape: {upper_2d.shape} [spatial, wavelength]")
    print(f"   Lower shape: {lower_2d.shape} [spatial, wavelength]")
    
    # Stack as [state, spatial, wavelength]
    dirty_flat = np.stack([upper_2d, lower_2d], axis=0)
    print(f"\n2. Stacked input shape: {dirty_flat.shape} [state, spatial, wavelength]")
    
    # Define ROI
    roi = parse_shape(f'[2:{dirty_flat.shape[1]-2},2:{dirty_flat.shape[2]-2}]')
    print(f"   ROI: {roi}")
    
    # Configure spectroflat
    print("\n3. Configuring spectroflat...")
    sf_config = SpectroflatConfig(roi=roi, iterations=2)
    sf_config.sensor_flat = SensorFlatConfig(
        spacial_degree=13,
        sigma_mask=2,
        fit_border=1,
        average_column_response_map=True,
        ignore_gradient=False,
        roi=roi
    )
    sf_config.smile = SmileConfig(
        line_distance=16,
        strong_smile_deg=2,
        max_dispersion_deg=5,
        line_prominence=0.1,
        height_sigma=0.04,
        smooth=True,
        emission_spectrum=False,
        state_aware=False,
        align_states=True,
        smile_deg=3,
        rotation_correction=0,
        detrend=True,
        roi=roi
    )
    
    # Create report directory
    report_dir = Path(config.directories.figures) / 'spectroflat_test_report'
    report_dir.mkdir(exist_ok=True, parents=True)
    print(f"   Report directory: {report_dir}")
    
    # Run spectroflat with report
    print("\n4. Running spectroflat analysis WITH report generation...")
    print("   (This may fail with plotting error)")
    
    try:
        analyser = Analyser(dirty_flat, sf_config, str(report_dir))
        analyser.run()
        print(f"   ✓ Success! Report saved to: {report_dir}")
        
    except Exception as e:
        print(f"   ✗ Error with report generation:")
        print(f"     {type(e).__name__}: {e}")
        print("\n   Trying WITHOUT report generation...")
        
        # Try without report (pass None as third positional argument)
        analyser_no_report = Analyser(dirty_flat, sf_config, None)
        analyser_no_report.run()
        print(f"   ✓ Success without report!")
        
        # Check output
        print(f"\n5. Checking output shapes:")
        print(f"   dust_flat shape: {analyser_no_report.dust_flat.shape}")
        print(f"   offset_map type: {type(analyser_no_report.offset_map)}")
        
        return analyser_no_report
    
    # If report generation succeeded, check output
    print(f"\n5. Checking output shapes:")
    print(f"   dust_flat shape: {analyser.dust_flat.shape}")
    print(f"   offset_map type: {type(analyser.offset_map)}")
    
    return analyser


def test_spectroflat_with_more_states():
    """Test spectroflat with more than 2 states to see if that fixes the plotting."""
    print("\n" + "="*70)
    print("Testing spectroflat with 4-state input (duplicated upper/lower)")
    print("="*70)
    
    # Load configuration
    config = get_config(
        config_path='configs/sample_dataset_sr_2025-07-07.toml',
        auto_discover_files=True,
        auto_create_dirs=False
    )
    
    data_type = 'flat_center'
    
    # Read L0 data
    print(f"\n1. Loading {data_type} L0 data...")
    data, header = tio.read_any_file(config, data_type, verbose=False, status='l0')
    
    upper_2d = data[0]['upper'].data
    lower_2d = data[0]['lower'].data
    
    # Create 4 states by duplicating upper and lower
    dirty_flat = np.stack([upper_2d, lower_2d, upper_2d, lower_2d], axis=0)
    print(f"\n2. Stacked input shape: {dirty_flat.shape} [state, spatial, wavelength]")
    
    # Define ROI
    roi = parse_shape(f'[100:{dirty_flat.shape[1]-100},100:{dirty_flat.shape[2]-100}]')
    
    # Configure spectroflat (minimal config for testing)
    sf_config = SpectroflatConfig(roi=roi, iterations=1)
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
    
    # Create report directory
    report_dir = Path(config.directories.figures) / 'spectroflat_4state_test_report'
    report_dir.mkdir(exist_ok=True, parents=True)
    
    # Run spectroflat with report
    print("\n3. Running spectroflat analysis with 4 states...")
    
    try:
        analyser = Analyser(dirty_flat, sf_config, str(report_dir))
        analyser.run()
        print(f"   ✓ Success with 4 states! Report saved to: {report_dir}")
        print(f"   dust_flat shape: {analyser.dust_flat.shape}")
        plt.imshow(analyser.dust_flat[0], vmin=0.97, vmax=1.03)
        plt.imshow(analyser.illumination_pattern[0])
        
    except Exception as e:
        print(f"   ✗ Error even with 4 states:")
        print(f"     {type(e).__name__}: {e}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("SPECTROFLAT STANDALONE TEST")
    print("="*70)
    
    # Test 1: With 2 states (our actual use case)
    print("\n[TEST 1: 2 states with/without report] - not working")
   # analyser = test_spectroflat_with_report()
    
    # Test 2: With 4 states (diagnostic only)
    print("\n[TEST 2: 4 states with report - diagnostic]")
    test_spectroflat_with_more_states()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")

