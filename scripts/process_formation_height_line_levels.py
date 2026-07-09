#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 4 15:24:27 2025

@author: zeuner


    # Example usage: load L4 data for different positions and lines

    # Option 1: Load specific positions
    ti_disk_center_l4, ti_disk_center_header = load_l4_for_position('ti', 'disk_center')
    print(f"Loaded TI disk center L4 data: {ti_disk_center_l4.keys() if hasattr(ti_disk_center_l4, 'keys') else 'scalar'}")

    # Option 2: Iterate over all positions for a line
    print("\nIterating over all TI positions:")
    for position, sequence in iterate_positions('ti'):
        print(f"  Position: {position}, Sequence: {sequence}")

    # Option 3: Load all L4 data for a line at once
    print("\nLoading all TI L4 data:")
    ti_all_l4 = load_all_l4_for_line('ti')

    # Option 4: Get all available lines and positions
    print(f"\nAvailable lines: {get_all_lines()}")
    print(f"TI positions: {get_all_positions('ti')}")
    print(f"SR positions: {get_all_positions('sr')}")

    # Option 5: Loop over all lines and positions
    print("\nLooping over all lines and positions:")
    for line in get_all_lines():
        print(f"  Line: {line}")
        for position in get_all_positions(line):
            sequence = get_formation_height_config(line, position)
            print(f"    {position}: sequence {sequence}")


"""

from themis.core import themis_tools as tt
from themis.core import themis_data_reduction as tdr
from themis.core import themis_io as tio
from themis.datasets.themis_datasets_2025 import get_config
from themis.plots import plot_power_spectrum

from spectator.controllers.app_controller import display_data # from spectator

import matplotlib.pyplot as plt
import numpy as np
import gc
import pickle
from pathlib import Path
from datetime import datetime
from astropy.io import fits
from lmfit.models import VoigtModel, ConstantModel

#----------------------------------------------------------------------
"""
MAIN: 
   
"""

# Configuration file path
CONFIG_PATH = 'configs/formation_dataset_2025-07-05.toml'

# Configuration for formation height analysis: maps disk positions to sequences
# Based on comments in formation_dataset_2025-07-05.toml
# m := minus (e.g., m40 = -40)
# Both TI and SR lines share the same positions and sequences
POSITION_TO_SEQUENCE = {
    'disk_center': 27,
    'm40': 24,
    'm30': 18,
    '30': 50,
    '40': 36,
    '45': 38,
    'm45': 48,
    'm55': 92,
    '55': 86,
}

# Available spectral lines
AVAILABLE_LINES = ['ti', 'sr']

# Helper: convert position name to numeric value (for plotting labels, etc.)
POSITION_TO_NUMERIC = {
    'disk_center': 0,
    'm40': -40,
    'm30': -30,
    '30': 30,
    '40': 40,
    '45': 45,
    'm45': -45,
    'm55': -55,
    '55': 55,
}

# Predefined crop configurations for each line
CROP_CONFIGS = {
    'sr': {'scan': (0, -1), 'slit': (2, -55), 'wavelength': (276, 886)},
    'ti': {'scan': (0, -1), 'slit': (2, -55), 'wavelength': (406, 1016)},
}

# Predefined split configurations for each line (wavelength ranges in nm)
SPLIT_CONFIGS = {
    'sr': {
        'continuum': (460.716, 460.719),
        'line': (460.719, 460.746),
        'residual': (460.746, 460.783),
    },
    'ti': {
        'continuum': (453.654, 453.66),
        'line': (453.631, 453.654),
        'residual': (453.58, 453.632),
    },
}

def crop_to_array(crop_dict):
    """
    Convert crop dictionary to array format for internal use.

    Args:
        crop_dict: Dictionary with keys 'slit', 'scan', 'wavelength',
                   each containing a tuple (start, end).

    Returns:
        list: Array [scan_start, scan_end, slit_start, slit_end, wl_start, wl_end]
    """
    return [
        crop_dict['scan'][0], crop_dict['scan'][1],
        crop_dict['slit'][0], crop_dict['slit'][1],
        crop_dict['wavelength'][0], crop_dict['wavelength'][1]
    ]

def split_spectrum(intensity, wavelength, split_config):
    """
    Split intensity and wavelength data into continuum, line, and residual parts
    based on provided wavelength ranges, and normalize by the continuum mean.

    Args:
        intensity: Dictionary mapping line -> position -> intensity array
        wavelength: Dictionary mapping line -> wavelength array
        split_config: Dictionary mapping line -> split configuration,
                      where each split config has keys 'continuum', 'line', 'residual',
                      each containing a tuple (wl_min, wl_max) in nm

    Returns:
        dict: Dictionary with same structure as intensity, but each intensity array
              is replaced by a dict with keys 'continuum', 'line', 'residual',
              each containing a tuple (intensity_part, wavelength_part).
              All intensity parts are normalized by the scan and spectral mean
              of the continuum region.
    """
    result = {}

    for line in intensity.keys():
        result[line] = {}
        line_split_config = split_config[line]
        wl = wavelength[line]

        for position in intensity[line].keys():
            intensity_data = intensity[line][position]
            result[line][position] = {}

            # First, extract continuum to compute normalization factor
            continuum_wl_min, continuum_wl_max = line_split_config['continuum']
            mask = (wl >= continuum_wl_min) & (wl <= continuum_wl_max)
            wl_indices = np.where(mask)[0]

            if len(wl_indices) == 0:
                print(f"Warning: No pixels found for {line} {position} continuum range {continuum_wl_min}-{continuum_wl_max} nm")
                continuum_mean = 1.0
            else:
                wl_start, wl_end = wl_indices[0], wl_indices[-1] + 1

                # Slice continuum intensity
                if intensity_data.ndim == 1:
                    continuum_intensity = intensity_data[wl_start:wl_end]
                    continuum_mean = np.mean(continuum_intensity)
                elif intensity_data.ndim == 2:
                    continuum_intensity = intensity_data[:, wl_start:wl_end]
                    continuum_mean = np.mean(continuum_intensity)
                elif intensity_data.ndim == 3:
                    continuum_intensity = intensity_data[:, :, wl_start:wl_end]
                    continuum_mean = np.mean(continuum_intensity)
                else:
                    raise ValueError(f"Unsupported intensity shape: {intensity_data.shape}")

            # Now split all parts and normalize
            for part_name, (wl_min, wl_max) in line_split_config.items():
                # Find pixel indices corresponding to wavelength range
                mask = (wl >= wl_min) & (wl <= wl_max)
                wl_indices = np.where(mask)[0]

                if len(wl_indices) == 0:
                    print(f"Warning: No pixels found for {line} {position} {part_name} range {wl_min}-{wl_max} nm")
                    result[line][position][part_name] = (None, None)
                    continue

                wl_start, wl_end = wl_indices[0], wl_indices[-1] + 1

                # Slice wavelength
                wl_part = wl[wl_start:wl_end]

                # Slice intensity (last dimension is wavelength)
                if intensity_data.ndim == 1:
                    intensity_part = intensity_data[wl_start:wl_end] / continuum_mean
                elif intensity_data.ndim == 2:
                    intensity_part = intensity_data[:, wl_start:wl_end] / continuum_mean
                elif intensity_data.ndim == 3:
                    intensity_part = intensity_data[:, :, wl_start:wl_end] / continuum_mean
                else:
                    raise ValueError(f"Unsupported intensity shape: {intensity_data.shape}")

                result[line][position][part_name] = (intensity_part, wl_part)

    return result


class SpectrumPart:
    """
    Class to hold a single part of a split spectrum (continuum, line, or residual).

    Attributes:
        data: Intensity array (normalized), organized: wavelength, direction along the slit, scan direction (slit positions)
        wvl: Wavelength array
        fit_result: Optional fit result dictionary
    """
    def __init__(self, data, wvl, fit_result=None):
        self.data = data
        self.wvl = wvl
        self.fit_result = fit_result

    def __repr__(self):
        if self.data is None:
            return f"SpectrumPart(data=None, wvl=None)"
        return f"SpectrumPart(data.shape={self.data.shape}, wvl.shape={self.wvl.shape})"

    def save(self, filepath, config_path=None, sequence=None, python_file=None):
        """
        Save the spectrum part to a FITS file.

        Args:
            filepath: Path to save the FITS file
            config_path: Path to the configuration file (saved in header)
            sequence: Sequence number used (saved in header)
            python_file: Name of the Python file that created this file (saved in header)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create primary HDU with data
        primary_hdu = fits.PrimaryHDU(data=self.data)

        # Create extension HDU with wavelength
        wavelength_hdu = fits.ImageHDU(data=self.wvl, name='WAVELENGTH')

        # Add header information
        header = primary_hdu.header
        header['CONFIG'] = str(config_path) if config_path else 'unknown'
        header['DATE'] = datetime.now().isoformat()
        header['SEQUENCE'] = str(sequence) if sequence is not None else 'unknown'
        header['PYTHON'] = str(python_file) if python_file else 'unknown'

        # Create HDU list and write to file
        hdul = fits.HDUList([primary_hdu, wavelength_hdu])
        hdul.writeto(filepath, overwrite=True)

    @classmethod
    def load(cls, filepath):
        """
        Load a spectrum part from a FITS file.

        Args:
            filepath: Path to the FITS file

        Returns:
            SpectrumPart object
        """
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            wvl = hdul['WAVELENGTH'].data
            # Fit result is not stored in FITS, set to None
            return cls(data, wvl, fit_result=None)

    def plot(self, ax=None, scan_idx=None, slit_idx=None, **kwargs):
        """
        Plot the spectrum part.

        Args:
            ax: Optional matplotlib axes. If None, creates a new figure.
            scan_idx: Optional scan index (slit positions) to plot specific pixel. If None, averages over scan.
            slit_idx: Optional slit index (direction along the slit) to plot specific pixel. If None, averages over slit.
            **kwargs: Additional arguments passed to ax.plot

        Returns:
            ax: The matplotlib axes object
        """
        if self.data is None or self.wvl is None:
            print("Cannot plot: data or wvl is None")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Handle different data dimensions
        # Data is now in (wavelength, slit, scan) order
        if self.data.ndim == 1:
            ax.plot(self.wvl, self.data, **kwargs)
        elif self.data.ndim == 2:
            # (wavelength, slit)
            if slit_idx is not None:
                # Plot specific slit
                ax.plot(self.wvl, self.data[:, slit_idx], **kwargs)
            else:
                # Average over slit
                slit_avg = np.mean(self.data, axis=1)
                ax.plot(self.wvl, slit_avg, **kwargs)
        elif self.data.ndim == 3:
            # (wavelength, slit, scan)
            if scan_idx is not None and slit_idx is not None:
                # Plot specific pixel
                ax.plot(self.wvl, self.data[:, slit_idx, scan_idx], **kwargs)
            elif scan_idx is not None:
                # Average over slit for specific scan
                ax.plot(self.wvl, np.mean(self.data[:, :, scan_idx], axis=1), **kwargs)
            elif slit_idx is not None:
                # Average over scan for specific slit
                ax.plot(self.wvl, np.mean(self.data[:, slit_idx, :], axis=1), **kwargs)
            else:
                # Average over slit and scan
                slit_avg = np.mean(self.data, axis=(1, 2))
                ax.plot(self.wvl, slit_avg, **kwargs)
        else:
            raise ValueError(f"Unsupported data shape: {self.data.shape}")

        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Intensity (normalized)")
        ax.grid(True, alpha=0.3)

        if ax is None:
            plt.tight_layout()
            plt.show()

        return ax


class Spectrum:
    """
    Class to hold split spectrum data for a specific line and position.

    Attributes:
        continuum: SpectrumPart for continuum region
        line: SpectrumPart for line region
        residual: SpectrumPart for residual region
        fit_sum: Full fit result (wavelength, slit, scan) - fitted data array
    """
    def __init__(self, continuum, line, residual, line_name=None):
        self.continuum = continuum
        self.line = line
        self.residual = residual
        self.line_name = line_name  # 'ti' or 'sr'
        self.fit_sum = None  # Will store full fit result after fitting

    def __repr__(self):
        return f"Spectrum(continuum, line, residual)"

    def save(self, filepath, config_path=None, sequence=None, python_file=None):
        """
        Save each SpectrumPart to separate FITS files.

        Args:
            filepath: Base filepath (without extension). Each part will be saved as
                     filepath_continuum.fits, filepath_line.fits, filepath_residual.fits
            config_path: Path to the configuration file (saved in header)
            sequence: Sequence number used (saved in header)
            python_file: Name of the Python file that created this file (saved in header)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save each part separately
        self.continuum.save(f"{filepath}_continuum.fits", config_path, sequence, python_file)
        self.line.save(f"{filepath}_line.fits", config_path, sequence, python_file)
        self.residual.save(f"{filepath}_residual.fits", config_path, sequence, python_file)

    @classmethod
    def load(cls, filepath, line_name=None):
        """
        Load a split spectrum from separate FITS files.

        Args:
            filepath: Base filepath (without extension). Expects files
                     filepath_continuum.fits, filepath_line.fits, filepath_residual.fits
            line_name: Optional line name ('ti' or 'sr')

        Returns:
            Spectrum object
        """
        filepath = Path(filepath)
        continuum = SpectrumPart.load(f"{filepath}_continuum.fits")
        line = SpectrumPart.load(f"{filepath}_line.fits")
        residual = SpectrumPart.load(f"{filepath}_residual.fits")
        return cls(continuum, line, residual, line_name=line_name)

    def fit(self, slit=None, scan=None):
        """
        Fit the spectrum using line-specific models.

        For TI line: Uses 4 Voigt profiles (A, B, C, D) + constant model.
        Fits each spatial pixel individually, or a single pixel if indices are provided.

        For SR line: Not implemented yet (placeholder).

        Args:
            slit: Optional slit index to fit only that pixel. If None, fits all pixels.
            scan: Optional scan index to fit only that pixel. If None, fits all pixels.
                      Must be provided together with slit_idx.
        """
        if self.line_name == 'ti':
            self._fit_ti(slit=slit, scan=scan)
        elif self.line_name == 'sr':
            print("SR line fitting not implemented yet")
            return
        else:
            print(f"Unknown line: {self.line_name}")
            return

    def _fit_ti(self, slit=None, scan=None):
        """
        Fit TI line using 4 Voigt profiles (A, B, C, D) + constant.

        Procedure:
        1. Concatenate residual+line for full spectrum
        2. Split into blue and red parts (per pixel, using minimum of the line)
        3. Fit blue and red parts separately with full model (A+B+C+D+const)
        4. Full fit = blue fit + red fit
        5. TI line = voigtA_blue+const concatenated with voigtA_red+const
        6. Store voigtA_blue and voigtA_red separately for plotting

        Args:
            slit: Optional slit index to fit only that pixel. If None, fits all pixels.
            scan: Optional scan index to fit only that pixel. If None, fits all pixels.
                  Must be provided together with slit.
        """
        # Concatenate line and residual data for full spectrum fitting
        wvl_line = self.line.wvl
        wvl_residual = self.residual.wvl
        data_line = self.line.data  # shape: (n_wvl_line, n_slit, n_scan)
        data_residual = self.residual.data  # shape: (n_wvl_residual, n_slit, n_scan)

        # Concatenate along wavelength axis
        wvl_full = np.concatenate([wvl_line, wvl_residual])
        data_full = np.concatenate([data_line, data_residual], axis=0)

        n_wvl_full, n_slit, n_scan = data_full.shape
        n_wvl_line = data_line.shape[0]

        # Sort by wavelength to ensure proper blue/red splitting
        sort_idx = np.argsort(wvl_full)
        wvl_full = wvl_full[sort_idx]
        data_full = data_full[sort_idx, :, :]

        # Track which indices in sorted array correspond to original line region
        # Create a mask for line region (first n_wvl_line elements before sorting)
        line_mask = np.zeros(n_wvl_full, dtype=bool)
        line_mask[:n_wvl_line] = True
        # Apply the sort to the mask
        line_mask_sorted = line_mask[sort_idx]
        # Get indices where line region is in sorted array
        line_indices_sorted = np.where(line_mask_sorted)[0]

        # Determine which pixels to fit
        if slit is not None and scan is not None:
            # Fit single pixel
            slit_indices = [slit]
            scan_indices = [scan]
            single_pixel = True
        elif slit is None and scan is None:
            # Fit all pixels
            slit_indices = range(n_slit)
            scan_indices = range(n_scan)
            single_pixel = False
        else:
            raise ValueError("Both slit and scan must be provided together, or both must be None")

        # Initialize output arrays for full spectrum
        line_fit_full = np.zeros_like(data_full)  # Ti component (A_ + const)
        residual_fit_full = np.zeros_like(data_full)  # Parasitic components (B_ + C_ + D_)
        fit_sum_full = np.zeros_like(data_full)  # Full fit (A_ + B_ + C_ + D_ + const)
        voigtA_blue_full = np.zeros_like(data_full)  # VoigtA component from blue fit
        voigtA_red_full = np.zeros_like(data_full)  # VoigtA component from red fit

        # TI line center wavelength
        ti_center_wl = 453.6385

        # Pixel overlap between blue and red parts
        pixel_overlap = 2

        # Initialize model parameters
        def initialize_ti_parameters(params, ti_center_wl, continuum_mean):
            params['A_amplitude'].value = -0.01
            params['A_center'].value = ti_center_wl
            params['A_sigma'].value = 0.002
            params['A_gamma'].value = 1

            params['B_amplitude'].value = -0.01
            params['B_center'].value = 453.6268
            params['B_sigma'].value = 0.002
            params['B_gamma'].value = 1

            params['C_amplitude'].value = -0.01
            params['C_center'].value = 453.6050
            params['C_sigma'].value = 0.002
            params['C_gamma'].value = 1

            params['D_amplitude'].value = -0.01
            params['D_center'].value = 453.5926
            params['D_sigma'].value = 0.002
            params['D_gamma'].value = 1

            params['c_c'].value = continuum_mean
            #params['c_c'].vary = False  # Fix constant to continuum mean, don't fit
            return params

        # Fit each spatial pixel
        for i, slit in enumerate(slit_indices):
            for j, scan in enumerate(scan_indices):
                profile = data_full[:, slit, scan]

                # Find center of the line for this pixel (using minimum of line region)
                # First, find the wavelength of the minimum in the line region
                line_profile = data_line[:, slit, scan]
                idx_min_line = np.argmin(line_profile)
                wvl_min_line = wvl_line[idx_min_line]

                # Then find the index of this wavelength in the full sorted spectrum
                idx_center_pixel = np.argmin(np.abs(wvl_full - wvl_min_line))

                # Calculate continuum mean for this pixel
                continuum_profile = self.continuum.data[:, slit, scan]
                continuum_mean = np.mean(continuum_profile)

                # Create models
                voigtA = VoigtModel(prefix='A_')
                pars = voigtA.make_params()
                voigtB = VoigtModel(prefix='B_')
                pars.update(voigtB.make_params())
                voigtC = VoigtModel(prefix='C_')
                pars.update(voigtC.make_params())
                voigtD = VoigtModel(prefix='D_')
                pars.update(voigtD.make_params())
                const = ConstantModel(prefix='c_')
                pars.update(const.make_params())

                pars = initialize_ti_parameters(pars, ti_center_wl, continuum_mean)

                # Build separate models for blue and red
                model_blue = voigtA + voigtB + voigtC + voigtD + const
                model_red = voigtA + const

                # Extract blue and red parts for this pixel
                wvl_blue_pixel = wvl_full[:idx_center_pixel + pixel_overlap]
                wvl_red_pixel = wvl_full[idx_center_pixel - pixel_overlap:]
                profile_blue = profile[:idx_center_pixel + pixel_overlap]
                profile_red = profile[idx_center_pixel - pixel_overlap:]

                # Plot initial guess if fitting single pixel
                if single_pixel:
                    
                    fig, ax = plt.subplots(figsize=(12, 6))

                    # Blue part
                    initial_guess_blue = model_blue.eval(pars, x=wvl_blue_pixel)
                    ax.plot(wvl_blue_pixel, profile_blue, '--', label='Data (blue)', alpha=0.6, color='blue')
                    ax.plot(wvl_blue_pixel, initial_guess_blue, '-', label='Initial guess (blue)', linewidth=2, color='blue')

                    # Red part
                    initial_guess_red = model_red.eval(pars, x=wvl_red_pixel)
                    ax.plot(wvl_red_pixel, profile_red, '--', label='Data (red)', alpha=0.6, color='red')
                    ax.plot(wvl_red_pixel, initial_guess_red, '-', label='Initial guess (red)', linewidth=2, color='red')

                    ax.set_xlabel('Wavelength')
                    ax.set_ylabel('Intensity')
                    ax.set_title('Initial guess - Blue and red parts')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.show()

                # Fit blue part
                try:
                    out_blue = model_blue.fit(profile_blue, pars, x=wvl_blue_pixel)
                    components_blue = out_blue.eval_components(x=wvl_blue_pixel)

                    # Fit red part
                    out_red = model_red.fit(profile_red, pars, x=wvl_red_pixel)
                    components_red = out_red.eval_components(x=wvl_red_pixel)

                    # Store results
                    # VoigtA components (for plotting in different colors)
                    voigtA_blue_full[:idx_center_pixel + pixel_overlap, slit, scan] = components_blue['A_']
                    voigtA_red_full[idx_center_pixel - pixel_overlap:, slit, scan] = components_red['A_']

                    # Line fit: Ti component (A_ + const)
                    line_fit_full[:idx_center_pixel + pixel_overlap, slit, scan] = components_blue['A_'] + components_blue['c_']
                    line_fit_full[idx_center_pixel - pixel_overlap:, slit, scan] = components_red['A_'] + components_red['c_']

                    # Residual fit: Parasitic components (B_ + C_ + D_)
                    # Only blue part has parasitic components, red part has none
                    residual_fit_full[:idx_center_pixel + pixel_overlap, slit, scan] = components_blue['B_'] + components_blue['C_'] + components_blue['D_']
                    residual_fit_full[idx_center_pixel - pixel_overlap:, slit, scan] = 0  # No parasitic components in red part

                    # Full fit: blue fit + red fit
                    fit_sum_full[:idx_center_pixel + pixel_overlap, slit, scan] = out_blue.eval(x=wvl_blue_pixel)
                    fit_sum_full[idx_center_pixel - pixel_overlap:, slit, scan] = out_red.eval(x=wvl_red_pixel)

                except Exception as e:
                    print(f"Fit failed for slit={slit}, scan={scan}: {e}")
                    # Use zeros for failed fits
                    line_fit_full[:, slit, scan] = 0
                    residual_fit_full[:, slit, scan] = 0
                    fit_sum_full[:, slit, scan] = 0
                    voigtA_blue_full[:, slit, scan] = 0
                    voigtA_red_full[:, slit, scan] = 0

        # Separate results back into line and residual parts
        # Line region results go to line wavelength region
        self.line.fit = line_fit_full[line_indices_sorted, :, :]
        self.residual.fit = residual_fit_full[line_indices_sorted, :, :]
        self.voigtA_blue = voigtA_blue_full[line_indices_sorted, :, :]
        self.voigtA_red = voigtA_red_full[line_indices_sorted, :, :]

        # Full fit should be in the full wavelength region (blue+red)
        self.fit_sum = fit_sum_full
        self.fit_sum_wvl = wvl_full  # Store full wavelength array for plotting

        if single_pixel:
            print(f"TI line fitting completed for single pixel (slit={slit}, scan={scan})")
        else:
            print(f"TI line fitting completed for {n_slit} x {n_scan} spatial pixels")

    def plot(self, ax=None, show_continuum=True, show_line=True, show_residual=True,
              scan=None, slit=None, fit=False):
        """
        Plot all parts of the split spectrum.

        Args:
            ax: Optional matplotlib axes. If None, creates a new figure.
            show_continuum: Whether to plot continuum
            show_line: Whether to plot line
            show_residual: Whether to plot residual
            scan_idx: Optional scan index to plot specific pixel. If None, averages over scan.
            slit_idx: Optional slit index to plot specific pixel. If None, averages over slit.
            fit: Whether to plot fit results (line.fit, residual.fit, fit_sum)

        Returns:
            ax: The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if show_continuum and self.continuum.data is not None:
            self.continuum.plot(ax, label='Continuum', alpha=0.7,
                              scan_idx=scan, slit_idx=slit)
        if show_line and self.line.data is not None:
            self.line.plot(ax, label='Line', alpha=0.9,
                          scan_idx=scan, slit_idx=slit)
        if show_residual and self.residual.data is not None:
            self.residual.plot(ax, label='Residual', alpha=0.7,
                              scan_idx=scan, slit_idx=slit)

        # Plot fit results if available and requested
        if fit and self.fit_sum is not None:
            wvl = self.line.wvl

            # Helper function to extract spectrum from 3D array
            def extract_spectrum(data, scan, slit):
                if scan is not None and slit is not None:
                    return data[:, slit, scan]
                elif scan is not None:
                    return np.mean(data[:, :, scan], axis=1)
                elif slit is not None:
                    return np.mean(data[:, slit, :], axis=1)
                else:
                    return np.mean(data, axis=(1, 2))

            # Plot individual fit components (in line wavelength region)
            if hasattr(self.line, 'fit') and self.line.fit is not None:
                line_fit = extract_spectrum(self.line.fit, scan, slit)
                ax.plot(wvl, line_fit, '--', label='Line fit (Ti)', alpha=0.8, linewidth=1.5)

            if hasattr(self.residual, 'fit') and self.residual.fit is not None:
                residual_fit = extract_spectrum(self.residual.fit, scan, slit)
                ax.plot(wvl, residual_fit, '--', label='Residual fit (parasitic)', alpha=0.8, linewidth=1.5)

            # Plot voigtA components in different colors (in line wavelength region)
            if hasattr(self, 'voigtA_blue') and self.voigtA_blue is not None:
                voigtA_blue = extract_spectrum(self.voigtA_blue, scan, slit)
                ax.plot(wvl, voigtA_blue, ':', label='VoigtA (blue)', alpha=0.7, linewidth=1.5, color='blue')

            if hasattr(self, 'voigtA_red') and self.voigtA_red is not None:
                voigtA_red = extract_spectrum(self.voigtA_red, scan, slit)
                ax.plot(wvl, voigtA_red, ':', label='VoigtA (red)', alpha=0.7, linewidth=1.5, color='red')

            # Plot full fit sum (in full wavelength region)
            if hasattr(self, 'fit_sum_wvl') and self.fit_sum_wvl is not None:
                fit_sum = extract_spectrum(self.fit_sum, scan, slit)
                ax.plot(self.fit_sum_wvl, fit_sum, '-', label='Full fit', alpha=0.9, linewidth=2)

        ax.legend()
        ax.set_title("Split Spectrum")

        if ax is None:
            plt.tight_layout()
            plt.show()

        return ax


class SpectrumContainer:
    """
    Container class to hold multiple Spectrum objects organized by line and position.

    Allows access via string keys: container['ti']['disk_center']
    """
    def __init__(self, spectra_dict):
        self._spectra = spectra_dict

    def __getitem__(self, key):
        return self._spectra[key]

    def __setitem__(self, key, value):
        self._spectra[key] = value

    def __repr__(self):
        lines = list(self._spectra.keys())
        n_spectra = sum(len(positions) for positions in self._spectra.values())
        return f"SpectrumContainer(lines={lines}, n_spectra={n_spectra})"

    def keys(self):
        return self._spectra.keys()

    def values(self):
        return self._spectra.values()

    def items(self):
        return self._spectra.items()

    def save_all(self, config_path=None):
        """
        Save all spectra to individual FITS files.

        Creates a directory structure: config_directories.reduced/spectra/line/position_*.fits
        Each SpectrumPart is saved separately as continuum, line, residual.

        Args:
            config_path: Path to the configuration file. If None, uses CONFIG_PATH.
                         This determines the base directory for saving files.
        """
        if config_path is None:
            config_path = CONFIG_PATH

        # Load config to get the reduced directory (use any line, directories are the same)
        config = get_config(line='ti', config_path=config_path, auto_discover_files=True, auto_create_dirs=False)
        base_path = Path(config.directories.reduced) / 'spectra'

        python_file = 'process_formation_height_line_levels'

        n_spectra = 0
        for line in self._spectra.keys():
            for position in self._spectra[line].keys():
                position_dir = base_path / line / position
                position_dir.mkdir(parents=True, exist_ok=True)
                filepath = position_dir / position
                sequence = POSITION_TO_SEQUENCE.get(position, 'unknown')
                self._spectra[line][position].save(filepath, config_path, sequence, python_file)
                n_spectra += 1

        print(f"Saved {n_spectra} spectra to {base_path}")

    def plot(self, line=None, position=None, fit=False, scan=None, slit=None,
             show_continuum=True, show_line=True, show_residual=True):
        """
        Plot a specific spectrum from the container.

        Args:
            line: Line name (e.g., 'ti', 'sr'). If None, uses first available line.
            position: Position name (e.g., 'disk_center'). If None, uses first available position.
            fit: Whether to plot fit results
            scan: Optional scan index for specific pixel
            slit: Optional slit index for specific pixel
            show_continuum: Whether to plot continuum
            show_line: Whether to plot line
            show_residual: Whether to plot residual
        """
        if line is None:
            line = list(self._spectra.keys())[0]
            print(f"Using default line: {line}")

        if position is None:
            position = list(self._spectra[line].keys())[0]
            print(f"Using default position: {position}")

        spectrum = self._spectra[line][position]
        return spectrum.plot(ax=None, show_continuum=show_continuum, show_line=show_line,
                           show_residual=show_residual, scan_idx=scan, slit_idx=slit, fit=fit)

    def plot_all(self, show_continuum=True, show_line=True, show_residual=True):
        """
        Plot all spectra in a grid layout.

        Args:
            show_continuum: Whether to plot continuum
            show_line: Whether to plot line
            show_residual: Whether to plot residual
        """
        lines = list(self._spectra.keys())
        n_lines = len(lines)

        fig, axes = plt.subplots(n_lines, 1, figsize=(12, 6 * n_lines))

        if n_lines == 1:
            axes = [axes]

        for idx, line in enumerate(lines):
            ax = axes[idx]
            positions = list(self._spectra[line].keys())
            for position in positions:
                spectrum = self._spectra[line][position]
                spectrum.plot(ax, show_continuum=show_continuum,
                            show_line=show_line, show_residual=show_residual,
                            label=position)
            ax.legend()
            ax.set_title(f"{line.upper()} - All Positions")

        plt.tight_layout()
        plt.show()

    @classmethod
    def load_all(cls, config_path=None):
        """
        Load all spectra from a directory structure.

        Expects directory structure: config_directories.reduced/spectra/line/position_*.fits

        Args:
            config_path: Path to the configuration file. If None, uses CONFIG_PATH.
                         This determines the base directory for loading files.

        Returns:
            SpectrumContainer with loaded spectra
        """
        if config_path is None:
            config_path = CONFIG_PATH

        # Load config to get the reduced directory (use any line, directories are the same)
        config = get_config(line='ti', config_path=config_path, auto_discover_files=True, auto_create_dirs=False)
        base_path = Path(config.directories.reduced) / 'spectra'
        spectra_dict = {}

        for line_dir in base_path.iterdir():
            if line_dir.is_dir():
                line = line_dir.name
                spectra_dict[line] = {}
                # Look for position directories (each containing continuum, line, residual FITS files)
                for position_dir in line_dir.iterdir():
                    if position_dir.is_dir():
                        position = position_dir.name
                        # Load Spectrum from the position directory
                        # Files are named: position_continuum.fits, position_line.fits, position_residual.fits
                        base_filepath = position_dir / position
                        spectra_dict[line][position] = Spectrum.load(base_filepath, line_name=line)

        n_spectra = sum(len(v) for v in spectra_dict.values())
        print(f"Loaded {n_spectra} spectra from {base_path}")
        return cls(spectra_dict)


def convert_to_spectrum(split_data):
    """
    Convert the dictionary output from split_spectrum to Spectrum objects.

    Reorders axes from (scan, slit, wavelength) to (wavelength, slit, scan).

    Args:
        split_data: Dictionary from split_spectrum function

    Returns:
        SpectrumContainer with Spectrum objects organized by line and position
    """
    result = {}
    for line in split_data.keys():
        result[line] = {}
        for position in split_data[line].keys():
            parts = split_data[line][position]

            # Reorder axes for each part: (scan, slit, wavelength) -> (wavelength, slit, scan)
            continuum_data = _reorder_axes(parts['continuum'][0])
            line_data = _reorder_axes(parts['line'][0])
            residual_data = _reorder_axes(parts['residual'][0])

            result[line][position] = Spectrum(
                SpectrumPart(continuum_data, parts['continuum'][1]),
                SpectrumPart(line_data, parts['line'][1]),
                SpectrumPart(residual_data, parts['residual'][1]),
                line_name=line
            )
    return SpectrumContainer(result)


def _reorder_axes(data):
    """
    Reorder data axes from (scan, slit, wavelength) to (wavelength, slit, scan).

    Args:
        data: Intensity array with shape (n_scan, n_slit, n_wavelength) or similar

    Returns:
        Reordered array with shape (n_wavelength, n_slit, n_scan)
    """
    if data is None:
        return None
    if data.ndim == 1:
        return data  # Already 1D, no reordering needed
    elif data.ndim == 2:
        # (scan, wavelength) -> (wavelength, scan)
        return data.T
    elif data.ndim == 3:
        # (scan, slit, wavelength) -> (wavelength, slit, scan)
        return np.transpose(data, (2, 1, 0))
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

def get_formation_height_config(line: str, position: str):
    """
    Get configuration for a specific line and disk position.

    Args:
        line: Spectral line ('ti' or 'sr')
        position: Disk position ('disk_center', 'm40', etc.)

    Returns:
        int: Sequence number for the given line and position
    """
    return POSITION_TO_SEQUENCE[position]

def get_all_lines():
    """
    Get all available spectral lines.

    Returns:
        list: List of line identifiers (e.g., ['ti', 'sr'])
    """
    return AVAILABLE_LINES

def get_all_positions(line: str = None):
    """
    Get all available disk positions.

    Args:
        line: Spectral line (optional, kept for backward compatibility)
              Since positions are shared, this parameter is not used.

    Returns:
        list: List of position strings (e.g., ['disk_center', 'm40', 'm30', ...])
    """
    return list(POSITION_TO_SEQUENCE.keys())

def iterate_positions(line: str = None):
    """
    Iterate over all positions, yielding (position, sequence) pairs.

    Args:
        line: Spectral line (optional, kept for backward compatibility)
              Since positions are shared, this parameter is not used.

    Yields:
        tuple: (position, sequence) for each position
    """
    for position, sequence in POSITION_TO_SEQUENCE.items():
        yield position, sequence

def load_all_l4_for_line(line: str):
    """
    Load L4 data for all positions of a given line.

    Args:
        line: Spectral line ('ti' or 'sr')

    Returns:
        dict: Dictionary mapping position -> (l4_data, header)
    """
    results = {}
    for position, sequence in iterate_positions(line):
        l4_data, header = load_l4_for_position(line, position)
        results[position] = (l4_data, header)
        print(f"Loaded {line.upper()} at {position} (sequence {sequence})")
    return results

def load_l4_for_position(line: str, position: str):
    """
    Load L4 data for a specific line and disk position.

    Args:
        line: Spectral line ('ti' or 'sr')
        position: Disk position ('disk_center', '-40', etc.)

    Returns:
        tuple: (l4_data, header) from tio.read_any_file
    """
    sequence = get_formation_height_config(line, position)
    config = get_config(
        line=line,
        config_path=CONFIG_PATH,
        auto_discover_files=True,
        auto_create_dirs=False
    )
    # Override sequence for this position
    config.dataset['scan']['sequence'] = sequence
    return tio.read_any_file(config, data_type='scan', status='l4')


def calculate_intensity(line: str, position: str, crop=None):
    """
    Calculate intensity for a specific line and position.

    Args:
        line: Spectral line ('ti' or 'sr')
        position: Disk position ('disk_center', '-40', etc.)
        crop: Optional crop specification. Can be:
              - None: no cropping
              - Array: [scan_start, scan_end, slit_start, slit_end, wl_start, wl_end]
              - Dict: {'slit': (start, end), 'scan': (start, end), 'wavelength': (start, end)}

    Returns:
        numpy array: Intensity values with shape (scan, slit, wavelength)
    """
    data, header = load_l4_for_position(line, position)

    # Check that only one map exists in the cycle set
    map_indices = sorted(set(k[2] for k in data.keys()))
    if len(map_indices) > 1:
        raise ValueError(f"Multiple maps found in cycle set for {line} at {position}: {map_indices}. "
                         f"Expected exactly one map.")
    map_idx = map_indices[0]

    if line == 'sr':
        result = tt.compute_polarimetry(data)
        # Collapse the map dimension (axis 1) to match TI shape
        sr_intensity = result.uml.I[:, map_idx, :, :]
        intensity = sr_intensity.squeeze(axis=1) if sr_intensity.shape[1] == 1 else sr_intensity
    elif line == 'ti': # use only lower state, as argued in the data reduction documentation
        # Since we verified only one map exists, stacking all is fine
        intensity = data.get_state('pQ').stack_all('lower')

    # Apply cropping if specified
    if crop is not None:
        # Convert dict to array if needed
        if isinstance(crop, dict):
            crop = crop_to_array(crop)
        scan_start, scan_end, slit_start, slit_end, wl_start, wl_end = crop
        intensity = intensity[scan_start:scan_end, slit_start:slit_end, wl_start:wl_end]

    return intensity
            
    
    

def calculate_intensity_for_all(crop=False):
    """
    Calculate intensity for all lines and positions and plot scan-averaged spectra.

    Args:
        crop: Boolean to control cropping. If True, uses predefined crop configurations
              from CROP_CONFIGS (line-dependent). If False, no cropping is applied.

    Returns:
        tuple: (intensity, wavelength) where intensity is a dictionary mapping
               line -> position -> full intensity array, and wavelength is a
               dictionary mapping line -> wavelength array
    """
    intensity = {}
    wavelength = {}
    scan_avg_intensity = {}  # For plotting only

    for line in get_all_lines():
        print(f"  Line: {line}")
        intensity[line] = {}
        wavelength[line] = {}
        scan_avg_intensity[line] = {}

        # Load wavelength once per line
        config = get_config(
            line=line,
            config_path=CONFIG_PATH,
            auto_discover_files=True,
            auto_create_dirs=False
        )
        wavelength[line], _ = tio.read_any_file(config, data_type='scan', status='wl')

        # Determine crop configuration for this line
        line_crop = CROP_CONFIGS[line] if crop else None

        # Apply wavelength cropping if crop is enabled
        if crop and line_crop is not None:
            wl_crop = line_crop['wavelength']
            wavelength[line] = wavelength[line][wl_crop[0]:wl_crop[1]]

        for position in get_all_positions(line):
            print(f"    Position: {position}")
            intensity_data = calculate_intensity(line, position, line_crop)

            # Store full intensity data
            intensity[line][position] = intensity_data

            # Calculate scan average (mean over scan axis) for plotting
            # Assuming shape is (n_scan, n_wavelength) or (n_scan, n_slit, n_wavelength)
            if intensity_data.ndim == 2:
                scan_avg = np.mean(intensity_data, axis=0)
            elif intensity_data.ndim == 3:
                scan_avg = np.mean(intensity_data, axis=(0, 1))
            else:
                scan_avg = intensity_data

            scan_avg_intensity[line][position] = scan_avg

    # Create figure with 2 subplots (one for each line)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for idx, line in enumerate(get_all_lines()):
        ax = axes[idx]
        wl = wavelength[line]
        for position in get_all_positions(line):
            spec = scan_avg_intensity[line][position]
            ax.plot(wl, spec, label=position)

            # Plot a random pixel spectrum (dashed line)
            intensity_data = intensity[line][position]
            # Get scan dimensions (excluding wavelength)
            if intensity_data.ndim == 2:
                # Shape is (n_scan, n_wavelength)
                random_scan_idx = np.random.randint(0, intensity_data.shape[0])
                pixel_spec = intensity_data[random_scan_idx, :]
            elif intensity_data.ndim == 3:
                # Shape is (n_scan, n_slit, n_wavelength)
                random_scan_idx = np.random.randint(0, intensity_data.shape[0])
                random_slit_idx = np.random.randint(0, intensity_data.shape[1])
                pixel_spec = intensity_data[random_scan_idx, random_slit_idx, :]
            else:
                # Already 1D, use as is
                pixel_spec = intensity_data

            ax.plot(wl, pixel_spec, '--', alpha=0.5, linewidth=0.8)

        # Add single legend entry for dashed lines
        ax.plot([], [], '--', alpha=0.5, linewidth=0.8, label='Random pixel')

        ax.set_title(f"{line.upper()} Scan-Averaged Spectrum")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return intensity, wavelength

def display_one_position(position: str, crop=False):
    """
    Display intensity data for a single position for both lines.

    Args:
        position: Disk position ('disk_center', '-40', etc.)
        crop_sr: Optional crop specification for SR data (dict or array format)
        crop_ti: Optional crop specification for TI data (dict or array format)
    """
     # Determine crop configuration for this line
    line_crop = CROP_CONFIGS['sr'] if crop else None
    sr = calculate_intensity('sr', position, line_crop)
    line_crop = CROP_CONFIGS['ti'] if crop else None
    ti = calculate_intensity('ti', position, line_crop)
    data_plot = np.array([sr, ti])
    viewer = display_data( data_plot, ['states', 'spatial_y', 'spatial_x', 'spectral'],
                          title=position,
                          state_names=['Sr', 'Ti']
                          )

def run_all_preparation_steps(crop=True):

    intensity, wavelength = calculate_intensity_for_all(crop=crop)  # Use line-dependent crop configs
    split_data = split_spectrum(intensity, wavelength, SPLIT_CONFIGS) # also normalized to continuum per position
    
    spectra = convert_to_spectrum(split_data) # also converted to IDL expected wavelength, slit, scan
    # Access: spectra['ti']['disk_center'].continuum.data
    # Access: spectra['ti']['disk_center'].continuum.wvl
    # spectra['ti']['disk_center'].plot()
    # spectra['ti']['disk_center'].plot(scan_idx=10,slit_idx=300)
    spectra.save_all()
    return spectra


#%%
if __name__ == '__main__':

    # Use predefined crop configurations (easier to read and modify)

    # check out one position
    #display_one_position('m40', crop=True)
    
    # Prepare data into a SpectrumContainer class holding all information
    # spectra = run_all_preparation_steps(crop=True)
    # check out mean intensity
    # spectra['ti']['disk_center'].plot()
    
    # once the preparation is finished, you can just load the prepared data
    spectra = SpectrumContainer.load_all()
# -----------------------------------------------------

