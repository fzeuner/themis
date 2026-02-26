#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:46:34 2025

@author: franziskaz
"""

#!/usr/bin/env python3
"""
Determine image transformation in y direction between upper and lower image
using the target data.

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import shift as scipy_shift
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import erf
from numpy.polynomial import Polynomial

from themis.datasets.themis_datasets_2025 import get_config
from themis.core import themis_io as tio

from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation

from spectator.controllers.app_controller import display_data # from spectator
#from napari_simpleitk_image_processing import richardson_lucy_deconvolution - is not working at the moment 2025-12-23 
from skimage.restoration import richardson_lucy

from themis.core import themis_data_reduction as tdr

#%%




def fit_gaussians_along_axis(image, axis=1, fit_min=None, fit_max=None,
                             initial_sigma=5.0, initial_width=10.0,
                             initial_amplitude=None,
                             initial_center=None, initial_offset=None):
    """Fit a 1D rectangular function convolved with a Gaussian to each slice.

    Parameters
    ----------
    image : 2D ndarray
        Input image.
    axis : int, {0,1}
        Axis along which to take slices:
        - axis=0: slice index is y, Gaussian is fitted along x
        - axis=1: slice index is x, Gaussian is fitted along y
    fit_min, fit_max : int or None
        Pixel indices along the *fitted* axis where the Gaussian will be fit.
        If None, use the full range.

    Returns
    -------
    centers : 1D ndarray
        Fitted profile centers (pixel coordinates along the fit axis).
    fwhm : 1D ndarray
        Fitted full-width at half maximum (FWHM) of the rectangular component.
    model_image : 2D ndarray
        Image reconstructed from the fitted rectangular⊗Gaussian profiles.
    coord_along_slices, coord_fit_axis : 1D ndarrays
        Pixel coordinates along the slice index and fit axis, respectively.
    sigmas : 1D ndarray
        Fitted Gaussian sigmas used in the convolution.
    """

    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("image must be 2D")

    # Bring the fit axis to axis=1: (n_slices, n_fit)
    if axis == 0:
        slices = img
    elif axis == 1:
        slices = img.T
    else:
        raise ValueError("axis must be 0 or 1")

    n_slices, n_fit = slices.shape

    if fit_min is None:
        fit_min = 0
    if fit_max is None or fit_max > n_fit:
        fit_max = n_fit

    x = np.arange(fit_min, fit_max)
    centers = np.full(n_slices, np.nan, dtype=float)
    fwhm = np.full(n_slices, np.nan, dtype=float)
    sigmas = np.full(n_slices, np.nan, dtype=float)

    model_slices = np.zeros_like(slices, dtype=float)

    sqrt2 = np.sqrt(2.0)

    def rect_gauss(x, A, x0, width, sigma, C):
        """Rectangular (top-hat) convolved with Gaussian.

        The rectangular has width = FWHM; the Gaussian has sigma.
        """
        half = 0.5 * width
        t1 = (x - x0 + half) / (sqrt2 * sigma)
        t2 = (x - x0 - half) / (sqrt2 * sigma)
        return A * 0.5 * (erf(t1) - erf(t2)) + C

    for i in range(n_slices):
        y = slices[i, fit_min:fit_max].astype(float)
        if not np.any(np.isfinite(y)):
            continue

        # Initial guesses: decide if feature is a peak or a dip
        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
        ymed = np.nanmedian(y)

        if initial_amplitude is None:
            # Compare distance of max/min from median to decide sign
            if (ymax - ymed) >= (ymed - ymin):
                # Positive peak
                A0 = ymax - ymed
            else:
                # Negative dip
                A0 = ymin - ymed  # negative amplitude
        else:
            A0 = initial_amplitude

        # Center guess: user-provided or from max/min
        if initial_center is not None:
            x0_0 = float(initial_center)
        else:
            if A0 >= 0:
                x0_0 = x[np.nanargmax(y)]
            else:
                x0_0 = x[np.nanargmin(y)]

        # Offset guess: user-provided or median
        C0 = ymed if initial_offset is None else float(initial_offset)

        width0 = float(initial_width)
        sigma0 = float(initial_sigma)

        p0 = (A0, x0_0, width0, sigma0, C0)

        try:
            # Constrain parameters: width>0, sigma>0, center within fit window.
            popt, _ = curve_fit(
                rect_gauss,
                x,
                y,
                p0=p0,
                bounds=([-np.inf, fit_min, 0.0, 0.0, -np.inf],
                        [ np.inf, fit_max, np.inf, np.inf,  np.inf]),
                maxfev=5000,
            )
            A, x0, width_fit, sigma_fit, C = popt
            centers[i] = x0
            fwhm[i] = max(width_fit, 0.0)
            sigmas[i] = max(sigma_fit, 0.0)

            # Build fitted slice over full length
            x_full = np.arange(n_fit)
            model_slices[i, :] = rect_gauss(x_full, A, x0, width_fit, sigma_fit, C)
        except Exception:
            continue

    # Transform model_slices back to original orientation
    if axis == 0:
        model_image = model_slices
        coord_along_slices = np.arange(img.shape[0])
        coord_fit_axis = np.arange(img.shape[1])
    else:
        model_image = model_slices.T
        coord_along_slices = np.arange(img.shape[1])
        coord_fit_axis = np.arange(img.shape[0])

    return centers, fwhm, model_image, coord_along_slices, coord_fit_axis, sigmas


def plot_gaussian_slice_fits(image, axis=1, fit_min=None, fit_max=None,
                             initial_sigma=5.0, initial_width=10.0,
                             initial_amplitude=None,
                             initial_center=None, initial_offset=None,
                             cmap='viridis'):
    """Fit rectangular⊗Gaussian profiles along `axis` and create diagnostics.

    Panels:
      1. Input image
      2. Fitted-model image
      3. Input image with profile centers overplotted
      4. Input image with FWHM envelopes (center ± FWHM/2) overplotted
    """

    (centers, fwhm, model_image,
     coord_along_slices, coord_fit_axis, sigmas) = fit_gaussians_along_axis(
        image,
        axis=axis,
        fit_min=fit_min,
        fit_max=fit_max,
        initial_sigma=initial_sigma,
        initial_width=initial_width,
        initial_amplitude=initial_amplitude,
        initial_center=initial_center,
        initial_offset=initial_offset,
    )

    img = np.asarray(image)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel 1: input image
    im1 = ax1.imshow(img, origin='lower', aspect='auto', cmap=cmap)
    ax1.set_title("Input image")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Panel 2: fitted model image
    im2 = ax2.imshow(model_image, origin='lower', aspect='auto', cmap=cmap)
    ax2.set_title("Fitted rectangular⊗Gaussian model image")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Prepare center and FWHM/2 coordinates
    if axis == 0:
        x_centers = centers
        y_centers = coord_along_slices
        x_minus = centers - 0.5 * fwhm
        x_plus = centers + 0.5 * fwhm
        y_line = coord_along_slices
    else:
        x_centers = coord_along_slices
        y_centers = centers
        x_line = coord_along_slices
        y_minus = centers - 0.5 * fwhm
        y_plus = centers + 0.5 * fwhm

    # Panel 3: centers over input image
    ax3.imshow(img, origin='lower', aspect='auto', cmap=cmap)
    ax3.plot(x_centers, y_centers, 'r.', ms=3, label='Profile center')
    ax3.set_title("Centers of fitted profiles")
    ax3.legend(loc='best')

    # Panel 4: FWHM envelopes (center ± FWHM/2) over input image
    ax4.imshow(img, origin='lower', aspect='auto', cmap=cmap)
    if axis == 0:
        ax4.plot(x_minus, y_line, 'r--', lw=1, label='center - FWHM/2')
        ax4.plot(x_plus,  y_line, 'b--', lw=1, label='center + FWHM/2')
    else:
        ax4.plot(x_line, y_minus, 'r--', lw=1, label='center - FWHM/2')
        ax4.plot(x_line, y_plus,  'b--', lw=1, label='center + FWHM/2')
    ax4.set_title("FWHM envelopes of fitted profiles")
    ax4.legend(loc='best')

    for ax in axes.ravel():
        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')

    # ------------------------------------------------------------
    # Additional 1D diagnostic: first slice data vs fitted profile
    # ------------------------------------------------------------
    # Determine local fit window used for the 1D plot
    if axis == 0:
        n_fit = img.shape[1]
    else:
        n_fit = img.shape[0]

    fit_min_loc = 0 if fit_min is None else fit_min
    fit_max_loc = n_fit if fit_max is None or fit_max > n_fit else fit_max
    x_window = np.arange(fit_min_loc, fit_max_loc)

    if axis == 0:
        data_1d = img[0, fit_min_loc:fit_max_loc]
        model_1d = model_image[0, fit_min_loc:fit_max_loc]
    else:
        data_1d = img[fit_min_loc:fit_max_loc, 0]
        model_1d = model_image[fit_min_loc:fit_max_loc, 0]

    fig_slice, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(x_window, data_1d, 'k.', label='data (slice 0)')
    ax.plot(x_window, model_1d, 'r-', label='rect⊗Gauss fit')

    # Also show the initial rect⊗Gauss guess used for the fit (slice 0)
    # Reconstruct the same initial p0 logic for this slice.
    y = data_1d.astype(float)
    if np.any(np.isfinite(y)):
        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
        ymed = np.nanmedian(y)

        if initial_amplitude is None:
            if (ymax - ymed) >= (ymed - ymin):
                A0 = ymax - ymed
            else:
                A0 = ymin - ymed
        else:
            A0 = initial_amplitude

        if initial_center is not None:
            x0_0 = float(initial_center)
        else:
            # Note: x_window is in the global fit-axis coordinates
            if A0 >= 0:
                x0_0 = x_window[np.nanargmax(y)]
            else:
                x0_0 = x_window[np.nanargmin(y)]

        C0 = ymed if initial_offset is None else float(initial_offset)

        width0 = float(initial_width)
        sigma0 = float(initial_sigma)

        sqrt2 = np.sqrt(2.0)

        def _rect_gauss(x, A, x0, width, sigma, C):
            half = 0.5 * width
            t1 = (x - x0 + half) / (sqrt2 * sigma)
            t2 = (x - x0 - half) / (sqrt2 * sigma)
            return A * 0.5 * (erf(t1) - erf(t2)) + C

        y_init = _rect_gauss(x_window, A0, x0_0, width0, sigma0, C0)
        ax.plot(x_window, y_init, 'b--', label='initial guess')

    ax.set_xlabel('Pixel along fit axis')
    ax.set_ylabel('Intensity')
    ax.set_title('First slice: data and fitted rect⊗Gauss')
    ax.legend(loc='best')

    # ------------------------------------------------------------
    # Additional figure: centers, FWHM, and Gaussian sigma vs slice index
    # with quadratic fits for center and sigma.
    # ------------------------------------------------------------
    fig_params, axes_params = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # x-axis: slice index (coord_along_slices)
    x_slices = coord_along_slices.astype(float)

    # Quadratic fit for centers vs slice (if enough valid points)
    valid_centers = np.isfinite(centers)
    if np.count_nonzero(valid_centers) >= 3:
        x_c = x_slices[valid_centers]
        y_c = centers[valid_centers]
        # Use Polynomial.fit for better numerical stability and then
        # convert back to the standard power basis to get coefficients.
        p_center = Polynomial.fit(x_c, y_c, deg=2)
        p_center_std = p_center.convert()  # standard basis in x
        center_coeffs = p_center_std.coef  # [a0, a1, a2]
        center_fit = p_center(x_slices)
    else:
        center_coeffs = np.array([np.nan, np.nan, np.nan], dtype=float)
        center_fit = np.full_like(x_slices, np.nan, dtype=float)

    # Quadratic fit for sigmas vs slice (if enough valid points), with
    # rejection of outliers that are much larger than the mean.
    valid_sigmas = np.isfinite(sigmas)
    if np.count_nonzero(valid_sigmas) >= 2:
        y_all = sigmas[valid_sigmas]
        mu = np.mean(y_all)
        std = np.std(y_all)
        # keep values not much larger than the mean
        keep = y_all <= (mu + 3.0 * std)
        if np.count_nonzero(keep) >= 2:
            x_s = x_slices[valid_sigmas][keep]
            y_s = y_all[keep]
        else:
            # fall back to using all finite sigmas
            x_s = x_slices[valid_sigmas]
            y_s = y_all

        p_sigma = Polynomial.fit(x_s, y_s, deg=2)
        p_sigma_std = p_sigma.convert()
        sigma_coeffs = p_sigma_std.coef  # [b0, b1, b2]
        sigma_fit = p_sigma(x_slices)
    else:
        sigma_coeffs = np.array([np.nan, np.nan, np.nan], dtype=float)
        sigma_fit = np.full_like(x_slices, np.nan, dtype=float)

    # Plot parameters and their quadratic fits
    axes_params[0].plot(x_slices, centers, 'k.', label='center')
    axes_params[0].plot(x_slices, center_fit, 'r-', label='quad fit')
    axes_params[0].set_ylabel('Center [px]')
    axes_params[0].set_title('Fitted profile parameters vs slice index')
    axes_params[0].legend(loc='best')

    axes_params[1].plot(x_slices, fwhm, 'b.-')
    axes_params[1].set_ylabel('FWHM [px]')

    axes_params[2].plot(x_slices, sigmas, 'r.', label='sigma')
    axes_params[2].plot(x_slices, sigma_fit, 'm-', label='quad fit')
    axes_params[2].set_ylabel('Sigma [px]')
    axes_params[2].set_xlabel('Slice index')
    axes_params[2].legend(loc='best')

    # Set y-limits to include central 95% of data in each subplot
    # Panel 0: centers and center_fit
    y0 = np.concatenate([
        centers[np.isfinite(centers)],
        center_fit[np.isfinite(center_fit)],
    ])
    if y0.size > 0:
        y0_sorted = np.sort(y0)
        lo = np.percentile(y0_sorted, 2.5)
        hi = np.percentile(y0_sorted, 97.5)
        axes_params[0].set_ylim(lo, hi)

    # Panel 1: FWHM
    y1 = fwhm[np.isfinite(fwhm)]
    if y1.size > 0:
        y1_sorted = np.sort(y1)
        lo = np.percentile(y1_sorted, 2.5)
        hi = np.percentile(y1_sorted, 97.5)
        axes_params[1].set_ylim(lo, hi)

    # Panel 2: sigmas and sigma_fit
    y2 = np.concatenate([
        sigmas[np.isfinite(sigmas)],
        sigma_fit[np.isfinite(sigma_fit)],
    ])
    if y2.size > 0:
        y2_sorted = np.sort(y2)
        lo = np.percentile(y2_sorted, 2.5)
        hi = np.percentile(y2_sorted, 97.5)
        axes_params[2].set_ylim(lo, hi)

    fig_params.tight_layout()

    return fig, (centers, fwhm, center_fit, sigma_fit)


def reduce_all(config):
    lv0 = tdr.reduction_levels["l0"]
    
    data_types = ['dark', 'flat', 'flat_center', 'scan']
    for data_type in data_types:
       result = lv0.reduce(config, data_type=data_type, return_reduced = False)

    # lv1 = tdr.reduction_levels["l1"]
    # result = lv1.reduce(config, data_type='flat_center', return_reduced = False)
    return(0)
   
#%%

if __name__ == '__main__':
    """Main function."""
    print("="*70)
    print("Calibration data: determine image y shifts")
    print("="*70)

    # Load configuration
    config = get_config(
        config_path='configs/calibration_data_sr_2025-07-05.toml',
        auto_discover_files=True
    )

    # reduce (only once)
    #result = reduce_all(config)
    #%%  
    scan, header = tio.read_any_file(config, 'scan', verbose=False, status='l1')
    pQ_scan = scan.get_state('pQ')
    upper_scan = pQ_scan.stack_all('upper')
    lower_scan = pQ_scan.stack_all('lower')
    
   
    
    
    #%%
    fig, (centers_lower, fwhm_lower, center_lower, sigma_lower) = plot_gaussian_slice_fits(
    lower_scan[0],
    axis=1,        # slices along y, fit along x
    fit_min=498,
    fit_max=532,initial_sigma=5, initial_width=10.0,
                          initial_amplitude=-8000,
                          initial_center=515, initial_offset=8700
    )
    plt.show()

    fig, (centers_upper, fwhm_upper, center_upper, sigma_upper) = plot_gaussian_slice_fits(
    upper_scan[0],
    axis=1,        # slices along y, fit along x
    fit_min=498,
    fit_max=532,initial_sigma=5, initial_width=10.0,
                          initial_amplitude=-8000,
                          initial_center=515, initial_offset=8700
    )
    plt.show()

    # Define shifts (y-shifts) from fitted center curves
    fixed_value = center_lower[0]
    lower_shifts = center_lower - fixed_value
    upper_shifts = center_upper - fixed_value


    def xshift_to_2d_map(x_shift, ny):
        x_shift = np.asarray(x_shift, dtype=float)
        if x_shift.ndim != 1:
            raise ValueError(f"x_shift must be 1D (nx,), got shape {x_shift.shape}")
        return np.repeat(x_shift[None, :], int(ny), axis=0)


    def xshift_fade_from_y(x_shift_at_y, ny, y_ref=515, length=485, direction='up'):
        x_shift_at_y = np.asarray(x_shift_at_y, dtype=float)
        if x_shift_at_y.ndim != 1:
            raise ValueError(f"x_shift_at_y must be 1D (nx,), got shape {x_shift_at_y.shape}")
        if length <= 0:
            raise ValueError("length must be > 0")

        y = np.arange(int(ny), dtype=float)

        if direction == 'up':
            w = 1.0 - ((y - float(y_ref)) / float(length))
            w = np.where((y >= float(y_ref)) & (y <= float(y_ref + length)), w, 0.0)
            w = np.clip(w, 0.0, 1.0)
        elif direction == 'down':
            w = 1.0 - ((float(y_ref) - y) / float(length))
            w = np.where((y >= float(y_ref - length)) & (y <= float(y_ref)), w, 0.0)
            w = np.clip(w, 0.0, 1.0)
        elif direction == 'both':
            w = 1.0 - (np.abs(y - float(y_ref)) / float(length))
            w = np.where(np.abs(y - float(y_ref)) <= float(length), w, 0.0)
            w = np.clip(w, 0.0, 1.0)
        else:
            raise ValueError("direction must be one of {'up','down','both'}")

        return w[:, None] * x_shift_at_y[None, :]

    #%%

    config_test = get_config(config_path='configs/sample_dataset_sr_2025-07-07.toml')

    #%%
    scan, header = tio.read_any_file(config_test, 'scan', verbose=False, status='l1')

    upper = scan.get_state('pQ').stack_all('upper')[0]
    lower = scan.get_state('pQ').stack_all('lower')[0]

    lower_shift_map = xshift_to_2d_map(lower_shifts, ny=lower.shape[0])
    upper_shift_map = xshift_to_2d_map(upper_shifts, ny=upper.shape[0])

    lower_shifted = tdr._apply_yshift_map_to_half(lower, lower_shift_map)
    upper_shifted = tdr._apply_yshift_map_to_half(upper, upper_shift_map)

    shift_candidates = np.arange(-5, 5 + 0.1, 0.1)
    best_metric_1 = np.inf
    optimal_shift_1 = 0.0
    best_metric_2 = np.inf
    optimal_shift_2 = 0.0

    for step in shift_candidates:
        y_shift_additional = -step * np.ones(lower.shape[1])
        lower_shifted_tmp = tdr._apply_yshift_to_half(lower_shifted, y_shift_additional)
        shifted_diff_tmp = upper_shifted - lower_shifted_tmp
        current_contrast_1 = shifted_diff_tmp[30:50, 90:360].mean(axis=1).std()
        current_contrast_2 = shifted_diff_tmp[30:50, 700:1000].mean(axis=1).std()
        if current_contrast_1 < best_metric_1:
            best_metric_1 = current_contrast_1
            optimal_shift_1 = float(step)
        if current_contrast_2 < best_metric_2:
            best_metric_2 = current_contrast_2
            optimal_shift_2 = float(step)
            #print(current_contrast_1)
            #print(current_contrast_2)
    print(optimal_shift_1, optimal_shift_2)  # if both are the same then I think that there is no "additional" rotation
    x0 = 90#0.5 * (90 + 360)
    x1 = 0.5 * (700 + 1000)
    nx = lower.shape[1]
    xx = np.arange(nx, dtype=float)
    s0 = -optimal_shift_1
    s1 = -optimal_shift_2
    y_shift_additional = s0 + (s1 - s0) * (xx - x0) / (x1 - x0)
    y_shift_additional = np.where(xx < min(x0, x1), s0, y_shift_additional)
    y_shift_additional = np.where(xx > max(x0, x1), s1, y_shift_additional)

    #%%

    additional_shift_map = xshift_fade_from_y(y_shift_additional, ny=lower.shape[0], y_ref=40, length=300, direction='both')

    total_shift_map_lower = lower_shift_map + additional_shift_map

    # xshift_fade_from_y(upper_shifts, ny=upper.shape[0], y_ref=515, length=400, direction='both') # not as good as the above

    lower_final = tdr._apply_yshift_map_to_half(lower, total_shift_map_lower)

    #%%
    shifted_diff = upper_shifted - lower_final

    data_plot = np.array([shifted_diff, upper - lower])

    viewer = display_data( data_plot, ['states',  'spatial_x', 'spectral'],
                       title='scan', 
                       state_names=['y shifted', 'original'
                                    ])

    #%%
    # IF YOU ARE HAPPY WITH THE ABOVE
    # Save shifts as NumPy files and register as auxiliary files for flat_center
    line = config.dataset['line']
    seq_fc = config.dataset['flat_center']['sequence']
    seq_fc_str = f"t{seq_fc:03d}"

    shifts_lower_path = Path(config.directories.reduced) / f"{line}_calibration_target_yshift_lower.npy"
    shifts_upper_path = Path(config.directories.reduced) / f"{line}_calibration_target_yshift_upper.npy"

    np.save(shifts_lower_path, total_shift_map_lower)
    np.save(shifts_upper_path, upper_shift_map)

    files_fc = config.dataset['flat_center']['files']
    if not hasattr(files_fc, 'auxiliary'):
        files_fc.auxiliary = {}

    files_fc.auxiliary['yshift_lower'] = shifts_lower_path
    files_fc.auxiliary['yshift_upper'] = shifts_upper_path

    print(f"Saved y-shift auxiliary files for flat_center: {shifts_lower_path.name}, {shifts_upper_path.name}")    
    
    
 #%%
     
    
#OLD STUFF   



# def convolve_slices_with_gaussian(image, sigmas, slice_axis=1,
#                                   mode='reflect', cval=0.0):
#     """Convolve each slice of a 2D image with a 1D Gaussian.

#     Parameters
#     ----------
#     image : 2D ndarray
#         Input image.
#     sigmas : 1D array-like
#         Gaussian sigma for each slice along `slice_axis`.
#         Length must equal ``image.shape[slice_axis]``.
#     slice_axis : int, {0,1}
#         Axis that indexes slices. Convolution is performed along the
#         *perpendicular* axis, i.e. along axis 1 if ``slice_axis == 0``
#         and along axis 0 if ``slice_axis == 1``.
#     mode, cval :
#         Passed through to ``scipy.ndimage.gaussian_filter1d``.

#     Returns
#     -------
#     convolved : 2D ndarray (float)
#         Image where each slice has been convolved with a Gaussian of
#         the corresponding sigma. If a sigma is non-finite or <= 0, the
#         original slice is copied unchanged.
#     """

#     img = np.asarray(image, dtype=float)
#     if img.ndim != 2:
#         raise ValueError("image must be 2D")

#     sigmas = np.asarray(sigmas, dtype=float)
#     if sigmas.ndim != 1:
#         raise ValueError("sigmas must be 1D (one value per slice)")

#     if slice_axis not in (0, 1):
#         raise ValueError("slice_axis must be 0 or 1")

#     n_slices = img.shape[slice_axis]
#     if sigmas.size != n_slices:
#         raise ValueError(
#             f"sigmas length {sigmas.size} does not match number of slices {n_slices}"
#         )

#     out = np.empty_like(img, dtype=float)

#     if slice_axis == 0:
#         # Slices are rows; convolve along columns (axis=1)
#         for i in range(n_slices):
#             s = sigmas[i]
#             row = img[i, :]
#             if not np.isfinite(s) or s <= 0:
#                 out[i, :] = row
#             else:
#                 out[i, :] = gaussian_filter1d(row, sigma=s, axis=-1,
#                                               mode=mode, cval=cval)
#     else:
#         # Slices are columns; convolve along rows (axis=0)
#         for j in range(n_slices):
#             s = sigmas[j]
#             col = img[:, j]
#             if not np.isfinite(s) or s <= 0:
#                 out[:, j] = col
#             else:
#                 out[:, j] = gaussian_filter1d(col, sigma=s, axis=0,
#                                               mode=mode, cval=cval)

#     return out

# perfect_psf = np.zeros( (25,25) )
# perfect_psf[12, 12] = 1
# psf = gaussian_filter(perfect_psf, sigma=2.5)
# psf/= psf.sum()
# number_of_iterations = 10
# deconvolved = richardson_lucy(
#     upper_scan[0],
#     psf,
#     num_iter=number_of_iterations,  # <-- correct name
#     clip=False,
# )
# deconvolved_s= deconvolved*upper_scan[0].mean() / deconvolved.mean()
# fig, (centers_upper, fwhm, center_coeffs_upper, sigma_coeffs_upper) = plot_gaussian_slice_fits(
# deconvolved_s,
# axis=1,        # slices along y, fit along x
# fit_min=498,
# fit_max=532,initial_sigma=5, initial_width=10.0,
#                       initial_amplitude=-8000,
#                       initial_center=515, initial_offset=8700
# )
# plt.show()    
    
    
    
 
# refocused_lower = convolve_slices_with_gaussian(lower_scan[0], np.sqrt(3.5**2-sigma_lower**2), slice_axis=1,
#                                       mode='reflect', cval=0.0)
# fig, (centers_lower2, fwhm2, center_lower2, sigma_lower2) = plot_gaussian_slice_fits(
# refocused_lower,
# axis=1,        # slices along y, fit along x
# fit_min=498,
# fit_max=532,initial_sigma=5, initial_width=10.0,
#                       initial_amplitude=-8000,
#                       initial_center=515, initial_offset=8700
# )
# plt.show()
    

    
#   #  #%%
# refocused_upper = convolve_slices_with_gaussian(upper_scan[0], np.sqrt(3.5**2-sigma_upper**2), slice_axis=1,
#                                       mode='reflect', cval=0.0)
# fig, (centers_upper2, fwhm2, center_upper2, sigma_upper2) = plot_gaussian_slice_fits(
# refocused_upper,
# axis=1,        # slices along y, fit along x
# fit_min=498,
# fit_max=532,initial_sigma=5, initial_width=10.0,
#                       initial_amplitude=-8000,
#                       initial_center=515, initial_offset=8700
# )
# plt.show()
    
#     #%%
# plt.plot(sigma_lower2, label='convolved sigmas lower')   
# plt.plot(sigma_lower, label='original sigmas lower')
# plt.plot(sigma_upper2, label='convolved sigmas upper')   
# plt.plot(sigma_upper, label='original sigmas upper')  
# plt.legend() 
    
# #%%
#         # Crop edges to avoid border effects
# margin = 50
# upper_scan_norm = (upper_scan[0] / np.mean(upper_scan[0]))[margin:-margin, margin:-margin]
# lower_scan_norm = (lower_scan[0] / np.mean(lower_scan[0]))[margin:-margin, margin:-margin]

# shift_xy, _, _ = phase_cross_correlation(upper_scan_norm, lower_scan_norm, upsample_factor=10) # means 0.1 pixel

# shifted_fft = fourier_shift(np.fft.fftn(lower_scan_norm), shift_xy)
# lower_scan_aligned = np.fft.ifftn(shifted_fft).real

# #fig = ird.imreg.imshow(upper_scan_norm, lower_scan_norm, result['timg'], cmap ='plasma_r')
# plt.imshow(upper_scan_norm- lower_scan_aligned)
# #%%
# data_plot = np.array([upper_scan[0],lower_scan[0] ])
# viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
#                    title='target scan', 
#                   state_names=['upper', 'lower'])
    
# #%%

    
# # result = ird.similarity(
# #     upper_scan_norm, 
# #     lower_scan_norm, # preshifted for better results
# #     numiter=5,
# #     constraints={
# #     'tx': (-1,1),      # x translation bounds
# #     'ty': (-2, 2),      # y translation bounds  
# #     'angle': (-0.5,0.5),     # rotation bounds in degrees
# #     'scale': (0.9, 1.1),  # scale bounds
# #     }
# # )
# lower_scan_aligned = scipy_shift(lower_scan_norm, (0, -1), order=3, mode='constant')
# # lower_scan_aligned_rotated = ird.transform_img(lower_scan_aligned,tvec=[0,0], angle=-0.5, scale=1)
# # upper_scan_rotated = ird.transform_img(lower_scan_norm,tvec=[0,0], angle=-0.3, scale=1)
# # print(f"   ✓ Shift: ({result['tvec'][0]:.3f}, {result['tvec'][1]:.3f}) px")
# # print(f"     Rotation: {result['angle']:.4f}°")
# # print(f"     Scale: {result['scale']:.6f}")
   

# viewer = display_data( np.array([upper_scan_norm,lower_scan_aligned_rotated, upper_scan_norm-lower_scan_aligned_rotated]), 'states',  'spatial', 'spectral',
#                    title='target scan', 
#                   state_names=['upper', 'lower aligned', 'upper-lower aligned'])
    