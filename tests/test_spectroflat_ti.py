#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import tempfile
import shutil

import numpy as np
from astropy.io import fits

from themis.datasets.themis_datasets_2025 import get_config
from themis.core import themis_io as tio


def _run_spectroflat_single(frame_2d, tmp_dir, report_dir=None):
    project_root = Path(__file__).resolve().parents[1]
    run_script = project_root / "scripts" / "run_spectroflat"

    if report_dir is None:
        report_dir = Path(tempfile.mkdtemp(prefix="spectroflat_report_"))
    else:
        report_dir = Path(report_dir)
        report_dir.mkdir(exist_ok=True, parents=True)

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    in_fits = tmp_dir / "input_frame.fits"
    out_offset = tmp_dir / "offset_map.fits"
    out_illum = tmp_dir / "illumination_pattern.fits"
    out_dust = tmp_dir / "dust_flat.fits"

    fits.PrimaryHDU(data=frame_2d.astype("float32")).writeto(in_fits, overwrite=True)

    print("\nPlease run this command in an EXTERNAL terminal:")
    print("-" * 68)
    print("conda activate atlas-fit")
    print(f"python {run_script} \\")
    print(f"    {in_fits} \\")
    print(f"    {out_offset} \\")
    print(f"    {out_illum} \\")
    print(f"    --report_dir {report_dir} \\")
    print(f"    --dust_flat_out {out_dust}")
    print("-" * 68)

    while True:
        user_input = input("Did run_spectroflat complete successfully? (y/n): ").strip().lower()
        if user_input == "y":
            break
        if user_input == "n":
            raise RuntimeError("run_spectroflat aborted by user")
        print('Please enter "y" or "n"')

    if not out_dust.exists():
        raise RuntimeError(f"Dust flat output not found: {out_dust}")
    if not out_offset.exists():
        raise RuntimeError(f"Offset map output not found: {out_offset}")

    with fits.open(out_dust) as hdul:
        dust_flat = np.array(hdul[0].data, dtype="float32")

    with fits.open(out_offset) as hdul:
        offset_map = np.array(hdul[0].data, dtype="float32")
        offset_err = np.array(hdul[1].data, dtype="float32") if len(hdul) > 1 else None

    if offset_map.ndim == 3:
        offset_map = offset_map[0]
    if offset_err is not None and offset_err.ndim == 3:
        offset_err = offset_err[0]

    return dust_flat, offset_map, offset_err


def _split_indices(n_rows, overlap=15, extra_overlap=20):
    overlap_total = overlap + extra_overlap
    mid = n_rows // 2

    top_start = 0
    top_end = min(n_rows, mid + overlap_total)

    bottom_start = max(0, mid - overlap_total)
    bottom_end = n_rows

    return (top_start, top_end), (bottom_start, bottom_end)


def _stitch_rows(shape, segments):
    acc = np.zeros(shape, dtype="float64")
    wgt = np.zeros(shape, dtype="float64")

    for arr, row_start, row_end in segments:
        acc[row_start:row_end, :] += arr
        wgt[row_start:row_end, :] += 1.0

    wgt[wgt == 0] = 1.0
    stitched = acc / wgt
    return stitched.astype("float32")


def _valid_segment(arr, row_start, row_end, nx):
    if arr is None:
        return False
    if arr.ndim != 2:
        return False
    expected_rows = row_end - row_start
    return arr.shape == (expected_rows, nx)


def _trim_segment(arr, row_start, row_end, trim_top=0, trim_bottom=0):
    new_start = row_start + trim_top
    new_end = row_end - trim_bottom
    if new_end <= new_start:
        raise ValueError(
            f"Invalid trim: [{row_start}:{row_end}) with trim_top={trim_top}, trim_bottom={trim_bottom}"
        )
    return arr[trim_top : arr.shape[0] - trim_bottom, :], new_start, new_end


def run_split_spectroflat_ti_upper(config_path="configs/sample_dataset_ti_2025-07-07.toml"):
    config = get_config(
        config_path=config_path,
        auto_discover_files=True,
        auto_create_dirs=False,
    )

    data_type = "flat_center"
    l0_frames, _ = tio.read_any_file(config, data_type, verbose=False, status="l0")
    frame0 = l0_frames.get(0)
    if frame0 is None:
        raise RuntimeError("No L0 flat_center frame found. Run raw->l0 first.")

    upper_2d = frame0.get_half("upper").data.astype("float32")
    ny, nx = upper_2d.shape

    line = config.dataset["line"]
    seq = config.dataset[data_type]["sequence"]
    seq_str = f"t{seq:03d}"
    reduced_dir = Path(config.directories.reduced)

    temp_root = reduced_dir / f"temp_split_spectroflat_{line}_{data_type}_{seq_str}_upper"
    tmp_top = temp_root / "top"
    tmp_bottom = temp_root / "bottom"
    temp_root.mkdir(exist_ok=True, parents=True)

    overlap = 15
    extra_overlap = 20
    (top_start, top_end), (bot_start, bot_end) = _split_indices(ny, overlap=overlap, extra_overlap=extra_overlap)

    upper_top = upper_2d[top_start:top_end, :]
    upper_bottom = upper_2d[bot_start:bot_end, :]

    report_base = Path(config.directories.figures) / "spectroflat_report" / "upper_split_ti"
    dust_top, omap_top, oerr_top = _run_spectroflat_single(upper_top, tmp_dir=tmp_top, report_dir=report_base / "top")
    dust_bot, omap_bot, oerr_bot = _run_spectroflat_single(upper_bottom, tmp_dir=tmp_bottom, report_dir=report_base / "bottom")

    dust_top, top_start_trim, top_end_trim = _trim_segment(
        dust_top,
        top_start,
        top_end,
        trim_bottom=extra_overlap,
    )
    dust_bot, bot_start_trim, bot_end_trim = _trim_segment(
        dust_bot,
        bot_start,
        bot_end,
        trim_top=extra_overlap,
    )

    omap_top, _, _ = _trim_segment(
        omap_top,
        top_start,
        top_end,
        trim_bottom=extra_overlap,
    )
    omap_bot, _, _ = _trim_segment(
        omap_bot,
        bot_start,
        bot_end,
        trim_top=extra_overlap,
    )

    if oerr_top is not None:
        oerr_top, _, _ = _trim_segment(
            oerr_top,
            top_start,
            top_end,
            trim_bottom=extra_overlap,
        )
    if oerr_bot is not None:
        oerr_bot, _, _ = _trim_segment(
            oerr_bot,
            bot_start,
            bot_end,
            trim_top=extra_overlap,
        )

    dust_stitched = _stitch_rows(
        (ny, nx),
        [
            (dust_top, top_start_trim, top_end_trim),
            (dust_bot, bot_start_trim, bot_end_trim),
        ],
    )
    omap_stitched = _stitch_rows(
        (ny, nx),
        [
            (omap_top, top_start_trim, top_end_trim),
            (omap_bot, bot_start_trim, bot_end_trim),
        ],
    )
    oerr_stitched = None
    has_valid_oerr = (
        _valid_segment(oerr_top, top_start_trim, top_end_trim, nx)
        and _valid_segment(oerr_bot, bot_start_trim, bot_end_trim, nx)
    )
    print(
        "Offset error shapes: "
        f"top={None if oerr_top is None else oerr_top.shape}, "
        f"bottom={None if oerr_bot is None else oerr_bot.shape}"
    )
    if has_valid_oerr:
        oerr_stitched = _stitch_rows(
            (ny, nx),
            [
                (oerr_top, top_start_trim, top_end_trim),
                (oerr_bot, bot_start_trim, bot_end_trim),
            ],
        )
    else:
        print("Offset-map error HDU missing/invalid for at least one split; writing stitched offset map without error HDU.")

    dust_flat_out = reduced_dir / f"{line}_{data_type}_{seq_str}_dust_flat_upper.fits"
    offset_map_out = reduced_dir / f"{line}_{data_type}_{seq_str}_offset_map_upper.fits"

    fits.PrimaryHDU(data=dust_stitched).writeto(dust_flat_out, overwrite=True)
    if oerr_stitched is None:
        fits.PrimaryHDU(data=omap_stitched).writeto(offset_map_out, overwrite=True)
    else:
        fits.HDUList([
            fits.PrimaryHDU(data=omap_stitched),
            fits.ImageHDU(data=oerr_stitched),
        ]).writeto(offset_map_out, overwrite=True)

    if temp_root.exists():
        shutil.rmtree(temp_root)
        print(f"Removed temporary split folder: {temp_root}")

    print("=" * 70)
    print("Split spectroflat finished for TI upper frame")
    print(f"Input frame shape: {upper_2d.shape}")
    print(f"Top rows:    [{top_start}:{top_end})")
    print(f"Bottom rows: [{bot_start}:{bot_end})")
    print(f"Trimmed stitch rows: top=[{top_start_trim}:{top_end_trim}), bottom=[{bot_start_trim}:{bot_end_trim})")
    print("Overlap handling: 15 px target overlap + 20 px extra trimmed at cut edges")
    print(f"Saved dust flat:  {dust_flat_out}")
    print(f"Saved offset map: {offset_map_out}")
    print("=" * 70)

    return {
        "dust_flat_upper": dust_flat_out,
        "offset_map_upper": offset_map_out,
    }


if __name__ == "__main__":
    run_split_spectroflat_ti_upper(config_path="configs/formation_dataset_ti_2025-07-05.toml")
