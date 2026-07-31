"""
GUI for comparing our line level maps with IDL run_voigt_as.pro output side-by-side.

Usage:
    uv run python scripts/compare_line_levels.py

Controls:
    - Line dropdown: select spectral line (ti or sr)
    - Position dropdown: select limb distance (disk_center, m40, m30, ...)
    - Level slider: drag the vertical line in the level bar to select level 0-24
    - Left panel: our line levels, Right panel: IDL tc line levels
"""

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from astropy.io import fits

from process_formation_height_line_levels import (
    SpectrumContainer, POSITION_TO_SEQUENCE, AVAILABLE_LINES, CONFIG_PATH,
    get_config,
)


def _load_idl_tc_levels(line, position, spec_res_angstrom):
    """Load IDL tc levels FITS file for a given line and position.

    File naming: {line}_{position}_fit_levels_tc.fits
    Axes in file: (n_levels, n_slit, n_scan)

    HDU1 contains 14 fit parameters per pixel (n_params, n_slit, n_scan):
        12: Half width at 0.95*blue wing continuum (blue, in spectral res units)
        13: Half width at 0.95*blue wing continuum (red, in spectral res units)

    The 25 levels are evenly distributed between 0 and 0.95*CLEVEL (inclusive).
    Width at level i = (i / (n_levels-1)) * (param13 + param14) * spec_res

    Returns: dict with 'intensities' (n_levels, n_slit, n_scan) and 'widths' (mean, in Angstrom) or None.
    """
    config = get_config(line='ti', config_path=CONFIG_PATH,
                        auto_discover_files=True, auto_create_dirs=False)
    base_path = Path(config.directories.reduced) / 'spectra'
    filepath = base_path / line / position / f"{line}_{position}_fit_levels_tc.fits"
    if not filepath.exists():
        return None
    try:
        with fits.open(filepath) as hdul:
            levels = hdul[0].data  # (n_levels, n_slit, n_scan)
            n_levels = levels.shape[0]

            # Compute per-pixel total width at 0.95*continuum from params 13+14
            widths = None
            if len(hdul) > 1:
                params = hdul[1].data  # (n_params, n_slit, n_scan)
                if params.ndim == 3 and params.shape[0] >= 14:
                    # param 13 (idx 12) + param 14 (idx 13) = total half-width at 0.95
                    # in spectral resolution units
                    total_width_095 = (params[12] + params[13]) * spec_res_angstrom
                    # Mean over spatial pixels for a representative width array
                    mean_total = np.nanmean(total_width_095)
                    widths = np.linspace(0, mean_total, n_levels)
            return {
                'intensities': levels,
                'widths': widths,
            }
    except Exception as e:
        print(f"Error loading IDL tc file {filepath}: {e}")
        return None


class CompareLineLevelsInspector(QtWidgets.QMainWindow):
    def __init__(self, spectra):
        super().__init__()
        self.spectra = spectra
        self.setWindowTitle("Compare Line Levels: Python vs IDL")
        self.resize(1400, 800)

        self.positions = list(POSITION_TO_SEQUENCE.keys())
        self.current_position = self.positions[0]
        self.current_line = AVAILABLE_LINES[0]
        self.level_idx = 0

        # Two panels: 'ours' and 'idl'
        self.panel_keys = ['ours', 'idl']
        self.rects = {k: [] for k in self.panel_keys}
        self._rect_items = {k: [] for k in self.panel_keys}
        self.rect_mode = False
        self._rect_start = None
        self._preview_item = None
        self._dragging_rect = None

        # Determine spatial dimensions and n_levels from first available data
        self.n_slit = 1
        self.n_scan = 1
        self.n_levels = 25
        for line in AVAILABLE_LINES:
            for position in self.positions:
                spec = self._get_spectrum(line, position)
                if spec is not None and spec.fit_line is not None:
                    self.n_slit = spec.fit_line.data.shape[1]
                    self.n_scan = spec.fit_line.data.shape[2]
                    if spec._has_3d_levels():
                        self.n_levels = spec.levels['intensities'].shape[0]
                    break
            else:
                continue
            break

        self._build_ui()
        self._update_plots()

    def _get_spectrum(self, line, position):
        if line in self.spectra.keys() and position in self.spectra[line].keys():
            return self.spectra[line][position]
        return None

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # --- Controls row ---
        ctrl = QtWidgets.QHBoxLayout()

        ctrl.addWidget(QtWidgets.QLabel("Line:"))
        self.line_combo = pg.ComboBox()
        self.line_combo.setItems(list(AVAILABLE_LINES))
        self.line_combo.currentTextChanged.connect(self._on_line_changed)
        ctrl.addWidget(self.line_combo)

        ctrl.addWidget(QtWidgets.QLabel("Position:"))
        self.pos_combo = pg.ComboBox()
        self.pos_combo.setItems(self.positions)
        self.pos_combo.currentTextChanged.connect(self._on_position_changed)
        ctrl.addWidget(self.pos_combo)

        # Toggle for histogram auto-update
        self.hist_toggle = pg.ComboBox()
        self.hist_toggle.setItems(['Histogram: Auto', 'Histogram: Fixed'])
        self.hist_toggle.currentTextChanged.connect(self._on_hist_toggle)
        ctrl.addWidget(self.hist_toggle)
        self.hist_auto = True

        self.hist_info = QtWidgets.QLabel(
            "Auto: blue=min, black=mean, red=max (updates per level)")
        self.hist_info.setStyleSheet("color: #aaa; font-style: italic;")
        ctrl.addWidget(self.hist_info)

        # Rectangle controls
        self.btn_rect = QtWidgets.QPushButton("Add Rectangle")
        self.btn_rect.setCheckable(True)
        self.btn_rect.toggled.connect(self._on_rect_mode_toggled)
        ctrl.addWidget(self.btn_rect)

        self.btn_clear = QtWidgets.QPushButton("Clear Rectangles")
        self.btn_clear.clicked.connect(self._clear_rects)
        ctrl.addWidget(self.btn_clear)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # --- Level slider ---
        pg.setConfigOptions(antialias=True)

        # Colormap base colors: blue -> cyan -> black -> magenta -> red
        self.cmap_colors = np.array([
            [0, 0, 255, 255],     # blue
            [0, 255, 255, 255],   # cyan
            [0, 0, 0, 255],       # black
            [255, 0, 255, 255],   # magenta
            [255, 0, 0, 255],     # red
        ], dtype=np.float64)
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        self.cmap = pg.ColorMap(pos, self.cmap_colors)
        self.lut = self.cmap.getLookupTable(0.0, 1.0, 256)

        self.level_plot = pg.PlotWidget()
        self.level_plot.setMaximumHeight(100)
        self.level_plot.setLabel('bottom', 'Level')
        self.level_plot.hideAxis('left')
        self.level_plot.setMouseEnabled(x=False, y=False)
        vb = self.level_plot.getPlotItem().getViewBox()
        vb.setLimits(xMin=-0.5, xMax=self.n_levels - 0.5, yMin=0, yMax=1)
        self.level_plot.setXRange(-0.5, self.n_levels - 0.5, padding=0.02)
        self.level_plot.setYRange(0, 1, padding=0)
        self.level_plot.setAutoVisible(x=False, y=False)
        self.level_plot.enableAutoRange(False)

        for i in range(self.n_levels + 1):
            boundary = i - 0.5
            color = pg.mkPen('w', width=1) if (i % 5 == 0 or i == self.n_levels) else pg.mkPen(100, 100, 100, width=0.5)
            self.level_plot.addItem(pg.InfiniteLine(pos=boundary, angle=90, pen=color))

        self.level_line = pg.InfiniteLine(pos=0, angle=90, movable=True,
                                          pen=pg.mkPen('y', width=8),
                                          label='{value:.0f}',
                                          labelOpts={'position': 0.5, 'color': 'y',
                                                     'fill': (0, 0, 0, 200)})
        self.level_line.setBounds([-0.5, self.n_levels - 0.5])
        self.level_line.sigDragged.connect(self._on_level_dragged)
        self.level_plot.addItem(self.level_line)
        layout.addWidget(self.level_plot)

        # --- Map row: two panels side by side ---
        self.img_items = {}
        self.hist_items = {}
        self.plot_items = {}

        panel_labels = {'ours': 'python', 'idl': 'IDL (tc)'}

        map_row = QtWidgets.QHBoxLayout()
        for key in self.panel_keys:
            glw = pg.GraphicsLayoutWidget()

            plot_item = glw.addPlot(title=panel_labels[key])
            plot_item.setLabel('bottom', 'Scan')
            plot_item.setLabel('left', 'Slit')
            img_item = pg.ImageItem()
            img_item.setLookupTable(self.lut)
            plot_item.addItem(img_item)
            self.img_items[key] = img_item
            self.plot_items[key] = plot_item

            hist = pg.HistogramLUTItem()
            hist.setImageItem(img_item)
            hist.gradient.setColorMap(self.cmap)
            glw.addItem(hist)
            self.hist_items[key] = hist

            self._patch_rect_events(key, plot_item.getViewBox())

            map_row.addWidget(glw)
        layout.addLayout(map_row, stretch=1)

    def _on_line_changed(self, line):
        self.current_line = line
        self._update_plots()

    def _on_rect_mode_toggled(self, checked):
        self.rect_mode = checked

    def _clear_rects(self):
        for key in self.panel_keys:
            self.rects[key] = []
            self._redraw_rects(key)

    def _patch_rect_events(self, key, vb):
        orig_press = vb.mousePressEvent
        orig_move = vb.mouseMoveEvent
        orig_release = vb.mouseReleaseEvent

        HIT = 3.0

        def on_press(ev):
            if self.rect_mode:
                ev.accept()
                pos = vb.mapSceneToView(ev.scenePos())
                px, py = pos.x(), pos.y()
                for i, (x0, y0, x1, y1) in enumerate(self.rects[key]):
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    if abs(px - cx) < HIT and abs(py - cy) < HIT:
                        self._dragging_rect = (key, i)
                        self._drag_offset = (px - cx, py - cy)
                        return
                self._rect_start = (key, px, py)
                self._preview_item = None
            else:
                orig_press(ev)

        def on_move(ev):
            if self.rect_mode and self._dragging_rect is not None:
                ev.accept()
                pos = vb.mapSceneToView(ev.scenePos())
                k, idx = self._dragging_rect
                x0, y0, x1, y1 = self.rects[k][idx]
                w, h = x1 - x0, y1 - y0
                new_cx = pos.x() - self._drag_offset[0]
                new_cy = pos.y() - self._drag_offset[1]
                self.rects[k][idx] = [new_cx - w/2, new_cy - h/2,
                                      new_cx + w/2, new_cy + h/2]
                self._redraw_rects(k)
            elif self.rect_mode and self._rect_start is not None:
                ev.accept()
                pos = vb.mapSceneToView(ev.scenePos())
                k, sx, sy = self._rect_start
                x0, y0 = min(sx, pos.x()), min(sy, pos.y())
                x1, y1 = max(sx, pos.x()), max(sy, pos.y())
                if x1 - x0 < 0.5 or y1 - y0 < 0.5:
                    return
                if self._preview_item is not None:
                    self.plot_items[k].removeItem(self._preview_item)
                self._preview_item = pg.QtWidgets.QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
                self._preview_item.setPen(pg.mkPen('w', width=2))
                self._preview_item.setBrush(pg.mkBrush(0, 0, 0, 0))
                self.plot_items[k].addItem(self._preview_item)
            else:
                orig_move(ev)

        def on_release(ev):
            if self.rect_mode and self._dragging_rect is not None:
                ev.accept()
                self._dragging_rect = None
            elif self.rect_mode and self._rect_start is not None:
                ev.accept()
                pos = vb.mapSceneToView(ev.scenePos())
                k, sx, sy = self._rect_start
                x0, y0 = min(sx, pos.x()), min(sy, pos.y())
                x1, y1 = max(sx, pos.x()), max(sy, pos.y())
                if (x1 - x0) > 0.5 and (y1 - y0) > 0.5:
                    self.rects[k].append([x0, y0, x1, y1])
                if self._preview_item is not None:
                    self.plot_items[k].removeItem(self._preview_item)
                    self._preview_item = None
                self._rect_start = None
                self._redraw_rects(k)
            else:
                orig_release(ev)

        vb.mousePressEvent = on_press
        vb.mouseMoveEvent = on_move
        vb.mouseReleaseEvent = on_release

    def _redraw_rects(self, key=None):
        keys = [key] if key else self.panel_keys
        for k in keys:
            for item in self._rect_items.get(k, []):
                try:
                    self.plot_items[k].removeItem(item)
                except Exception:
                    pass
            self._rect_items[k] = []
            for (x0, y0, x1, y1) in self.rects[k]:
                rect = pg.QtWidgets.QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
                rect.setPen(pg.mkPen('w', width=2))
                rect.setBrush(pg.mkBrush(0, 0, 0, 0))
                self.plot_items[k].addItem(rect)
                self._rect_items[k].append(rect)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                cross_size = max(min(x1 - x0, y1 - y0) * 0.1, 1.0)
                h_line = pg.PlotCurveItem(
                    [cx - cross_size, cx + cross_size], [cy, cy],
                    pen=pg.mkPen('w', width=1.5))
                v_line = pg.PlotCurveItem(
                    [cx, cx], [cy - cross_size, cy + cross_size],
                    pen=pg.mkPen('w', width=1.5))
                self.plot_items[k].addItem(h_line)
                self.plot_items[k].addItem(v_line)
                self._rect_items[k].append(h_line)
                self._rect_items[k].append(v_line)

    def _on_position_changed(self, pos):
        self.current_position = pos
        self._update_plots()

    def _on_hist_toggle(self, text):
        self.hist_auto = 'Auto' in text
        if self.hist_auto:
            self.hist_info.setText("Auto: blue=min, black=mean, red=max (updates per level)")
        else:
            self.hist_info.setText("Fixed: color scale locked across levels")

    def _on_level_dragged(self):
        val = int(round(self.level_line.value()))
        val = max(0, min(val, self.n_levels - 1))
        self.level_line.setPos(val)
        self.level_idx = val
        self._update_plots()

    def _get_our_map(self, level_idx):
        """Our levels: shape (n_levels, n_scan, n_slit) -> return (n_slit, n_scan).

        Widths are rebuilt from the spatial mean of width_continuum (per-pixel
        total width at 0.95*continuum) rather than the stored global widths
        array, which is derived from a single reference pixel (0,0) and is
        not representative of the whole map. This makes the displayed width
        comparable to the IDL panel's spatial-mean width.
        """
        spec = self._get_spectrum(self.current_line, self.current_position)
        if spec is None or not spec._has_3d_levels():
            return None, None
        data = spec.levels['intensities'][level_idx, :, :]  # (n_scan, n_slit)
        n_levels = spec.levels['intensities'].shape[0]
        width_continuums = spec.levels['metadata'].get('width_continuums')
        if width_continuums is not None:
            mean_width_continuum = np.nanmean(width_continuums)
            widths = np.linspace(0, mean_width_continuum, n_levels)
        else:
            widths = spec.levels['widths']
        return np.transpose(data), widths  # -> (n_slit, n_scan)

    def _get_idl_map(self, level_idx):
        """IDL tc levels: shape (n_levels, n_slit, n_scan) -> return (n_slit, n_scan)."""
        spec = self._get_spectrum(self.current_line, self.current_position)
        spec_res = None
        if spec is not None and spec.line.wvl is not None and len(spec.line.wvl) > 1:
            spec_res = float(spec.line.wvl[1] - spec.line.wvl[0])
        idl = _load_idl_tc_levels(self.current_line, self.current_position, spec_res)
        if idl is None:
            return None, None
        data = idl['intensities']
        if level_idx >= data.shape[0]:
            return None, None
        return data[level_idx, :, :], idl['widths']

    def _update_plots(self):
        for key in self.panel_keys:
            img_item = self.img_items[key]

            if key == 'ours':
                map_display, widths = self._get_our_map(self.level_idx)
            else:
                map_display, widths = self._get_idl_map(self.level_idx)

            if map_display is None:
                img_item.clear()
                self.plot_items[key].setTitle(
                    f"{'python' if key == 'ours' else 'IDL (tc)'} — no data")
                continue

            finite = map_display[np.isfinite(map_display)]
            if len(finite) == 0:
                img_item.clear()
                continue

            vmin, vmax = np.nanmin(finite), np.nanmax(finite)
            if vmin == vmax:
                vmax = vmin + 1e-6

            mean_val = np.mean(finite)

            if self.hist_auto:
                mean_frac = (mean_val - vmin) / (vmax - vmin)
                mean_frac = max(0.01, min(0.99, mean_frac))
                blue_to_black = mean_frac
                black_to_red = 1.0 - mean_frac
                cmap_pos = np.array([
                    0.0,
                    blue_to_black * 0.5,
                    blue_to_black,
                    blue_to_black + black_to_red * 0.5,
                    1.0,
                ])
                dyn_cmap = pg.ColorMap(cmap_pos, self.cmap_colors)
                dyn_lut = dyn_cmap.getLookupTable(0.0, 1.0, 256)
                img_item.setLookupTable(dyn_lut)
                self.hist_items[key].gradient.setColorMap(dyn_cmap)
                img_item.setImage(map_display, levels=(vmin, vmax))
            else:
                current_levels = img_item.levels
                if current_levels is not None:
                    img_item.setImage(map_display, levels=tuple(current_levels))
                else:
                    img_item.setImage(map_display, levels=(vmin, vmax))

            # Update title
            label = 'python' if key == 'ours' else 'IDL (tc)'
            if widths is not None and self.level_idx < len(widths):
                w_mA = widths[self.level_idx] * 1000
                title = f"{label} — {self.current_line.upper()} {self.current_position} level {self.level_idx} ({w_mA:.0f} mA)"
            else:
                title = f"{label} — {self.current_line.upper()} {self.current_position} level {self.level_idx}"
            self.plot_items[key].setTitle(title)

        self._redraw_rects()


def main():
    app = pg.mkQApp()
    print("Loading spectra...")
    spectra = SpectrumContainer.load_all()
    print("Loaded. Launching GUI...")
    window = CompareLineLevelsInspector(spectra)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
