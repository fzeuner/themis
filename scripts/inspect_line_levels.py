"""
GUI for inspecting line level maps for Ti and Sr side-by-side.

Usage:
    uv run python scripts/inspect_line_levels.py

Controls:
    - Position dropdown: select limb distance (disk_center, m40, m30, ...)
    - Level slider: drag the vertical line in the level bar to select level 0-24
    - Each panel shows a 2D spatial map (slit x scan) for the selected level.
"""

import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from process_formation_height_line_levels import (
    SpectrumContainer, POSITION_TO_SEQUENCE, AVAILABLE_LINES,
)


class LineLevelsInspector(QtWidgets.QMainWindow):
    def __init__(self, spectra):
        super().__init__()
        self.spectra = spectra
        self.setWindowTitle("Line Levels Inspector")
        self.resize(1400, 800)

        self.positions = list(POSITION_TO_SEQUENCE.keys())
        self.current_position = self.positions[0]
        self.level_idx = 0

        # Persistent rectangles per line: list of [x0, y0, x1, y1] in data coords
        self.rects = {line: [] for line in AVAILABLE_LINES}
        self._rect_items = {line: [] for line in AVAILABLE_LINES}
        self.rect_mode = False
        self._rect_start = None
        self._preview_item = None
        self._dragging_rect = None  # (line, index) when dragging an existing rect center

        # Determine spatial dimensions and n_levels from first available spectrum
        self.n_slit = 1
        self.n_scan = 1
        self.n_levels = 25  # default
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

        # --- Controls row using pyqtgraph widgets ---
        ctrl = QtWidgets.QHBoxLayout()

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

        # --- Level slider as a pyqtgraph draggable line ---
        pg.setConfigOptions(antialias=True)

        # Colormap base colors: blue -> cyan -> black -> magenta -> red
        self.cmap_colors = np.array([
            [0, 0, 255, 255],     # blue
            [0, 255, 255, 255],   # cyan
            [0, 0, 0, 255],       # black
            [255, 0, 255, 255],   # magenta
            [255, 0, 0, 255],     # red
        ], dtype=np.float64)
        # Default colormap with black at center
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        self.cmap = pg.ColorMap(pos, self.cmap_colors)
        self.lut = self.cmap.getLookupTable(0.0, 1.0, 256)

        # Level slider plot — a thin horizontal bar with a draggable vertical line
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

        # Draw level boundaries as vertical lines at half-integer positions
        for i in range(self.n_levels + 1):
            boundary = i - 0.5
            color = pg.mkPen('w', width=1) if (i % 5 == 0 or i == self.n_levels) else pg.mkPen(100, 100, 100, width=0.5)
            self.level_plot.addItem(pg.InfiniteLine(pos=boundary, angle=90, pen=color))

        # Draggable vertical line for level selection — thick, snaps to integers
        self.level_line = pg.InfiniteLine(pos=0, angle=90, movable=True,
                                          pen=pg.mkPen('y', width=8),
                                          label='{value:.0f}',
                                          labelOpts={'position': 0.5, 'color': 'y',
                                                     'fill': (0, 0, 0, 200)})
        self.level_line.setBounds([-0.5, self.n_levels - 0.5])
        self.level_line.sigDragged.connect(self._on_level_dragged)
        self.level_plot.addItem(self.level_line)
        layout.addWidget(self.level_plot)

        # --- Map row ---
        self.img_items = {}
        self.hist_items = {}
        self.plot_items = {}

        map_row = QtWidgets.QHBoxLayout()
        for line in AVAILABLE_LINES:
            glw = pg.GraphicsLayoutWidget()

            plot_item = glw.addPlot(title=f"{line.upper()}")
            plot_item.setLabel('bottom', 'Scan')
            plot_item.setLabel('left', 'Slit')
            img_item = pg.ImageItem()
            img_item.setLookupTable(self.lut)
            plot_item.addItem(img_item)
            self.img_items[line] = img_item
            self.plot_items[line] = plot_item

            hist = pg.HistogramLUTItem()
            hist.setImageItem(img_item)
            hist.gradient.setColorMap(self.cmap)
            glw.addItem(hist)
            self.hist_items[line] = hist

            # Patch mouse events for rectangle creation
            self._patch_rect_events(line, plot_item.getViewBox())

            map_row.addWidget(glw)
        layout.addLayout(map_row, stretch=1)

    def _on_rect_mode_toggled(self, checked):
        self.rect_mode = checked

    def _clear_rects(self):
        for line in AVAILABLE_LINES:
            self.rects[line] = []
            self._redraw_rects(line)

    def _patch_rect_events(self, line, vb):
        orig_press = vb.mousePressEvent
        orig_move = vb.mouseMoveEvent
        orig_release = vb.mouseReleaseEvent

        HIT = 3.0  # hit-test threshold in data units

        def on_press(ev):
            if self.rect_mode:
                ev.accept()
                pos = vb.mapSceneToView(ev.scenePos())
                px, py = pos.x(), pos.y()
                # Check if clicking near an existing rect center
                for i, (x0, y0, x1, y1) in enumerate(self.rects[line]):
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    if abs(px - cx) < HIT and abs(py - cy) < HIT:
                        self._dragging_rect = (line, i)
                        self._drag_offset = (px - cx, py - cy)
                        return
                # Otherwise start creating a new rectangle
                self._rect_start = (line, px, py)
                self._preview_item = None
            else:
                orig_press(ev)

        def on_move(ev):
            if self.rect_mode and self._dragging_rect is not None:
                ev.accept()
                pos = vb.mapSceneToView(ev.scenePos())
                ln, idx = self._dragging_rect
                x0, y0, x1, y1 = self.rects[ln][idx]
                w, h = x1 - x0, y1 - y0
                new_cx = pos.x() - self._drag_offset[0]
                new_cy = pos.y() - self._drag_offset[1]
                self.rects[ln][idx] = [new_cx - w/2, new_cy - h/2,
                                       new_cx + w/2, new_cy + h/2]
                self._redraw_rects(ln)
            elif self.rect_mode and self._rect_start is not None:
                ev.accept()
                pos = vb.mapSceneToView(ev.scenePos())
                ln, sx, sy = self._rect_start
                x0, y0 = min(sx, pos.x()), min(sy, pos.y())
                x1, y1 = max(sx, pos.x()), max(sy, pos.y())
                if x1 - x0 < 0.5 or y1 - y0 < 0.5:
                    return
                if self._preview_item is not None:
                    self.plot_items[ln].removeItem(self._preview_item)
                self._preview_item = pg.QtWidgets.QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
                self._preview_item.setPen(pg.mkPen('w', width=2))
                self._preview_item.setBrush(pg.mkBrush(0, 0, 0, 0))
                self.plot_items[ln].addItem(self._preview_item)
            else:
                orig_move(ev)

        def on_release(ev):
            if self.rect_mode and self._dragging_rect is not None:
                ev.accept()
                self._dragging_rect = None
            elif self.rect_mode and self._rect_start is not None:
                ev.accept()
                pos = vb.mapSceneToView(ev.scenePos())
                ln, sx, sy = self._rect_start
                x0, y0 = min(sx, pos.x()), min(sy, pos.y())
                x1, y1 = max(sx, pos.x()), max(sy, pos.y())
                if (x1 - x0) > 0.5 and (y1 - y0) > 0.5:
                    self.rects[ln].append([x0, y0, x1, y1])
                if self._preview_item is not None:
                    self.plot_items[ln].removeItem(self._preview_item)
                    self._preview_item = None
                self._rect_start = None
                self._redraw_rects(ln)
            else:
                orig_release(ev)

        vb.mousePressEvent = on_press
        vb.mouseMoveEvent = on_move
        vb.mouseReleaseEvent = on_release

    def _redraw_rects(self, line=None):
        lines = [line] if line else AVAILABLE_LINES
        for ln in lines:
            for item in self._rect_items.get(ln, []):
                try:
                    self.plot_items[ln].removeItem(item)
                except Exception:
                    pass
            self._rect_items[ln] = []
            for (x0, y0, x1, y1) in self.rects[ln]:
                # Rectangle outline
                rect = pg.QtWidgets.QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
                rect.setPen(pg.mkPen('w', width=2))
                rect.setBrush(pg.mkBrush(0, 0, 0, 0))
                self.plot_items[ln].addItem(rect)
                self._rect_items[ln].append(rect)
                # Center cross
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                cross_size = max(min(x1 - x0, y1 - y0) * 0.1, 1.0)
                h_line = pg.PlotCurveItem(
                    [cx - cross_size, cx + cross_size], [cy, cy],
                    pen=pg.mkPen('w', width=1.5))
                v_line = pg.PlotCurveItem(
                    [cx, cx], [cy - cross_size, cy + cross_size],
                    pen=pg.mkPen('w', width=1.5))
                self.plot_items[ln].addItem(h_line)
                self.plot_items[ln].addItem(v_line)
                self._rect_items[ln].append(h_line)
                self._rect_items[ln].append(v_line)

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

    def _get_map_data(self, spec, level_idx):
        """Extract the 2D intensity map (n_slit, n_scan) for the selected level."""
        if not spec._has_3d_levels():
            return None

        return spec.levels['intensities'][level_idx, :, :]

    def _update_plots(self):
        for line in AVAILABLE_LINES:
            img_item = self.img_items[line]

            spec = self._get_spectrum(line, self.current_position)
            if spec is None or not spec._has_3d_levels():
                img_item.clear()
                continue

            map_data = self._get_map_data(spec, self.level_idx)
            if map_data is None:
                img_item.clear()
                continue

            # Transpose so slit is on y-axis, scan on x-axis
            map_display = np.transpose(map_data)

            finite = map_display[np.isfinite(map_display)]
            if len(finite) == 0:
                img_item.clear()
                continue

            vmin, vmax = np.nanmin(finite), np.nanmax(finite)
            if vmin == vmax:
                vmax = vmin + 1e-6

            mean_val = np.mean(finite)

            if self.hist_auto:
                # Build a colormap where black sits at the mean fraction
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
                self.hist_items[line].gradient.setColorMap(dyn_cmap)
                img_item.setImage(map_display, levels=(vmin, vmax))
            else:
                # Keep existing levels, just update the image data
                current_levels = img_item.levels
                if current_levels is not None:
                    img_item.setImage(map_display, levels=tuple(current_levels))
                else:
                    img_item.setImage(map_display, levels=(vmin, vmax))

            # Update title with width info
            widths = spec.levels['widths']
            w_mA = widths[self.level_idx] * 1000
            title = f"{line.upper()} — level {self.level_idx} ({w_mA:.0f} mA)"
            img_item.getViewBox().parentItem().setTitle(title)

        # Redraw persistent rectangles on top of updated maps
        self._redraw_rects()


def main():
    app = pg.mkQApp()
    print("Loading spectra...")
    spectra = SpectrumContainer.load_all()
    print("Loaded. Launching GUI...")
    window = LineLevelsInspector(spectra)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
