"""
VaspViz — chgcar_viewer.py
3D volumetric visualization of CHGCAR / PARCHG.
"""

import sys
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSlider, QFileDialog, QMessageBox, QComboBox,
    QCheckBox, QSplitter, QFrame, QStyle, QDoubleSpinBox, QProgressBar, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    _HAS_PYQTGRAPH = True
except ImportError:
    _HAS_PYQTGRAPH = False

from parsers import ChgcarParser
from constants import SIDEBAR_STYLE

class ChgcarLoadThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, fp):
        super().__init__()
        self.fp = fp

    def run(self):
        try:
            res = ChgcarParser(self.fp).parse()
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class ChargeDensityWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.data = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 6)
        lay.setSpacing(4)
        
        # Header
        header = QLabel("<b style='font-size:13px;color:#1e293b'>Charge Density Visualizer</b>  "
                        "<span style='font-size:10px;color:#64748b'>3D Volumetric Rendering of CHGCAR/PARCHG</span>")
        lay.addWidget(header)
        
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        _G_STYLE = """
        QGroupBox {
            font-size: 11px; font-weight: bold; color: #334155;
            border: 1px solid #e2e8f0; border-radius: 6px;
            margin-top: 10px; padding-top: 12px; background: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin; subcontrol-position: top left;
            left: 8px; padding: 0 3px; color: #2563eb;
        }
        QLabel { font-weight: normal; color: #475569; }
        """
        
        # ── Left sidebar ──
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setMinimumWidth(250); left_scroll.setMaximumWidth(310)
        
        left = QWidget(); left.setStyleSheet(SIDEBAR_STYLE + _G_STYLE)
        ll = QVBoxLayout(left); ll.setContentsMargins(4, 4, 4, 4); ll.setSpacing(4)
        
        g_file = QGroupBox("File / Data"); gl1 = QVBoxLayout(g_file); gl1.setSpacing(6)
        btn_load = QPushButton("Load CHGCAR")
        btn_load.setObjectName("primary")
        btn_load.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        btn_load.clicked.connect(self._load)
        gl1.addWidget(btn_load)
        
        self.lbl_info = QLabel("No file loaded"); self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-size:11px;color:#0f172a;padding:6px;background:#f8fafc;border-radius:4px;border:1px solid #cbd5e1;")
        gl1.addWidget(self.lbl_info)
        ll.addWidget(g_file)
        
        g_disp = QGroupBox("Volume Rendering"); gl2 = QVBoxLayout(g_disp); gl2.setSpacing(6)
        
        gl2.addWidget(QLabel("Data to display:"))
        self.cmb_data = QComboBox()
        self.cmb_data.addItems(["Total Charge Density"])
        self.cmb_data.currentIndexChanged.connect(self._update_volume)
        gl2.addWidget(self.cmb_data)
        
        # Iso-threshold with label
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Iso-threshold:")); self.lbl_th = QLabel("0.5 %"); h1.addWidget(self.lbl_th, 0, Qt.AlignmentFlag.AlignRight)
        gl2.addLayout(h1)
        self.sl_thresh = QSlider(Qt.Orientation.Horizontal)
        self.sl_thresh.setRange(0, 1000); self.sl_thresh.setValue(5) # 0.5% default to see bonds
        self.sl_thresh.valueChanged.connect(self._on_sliders_changed)
        gl2.addWidget(self.sl_thresh)
        
        # Max clip with label
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Max value clip:")); self.lbl_max = QLabel("30.0 %"); h2.addWidget(self.lbl_max, 0, Qt.AlignmentFlag.AlignRight)
        gl2.addLayout(h2)
        self.sl_max = QSlider(Qt.Orientation.Horizontal)
        self.sl_max.setRange(1, 1000); self.sl_max.setValue(300) # 30% default to clip core electrons
        self.sl_max.valueChanged.connect(self._on_sliders_changed)
        gl2.addWidget(self.sl_max)
        
        gl2.addWidget(QLabel("Colormap:"))
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["plasma", "viridis", "inferno", "magma", "cyan", "red", "blue"])
        self.cmb_cmap.currentIndexChanged.connect(self._update_volume)
        gl2.addWidget(self.cmb_cmap)
        
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Opacity factor:"))
        self.sp_opacity = QDoubleSpinBox()
        self.sp_opacity.setRange(0.01, 10.0); self.sp_opacity.setValue(1.0); self.sp_opacity.setSingleStep(0.1)
        self.sp_opacity.valueChanged.connect(self._update_volume)
        h3.addWidget(self.sp_opacity)
        gl2.addLayout(h3)
        
        self.chk_atoms = QCheckBox("Show Atoms & Cell Frame")
        self.chk_atoms.setChecked(True)
        self.chk_atoms.stateChanged.connect(self._update_structure)
        gl2.addWidget(self.chk_atoms)
        
        ll.addWidget(g_disp)
        
        self.pbar = QProgressBar(); self.pbar.setVisible(False)
        ll.addWidget(self.pbar)
        
        ll.addStretch()
        left_scroll.setWidget(left)
        sp.addWidget(left_scroll)
        
        # ── Right viewport ──
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        if _HAS_PYQTGRAPH:
            self.glw = gl.GLViewWidget()
            self.glw.setBackgroundColor(pg.mkColor("#0f172a"))
            self.vol_item = gl.GLVolumeItem(np.zeros((2,2,2,4)))
            self.glw.addItem(self.vol_item)
            rl.addWidget(self.glw)
        else:
            rl.addWidget(QLabel("PyQtGraph is required for volumetric rendering. Please run: pip install pyqtgraph"))
            
        sp.addWidget(right)
        sp.setStretchFactor(0, 0); sp.setStretchFactor(1, 1)
        lay.addWidget(sp)
        
        self.cell_items = []
        self.atom_item = None

    def _on_sliders_changed(self):
        self.lbl_th.setText(f"{self.sl_thresh.value()/10.0:.1f} %")
        self.lbl_max.setText(f"{self.sl_max.value()/10.0:.1f} %")
        self._update_volume()

    def _load(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Select CHGCAR / PARCHG")
        if not fp: return
        self.pbar.setVisible(True); self.pbar.setRange(0,0)
        self.thread = ChgcarLoadThread(fp)
        self.thread.finished.connect(self._on_loaded)
        self.thread.error.connect(self._on_error)
        self.thread.start()

    def _on_error(self, err):
        self.pbar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Failed to parse CHGCAR:\n{err}")

    def _on_loaded(self, data):
        self.pbar.setVisible(False)
        self.data = data
        nx, ny, nz = data["grid"]
        n_atoms = len(data["cart_positions"])
        
        info = f"Grid: {nx}×{ny}×{nz}\nAtoms: {n_atoms}"
        if data["is_spin_polarized"]:
            info += "\nSpin Polarized: Yes"
        self.lbl_info.setText(info)
        
        self.cmb_data.blockSignals(True)
        self.cmb_data.clear()
        self.cmb_data.addItem("Total Charge Density")
        if data["is_spin_polarized"]:
            self.cmb_data.addItem("Magnetization Density")
            self.cmb_data.addItem("Spin Up (Total + Mag)/2")
            self.cmb_data.addItem("Spin Down (Total - Mag)/2")
        self.cmb_data.blockSignals(False)
        
        self._update_structure()
        self._update_volume()
        
        if _HAS_PYQTGRAPH:
            # Auto-fit camera
            maxr = float(np.max(np.abs(data["lattice"]))) * 2.0
            self.glw.setCameraPosition(distance=maxr)

    def _update_structure(self):
        if not _HAS_PYQTGRAPH or not self.data: return
        
        for item in self.cell_items:
            self.glw.removeItem(item)
        if self.atom_item:
            self.glw.removeItem(self.atom_item)
        self.cell_items.clear(); self.atom_item = None
            
        if not self.chk_atoms.isChecked(): return
        
        lat = self.data["lattice"]
        # Draw box
        # corners
        corners = np.array([
            [0,0,0], lat[0], lat[1], lat[0]+lat[1],
            lat[2], lat[0]+lat[2], lat[1]+lat[2], lat[0]+lat[1]+lat[2]
        ])
        edges = [
            (0,1),(0,2),(1,3),(2,3),
            (4,5),(4,6),(5,7),(6,7),
            (0,4),(1,5),(2,6),(3,7)
        ]
        for i, j in edges:
            pts = np.vstack((corners[i], corners[j]))
            item = gl.GLLinePlotItem(pos=pts, color=(0.7,0.7,0.8,0.5), width=2.0)
            self.glw.addItem(item)
            self.cell_items.append(item)
            
        # Draw atoms
        if len(self.data["cart_positions"]) > 0:
            pos = self.data["cart_positions"]
            self.atom_item = gl.GLScatterPlotItem(pos=pos, color=(1.0,0.8,0.3,1.0), size=12.0, pxMode=True)
            self.glw.addItem(self.atom_item)

    def _get_cmap(self, name, size=256):
        import matplotlib.pyplot as plt
        try:
            cmap = plt.get_cmap(name)
            colors = cmap(np.linspace(0, 1, size))
        except Exception:
            colors = np.zeros((size, 4))
            colors[:,3] = 1.0
            if name == "cyan":
                colors[:,1] = np.linspace(0,1,size); colors[:,2] = np.linspace(0,1,size)
            elif name == "red":
                colors[:,0] = np.linspace(0,1,size)
            else:
                colors[:,2] = np.linspace(0,1,size)
        return colors

    def _update_volume(self):
        if not _HAS_PYQTGRAPH or not self.data: return
        
        choice = self.cmb_data.currentText()
        if choice == "Total Charge Density":
            vol_data = self.data["chg_total"].copy()
        elif choice == "Magnetization Density":
            vol_data = self.data["chg_diff"].copy()
        elif choice == "Spin Up (Total + Mag)/2":
            vol_data = (self.data["chg_total"] + self.data["chg_diff"]) / 2.0
        elif choice == "Spin Down (Total - Mag)/2":
            vol_data = (self.data["chg_total"] - self.data["chg_diff"]) / 2.0
        else:
            vol_data = self.data["chg_total"].copy()
            
        vmin, vmax = vol_data.min(), vol_data.max()
        if vmax == vmin: vmax = vmin + 1.0
        
        thresh_pct = self.sl_thresh.value() / 1000.0
        max_pct = self.sl_max.value() / 1000.0
        if thresh_pct >= max_pct: thresh_pct = max_pct - 0.001
        
        clip_min = vmin + (vmax - vmin) * thresh_pct
        clip_max = vmin + (vmax - vmin) * max_pct
        
        vol_norm = np.clip((vol_data - clip_min) / (clip_max - clip_min + 1e-9), 0.0, 1.0)
        vol_norm[vol_data < clip_min] = 0.0
        
        cmap_name = self.cmb_cmap.currentText()
        cmap_colors = self._get_cmap(cmap_name, 256)
        
        indices = (vol_norm * 255).astype(np.int32)
        rgba_vol = cmap_colors[indices]
        
        base_alpha = self.sp_opacity.value()
        # Scale alpha so low density is more transparent
        rgba_vol[..., 3] *= vol_norm * base_alpha
        rgba_vol[..., 3] = np.clip(rgba_vol[..., 3], 0.0, 1.0)
        
        # MUST convert to unsigned byte (0-255) for pyqtgraph GLVolumeItem OpenGL textures
        rgba_vol = (rgba_vol * 255).astype(np.ubyte)
        
        # Optimization: GLVolumeItem is faster if we only send float32 or uint8 data. 
        # But wait, GLVolumeItem can be very heavy. 
        # Actually it's best to reduce resolution if it's too large, but let's try direct first.
        # Check if the grid is massive
        nx, ny, nz = self.data["grid"]
        if nx * ny * nz > 120**3:
            # Subsample by 2 to prevent extreme lag
            rgba_vol = rgba_vol[::2, ::2, ::2]
            nx, ny, nz = rgba_vol.shape[:3]
        
        self.vol_item.setData(rgba_vol)
        
        lat = self.data["lattice"]
        # Transformation matrix to align volumetric grid to Cartesian lattice
        M = np.eye(4)
        M[0:3, 0] = lat[0] / nx
        M[0:3, 1] = lat[1] / ny
        M[0:3, 2] = lat[2] / nz
        
        # Pyqtgraph Transform3D from numpy array
        tr = pg.Transform3D(M.T) 
        self.vol_item.setTransform(tr)
