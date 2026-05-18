"""
VaspViz v4.1 — Professional VASP Electronic Structure Suite
============================================================
Developer: Soumyajit Das, NIT Silchar

Usage:  python main.py [vasprun.xml]

File structure:
  main.py          — MainWindow + entry point
  constants.py     — Element tables, styles
  parsers.py       — vasprun.xml, POSCAR, Wannier90, OUTCAR parsers
  analysis.py      — PlotEngine, band gap, effective mass, optical
  layer_builder.py — LayerBuilderWidget
  gl_viewer.py     — VESTA-style OpenGL PoscarViewerWidget
  widgets.py       — KpointHelper, PlotEditor, AnalysisPanel, WannierCompare
"""

import sys, os, csv, re, math
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from collections import OrderedDict
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from chgcar_viewer import ChargeDensityWidget

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QFileDialog, QComboBox,
    QDoubleSpinBox, QCheckBox, QGroupBox, QTabWidget, QSpinBox,
    QStatusBar, QToolBar, QFrame, QScrollArea, QSizePolicy,
    QMessageBox, QGridLayout, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QDialog, QDialogButtonBox, QListWidget,
    QListWidgetItem, QAbstractItemView, QColorDialog, QFormLayout,
    QTreeWidget, QTreeWidgetItem, QSlider, QMenu, QToolButton,
    QButtonGroup, QRadioButton, QStackedWidget, QProgressBar,
    QInputDialog, QFontComboBox, QStyle
)
from PyQt6.QtCore import Qt, QSize, QTimer, QMimeData, pyqtSignal
from PyQt6.QtGui import (
    QAction, QFont, QColor, QPalette, QPixmap, QIcon,
    QSurfaceFormat, QMatrix4x4, QVector3D, QQuaternion, QKeySequence, QShortcut,
)
# OpenGL imports (needed for PoscarViewerWidget in gl_viewer.py)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_hex
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

# ── Local modules ──────────────────────────────────────────────────────────────
from constants import *
from constants import GLOBAL_APP_STYLE
from parsers   import VasprunParser, PoscarParser, Wannier90Parser, read_outcar_info
from analysis  import (PlotEngine, find_band_gap, fit_effective_mass,
                        compute_jdos, compute_optical_spectrum)
from layer_builder import LayerBuilderWidget
from gl_viewer  import PoscarViewerWidget
from widgets    import (KpointHelperWidget, PlotEditorPanel,
                        AnalysisPanel, WannierCompareWidget, P4VaspPanel,
                        KPathSeekerWidget, FatBandPanel)

def get_asset_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "assets", filename)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data=None; self.engine=None
        # ── Performance: debounce replot ──────────────────────────────────────
        self._plot_timer = QTimer(self); self._plot_timer.setSingleShot(True)
        self._plot_timer.timeout.connect(self._do_replot)
        self._last_mode = -1          # track mode to avoid unnecessary figure.clear()
        self._last_rcfont = {}        # cache rcParams to avoid redundant sets
        # ─────────────────────────────────────────────────────────────────────
        self._build()
        self.setWindowTitle(f"VaspViz v{VERSION} — by {DEVELOPER}")
        self.setWindowIcon(QIcon(get_asset_path("logo-vasvz.png")))
        self.resize(1600,960)
        self.status("Ready — open a vasprun.xml")

    def _build(self):
        self.menuBar().setStyleSheet(MENU_STYLE)
        cw=QWidget(); self.setCentralWidget(cw)
        ml=QVBoxLayout(cw); ml.setContentsMargins(0,0,0,0); ml.setSpacing(0)

        self.tabs=QTabWidget(); self.tabs.setDocumentMode(True)

        # ── Tab 1: Electronic Structure ──
        es=QWidget(); esl=QHBoxLayout(es); esl.setContentsMargins(0,0,0,0)
        main_split=QSplitter(Qt.Orientation.Horizontal); main_split.setHandleWidth(1)
        self.left=self._build_left(); main_split.addWidget(self.left)
        self.plotw=self._build_plot(); main_split.addWidget(self.plotw)
        self.plot_editor=PlotEditorPanel(); self.plot_editor.settings_changed.connect(self._on_editor_change)
        self.plot_editor.setMinimumWidth(248); self.plot_editor.setMaximumWidth(268)
        main_split.addWidget(self.plot_editor)
        main_split.setSizes([228,1120,248]); esl.addWidget(main_split)
        self.tabs.addTab(es, self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon), " Electronic Structure")

        # ── Tabs 2-6 ──
        self.lb=LayerBuilderWidget(); self.tabs.addTab(self.lb, self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView), " Layer Builder")
        self.pv=PoscarViewerWidget(); self.tabs.addTab(self.pv, self.style().standardIcon(QStyle.StandardPixmap.SP_DesktopIcon), " POSCAR Viewer")
        self.wc=WannierCompareWidget(); self.tabs.addTab(self.wc, self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView), " Wannier90")
        self.fb=FatBandPanel(); self.tabs.addTab(self.fb, self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp), " Fat Bands")
        self.khelper=KpointHelperWidget()
        self.khelper.kpath_requested.connect(self._insert_kpath_label)
        self.tabs.addTab(self.khelper, self.style().standardIcon(QStyle.StandardPixmap.SP_FileLinkIcon), " K-point Helper")
        self.analysis=AnalysisPanel(); self.tabs.addTab(self.analysis, self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView), " Analysis & Export")
        self.p4vasp=P4VaspPanel(); self.tabs.addTab(self.p4vasp, self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload), " P4Vasp Tools")
        self.kpath_seeker=KPathSeekerWidget(); self.tabs.addTab(self.kpath_seeker, self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight), " K-Path Seeker")
        self.chgcar_viewer = ChargeDensityWidget(); self.tabs.addTab(self.chgcar_viewer, self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon), " Charge Density")

        ml.addWidget(self.tabs)

        # ── Status bar with progress widget ──
        sb=QStatusBar(); self.setStatusBar(sb)
        self._progress=QProgressBar()
        self._progress.setRange(0,0)   # indeterminate spinner
        self._progress.setFixedWidth(110)
        self._progress.setFixedHeight(12)
        self._progress.setVisible(False)
        sb.addPermanentWidget(self._progress)

        self._setup_menu(); self._setup_toolbar()

        # ── Ctrl+1-6 tab shortcuts ──
        for i in range(min(self.tabs.count(), 9)):
            sc=QShortcut(QKeySequence(f"Ctrl+{i+1}"),self)
            sc.activated.connect(lambda _=None,idx=i: self.tabs.setCurrentIndex(idx))


    def _build_left(self):
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(220); scroll.setMaximumWidth(260)
        panel=QWidget(); panel.setStyleSheet(SIDEBAR_STYLE); panel.setObjectName("sidebar")
        lay=QVBoxLayout(panel); lay.setContentsMargins(7,6,7,6); lay.setSpacing(4)

        # ── File info ──
        g=QGroupBox("File"); gl=QVBoxLayout(g)
        self.lbl_sys=QLabel("No file loaded"); self.lbl_sys.setWordWrap(True); self.lbl_sys.setStyleSheet("font-size:12px;font-weight:600;color:#2563EB;")
        self.lbl_info=QLabel(""); self.lbl_info.setWordWrap(True); self.lbl_info.setStyleSheet("font-size:10px;color:#64748B;")
        gl.addWidget(self.lbl_sys); gl.addWidget(self.lbl_info); lay.addWidget(g)

        # ── Overlay ──
        gc=QGroupBox("Overlay (2nd file)"); gcl=QVBoxLayout(gc); gcl.setSpacing(4)
        self.lbl_f2=QLabel("None"); self.lbl_f2.setStyleSheet("font-size:10px;color:#64748B;"); gcl.addWidget(self.lbl_f2)
        br=QHBoxLayout()
        b1=QPushButton("Load 2nd"); b1.clicked.connect(self._load_f2); br.addWidget(b1)
        b2=QPushButton("Clear"); b2.clicked.connect(self._clear_f2); br.addWidget(b2)
        gcl.addLayout(br); lay.addWidget(gc)

        # ── Plot Mode ──
        g=QGroupBox("Plot Mode"); gl=QVBoxLayout(g)
        self.mode_combo=QComboBox()
        self.mode_combo.addItems(["Band Structure","Fat Bands (bubble)","Fat Bands (colormap)",
                                   "DOS","Band + DOS","Stacked DOS (per-ion)","Brillouin Zone (auto)",
                                   "Fermi Surface","Spin Texture (SP only)","Group Velocity"])
        self.mode_combo.currentIndexChanged.connect(self._mode_change); gl.addWidget(self.mode_combo); lay.addWidget(g)

        # ── Energy Range ──
        g=QGroupBox("Energy Range"); gl=QGridLayout(g)
        gl.addWidget(QLabel("E min:"),0,0); self.sp_emin=QDoubleSpinBox(); self.sp_emin.setRange(-50,0); self.sp_emin.setValue(-6.); self.sp_emin.setSingleStep(.5); self.sp_emin.valueChanged.connect(self._replot); gl.addWidget(self.sp_emin,0,1)
        gl.addWidget(QLabel("E max:"),1,0); self.sp_emax=QDoubleSpinBox(); self.sp_emax.setRange(0,50); self.sp_emax.setValue(6.); self.sp_emax.setSingleStep(.5); self.sp_emax.valueChanged.connect(self._replot); gl.addWidget(self.sp_emax,1,1)
        self.chk_shift=QCheckBox("Shift E_F → 0"); self.chk_shift.setChecked(True); self.chk_shift.stateChanged.connect(self._replot); gl.addWidget(self.chk_shift,2,0,1,2); lay.addWidget(g)

        # ── Fat bands ──
        self.grp_fat=QGroupBox("Fat Bands / Projections"); fl=QVBoxLayout(self.grp_fat); fl.setSpacing(4)
        fl.addWidget(QLabel("Orbital:")); self.orb_combo=QComboBox(); self.orb_combo.addItems(list(ORBITAL_GROUPS.keys())); self.orb_combo.currentIndexChanged.connect(self._replot); fl.addWidget(self.orb_combo)
        fl.addWidget(QLabel("Atom idx (−1=all):")); self.sp_atom=QSpinBox(); self.sp_atom.setRange(-1,999); self.sp_atom.setValue(-1); self.sp_atom.valueChanged.connect(self._replot); fl.addWidget(self.sp_atom)
        sr=QHBoxLayout(); sr.addWidget(QLabel("Scale:")); self.sp_scale=QSpinBox(); self.sp_scale.setRange(5,500); self.sp_scale.setValue(40); self.sp_scale.setSingleStep(5); self.sp_scale.valueChanged.connect(self._replot); sr.addWidget(self.sp_scale); fl.addLayout(sr)
        fl.addWidget(QLabel("Colormap:")); self.cmap_combo=QComboBox()
        self.cmap_combo.addItems(["plasma","viridis","inferno","coolwarm","RdBu","Spectral","hot","Blues","turbo","magma","cividis"]); self.cmap_combo.currentIndexChanged.connect(self._replot); fl.addWidget(self.cmap_combo)
        wr=QHBoxLayout(); wr.addWidget(QLabel("W:")); self.sp_wmin=QDoubleSpinBox(); self.sp_wmin.setRange(0,1); self.sp_wmin.setValue(0.); self.sp_wmin.setSingleStep(.05); self.sp_wmin.valueChanged.connect(self._replot); wr.addWidget(self.sp_wmin)
        wr.addWidget(QLabel("—")); self.sp_wmax=QDoubleSpinBox(); self.sp_wmax.setRange(0,2); self.sp_wmax.setValue(1.); self.sp_wmax.setSingleStep(.05); self.sp_wmax.valueChanged.connect(self._replot); wr.addWidget(self.sp_wmax); fl.addLayout(wr)
        self.grp_fat.setVisible(False); lay.addWidget(self.grp_fat)

        # ── DOS ──
        self.grp_dos=QGroupBox("DOS Settings"); dl=QVBoxLayout(self.grp_dos); dl.setSpacing(3)
        self.chk_s=QCheckBox("s"); self.chk_p=QCheckBox("p"); self.chk_d=QCheckBox("d"); self.chk_fill=QCheckBox("Fill")
        self.chk_spin_mir=QCheckBox("Mirror spin-dn")
        for chk in [self.chk_s,self.chk_p,self.chk_d,self.chk_fill]: chk.setChecked(True); chk.stateChanged.connect(self._replot); dl.addWidget(chk)
        self.chk_spin_mir.stateChanged.connect(self._replot); dl.addWidget(self.chk_spin_mir)
        dl.addWidget(QLabel("Ion filter (blank=all):")); self.le_ions=QLineEdit(); self.le_ions.setPlaceholderText("e.g. 0,1"); self.le_ions.editingFinished.connect(self._replot); dl.addWidget(self.le_ions)
        self.grp_dos.setVisible(False); lay.addWidget(self.grp_dos)

        # ── BZ / Fermi ──
        self.grp_fs=QGroupBox("Fermi Surface — Band"); fsl=QVBoxLayout(self.grp_fs)
        self.sp_fs=QSpinBox(); self.sp_fs.setRange(1,999); self.sp_fs.valueChanged.connect(self._replot); fsl.addWidget(self.sp_fs)
        self.grp_fs.setVisible(False); lay.addWidget(self.grp_fs)

        # ── K-path ──
        gk=QGroupBox("K-path"); gkl=QVBoxLayout(gk)
        btn_kp=QPushButton("Edit k-path labels..."); btn_kp.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView)); btn_kp.clicked.connect(self._edit_kpath); gkl.addWidget(btn_kp)
        # Tab index 5 = K-point Helper (0=Elec.Structure, 1=LayerBuilder, 2=POSCAR, 3=Wannier, 4=FatBands, 5=KHelper)
        btn_kh=QPushButton("K-point Helper"); btn_kh.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight)); btn_kh.clicked.connect(lambda: self.tabs.setCurrentIndex(5)); gkl.addWidget(btn_kh)
        btn_chg=QPushButton("Charge Density"); btn_chg.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)); btn_chg.clicked.connect(lambda: self.tabs.setCurrentIndex(9)); gkl.addWidget(btn_chg)
        lay.addWidget(gk)

        lay.addStretch(); scroll.setWidget(panel); return scroll

    def _build_plot(self):
        cw=QWidget(); vl=QVBoxLayout(cw); vl.setContentsMargins(0,0,0,0); vl.setSpacing(0)
        self.figure=Figure(figsize=(11,8),dpi=100); self.figure.patch.set_facecolor("#ffffff")
        self.canvas=FigureCanvas(self.figure); self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        self.nav=NavigationToolbar(self.canvas,cw); self.nav.setIconSize(QSize(16,16))
        vl.addWidget(self.nav); vl.addWidget(self.canvas)
        self.engine=PlotEngine(self.figure)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._welcome(); return cw

    def _setup_menu(self):
        mb=self.menuBar()
        # File
        fm=mb.addMenu("File")
        for lbl,sc,fn in [("Open vasprun.xml…","Ctrl+O",self._open),("Load 2nd file","",self._load_f2)]:
            a=QAction(lbl,self); a.setShortcut(sc); a.triggered.connect(fn); fm.addAction(a)
        fm.addSeparator()
        for lbl,sc,fn in [("Save Figure…","Ctrl+S",self._save),("Save High-Res PNG (600 DPI)…","",self._save_hq_png),("Save LaTeX-ready…","",self._save_latex_ready),("Save Band CSV","",self.analysis.export_bands),("Save DOS CSV","",self.analysis.export_dos)]:
            a=QAction(lbl,self); a.setShortcut(sc); a.triggered.connect(fn); fm.addAction(a)
        fm.addSeparator()
        a=QAction("Open POSCAR/CONTCAR…",self); a.triggered.connect(lambda: (self.tabs.setCurrentIndex(2),self.pv._open())); fm.addAction(a)
        a=QAction("Load Wannier bands…",self); a.triggered.connect(lambda: (self.tabs.setCurrentIndex(3),self.wc._load())); fm.addAction(a)
        a=QAction("Read OUTCAR…",self); a.triggered.connect(self._read_outcar); fm.addAction(a)
        a=QAction("Open POSCAR for K-Path…",self); a.triggered.connect(lambda: (self.tabs.setCurrentIndex(7),self.kpath_seeker._load_poscar())); fm.addAction(a)
        fm.addSeparator()
        a=QAction("Quit",self); a.setShortcut("Ctrl+Q"); a.triggered.connect(self.close); fm.addAction(a)

        # View
        vm=mb.addMenu("View")
        for lbl,idx in [("Band Structure",0),("Fat Bands (bubble)",1),("Fat Bands (colormap)",2),("DOS",3),("Band+DOS",4),("Brillouin Zone",5),("Fermi Surface",6)]:
            a=QAction(lbl,self); a.triggered.connect(lambda _,i=idx: self.mode_combo.setCurrentIndex(i)); vm.addAction(a)
        vm.addSeparator()
        for lbl,tidx in [("Layer Builder",1),("POSCAR Viewer",2),("Wannier90",3),("Fat Bands",4),("K-point Helper",5),("Analysis",6),("P4Vasp",7),("K-Path Seeker",8),("Charge Density",9)]:
            a=QAction(lbl,self); a.triggered.connect(lambda _,i=tidx: self.tabs.setCurrentIndex(i)); vm.addAction(a)

        # Plot
        pm=mb.addMenu("Plot")
        a=QAction("Toggle Dark Mode",self); a.setCheckable(True); a.triggered.connect(lambda c: (self.plot_editor.chk_dark.setChecked(c),self._replot())); pm.addAction(a)
        pm.addSeparator()
        a=QAction("Edit k-path labels…",self); a.triggered.connect(self._edit_kpath); pm.addAction(a)
        a=QAction("Band selector…",self); a.triggered.connect(self._band_selector); pm.addAction(a)
        pm.addSeparator()
        # Colour themes submenu
        tm=pm.addMenu("Apply theme…")
        for th in ["Publication","Dark Neon","Nature","Pastel","High Contrast"]:
            ta=QAction(th,self); ta.triggered.connect(lambda _,t=th: self._apply_theme(t)); tm.addAction(ta)
        pm.addSeparator()
        a=QAction("Reset plot settings",self); a.triggered.connect(self._reset_plot_settings); pm.addAction(a)

        # Analysis
        am=mb.addMenu("Analysis")
        for lbl,fn in [("Band Gap",self._quick_gap),("Effective Mass",self._quick_mstar),
                        ("JDOS",lambda: (self.tabs.setCurrentIndex(6),self.analysis.calc_optical())),
                        ("Band Curvature",lambda: (self.tabs.setCurrentIndex(6),self.analysis.calc_curvature()))]:
            a=QAction(lbl,self); a.triggered.connect(fn); am.addAction(a)
        am.addSeparator()
        for lbl,fn in [("Export Band CSV",self.analysis.export_bands),("Export DOS CSV",self.analysis.export_dos),("Export Gap Report",self.analysis.export_gap)]:
            a=QAction(lbl,self); a.triggered.connect(fn); am.addAction(a)

        # Help
        hm=mb.addMenu("Help")
        a=QAction("Keyboard Shortcuts",self); a.triggered.connect(self._show_shortcuts); hm.addAction(a)
        hm.addSeparator()
        a=QAction(f"About VaspViz v{VERSION}",self); a.triggered.connect(self._about); hm.addAction(a)
        a=QAction("Developer Info",self); a.triggered.connect(self._developer_info); hm.addAction(a)

    def _setup_toolbar(self):
        tb=QToolBar("Main"); tb.setIconSize(QSize(20,20))
        tb.setMovable(False); tb.setFloatable(False)
        self.addToolBar(tb)
        sp = self.style()
        # File group
        file_actions = [
            (sp.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),     "Open XML",  "Open vasprun.xml (Ctrl+O)",  self._open),
            (sp.standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder), "2nd File","Overlay a 2nd vasprun.xml", self._load_f2),
        ]
        for icon,lbl,tip,fn in file_actions:
            a=QAction(icon, lbl, self); a.setToolTip(tip); a.triggered.connect(fn); tb.addAction(a)
        tb.addSeparator()
        # Save group
        save_actions = [
            (sp.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Save Figure",   "Save figure as PNG/PDF/SVG (Ctrl+S)", self._save),
            (sp.standardIcon(QStyle.StandardPixmap.SP_ArrowDown),        "HQ PNG",        "Save 600 DPI PNG",                    self._save_hq_png),
            (sp.standardIcon(QStyle.StandardPixmap.SP_FileLinkIcon),     "LaTeX PDF+SVG", "Save LaTeX-ready PDF+SVG pair",       self._save_latex_ready),
        ]
        for icon,lbl,tip,fn in save_actions:
            a=QAction(icon, lbl, self); a.setToolTip(tip); a.triggered.connect(fn); tb.addAction(a)
        tb.addSeparator()
        # Analysis group
        analysis_actions = [
            (sp.standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation), "Band Gap", "Quick band-gap report",  self._quick_gap),
            (sp.standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView), "m* Fit",  "Fit effective mass (CBM)", self._quick_mstar),
        ]
        for icon,lbl,tip,fn in analysis_actions:
            a=QAction(icon, lbl, self); a.setToolTip(tip); a.triggered.connect(fn); tb.addAction(a)
        tb.addSeparator()
        # Navigation group
        nav_tabs=[
            (sp.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView), "Layers",   1),
            (sp.standardIcon(QStyle.StandardPixmap.SP_DesktopIcon),            "POSCAR",   2),
            (sp.standardIcon(QStyle.StandardPixmap.SP_FileDialogListView),     "Wannier",  3),
            (sp.standardIcon(QStyle.StandardPixmap.SP_FileLinkIcon),           "K-pts",    5),
            (sp.standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView), "Analysis", 6),
            (sp.standardIcon(QStyle.StandardPixmap.SP_ArrowRight),             "K-Path",   8),
        ]
        for icon,lbl,idx in nav_tabs:
            a=QAction(icon, lbl, self); a.setToolTip(f"Go to {lbl} tab")
            a.triggered.connect(lambda _=None,i=idx: self.tabs.setCurrentIndex(i)); tb.addAction(a)


    # ── file ops ──────────────────────────────────────────────────────────────

    def _open(self):
        p,_=QFileDialog.getOpenFileName(self,"Open vasprun.xml","","VASP (vasprun.xml *.xml);;All (*)")
        if p: self._load(p)

    def _load(self, path):
        self._progress.setVisible(True); QApplication.processEvents()
        self.status(f"Parsing {Path(path).name}…")
        try:
            data=VasprunParser(path).parse(); self._set_data(data)
            self.setWindowTitle(f"VaspViz v{VERSION} — {Path(path).name}")
            self.status(f"{Path(path).name} | {data['nbands']} bands | {data['nkpoints']} k-pts | "
                        f"E_F={data['efermi']:.4f} eV | {'SP' if data['spin_polarized'] else 'NSP'} | ~{data['n_electrons']} e⁻")
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self,"Parse Error",str(e)); self.status("Error.")
        finally:
            self._progress.setVisible(False)



    def _load_f2(self):
        p,_=QFileDialog.getOpenFileName(self,"2nd vasprun.xml","","VASP (*.xml);;All (*)")
        if not p: return
        try:
            d2=VasprunParser(p).parse(); self.engine.set_data2(d2)
            self.lbl_f2.setText(f"[OK] {Path(p).name}"); self._replot()
        except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def _clear_f2(self):
        self.engine.set_data2(None); self.lbl_f2.setText("None"); self._replot()

    def _set_data(self, data):
        self.data=data; self.engine.set_data(data)
        self.lbl_sys.setText(data["system"])
        ions=", ".join(dict.fromkeys(data["ions"])) or "?"
        kpath="→".join(l for _,l in data["klabels"] if l) or "auto"
        self.lbl_info.setText(f"{data['nbands']} bands | {data['nkpoints']} k-pts\n"
                               f"E_F={data['efermi']:.4f} eV | ~{data['n_electrons']} e⁻\n"
                               f"{'Spin-pol.' if data['spin_polarized'] else 'NSP'}\n"
                               f"Ions ({data['n_ions']}): {ions}\nPath: {kpath}")
        if data["ions"]: self.sp_atom.setMaximum(max(0, len(data["ions"])-1))
        self.sp_fs.setMaximum(data["nbands"])
        self._last_mode = -1   # force full redraw on new data
        self.analysis.set_data(data); self.wc.set_vasp_data(data); self.fb.set_vasp_data(data)
        self._do_replot()

    def _on_editor_change(self, settings):
        if settings.pop("_zoom_gap", False):
            if self.data and self.engine:
                emin, emax = self.engine.zoom_to_gap()
                if emin is not None:
                    self.sp_emin.blockSignals(True); self.sp_emax.blockSignals(True)
                    self.sp_emin.setValue(emin); self.sp_emax.setValue(emax)
                    self.sp_emin.blockSignals(False); self.sp_emax.blockSignals(False)
                    self.status(f"Zoomed to gap region: [{emin:.2f}, {emax:.2f}] eV")
                else:
                    self.status("Metallic — no gap to zoom to")
        elif settings.pop("_reset_e", False):
            self.sp_emin.blockSignals(True); self.sp_emax.blockSignals(True)
            self.sp_emin.setValue(-6.0); self.sp_emax.setValue(6.0)
            self.sp_emin.blockSignals(False); self.sp_emax.blockSignals(False)
            self.status("Energy window reset to ±6 eV")
        elif settings.pop("_show_sym", False):
            if self.data:
                ev = self.data["eigenvalues"].copy() - self.data["efermi"]
                info = find_band_gap(ev)
                if info["type"] != "metal":
                    vb_bands = [max(0, info["vbm_b"]-2+i) for i in range(3)]
                    cb_bands = [min(self.data["nbands"]-1, info["cbm_b"]+i) for i in range(3)]
                    bands_str = ",".join(str(b+1) for b in sorted(set(vb_bands+cb_bands)))
                    self.plot_editor.le_bands.setText(bands_str)
                    self.status(f"Showing VB/CB bands: {bands_str}")
                else:
                    self.status("Metallic — cannot auto-select VB/CB")
        self._replot()

    def _mode_change(self, idx):
        self.grp_fat.setVisible(idx in (1,2))
        self.grp_dos.setVisible(idx in (3,4,5))
        self.grp_fs.setVisible(idx==7)
        self._do_replot()   # immediate — skip debounce for mode changes

    def _get_settings(self):
        ions_raw = self.le_ions.text().strip(); ion_filter = None
        if ions_raw:
            try: ion_filter = [int(x.strip()) for x in ions_raw.split(",")]
            except: pass
        eds = self.plot_editor.get_settings()   # includes all sumo settings
        s = eds.copy()
        s.update({
            # Sidebar controls
            "shift_efermi": self.chk_shift.isChecked(),
            "emin": self.sp_emin.value(), "emax": self.sp_emax.value(),
            "cmap": self.cmap_combo.currentText(),
            "wmin": self.sp_wmin.value(), "wmax": self.sp_wmax.value(),
            "dos_s": self.chk_s.isChecked(), "dos_p": self.chk_p.isChecked(),
            "dos_d": self.chk_d.isChecked(), "dos_fill": self.chk_fill.isChecked(),
            "spin_mirror": self.chk_spin_mir.isChecked(),
            "ion_filter": ion_filter,
        })
        return s

    def _replot(self):
        """Debounced replot — batches rapid control changes (250 ms)."""
        if not self.data: return
        self._plot_timer.start(250)

    def _do_replot(self):
        """Actual replot. Called by timer or directly for immediate updates."""
        if not self.data: return
        s = self._get_settings(); self.engine.set_settings(**s)
        dk = s["dark"]
        mode = self.mode_combo.currentIndex()
        mode_changed = (mode != self._last_mode)
        self._last_mode = mode

        # NOTE: fig_width/fig_height are for EXPORT only (applied in _save).
        # Do NOT call set_size_inches here — it fights the canvas widget size.

        # Always clear for modes with colorbars or gridspec (avoids stale axes)
        colorbar_modes = {2, 4, 7, 8, 9}   # colormap, band+DOS, fermi, spin tex, vel
        if mode_changed or mode in colorbar_modes:
            self.figure.clear()
        self.figure.patch.set_facecolor("#0f172a" if dk else "#ffffff")

        if mode == 4:
            self.engine.plot_band_dos()
        else:
            ax = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
            if   mode==0: self.engine.plot_bands(ax, s["show_gap"], s["show_mstar"])
            elif mode==1: self.engine.plot_fatbands(ax, self.orb_combo.currentText(),
                              self.sp_atom.value() if self.sp_atom.value()>=0 else None,
                              self.sp_scale.value())
            elif mode==2: self.engine.plot_colormap_bands(ax, self.orb_combo.currentText(),
                              self.sp_atom.value() if self.sp_atom.value()>=0 else None)
            elif mode==3: self.engine.plot_dos(ax, show_s=s["dos_s"], show_p=s["dos_p"],
                              show_d=s["dos_d"], fill=s["dos_fill"],
                              spin_mirror=s["spin_mirror"], ion_filter=s["ion_filter"])
            elif mode==5: self.engine.plot_stacked_dos(ax, ion_labels=self.data.get("ions",[]))
            elif mode==6: self.engine.plot_bz_auto(ax)
            elif mode==7: self.engine.plot_fermi_surface(ax, self.sp_fs.value()-1)
            elif mode==8: self.engine.plot_spin_texture(ax)
            elif mode==9: self.engine.plot_band_velocity(ax)
            try:
                self.figure.tight_layout(pad=1.5)
            except Exception:
                # tight_layout fails with colorbars / twin axes — use manual padding
                self.figure.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.12)
        self.canvas.draw_idle()  # non-blocking render

    def _edit_kpath(self):
        if not self.data: return
        dlg=QDialog(self); dlg.setWindowTitle("Edit k-path labels"); dlg.resize(440,360)
        v=QVBoxLayout(dlg)
        v.addWidget(QLabel("<b>One per line:  index , label</b><br>"
                           "Use | for connection points (e.g. <code>60,K|Γ</code>)<br>"
                           "Tip: use the K-point Helper tab to find labels."))
        te=QTextEdit(); te.setFont(QFont("Courier New",10)); te.setStyleSheet("background:#fff;color:#1e293b;")
        te.setPlainText("\n".join(f"{i},{l}" for i,l in self.data["klabels"])); v.addWidget(te)
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); v.addWidget(bb)
        if dlg.exec()==QDialog.DialogCode.Accepted:
            nl=[]
            for line in te.toPlainText().strip().split("\n"):
                p=line.strip().split(",",1)
                if len(p)==2:
                    try: nl.append((int(p[0]),p[1].strip()))
                    except: pass
            if nl: self.data["klabels"]=nl; self.engine.set_data(self.data); self._replot()

    def _insert_kpath_label(self,label):
        """Called from K-point helper when user clicks 'Add to k-path'."""
        QMessageBox.information(self,"K-path label",
            f"Symbol: {label}\nCopy the index from your k-point list and add:\n  <index>,{label}\nin the k-path editor (Plot → Edit k-path labels…)")

    def _band_selector(self):
        if not self.data: return
        dlg = QDialog(self); dlg.setWindowTitle("Band Selector"); v = QVBoxLayout(dlg)
        nb = self.data["nbands"]
        v.addWidget(QLabel(f"<b>Total bands: {nb}</b><br>"
            "Syntax: <code>5,8,14</code> &nbsp;|&nbsp; <code>1-6,10-15</code> &nbsp;|&nbsp; blank = all"))
        v.addWidget(QLabel("Plot only bands:"))
        le_plot = QLineEdit(self.plot_editor.le_bands.text())
        le_plot.setPlaceholderText("e.g.  1-4,6,8   or blank for all")
        le_plot.setStyleSheet("background:#fff;color:#1e293b;font-family:monospace;")
        v.addWidget(le_plot)
        v.addWidget(QLabel("Highlight bands (thicker accent colour):"))
        le_hl = QLineEdit(self.plot_editor.le_highlight.text())
        le_hl.setPlaceholderText("e.g.  5,6  or blank for none")
        le_hl.setStyleSheet("background:#fff;color:#1e293b;font-family:monospace;")
        v.addWidget(le_hl)
        # Show band energy table
        ev = self.data["eigenvalues"][0,:,:] - self.data["efermi"]
        nk = ev.shape[0]
        v.addWidget(QLabel("Band ranges (min→max E−E_F eV):"))
        tbl = QTableWidget(min(nb,30), 3)
        tbl.setHorizontalHeaderLabels(["Band","E min (eV)","E max (eV)"])
        tbl.setStyleSheet("background:#fff;color:#1e293b;font-size:11px;")
        tbl.setMaximumHeight(200)
        for ib in range(min(nb,30)):
            e=ev[:,ib]; tbl.setItem(ib,0,QTableWidgetItem(str(ib+1)))
            tbl.setItem(ib,1,QTableWidgetItem(f"{e.min():.3f}"))
            tbl.setItem(ib,2,QTableWidgetItem(f"{e.max():.3f}"))
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v.addWidget(tbl)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); v.addWidget(bb)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.plot_editor.le_bands.setText(le_plot.text())
            self.plot_editor.le_highlight.setText(le_hl.text())
            self._replot()

    def _reset_plot_settings(self):
        self.plot_editor.le_bands.setText(""); self.plot_editor.le_title.setText(""); self._replot()

    def _quick_gap(self):
        if not self.data: return
        ev=self.data["eigenvalues"].copy()-self.data["efermi"]; info=find_band_gap(ev)
        if info["type"]=="metal": QMessageBox.information(self,"Band Gap","Metallic — no gap found")
        else: QMessageBox.information(self,"Band Gap",f"Eg = {info['gap']:.4f} eV  ({info['type']})\nVBM = {info['vbm']:.4f} eV  (band {info['vbm_b']+1})\nCBM = {info['cbm']:.4f} eV  (band {info['cbm_b']+1})")

    def _apply_theme(self, theme):
        """Apply a colour/style preset to the plot editor."""
        themes = {
            "Publication": {"c1":"#000000","c2":"#CC0000","dark":False,"linewidth":1.2,
                           "font_family":"STIXGeneral","font_size":12,"grid_major":False},
            "Dark Neon":   {"c1":"#00FFFF","c2":"#FF69B4","dark":True,"linewidth":1.6,
                           "font_family":"DejaVu Sans","font_size":11,"grid_major":True},
            "Nature":      {"c1":"#2166AC","c2":"#D6604D","dark":False,"linewidth":1.4,
                           "font_family":"Arial","font_size":11,"grid_major":False},
            "Pastel":      {"c1":"#7BAFD4","c2":"#F4A36C","dark":False,"linewidth":1.5,
                           "font_family":"DejaVu Sans","font_size":11,"grid_major":True},
            "High Contrast":{"c1":"#1F78B4","c2":"#E31A1C","dark":False,"linewidth":2.0,
                            "font_family":"DejaVu Sans","font_size":13,"grid_major":True},
        }
        t = themes.get(theme,{}); pe = self.plot_editor
        if "c1"        in t: pe.c1=t["c1"];  pe.btn_c1.setStyleSheet(f"background:{t['c1']};border:1.5px solid #e2e8f0;border-radius:5px;")
        if "c2"        in t: pe.c2=t["c2"];  pe.btn_c2.setStyleSheet(f"background:{t['c2']};border:1.5px solid #e2e8f0;border-radius:5px;")
        if "dark"      in t: pe.chk_dark.setChecked(t["dark"])
        if "linewidth" in t: pe.sp_lw.setValue(t["linewidth"])
        if "font_family" in t:
            idx = AVAILABLE_FONTS.index(t["font_family"]) if t["font_family"] in AVAILABLE_FONTS else 0
            pe.font_combo.setCurrentIndex(idx)
        if "font_size" in t: pe.sp_fs.setValue(t["font_size"])
        if "grid_major" in t: pe.chk_grid.setChecked(t["grid_major"])
        self._replot()
        self.status(f"Applied theme: {theme}")

    def _zoom_to_gap_quick(self):
        if not self.data or not self.engine: return
        emin, emax = self.engine.zoom_to_gap()
        if emin is not None:
            self.sp_emin.blockSignals(True); self.sp_emax.blockSignals(True)
            self.sp_emin.setValue(emin); self.sp_emax.setValue(emax)
            self.sp_emin.blockSignals(False); self.sp_emax.blockSignals(False)
            self._replot(); self.status(f"Zoomed to gap: [{emin:.2f}, {emax:.2f}] eV")
        else: self.status("Metallic — no gap to zoom to")

    def _quick_mstar(self):
        if not self.data: return
        ev=self.data["eigenvalues"].copy()-self.data["efermi"]; gap=find_band_gap(ev)
        if gap["type"]=="metal": QMessageBox.information(self,"m*","Metallic"); return
        ms=fit_effective_mass(self.data["kdist"],ev[0,:,gap["cbm_b"]],gap["cbm_k"])
        if ms: QMessageBox.information(self,"Effective mass",f"m* ≈ {abs(ms):.4f} mₑ  (electron-like, CBM parabolic fit)")
        else: QMessageBox.information(self,"Effective mass","Parabolic fit failed (band too flat?)")

    def _read_outcar(self):
        p,_=QFileDialog.getOpenFileName(self,"Open OUTCAR","","OUTCAR files (OUTCAR *);;All (*)")
        if not p: return
        info=read_outcar_info(p)
        lines=["OUTCAR Summary","="*44,
               f"k-points:     {info['nkpoints']}",
               f"Bands:        {info['nbands']}",
               f"NELECT:       {info['nelect']}",
               f"E_Fermi:      {info['efermi']} eV",
               f"Total energy: {info['total_energy']} eV",
               f"Ionic steps:  {info['ionic_steps']}",
               f"Timing:       {info['timing']}",""]
        if info["warnings"]:
            lines.append(f"Warnings ({len(info['warnings'])}):")
            for w in info["warnings"][:10]: lines.append(f"  ! {w}")
        dlg=QDialog(self); dlg.setWindowTitle("OUTCAR Summary"); dlg.resize(520,400)
        v=QVBoxLayout(dlg); te=QTextEdit(); te.setFont(QFont("Courier New",10))
        te.setStyleSheet("background:#fff;color:#1e293b;"); te.setPlainText("\n".join(lines)); te.setReadOnly(True); v.addWidget(te)
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok); bb.accepted.connect(dlg.accept); v.addWidget(bb); dlg.exec()

    def _save(self):
        p,_=QFileDialog.getSaveFileName(self,"Save Figure","bandstructure","PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;EPS (*.eps)")
        if not p: return
        s = self._get_settings()
        fw = s.get("fig_width", 11); fh = s.get("fig_height", 8)
        transparent = s.get("transparent", False)
        orig = self.figure.get_size_inches()
        self.figure.set_size_inches(fw, fh)
        self.figure.savefig(p, dpi=self.plot_editor.sp_dpi.value(), bbox_inches="tight", transparent=transparent)
        self.figure.set_size_inches(*orig)          # restore live size
        self.status(f"Saved: {p}  ({fw}×{fh} in, {self.plot_editor.sp_dpi.value()} DPI)")

    def _save_hq_png(self):
        """Export High-Quality PNG at 600 DPI."""
        p,_=QFileDialog.getSaveFileName(self,"Save High-Res PNG","bandstructure_hq","PNG (*.png)")
        if not p: return
        s = self._get_settings()
        fw = s.get("fig_width", 11); fh = s.get("fig_height", 8)
        transparent = s.get("transparent", False)
        orig = self.figure.get_size_inches()
        self.figure.set_size_inches(fw, fh)
        self.figure.savefig(p, dpi=600, bbox_inches="tight", transparent=transparent)
        self.figure.set_size_inches(*orig)
        self.status(f"Saved: {p} at 600 DPI")

    def _save_latex_ready(self):
        """Export publication-ready PDF + SVG pair, tight bbox."""
        p,_=QFileDialog.getSaveFileName(self,"Save LaTeX-ready","bandstructure","PDF (*.pdf)")
        if not p: return
        s = self._get_settings()
        transparent = s.get("transparent", False)
        base=str(Path(p).with_suffix(""))
        self.figure.savefig(base+".pdf",dpi=300,bbox_inches="tight",
                            facecolor=self.figure.get_facecolor(), transparent=transparent)
        self.figure.savefig(base+".svg",dpi=150,bbox_inches="tight",
                            facecolor=self.figure.get_facecolor(), transparent=transparent)
        self.status(f"Saved: {base}.pdf + {base}.svg")

    def _show_shortcuts(self):
        msg = (
            "Ctrl+O  ....  Open vasprun.xml\n"
            "Ctrl+S  ....  Save figure (PNG/PDF/SVG)\n"
            "Ctrl+Q  ....  Quit\n\n"
            "Plot toolbar: zoom, pan, reset (home button)\n\n"
            "Quick actions:\n"
            "  Gap button  -- zoom energy window to band gap\n"
            "  Bands button -- open band selector dialog\n\n"
            "Plot Editor (right panel):\n"
            "  Band selection: e.g. 1-6,10  -- plot only those bands\n"
            "  Highlight bands: e.g. 5,6  -- draws thick accent line\n"
            "  Zoom to Gap / Reset / Show VB+CB buttons\n"
            "  Per-spin color, overlay color, highlight color pickers\n\n"
            "K-point Helper tab:\n"
            "  Click any k-point -> copy symbol/LaTeX/coords\n\n"
            "Layer Builder:\n"
            "  Double-click a layer entry to edit it\n"
            "  Drag rows to reorder layers\n\n"
            "POSCAR Viewer:\n"
            "  Switch xy/xz/yz projection\n"
            "  Toggle bonds (covalent radius detection)\n\n"
            "Charge Density:\n"
            "   Loads CHGCAR/PARCHG files for 3D volumetric rendering.\n"
            "   Supports spin-polarized magnetization density visualization."
        )
        QMessageBox.information(self, "Keyboard Shortcuts", msg)

    def _on_mouse_move(self, event):
        """Show k-path position and energy in status bar on hover."""
        if event.inaxes and self.data:
            kd = self.data["kdist"]
            x, y = event.xdata, event.ydata
            if x is not None and y is not None and len(kd):
                ik = int(np.argmin(np.abs(kd - x)))
                ef = self.data["efermi"]
                e_abs = y + ef if self.chk_shift.isChecked() else y
                kpt = self.data["kpoints"][ik] if ik < len(self.data["kpoints"]) else [0,0,0]
                self.statusBar().showMessage(
                    f"k-dist={x:.4f}  |  k=[{kpt[0]:.3f},{kpt[1]:.3f},{kpt[2]:.3f}]  |  "
                    f"E−E₁={y:+.4f} eV  |  E={e_abs:.4f} eV  |  k-idx={ik}")

    def _welcome(self):
        self.figure.patch.set_facecolor("#f8fafc")
        ax = self.figure.add_subplot(111); ax.set_facecolor("#f8fafc"); ax.set_axis_off()
        t = ax.transAxes
        
        # Load and display logo if available
        logo_loaded = False
        try:
            import matplotlib.image as mpimg
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            img = mpimg.imread(get_asset_path("logo-vasvz.png"))
            # Adjust zoom to fit nicely above the title
            imagebox = OffsetImage(img, zoom=0.15)
            ab = AnnotationBbox(imagebox, (0.5, 0.88), frameon=False, xycoords='axes fraction')
            ax.add_artist(ab)
            logo_loaded = True
        except Exception:
            pass

        # Adjust vertical spacing based on whether the logo is displayed
        y_title = 0.74 if logo_loaded else 0.84
        y_badge = 0.66 if logo_loaded else 0.76
        y_desc  = 0.58 if logo_loaded else 0.68
        y_line  = 0.515 if logo_loaded else 0.615

        # Title
        ax.text(0.5, y_title, "VaspViz", transform=t, ha="center", va="center",
                fontsize=46, color="#1e293b", fontweight="bold")
        # Version badge
        ax.text(0.5, y_badge, f" v{VERSION} ", transform=t, ha="center", va="center",
                fontsize=13, color="#ffffff", fontweight="600",
                bbox=dict(boxstyle="round,pad=0.4", fc="#2563EB", ec="#1d4ed8", lw=1.5))
        ax.text(0.5, y_desc, f"Professional VASP Electronic Structure Suite  ·  {DEVELOPER}",
                transform=t, ha="center", va="center", fontsize=11, color="#64748b")
        # Separator
        ax.plot([0.08, 0.92], [y_line, y_line], color="#e2e8f0", lw=1.2,
                transform=t, clip_on=False)
        # Feature cards
        cards = [
            ("POSCAR & OUTCAR",  "Structure Tools",       "POSCAR viewer, Layer builder\nWannier90 comparison"),
            ("Analysis Tools",   "Electronic Structure",  "Bands, DOS, FatBands, Projections\nInteractive k-path seeker"),
            ("Convergence",      "P4Vasp Port",           "Monitor energy, forces, stress\nacross ionic steps"),
            ("Charge Density",   "Volumetric Data",       "3D Visualization of CHGCAR / PARCHG\nSupports spin-polarization")
        ]
        y_card_title = 0.40 if logo_loaded else 0.48
        y_card_desc = 0.31 if logo_loaded else 0.39

        for i, (icon, title, desc) in enumerate(cards):
            cx = 0.09 + i * 0.235
            ax.text(cx+0.07, y_card_title, title, transform=t, ha="center", fontsize=10,
                    fontweight="bold", color="#1e293b", va="center")
            ax.text(cx+0.07, y_card_desc, desc,  transform=t, ha="center", fontsize=8.5,
                    color="#64748b", va="center", linespacing=1.6)
        
        y_hint = 0.17 if logo_loaded else 0.21
        y_footer = 0.07 if logo_loaded else 0.09
        
        # Quick-start hint
        ax.text(0.5, y_hint,
                "Ctrl+O — open file    ·    Ctrl+1–6 — switch tabs",
                transform=t, ha="center", va="center", fontsize=10, color="#2563EB",
                bbox=dict(boxstyle="round,pad=0.55", fc="#eff6ff", ec="#bfdbfe", lw=1.2))
        ax.text(0.5, y_footer, "National Institute of Technology Silchar  ·  Assam, India",
                transform=t, ha="center", va="center", fontsize=9, color="#94a3b8")
        self.canvas.draw_idle()

    def _about(self):
        QMessageBox.about(self,f"About VaspViz v{VERSION}",
            f"<b>VaspViz v{VERSION}</b><br>"
            f"Professional VASP Electronic Structure Suite<br>"
            f"<i>Developer: {DEVELOPER}</i><br><br>"
            "<b>Features:</b><br>"
            "• Band structure, fat bands (bubble/colormap), DOS, Band+DOS<br>"
            "• Optical properties: σ, ε, n+ik, α, EELS, R (9 spectra)<br>"
            "• Automatic Brillouin zone from real reciprocal lattice<br>"
            "• Band gap finder, effective mass, curvature analysis<br>"
            "• Band selector (plot any subset of bands)<br>"
            "• K-point Helper with LaTeX names for all crystal systems<br>"
            "• Layer Builder: 24 materials, ZrX₂-style, moire, strain<br>"
            "• POSCAR Viewer: 2D/3D, bond detection, element radii<br>"
            "• Wannier90 comparison overlay<br>"
            "• Electron count display from NELECT/valence table<br>"
            "• Free-form Plot Editor: colors, fonts, DPI, grid, band selection<br>"
            "• CSV export: bands, DOS, projections, gap report<br><br>"
            "<b>Requirements:</b> PyQt6, matplotlib, numpy, scipy, lxml<br><br>"
            f"<i>VaspViz v{VERSION} — {DEVELOPER}</i><br>"
            "National Institute of Technology Silchar, Assam, India")

    def _developer_info(self):
        QMessageBox.information(self,"Developer",
            f"VaspViz v{VERSION}\n\nDeveloped by:\n{DEVELOPER}\n\n"
            "National Institute of Technology Silchar\nAssam, India\n\n"
            "For issues and contributions, please contact the developer.\n"
            "Built with PyQt6, matplotlib, numpy, scipy.")

    def status(self,m): self.statusBar().showMessage(m)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VaspViz")
    app.setWindowIcon(QIcon(get_asset_path("logo-vasvz.png")))
    app.setStyle("Fusion")

    # System-wide palette
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor("#f8fafc"))
    p.setColor(QPalette.ColorRole.WindowText,      QColor("#0f172a"))
    p.setColor(QPalette.ColorRole.Base,            QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor("#f1f5f9"))
    p.setColor(QPalette.ColorRole.Button,          QColor("#f1f5f9"))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor("#0f172a"))
    p.setColor(QPalette.ColorRole.Highlight,       QColor("#2563EB"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.Text,            QColor("#0f172a"))
    p.setColor(QPalette.ColorRole.BrightText,      QColor("#0f172a"))
    app.setPalette(p)

    # Global QSS — applied once; covers every widget type
    app.setStyleSheet(GLOBAL_APP_STYLE)

    # Premium typography — Segoe UI on Windows, falls back gracefully
    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)

    win = MainWindow(); win.show()
    if len(sys.argv) > 1:
        QTimer.singleShot(200, lambda: win._load(sys.argv[1]))
    sys.exit(app.exec())

if __name__=="__main__":
    main()