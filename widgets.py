"""VaspViz — widgets.py: KpointHelperWidget, PlotEditorPanel, AnalysisPanel, WannierCompareWidget."""
import csv, re, math
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QFileDialog, QComboBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QSpinBox, QFrame, QScrollArea, QSizePolicy, QMessageBox,
    QGridLayout, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QLineEdit, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QColorDialog, QTreeWidget, QTreeWidgetItem,
    QApplication, QTabWidget, QFormLayout, QStyle)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPixmap, QIcon, QAction

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

from constants import (HSYM_KPOINTS, AVAILABLE_FONTS, ORBITAL_GROUPS,
                       SPIN_COLORS, SIDEBAR_STYLE, VERSION, DEVELOPER,
                       VALENCE_ELECTRONS, ELEMENT_NUMBERS)
from parsers import Wannier90Parser, read_outcar_info, PoscarParser
from analysis import (find_band_gap, fit_effective_mass, compute_jdos,
                      PlotEngine)

class KpointHelperWidget(QWidget):
    kpath_requested = pyqtSignal(str)

    LINE_PATHS = {
        "Cubic (SC)": {
            "Γ-X-M-Γ-R-X|M-R": [("Γ","0 0 0"),("X","1/2 0 0"),("M","1/2 1/2 0"),("Γ","0 0 0"),("R","1/2 1/2 1/2"),("X","1/2 0 0"),None,("M","1/2 1/2 0"),("R","1/2 1/2 1/2")],
        },
        "FCC": {
            "Γ-X-W-K-Γ-L-U-W": [("Γ","0 0 0"),("X","1/2 0 1/2"),("W","1/2 1/4 3/4"),("K","3/8 3/8 3/4"),("Γ","0 0 0"),("L","1/2 1/2 1/2"),("U","5/8 1/4 5/8"),("W","1/2 1/4 3/4")],
            "Γ-X-M-Γ-R": [("Γ","0 0 0"),("X","1/2 0 1/2"),("M","1/2 1/2 0"),("Γ","0 0 0"),("R","1/2 1/2 1/2")],
        },
        "BCC": {
            "Γ-H-N-Γ-P-H|P-N": [("Γ","0 0 0"),("H","1/2 -1/2 1/2"),("N","0 0 1/2"),("Γ","0 0 0"),("P","1/4 1/4 1/4"),("H","1/2 -1/2 1/2"),None,("P","1/4 1/4 1/4"),("N","0 0 1/2")],
        },
        "Hexagonal": {
            "Γ-M-K-Γ (2D)": [("Γ","0 0 0"),("M","1/2 0 0"),("K","1/3 1/3 0"),("Γ","0 0 0")],
            "Γ-M-K-Γ-A-L-H-A": [("Γ","0 0 0"),("M","1/2 0 0"),("K","1/3 1/3 0"),("Γ","0 0 0"),("A","0 0 1/2"),("L","1/2 0 1/2"),("H","1/3 1/3 1/2"),("A","0 0 1/2"),None,("L","1/2 0 1/2"),("M","1/2 0 0"),None,("K","1/3 1/3 0"),("H","1/3 1/3 1/2")],
        },
        "Tetragonal": {
            "Γ-X-M-Γ-Z-R-A-Z|X-R|M-A": [("Γ","0 0 0"),("X","1/2 0 0"),("M","1/2 1/2 0"),("Γ","0 0 0"),("Z","0 0 1/2"),("R","1/2 0 1/2"),("A","1/2 1/2 1/2"),("Z","0 0 1/2"),None,("X","1/2 0 0"),("R","1/2 0 1/2"),None,("M","1/2 1/2 0"),("A","1/2 1/2 1/2")],
        },
        "Orthorhombic": {
            "Γ-X-S-Y-Γ-Z-U-R-T-Z": [("Γ","0 0 0"),("X","1/2 0 0"),("S","1/2 1/2 0"),("Y","0 1/2 0"),("Γ","0 0 0"),("Z","0 0 1/2"),("U","1/2 0 1/2"),("R","1/2 1/2 1/2"),("T","0 1/2 1/2"),("Z","0 0 1/2")],
        },
        "Monoclinic": {
            "Γ-Y-H-C-E-M-A-X-Γ": [("Γ","0 0 0"),("Y","0 1/2 0"),("H","0 1/2 1/2"),("C","0 0 1/2"),("E","1/2 1/2 1/2"),("M","1/2 1/2 0"),("A","1/2 0 0"),("X","1/2 0 1/2"),("Γ","0 0 0")],
        },
        "Rhombohedral": {
            "Γ-T-L-Γ-F": [("Γ","0 0 0"),("T","1/2 1/2 1/2"),("L","1/2 0 0"),("Γ","0 0 0"),("F","1/2 1/2 0")],
        },
    }

    def __init__(self):
        super().__init__(); self._build()

    def _build(self):
        lay=QVBoxLayout(self); lay.setContentsMargins(6,4,6,6); lay.setSpacing(4)
        lay.addWidget(QLabel("<b style='font-size:12px;color:#1e293b'>K-point Helper & Line-mode Generator</b>"))
        sp = QSplitter(Qt.Orientation.Horizontal)

        # Left: lookup table
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(4,4,4,4); ll.setSpacing(4)
        g0 = QGroupBox("High-Symmetry Points"); g0l = QVBoxLayout(g0); g0l.setSpacing(3)
        top=QHBoxLayout(); top.setSpacing(3); top.addWidget(QLabel("Lattice:"))
        self.lat_combo=QComboBox(); self.lat_combo.addItems(list(HSYM_KPOINTS.keys()))
        self.lat_combo.currentIndexChanged.connect(self._populate); top.addWidget(self.lat_combo)
        g0l.addLayout(top)
        self.table=QTableWidget(0,4); self.table.setHorizontalHeaderLabels(["Sym","LaTeX","Coords","Desc"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet("background:#fff;color:#1e293b;font-size:11px;")
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        g0l.addWidget(self.table)
        det=QHBoxLayout(); det.setSpacing(4)
        self.lbl_sym=QLabel("—"); self.lbl_sym.setStyleSheet("font-size:14px;font-weight:bold;min-width:30px;")
        self.lbl_coords=QLabel("—"); self.lbl_coords.setStyleSheet("font-size:11px;color:#475569;")
        det.addWidget(self.lbl_sym); det.addWidget(self.lbl_coords); det.addStretch(); g0l.addLayout(det)
        br=QHBoxLayout(); br.setSpacing(3)
        for t,f in [("Sym",self._copy_sym),("LaTeX",self._copy_latex),("Add k-path",self._add_to_kpath)]:
            b=QPushButton(t); b.clicked.connect(f)
            if "➕" in t: b.setObjectName("primary")
            br.addWidget(b)
        g0l.addLayout(br); ll.addWidget(g0); ll.addStretch()
        sp.addWidget(left)

        # Right: Line-mode generator
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(4,4,4,4); rl.setSpacing(4)
        g1=QGroupBox("Line-mode KPOINTS Generator"); g1l=QVBoxLayout(g1); g1l.setSpacing(3)
        r1=QHBoxLayout(); r1.setSpacing(3); r1.addWidget(QLabel("Crystal:"))
        self.crystal_combo=QComboBox(); self.crystal_combo.addItems(list(self.LINE_PATHS.keys()))
        self.crystal_combo.currentIndexChanged.connect(self._on_crystal); r1.addWidget(self.crystal_combo)
        g1l.addLayout(r1)
        r2=QHBoxLayout(); r2.setSpacing(3); r2.addWidget(QLabel("Path:"))
        self.path_combo=QComboBox(); r2.addWidget(self.path_combo); g1l.addLayout(r2)
        r3=QHBoxLayout(); r3.setSpacing(3); r3.addWidget(QLabel("Divisions:"))
        self.sp_div=QSpinBox(); self.sp_div.setRange(10,200); self.sp_div.setValue(40)
        r3.addWidget(self.sp_div); r3.addStretch(); g1l.addLayout(r3)
        bg=QPushButton("Generate KPOINTS"); bg.setObjectName("primary"); bg.clicked.connect(self._generate); g1l.addWidget(bg)
        rl.addWidget(g1)

        g2=QGroupBox("Generated KPOINTS"); g2l=QVBoxLayout(g2); g2l.setSpacing(3)
        self.te_output=QTextEdit(); self.te_output.setReadOnly(True); self.te_output.setFont(QFont("Courier New",10))
        self.te_output.setStyleSheet("background:#fff;color:#1e293b;border:1px solid #e2e8f0;border-radius:4px;")
        self.te_output.setPlaceholderText("Click 'Generate KPOINTS' to create line-mode KPOINTS...")
        g2l.addWidget(self.te_output)
        br2=QHBoxLayout(); br2.setSpacing(3)
        bc=QPushButton("Copy"); bc.clicked.connect(self._copy_output); br2.addWidget(bc)
        bs=QPushButton("Save"); bs.clicked.connect(self._save_output); br2.addWidget(bs)
        g2l.addLayout(br2); rl.addWidget(g2)

        g3=QGroupBox("Quick Reference"); g3l=QVBoxLayout(g3); g3l.setSpacing(2)
        for lbl,pth in [("Hex 2D:","Γ—M—K—Γ"),("Hex bulk:","Γ—M—K—Γ—A—L—H—A"),("FCC:","Γ—X—W—K—Γ—L"),
                        ("BCC:","Γ—H—N—Γ—P—H"),("SC:","Γ—X—M—Γ—R—X"),("Tetra:","Γ—X—M—Γ—Z—R—A—Z")]:
            rr=QHBoxLayout(); rr.setSpacing(2)
            rr.addWidget(QLabel(f"<b style='font-size:10px'>{lbl}</b>"))
            le=QLineEdit(pth); le.setReadOnly(True); le.setStyleSheet("background:#f8fafc;font-size:10px;"); rr.addWidget(le,1)
            bc2=QPushButton("Copy path"); bc2.setFixedWidth(60); bc2.clicked.connect(lambda _,p=pth: QApplication.clipboard().setText(p)); rr.addWidget(bc2)
            g3l.addLayout(rr)
        rl.addWidget(g3); rl.addStretch()
        sp.addWidget(right); sp.setStretchFactor(0,1); sp.setStretchFactor(1,1)
        lay.addWidget(sp,stretch=1)
        self.table.itemSelectionChanged.connect(self._on_select)
        self._populate(); self._on_crystal()

    def _populate(self):
        lat=self.lat_combo.currentText(); points=HSYM_KPOINTS.get(lat,[])
        self.table.setRowCount(len(points))
        for row,(sym,latex,coords,desc) in enumerate(points):
            for col,val in enumerate([sym,f"${latex}$",coords,desc]):
                self.table.setItem(row,col,QTableWidgetItem(val))
        if points: self.table.selectRow(0)

    def _on_select(self):
        row=self.table.currentRow(); lat=self.lat_combo.currentText()
        points=HSYM_KPOINTS.get(lat,[])
        if row<len(points):
            sym,_,coords,desc=points[row]
            self.lbl_sym.setText(sym); self.lbl_coords.setText(f"{coords}  {desc}")

    def _on_crystal(self):
        crystal=self.crystal_combo.currentText()
        self.path_combo.clear(); self.path_combo.addItems(list(self.LINE_PATHS.get(crystal,{}).keys()))

    def _generate(self):
        crystal=self.crystal_combo.currentText(); pname=self.path_combo.currentText()
        nodes=self.LINE_PATHS.get(crystal,{}).get(pname,[])
        if not nodes: self.te_output.setPlainText("No path selected"); return
        ndiv=self.sp_div.value()
        lines=[f"K-POINTS along {pname}",str(ndiv),"Line-mode","Reciprocal"]
        i=0
        while i<len(nodes):
            if nodes[i] is None: lines.append(""); i+=1; continue
            if i+1<len(nodes) and nodes[i+1] is not None:
                la,ca=nodes[i]; lb,cb=nodes[i+1]
                lines.append(f"  {ca}   ! {la}"); lines.append(f"  {cb}   ! {lb}"); lines.append(""); i+=2
            else:
                la,ca=nodes[i]; lines.append(f"  {ca}   ! {la}"); i+=1
        self.te_output.setPlainText("\n".join(lines))

    def _copy_output(self):
        t=self.te_output.toPlainText()
        if t: QApplication.clipboard().setText(t)

    def _save_output(self):
        t=self.te_output.toPlainText()
        if not t: return
        p,_=QFileDialog.getSaveFileName(self,"Save KPOINTS","KPOINTS","All (*)")
        if p:
            with open(p,"w") as f: f.write(t)

    def _get_current(self):
        row=self.table.currentRow(); pts=HSYM_KPOINTS.get(self.lat_combo.currentText(),[])
        if row<len(pts): return pts[row]
        return None

    def _copy_sym(self):
        p=self._get_current()
        if p: QApplication.clipboard().setText(p[0])

    def _copy_latex(self):
        p=self._get_current()
        if p: QApplication.clipboard().setText(f"${p[1]}$")

    def _add_to_kpath(self):
        p=self._get_current()
        if p: self.kpath_requested.emit(p[0])


# ══════════════════════════════════════════════════════════════════════════════
#  OPTICAL PROPERTIES TAB
# ══════════════════════════════════════════════════════════════════════════════

class PlotEditorPanel(QWidget):
    settings_changed=pyqtSignal(dict)

    def __init__(self):
        super().__init__(); self._build()

    def _build(self):
        # ── Outer layout: just holds the scroll area ──────────────────────────
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea { background:#f8fafc; border:none; }
            QScrollBar:vertical { background:#f1f5f9; width:7px; border-radius:3px; }
            QScrollBar::handle:vertical { background:#cbd5e1; border-radius:3px; min-height:20px; }
            QScrollBar::handle:vertical:hover { background:#94a3b8; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
        """)

        # Content widget that lives inside the scroll area
        content = QWidget(); content.setStyleSheet("background:#f8fafc;")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(6, 6, 6, 10)
        lay.setSpacing(6)

        lay.addWidget(QLabel("<b style='font-size:13px;color:#1e293b'>Plot Editor</b>"))

        # Axes & ticks
        g1=QGroupBox("Axes & Ticks"); g1l=QGridLayout(g1); g1l.setSpacing(4); g1l.setContentsMargins(6,10,6,6)
        g1l.addWidget(QLabel("Y-tick:"),0,0); self.sp_ytick=QDoubleSpinBox(); self.sp_ytick.setRange(0.1,5); self.sp_ytick.setValue(2.0); self.sp_ytick.setSingleStep(0.5); g1l.addWidget(self.sp_ytick,0,1)
        g1l.addWidget(QLabel("Line w:"),1,0); self.sp_lw=QDoubleSpinBox(); self.sp_lw.setRange(0.3,5); self.sp_lw.setValue(1.5); self.sp_lw.setSingleStep(0.1); g1l.addWidget(self.sp_lw,1,1)
        g1l.addWidget(QLabel("Band α:"),2,0); self.sp_band_alpha=QDoubleSpinBox(); self.sp_band_alpha.setRange(0.1,1.0); self.sp_band_alpha.setValue(1.0); self.sp_band_alpha.setSingleStep(0.05); g1l.addWidget(self.sp_band_alpha,2,1)
        g1l.addWidget(QLabel("Font sz:"),3,0); self.sp_fs=QSpinBox(); self.sp_fs.setRange(7,20); self.sp_fs.setValue(11); g1l.addWidget(self.sp_fs,3,1)
        g1l.addWidget(QLabel("Font:"),4,0); self.font_combo=QComboBox(); self.font_combo.addItems(AVAILABLE_FONTS); g1l.addWidget(self.font_combo,4,1)
        g1l.addWidget(QLabel("DPI:"),5,0); self.sp_dpi=QSpinBox(); self.sp_dpi.setRange(72,600); self.sp_dpi.setValue(200); g1l.addWidget(self.sp_dpi,5,1)
        g1l.addWidget(QLabel("Marker:"),6,0); self.marker_combo=QComboBox(); self.marker_combo.addItems(["None","dot","circle","square","diamond"]); g1l.addWidget(self.marker_combo,6,1)
        lay.addWidget(g1)

        # Colors
        g2=QGroupBox("Colors"); g2l=QGridLayout(g2); g2l.setSpacing(4)
        colors_list = [("Spin↑ / Band:","c1","#2563EB"),("Spin↓:","c2","#DC2626"),
                       ("Overlay:","c3","#A855F7"),("Wannier:","c4","#16A34A"),
                       ("DOS s:","dos_s_col","#2563EB"),("DOS p:","dos_p_col","#16A34A"),
                       ("DOS d:","dos_d_col","#EA580C"),("DOS Tot:","dos_tot_col","#374151")]
        for i, (lbl,attr,col) in enumerate(colors_list):
            r, c = divmod(i, 2)
            g2l.addWidget(QLabel(lbl), r, c*2)
            btn=QPushButton(); btn.setFixedSize(30,22)
            setattr(self,attr,col)
            btn.setStyleSheet(f"background:{col};border:1.5px solid #e2e8f0;border-radius:5px;")
            btn.clicked.connect(lambda _,a=attr: self._pick(a))
            setattr(self,f"btn_{attr}",btn); g2l.addWidget(btn, r, c*2+1)
        lay.addWidget(g2)

        # Checkboxes
        CHK_STYLE = """
            QCheckBox { color:#374151; font-size:12px; spacing:8px; padding:2px 0; }
            QCheckBox::indicator { width:17px; height:17px; border-radius:4px;
                border:2px solid #cbd5e1; background:#fff; }
            QCheckBox::indicator:hover { border-color:#93c5fd; background:#eff6ff; }
            QCheckBox::indicator:checked { background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 #3b82f6,stop:1 #2563EB); border-color:#2563EB; }
        """
        g3=QGroupBox("Display Options"); g3l=QVBoxLayout(g3); g3l.setSpacing(5)
        self.chk_dark=QCheckBox("Dark background")
        self.chk_fermi=QCheckBox("Fermi level line"); self.chk_fermi.setChecked(True)
        self.chk_klines=QCheckBox("K-symmetry lines"); self.chk_klines.setChecked(True)
        self.chk_gap=QCheckBox("Annotate band gap")
        self.chk_mstar=QCheckBox("Show m* on plot")
        self.chk_grid=QCheckBox("Major grid"); self.chk_grid.setChecked(True)
        self.chk_minor=QCheckBox("Minor ticks"); self.chk_minor.setChecked(True)
        self.chk_spines=QCheckBox("Show spines"); self.chk_spines.setChecked(True)
        for chk in [self.chk_dark,self.chk_fermi,self.chk_klines,self.chk_gap,self.chk_mstar,self.chk_grid,self.chk_minor,self.chk_spines]:
            chk.setStyleSheet(CHK_STYLE); g3l.addWidget(chk)
        fr=QHBoxLayout(); fr.addWidget(QLabel("E_F style:"))
        self.fermi_style_combo=QComboBox(); self.fermi_style_combo.addItems(["dashed","solid","dotted","dashdot"]); fr.addWidget(self.fermi_style_combo,1)
        self.fermi_col="#64748b"; self.btn_fermi_col=QPushButton(); self.btn_fermi_col.setFixedSize(30,22)
        self.btn_fermi_col.setStyleSheet(f"background:{self.fermi_col};border:1px solid #e2e8f0;border-radius:4px;")
        self.btn_fermi_col.setToolTip("Fermi line color"); self.btn_fermi_col.clicked.connect(self._pick_fermi_col); fr.addWidget(self.btn_fermi_col)
        g3l.addLayout(fr)
        lay.addWidget(g3)

        # ── Spine Customization ──
        g_sp = QGroupBox("Spine Customization"); g_spl = QGridLayout(g_sp); g_spl.setSpacing(4); g_spl.setContentsMargins(6,10,6,6)

        # Which sides to show
        g_spl.addWidget(QLabel("Sides:"), 0, 0)
        spine_sides_row = QHBoxLayout(); spine_sides_row.setSpacing(6)
        self.chk_sp_left   = QCheckBox("L");  self.chk_sp_left.setChecked(True)
        self.chk_sp_right  = QCheckBox("R");  self.chk_sp_right.setChecked(False)
        self.chk_sp_bottom = QCheckBox("B");  self.chk_sp_bottom.setChecked(True)
        self.chk_sp_top    = QCheckBox("T");  self.chk_sp_top.setChecked(False)
        for chk in [self.chk_sp_left, self.chk_sp_right, self.chk_sp_bottom, self.chk_sp_top]:
            chk.setStyleSheet(CHK_STYLE); spine_sides_row.addWidget(chk)
        g_spl.addLayout(spine_sides_row, 0, 1)

        # Color
        g_spl.addWidget(QLabel("Color:"), 1, 0)
        self.spine_col = "#1e293b"
        self.btn_spine_col = QPushButton()
        self.btn_spine_col.setFixedSize(30, 22)
        self.btn_spine_col.setStyleSheet(f"background:{self.spine_col};border:1px solid #e2e8f0;border-radius:4px;")
        self.btn_spine_col.setToolTip("Spine color")
        self.btn_spine_col.clicked.connect(self._pick_spine_col)
        g_spl.addWidget(self.btn_spine_col, 1, 1)

        # Line width
        g_spl.addWidget(QLabel("Width:"), 2, 0)
        self.sp_spine_lw = QDoubleSpinBox()
        self.sp_spine_lw.setRange(0.3, 5.0); self.sp_spine_lw.setValue(1.0); self.sp_spine_lw.setSingleStep(0.25)
        g_spl.addWidget(self.sp_spine_lw, 2, 1)

        # Line style
        g_spl.addWidget(QLabel("Style:"), 3, 0)
        self.cmb_spine_style = QComboBox()
        self.cmb_spine_style.addItems(["solid", "dashed", "dotted", "dashdot"])
        g_spl.addWidget(self.cmb_spine_style, 3, 1)

        lay.addWidget(g_sp)

        # ── Tick Customization ──
        g_tk = QGroupBox("Tick Customization"); g_tkl = QGridLayout(g_tk); g_tkl.setSpacing(4); g_tkl.setContentsMargins(6,10,6,6)
        
        g_tkl.addWidget(QLabel("Direction:"), 0, 0)
        self.cmb_tick_dir = QComboBox()
        self.cmb_tick_dir.addItems(["out", "in", "inout"])
        g_tkl.addWidget(self.cmb_tick_dir, 0, 1)

        g_tkl.addWidget(QLabel("Length:"), 1, 0)
        self.sp_tick_len = QDoubleSpinBox()
        self.sp_tick_len.setRange(1.0, 15.0); self.sp_tick_len.setValue(4.0); self.sp_tick_len.setSingleStep(0.5)
        g_tkl.addWidget(self.sp_tick_len, 1, 1)

        g_tkl.addWidget(QLabel("Width:"), 2, 0)
        self.sp_tick_wid = QDoubleSpinBox()
        self.sp_tick_wid.setRange(0.2, 5.0); self.sp_tick_wid.setValue(0.8); self.sp_tick_wid.setSingleStep(0.2)
        g_tkl.addWidget(self.sp_tick_wid, 2, 1)

        g_tkl.addWidget(QLabel("Padding:"), 3, 0)
        self.sp_tick_pad = QDoubleSpinBox()
        self.sp_tick_pad.setRange(0.0, 15.0); self.sp_tick_pad.setValue(3.0); self.sp_tick_pad.setSingleStep(0.5)
        g_tkl.addWidget(self.sp_tick_pad, 3, 1)

        lay.addWidget(g_tk)

        # Title / band selection
        g4=QGroupBox("Figure Text & Bands"); g4l=QVBoxLayout(g4); g4l.setSpacing(6)
        g4l.addWidget(QLabel("Figure title:"))
        self.le_title=QLineEdit(); self.le_title.setPlaceholderText("Optional title…"); g4l.addWidget(self.le_title)
        g4l.addWidget(QLabel("Plot only bands (e.g. 5,8,14-20):"))
        self.le_bands=QLineEdit(); self.le_bands.setPlaceholderText("blank = all bands"); g4l.addWidget(self.le_bands)
        g4l.addWidget(QLabel("Highlight bands (e.g. 6,7):"))
        hl_row=QHBoxLayout(); hl_row.setSpacing(6)
        self.le_highlight=QLineEdit(); self.le_highlight.setPlaceholderText("blank = none"); hl_row.addWidget(self.le_highlight,1)
        self.btn_hl_col=QPushButton(); self.btn_hl_col.setFixedSize(32,26); self.hl_col="#F59E0B"
        self.btn_hl_col.setStyleSheet(f"background:{self.hl_col};border:1.5px solid #e2e8f0;border-radius:5px;")
        self.btn_hl_col.setToolTip("Highlight color"); self.btn_hl_col.clicked.connect(self._pick_hl); hl_row.addWidget(self.btn_hl_col)
        g4l.addLayout(hl_row)
        lay.addWidget(g4)

        # Quick actions
        g5=QGroupBox("Quick Actions"); g5l=QVBoxLayout(g5); g5l.setSpacing(6)
        btn_style=("QPushButton{background:#f8fafc;color:#374151;border:1px solid #e2e8f0;"
                   "border-radius:6px;padding:7px 10px;font-size:12px;font-weight:500;min-height:30px;}"
                   "QPushButton:hover{background:#f1f5f9;border-color:#bfdbfe;}")
        self.btn_zoom_gap=QPushButton("Zoom to Band Gap"); self.btn_zoom_gap.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown)); self.btn_zoom_gap.setStyleSheet(btn_style); self.btn_zoom_gap.clicked.connect(self._zoom_gap); g5l.addWidget(self.btn_zoom_gap)
        self.btn_reset_e=QPushButton("Reset Energy Window"); self.btn_reset_e.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)); self.btn_reset_e.setStyleSheet(btn_style); self.btn_reset_e.clicked.connect(self._reset_e); g5l.addWidget(self.btn_reset_e)
        self.btn_sym_only=QPushButton("Show Only VB+CB"); self.btn_sym_only.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)); self.btn_sym_only.setStyleSheet(btn_style); self.btn_sym_only.clicked.connect(self._show_sym); g5l.addWidget(self.btn_sym_only)
        lay.addWidget(g5)
        # ── Advanced Options ──
        g8=QGroupBox("Advanced Options"); g8l=QGridLayout(g8); g8l.setSpacing(4); g8l.setContentsMargins(6,10,6,6)
        self.chk_band_edges=QCheckBox("Mark VBM/CBM edges"); g8l.addWidget(self.chk_band_edges,0,0,1,2)
        self.chk_total_only=QCheckBox("Total DOS only"); g8l.addWidget(self.chk_total_only,1,0,1,2)
        self.chk_no_total=QCheckBox("No total DOS"); g8l.addWidget(self.chk_no_total,2,0,1,2)
        g8l.addWidget(QLabel("DOS fill α:"),3,0); self.sp_dos_fill_alpha=QDoubleSpinBox(); self.sp_dos_fill_alpha.setRange(0,1); self.sp_dos_fill_alpha.setValue(0.25); self.sp_dos_fill_alpha.setSingleStep(0.05); g8l.addWidget(self.sp_dos_fill_alpha,3,1)
        g8l.addWidget(QLabel("DOS smooth σ:"),4,0); self.sp_dos_sigma=QDoubleSpinBox(); self.sp_dos_sigma.setRange(0,5); self.sp_dos_sigma.setValue(0.0); self.sp_dos_sigma.setSingleStep(0.1); g8l.addWidget(self.sp_dos_sigma,4,1)
        g8l.addWidget(QLabel("DOS max limit:"),5,0); self.sp_dos_max=QDoubleSpinBox(); self.sp_dos_max.setRange(0,10000); self.sp_dos_max.setValue(0.0); self.sp_dos_max.setSingleStep(5.0); self.sp_dos_max.setToolTip("0 = auto"); g8l.addWidget(self.sp_dos_max,5,1)
        self.chk_vector=QCheckBox("Vector export (no raster)"); self.chk_vector.setChecked(True); g8l.addWidget(self.chk_vector,6,0,1,2)
        self.chk_transparent=QCheckBox("Transparent background"); g8l.addWidget(self.chk_transparent,7,0,1,2)
        g8l.addWidget(QLabel("Export W (in):"),8,0); self.sp_fw=QDoubleSpinBox(); self.sp_fw.setRange(4,20); self.sp_fw.setValue(11); self.sp_fw.setSingleStep(0.5); g8l.addWidget(self.sp_fw,8,1)
        g8l.addWidget(QLabel("Export H (in):"),9,0); self.sp_fh=QDoubleSpinBox(); self.sp_fh.setRange(3,16); self.sp_fh.setValue(8); self.sp_fh.setSingleStep(0.5); g8l.addWidget(self.sp_fh,9,1)
        g8l.addWidget(QLabel("Y-label:"),10,0); self.le_ylabel=QLineEdit(); self.le_ylabel.setPlaceholderText("Energy (eV)"); g8l.addWidget(self.le_ylabel,10,1)
        lay.addWidget(g8)

        # Apply button
        btn_apply=QPushButton("Apply to Plot")
        btn_apply.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        btn_apply.setObjectName("primary")
        btn_apply.setMinimumHeight(36)
        btn_apply.clicked.connect(self._emit); lay.addWidget(btn_apply)
        lay.addStretch()

        # Finalise scroll area
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # Connect all widgets for live update
        live = [
            self.sp_ytick, self.sp_lw, self.sp_band_alpha, self.sp_fs, self.sp_dpi,
            self.font_combo, self.marker_combo, self.fermi_style_combo,
            self.chk_dark, self.chk_fermi, self.chk_klines, self.chk_gap, self.chk_mstar,
            self.chk_grid, self.chk_minor, self.chk_spines,
            # Spine customization
            self.chk_sp_left, self.chk_sp_right, self.chk_sp_bottom, self.chk_sp_top,
            self.sp_spine_lw, self.cmb_spine_style,
            # Tick customization
            self.cmb_tick_dir, self.sp_tick_len, self.sp_tick_wid, self.sp_tick_pad,
            self.le_title, self.le_bands, self.le_highlight,
            self.chk_band_edges, self.sp_dos_fill_alpha, self.sp_dos_sigma, self.sp_dos_max,
            self.chk_total_only, self.chk_no_total, self.le_ylabel, self.chk_vector, self.chk_transparent,
            self.sp_fw, self.sp_fh,
        ]
        for w in live:
            if hasattr(w,"valueChanged"): w.valueChanged.connect(self._emit)
            elif hasattr(w,"stateChanged"): w.stateChanged.connect(self._emit)
            elif hasattr(w,"currentIndexChanged"): w.currentIndexChanged.connect(self._emit)
            elif hasattr(w,"editingFinished"): w.editingFinished.connect(self._emit)

    def _pick(self,which):
        cur=getattr(self,which); c=QColorDialog.getColor(QColor(cur),self)
        if c.isValid():
            setattr(self,which,c.name())
            btn=getattr(self,f"btn_{which}")
            btn.setStyleSheet(f"background:{c.name()};border:1px solid #e2e8f0;border-radius:4px;")
            self._emit()

    def _parse_bands(self,s):
        """Parse '5,8,14-20' into list of 0-based indices."""
        if not s.strip(): return None
        result=[]
        for part in s.split(","):
            part=part.strip()
            if "-" in part:
                lo,hi=part.split("-",1)
                try: result.extend(range(int(lo)-1,int(hi)))
                except: pass
            else:
                try: result.append(int(part)-1)
                except: pass
        return result if result else None

    def _pick_hl(self):
        c=QColorDialog.getColor(QColor(self.hl_col),self)
        if c.isValid():
            self.hl_col=c.name()
            self.btn_hl_col.setStyleSheet(f"background:{c.name()};border:1px solid #e2e8f0;border-radius:4px;")
            self._emit()

    def _zoom_gap(self):
        # Signal main window to zoom — emit a special settings update
        self.settings_changed.emit({**self.get_settings(), "_zoom_gap": True})

    def _reset_e(self):
        self.settings_changed.emit({**self.get_settings(), "_reset_e": True})

    def _show_sym(self):
        self.settings_changed.emit({**self.get_settings(), "_show_sym": True})

    def _emit(self,*_):
        self.settings_changed.emit(self.get_settings())

    def _pick_fermi_col(self):
        c=QColorDialog.getColor(QColor(self.fermi_col),self)
        if c.isValid():
            self.fermi_col=c.name()
            self.btn_fermi_col.setStyleSheet(f"background:{c.name()};border:1px solid #e2e8f0;border-radius:4px;")
            self._emit()

    def _pick_spine_col(self):
        c = QColorDialog.getColor(QColor(self.spine_col), self)
        if c.isValid():
            self.spine_col = c.name()
            self.btn_spine_col.setStyleSheet(f"background:{c.name()};border:1px solid #e2e8f0;border-radius:4px;")
            self._emit()

    def get_settings(self):
        marker_map={"None":None,"dot":".","circle":"o","square":"s","diamond":"D"}
        return {
            "ytick_step":self.sp_ytick.value(), "linewidth":self.sp_lw.value(),
            "band_alpha":self.sp_band_alpha.value(),
            "font_size":self.sp_fs.value(), "font_family":self.font_combo.currentText(),
            "export_dpi":self.sp_dpi.value(),
            "marker_style":marker_map.get(self.marker_combo.currentText()),
            "spin_colors":[self.c1, self.c2], "band2_color":self.c3, "wannier_color":self.c4,
            "dos_colors":{"s":self.dos_s_col,"p":self.dos_p_col,"d":self.dos_d_col,"tot":self.dos_tot_col},
            "dark":self.chk_dark.isChecked(), "show_fermi":self.chk_fermi.isChecked(),
            "fermi_style":self.fermi_style_combo.currentText(),
            "fermi_color":self.fermi_col,
            "show_klines":self.chk_klines.isChecked(), "show_gap":self.chk_gap.isChecked(),
            "show_mstar":self.chk_mstar.isChecked(), "grid_major":self.chk_grid.isChecked(),
            "minor_ticks":self.chk_minor.isChecked(), "show_spines":self.chk_spines.isChecked(),
            # Spine customization
            "spine_sides": {
                "left":   self.chk_sp_left.isChecked(),
                "right":  self.chk_sp_right.isChecked(),
                "bottom": self.chk_sp_bottom.isChecked(),
                "top":    self.chk_sp_top.isChecked(),
            },
            "spine_color": self.spine_col,
            "spine_lw":    self.sp_spine_lw.value(),
            "spine_style": self.cmb_spine_style.currentText(),
            "tick_dir":    self.cmb_tick_dir.currentText(),
            "tick_len":    self.sp_tick_len.value(),
            "tick_width":  self.sp_tick_wid.value(),
            "tick_pad":    self.sp_tick_pad.value(),
            "title":self.le_title.text(),
            "selected_bands":self._parse_bands(self.le_bands.text()),
            "highlight_bands":self._parse_bands(getattr(self,"le_highlight",type("x",(),{"text":lambda s:""})()).text()),
            "highlight_color":getattr(self,"hl_col","#F59E0B"),
            # Advanced options
            "fig_width":self.sp_fw.value(), "fig_height":self.sp_fh.value(),
            "spin_channel":"both", "interp_factor":1,
            "dos_sigma":self.sp_dos_sigma.value(), "dos_max":self.sp_dos_max.value(), "legend_on":True, "legend_pos":"upper right", "show_zero":True,
            "band_edges":self.chk_band_edges.isChecked(),
            "vector_export":self.chk_vector.isChecked(),
            "transparent":self.chk_transparent.isChecked(),
            "dos_fill_alpha":self.sp_dos_fill_alpha.value(),
            "total_only":self.chk_total_only.isChecked(),
            "no_total":self.chk_no_total.isChecked(),
            "ylabel":self.le_ylabel.text() or None,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS PANEL
# ══════════════════════════════════════════════════════════════════════════════


class AnalysisPanel(QWidget):
    def __init__(self):
        super().__init__(); self.data=None; self._build()

    def _build(self):
        ml=QVBoxLayout(self); ml.setContentsMargins(6,4,6,6); ml.setSpacing(4)
        ml.addWidget(QLabel("<b style='font-size:12px;color:#1e293b'>Analysis & Export</b>"))
        sp=QSplitter(Qt.Orientation.Horizontal)
        # Left
        lw=QWidget(); lw.setStyleSheet(SIDEBAR_STYLE); ll=QVBoxLayout(lw); ll.setSpacing(5); ll.setContentsMargins(4,4,4,4)
        # Gap
        g1=QGroupBox("Band Gap"); g1l=QVBoxLayout(g1)
        self.gap_label=QLabel("Load data first"); self.gap_label.setWordWrap(True)
        self.gap_label.setStyleSheet("font-size:12px;padding:6px;background:#f1f5f9;border-radius:6px;"); g1l.addWidget(self.gap_label)
        b=QPushButton("Calculate Band Gap"); b.setObjectName("primary"); b.clicked.connect(self.calc_gap); g1l.addWidget(b)
        bcp=QPushButton("Copy result"); bcp.clicked.connect(self._copy_gap_result); g1l.addWidget(bcp)
        ll.addWidget(g1)
        # Effective mass
        g2=QGroupBox("Effective Mass"); g2l=QVBoxLayout(g2)
        self.mstar_label=QLabel("—"); self.mstar_label.setWordWrap(True)
        self.mstar_label.setStyleSheet("font-size:11px;padding:5px;background:#f1f5f9;border-radius:6px;"); g2l.addWidget(self.mstar_label)
        grd=QGridLayout()
        for row,(lbl,attr,lo,hi,val) in enumerate([("Band:","sp_band",1,999,1),("k-pt:","sp_kpt",0,9999,0),("Fit pts:","sp_npts",4,20,8)]):
            grd.addWidget(QLabel(lbl),row,0); sb=QSpinBox(); sb.setRange(lo,hi); sb.setValue(val); sb.setStyleSheet("background:#fff;color:#1e293b;"); setattr(self,attr,sb); grd.addWidget(sb,row,1)
        g2l.addLayout(grd)
        b2=QPushButton("📈 Fit m*"); b2.clicked.connect(self.calc_mstar); g2l.addWidget(b2); ll.addWidget(g2)
        # Optical
        g3=QGroupBox("JDOS (quick)"); g3l=QVBoxLayout(g3)
        gr=QHBoxLayout(); gr.addWidget(QLabel("Broadening:")); self.sp_broad=QDoubleSpinBox(); self.sp_broad.setRange(.01,1.); self.sp_broad.setValue(.1); self.sp_broad.setStyleSheet("background:#fff;color:#1e293b;"); gr.addWidget(self.sp_broad)
        g3l.addLayout(gr)
        b3=QPushButton("Show JDOS"); b3.clicked.connect(self.calc_optical); g3l.addWidget(b3)
        b3b=QPushButton("∫ DOS electron count"); b3b.clicked.connect(self._show_dos_integration); g3l.addWidget(b3b)
        ll.addWidget(g3)
        # Curvature
        g4=QGroupBox("Band Curvature"); g4l=QVBoxLayout(g4)
        cr=QHBoxLayout(); cr.addWidget(QLabel("Band:")); self.sp_curv_band=QSpinBox(); self.sp_curv_band.setRange(1,999); self.sp_curv_band.setStyleSheet("background:#fff;color:#1e293b;"); cr.addWidget(self.sp_curv_band)
        g4l.addLayout(cr); b4=QPushButton("📐 Curvature"); b4.clicked.connect(self.calc_curvature); g4l.addWidget(b4); ll.addWidget(g4)
        ll.addStretch()
        # Right
        rw=QWidget(); rw.setStyleSheet(SIDEBAR_STYLE); rl=QVBoxLayout(rw); rl.setSpacing(5); rl.setContentsMargins(4,4,4,4)
        g5=QGroupBox("Structure & INCAR"); g5l=QVBoxLayout(g5)
        self.incar_table=QTableWidget(0,2); self.incar_table.setHorizontalHeaderLabels(["Parameter","Value"])
        self.incar_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.incar_table.setAlternatingRowColors(True)
        self.incar_table.setStyleSheet("background:#fff;color:#1e293b;"); g5l.addWidget(self.incar_table)
        # Electron count display
        self.lbl_elec_info=QLabel("—"); self.lbl_elec_info.setWordWrap(True)
        self.lbl_elec_info.setStyleSheet("font-size:11px;padding:6px;background:#f0fdf4;border-radius:6px;border:1px solid #bbf7d0;color:#166534;")
        g5l.addWidget(self.lbl_elec_info); rl.addWidget(g5)
        g6=QGroupBox("Export Data"); g6l=QVBoxLayout(g6)
        for txt,fn in [("Band energies CSV",self.export_bands),("DOS CSV",self.export_dos),
                        ("Projections CSV",self.export_proj),("Gap report TXT",self.export_gap)]:
            b=QPushButton(txt); b.clicked.connect(fn); g6l.addWidget(b)
        rl.addWidget(g6); rl.addStretch()
        sp.addWidget(lw); sp.addWidget(rw)
        sp.setStretchFactor(0, 1); sp.setStretchFactor(1, 1)
        ml.addWidget(sp, stretch=1)

    def set_data(self,data):
        self.data=data
        incar=data.get("incar",{})
        self.incar_table.setRowCount(len(incar))
        for row,(k,v) in enumerate(incar.items()):
            self.incar_table.setItem(row,0,QTableWidgetItem(k)); self.incar_table.setItem(row,1,QTableWidgetItem(str(v)))
        self.sp_band.setMaximum(data["nbands"]); self.sp_kpt.setMaximum(data["nkpoints"]-1)
        self.sp_curv_band.setMaximum(data["nbands"]); self.gap_label.setText("Click 'Calculate Band Gap'")
        # Electron info
        ions=data.get("ions",[]); n_elec=data.get("n_electrons",0); n_ions=len(ions)
        species_str=", ".join(f"{s}" for s in dict.fromkeys(ions))
        self.lbl_elec_info.setText(
            f"🔬 System: {data['system']}\n"
            f"⚛️  Ions ({n_ions}): {species_str}\n"
            f"⚡ Valence electrons: ~{n_elec}\n"
            f"📊 Bands: {data['nbands']}  |  k-pts: {data['nkpoints']}\n"
            f"🌊 E_F = {data['efermi']:.4f} eV\n"
            f"🔄 {'Spin-polarized' if data.get('spin_polarized') else 'Non-spin-polarized'}"
        )

    def calc_gap(self):
        if not self.data: QMessageBox.warning(self,"","Load data first."); return
        ev=self.data["eigenvalues"].copy()-self.data["efermi"]
        info=find_band_gap(ev); kd=self.data["kdist"]
        if info["type"]=="metal": self.gap_label.setText("METALLIC — no band gap")
        else:
            self.gap_label.setText(f"<b>Eg = {info['gap']:.4f} eV ({info['type']})</b><br>"
                                    f"VBM: {info['vbm']:.4f} eV @ k[{info['vbm_k']}] kd={kd[info['vbm_k']]:.3f} (band {info['vbm_b']+1})<br>"
                                    f"CBM: {info['cbm']:.4f} eV @ k[{info['cbm_k']}] kd={kd[info['cbm_k']]:.3f} (band {info['cbm_b']+1})")

    def _copy_gap_result(self):
        """Copy gap label text to clipboard."""
        import re
        txt = self.gap_label.text()
        txt = re.sub(r"<[^>]+>", "", txt).replace("&lt;","<").replace("&gt;",">")
        QApplication.clipboard().setText(txt.strip())

    def calc_mstar(self):
        if not self.data: QMessageBox.warning(self,"","Load data first."); return
        ib=self.sp_band.value()-1; ik=self.sp_kpt.value()
        ev=self.data["eigenvalues"][0,:,ib]-self.data["efermi"]
        ms=fit_effective_mass(self.data["kdist"],ev,ik,self.sp_npts.value())
        if ms is None: self.mstar_label.setText("❌ Fit failed (flat or too few pts)")
        else:
            sign="electron-like" if ms>0 else "hole-like"
            self.mstar_label.setText(f"<b>m* = {abs(ms):.4f} mₑ</b> ({sign})<br>Band {ib+1} @ k[{ik}] kd={self.data['kdist'][ik]:.3f}")

    def calc_optical(self):
        if not self.data: QMessageBox.warning(self,"","Load data first."); return
        ev=self.data["eigenvalues"][0,:,:]-self.data["efermi"]
        energies=np.linspace(0.01,8,500); jdos=compute_jdos(energies,ev,self.sp_broad.value())
        fig=Figure(figsize=(8,5)); canvas=FigureCanvas(fig); ax=fig.add_subplot(111)
        if jdos.max()>0: jdos/=jdos.max()
        ax.plot(energies,jdos,lw=1.6,color="#2563EB"); ax.fill_between(energies,0,jdos,alpha=0.25,color="#2563EB")
        ax.set_xlabel("Photon energy (eV)"); ax.set_ylabel("JDOS (norm.)"); ax.set_ylim(0,1.1)
        ax.set_title(f"JDOS — {self.data['system']}"); ax.grid(alpha=0.15); fig.tight_layout(); canvas.draw()
        dlg=QDialog(self); dlg.setWindowTitle("JDOS"); dlg.resize(740,520)
        v=QVBoxLayout(dlg); v.addWidget(canvas)
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok); bb.accepted.connect(dlg.accept); v.addWidget(bb); dlg.exec()

    def calc_curvature(self):
        if not self.data: QMessageBox.warning(self,"","Load data first."); return
        ib=self.sp_curv_band.value()-1
        ev=self.data["eigenvalues"][0,:,ib]-self.data["efermi"]; kd=self.data["kdist"]
        ev_sm=gaussian_filter1d(ev,sigma=1.0); curv=np.gradient(np.gradient(ev_sm,kd),kd)
        fig=Figure(figsize=(10,5)); canvas=FigureCanvas(fig)
        ax1=fig.add_subplot(121)
        ax1.plot(kd,ev,"b-",lw=0.9,alpha=0.5,label="Original"); ax1.plot(kd,ev_sm,"r-",lw=1.6,label="Smoothed")
        ax1.axhline(0,color="gray",ls="--",alpha=0.5); ax1.set_xlabel("k-path"); ax1.set_ylabel("E−E_F (eV)")
        ax1.set_title(f"Band {ib+1}"); ax1.legend(fontsize=9); ax1.grid(alpha=0.2)
        ax2=fig.add_subplot(122)
        ax2.plot(kd,curv,"g-",lw=1.6); ax2.axhline(0,color="gray",ls="--",alpha=0.5)
        ax2.set_xlabel("k-path"); ax2.set_ylabel("d²E/dk² (eV/Å²)"); ax2.set_title("Curvature"); ax2.grid(alpha=0.2)
        fig.tight_layout(); canvas.draw()
        dlg=QDialog(self); dlg.setWindowTitle(f"Curvature — Band {ib+1}"); dlg.resize(920,500)
        v=QVBoxLayout(dlg); v.addWidget(canvas)
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok); bb.accepted.connect(dlg.accept); v.addWidget(bb); dlg.exec()

    def _show_dos_integration(self):
        """Integrate DOS up to E_F and compare with NELECT."""
        if not self.data: QMessageBox.warning(self,"","Load data first."); return
        dos = self.data.get("dos")
        if not dos: QMessageBox.warning(self,"","No DOS data.\nRun VASP with ISTART=0, NSW=0."); return
        total = dos["total"].get("spin 1")
        if total is None: QMessageBox.warning(self,"","No total DOS."); return
        e = total[:,0]; d = total[:,1]; ef = self.data["efermi"]
        mask = e <= ef
        if not mask.any(): QMessageBox.warning(self,"","No states below E_F."); return
        de = np.diff(e); de = np.append(de, de[-1])
        n_up = float(np.sum(d[mask]*de[mask]))
        n_total = n_up
        spin_line = "Non-spin-polarized"
        if self.data.get("spin_polarized"):
            tot2 = dos["total"].get("spin 2")
            if tot2 is not None:
                d2 = tot2[:,1]; n_dn = float(np.sum(d2[mask]*de[mask]))
                n_total = n_up + n_dn
                spin_line = f"Spin-up: {n_up:.3f} | Spin-dn: {n_dn:.3f} | Total: {n_total:.3f}"
        nelect = self.data.get("n_electrons", "?")
        msg = (f"E_F = {ef:.4f} eV\n\n"
               f"Integrated electrons (up to E_F):\n"
               f"  From DOS: {n_total:.3f}\n"
               f"  NELECT:   {nelect}\n\n"
               f"Spin: {spin_line}\n\n"
               f"Tip: Accuracy improves with denser k-mesh.")
        QMessageBox.information(self, "DOS Electron Count", msg)

    def export_bands(self):
        if not self.data: return
        p,_=QFileDialog.getSaveFileName(self,"Export","bands.csv","CSV (*.csv)")
        if not p: return
        ev=self.data["eigenvalues"]; kd=self.data["kdist"]; ef=self.data["efermi"]
        with open(p,"w",newline="") as f:
            w=csv.writer(f)
            h=["k_dist","kx","ky","kz"]+[f"band{i+1}_s1" for i in range(ev.shape[2])]
            if ev.shape[0]==2: h+=[f"band{i+1}_s2" for i in range(ev.shape[2])]
            w.writerow(h)
            for ik in range(ev.shape[1]):
                row=[kd[ik]]+list(self.data["kpoints"][ik])+[ev[0,ik,ib]-ef for ib in range(ev.shape[2])]
                if ev.shape[0]==2: row+=[ev[1,ik,ib]-ef for ib in range(ev.shape[2])]
                w.writerow(row)
        QMessageBox.information(self,"Done",f"Saved: {p}")

    def export_dos(self):
        if not self.data or not self.data.get("dos"): return
        p,_=QFileDialog.getSaveFileName(self,"Export","dos.csv","CSV (*.csv)")
        if not p: return
        total=self.data["dos"]["total"].get("spin 1")
        if total is None: return
        ef=self.data["efermi"]
        with open(p,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["energy_eV","E_minus_Ef","tdos","idos"])
            for row in total: w.writerow([row[0],row[0]-ef,row[1],row[2]])
        QMessageBox.information(self,"Done",f"Saved: {p}")

    def export_proj(self):
        if not self.data or self.data.get("projections") is None: return
        p,_=QFileDialog.getSaveFileName(self,"Export","proj.csv","CSV (*.csv)")
        if not p: return
        proj=self.data["projections"][0]; kd=self.data["kdist"]; ef=self.data["efermi"]
        nk,nb,ni,no=proj.shape
        with open(p,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["k_idx","k_dist","band","ion","s","py","pz","px","dxy","dyz","dz2","dxz","dx2"])
            for ik in range(nk):
                for ib in range(nb):
                    for ii in range(ni): w.writerow([ik,kd[ik],ib+1,ii]+list(proj[ik,ib,ii,:min(no,9)]))
        QMessageBox.information(self,"Done",f"Saved: {p}")

    def export_gap(self):
        if not self.data: return
        p,_=QFileDialog.getSaveFileName(self,"Export","gap_report.txt","Text (*.txt)")
        if not p: return
        ev=self.data["eigenvalues"].copy()-self.data["efermi"]; info=find_band_gap(ev)
        with open(p,"w") as f:
            f.write(f"Band Gap Report — VaspViz v{VERSION}\nDeveloper: {DEVELOPER}\n"+"="*50+"\n")
            f.write(f"System: {self.data['system']}\nFermi Energy: {self.data['efermi']:.4f} eV\n")
            f.write(f"Ions: {self.data['ions']}\nValence electrons: ~{self.data.get('n_electrons',0)}\n\n")
            if info["type"]=="metal": f.write("System is METALLIC\n")
            else:
                f.write(f"Band Gap: {info['gap']:.4f} eV ({info['type']})\n")
                f.write(f"VBM: {info['vbm']:.4f} eV @ k[{info['vbm_k']}] (band {info['vbm_b']+1})\n")
                f.write(f"CBM: {info['cbm']:.4f} eV @ k[{info['cbm_k']}] (band {info['cbm_b']+1})\n")
        QMessageBox.information(self,"Done",f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  WANNIER COMPARISON
# ══════════════════════════════════════════════════════════════════════════════


class WannierCompareWidget(QWidget):
    def __init__(self):
        super().__init__(); self.vasp_data=None; self.wannier_data=None; self._build()

    def set_vasp_data(self,d):
        self.vasp_data=d
        if d: self.lbl_vasp.setText(f"✓ {d['system']}\n{d['nbands']} bands | {d['nkpoints']} k-pts"); self.lbl_vasp.setStyleSheet("font-size:10px;color:#16A34A;")

    def _build(self):
        W_STYLE = (
            "QWidget{background:#f8fafc;color:#1e293b;}"
            "QGroupBox{font-weight:600;border:1px solid #e2e8f0;border-radius:8px;"
            "margin-top:10px;padding-top:8px;background:#ffffff;color:#1e293b;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;"
            "background:#ffffff;color:#475569;}"
            "QLabel{color:#374151;font-size:12px;}"
            "QPushButton{background:#f8fafc;color:#374151;border:1px solid #e2e8f0;"
            "border-radius:6px;padding:5px 10px;font-size:11px;min-height:26px;}"
            "QPushButton:hover{background:#f1f5f9;border-color:#93c5fd;}"
            "QPushButton#primary{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #3b82f6,stop:1 #2563EB);color:#fff;border-color:#1d4ed8;font-weight:600;}"
            "QDoubleSpinBox,QSpinBox,QComboBox,QLineEdit{background:#fff;color:#1e293b;"
            "border:1px solid #e2e8f0;border-radius:5px;padding:2px 6px;font-size:11px;min-height:22px;}"
            "QCheckBox{color:#374151;font-size:12px;spacing:6px;}"
            "QCheckBox::indicator{width:16px;height:16px;border-radius:4px;border:2px solid #cbd5e1;background:#fff;}"
            "QCheckBox::indicator:checked{background:#2563EB;border-color:#2563EB;}"
            "QScrollBar:vertical{background:#f1f5f9;width:7px;border-radius:3px;}"
            "QScrollBar::handle:vertical{background:#cbd5e1;border-radius:3px;min-height:20px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
            "QTextEdit{background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;color:#1e293b;padding:4px;}"
        )
        lay=QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        scroll_left=QScrollArea(); scroll_left.setWidgetResizable(True)
        scroll_left.setFrameShape(QFrame.Shape.NoFrame)
        scroll_left.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_left.setFixedWidth(298)
        scroll_left.setStyleSheet("QScrollArea{background:#f8fafc;border:none;border-right:1px solid #e2e8f0;}")
        cw=QWidget(); cw.setStyleSheet(W_STYLE); cl=QVBoxLayout(cw)
        cl.setContentsMargins(10,10,10,10); cl.setSpacing(8)
        cl.addWidget(QLabel("<b style='font-size:13px;color:#1e293b'>Wannier90 Tools</b>"))
        tabs=QTabWidget(); tabs.setStyleSheet(
            "QTabBar::tab{padding:5px 10px;font-size:11px;color:#64748b;border:none;background:transparent;}"
            "QTabBar::tab:selected{color:#2563EB;border-bottom:2px solid #2563EB;font-weight:600;}"
            "QTabWidget::pane{border:none;background:#f8fafc;}")
        # ── Tab 0: Compare ─────────────────────────────────────────────────────
        t0=QWidget(); t0l=QVBoxLayout(t0); t0l.setSpacing(6); t0l.setContentsMargins(0,6,0,0)
        g1=QGroupBox("VASP data"); g1l=QVBoxLayout(g1)
        self.lbl_vasp=QLabel("Load vasprun.xml in main tab")
        self.lbl_vasp.setWordWrap(True); self.lbl_vasp.setStyleSheet("font-size:10px;color:#64748B;")
        g1l.addWidget(self.lbl_vasp); t0l.addWidget(g1)
        g2=QGroupBox("Wannier90 bands"); g2l=QVBoxLayout(g2)
        g2l.addWidget(QLabel("Accepts: wannier90_band.dat,\n  seedname_band.dat, bands.dat"))
        btn_w=QPushButton("📂 Load Wannier file"); btn_w.setObjectName("primary")
        btn_w.clicked.connect(self._load); g2l.addWidget(btn_w)
        self.lbl_wannier=QLabel("No file loaded")
        self.lbl_wannier.setStyleSheet("font-size:10px;color:#64748B;")
        g2l.addWidget(self.lbl_wannier); t0l.addWidget(g2)
        g3=QGroupBox("Plot Settings"); g3l=QGridLayout(g3); g3l.setSpacing(5)
        g3l.addWidget(QLabel("VASP color"),0,0)
        self.btn_vc=QPushButton(); self.btn_vc.setFixedSize(40,24); self.vc="#2563EB"
        self.btn_vc.setStyleSheet(f"background:{self.vc};border:1px solid #e2e8f0;border-radius:4px;")
        self.btn_vc.clicked.connect(lambda: self._pick("vc")); g3l.addWidget(self.btn_vc,0,1)
        g3l.addWidget(QLabel("Wannier color"),1,0)
        self.btn_wc=QPushButton(); self.btn_wc.setFixedSize(40,24); self.wc="#DC2626"
        self.btn_wc.setStyleSheet(f"background:{self.wc};border:1px solid #e2e8f0;border-radius:4px;")
        self.btn_wc.clicked.connect(lambda: self._pick("wc")); g3l.addWidget(self.btn_wc,1,1)
        for row,(lbl,attr,lo,hi,val) in enumerate([
                ("E min","sp_wmin",-20,0,-6.),("E max","sp_wmax",0,20,6.),
                ("LW VASP","sp_lv",.3,5,1.5),("LW Wan.","sp_lw",.3,5,1.0)],2):
            sb=QDoubleSpinBox(); sb.setRange(lo,hi); sb.setValue(val)
            g3l.addWidget(QLabel(lbl),row,0); g3l.addWidget(sb,row,1); setattr(self,attr,sb)
        self.chk_shift=QCheckBox("Shift E_F→0"); self.chk_shift.setChecked(True)
        g3l.addWidget(self.chk_shift,6,0,1,2)
        self.chk_show_dev=QCheckBox("Show max |ΔE| deviation"); self.chk_show_dev.setChecked(True)
        g3l.addWidget(self.chk_show_dev,7,0,1,2)
        t0l.addWidget(g3)
        g4=QGroupBox("Energy Window Detector"); g4l=QVBoxLayout(g4); g4l.setSpacing(4)
        g4l.addWidget(QLabel("Auto-suggest dis_win / dis_froz from VASP bands:"))
        gw=QGridLayout(); gw.setSpacing(4)
        gw.addWidget(QLabel("Target # WFs:"),0,0)
        self.sp_nwf=QSpinBox(); self.sp_nwf.setRange(1,200); self.sp_nwf.setValue(8); gw.addWidget(self.sp_nwf,0,1)
        gw.addWidget(QLabel("Frozen margin (eV):"),1,0)
        self.sp_froz=QDoubleSpinBox(); self.sp_froz.setRange(0,3); self.sp_froz.setValue(0.5); gw.addWidget(self.sp_froz,1,1)
        gw.addWidget(QLabel("Outer margin (eV):"),2,0)
        self.sp_outer=QDoubleSpinBox(); self.sp_outer.setRange(0,8); self.sp_outer.setValue(2.0); gw.addWidget(self.sp_outer,2,1)
        g4l.addLayout(gw)
        btn_det=QPushButton("Detect Energy Windows"); btn_det.setObjectName("primary")
        btn_det.clicked.connect(self._detect_wannier_windows); g4l.addWidget(btn_det)
        self.lbl_windows=QLabel("—"); self.lbl_windows.setWordWrap(True)
        self.lbl_windows.setStyleSheet("font-size:10px;padding:5px;background:#f0f9ff;border-radius:4px;"
                                        "border:1px solid #bae6fd;color:#0c4a6e;font-family:monospace;")
        g4l.addWidget(self.lbl_windows)
        btn_cw=QPushButton("Copy wannier90.win block"); btn_cw.clicked.connect(self._copy_windows)
        g4l.addWidget(btn_cw); t0l.addWidget(g4)
        btn_plot=QPushButton("Update Plot"); btn_plot.setObjectName("primary")
        btn_plot.clicked.connect(self._plot); t0l.addWidget(btn_plot)
        btn_save=QPushButton("Save Figure"); btn_save.clicked.connect(self._save)
        t0l.addWidget(btn_save); t0l.addStretch(); tabs.addTab(t0,"Compare")
        # ── Tab 1: Win Generator ───────────────────────────────────────────────
        t1=QWidget(); t1l=QVBoxLayout(t1); t1l.setSpacing(6); t1l.setContentsMargins(0,6,0,0)
        g5=QGroupBox("wannier90.win Generator"); g5l=QVBoxLayout(g5); g5l.setSpacing(5)
        gw2=QGridLayout(); gw2.setSpacing(4)
        gw2.addWidget(QLabel("num_wann:"),0,0)
        self.sp_win_nwann=QSpinBox(); self.sp_win_nwann.setRange(1,200); self.sp_win_nwann.setValue(8); gw2.addWidget(self.sp_win_nwann,0,1)
        gw2.addWidget(QLabel("num_bands:"),1,0)
        self.sp_win_nbands=QSpinBox(); self.sp_win_nbands.setRange(1,500); self.sp_win_nbands.setValue(20); gw2.addWidget(self.sp_win_nbands,1,1)
        gw2.addWidget(QLabel("mp_grid:"),2,0)
        self.le_win_mpgrid=QLineEdit(); self.le_win_mpgrid.setPlaceholderText("e.g. 8 8 1"); gw2.addWidget(self.le_win_mpgrid,2,1)
        gw2.addWidget(QLabel("Projections:"),3,0)
        self.le_win_proj=QLineEdit(); self.le_win_proj.setPlaceholderText("e.g. Mo:d; S:p"); gw2.addWidget(self.le_win_proj,3,1)
        g5l.addLayout(gw2)
        cr=QHBoxLayout()
        self.chk_win_dis=QCheckBox("dis_win"); self.chk_win_dis.setChecked(True); cr.addWidget(self.chk_win_dis)
        self.chk_win_froz=QCheckBox("dis_froz"); self.chk_win_froz.setChecked(True); cr.addWidget(self.chk_win_froz)
        self.chk_win_kpts=QCheckBox("begin kpoints"); cr.addWidget(self.chk_win_kpts)
        g5l.addLayout(cr)
        btn_gen_win=QPushButton("Generate wannier90.win Block"); btn_gen_win.setObjectName("primary")
        btn_gen_win.clicked.connect(self._gen_win_block); g5l.addWidget(btn_gen_win)
        t1l.addWidget(g5)
        self.te_win_output=QTextEdit(); self.te_win_output.setReadOnly(True)
        self.te_win_output.setFont(QFont("Courier New",10))
        self.te_win_output.setPlaceholderText("Generated wannier90.win block appears here...")
        self.te_win_output.setMinimumHeight(200); t1l.addWidget(self.te_win_output,1)
        btn_copy_win=QPushButton("Copy to Clipboard")
        btn_copy_win.clicked.connect(self._copy_win_output); t1l.addWidget(btn_copy_win)
        t1l.addStretch(); tabs.addTab(t1,"Win Generator")
        cl.addWidget(tabs); cl.addStretch()
        scroll_left.setWidget(cw); lay.addWidget(scroll_left)
        self.fig=Figure(figsize=(10,7),dpi=100); self.fig.patch.set_facecolor("#ffffff")
        self.canvas=FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        nav=NavigationToolbar(self.canvas,self); nav.setIconSize(QSize(14,14))
        rw=QWidget(); rl=QVBoxLayout(rw); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0)
        rl.addWidget(nav); rl.addWidget(self.canvas); lay.addWidget(rw,1)
        self._placeholder()


    def _placeholder(self):
        self.fig.clear(); ax=self.fig.add_subplot(111); ax.set_axis_off()
        for txt,y in [("VASP vs Wannier90 Band Comparison",.60),("1. Load vasprun.xml in Electronic Structure tab",.47),("2. Load Wannier90 output file here",.39),("Format: 2-column k-dist  energy (blank lines separate bands)",.31)]:
            ax.text(.5,y,txt,transform=ax.transAxes,ha="center",va="center",fontsize=12 if y>.5 else 10,color="#94A3B8" if y>.5 else "#CBD5E1")
        self.canvas.draw()

    def _pick(self,which):
        c=QColorDialog.getColor(QColor(getattr(self,which)),self)
        if c.isValid():
            setattr(self,which,c.name())
            getattr(self,f"btn_{which}").setStyleSheet(f"background:{c.name()};border:1px solid #e2e8f0;")

    def _load(self):
        p,_=QFileDialog.getOpenFileName(self,"Load Wannier bands","","Wannier90 (*.dat *.gnu);;All (*)")
        if not p: return
        try:
            self.wannier_data=Wannier90Parser(p).parse()
            self.lbl_wannier.setText(f"✓ {Path(p).name}\n{self.wannier_data['nbands']} bands"); self.lbl_wannier.setStyleSheet("font-size:10px;color:#16A34A;"); self._plot()
        except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def _gen_win_block(self):
        """Generate a ready-to-paste wannier90.win block."""
        nwann=self.sp_win_nwann.value(); nbands=self.sp_win_nbands.value()
        mpgrid=self.le_win_mpgrid.text().strip() or "8 8 1"
        proj_raw=self.le_win_proj.text().strip() or "f=0,0,0:l=2"
        lines=[]
        if self.vasp_data:
            lines.append(f"# System: {self.vasp_data.get('system','unknown')}")
            lines.append(f"# E_F = {self.vasp_data.get('efermi',0):.4f} eV")
            lines.append("")
        lines.append(f"num_wann  = {nwann}")
        lines.append(f"num_bands = {nbands}")
        lines.append("")
        if self.chk_win_dis.isChecked():
            if hasattr(self,"_last_windows"):
                w=self._last_windows
                lines.append(f"dis_win_min  = {w['dis_win_min']:.4f}")
                lines.append(f"dis_win_max  = {w['dis_win_max']:.4f}")
            else:
                lines.append("dis_win_min  = # run Detect Windows first")
                lines.append("dis_win_max  = # run Detect Windows first")
        if self.chk_win_froz.isChecked():
            if hasattr(self,"_last_windows"):
                w=self._last_windows
                lines.append(f"dis_froz_min = {w['dis_froz_min']:.4f}")
                lines.append(f"dis_froz_max = {w['dis_froz_max']:.4f}")
            else:
                lines.append("dis_froz_min = # run Detect Windows first")
                lines.append("dis_froz_max = # run Detect Windows first")
        lines.append("")
        lines.append(f"mp_grid : {mpgrid}")
        lines.append("")
        lines.append("begin projections")
        for proj in proj_raw.split(";"):
            lines.append(f"  {proj.strip()}")
        lines.append("end projections")
        if self.chk_win_kpts.isChecked() and self.vasp_data:
            lines.append("\nbegin kpoints")
            for kpt in self.vasp_data.get("kpoints",[]):
                lines.append(f"  {kpt[0]:.8f}  {kpt[1]:.8f}  {kpt[2]:.8f}")
            lines.append("end kpoints")
        self.te_win_output.setPlainText("\n".join(lines))

    def _copy_win_output(self):
        txt=self.te_win_output.toPlainText()
        if txt: QApplication.clipboard().setText(txt)
        else: QMessageBox.warning(self,"","Nothing to copy — click Generate first.")



    def _plot(self):
        if not self.vasp_data: QMessageBox.warning(self,"","Load VASP data first."); return
        if not self.wannier_data: QMessageBox.warning(self,"","Load Wannier file first."); return
        self.fig.clear(); ax=self.fig.add_subplot(111)
        emin=self.sp_wmin.value(); emax=self.sp_wmax.value()
        shift=self.vasp_data["efermi"] if self.chk_shift.isChecked() else 0.
        ev=self.vasp_data["eigenvalues"][0,:,:]-shift
        kd_v=self.vasp_data["kdist"]; kd_w=self.wannier_data["kdist"].copy(); ev_w=self.wannier_data["eigenvalues"]
        if kd_w.max()>0: kd_w=kd_w/kd_w.max()*kd_v.max()
        for ib in range(ev.shape[1]):
            e=ev[:,ib]
            if e.max()<emin-.5 or e.min()>emax+.5: continue
            ax.plot(kd_v,e,color=self.vc,lw=self.sp_lv.value(),alpha=0.75,rasterized=False,label="VASP DFT" if ib==0 else "")
        for ib in range(ev_w.shape[1]):
            e=ev_w[:,ib]
            if e.max()<emin-.5 or e.min()>emax+.5: continue
            ax.plot(kd_w,e,color=self.wc,lw=self.sp_lw.value(),alpha=0.88,ls="--",rasterized=False,label="Wannier90" if ib==0 else "")
        for idx,lbl in self.vasp_data["klabels"]:
            if idx<len(kd_v): ax.axvline(kd_v[idx],color="#94A3B8",lw=0.8,ls="--",alpha=0.5)
        ax.axhline(0,color="#EF4444",lw=1.,ls="--",alpha=0.75)
        ticks=[]; labs=[]
        for idx,lbl in self.vasp_data["klabels"]:
            if idx<len(kd_v): ticks.append(kd_v[idx]); labs.append(lbl or "")
        ax.set_xticks(ticks); ax.set_xticklabels(labs,fontsize=12)
        ax.set_xlim(kd_v[0],kd_v[-1]); ax.set_ylim(emin,emax)
        ax.set_ylabel("E − $E_F$ (eV)" if self.chk_shift.isChecked() else "Energy (eV)",fontsize=11)
        ax.tick_params(axis="x",bottom=False); ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.grid(axis="y",alpha=0.1,lw=0.5); ax.legend(fontsize=10,framealpha=0.9,loc="upper right")
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()

    def _save(self):
        p,_=QFileDialog.getSaveFileName(self,"Save","wannier_compare","PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if p: self.fig.savefig(p,dpi=200,bbox_inches="tight")

    def _detect_wannier_windows(self):
        """Auto-suggest Wannier90 energy windows from VASP band structure."""
        if not self.vasp_data: QMessageBox.warning(self,"","Load VASP data first."); return
        ev=self.vasp_data["eigenvalues"].copy()-self.vasp_data["efermi"]
        n_wf=self.sp_nwf.value(); froz_margin=self.sp_froz.value(); outer_margin=self.sp_outer.value()
        band_means=np.mean(ev[0],axis=0)
        sorted_bands=np.argsort(np.abs(band_means))
        target_bands=sorted(sorted_bands[:n_wf].tolist())
        target_ev=ev[0][:,target_bands]
        e_min_tgt=float(target_ev.min()); e_max_tgt=float(target_ev.max())
        gap_info=find_band_gap(ev)
        if gap_info["type"]!="metal":
            froz_min=gap_info["vbm"]-froz_margin; froz_max=gap_info["cbm"]+froz_margin
        else:
            froz_min=e_min_tgt+(e_max_tgt-e_min_tgt)*0.25
            froz_max=e_min_tgt+(e_max_tgt-e_min_tgt)*0.75
        outer_min=e_min_tgt-outer_margin; outer_max=e_max_tgt+outer_margin
        ef=self.vasp_data["efermi"]
        self._last_windows={"dis_win_min":outer_min+ef,"dis_win_max":outer_max+ef,
                             "dis_froz_min":froz_min+ef,"dis_froz_max":froz_max+ef,"ef":ef}
        bstr=",".join(str(b+1) for b in target_bands[:8])+("..." if len(target_bands)>8 else "")
        txt=(f"Target bands: {bstr}\n\n"
             f"[Relative to EF=0]\n"
             f"  dis_win_min  = {outer_min:+.4f} eV\n"
             f"  dis_win_max  = {outer_max:+.4f} eV\n"
             f"  dis_froz_min = {froz_min:+.4f} eV\n"
             f"  dis_froz_max = {froz_max:+.4f} eV\n\n"
             f"[Absolute for wannier90.win]\n"
             f"  dis_win_min  = {outer_min+ef:.4f}\n"
             f"  dis_win_max  = {outer_max+ef:.4f}\n"
             f"  dis_froz_min = {froz_min+ef:.4f}\n"
             f"  dis_froz_max = {froz_max+ef:.4f}")
        self.lbl_windows.setText(txt)
        # NOTE: We do NOT call self._plot() here — that calls tight_layout which
        # triggers a Qt geometry event that can un-maximize the window on some
        # platforms. Instead we update the axes in-place.
        if not self.fig.axes:
            self._plot()
        if self.fig.axes:
            ax = self.fig.axes[0]
            # Remove old window spans if present
            ax.patches = [p for p in ax.patches if not getattr(p, "_vaspviz_window", False)]
            # Draw frozen window (green band)
            sp1 = ax.axhspan(froz_min, froz_max, alpha=0.13, color="#22C55E", zorder=0)
            sp1._vaspviz_window = True
            # Draw outer window (blue band)
            sp2 = ax.axhspan(outer_min, outer_max, alpha=0.06, color="#3B82F6", zorder=0)
            sp2._vaspviz_window = True
            # Boundary lines
            for y in [froz_min, froz_max]:
                ax.axhline(y, color="#22C55E", lw=1.3, ls="--", alpha=0.75, zorder=1)
            for y in [outer_min, outer_max]:
                ax.axhline(y, color="#3B82F6", lw=1.3, ls=":", alpha=0.75, zorder=1)
            # Minimal legend without triggering layout recalculation
            handles = [
                mpatches.Patch(color="#22C55E", alpha=0.5, label=f"Frozen [{froz_min:+.2f}, {froz_max:+.2f}] eV"),
                mpatches.Patch(color="#3B82F6", alpha=0.3, label=f"Outer [{outer_min:+.2f}, {outer_max:+.2f}] eV"),
            ]
            ax.legend(handles=handles, fontsize=8, framealpha=0.85, loc="upper right")
            self.canvas.draw_idle()  # draw_idle() avoids triggering resize events

    def _copy_windows(self):
        if not hasattr(self,"_last_windows"): QMessageBox.warning(self,"","Run Detect Windows first."); return
        w=self._last_windows
        block=("# Wannier90 energy windows (wannier90.win)\n"
               f"dis_win_min  = {w['dis_win_min']:.4f}\n"
               f"dis_win_max  = {w['dis_win_max']:.4f}\n"
               f"dis_froz_min = {w['dis_froz_min']:.4f}\n"
               f"dis_froz_max = {w['dis_froz_max']:.4f}\n")
        QApplication.clipboard().setText(block)
        self.lbl_windows.setText(self.lbl_windows.text()+"\n\n✓ Copied!")


# ══════════════════════════════════════════════════════════════════════════════
#  P4VASP PANEL
# ══════════════════════════════════════════════════════════════════════════════


class P4VaspPanel(QWidget):
    """P4Vasp-inspired panel: convergence monitor, force analysis, stress."""
    def __init__(self):
        super().__init__(); self._osz_data=None; self._force_data=None; self._stress_data=None; self._build()
    def _build(self):
        ml=QVBoxLayout(self); ml.setContentsMargins(6,4,6,6); ml.setSpacing(4)
        ml.addWidget(QLabel("<b style='font-size:12px;color:#1e293b'>P4Vasp Tools</b>"))
        sp=QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel (Scrollable)
        ls=QScrollArea(); ls.setWidgetResizable(True); ls.setFrameShape(QFrame.Shape.NoFrame)
        ls.setMinimumWidth(220); ls.setMaximumWidth(280)
        ls.setStyleSheet("QScrollArea{background:#f8fafc;border:none;border-right:1px solid #e2e8f0;}")
        
        W_STYLE = (
            "QWidget{background:#f8fafc;color:#1e293b;}"
            "QGroupBox{font-weight:600;border:1px solid #e2e8f0;border-radius:8px;margin-top:10px;padding-top:8px;background:#ffffff;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;background:#ffffff;color:#475569;}"
            "QLabel{color:#374151;font-size:11px;}"
            "QPushButton{background:#f8fafc;color:#374151;border:1px solid #e2e8f0;border-radius:6px;padding:5px;font-size:11px;min-height:24px;}"
            "QPushButton:hover{background:#f1f5f9;border-color:#93c5fd;}"
            "QPushButton#primary{background:#2563EB;color:#fff;border-color:#1d4ed8;font-weight:600;}"
        )
        left=QWidget(); left.setStyleSheet(W_STYLE)
        ll=QVBoxLayout(left); ll.setSpacing(8); ll.setContentsMargins(10,10,10,10)
        
        g_auto=QGroupBox("Auto Load"); gl_auto=QVBoxLayout(g_auto)
        btn_auto=QPushButton("Load OSZICAR + OUTCAR"); btn_auto.setObjectName("primary")
        btn_auto.clicked.connect(self._auto_load); gl_auto.addWidget(btn_auto); ll.addWidget(g_auto)

        g1=QGroupBox("OSZICAR"); g1l=QVBoxLayout(g1); g1l.setSpacing(4)
        self.lbl_osz=QLabel("No OSZICAR"); self.lbl_osz.setWordWrap(True)
        self.lbl_osz.setStyleSheet("font-size:10px;padding:4px;background:#f1f5f9;border-radius:4px;")
        g1l.addWidget(self.lbl_osz)
        b1=QPushButton("📂 Load OSZICAR"); b1.clicked.connect(self._load_oszicar); g1l.addWidget(b1)
        ll.addWidget(g1)

        g2=QGroupBox("Forces"); g2l=QVBoxLayout(g2); g2l.setSpacing(4)
        self.lbl_force=QLabel("No OUTCAR"); self.lbl_force.setWordWrap(True)
        self.lbl_force.setStyleSheet("font-size:10px;padding:4px;background:#f1f5f9;border-radius:4px;")
        g2l.addWidget(self.lbl_force)
        b2=QPushButton("📂 Load OUTCAR"); b2.clicked.connect(self._load_forces); g2l.addWidget(b2)
        ll.addWidget(g2)

        g3=QGroupBox("Stress"); g3l=QVBoxLayout(g3); g3l.setSpacing(4)
        self.lbl_stress=QLabel("—"); self.lbl_stress.setWordWrap(True)
        self.lbl_stress.setStyleSheet("font-size:10px;padding:4px;background:#f1f5f9;border-radius:4px;")
        g3l.addWidget(self.lbl_stress)
        b3=QPushButton("📂 Load Stress (OUTCAR)"); b3.clicked.connect(self._load_stress); g3l.addWidget(b3)
        ll.addWidget(g3)

        g4=QGroupBox("Convergence Info"); g4l=QVBoxLayout(g4); g4l.setSpacing(4)
        self.lbl_conv=QLabel("Load files"); self.lbl_conv.setWordWrap(True)
        self.lbl_conv.setStyleSheet("font-size:10px;padding:5px;background:#f0fdf4;border-radius:4px;border:1px solid #bbf7d0;color:#166534;")
        g4l.addWidget(self.lbl_conv)
        btn_table=QPushButton("Show Convergence Table")
        btn_table.clicked.connect(self._show_conv_table); g4l.addWidget(btn_table)
        ll.addWidget(g4); ll.addStretch()
        ls.setWidget(left); sp.addWidget(ls)

        # Right Panel (Plots)
        right=QWidget(); rl=QVBoxLayout(right); rl.setContentsMargins(2,2,2,2); rl.setSpacing(3)
        scroll_right=QScrollArea(); scroll_right.setWidgetResizable(True); scroll_right.setFrameShape(QFrame.Shape.NoFrame)
        rc=QWidget(); rcl=QVBoxLayout(rc); rcl.setContentsMargins(0,0,0,0); rcl.setSpacing(5)
        
        self.fig_osz=Figure(figsize=(8,2.5),dpi=90); self.fig_osz.patch.set_facecolor("#ffffff")
        self.canvas_osz=FigureCanvas(self.fig_osz); self.canvas_osz.setMinimumHeight(220); rcl.addWidget(self.canvas_osz)
        
        self.fig_force=Figure(figsize=(8,2.5),dpi=90); self.fig_force.patch.set_facecolor("#ffffff")
        self.canvas_force=FigureCanvas(self.fig_force); self.canvas_force.setMinimumHeight(220); rcl.addWidget(self.canvas_force)
        
        self.fig_stress=Figure(figsize=(8,2.5),dpi=90); self.fig_stress.patch.set_facecolor("#ffffff")
        self.canvas_stress=FigureCanvas(self.fig_stress); self.canvas_stress.setMinimumHeight(220); rcl.addWidget(self.canvas_stress)
        
        rcl.addStretch(); scroll_right.setWidget(rc)
        rl.addWidget(scroll_right)
        sp.addWidget(right); sp.setStretchFactor(0,0); sp.setStretchFactor(1,1); ml.addWidget(sp,stretch=1)

    def _auto_load(self):
        d=QFileDialog.getExistingDirectory(self,"Select VASP Run Directory","")
        if not d: return
        p=Path(d)
        osz=p/"OSZICAR"; out=p/"OUTCAR"
        if osz.exists(): self._load_oszicar(str(osz))
        if out.exists(): self._load_forces(str(out)); self._load_stress(str(out))
        if not osz.exists() and not out.exists(): QMessageBox.warning(self,"","No OSZICAR or OUTCAR found.")

    def _load_oszicar(self, path=None):
        p=path or QFileDialog.getOpenFileName(self,"Open OSZICAR","","OSZICAR (OSZICAR*);;All (*)")[0]
        if not p: return
        try:
            steps=[]; energies=[]; de_list=[]; mag_list=[]
            with open(p) as f:
                for line in f:
                    parts=line.split()
                    if not parts or "F=" not in line: continue
                    try:
                        step=int(parts[0]); idx_f=parts.index("F="); e=float(parts[idx_f+1]); steps.append(step); energies.append(e)
                        de=0.0
                        for i,x in enumerate(parts):
                            if x.startswith("d") and "E" in x and i+1<len(parts):
                                try: de=float(parts[i+1]); break
                                except: pass
                        de_list.append(de)
                        mag=0.0
                        if "mag=" in line:
                            try: mag=float(parts[parts.index("mag=")+1])
                            except: pass
                        mag_list.append(mag)
                    except: pass
            if not steps: self.lbl_osz.setText("No ionic steps"); return
            self._osz_data={"steps":np.array(steps),"energies":np.array(energies),"de":np.array(de_list),"mag":np.array(mag_list)}
            self.lbl_osz.setText(f"✓ {Path(p).name}\n{len(steps)} steps, E={energies[-1]:.6f} eV")
            self._replot_osz(); self._update_conv()
        except Exception as e: self.lbl_osz.setText(f"Error: {e}")

    def _replot_osz(self):
        if not self._osz_data: return
        d=self._osz_data; self.fig_osz.clear()
        ax1=self.fig_osz.add_subplot(121); ax1.plot(d["steps"],d["energies"],"o-",color="#2563EB",ms=3,lw=1.2)
        ax1.set_xlabel("Step",fontsize=9); ax1.set_ylabel("E (eV)",fontsize=9); ax1.set_title("Energy History",fontsize=10,fontweight="bold")
        ax1.tick_params(labelsize=8); ax1.grid(True,alpha=0.15)
        ax2=self.fig_osz.add_subplot(122); de=np.abs(d["de"]); de[de<1e-15]=1e-15
        ax2.semilogy(d["steps"],de,"s-",color="#DC2626",ms=3,lw=1.2,label="|ΔE|")
        if d["mag"].any():
            ax2t=ax2.twinx(); ax2t.plot(d["steps"],d["mag"],"^-",color="#16A34A",ms=3,lw=1,alpha=0.7)
            ax2t.set_ylabel("Mag. Moment",fontsize=8,color="#16A34A"); ax2t.tick_params(labelsize=7,colors="#16A34A")
        ax2.set_xlabel("Step",fontsize=9); ax2.set_ylabel("|ΔE| (eV)",fontsize=9); ax2.set_title("Energy Change & Magnetization",fontsize=10,fontweight="bold")
        ax2.tick_params(labelsize=8); ax2.grid(True,alpha=0.15); ax2.legend(fontsize=8)
        self.fig_osz.tight_layout(pad=1.0); self.canvas_osz.draw_idle()

    def _load_forces(self, path=None):
        p=path or QFileDialog.getOpenFileName(self,"Open OUTCAR for forces","","OUTCAR (OUTCAR*);;All (*)")[0]
        if not p: return
        try:
            mx=[]; av=[]; in_b=False; fs=[]
            with open(p) as f:
                for line in f:
                    if "TOTAL-FORCE" in line: in_b=True; fs=[]; next(f); continue
                    if in_b:
                        parts=line.split()
                        if len(parts)==6:
                            try: fs.append(np.sqrt(float(parts[3])**2+float(parts[4])**2+float(parts[5])**2))
                            except: pass
                        elif "---" in line and fs: in_b=False; mx.append(max(fs)); av.append(np.mean(fs))
            if not mx: self.lbl_force.setText("No forces found"); return
            self._force_data={"max":np.array(mx),"avg":np.array(av),"steps":np.arange(1,len(mx)+1)}
            self.lbl_force.setText(f"✓ {Path(p).name}\n{len(mx)} steps, max|F|={mx[-1]:.4f}")
            self._replot_forces(); self._update_conv()
        except Exception as e: self.lbl_force.setText(f"Error: {e}")

    def _replot_forces(self):
        if not self._force_data: return
        d=self._force_data; self.fig_force.clear(); ax=self.fig_force.add_subplot(111)
        ax.semilogy(d["steps"],d["max"],"o-",color="#EA580C",ms=3,lw=1.2,label="Max |F|")
        ax.semilogy(d["steps"],d["avg"],"s-",color="#7C3AED",ms=3,lw=1.0,alpha=0.8,label="Avg |F|")
        ax.axhline(0.01,color="#94A3B8",ls="--",lw=1,alpha=0.6,label="0.01 threshold")
        ax.set_xlabel("Step",fontsize=9); ax.set_ylabel("Force (eV/Å)",fontsize=9); ax.set_title("Force Convergence",fontsize=10,fontweight="bold")
        ax.tick_params(labelsize=8); ax.grid(True,alpha=0.15); ax.legend(fontsize=8)
        self.fig_force.tight_layout(pad=1.0); self.canvas_force.draw_idle()

    def _load_stress(self, path=None):
        p=path or QFileDialog.getOpenFileName(self,"Open OUTCAR for stress","","OUTCAR (OUTCAR*);;All (*)")[0]
        if not p: return
        try:
            stresses=[]
            with open(p) as f:
                for line in f:
                    if "in kB" in line:
                        parts=line.split()
                        try: stresses.append([float(x) for x in parts[-6:]])
                        except: pass
            if not stresses: self.lbl_stress.setText("No stress data"); return
            self._stress_data={"stresses":np.array(stresses),"steps":np.arange(1,len(stresses)+1)}
            final=stresses[-1]; labels=["XX","YY","ZZ","XY","YZ","ZX"]
            txt=[f"✓ {Path(p).name} — {len(stresses)} steps"]
            for l,v in zip(labels[:3],final[:3]): txt.append(f"  {l}: {v:+.1f} kB")
            txt.append(f"  P = {np.mean(final[:3]):.1f} kB")
            self.lbl_stress.setText("\n".join(txt)); self._replot_stress(); self._update_conv()
        except Exception as e: self.lbl_stress.setText(f"Error: {e}")

    def _replot_stress(self):
        if not self._stress_data: return
        d=self._stress_data; self.fig_stress.clear(); ax=self.fig_stress.add_subplot(111)
        s=d["stresses"]; steps=d["steps"]
        labels=["XX","YY","ZZ","XY","YZ","ZX"]; colors=["#EF4444","#3B82F6","#10B981","#F59E0B","#8B5CF6","#64748B"]
        for i in range(6):
            if i<3: ax.plot(steps,s[:,i],"-",color=colors[i],lw=1.5,label=labels[i])
            else: ax.plot(steps,s[:,i],"--",color=colors[i],lw=1,alpha=0.6,label=labels[i])
        ax.set_xlabel("Step",fontsize=9); ax.set_ylabel("Stress (kB)",fontsize=9); ax.set_title("Stress Tensor Components",fontsize=10,fontweight="bold")
        ax.tick_params(labelsize=8); ax.grid(True,alpha=0.15); ax.legend(fontsize=8,loc='center left',bbox_to_anchor=(1,0.5))
        self.fig_stress.tight_layout(pad=1.0); self.canvas_stress.draw_idle()

    def _update_conv(self):
        lines=[]
        if self._osz_data:
            de=abs(self._osz_data["de"][-1]) if len(self._osz_data["de"]) else 0
            lines.append(f"Energy: {'✅' if de<1e-4 else '⚠'}  |ΔE| = {de:.2e} eV")
        if self._force_data:
            mf=self._force_data["max"][-1]
            lines.append(f"Forces: {'✅' if mf<0.01 else '⚠'}  max|F| = {mf:.4f} eV/Å")
        if self._stress_data:
            ms=np.max(np.abs(self._stress_data["stresses"][-1][:3]))
            lines.append(f"Stress: {'✅' if ms<1.0 else '⚠'}  max|S| = {ms:.1f} kB")
        if not lines: lines.append("Load OSZICAR/OUTCAR to view convergence.")
        self.lbl_conv.setText("\n".join(lines))

    def _show_conv_table(self):
        if not self._osz_data and not self._force_data: QMessageBox.warning(self,"","No data loaded."); return
        dlg=QDialog(self); dlg.setWindowTitle("Convergence Data"); dlg.resize(600,400)
        lay=QVBoxLayout(dlg)
        tw=QTableWidget(); tw.setColumnCount(5); tw.setHorizontalHeaderLabels(["Step","Energy (eV)","ΔE (eV)","Max |F| (eV/Å)","Avg |F| (eV/Å)"])
        tw.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        steps=self._osz_data["steps"] if self._osz_data else self._force_data["steps"]
        tw.setRowCount(len(steps))
        for i,s in enumerate(steps):
            tw.setItem(i,0,QTableWidgetItem(str(s)))
            if self._osz_data and i<len(self._osz_data["energies"]):
                tw.setItem(i,1,QTableWidgetItem(f"{self._osz_data['energies'][i]:.6f}"))
                tw.setItem(i,2,QTableWidgetItem(f"{self._osz_data['de'][i]:.2e}"))
            if self._force_data and i<len(self._force_data["max"]):
                tw.setItem(i,3,QTableWidgetItem(f"{self._force_data['max'][i]:.4f}"))
                tw.setItem(i,4,QTableWidgetItem(f"{self._force_data['avg'][i]:.4f}"))
        lay.addWidget(tw)
        btn=QPushButton("Copy to Clipboard"); btn.clicked.connect(lambda: QApplication.clipboard().setText("\n".join(["\t".join([tw.item(r,c).text() if tw.item(r,c) else "" for c in range(5)]) for r in range(tw.rowCount())])))
        lay.addWidget(btn); dlg.exec()


# ══════════════════════════════════════════════════════════════════════════════
#  K-PATH SEEKER  (SeeK-path / Hinuma et al. 2017)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_bz_edges(rec_lat):
    """Compute Brillouin zone (Wigner-Seitz cell) edges from reciprocal lattice."""
    from scipy.spatial import Voronoi
    pts = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                pts.append(i * rec_lat[0] + j * rec_lat[1] + k * rec_lat[2])
    pts = np.array(pts)
    vor = Voronoi(pts)
    origin_idx = 13
    edges = set()
    for ridge_pts, ridge_verts in zip(vor.ridge_points, vor.ridge_vertices):
        if origin_idx in ridge_pts and -1 not in ridge_verts:
            verts = ridge_verts
            poly = np.array([vor.vertices[v] for v in verts])
            centroid = poly.mean(axis=0)
            normal = np.cross(poly[1] - poly[0], poly[2] - poly[0])
            nlen = np.linalg.norm(normal)
            if nlen < 1e-12:
                continue
            normal /= nlen
            u = poly[0] - centroid
            u -= np.dot(u, normal) * normal
            nu = np.linalg.norm(u)
            if nu < 1e-12:
                continue
            u /= nu
            v_dir = np.cross(normal, u)
            angles = [np.arctan2(np.dot(p - centroid, v_dir), np.dot(p - centroid, u)) for p in poly]
            order = np.argsort(angles)
            sv = [verts[o] for o in order]
            for ii in range(len(sv)):
                edges.add(tuple(sorted([sv[ii], sv[(ii + 1) % len(sv)]])))
    return [(vor.vertices[a], vor.vertices[b]) for a, b in edges]


# ── pyqtgraph OpenGL 3D widget ────────────────────────────────────────────────
_HAS_PYQTGRAPH = False
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    _HAS_PYQTGRAPH = True
except ImportError:
    pass


def _hex_rgb_float(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, 1.0)


def _make_tube_mesh(p1, p2, radius=0.025, color=(0.2, 0.5, 0.9, 1.0), n_sides=8):
    """Return a GLMeshItem cylinder between p1 and p2 — used for thick k-path rendering.

    This bypasses the OpenGL line-width limitation (glLineWidth > 1 is ignored on
    most modern drivers/core-profile contexts).  Returns None if pyqtgraph unavailable.
    """
    if not _HAS_PYQTGRAPH:
        return None
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    axis = p2 - p1
    length = np.linalg.norm(axis)
    if length < 1e-12:
        return None
    axis_n = axis / length
    # Two perpendicular unit vectors spanning the tube cross-section
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis_n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis_n, ref);  u /= np.linalg.norm(u)
    v = np.cross(axis_n, u)
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    verts = []
    for a in angles:
        rv = radius * (np.cos(a) * u + np.sin(a) * v)
        verts.append(p1 + rv)   # bottom ring
        verts.append(p2 + rv)   # top ring
    verts = np.array(verts, dtype=np.float32)
    faces = []
    for i in range(n_sides):
        i0, i1 = 2 * i, 2 * i + 1
        i2, i3 = 2 * ((i + 1) % n_sides), 2 * ((i + 1) % n_sides) + 1
        faces += [[i0, i1, i3], [i0, i3, i2]]
    faces = np.array(faces, dtype=np.uint32)
    face_colors = np.tile(np.array(color, dtype=np.float32), (len(faces), 1))
    md = gl.MeshData(vertexes=verts, faces=faces, faceColors=face_colors)
    return gl.GLMeshItem(meshdata=md, smooth=True, shader='shaded', glOptions='opaque')


class _BZGLWidget(gl.GLViewWidget if _HAS_PYQTGRAPH else QWidget):
    """OpenGL 3D Brillouin zone viewer powered by pyqtgraph."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dark_bg = False
        if _HAS_PYQTGRAPH:
            self.setBackgroundColor(pg.mkColor("#ffffff"))  # white by default
            self.setCameraPosition(distance=4)
            self.opts['fov'] = 45

    def set_dark(self, dark: bool):
        """Toggle background between dark (#0f172a) and light (#ffffff)."""
        self._dark_bg = dark
        if _HAS_PYQTGRAPH:
            self.setBackgroundColor(pg.mkColor("#0f172a" if dark else "#ffffff"))

    def show_bz(self, rec_lat, point_coords, path, cart_pts, fit_camera=True,
                path_width=2.5, show_labels=True, path_colors_rgba=None,
                show_bz_cell=True, show_vecs=True, bz_alpha=0.55):
        """Render BZ wireframe, k-path, and high-symmetry points."""
        if not _HAS_PYQTGRAPH:
            return
        # Clear previous items
        for item in list(self.items):
            self.removeItem(item)

        # ── Scene scale (for tube radius) ──
        if cart_pts:
            _all = np.array(list(cart_pts.values()))
            _scene_scale = max(float(np.abs(_all).max()), 0.5)
        else:
            _scene_scale = 1.0

        # ── BZ wireframe ──
        if show_bz_cell:
            try:
                _wire_col = (0.25, 0.45, 0.70, float(bz_alpha))
                for p1, p2 in _compute_bz_edges(rec_lat):
                    self.addItem(gl.GLLinePlotItem(
                        pos=np.array([p1, p2], dtype=np.float32),
                        color=_wire_col, width=1.0, antialias=True))
            except Exception:
                pass

        # ── Reciprocal lattice vectors — rendered as tubes + arrowhead cones ──
        # glLineWidth is ignored on modern OpenGL, so we use GLMeshItem tubes here too.
        if show_vecs:
            bvec_colors = [
                (0.85, 0.15, 0.15, 1.0),   # b1 — red
                (0.05, 0.62, 0.18, 1.0),   # b2 — green
                (0.10, 0.35, 0.88, 1.0),   # b3 — blue
            ]
            bvec_labels = ["b\u2081", "b\u2082", "b\u2083"]
            vec_tube_r = _scene_scale * 0.004   # thinner than path tubes
            for i in range(3):
                v = np.array(rec_lat[i], dtype=float)
                col = bvec_colors[i]
                # ── shaft (tube from origin → 85 % of vector) ──
                shaft_end = v * 0.85
                shaft = _make_tube_mesh(
                    np.zeros(3), shaft_end,
                    radius=vec_tube_r, color=col, n_sides=8)
                if shaft is not None:
                    self.addItem(shaft)
                # ── arrowhead cone (small sphere at tip for simplicity) ──
                tip_mesh = _make_tube_mesh(
                    shaft_end, v,
                    radius=vec_tube_r * 2.2, color=col, n_sides=10)
                if tip_mesh is not None:
                    self.addItem(tip_mesh)
                # ── label at 115 % of tip ──
                _lbl_col_v = (0.95, 0.95, 0.95, 1.0) if self._dark_bg else col
                try:
                    lbl_pos = (v * 1.18).astype(np.float32)
                    txt = gl.GLTextItem(
                        pos=lbl_pos,
                        text=bvec_labels[i],
                        color=pg.mkColor(*[int(c * 255) for c in _lbl_col_v]),
                    )
                    self.addItem(txt)
                except Exception:
                    pass

        # ── K-path segments — rendered as 3D cylinder tubes ──────────────────
        # glLineWidth > 1 is ignored by modern OpenGL core-profile drivers
        # (Windows, macOS metal), so we use GLMeshItem cylinders instead.
        _default_colors = [
            (0.15, 0.39, 0.92, 1.0), (0.49, 0.23, 0.93, 1.0), (0.92, 0.35, 0.05, 1.0),
            (0.03, 0.57, 0.74, 1.0), (0.86, 0.15, 0.15, 1.0), (0.09, 0.64, 0.26, 1.0),
            (0.85, 0.47, 0.02, 1.0), (0.39, 0.40, 0.95, 1.0), (0.93, 0.28, 0.60, 1.0),
        ]
        colors_to_use = path_colors_rgba if path_colors_rgba else _default_colors
        # path_width in the spinner is 0.5 – 10; map to tube radius as fraction of scene
        tube_radius = _scene_scale * 0.008 * float(path_width)
        for idx, (start, end) in enumerate(path):
            c1 = cart_pts.get(start, np.zeros(3))
            c2 = cart_pts.get(end,   np.zeros(3))
            col = colors_to_use[idx % len(colors_to_use)]
            mesh = _make_tube_mesh(c1, c2, radius=tube_radius, color=col)
            if mesh is not None:
                self.addItem(mesh)

        # ── High-symmetry point spheres ──
        if cart_pts:
            pos_arr = np.array(list(cart_pts.values()), dtype=np.float32)
            self.addItem(gl.GLScatterPlotItem(
                pos=pos_arr, size=12, color=(1.0, 0.70, 0.0, 1.0), pxMode=True))

        # ── Origin marker ──
        _orig_col = (0.9, 0.9, 0.9, 0.8) if self._dark_bg else (0.2, 0.2, 0.2, 0.7)
        self.addItem(gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]], dtype=np.float32),
            size=7, color=_orig_col, pxMode=True))

        # ── High-symmetry point labels via GLTextItem ─────────────────────────
        if show_labels and cart_pts:
            _lbl_col = (0.95, 0.95, 0.95, 1.0) if self._dark_bg else (0.05, 0.05, 0.15, 1.0)
            lbl_offset = _scene_scale * 0.08
            try:
                for name, pos in cart_pts.items():
                    disp = (
                        "\u0393" if name == "GAMMA"
                        else name.replace("_0", "\u2080")
                                  .replace("_1", "\u2081")
                                  .replace("_2", "\u2082")
                    )
                    lbl_pos = np.array(pos, dtype=float) + lbl_offset
                    txt = gl.GLTextItem(
                        pos=lbl_pos.astype(np.float32),
                        text=disp,
                        color=pg.mkColor(*[int(c * 255) for c in _lbl_col]),
                    )
                    self.addItem(txt)
            except Exception:
                pass   # GLTextItem not available in this pyqtgraph build

        # Auto-center camera only on initial load/reset (not on zoom redraws)
        if fit_camera and cart_pts:
            pos_arr = np.array(list(cart_pts.values()), dtype=np.float32)
            maxr = float(np.abs(pos_arr).max()) * 2.5
            self.setCameraPosition(distance=max(maxr, 2.0))


class KPathSeekerWidget(QWidget):
    """Interactive K-path seeker using SeeK-path (Hinuma et al. 2017)."""

    def __init__(self):
        super().__init__()
        self._poscar_data     = None
        self._seekpath_result = None
        self._use_gl     = _HAS_PYQTGRAPH
        self._zoom_level = 1.0   # matplotlib: ax.dist multiplier (<1 = zoom in)
        self._gl_zoom    = 1.0   # GL: geometry scale factor (>1 = zoom in)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 4, 6, 6); lay.setSpacing(4)

        # Header
        header = QLabel("<b style='font-size:13px;color:#1e293b'>K-Path Seeker</b>  "
                        "<span style='font-size:10px;color:#64748b'>Based on SeeK-path (Hinuma et al. 2017)</span>")
        lay.addWidget(header)

        sp = QSplitter(Qt.Orientation.Horizontal)

        # ── Left sidebar ──
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setMinimumWidth(250); left_scroll.setMaximumWidth(310)
        left = QWidget(); left.setStyleSheet(SIDEBAR_STYLE)
        ll = QVBoxLayout(left); ll.setContentsMargins(4, 4, 4, 4); ll.setSpacing(4)

        g0 = QGroupBox("POSCAR / CONTCAR"); g0l = QVBoxLayout(g0); g0l.setSpacing(3)
        btn_load = QPushButton("Load POSCAR"); btn_load.setObjectName("primary")
        btn_load.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        btn_load.clicked.connect(self._load_poscar); g0l.addWidget(btn_load)
        self.lbl_file = QLabel("No file loaded"); self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("font-size:10px;color:#64748B;padding:3px;background:#f1f5f9;border-radius:4px;")
        g0l.addWidget(self.lbl_file); ll.addWidget(g0)

        g1 = QGroupBox("Symmetry Info"); g1l = QVBoxLayout(g1); g1l.setSpacing(3)
        self.lbl_sym = QLabel("—"); self.lbl_sym.setWordWrap(True)
        self.lbl_sym.setStyleSheet("font-size:11px;padding:6px;background:#f0f9ff;border-radius:6px;border:1px solid #bae6fd;color:#0c4a6e;")
        g1l.addWidget(self.lbl_sym); ll.addWidget(g1)

        g2 = QGroupBox("Settings"); g2l = QGridLayout(g2); g2l.setSpacing(4)
        g2l.addWidget(QLabel("Symm. precision:"), 0, 0)
        self.sp_thresh = QDoubleSpinBox(); self.sp_thresh.setRange(1e-6, 0.1)
        self.sp_thresh.setDecimals(6); self.sp_thresh.setValue(1e-4); self.sp_thresh.setSingleStep(1e-5)
        self.sp_thresh.setToolTip("Symmetry detection tolerance. 1e-4 is recommended for DFT structures.")
        g2l.addWidget(self.sp_thresh, 0, 1)
        self.chk_tr = QCheckBox("Time reversal"); self.chk_tr.setChecked(True)
        g2l.addWidget(self.chk_tr, 1, 0, 1, 2)
        g2l.addWidget(QLabel("Divisions:"), 2, 0)
        self.sp_div = QSpinBox(); self.sp_div.setRange(10, 500); self.sp_div.setValue(40)
        g2l.addWidget(self.sp_div, 2, 1)
        self.chk_2d_auto = QCheckBox("Auto-detect 2D slab (spglib)")
        self.chk_2d_auto.setChecked(True)
        self.chk_2d_auto.setToolTip(
            "Detects monolayer/slab structures by checking vacuum gap (c/a > 3.5)\n"
            "and overrides the bulk k-path with the correct 2D BZ path.\n"
            "e.g. Hexagonal 2D (MoS2, graphene): Γ-M-K-Γ\n"
            "     Square 2D: Γ-X-M-Γ\n"
            "Reference: Hinuma et al., Comput. Mater. Sci. 128 (2017) 140."
        )
        g2l.addWidget(self.chk_2d_auto, 3, 0, 1, 2)
        ll.addWidget(g2)

        g_disp = QGroupBox("Display Options"); g_disp_l = QGridLayout(g_disp); g_disp_l.setSpacing(3)
        self.chk_show_bz   = QCheckBox("Show BZ cell");         self.chk_show_bz.setChecked(True)
        self.chk_show_vecs = QCheckBox("Show rec. vectors");     self.chk_show_vecs.setChecked(True)
        self.chk_show_lbls = QCheckBox("Show point labels");     self.chk_show_lbls.setChecked(True)
        g_disp_l.addWidget(self.chk_show_bz,   0, 0, 1, 2)
        g_disp_l.addWidget(self.chk_show_vecs, 1, 0, 1, 2)
        g_disp_l.addWidget(self.chk_show_lbls, 2, 0, 1, 2)
        g_disp_l.addWidget(QLabel("Path width:"), 3, 0)
        self.sp_pw = QDoubleSpinBox(); self.sp_pw.setRange(0.5, 10.0); self.sp_pw.setValue(2.5); self.sp_pw.setSingleStep(0.5)
        self.sp_pw.setToolTip("Tube radius of the k-path segments. Larger = thicker tubes.")
        g_disp_l.addWidget(self.sp_pw, 3, 1)
        g_disp_l.addWidget(QLabel("BZ opacity:"), 4, 0)
        self.sp_bz_alpha = QDoubleSpinBox(); self.sp_bz_alpha.setRange(0.05, 1.0); self.sp_bz_alpha.setValue(0.55); self.sp_bz_alpha.setSingleStep(0.05)
        g_disp_l.addWidget(self.sp_bz_alpha, 4, 1)
        g_disp_l.addWidget(QLabel("Color scheme:"), 5, 0)
        self.cmb_clr = QComboBox()
        self.cmb_clr.addItems(["Vivid", "Pastel", "Warm", "Mono"])
        g_disp_l.addWidget(self.cmb_clr, 5, 1)
        btn_redraw = QPushButton("↺  Redraw"); btn_redraw.clicked.connect(self._redraw)
        g_disp_l.addWidget(btn_redraw, 6, 0, 1, 2)
        ll.addWidget(g_disp)

        g3 = QGroupBox("Actions"); g3l = QVBoxLayout(g3); g3l.setSpacing(5)
        btn_seek = QPushButton("Seek K-Path"); btn_seek.setObjectName("primary")
        btn_seek.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward))
        btn_seek.clicked.connect(self._seek_path); g3l.addWidget(btn_seek)
        btn_gen = QPushButton("Generate KPOINTS")
        btn_gen.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        btn_gen.clicked.connect(self._generate_kpoints)
        g3l.addWidget(btn_gen)
        btn_copy = QPushButton("Copy KPOINTS")
        btn_copy.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        btn_copy.clicked.connect(self._copy_kpoints)
        g3l.addWidget(btn_copy)
        btn_save = QPushButton("Save KPOINTS")
        btn_save.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        btn_save.clicked.connect(self._save_kpoints)
        g3l.addWidget(btn_save)
        ll.addWidget(g3)

        g4 = QGroupBox("K-Path"); g4l = QVBoxLayout(g4); g4l.setSpacing(3)
        self.lbl_path = QLabel("—"); self.lbl_path.setWordWrap(True)
        self.lbl_path.setStyleSheet("font-size:11px;padding:4px;background:#f0fdf4;border-radius:4px;border:1px solid #bbf7d0;color:#166534;")
        g4l.addWidget(self.lbl_path); ll.addWidget(g4)

        g5 = QGroupBox("High-Symmetry Points"); g5l = QVBoxLayout(g5)
        self.tbl_pts = QTableWidget(0, 4)
        self.tbl_pts.setHorizontalHeaderLabels(["Label", "k₁", "k₂", "k₃"])
        self.tbl_pts.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_pts.setStyleSheet("background:#fff;color:#1e293b;font-size:11px;")
        self.tbl_pts.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_pts.setMaximumHeight(180)
        g5l.addWidget(self.tbl_pts); ll.addWidget(g5)

        ll.addStretch()
        left_scroll.setWidget(left); sp.addWidget(left_scroll)

        # ── Right: 3D viewport + KPOINTS output ──
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(2, 2, 2, 2); rl.setSpacing(4)

        # ── Zoom toolbar + dark/light toggle ──
        zoom_bar = QHBoxLayout(); zoom_bar.setSpacing(3)
        _ZB = "QPushButton{background:#f1f5f9;color:#1e293b;border:1px solid #e2e8f0;" \
              "border-radius:5px;padding:3px 10px;font-size:13px;font-weight:600;min-width:30px;}" \
              "QPushButton:hover{background:#e0e7ff;border-color:#a5b4fc;}"
        btn_zi = QPushButton("＋"); btn_zi.setToolTip("Zoom In  (scroll wheel also works)")
        btn_zi.setStyleSheet(_ZB); btn_zi.clicked.connect(self._zoom_in)
        btn_zo = QPushButton("－"); btn_zo.setToolTip("Zoom Out")
        btn_zo.setStyleSheet(_ZB); btn_zo.clicked.connect(self._zoom_out)
        btn_zr = QPushButton("⌂"); btn_zr.setToolTip("Reset zoom / fit")
        btn_zr.setStyleSheet(_ZB); btn_zr.clicked.connect(self._zoom_reset)
        zoom_lbl = QLabel("BZ View:"); zoom_lbl.setStyleSheet("font-size:11px;color:#64748b;")
        # ── Dark / Light toggle ──
        self._dark_mode = False
        self.btn_theme = QPushButton("🌙 Dark")
        self.btn_theme.setToolTip("Toggle dark / light background")
        self.btn_theme.setStyleSheet(
            "QPushButton{background:#1e293b;color:#e2e8f0;border:1px solid #334155;"
            "border-radius:5px;padding:3px 12px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#0f172a;}"
        )
        self.btn_theme.clicked.connect(self._toggle_bg)
        zoom_bar.addWidget(zoom_lbl)
        zoom_bar.addWidget(btn_zi); zoom_bar.addWidget(btn_zo); zoom_bar.addWidget(btn_zr)
        zoom_bar.addSpacing(12)
        zoom_bar.addWidget(self.btn_theme)
        zoom_bar.addStretch()
        rl.addLayout(zoom_bar)

        # 3D viewport — pyqtgraph GL or matplotlib fallback
        if self._use_gl:
            # Container so we can stack the GL view and an HTML label overlay
            gl_container = QWidget()
            gl_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            gl_stack = QVBoxLayout(gl_container)
            gl_stack.setContentsMargins(0, 0, 0, 0)
            self._gl_widget = _BZGLWidget()
            self._gl_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            gl_stack.addWidget(self._gl_widget)
            # Floating label overlay (absolute positioned inside the GL container)
            self._label_overlay = QLabel(gl_container)
            self._label_overlay.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            self._label_overlay.setStyleSheet(
                "background:transparent;color:#1e293b;font-size:12px;font-weight:700;padding:4px;"
            )
            self._label_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self._label_overlay.hide()
            rl.addWidget(gl_container, stretch=3)
        else:
            self.fig = Figure(figsize=(8, 7), dpi=80); self.fig.patch.set_facecolor("#ffffff")
            self.canvas = FigureCanvas(self.fig)
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            rl.addWidget(self.canvas, stretch=3)

        # KPOINTS output
        g6 = QGroupBox("Generated KPOINTS"); g6l = QVBoxLayout(g6)
        self.te_kpoints = QTextEdit(); self.te_kpoints.setReadOnly(True)
        self.te_kpoints.setFont(QFont("Courier New", 10)); self.te_kpoints.setMaximumHeight(160)
        self.te_kpoints.setStyleSheet("background:#fff;color:#1e293b;border:1px solid #e2e8f0;border-radius:4px;")
        self.te_kpoints.setPlaceholderText("Load a POSCAR and click 'Seek K-Path' then 'Generate KPOINTS'...")
        g6l.addWidget(self.te_kpoints); rl.addWidget(g6, stretch=1)

        sp.addWidget(right); sp.setStretchFactor(0, 0); sp.setStretchFactor(1, 1)
        lay.addWidget(sp, stretch=1)

        # Footer
        footer = QLabel("SeeK-path: Hinuma et al., Comput. Mater. Sci. 128 (2017) 140.  "
                        "|  2D slab auto-detect: spglib (Togo & Tanaka 2018)  "
                        "|  pip install seekpath spglib")
        footer.setStyleSheet("font-size:9px;color:#94a3b8;padding:2px 4px;")
        lay.addWidget(footer)

        if not self._use_gl:
            self._placeholder_mpl()

    # ── matplotlib fallback placeholder ──
    def _placeholder_mpl(self):
        self.fig.clear(); ax = self.fig.add_subplot(111); ax.set_axis_off()
        ax.set_facecolor("#ffffff")
        ax.text(0.5, 0.62, "Brillouin Zone Viewer", transform=ax.transAxes,
                ha="center", va="center", fontsize=22, color="#1e293b", fontweight="bold")
        ax.text(0.5, 0.52, "Based on SeeK-path (Hinuma et al. 2017)", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#64748b")
        ax.plot([0.1, 0.9], [0.46, 0.46], color="#e2e8f0", lw=1, transform=ax.transAxes, clip_on=False)
        ax.text(0.5, 0.38, "1.  Load a POSCAR / CONTCAR file (left panel)", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#475569")
        ax.text(0.5, 0.30, "2.  Click  'Seek K-Path'  to detect the Bravais lattice & k-path", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#475569")
        ax.text(0.5, 0.22, "3.  Adjust divisions and click 'Generate KPOINTS'", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#475569")
        ax.text(0.5, 0.10,
                "Requires:  seekpath   spglib          (pip install seekpath spglib)",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#94a3b8",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f1f5f9", ec="#e2e8f0", lw=1))
        self.canvas.draw_idle()

    def _load_poscar(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open POSCAR / CONTCAR", "",
                                           "VASP (POSCAR CONTCAR *.vasp);;All (*)")
        if not p:
            return
        try:
            data = PoscarParser(p).parse()
            self._poscar_data = data
            self.lbl_file.setText(
                f"[OK] {Path(p).name}\n"
                f"Atoms: {data['total_atoms']}  |  Species: {', '.join(data['species'])}\n"
                f"a={data['a']:.3f}  b={data['b']:.3f}  c={data['c']:.3f} A\n"
                f"alpha={data['alpha']:.1f}  beta={data['beta']:.1f}  gamma={data['gamma']:.1f}"
            )
            self.lbl_file.setStyleSheet("font-size:10px;color:#16A34A;padding:3px;background:#f0fdf4;border-radius:4px;border:1px solid #bbf7d0;")
            self._seekpath_result = None
            self.lbl_sym.setText("Click 'Seek K-Path' to analyze symmetry")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse POSCAR:\n{e}")

    # ── 2D material detection & k-path override ──────────────────────────────

    # 2D k-paths keyed by (Bravais lattice type, dimensionality hint)
    # Coordinates are in fractional reciprocal-lattice units of the primitive cell
    # Reference: Hinuma et al., Comput. Mater. Sci. 128 (2017) 140
    #            VASPKIT manual (Wang et al. 2021)
    _2D_PATHS = {
        # Hexagonal 2D: graphene, MoS2, h-BN, etc.  Space groups 156-194 (hP)
        "hexagonal_2d": {
            "path": [("GAMMA", "M"), ("M", "K"), ("K", "GAMMA")],
            "point_coords": {
                "GAMMA": [0.0, 0.0, 0.0],
                "M":     [0.5, 0.0, 0.0],
                "K":     [1/3, 1/3, 0.0],
            },
        },
        # Square/rectangular 2D: SnS2-like, CuO2 planes, etc. (tP / oP in-plane)
        "square_2d": {
            "path": [("GAMMA", "X"), ("X", "M"), ("M", "GAMMA")],
            "point_coords": {
                "GAMMA": [0.0, 0.0, 0.0],
                "X":     [0.5, 0.0, 0.0],
                "M":     [0.5, 0.5, 0.0],
            },
        },
        # Rectangular 2D (orthorhombic in-plane)
        "rectangular_2d": {
            "path": [("GAMMA", "X"), ("X", "S"), ("S", "Y"), ("Y", "GAMMA")],
            "point_coords": {
                "GAMMA": [0.0, 0.0, 0.0],
                "X":     [0.5, 0.0, 0.0],
                "Y":     [0.0, 0.5, 0.0],
                "S":     [0.5, 0.5, 0.0],
            },
        },
    }

    def _is_2d_slab(self, lattice_mat):
        """
        Detect whether the structure is a 2D slab / monolayer by checking
        whether one lattice parameter is much larger than the in-plane ones
        (vacuum gap indicator).  Returns (bool, vacuum_axis_idx, ratio).

        VASPKIT convention: if max_c / mean(a,b) > 3.5 it is treated as 2D.
        We also check that the out-of-plane direction has no atomic density
        (done implicitly because POSCAR usually has c as the vacuum axis).
        """
        import numpy as np
        a = float(np.linalg.norm(lattice_mat[0]))
        b = float(np.linalg.norm(lattice_mat[1]))
        c = float(np.linalg.norm(lattice_mat[2]))
        lengths = [a, b, c]
        max_len = max(lengths)
        idx_max = lengths.index(max_len)
        in_plane = [l for i, l in enumerate(lengths) if i != idx_max]
        mean_ip = sum(in_plane) / 2
        ratio = max_len / max(mean_ip, 1e-10)
        return ratio > 3.5, idx_max, ratio

    def _get_2d_bravais_type(self, sg_number, lattice_mat):
        """
        Given a 3D space group number and lattice, return the 2D Bravais type
        (hexagonal_2d, square_2d, or rectangular_2d) for path selection.
        Follows VASPKIT's approach.
        """
        import numpy as np
        a = float(np.linalg.norm(lattice_mat[0]))
        b = float(np.linalg.norm(lattice_mat[1]))
        # Hexagonal: SG 143-194 (trigonal + hexagonal)
        if 143 <= sg_number <= 194:
            return "hexagonal_2d"
        # Tetragonal: SG 75-142
        if 75 <= sg_number <= 142:
            return "square_2d"
        # Cubic: SG 195-230 (can become square 2D when slab)
        if 195 <= sg_number <= 230:
            return "square_2d"
        # Orthorhombic with a ≈ b → square; else rectangular
        if abs(a - b) / max(a, b) < 0.05:
            return "square_2d"
        return "rectangular_2d"

    def _build_2d_result_override(self, original_result, kind, sg_number, sg_sym, pg):
        """
        Build a fake seekpath-style result dict with the 2D k-path injected.
        Preserves reciprocal lattice from the original result so BZ drawing works.
        """
        tpl = self._2D_PATHS[kind]
        # Create an overridden result preserving the BZ geometry
        result_2d = dict(original_result)
        result_2d["path"] = tpl["path"]
        result_2d["point_coords"] = tpl["point_coords"]
        # Tag so we can show a banner
        result_2d["_vaspviz_2d"] = True
        result_2d["_vaspviz_2d_kind"] = kind
        result_2d["spacegroup_number"] = sg_number
        result_2d["spacegroup_international"] = sg_sym
        result_2d["pointgroup_international"] = pg
        return result_2d

    def _seek_path(self):
        if not self._poscar_data:
            QMessageBox.warning(self, "", "Load a POSCAR file first."); return
        try:
            import seekpath
        except ImportError:
            QMessageBox.critical(self, "Missing Package",
                                 "seekpath is not installed.\nRun: pip install seekpath spglib"); return
        data = self._poscar_data
        lattice = data["lattice"].tolist()
        n_atoms = data["total_atoms"]
        positions = data["frac_positions"][:n_atoms].tolist()
        numbers = []
        for lbl in data["ion_labels"][:n_atoms]:
            import re as _re
            clean = _re.sub(r'\d+', '', lbl).strip()
            n = ELEMENT_NUMBERS.get(lbl.strip(), 0) or ELEMENT_NUMBERS.get(clean, 1)
            numbers.append(n)

        symprec = self.sp_thresh.value()

        # ── Step 1: Use spglib for authoritative symmetry analysis ──────────────
        spglib_info = None
        try:
            import spglib as _spg
            cell = (lattice, positions, numbers)
            dataset = _spg.get_symmetry_dataset(cell, symprec=symprec)
            if dataset is not None:
                spglib_info = {
                    "sg_number":  dataset["number"],
                    "sg_sym":     dataset["international"],
                    "pg":         dataset["pointgroup"],
                }
        except Exception:
            pass  # spglib unavailable or failed — fall through to seekpath only

        # ── Step 2: Check 2D slab geometry ─────────────────────────────────────
        import numpy as np
        lattice_np = np.array(lattice)
        is_2d, vacuum_axis, c_ratio = self._is_2d_slab(lattice_np)

        # ── Step 3: Run seekpath for BZ geometry (always needed for 3D plot) ───
        try:
            result = seekpath.get_explicit_k_path(
                (lattice, positions, numbers),
                with_time_reversal=self.chk_tr.isChecked(),
                symprec=symprec
            )
        except Exception:
            try:
                result = seekpath.get_path(
                    (lattice, positions, numbers),
                    with_time_reversal=self.chk_tr.isChecked(),
                    symprec=symprec
                )
            except Exception as e2:
                import traceback; traceback.print_exc()
                QMessageBox.critical(self, "SeeK-path Error",
                    f"Could not determine k-path:\n{e2}\n\n"
                    f"Try increasing the symmetry precision (e.g. 0.001).")
                return

        # Merge spglib SG info into result if available (more reliable)
        if spglib_info:
            result["spacegroup_number"] = spglib_info["sg_number"]
            result["spacegroup_international"] = spglib_info["sg_sym"]
            result["pointgroup_international"] = spglib_info["pg"]

        # ── Step 4: Override k-path for 2D slabs ───────────────────────────────
        if is_2d and self.chk_2d_auto.isChecked():
            sg_num = result.get("spacegroup_number", 1)
            sg_sym = result.get("spacegroup_international", "?")
            pg     = result.get("pointgroup_international", result.get("pointgroup", "?"))
            kind_2d = self._get_2d_bravais_type(sg_num, lattice_np)
            result = self._build_2d_result_override(result, kind_2d, sg_num, sg_sym, pg)
            result["_vaspviz_c_ratio"] = c_ratio

        self._seekpath_result = result
        self._gl_zoom    = 1.0   # reset geometry zoom on new structure
        self._zoom_level = 1.0   # reset matplotlib zoom too
        self._update_sym_info(result)
        self._update_points_table(result)
        self._plot_bz(result, fit_camera=True)

    @staticmethod
    def _label_to_display(name):
        """Convert raw seekpath label (e.g. GAMMA, K_0) to display label (Γ, K₀)."""
        if name == "GAMMA":
            return "\u0393"   # Γ
        # Handle labels like K_0, M_0 etc
        name = name.replace("_0", "\u2080").replace("_1", "\u2081").replace("_2", "\u2082")
        return name

    def _update_sym_info(self, res):
        bravais = res.get("bravais_lattice", "?")
        sg_num = res.get("spacegroup_number", "?")
        sg_sym = res.get("spacegroup_international", "?")
        pg = res.get("pointgroup_international", res.get("pointgroup", "?"))
        path = res.get("path", [])
        is_2d_override = res.get("_vaspviz_2d", False)
        kind_2d = res.get("_vaspviz_2d_kind", "")
        c_ratio = res.get("_vaspviz_c_ratio", None)

        # Build a clean human-readable path string
        # e.g. [(GAMMA,M),(M,K),(K,GAMMA)] -> "Γ-M-K-Γ"
        merged_segments = []
        for start, end in path:
            ds = self._label_to_display(start)
            de = self._label_to_display(end)
            if merged_segments and merged_segments[-1][-1] == ds:
                merged_segments[-1].append(de)
            else:
                merged_segments.append([ds, de])

        path_parts = ["-".join(chain) for chain in merged_segments]
        path_str = "  |  ".join(path_parts) if path_parts else "No path found"

        # Build symmetry info text
        sym_lines = [
            f"Space Group: {sg_sym} (#{sg_num})",
            f"Bravais Lattice: {bravais}",
            f"Point Group: {pg}",
        ]
        if is_2d_override:
            kind_label = {
                "hexagonal_2d":    "Hexagonal 2D  (e.g. MoS₂, graphene)",
                "square_2d":       "Square 2D     (e.g. SnSe₂, h-BN-like)",
                "rectangular_2d": "Rectangular 2D",
            }.get(kind_2d, "2D")
            ratio_txt = f"  (c/a = {c_ratio:.1f}×)" if c_ratio else ""
            sym_lines += [
                "",
                f"⚠ 2D Slab Detected{ratio_txt}",
                f"   Type: {kind_label}",
                "   → Using 2D BZ path (Hinuma 2017 / VASPKIT)",
            ]
            self.lbl_sym.setStyleSheet(
                "font-size:11px;padding:6px;background:#fef9c3;border-radius:6px;"
                "border:1px solid #fde047;color:#713f12;"
            )
        else:
            self.lbl_sym.setStyleSheet(
                "font-size:11px;padding:6px;background:#f0f9ff;border-radius:6px;"
                "border:1px solid #bae6fd;color:#0c4a6e;"
            )

        self.lbl_sym.setText("\n".join(sym_lines))
        self.lbl_path.setText(path_str)

    def _update_points_table(self, res):
        pts = res.get("point_coords", {})
        self.tbl_pts.setRowCount(len(pts))
        for row, (name, coords) in enumerate(sorted(pts.items())):
            disp = self._label_to_display(name)
            self.tbl_pts.setItem(row, 0, QTableWidgetItem(disp))
            for c in range(3):
                self.tbl_pts.setItem(row, 1 + c, QTableWidgetItem(f"{coords[c]:.5f}"))

    def _get_path_colors(self):
        scheme = self.cmb_clr.currentText()
        if scheme == "Pastel":
            return ["#93C5FD","#C4B5FD","#FCA5A5","#6EE7B7","#FCD34D","#F9A8D4","#A5F3FC","#BBF7D0","#FDE68A"]
        elif scheme == "Warm":
            return ["#DC2626","#EA580C","#D97706","#B45309","#92400E","#C2410C","#9A3412","#991B1B","#78350F"]
        elif scheme == "Mono":
            return ["#1e293b","#334155","#475569","#64748b","#94a3b8","#cbd5e1","#334155","#1e293b","#475569"]
        else:  # Vivid
            return ["#2563EB","#7C3AED","#EA580C","#0891B2","#DC2626","#16A34A","#D97706","#6366F1","#EC4899"]

    def _plot_bz(self, res, fit_camera=True):
        """Render Brillouin zone for the given seekpath result.

        Parameters
        ----------
        fit_camera : bool
            If True (default), GL camera auto-fits to the scene.
            Pass False on zoom redraws to keep the current viewpoint.
        """
        rec_lat      = np.array(res["reciprocal_primitive_lattice"])
        point_coords = res["point_coords"]
        path         = res["path"]
        cart_pts     = {name: np.array(frac) @ rec_lat for name, frac in point_coords.items()}

        if self._use_gl:
            # Zoom via geometry scaling — multiply all coordinates by _gl_zoom.
            # This is camera-API-independent and guaranteed to work.
            s = self._gl_zoom
            rec_lat_s  = rec_lat * s
            cart_pts_s = {k: v * s for k, v in cart_pts.items()}
            self._gl_widget.show_bz(
                rec_lat_s, point_coords, path, cart_pts_s,
                fit_camera    = fit_camera,
                path_width    = self.sp_pw.value(),
                show_labels   = self.chk_show_lbls.isChecked(),
                path_colors_rgba = self._get_path_colors_rgba(),
                show_bz_cell  = self.chk_show_bz.isChecked(),
                show_vecs     = self.chk_show_vecs.isChecked(),
                bz_alpha      = self.sp_bz_alpha.value(),
            )
        else:
            self._plot_bz_mpl(rec_lat, point_coords, path, cart_pts)

    def _get_path_colors_rgba(self):
        """Return path colors as (r,g,b,a) float tuples for the GL renderer."""
        def hex2rgba(h):
            h = h.lstrip("#")
            return (int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255, 1.0)
        return [hex2rgba(c) for c in self._get_path_colors()]

    def _update_gl_labels(self, *_):
        """Deprecated — labels are now GLTextItems rendered inside the GL viewport.
        The HTML overlay is kept hidden for backward compatibility.
        """
        if hasattr(self, '_label_overlay') and self._label_overlay is not None:
            self._label_overlay.hide()

    def _plot_bz_mpl(self, rec_lat, point_coords, path, cart_pts):
        """Matplotlib 3D BZ rendering — clean, no axes, no ticks."""
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d')
        ax.set_facecolor("#ffffff"); self.fig.patch.set_facecolor("#ffffff")

        # ── Completely remove all axis decorations ──────────────────────────────
        ax.set_axis_off()   # removes tick marks, tick labels, axis lines & labels
        # Blank out the pane fill/edges and grid (belt-and-suspenders)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor('none')
            axis.line.set_color('none')
        ax.grid(False)

        # ── BZ wireframe ────────────────────────────────────────────────────────
        if self.chk_show_bz.isChecked():
            try:
                alpha = self.sp_bz_alpha.value()
                for p1, p2 in _compute_bz_edges(rec_lat):
                    ax.plot3D(*zip(p1, p2), color="#64748b", lw=1.2, alpha=alpha)
            except Exception:
                pass

        # ── Reciprocal lattice vectors ─────────────────────────────────────────
        if self.chk_show_vecs.isChecked():
            bvc  = ["#DC2626", "#16A34A", "#2563EB"]
            lblv = ["b\u2081", "b\u2082", "b\u2083"]
            for i in range(3):
                v = rec_lat[i]
                ax.quiver(0, 0, 0, v[0], v[1], v[2],
                          color=bvc[i], arrow_length_ratio=0.10, lw=2.0, alpha=0.9)
                ax.text(v[0]*1.14, v[1]*1.14, v[2]*1.14, lblv[i],
                        color=bvc[i], fontsize=10, fontweight="bold")

        # ── K-path segments ────────────────────────────────────────────────────
        pw     = self.sp_pw.value()
        colors = self._get_path_colors()
        for idx, (start, end) in enumerate(path):
            c1 = cart_pts.get(start, np.zeros(3))
            c2 = cart_pts.get(end,   np.zeros(3))
            ax.plot3D([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                      color=colors[idx % len(colors)], lw=pw, alpha=0.9, zorder=5)

        # ── High-symmetry points & labels ──────────────────────────────────────
        for name, cart in cart_pts.items():
            disp = "\u0393" if name == "GAMMA" else name
            ax.scatter(*cart, s=70, c="#f59e0b", zorder=10,
                       edgecolors="#1e293b", linewidths=1.2)
            if self.chk_show_lbls.isChecked():
                ax.text(cart[0], cart[1], cart[2] + 0.02, f"  {disp}",
                        fontsize=9, fontweight="bold", color="#1e293b", zorder=11)

        # ── Axis limits: always encompass ALL data — no clipping ───────────────
        # Zoom is achieved by changing the camera distance (ax.dist), NOT by
        # narrowing the axis limits (which would clip the BZ wireframe edges).
        all_pts = np.array(list(cart_pts.values())) if cart_pts else np.zeros((1, 3))
        base_r  = max(float(np.abs(all_pts).max()) * 1.35, 0.5)
        ax.set_xlim(-base_r, base_r)
        ax.set_ylim(-base_r, base_r)
        ax.set_zlim(-base_r, base_r)

        # ax.dist: default = 10.  smaller → camera closer → zoom in.
        # _zoom_level: 1.0 = default.  <1 = zoomed in.  >1 = zoomed out.
        # For matplotlib >= 3.6, ax.set_box_aspect with zoom kwarg is used instead of ax.dist
        zoom_factor = 1.0 / self._zoom_level
        try:
            if hasattr(ax, "set_box_aspect"):
                try:
                    ax.set_box_aspect(None, zoom=zoom_factor)
                except TypeError:
                    pass
            ax.dist = 10.0 * self._zoom_level
        except Exception:
            pass

        try:
            self.fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        except Exception:
            pass
        self.canvas.draw_idle()

    def _generate_kpoints(self):
        if not self._seekpath_result:
            QMessageBox.warning(self, "", "Run 'Seek K-Path' first."); return
        res = self._seekpath_result
        is_2d = res.get("_vaspviz_2d", False)
        ndiv  = self.sp_div.value()
        path  = res["path"]
        pts   = res["point_coords"]

        segs_lbl = "-".join(
            self._label_to_display(n)
            for n in dict.fromkeys([pt for seg in path for pt in seg])
        )

        # For 2D overrides, always use point_coords directly (the 2D path).
        # For bulk results, prefer explicit_kpoints_rel if available (from
        # get_explicit_k_path) for accuracy, but fall back to point_coords.
        use_explicit = (
            not is_2d
            and "explicit_kpoints_rel" in res
            and "explicit_kpoints_labels" in res
        )

        if is_2d:
            kind_2d = res.get("_vaspviz_2d_kind", "hexagonal_2d")
            c_ratio = res.get("_vaspviz_c_ratio", 0.0)
            sg_sym  = res.get("spacegroup_international", "?")
            sg_num  = res.get("spacegroup_number", "?")
            header  = (
                f"K-POINTS  {segs_lbl}  [2D slab — {kind_2d.replace('_2d','').capitalize()}]"
            )
            comment_lines = [
                f"# 2D BZ path: {segs_lbl}",
                f"# Space group: {sg_sym} (#{sg_num}),  c/a = {c_ratio:.1f}x (slab detected)",
                "# Ref: Hinuma et al., Comput. Mater. Sci. 128 (2017) 140 / VASPKIT convention",
            ]
        else:
            header = f"K-POINTS  {segs_lbl}  (SeeK-path / Hinuma 2017)"
            comment_lines = []

        lines = comment_lines + [header, str(ndiv), "Line-mode", "Reciprocal"]
        for start, end in path:
            cs, ce = pts[start], pts[end]
            ds = "Gamma" if start == "GAMMA" else start
            de = "Gamma" if end   == "GAMMA" else end
            lines.append(f"  {cs[0]:.8f}  {cs[1]:.8f}  {cs[2]:.8f}   ! {ds}")
            lines.append(f"  {ce[0]:.8f}  {ce[1]:.8f}  {ce[2]:.8f}   ! {de}")
            lines.append("")
        self.te_kpoints.setPlainText("\n".join(lines))

    def _copy_kpoints(self):
        t = self.te_kpoints.toPlainText()
        if t: QApplication.clipboard().setText(t)

    def _save_kpoints(self):
        t = self.te_kpoints.toPlainText()
        if not t: QMessageBox.warning(self, "", "Generate KPOINTS first."); return
        p, _ = QFileDialog.getSaveFileName(self, "Save KPOINTS", "KPOINTS", "All (*)")
        if p:
            with open(p, "w") as f: f.write(t)
            QMessageBox.information(self, "Saved", f"KPOINTS saved to:\n{p}")

    def _redraw(self):
        if self._seekpath_result:
            self._plot_bz(self._seekpath_result, fit_camera=False)

    # ── Zoom controls ─────────────────────────────────────────────────────────
    # GL backend: zoom by scaling geometry (camera-API-independent).
    # MPL backend: zoom by adjusting ax.dist (camera-distance multiplier).

    def _zoom_in(self):
        if not self._seekpath_result:
            return
        if self._use_gl:
            self._gl_zoom = min(10.0, self._gl_zoom * 1.33)
            self._plot_bz(self._seekpath_result, fit_camera=False)
        else:
            self._zoom_level = max(0.05, self._zoom_level * 0.75)  # smaller dist → zoom in
            self._plot_bz(self._seekpath_result)

    def _zoom_out(self):
        if not self._seekpath_result:
            return
        if self._use_gl:
            self._gl_zoom = max(0.05, self._gl_zoom / 1.33)
            self._plot_bz(self._seekpath_result, fit_camera=False)
        else:
            self._zoom_level = min(8.0, self._zoom_level * 1.33)  # bigger dist → zoom out
            self._plot_bz(self._seekpath_result)

    def _zoom_reset(self):
        if not self._seekpath_result:
            return
        self._gl_zoom    = 1.0
        self._zoom_level = 1.0
        self._plot_bz(self._seekpath_result, fit_camera=True)

    # ── Background theme toggle ──────────────────────────────────────────

    def _toggle_bg(self):
        """Switch the GL viewer (and matplotlib fallback) between dark and light."""
        self._dark_mode = not self._dark_mode
        dark = self._dark_mode

        if self._use_gl:
            self._gl_widget.set_dark(dark)
            # Update button appearance
            if dark:
                self.btn_theme.setText("☀ Light")
                self.btn_theme.setStyleSheet(
                    "QPushButton{background:#f8fafc;color:#1e293b;border:1px solid #cbd5e1;"
                    "border-radius:5px;padding:3px 12px;font-size:12px;font-weight:600;}"
                    "QPushButton:hover{background:#e2e8f0;}"
                )
            else:
                self.btn_theme.setText("🌙 Dark")
                self.btn_theme.setStyleSheet(
                    "QPushButton{background:#1e293b;color:#e2e8f0;border:1px solid #334155;"
                    "border-radius:5px;padding:3px 12px;font-size:12px;font-weight:600;}"
                    "QPushButton:hover{background:#0f172a;}"
                )
            # Re-render so label colours update for the new background
            if self._seekpath_result:
                self._plot_bz(self._seekpath_result, fit_camera=False)
        else:
            # Matplotlib fallback — toggle figure background
            bg = "#0f172a" if dark else "#ffffff"
            txt_col = "#e2e8f0" if dark else "#1e293b"
            if self._seekpath_result:
                self.fig.patch.set_facecolor(bg)
                if self.fig.axes:
                    self.fig.axes[0].set_facecolor(bg)
                self.canvas.draw_idle()
            if dark:
                self.btn_theme.setText("☀ Light")
                self.btn_theme.setStyleSheet(
                    "QPushButton{background:#f8fafc;color:#1e293b;border:1px solid #cbd5e1;"
                    "border-radius:5px;padding:3px 12px;font-size:12px;font-weight:600;}"
                    "QPushButton:hover{background:#e2e8f0;}"
                )
            else:
                self.btn_theme.setText("🌙 Dark")
                self.btn_theme.setStyleSheet(
                    "QPushButton{background:#1e293b;color:#e2e8f0;border:1px solid #334155;"
                    "border-radius:5px;padding:3px 12px;font-size:12px;font-weight:600;}"
                    "QPushButton:hover{background:#0f172a;}"
                )


# ══════════════════════════════════════════════════════════════════════════════
#  FAT BAND PANEL (ORBITAL PROJECTIONS)
# ══════════════════════════════════════════════════════════════════════════════

from parsers import ProcarParser

class FatBandPanel(QWidget):
    """Visualizes orbital contributions using PROCAR data as bubble scatter plots."""
    def __init__(self):
        super().__init__(); self.vasp_data = None; self.procar_data = None; self._build()
        
    def set_vasp_data(self, d):
        self.vasp_data = d
        if d:
            self.lbl_vasp.setText(f"[OK] {d.get('system','unknown')}\n{d.get('nbands',0)} bands | {d.get('nkpoints',0)} k-pts")
            self.lbl_vasp.setStyleSheet("font-size:10px;color:#16A34A;")
            self._plot()

    def _build(self):
        ml = QVBoxLayout(self); ml.setContentsMargins(0,0,0,0); ml.setSpacing(0)
        sp = QSplitter(Qt.Orientation.Horizontal); sp.setHandleWidth(1)
        
        # Left Panel (Controls)
        cw = QWidget(); cw.setStyleSheet(SIDEBAR_STYLE); cl = QVBoxLayout(cw)
        cl.setContentsMargins(10,10,10,10); cl.setSpacing(8); cw.setMaximumWidth(280)
        
        cl.addWidget(QLabel("<b style='font-size:13px;color:#1e293b'>Fat Band Analysis</b>"))
        
        g1 = QGroupBox("Data Source"); g1l = QVBoxLayout(g1); g1l.setSpacing(4)
        self.lbl_vasp = QLabel("Load vasprun.xml in main tab"); self.lbl_vasp.setWordWrap(True)
        self.lbl_vasp.setStyleSheet("font-size:10px;color:#64748B;")
        g1l.addWidget(self.lbl_vasp)
        
        self.lbl_procar = QLabel("No PROCAR loaded"); self.lbl_procar.setWordWrap(True)
        self.lbl_procar.setStyleSheet("font-size:10px;color:#64748B;")
        g1l.addWidget(self.lbl_procar)
        btn_load = QPushButton("Load PROCAR"); btn_load.setObjectName("primary")
        btn_load.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        btn_load.clicked.connect(self._load_procar); g1l.addWidget(btn_load)
        cl.addWidget(g1)
        
        g2 = QGroupBox("Projection Settings"); g2l = QFormLayout(g2); g2l.setSpacing(6)
        self.cmb_ion = QComboBox()
        self.cmb_ion.addItem("All Ions", -1)
        g2l.addRow("Ion:", self.cmb_ion)
        
        self.cmb_orb = QComboBox()
        self.cmb_orb.addItems(["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2", "Total p", "Total d"])
        g2l.addRow("Orbital:", self.cmb_orb)
        cl.addWidget(g2)
        
        g3 = QGroupBox("Plot Styling"); g3l = QFormLayout(g3); g3l.setSpacing(6)
        self.sp_scale = QDoubleSpinBox(); self.sp_scale.setRange(1, 500); self.sp_scale.setValue(50); self.sp_scale.setSingleStep(10)
        g3l.addRow("Bubble Scale:", self.sp_scale)
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Reds", "Greens"])
        g3l.addRow("Colormap:", self.cmb_cmap)
        self.sp_emin = QDoubleSpinBox(); self.sp_emin.setRange(-30, 0); self.sp_emin.setValue(-6)
        g3l.addRow("E Min (eV):", self.sp_emin)
        self.sp_emax = QDoubleSpinBox(); self.sp_emax.setRange(0, 30); self.sp_emax.setValue(6)
        g3l.addRow("E Max (eV):", self.sp_emax)
        cl.addWidget(g3)
        
        btn_plot = QPushButton("Update Plot"); btn_plot.setObjectName("primary")
        btn_plot.clicked.connect(self._plot); cl.addWidget(btn_plot)
        cl.addStretch(); sp.addWidget(cw)
        
        # Right Panel (Plot)
        rw = QWidget(); rl = QVBoxLayout(rw); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0)
        self.fig = Figure(figsize=(10,7), dpi=100); self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        nav = NavigationToolbar(self.canvas, self); nav.setIconSize(QSize(14,14))
        rl.addWidget(nav); rl.addWidget(self.canvas); sp.addWidget(rw)
        
        ml.addWidget(sp)
        self._placeholder()

    def _placeholder(self):
        self.fig.clear(); ax = self.fig.add_subplot(111); ax.set_axis_off()
        ax.text(0.5, 0.5, "Load vasprun.xml and PROCAR to view Fat Bands", transform=ax.transAxes, ha="center", va="center", color="#94A3B8", fontsize=12)
        self.canvas.draw_idle()

    def _load_procar(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open PROCAR", "", "PROCAR (PROCAR*);;All (*)")
        if not p: return
        try:
            self.procar_data = ProcarParser(p).parse()
            self.lbl_procar.setText(f"[OK] {Path(p).name}\n{self.procar_data['nions']} ions | {self.procar_data['nbands']} bands")
            self.lbl_procar.setStyleSheet("font-size:10px;color:#16A34A;")
            
            # Populate ions
            self.cmb_ion.clear()
            self.cmb_ion.addItem("All Ions", -1)
            for i in range(self.procar_data['nions']):
                self.cmb_ion.addItem(f"Ion {i+1}", i)
            
            self._plot()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse PROCAR:\n{e}")

    def _plot(self):
        if not self.vasp_data or not self.procar_data: return
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            # Use vasprun k-dist if available, else generate linear
            if "kdist" in self.vasp_data and len(self.vasp_data["kdist"]) == self.procar_data["nkpoints"]:
                kdist = self.vasp_data["kdist"]
            else:
                kdist = np.arange(self.procar_data["nkpoints"])
            
            ef = self.vasp_data.get("efermi", 0)
            
            # Gather weights
            w = self.procar_data["weights"] # (nspin, nk, nb, nion, norb)
            e = self.procar_data["energies"] - ef # (nspin, nk, nb)
            
            spin_idx = 0 # Currently plot spin 1 only
            ion_idx = self.cmb_ion.currentData()
            orb_name = self.cmb_orb.currentText()
            
            orb_map = {"s":0, "py":1, "pz":2, "px":3, "dxy":4, "dyz":5, "dz2":6, "dxz":7, "dx2-y2":8}
            
            if ion_idx == -1: # All ions
                w_ion = np.sum(w[spin_idx], axis=2) # sum over ions -> (nk, nb, norb)
            else:
                w_ion = w[spin_idx, :, :, ion_idx, :] # (nk, nb, norb)
                
            if orb_name == "Total p":
                w_plot = np.sum(w_ion[:, :, 1:4], axis=2)
            elif orb_name == "Total d":
                w_plot = np.sum(w_ion[:, :, 4:9], axis=2)
            else:
                idx = orb_map.get(orb_name, 0)
                if idx < w_ion.shape[2]:
                    w_plot = w_ion[:, :, idx]
                else:
                    w_plot = np.zeros_like(e[spin_idx])
            
            emin = self.sp_emin.value()
            emax = self.sp_emax.value()
            scale = self.sp_scale.value()
            cmap = plt.get_cmap(self.cmb_cmap.currentText())
            
            # Plot standard bands first as faint lines
            for ib in range(e.shape[2]):
                band_e = e[spin_idx, :, ib]
                if np.min(band_e) < emax and np.max(band_e) > emin:
                    ax.plot(kdist, band_e, color="#cbd5e1", lw=0.8, zorder=1)
            
            # Scatter plot for fat bands
            K, B = np.meshgrid(kdist, np.arange(e.shape[2]), indexing='ij')
            E_flat = e[spin_idx].flatten()
            K_flat = K.flatten()
            W_flat = w_plot.flatten()
            
            mask = (E_flat >= emin) & (E_flat <= emax) & (W_flat > 0.01)
            
            scatter = ax.scatter(
                K_flat[mask], E_flat[mask], 
                s=W_flat[mask] * scale, 
                c=W_flat[mask], cmap=cmap, 
                alpha=0.7, edgecolors='none', zorder=2
            )
            
            # K-labels from vasprun
            if "klabels" in self.vasp_data and self.vasp_data["klabels"]:
                ticks = []; labels = []
                for idx, lbl in self.vasp_data["klabels"]:
                    if idx < len(kdist):
                        ticks.append(kdist[idx])
                        labels.append(f"${lbl}$" if lbl else "")
                        ax.axvline(kdist[idx], color="#94A3B8", lw=0.8, alpha=0.5, zorder=0)
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels, fontsize=12)
            
            ax.axhline(0, color="#94A3B8", lw=1, ls="--", zorder=0) # Fermi level
            ax.set_xlim(kdist[0], kdist[-1])
            ax.set_ylim(emin, emax)
            ax.set_ylabel(r"$E - E_F$ (eV)", fontsize=11)
            
            title = f"Fat Bands: Ion {'All' if ion_idx == -1 else ion_idx+1} | Orbital: {orb_name}"
            ax.set_title(title, fontsize=11, fontweight="bold")
            
            try: self.fig.colorbar(scatter, ax=ax, label="Weight")
            except: pass
            
            self.fig.tight_layout()
            self.canvas.draw_idle()
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", str(e))


