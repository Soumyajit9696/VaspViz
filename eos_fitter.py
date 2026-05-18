"""
VaspViz — eos_fitter.py
Equation of State (EOS) Fitting tool for calculating equilibrium volume and bulk modulus.
"""

import numpy as np
from scipy.optimize import curve_fit
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFileDialog, QMessageBox, QComboBox, QPlainTextEdit,
    QSplitter, QFrame, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from constants import SIDEBAR_STYLE
from parsers import VasprunParser

# Conversion factor: eV/Angstrom^3 to GPa
EV_A3_TO_GPA = 160.21766208

# ── EOS Equations ──
def eos_murnaghan(V, E0, B0, Bp, V0):
    return E0 + B0 * V / Bp * (((V0 / V)**Bp) / (Bp - 1.0) + 1.0) - V0 * B0 / (Bp - 1.0)

def eos_birch_murnaghan(V, E0, B0, Bp, V0):
    eta = (V0 / V)**(2.0 / 3.0)
    return E0 + 9.0 * V0 * B0 / 16.0 * (eta - 1.0)**2 * (6.0 + Bp * (eta - 1.0) - 4.0 * eta)

def eos_vinet(V, E0, B0, Bp, V0):
    eta = (V / V0)**(1.0 / 3.0)
    return E0 + 2.0 * B0 * V0 / (Bp - 1.0)**2 * (2.0 - (5.0 + 3.0 * Bp * (eta - 1.0) - 3.0 * eta) * np.exp(-1.5 * (Bp - 1.0) * (eta - 1.0)))

def eos_parabola(V, a, b, c):
    return a * V**2 + b * V + c

EOS_MODELS = {
    "Birch-Murnaghan (3rd Order)": eos_birch_murnaghan,
    "Murnaghan": eos_murnaghan,
    "Rose-Vinet": eos_vinet,
    "Parabola (Quadratic)": eos_parabola
}

class EOSWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.volumes = np.array([])
        self.energies = np.array([])
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 6)
        lay.setSpacing(4)
        
        # Header
        header = QLabel("<b style='font-size:13px;color:#1e293b'>Equation of State (EOS) Fitter</b>  "
                        "<span style='font-size:10px;color:#64748b'>Calculate Bulk Modulus & Equilibrium Volume</span>")
        header.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
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
        left_scroll.setMinimumWidth(260); left_scroll.setMaximumWidth(320)
        
        left = QWidget(); left.setStyleSheet(SIDEBAR_STYLE + _G_STYLE)
        ll = QVBoxLayout(left); ll.setContentsMargins(4, 4, 4, 4); ll.setSpacing(6)
        
        g_data = QGroupBox("Energy-Volume Data"); gl1 = QVBoxLayout(g_data); gl1.setSpacing(6)
        
        h1 = QHBoxLayout()
        btn_load = QPushButton("Load vasprun.xml files")
        btn_load.setObjectName("primary")
        btn_load.clicked.connect(self._load_files)
        h1.addWidget(btn_load)
        gl1.addLayout(h1)
        
        gl1.addWidget(QLabel("Or paste Volume & Energy columns here:"))
        self.txt_data = QPlainTextEdit()
        self.txt_data.setPlaceholderText("Volume (Å³)    Energy (eV)\n30.12          -120.45\n31.50          -121.10\n...")
        self.txt_data.setMinimumHeight(150)
        self.txt_data.textChanged.connect(self._parse_text)
        gl1.addWidget(self.txt_data)
        ll.addWidget(g_data)
        
        g_fit = QGroupBox("Fitting Model"); gl2 = QVBoxLayout(g_fit); gl2.setSpacing(6)
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(list(EOS_MODELS.keys()))
        self.cmb_model.currentIndexChanged.connect(self._fit_and_plot)
        gl2.addWidget(self.cmb_model)
        
        # Results table
        self.tbl_res = QTableWidget(4, 2)
        self.tbl_res.horizontalHeader().setVisible(False)
        self.tbl_res.verticalHeader().setVisible(False)
        self.tbl_res.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_res.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.tbl_res.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_res.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        
        font = self.tbl_res.font()
        font.setPointSize(11)
        self.tbl_res.setFont(font)
        
        self.tbl_res.setItem(0, 0, QTableWidgetItem("V₀ (Å³):"))
        self.tbl_res.setItem(1, 0, QTableWidgetItem("E₀ (eV):"))
        self.tbl_res.setItem(2, 0, QTableWidgetItem("B₀ (GPa):"))
        self.tbl_res.setItem(3, 0, QTableWidgetItem("B'₀:"))
        for i in range(4): self.tbl_res.setItem(i, 1, QTableWidgetItem("—"))
        self.tbl_res.setMinimumHeight(125)
        gl2.addWidget(self.tbl_res)
        
        ll.addWidget(g_fit)
        
        ll.addStretch()
        left_scroll.setWidget(left)
        sp.addWidget(left_scroll)
        
        # ── Right viewport (Matplotlib) ──
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, right)
        self.ax = self.fig.add_subplot(111)
        
        rl.addWidget(self.toolbar)
        rl.addWidget(self.canvas)
        
        sp.addWidget(right)
        sp.setStretchFactor(0, 0); sp.setStretchFactor(1, 1)
        lay.addWidget(sp)

        self._reset_plot()

    def _reset_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Volume (Å³)", fontsize=11, fontweight='bold')
        self.ax.set_ylabel("Total Energy (eV)", fontsize=11, fontweight='bold')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.fig.tight_layout()
        self.canvas.draw()

    def _load_files(self):
        fps, _ = QFileDialog.getOpenFileNames(self, "Select vasprun.xml files", "", "VASP (*.xml)")
        if not fps: return
        
        vols, enes = [], []
        try:
            for fp in fps:
                data = VasprunParser(fp).parse()
                lat = data["lattice"]
                v = np.dot(lat[0], np.cross(lat[1], lat[2]))
                vols.append(abs(v))
                enes.append(data["energy"])
            
            txt = ""
            for v, e in zip(vols, enes):
                txt += f"{v:.6f}\t{e:.6f}\n"
            
            self.txt_data.blockSignals(True)
            self.txt_data.setPlainText(txt)
            self.txt_data.blockSignals(False)
            
            self._parse_text()
        except Exception as e:
            QMessageBox.critical(self, "Error parsing files", str(e))

    def _parse_text(self):
        txt = self.txt_data.toPlainText().strip()
        vols, enes = [], []
        for line in txt.split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    v = float(parts[0].replace(',', ''))
                    e = float(parts[1].replace(',', ''))
                    vols.append(v)
                    enes.append(e)
                except ValueError:
                    pass
        
        self.volumes = np.array(vols)
        self.energies = np.array(enes)
        self._fit_and_plot()

    def _fit_and_plot(self):
        self._reset_plot()
        
        for i in range(4): self.tbl_res.setItem(i, 1, QTableWidgetItem("—"))
        
        if len(self.volumes) < 4:
            if len(self.volumes) > 0:
                self.ax.plot(self.volumes, self.energies, 'o', color="#2563EB", markersize=8)
                self.canvas.draw()
            return
            
        V = self.volumes
        E = self.energies
        
        idx = np.argsort(V)
        V = V[idx]
        E = E[idx]
        
        self.ax.plot(V, E, 'o', color="#2563EB", markersize=8, label="DFT Data")
        
        try:
            poly = np.polyfit(V, E, 2)
            a, b, c = poly
            
            V0_guess = -b / (2 * a)
            if V0_guess < V.min() * 0.5 or V0_guess > V.max() * 1.5 or a < 0:
                idx_min = np.argmin(E)
                V0_guess = V[idx_min]
                E0_guess = E[idx_min]
                B0_guess = 50.0 / EV_A3_TO_GPA
            else:
                E0_guess = a * V0_guess**2 + b * V0_guess + c
                B0_guess = V0_guess * 2 * a
                
            if B0_guess < 0: B0_guess = 50.0 / EV_A3_TO_GPA
            Bp_guess = 4.0
            
            model_name = self.cmb_model.currentText()
            func = EOS_MODELS[model_name]
            
            if model_name == "Parabola (Quadratic)":
                popt = poly
                V0 = V0_guess
                E0 = E0_guess
                B0_eV = B0_guess
                Bp = 0.0
                
                v_fit = np.linspace(V.min()*0.95, V.max()*1.05, 100)
                e_fit = eos_parabola(v_fit, *popt)
            else:
                p0 = [E0_guess, B0_guess, Bp_guess, V0_guess]
                popt, pcov = curve_fit(func, V, E, p0=p0, maxfev=10000)
                E0, B0_eV, Bp, V0 = popt
                
                v_fit = np.linspace(V.min()*0.95, V.max()*1.05, 100)
                e_fit = func(v_fit, *popt)
                
            B0_gpa = B0_eV * EV_A3_TO_GPA
            
            self.ax.plot(v_fit, e_fit, '-', color="#DC2626", linewidth=2, label=f"{model_name} Fit")
            self.ax.legend(frameon=False)
            
            self.ax.plot(V0, E0, 'x', color="#16A34A", markersize=10, markeredgewidth=2)
            self.ax.axvline(V0, color="#16A34A", linestyle=":", alpha=0.5)
            
            self.tbl_res.setItem(0, 1, QTableWidgetItem(f"{V0:.4f}"))
            self.tbl_res.setItem(1, 1, QTableWidgetItem(f"{E0:.5f}"))
            self.tbl_res.setItem(2, 1, QTableWidgetItem(f"{B0_gpa:.2f}"))
            if model_name != "Parabola (Quadratic)":
                self.tbl_res.setItem(3, 1, QTableWidgetItem(f"{Bp:.3f}"))
            else:
                self.tbl_res.setItem(3, 1, QTableWidgetItem("N/A"))
                
        except Exception as e:
            self.ax.set_title("Fit failed to converge", color="red", fontsize=10)
            
        self.canvas.draw()
