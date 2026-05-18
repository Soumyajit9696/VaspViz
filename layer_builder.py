"""VaspViz — layer_builder.py: LayerBuilderWidget for 2D heterostructure design."""
import csv, math
import numpy as np
from pathlib import Path
from collections import OrderedDict

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QGroupBox, QLabel, QPushButton, QFileDialog, QComboBox, QDoubleSpinBox,
    QCheckBox, QSpinBox, QFrame, QSizePolicy, QListWidget, QListWidgetItem,
    QAbstractItemView, QDialog, QDialogButtonBox, QTextEdit, QColorDialog,
    QGridLayout, QLineEdit, QStackedWidget)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QPixmap, QIcon, QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import numpy as np

from constants import (MATERIAL_DB, STACKING_PRESETS, MULTILAYER_PRESETS,
                       LAYER_COLORS, ATOM_COLORS, ATOM_SIZES, SIDEBAR_STYLE)
from gl_viewer import StructureGLWidget

def hex_vecs(a):
    return np.array([a,0.]),np.array([a*.5,a*np.sqrt(3)/2])

class LayerBuilderWidget(QWidget):
    def __init__(self):
        super().__init__(); self.layers=[]; self.is_dark=False; self._meas=[]; self._gl_scene_cache=None; self._build()

    def _build(self):
        root=QHBoxLayout(self); root.setContentsMargins(0,0,0,0)
        # Controls scroll
        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame); scroll.setMaximumWidth(310)
        ctrl=QWidget(); ctrl.setStyleSheet(SIDEBAR_STYLE); cl=QVBoxLayout(ctrl); cl.setContentsMargins(10,10,10,10); cl.setSpacing(8)

        # ── Material picker ──
        g=QGroupBox("Add Layer"); gl=QVBoxLayout(g); gl.setSpacing(6)

        # Material grid (visual selector)
        gl.addWidget(QLabel("Material:"))
        self.mat_combo=QComboBox(); self.mat_combo.addItems(list(MATERIAL_DB.keys())); gl.addWidget(self.mat_combo)

        # Quick category buttons
        cat_row=QHBoxLayout()
        for cat,mats in [("TMD",["MoS₂","WS₂","ZrS₂","MoSe₂"]),("Mag",["CrI₃","Fe₃GeTe₂"]),("Hex",["Graphene","hBN","Silicene"])]:
            btn=QPushButton(cat); btn.setFixedHeight(24); btn.setFixedWidth(50)
            btn.clicked.connect(lambda _,m=mats[0]: self.mat_combo.setCurrentText(m))
            cat_row.addWidget(btn)
        gl.addLayout(cat_row)

        grd=QGridLayout(); grd.setSpacing(4)
        grd.addWidget(QLabel("Stacking:"),0,0); self.stk_combo=QComboBox()
        self.stk_combo.addItems(["Custom"]+list(STACKING_PRESETS.keys()))
        self.stk_combo.currentTextChanged.connect(self._apply_stk); grd.addWidget(self.stk_combo,0,1)
        params=[("dx (frac)","sp_dx",-1,1,0,.05,4),("dy (frac)","sp_dy",-1,1,0,.05,4),
                ("Twist (°)","sp_twist",-30,30,0,.1,3),("Gap (Å)","sp_gap",2,10,3.35,.05,2)]
        for row,(lbl,attr,lo,hi,val,step,dec) in enumerate(params,1):
            grd.addWidget(QLabel(lbl),row,0)
            sb=QDoubleSpinBox(); sb.setRange(lo,hi); sb.setSingleStep(step); sb.setDecimals(dec)
            if val: sb.setValue(val)
            setattr(self,attr,sb); grd.addWidget(sb,row,1)
        gl.addLayout(grd)
        btn_add=QPushButton("＋ Add Layer"); btn_add.setObjectName("primary"); btn_add.clicked.connect(self._add); gl.addWidget(btn_add)
        cl.addWidget(g)

        # ── Stacking Pattern Generator ──
        g2=QGroupBox("Stacking Pattern Generator"); g2l=QVBoxLayout(g2); g2l.setSpacing(5)
        nr=QHBoxLayout(); nr.addWidget(QLabel("N layers:"))
        self.sp_nlayers=QSpinBox(); self.sp_nlayers.setRange(2,6); self.sp_nlayers.setValue(2)
        self.sp_nlayers.valueChanged.connect(self._update_patterns); nr.addWidget(self.sp_nlayers)
        nr.addWidget(QLabel("Gap (Å):"))
        self.sp_stack_gap=QDoubleSpinBox(); self.sp_stack_gap.setRange(2,12); self.sp_stack_gap.setValue(3.35); self.sp_stack_gap.setSingleStep(0.05)
        nr.addWidget(self.sp_stack_gap); g2l.addLayout(nr)
        g2l.addWidget(QLabel("Pattern:"))
        self.pattern_combo=QComboBox(); g2l.addWidget(self.pattern_combo)
        # quick pattern buttons row
        qr=QHBoxLayout(); qr.setSpacing(3)
        for pat in ["AA","AB","ABA","ABAB","ABCABC"]:
            b=QPushButton(pat); b.setFixedHeight(24)
            b.clicked.connect(lambda _,p=pat: self._quick_pattern(p))
            qr.addWidget(b)
        g2l.addLayout(qr)
        g2l.addWidget(QLabel("Custom twist per layer (°, comma-sep):"))
        self.le_twist_stack=QLineEdit(); self.le_twist_stack.setPlaceholderText("e.g. 0,1.5,0,1.5")
        g2l.addWidget(self.le_twist_stack)
        btn_gen=QPushButton("Generate Stack"); btn_gen.setObjectName("primary")
        btn_gen.clicked.connect(self._generate_stack); g2l.addWidget(btn_gen)
        cl.addWidget(g2)
        self._update_patterns()

        # ── Layer stack ──
        g3=QGroupBox("Layer Stack"); g3l=QVBoxLayout(g3); g3l.setSpacing(4)
        self.llist=QListWidget(); self.llist.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.llist.setMinimumHeight(130); self.llist.model().rowsMoved.connect(self._reorder)
        self.llist.itemDoubleClicked.connect(self._edit_layer); g3l.addWidget(self.llist)
        br=QHBoxLayout(); br.setSpacing(4)
        for txt,fn in [("Remove",self._remove),("Clear",self._clear),("👁",self._toggle_vis)]:
            b=QPushButton(txt); b.setFixedHeight(28); b.clicked.connect(fn); br.addWidget(b)
        g3l.addLayout(br); cl.addWidget(g3)

        # ── Info ──
        g4=QGroupBox("Structure Info"); g4l=QVBoxLayout(g4)
        self.info_label=QLabel("No layers"); self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size:11px;color:#374151;"); g4l.addWidget(self.info_label); cl.addWidget(g4)

        # ── View options ──
        g5=QGroupBox("View"); g5l=QVBoxLayout(g5); g5l.setSpacing(3)
        self.chk_moire=QCheckBox("Moiré pattern (2D)")
        self.chk_strain=QCheckBox("Strain map (2D)")
        self.chk_charge=QCheckBox("Charge density sketch (2D)")
        self.chk_bonds=QCheckBox("Show Bonds (3D)"); self.chk_bonds.setChecked(True)
        for chk in [self.chk_moire,self.chk_strain,self.chk_charge,self.chk_bonds]:
            chk.stateChanged.connect(self._draw); g5l.addWidget(chk)
            
        r_style=QHBoxLayout(); r_style.addWidget(QLabel("Style:"))
        self.mode_cb=QComboBox()
        self.mode_cb.addItems(["Ball & Stick","Space Fill","Stick Only","Wireframe"])
        self.mode_cb.currentIndexChanged.connect(self._on_mode); r_style.addWidget(self.mode_cb)
        g5l.addLayout(r_style)
            
        self.btn_bg = QPushButton("Toggle Dark/Light BG")
        self.btn_bg.clicked.connect(self._toggle_bg)
        g5l.addWidget(self.btn_bg)
        
        scr=QHBoxLayout(); scr.addWidget(QLabel("Supercell (x,y):"))
        self.sp_sc=QSpinBox(); self.sp_sc.setRange(1,12); self.sp_sc.setValue(3); self.sp_sc.valueChanged.connect(self._draw); scr.addWidget(self.sp_sc)
        g5l.addLayout(scr); cl.addWidget(g5)

        # ── Moiré Analysis ──
        g_ma=QGroupBox("Moiré Analysis"); g_mal=QVBoxLayout(g_ma)
        btn_ma=QPushButton("Find Commensurate Angles"); btn_ma.clicked.connect(self._find_commensurate)
        g_mal.addWidget(btn_ma); cl.addWidget(g_ma)

        # ── Measurement ──
        g7=QGroupBox("Measurement"); g7l=QVBoxLayout(g7)
        mr=QHBoxLayout()
        for t,f in [("📏 Dist",self._meas_d),("∠ Angle",self._meas_a),("✕ Clear",self._meas_clr)]:
            b=QPushButton(t); b.clicked.connect(f); mr.addWidget(b)
        g7l.addLayout(mr)
        self.lbl_meas=QLabel("Click atoms in 3D view"); self.lbl_meas.setWordWrap(True)
        self.lbl_meas.setStyleSheet("font-size:11px;padding:4px;background:#2a2a1e;color:#fde68a;border-radius:4px;")
        g7l.addWidget(self.lbl_meas); cl.addWidget(g7)

        # ── Export ──
        g6=QGroupBox("Export"); g6l=QVBoxLayout(g6); g6l.setSpacing(4)
        for txt,fn in [("Export PNG/PDF",self._export),("POSCAR template",self._poscar),
                       ("CIF template",self._cif),("XYZ format",self._xyz),("Strain report",self._strain_report)]:
            b=QPushButton(txt); b.clicked.connect(fn); g6l.addWidget(b)
        cl.addWidget(g6); cl.addStretch(); scroll.setWidget(ctrl)

        # Canvas
        self.fig=Figure(figsize=(9,7),dpi=100); self.fig.patch.set_facecolor("#ffffff")
        self.canvas=FigureCanvas(self.fig); self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        nav=NavigationToolbar(self.canvas,self); nav.setIconSize(QSize(14,14))
        rw=QWidget(); rl=QVBoxLayout(rw); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0)
        rl.addWidget(nav); rl.addWidget(self.canvas)
        
        self.gl = StructureGLWidget(self)
        self.gl.atom_picked.connect(self._on_pick)
        self.gl.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        
        self.stack = QStackedWidget()
        self.stack.addWidget(self.gl)
        self.stack.addWidget(rw)
        
        # Viewport toolbar
        tb=QWidget(); tb.setFixedHeight(36); tb.setStyleSheet("background:#16213e;border-bottom:1px solid #2d2d50;")
        tl=QHBoxLayout(tb); tl.setContentsMargins(8,3,8,3); tl.setSpacing(8)
        lbl_v = QLabel("View along:")
        lbl_v.setStyleSheet("color:#e2e8f0; font-weight:bold; font-size:11px;")
        tl.addWidget(lbl_v)
        for t,az,el in [("a",90,0), ("b",0,0), ("c",0,90), ("Iso",-45,25)]:
            b=QPushButton(t); b.setStyleSheet("background:#2d2d50;color:#c0c0d0;border:1px solid #3d3d6b;border-radius:4px;padding:2px 8px;font-size:11px;")
            b.clicked.connect(lambda _,a=az,e=el:(self.gl.camera.preset(a,e),self.gl.update()))
            tl.addWidget(b)
        tl.addStretch()
        b_rst=QPushButton("Reset Camera"); b_rst.setStyleSheet("background:#2d2d50;color:#c0c0d0;border:1px solid #3d3d6b;border-radius:4px;padding:2px 8px;font-size:11px;")
        b_rst.clicked.connect(lambda: (self.gl.camera.reset(self.gl._scene_r), self.gl.update()))
        tl.addWidget(b_rst)

        viewport_layout = QVBoxLayout()
        viewport_layout.setContentsMargins(0,0,0,0); viewport_layout.setSpacing(0)
        viewport_layout.addWidget(tb)
        viewport_layout.addWidget(self.stack)
        
        root.addWidget(scroll); root.addLayout(viewport_layout)

    def _apply_stk(self,n):
        if n in STACKING_PRESETS: s=STACKING_PRESETS[n]; self.sp_dx.setValue(s[0]); self.sp_dy.setValue(s[1])

    def _add(self):
        mat=self.mat_combo.currentText(); db=MATERIAL_DB[mat]
        self.layers.append({"material":mat,"a":db["a"],"c":db["c"],
                             "shift":[self.sp_dx.value(),self.sp_dy.value()],
                             "twist":self.sp_twist.value(),"gap":self.sp_gap.value(),
                             "color":LAYER_COLORS[len(self.layers)%len(LAYER_COLORS)],"vis":True})
        self._refresh(); self._update_info(); self._draw()

    # ── stacking pattern generator ──
    # Shift map: A=[0,0], B=[1/3,2/3], C=[2/3,1/3], D=[2/3,0], S=[0.5,0.5]
    _SHIFT_MAP = {
        "A":[0.0, 0.0], "B":[1/3, 2/3], "C":[2/3, 1/3],
        "D":[0.5, 0.0], "S":[0.5, 0.5],
    }
    _PATTERNS = {
        2: ["AA","AB","BA","AC","AD","SS"],
        3: ["AAA","ABA","BAB","ABC","ACA","ABB","ADB","ASA"],
        4: ["ABAB","ABBA","AAAA","ABBC","AABB","ABCD","ABCA","ABDB"],
        5: ["ABABA","ABCBA","AABBA","ABCAB","ABABC"],
        6: ["ABABAB","ABCABC","AABBCC","ABCBCA","ABABBA","ABCACB"],
    }

    def _update_patterns(self):
        n=self.sp_nlayers.value()
        self.pattern_combo.clear()
        self.pattern_combo.addItems(self._PATTERNS.get(n,[]))

    def _quick_pattern(self, pat):
        n=len(pat)
        self.sp_nlayers.setValue(n)
        self._update_patterns()
        idx=self.pattern_combo.findText(pat)
        if idx>=0: self.pattern_combo.setCurrentIndex(idx)
        self._generate_stack()

    def _generate_stack(self):
        mat=self.mat_combo.currentText(); db=MATERIAL_DB[mat]
        pat=self.pattern_combo.currentText()
        if not pat: return
        gap=self.sp_stack_gap.value()
        # parse twist angles
        twist_vals=[0.0]*len(pat)
        raw=self.le_twist_stack.text().strip()
        if raw:
            parts=raw.split(",")
            for i,p in enumerate(parts):
                if i>=len(pat): break
                try: twist_vals[i]=float(p.strip())
                except: pass
        self.layers=[]
        for i,letter in enumerate(pat):
            shift=list(self._SHIFT_MAP.get(letter.upper(),[0.0,0.0]))
            self.layers.append({
                "material":mat,"a":db["a"],"c":db["c"],
                "shift":shift,"twist":twist_vals[i],
                "gap":gap if i>0 else db["c"]/2,
                "color":LAYER_COLORS[i%len(LAYER_COLORS)],"vis":True,
                "_pattern_letter":letter.upper(),
            })
        self._refresh(); self._update_info(); self._draw()

    def _load_preset(self):
        """Legacy: kept for compatibility, delegates to _generate_stack."""
        pass


    def _remove(self):
        r=self.llist.currentRow()
        if 0<=r<len(self.layers): self.layers.pop(r); self._refresh(); self._update_info(); self._draw()

    def _clear(self):
        self.layers=[]; self._refresh(); self._update_info(); self._draw()

    def _toggle_vis(self):
        r=self.llist.currentRow()
        if 0<=r<len(self.layers): self.layers[r]["vis"]=not self.layers[r]["vis"]; self._refresh(); self._draw()

    def _reorder(self):
        new=[]
        for row in range(self.llist.count()):
            item=self.llist.item(row)
            idx=item.data(Qt.ItemDataRole.UserRole)
            if idx is not None and 0<=idx<len(self.layers):
                new.append(self.layers[idx])
        if len(new)==len(self.layers): self.layers=new
        self._refresh(); self._draw()

    def _edit_layer(self,item):
        row=self.llist.currentRow()
        if row<0 or row>=len(self.layers): return
        lay=self.layers[row]
        dlg=QDialog(self); dlg.setWindowTitle(f"Edit Layer {row+1}"); v=QVBoxLayout(dlg)
        form=QFormLayout()
        wdgs={}
        for lbl,key,lo,hi,dec in [("dx",("shift",0),-1,1,4),("dy",("shift",1),-1,1,4),
                                    ("Twist°","twist",-30,30,3),("Gap Å","gap",2,10,2)]:
            sb=QDoubleSpinBox(); sb.setRange(lo,hi); sb.setDecimals(dec)
            sb.setStyleSheet("background:#fff;color:#1e293b;")
            val=lay[key] if isinstance(key,str) else lay[key[0]][key[1]]; sb.setValue(val)
            wdgs[lbl]=(sb,key); form.addRow(f"{lbl}:",sb)
        v.addLayout(form)
        btn_c=QPushButton("Change Color"); btn_c.clicked.connect(lambda: self._pick_col(row,btn_c)); v.addWidget(btn_c)
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); v.addWidget(bb)
        if dlg.exec()==QDialog.DialogCode.Accepted:
            for lbl,(sb,key) in wdgs.items():
                if isinstance(key,str): lay[key]=sb.value()
                else: lay[key[0]][key[1]]=sb.value()
            self._refresh(); self._update_info(); self._draw()

    def _pick_col(self,row,btn):
        c=QColorDialog.getColor(QColor(self.layers[row]["color"]),self)
        if c.isValid(): self.layers[row]["color"]=c.name(); btn.setStyleSheet(f"background:{c.name()};color:#fff;")

    def _refresh(self):
        self.llist.clear()
        # Build full pattern string
        pat_str="".join(l.get("_pattern_letter","?") for l in self.layers)
        for i,l in enumerate(self.layers):
            s=l["shift"]; vis="👁" if l["vis"] else "🙈"
            stk_label=l.get("_pattern_letter","")
            if not stk_label:
                sx,sy=round(s[0],3),round(s[1],3)
                for name2,(px2,py2) in [("A",(0,0)),("B",(1/3,2/3)),("C",(2/3,1/3)),("D",(0.5,0)),("S",(0.5,0.5))]:
                    if abs(sx-px2)<0.01 and abs(sy-py2)<0.01:
                        stk_label=name2; break
            item=QListWidgetItem(f"L{i+1}[{stk_label}]: {l['material']}  θ={l['twist']:.1f}°  d={l.get('gap',3.35):.2f}Å  {vis}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            px=QPixmap(16,16); px.fill(QColor(l["color"])); item.setIcon(QIcon(px)); self.llist.addItem(item)
        if len(self.layers)>1 and pat_str:
            hdr=QListWidgetItem(f"  Pattern: {pat_str}")
            hdr.setFlags(Qt.ItemFlag.NoItemFlags)
            hdr.setForeground(QColor("#2563EB"))
            self.llist.insertItem(0,hdr)

    def _update_info(self):
        if not self.layers: self.info_label.setText("No layers"); return
        lines=[f"<b>{len(self.layers)}-layer heterostructure</b>"]
        for i,l in enumerate(self.layers): lines.append(f"L{i+1}: {l['material']}  a={l['a']:.3f}Å")
        for i in range(1,len(self.layers)):
            s=100*(self.layers[i]["a"]-self.layers[i-1]["a"])/self.layers[i-1]["a"]
            lines.append(f"  ε(L{i}→L{i+1})= {s:+.2f}%")
        for i,l in enumerate(self.layers):
            if abs(l["twist"])>0.01:
                mp=moire_period(l["a"],l["twist"]); lines.append(f"  λ_M(L{i+1})= {mp:.1f}Å")
        self.info_label.setText("<br>".join(lines))

    # ── drawing ──────────────────────────────────────────────────────────────

    def _toggle_bg(self):
        self.is_dark = not self.is_dark
        if self.is_dark:
            self.gl.set_bg(0.08, 0.08, 0.14)
            self.fig.patch.set_facecolor("#0f172a")
        else:
            self.gl.set_bg(1.0, 1.0, 1.0)
            self.fig.patch.set_facecolor("#ffffff")
        self._draw()

    def _get_gl_scene(self):
        if not self.layers: return [], [], None
        sc = self.sp_sc.value()
        a0 = self.layers[0]["a"]
        a1_v, a2_v = hex_vecs(a0)
        c_tot = sum(l.get("gap", 3.35) for l in self.layers[1:]) + 20.0
        lattice = np.array([[a1_v[0], a1_v[1], 0],
                            [a2_v[0], a2_v[1], 0],
                            [0, 0, c_tot]])
        
        pos = []
        labels = []
        z = 10.0
        for i, lay in enumerate(self.layers):
            if not lay.get("vis", True): continue
            if i > 0: z += lay.get("gap", 3.35)
            a = lay["a"]
            la1, la2 = hex_vecs(a)
            s = lay.get("shift", [0,0])
            sc_c = s[0]*la1 + s[1]*la2
            
            th = np.radians(lay.get("twist", 0.0))
            R = np.array([[np.cos(th), -np.sin(th), 0],
                          [np.sin(th),  np.cos(th), 0],
                          [0, 0, 1]])
            
            mat = MATERIAL_DB[lay["material"]]
            db_atoms = mat["atoms"]
            
            for nx in range(-sc, sc+1):
                for ny in range(-sc, sc+1):
                    for sym, frac in db_atoms:
                        af = np.array([frac[0], frac[1], frac[2] if len(frac)>2 else 0.0])
                        p2d = nx*la1 + ny*la2 + af[0]*la1 + af[1]*la2 + sc_c
                        p3d = np.array([p2d[0], p2d[1], z + af[2]*a])
                        p_rot = R @ p3d
                        pos.append(p_rot)
                        labels.append(sym)
        return pos, labels, lattice

    def _draw(self):
        self.fig.clear()
        if not self.layers:
            self.gl.set_scene([], [], None)
            ax=self.fig.add_subplot(111); ax.set_axis_off(); ax.set_facecolor("#0f172a" if self.is_dark else "#ffffff"); self.fig.patch.set_facecolor("#0f172a" if self.is_dark else "#ffffff")
            ax.text(.5,.5,"Add layers from the panel on the left",transform=ax.transAxes,ha="center",va="center",fontsize=12,color="#64748B",style="italic")
            self.canvas.draw(); return
            
        if self.chk_strain.isChecked():
            self.stack.setCurrentIndex(1)
            self._draw_strain()
            self.canvas.draw()
        elif self.chk_moire.isChecked():
            self.stack.setCurrentIndex(1)
            self._draw_moire()
            self.canvas.draw()
        elif self.chk_charge.isChecked():
            self.stack.setCurrentIndex(1)
            self._draw_charge()
            self.canvas.draw()
        else:
            self.stack.setCurrentIndex(0)
            self._draw_3d()

    def _draw_3d(self):
        pos, labels, lattice = self._get_gl_scene()
        self.gl.show_bonds = self.chk_bonds.isChecked()
        self.gl.show_cell = True
        self.gl.set_scene(pos, labels, lattice, 1.15)

    def _draw_moire(self):
        if len(self.layers)<2:
            ax=self.fig.add_subplot(111); ax.text(.5,.5,"Need ≥2 layers",transform=ax.transAxes,ha="center",va="center",fontsize=12,color="#94A3B8"); self.canvas.draw(); return
        ax=self.fig.add_subplot(111,aspect="equal"); ax.set_axis_off()
        bg_col = "#0f172a" if self.is_dark else "#ffffff"
        self.fig.patch.set_facecolor(bg_col); ax.set_facecolor(bg_col)
        l0=self.layers[0]; l1=self.layers[1]
        a1_v,a2_v=hex_vecs(l0["a"]); s0=l0["shift"]; s1=l1["shift"]
        sc0=s0[0]*a1_v+s0[1]*a2_v; sc1=s1[0]*a1_v+s1[1]*a2_v
        th=np.radians(l1["twist"]); R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        a1_r,a2_r=hex_vecs(l1["a"])
        sc_val=self.sp_sc.value()*2; ext=sc_val*l0["a"]
        size=200
        xs=np.linspace(-ext/2,ext/2,size); ys=np.linspace(-ext/2,ext/2,size)
        XX,YY=np.meshgrid(xs,ys); pts=np.stack([XX.ravel(),YY.ravel()],axis=1)
        def density(vecs,off,pts):
            a1,a2=vecs; inv=np.linalg.inv(np.stack([a1,a2],axis=1))
            frac=(inv@(pts-off).T).T; frac-=np.round(frac)
            return np.exp(-10*(frac**2).sum(axis=1))
        d0=density((a1_v,a2_v),sc0,pts)
        d1=density((R@a1_r,R@a2_r),sc1,pts)
        moire=(d0*d1).reshape(size,size)
        ax.imshow(moire,extent=[-ext/2,ext/2,-ext/2,ext/2],cmap="inferno",origin="lower",alpha=0.92)
        mp=moire_period(l0["a"],l1["twist"]) if abs(l1["twist"])>0.01 else np.inf
        title=f"Moiré: {l0['material']}/{l1['material']}"
        if mp<500: title+=f"  —  λ_M≈{mp:.1f}Å (θ={l1['twist']:.2f}°)"
        self.fig.suptitle(title,fontsize=10,color="#e2e8f0" if self.is_dark else "#1e293b",y=0.98)

    def _draw_strain(self):
        if len(self.layers)<2:
            ax=self.fig.add_subplot(111); ax.text(.5,.5,"Need ≥2 layers",transform=ax.transAxes,ha="center",va="center",fontsize=12,color="#888"); self.canvas.draw(); return
        ax=self.fig.add_subplot(111)
        bg_col = "#0f172a" if self.is_dark else "#ffffff"
        fg_col = "#e2e8f0" if self.is_dark else "#1e293b"
        self.fig.patch.set_facecolor(bg_col); ax.set_facecolor(bg_col); fg=fg_col
        pairs=[f"L{i}→L{i+1}" for i in range(1,len(self.layers))]
        strains=[100*(self.layers[i]["a"]-self.layers[i-1]["a"])/self.layers[i-1]["a"] for i in range(1,len(self.layers))]
        colors=["#16A34A" if abs(s)<1 else "#EA580C" if abs(s)<3 else "#DC2626" for s in strains]
        bars=ax.bar(pairs,strains,color=colors,alpha=0.85,edgecolor="#334155",width=0.5)
        ax.axhline(0,color="#94A3B8",lw=0.8,ls="--"); ax.axhspan(-1,1,alpha=0.06,color="#16A34A")
        for bar,s in zip(bars,strains):
            ax.text(bar.get_x()+bar.get_width()/2,s+(.1 if s>=0 else -.2),f"{s:+.2f}%",
                    ha="center",va="bottom" if s>=0 else "top",fontsize=9,color=fg)
        ax.set_ylabel("Lattice strain (%)",color=fg,fontsize=11); ax.set_title("Interlayer lattice strain",color=fg,fontsize=12)
        ax.tick_params(colors=fg); [sp.set_color("#334155") for sp in ax.spines.values()]

    def _draw_charge(self):
        """Schematic charge density sketch."""
        if not self.layers: return
        ax=self.fig.add_subplot(111)
        bg_col = "#0f172a" if self.is_dark else "#ffffff"
        fg_col = "#e2e8f0" if self.is_dark else "#1e293b"
        ax.set_facecolor(bg_col); self.fig.patch.set_facecolor(bg_col); fg=fg_col
        x=np.linspace(-5,5,300)
        z=0.
        for i,lay in enumerate(self.layers):
            if i>0: z+=lay["gap"]
            rho=np.exp(-((x)**2)/2)+0.5*np.exp(-((x-2)**2)/1.5)+0.5*np.exp(-((x+2)**2)/1.5)
            rho*=max(0.5,1.-i*.1)
            ax.plot(x,rho+z,color=lay["color"],lw=2)
            ax.fill_between(x,z,rho+z,color=lay["color"],alpha=0.2)
            ax.text(5.2,z+0.5,f"L{i+1}: {lay['material']}",color=lay["color"],fontsize=9,va="center")
        ax.set_xlim(-6,8); ax.set_ylim(-1,z+2.5)
        ax.set_xlabel("x (Å)",color=fg); ax.set_ylabel("Charge Density (a.u.) / z",color=fg)
        ax.tick_params(colors=fg); [sp.set_color("#334155") for sp in ax.spines.values()]
        self.fig.suptitle("Charge Density Sketch (cross-section)",fontsize=10,color=fg,y=0.98)

    # ── Advanced Features ──

    def _on_mode(self):
        self.gl.render_mode = self.mode_cb.currentIndex()
        self.gl.update()

    def _on_pick(self, idx):
        if idx not in [m[0] for m in self._meas]:
            pos = self.gl._pos[idx]
            self._meas.append((idx, pos))
            if len(self._meas) > 3: self._meas.pop(0)
        
        self.gl._selected = set(m[0] for m in self._meas)
        self.gl._up_spheres()
        self.gl.update()

        if not self._meas:
            self.lbl_meas.setText("Click atoms in 3D view")
        else:
            txt = "Selected:\n"
            for i, (ix, p) in enumerate(self._meas):
                txt += f"Atom {ix}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})\n"
            self.lbl_meas.setText(txt.strip())

    def _meas_d(self):
        if len(self._meas) < 2:
            self.lbl_meas.setText("Select at least 2 atoms first.")
            return
        p1 = self._meas[-2][1]
        p2 = self._meas[-1][1]
        d = np.linalg.norm(p1 - p2)
        self.lbl_meas.setText(f"Distance: {d:.4f} Å")
        
    def _meas_a(self):
        if len(self._meas) < 3:
            self.lbl_meas.setText("Select 3 atoms first.")
            return
        p1 = self._meas[-3][1]
        p2 = self._meas[-2][1]  # vertex
        p3 = self._meas[-1][1]
        v1 = p1 - p2; n1 = np.linalg.norm(v1)
        v2 = p3 - p2; n2 = np.linalg.norm(v2)
        if n1 > 0: v1 /= n1
        if n2 > 0: v2 /= n2
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
        self.lbl_meas.setText(f"Angle: {ang:.2f}°")
        
    def _meas_clr(self):
        self._meas = []
        self.gl.selected_atoms = []
        self.lbl_meas.setText("Click atoms in 3D view")
        self.gl.update()

    def _xyz(self):
        if not self.layers: return
        pos, labels, _ = self._get_gl_scene()
        if not pos: return
        path, _ = QFileDialog.getSaveFileName(self, "Export XYZ", "heterostructure.xyz", "XYZ Files (*.xyz)")
        if not path: return
        with open(path, "w") as f:
            f.write(f"{len(pos)}\n")
            f.write(f"Generated by VaspViz Layer Builder\n")
            for p, l in zip(pos, labels):
                f.write(f"{l} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        self.info_label.setText(f"Exported to {Path(path).name}")

    def _find_commensurate(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Commensurate Twist Angles")
        layout = QVBoxLayout(dlg)
        
        if len(self.layers) < 2:
            layout.addWidget(QLabel("Add at least 2 layers to calculate commensurate angles."))
        else:
            l1, l2 = self.layers[0], self.layers[1]
            a1, a2 = l1["a"], l2["a"]
            info = QLabel(f"L1 ({l1['material']}): a={a1:.3f} Å\nL2 ({l2['material']}): a={a2:.3f} Å")
            layout.addWidget(info)
            
            table = QTableWidget()
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(["Twist Angle (°)", "n", "m", "SC Size (Å)"])
            
            angles = []
            if abs(a1 - a2) < 0.01:
                for n in range(1, 15):
                    for m in range(1, n+1):
                        cos_th = (n**2 + 4*n*m + m**2) / (2*(n**2 + n*m + m**2))
                        if cos_th <= 1.0:
                            th = np.degrees(np.arccos(cos_th))
                            if th > 0.1:
                                sc = a1 * np.sqrt(n**2 + n*m + m**2)
                                angles.append((th, n, m, sc))
                
                angles.sort(key=lambda x: x[0])
                table.setRowCount(len(angles))
                for i, (th, n, m, sc) in enumerate(angles):
                    table.setItem(i, 0, QTableWidgetItem(f"{th:.3f}"))
                    table.setItem(i, 1, QTableWidgetItem(str(n)))
                    table.setItem(i, 2, QTableWidgetItem(str(m)))
                    table.setItem(i, 3, QTableWidgetItem(f"{sc:.2f}"))
                layout.addWidget(table)
            else:
                layout.addWidget(QLabel("Mismatched lattices require strain for commensurability.\nFor heterobilayers, use the 'Strain map' view to find local minima."))

        dlg.setLayout(layout)
        dlg.resize(400, 300)
        dlg.exec()

    def _export(self):
        p,_=QFileDialog.getSaveFileName(self,"Export","layer_structure","PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if p: self.fig.savefig(p,dpi=200,bbox_inches="tight")

    def _poscar(self):
        if not self.layers: return
        a=self.layers[0]["a"]; c=sum(l["gap"] for l in self.layers[1:])+20.
        a1,a2=hex_vecs(a)
        all_atoms=[]
        z=10.
        for i,lay in enumerate(self.layers):
            if i>0: z+=lay["gap"]
            s=lay["shift"]
            for sym,frac in MATERIAL_DB[lay["material"]]["atoms"]:
                all_atoms.append((sym,frac[0]+s[0],frac[1]+s[1],z/c,i+1))
        spec_order=list(OrderedDict.fromkeys(a[0] for a in all_atoms))
        spec_counts=[sum(1 for a in all_atoms if a[0]==sp) for sp in spec_order]
        lines=[f"{'+'.join(dict.fromkeys(l['material'] for l in self.layers))} {len(self.layers)}-layer","1.0",
               f"  {a1[0]:.6f}   {a1[1]:.6f}   0.000000",
               f"  {a2[0]:.6f}   {a2[1]:.6f}   0.000000",
               f"  0.000000   0.000000   {c:.6f}",
               "  ".join(spec_order),"  ".join(str(v) for v in spec_counts),"Direct"]
        for sp in spec_order:
            for sym,fx,fy,fz,layer in all_atoms:
                if sym==sp: lines.append(f"  {fx:.6f}   {fy:.6f}   {fz:.6f}   ! {sym} L{layer}")
        self._show_text(f"POSCAR — {dict(zip(spec_order,spec_counts))}","\n".join(lines))

    def _cif(self):
        if not self.layers: return
        a=self.layers[0]["a"]; c=sum(l["gap"] for l in self.layers[1:])+20.
        lines=["data_layer_structure",f"_cell_length_a   {a:.4f}",f"_cell_length_b   {a:.4f}",
               f"_cell_length_c   {c:.4f}","_cell_angle_alpha   90","_cell_angle_beta    90",
               "_cell_angle_gamma   120","_symmetry_space_group_name_H-M   'P 1'",
               "loop_","_atom_site_label","_atom_site_type_symbol",
               "_atom_site_fract_x","_atom_site_fract_y","_atom_site_fract_z"]
        z=10.
        for i,lay in enumerate(self.layers):
            if i>0: z+=lay["gap"]
            s=lay["shift"]
            for j,(sym,frac) in enumerate(MATERIAL_DB[lay["material"]]["atoms"]):
                lines.append(f"{sym}{i+1}{j+1}  {sym}  {frac[0]+s[0]:.4f}  {frac[1]+s[1]:.4f}  {z/c:.4f}")
        self._show_text("CIF Template","\n".join(lines))

    def _strain_report(self):
        if len(self.layers)<2: return
        lines=["Interlayer Strain & Moiré Report","="*45]
        for i in range(1,len(self.layers)):
            a1=self.layers[i-1]["a"]; a2=self.layers[i]["a"]; s=100*(a2-a1)/a1
            mp_val=moire_period(a1,self.layers[i]["twist"]) if abs(self.layers[i]["twist"])>0.01 else None
            mp_str=f"{mp_val:.1f} Å" if mp_val is not None else "N/A"
            lines+=[f"L{i}({self.layers[i-1]['material']})→L{i+1}({self.layers[i]['material']}):",
                    f"  Strain={s:+.3f}%  Twist={self.layers[i]['twist']:.3f}°  λ_M={mp_str}  d_vdW={self.layers[i]['gap']:.2f}Å"]
        self._show_text("Strain Report","\n".join(lines))

    def _show_text(self,title,text):
        dlg=QDialog(self); dlg.setWindowTitle(title); dlg.resize(560,460)
        v=QVBoxLayout(dlg); te=QTextEdit(); te.setFont(QFont("Courier New",10))
        te.setStyleSheet("background:#fff;color:#1e293b;"); te.setPlainText(text); v.addWidget(te)
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok); bb.accepted.connect(dlg.accept); v.addWidget(bb); dlg.exec()


# ══════════════════════════════════════════════════════════════════════════════
#  POSCAR VIEWER (enhanced)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  POSCAR VIEWER — VESTA-STYLE (complete replacement)
#  Features: ball-and-stick / spacefill / wireframe, spglib symmetry,
#             supercell, distance/angle measurement, rotation, export,
#             p4vasp-inspired: force vectors, polyhedra hints, coordination
# ══════════════════════════════════════════════════════════════════════════════
