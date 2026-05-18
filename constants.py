"""VaspViz — constants.py: element tables, physical constants, UI stylesheets."""

VERSION   = "4.1"
DEVELOPER = "Soumyajit Das, NIT Silchar"
HBAR2_OVER_2M = 3.81
EV_TO_CM1     = 8065.54
EPSILON0      = 8.854e-12

ORBITAL_NAMES = ["s","py","pz","px","dxy","dyz","dz2","dxz","dx2-y2","fy3x2","fxyz","fyz2","fz3","fxz2","fzx2","fx3"]
ORBITAL_GROUPS = {
    "s":[0],"p":[1,2,3],"px":[3],"py":[1],"pz":[2],
    "d":[4,5,6,7,8],"dxy":[4],"dyz":[5],"dz2":[6],"dxz":[7],"dx2-y2":[8],
    "f":[9,10,11,12,13,14,15],"all":list(range(9)),
}
SPIN_COLORS = ["#2563EB","#DC2626"]
ORBITAL_COLORS = {"s":"#2563EB","p":"#16A34A","px":"#15803D","py":"#166534","pz":"#14532D","d":"#EA580C","dxy":"#C2410C","dyz":"#B45309","dz2":"#92400E","dxz":"#78350F","dx2-y2":"#F59E0B","f":"#7C3AED","all":"#0891B2"}
LAYER_COLORS = ["#2563EB","#DC2626","#16A34A","#7C3AED","#D97706","#0891B2","#BE185D","#4B5563","#065F46","#7F1D1D"]
KNAME_MAP = {"gamma":"Γ","Gamma":"Γ","GAMMA":"Γ","G":"Γ","g":"Γ","\\Gamma":"Γ","M":"M","K":"K","X":"X","L":"L","W":"W","H":"H","N":"N","P":"P","A":"A","Z":"Z","R":"R","U":"U","T":"T","S":"S","F":"F","Q":"Q"}

VALENCE_ELECTRONS = {
    "H":1,"He":2,"Li":1,"Be":2,"B":3,"C":4,"N":5,"O":6,"F":7,"Ne":8,"Na":1,"Mg":2,"Al":3,"Si":4,"P":5,"S":6,"Cl":7,"Ar":8,"K":1,"Ca":2,"Sc":3,"Ti":4,"V":5,"Cr":6,"Mn":7,"Fe":8,"Co":9,"Ni":10,"Cu":11,"Zn":12,"Ga":3,"Ge":4,"As":5,"Se":6,"Br":7,"Kr":8,"Rb":1,"Sr":2,"Y":3,"Zr":4,"Nb":5,"Mo":6,"Tc":7,"Ru":8,"Rh":9,"Pd":10,"Ag":11,"Cd":12,"In":3,"Sn":4,"Sb":5,"Te":6,"I":7,"Xe":8,"Cs":1,"Ba":2,"La":3,"Hf":4,"Ta":5,"W":6,"Re":7,"Os":8,"Ir":9,"Pt":10,"Au":11,"Hg":12,"Tl":3,"Pb":4,"Bi":5,
}
COVALENT_RADII = {"H":0.31,"C":0.77,"N":0.75,"O":0.73,"F":0.71,"S":1.03,"P":1.07,"Si":1.17,"Ge":1.22,"As":1.19,"Se":1.20,"Br":1.20,"Mo":1.46,"W":1.50,"Zr":1.75,"Nb":1.64,"Ta":1.70,"V":1.53,"Ti":1.60,"Cr":1.39,"Fe":1.32,"Co":1.26,"Ni":1.24,"Cu":1.32,"Zn":1.22,"Ga":1.22,"In":1.42,"Sn":1.39,"Pb":1.46,"Bi":1.51,"B":0.82,"Al":1.21,"I":1.39,"Cl":0.99,"Hf":1.75,"Mn":1.39,"default":1.50}
VDW_RADII = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"F":1.47,"S":1.80,"P":1.80,"Si":2.10,"Ge":2.11,"Cl":1.75,"Br":1.85,"I":1.98,"Zr":2.36,"Mo":2.09,"W":2.10,"Nb":2.07,"V":2.07,"Ti":2.11,"Fe":2.05,"Co":2.00,"Ni":1.99,"Cu":1.96,"Zn":2.01,"default":2.00}
DISPLAY_RADII = {"H":0.31,"C":0.77,"N":0.75,"O":0.73,"F":0.64,"S":1.03,"P":1.06,"Si":1.11,"Ge":1.22,"Zr":1.56,"Mo":1.30,"W":1.35,"Nb":1.46,"V":1.34,"Ti":1.40,"Cr":1.28,"Fe":1.24,"Co":1.18,"Ni":1.17,"Cu":1.28,"Zn":1.18,"In":1.42,"Ga":1.22,"Sn":1.39,"Pb":1.54,"Bi":1.51,"Hf":1.56,"Ta":1.46,"Se":1.16,"Te":1.36,"default":1.25}
ATOM_COLORS = {"H":"#FFFFFF","He":"#D9FFFF","Li":"#CC80FF","Be":"#C2FF00","B":"#FFB5B5","C":"#909090","N":"#3050F8","O":"#FF0D0D","F":"#90E050","Ne":"#B3E3F5","Na":"#AB5CF2","Mg":"#8AFF00","Al":"#BFA6A6","Si":"#F0C8A0","P":"#FF8000","S":"#FFFF30","Cl":"#1FF01F","Ar":"#80D1E3","K":"#8F40D4","Ca":"#3DFF00","Sc":"#E6E6E6","Ti":"#BFC2C7","V":"#A6A6AB","Cr":"#8A99C7","Mn":"#9C7AC7","Fe":"#E06633","Co":"#F090A0","Ni":"#50D050","Cu":"#C88033","Zn":"#7D80B0","Ga":"#C28F8F","Ge":"#668F8F","As":"#BD80E3","Se":"#FFA100","Br":"#A62929","Kr":"#5CB8D1","Rb":"#702EB0","Sr":"#00FF00","Y":"#94FFFF","Zr":"#94E0E0","Nb":"#73C2C9","Mo":"#54B5B5","Ru":"#248F8F","Rh":"#0A7D8C","Pd":"#006985","Ag":"#C0C0C0","Cd":"#FFD98F","In":"#A67573","Sn":"#668080","Sb":"#9E63B5","Te":"#D47A00","I":"#940094","W":"#2194D6","Ta":"#4DA6FF","Hf":"#4DC2FF","Re":"#267DAB","Os":"#266696","Ir":"#175487","Pt":"#D0D0E0","Au":"#FFD123","Pb":"#575961","Bi":"#9E4FB5","default":"#888888"}
ATOM_SIZES = {"Zr":140,"Mo":120,"W":130,"Nb":125,"V":110,"Ti":120,"Ta":130,"In":115,"Ga":105,"Cr":115,"Sn":130,"Pb":140,"Bi":140,"C":80,"N":80,"O":75,"H":50,"S":95,"Se":105,"Te":120,"I":110,"Fe":110,"Co":105,"Ni":105,"Cu":105,"Zn":110,"default":95}
ELEMENT_NUMBERS = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
    "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
    "Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,
    "Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,
    "Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,
    "Sb":51,"Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,"La":57,"Ce":58,
    "Pr":59,"Nd":60,"Pm":61,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,
    "Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71,"Hf":72,"Ta":73,"W":74,
    "Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Tl":81,"Pb":82,
    "Bi":83,"Po":84,"At":85,"Rn":86,"Fr":87,"Ra":88,"Ac":89,"Th":90,
    "Pa":91,"U":92,"Np":93,"Pu":94,"Am":95,"Cm":96,"Bk":97,"Cf":98,
    "Es":99,"Fm":100,"Md":101,"No":102,"Lr":103,"Rf":104,"Db":105,
    "Sg":106,"Bh":107,"Hs":108,"Mt":109,"Ds":110,"Rg":111,"Cn":112,
    "Nh":113,"Fl":114,"Mc":115,"Lv":116,"Ts":117,"Og":118,
}

MATERIAL_DB = {
    "Graphene":   {"a":2.46,"c":3.35,"atoms":[("C",[0,0,0]),("C",[1/3,2/3,0])],"type":"hex","color":"#374151","mag":False},
    "hBN":        {"a":2.50,"c":3.33,"atoms":[("B",[0,0,0]),("N",[1/3,2/3,0])],"type":"hex","color":"#60A5FA","mag":False},
    "Silicene":   {"a":3.84,"c":4.20,"atoms":[("Si",[0,0,0]),("Si",[1/3,2/3,0.05])],"type":"hex","color":"#94A3B8","mag":False},
    "Germanene":  {"a":4.02,"c":4.30,"atoms":[("Ge",[0,0,0]),("Ge",[1/3,2/3,0.05])],"type":"hex","color":"#CBD5E1","mag":False},
    "MoS2":  {"a":3.18,"c":6.15,"atoms":[("Mo",[0,0,0]),("S",[2/3,1/3,0.25]),("S",[2/3,1/3,-0.25])],"type":"hex","color":"#F97316","mag":False},
    "WS2":   {"a":3.18,"c":6.18,"atoms":[("W",[0,0,0]),("S",[2/3,1/3,0.25]),("S",[2/3,1/3,-0.25])],"type":"hex","color":"#A855F7","mag":False},
    "MoSe2": {"a":3.32,"c":6.45,"atoms":[("Mo",[0,0,0]),("Se",[2/3,1/3,0.25]),("Se",[2/3,1/3,-0.25])],"type":"hex","color":"#EF4444","mag":False},
    "WSe2":  {"a":3.32,"c":6.48,"atoms":[("W",[0,0,0]),("Se",[2/3,1/3,0.25]),("Se",[2/3,1/3,-0.25])],"type":"hex","color":"#EC4899","mag":False},
    "ZrS2":  {"a":3.66,"c":5.85,"atoms":[("Zr",[0,0,0]),("S",[2/3,1/3,0.25]),("S",[2/3,1/3,-0.25])],"type":"hex","color":"#06B6D4","mag":False},
    "ZrSe2": {"a":3.77,"c":6.14,"atoms":[("Zr",[0,0,0]),("Se",[2/3,1/3,0.25]),("Se",[2/3,1/3,-0.25])],"type":"hex","color":"#0EA5E9","mag":False},
    "HfS2":  {"a":3.64,"c":5.84,"atoms":[("Hf",[0,0,0]),("S",[2/3,1/3,0.25]),("S",[2/3,1/3,-0.25])],"type":"hex","color":"#22D3EE","mag":False},
    "NbSe2": {"a":3.44,"c":6.27,"atoms":[("Nb",[0,0,0]),("Se",[2/3,1/3,0.25]),("Se",[2/3,1/3,-0.25])],"type":"hex","color":"#F59E0B","mag":False},
    "CrI3":  {"a":6.87,"c":6.61,"atoms":[("Cr",[0,0,0]),("I",[2/3,1/3,0.25]),("I",[0,1/3,0.15])],"type":"hex","color":"#F87171","mag":True},
    "InSe":  {"a":4.05,"c":8.32,"atoms":[("In",[0,0,0]),("Se",[2/3,1/3,0.25]),("Se",[2/3,1/3,-0.25])],"type":"hex","color":"#34D399","mag":False},
    "Custom":{"a":3.00,"c":6.00,"atoms":[("X",[0,0,0])],"type":"hex","color":"#64748B","mag":False},
}
STACKING_PRESETS = {"AA":[0.0,0.0],"AB":[1/3,2/3],"BA":[2/3,1/3],"SP":[0.5,0.0],"AC":[0.0,1/3],"Hollow":[0.5,0.5]}
MULTILAYER_PRESETS = {
    "Bilayer AA":[{"s":[0,0]},{"s":[0,0]}],"Bilayer AB":[{"s":[0,0]},{"s":[1/3,2/3]}],
    "Trilayer ABA":[{"s":[0,0]},{"s":[1/3,2/3]},{"s":[2/3,1/3]}],
    "4L ABAB":[{"s":[0,0]},{"s":[1/3,2/3]},{"s":[0,0]},{"s":[1/3,2/3]}],
    "6L ABABAB":[{"s":[0,0]},{"s":[1/3,2/3]},{"s":[0,0]},{"s":[1/3,2/3]},{"s":[0,0]},{"s":[1/3,2/3]}],
}
HSYM_KPOINTS = {
    "Cubic (SC)":[("Γ","\\Gamma","[0,0,0]","Zone centre"),("X","X","[0,1/2,0]",""),("M","M","[1/2,1/2,0]",""),("R","R","[1/2,1/2,1/2]","")],
    "FCC":[("Γ","\\Gamma","[0,0,0]",""),("X","X","[1/2,0,1/2]",""),("L","L","[1/2,1/2,1/2]",""),("W","W","[1/2,1/4,3/4]",""),("K","K","[3/8,3/8,3/4]","")],
    "BCC":[("Γ","\\Gamma","[0,0,0]",""),("H","H","[1/2,-1/2,1/2]",""),("N","N","[0,0,1/2]",""),("P","P","[1/4,1/4,1/4]","")],
    "Hexagonal":[("Γ","\\Gamma","[0,0,0]","Zone centre"),("M","M","[1/2,0,0]",""),("K","K","[1/3,1/3,0]",""),("A","A","[0,0,1/2]",""),("L","L","[1/2,0,1/2]",""),("H","H","[1/3,1/3,1/2]","")],
    "Tetragonal":[("Γ","\\Gamma","[0,0,0]",""),("X","X","[1/2,0,0]",""),("M","M","[1/2,1/2,0]",""),("Z","Z","[0,0,1/2]",""),("R","R","[1/2,0,1/2]",""),("A","A","[1/2,1/2,1/2]","")],
    "Orthorhombic":[("Γ","\\Gamma","[0,0,0]",""),("X","X","[1/2,0,0]",""),("Y","Y","[0,1/2,0]",""),("Z","Z","[0,0,1/2]",""),("S","S","[1/2,1/2,0]",""),("R","R","[1/2,1/2,1/2]","")],
}
AVAILABLE_FONTS = ["DejaVu Sans","DejaVu Serif","DejaVu Sans Mono","sans-serif","serif","monospace","STIXGeneral","Times New Roman","Arial","Courier New","Palatino"]

MENU_STYLE = """
QMenuBar { background:#ffffff; color:#374151; border-bottom:1px solid #e2e8f0; padding:2px 6px; spacing:1px; font-size:12px; }
QMenuBar::item { background:transparent; color:#374151; padding:5px 12px; border-radius:4px; }
QMenuBar::item:selected { background:#eff6ff; color:#2563EB; }
QMenu { background:#ffffff; color:#374151; border:1px solid #e2e8f0; padding:4px 0; font-size:12px; border-radius:8px; }
QMenu::item { padding:7px 28px 7px 14px; }
QMenu::item:selected { background:#eff6ff; color:#2563EB; }
QMenu::separator { height:1px; background:#f1f5f9; margin:4px 10px; }
QMenu::item:disabled { color:#94a3b8; }
"""
SIDEBAR_STYLE = """
QWidget#sidebar { background:#f8fafc; }
QGroupBox { background:#ffffff; }
"""
VIEWER_SIDE_STYLE = """
QWidget{background:#f8fafc;color:#1e293b;}
QGroupBox{font-weight:600;border:1px solid #e2e8f0;border-radius:8px;margin-top:12px;padding-top:10px;background:#ffffff;color:#1e293b;}
QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 6px;background:#ffffff;color:#475569;}
QLabel{color:#374151;font-size:12px;}
QCheckBox{color:#374151;font-size:12px;spacing:7px;}
QCheckBox::indicator{width:17px;height:17px;border-radius:4px;border:2px solid #cbd5e1;background:#ffffff;}
QCheckBox::indicator:checked{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #3b82f6,stop:1 #2563EB);border-color:#2563EB;}
QPushButton{background:#f8fafc;color:#374151;border:1px solid #e2e8f0;border-radius:6px;padding:6px 12px;font-size:12px;min-height:28px;}
QPushButton:hover{background:#f1f5f9;border-color:#93c5fd;}
QPushButton:pressed{background:#dbeafe;}
QPushButton#primary{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #3b82f6,stop:1 #2563EB);color:#ffffff;border-color:#1d4ed8;font-weight:700;}
QPushButton#danger{background:#fef2f2;color:#dc2626;border-color:#fca5a5;}
QComboBox,QSpinBox,QDoubleSpinBox,QLineEdit{background:#ffffff;color:#1e293b;border:1px solid #e2e8f0;border-radius:5px;padding:3px 8px;min-height:26px;font-size:12px;}
QComboBox:focus,QSpinBox:focus,QDoubleSpinBox:focus{border-color:#2563EB;}
QComboBox QAbstractItemView{background:#ffffff;color:#1e293b;border:1px solid #e2e8f0;selection-background-color:#dbeafe;}
QScrollBar:vertical{background:#f1f5f9;width:7px;border-radius:3px;}
QScrollBar::handle:vertical{background:#cbd5e1;border-radius:3px;min-height:20px;}
QScrollBar::handle:vertical:hover{background:#94a3b8;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}
QSlider::groove:horizontal{height:4px;background:#e2e8f0;border-radius:2px;}
QSlider::handle:horizontal{background:#2563EB;width:14px;height:14px;margin:-5px 0;border-radius:7px;}
QSlider::sub-page:horizontal{background:#2563EB;border-radius:2px;}
QTreeWidget,QTableWidget{background:#ffffff;color:#1e293b;border:1px solid #e2e8f0;font-size:11px;alternate-background-color:#f8fafc;}
QHeaderView::section{background:#f8fafc;color:#475569;border:none;padding:4px;font-weight:600;}
"""

GLOBAL_APP_STYLE = """
/* === Global reset === */
QWidget { font-family: 'Segoe UI', 'Inter', 'DejaVu Sans', sans-serif; font-size: 12px; color: #1e293b; }
QMainWindow { background: #f8fafc; }

/* === Toolbar (white) === */
QToolBar { background:#ffffff; border:none; border-bottom:1px solid #e2e8f0; padding:2px 8px; spacing:2px; }
QToolBar::separator { background:#e2e8f0; width:1px; margin:3px 5px; }
QToolButton { background:transparent; color:#374151; border:none; border-radius:5px; padding:3px 10px; font-size:11px; font-weight:500; }
QToolButton:hover { background:#f1f5f9; color:#1e293b; border:1px solid #e2e8f0; }
QToolButton:pressed { background:#dbeafe; color:#2563EB; }

/* === Tabs === */
QTabWidget::pane { border:none; background:#f8fafc; }
QTabBar { background:#f8fafc; border-bottom:1px solid #e2e8f0; }
QTabBar::tab { background:transparent; color:#64748b; padding:6px 14px; font-size:11px; font-weight:500; border:none; border-bottom:2px solid transparent; margin-right:1px; }
QTabBar::tab:selected { color:#2563EB; border-bottom:2px solid #2563EB; font-weight:600; }
QTabBar::tab:hover:!selected { color:#374151; background:#f1f5f9; }

/* === Status Bar (white) === */
QStatusBar { background:#ffffff; color:#64748b; font-size:11px; border-top:1px solid #e2e8f0; padding:0 8px; }
QStatusBar::item { border:none; }

/* === GroupBox === */
QGroupBox { font-weight:600; font-size:11px; border:1px solid #e2e8f0; border-radius:6px; margin-top:8px; padding-top:6px; background:#ffffff; color:#1e293b; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; background:#ffffff; color:#475569; }

/* === Inputs === */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background:#ffffff; color:#1e293b; border:1px solid #e2e8f0; border-radius:5px; padding:3px 6px; font-size:11px; min-height:22px; selection-background-color:#dbeafe; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border-color:#2563EB; }
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover { border-color:#93c5fd; }
QComboBox::drop-down { border:none; width:20px; }
QComboBox QAbstractItemView { background:#ffffff; color:#1e293b; border:1px solid #e2e8f0; border-radius:5px; selection-background-color:#dbeafe; selection-color:#1e293b; padding:2px; }

/* === Buttons === */
QPushButton { background:#f8fafc; color:#374151; border:1px solid #e2e8f0; border-radius:5px; padding:4px 10px; font-size:11px; font-weight:500; min-height:24px; }
QPushButton:hover { background:#f1f5f9; border-color:#bfdbfe; color:#1e293b; }
QPushButton:pressed { background:#dbeafe; border-color:#93c5fd; }
QPushButton:disabled { background:#f1f5f9; color:#94a3b8; border-color:#e2e8f0; }
QPushButton#primary { background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #3b82f6,stop:1 #2563EB); color:#ffffff; border:1px solid #1d4ed8; font-weight:600; }
QPushButton#primary:hover { background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #60a5fa,stop:1 #3b82f6); }
QPushButton#primary:pressed { background:#1d4ed8; }
QPushButton#danger { background:#fef2f2; color:#dc2626; border:1px solid #fca5a5; }
QPushButton#danger:hover { background:#fee2e2; }
QPushButton#success { background:#f0fdf4; color:#16a34a; border:1px solid #86efac; }

/* === Checkboxes === */
QCheckBox { color:#374151; font-size:12px; spacing:8px; padding:2px 0; }
QCheckBox::indicator { width:17px; height:17px; border-radius:5px; border:2px solid #cbd5e1; background:#ffffff; }
QCheckBox::indicator:hover { border-color:#93c5fd; background:#eff6ff; }
QCheckBox::indicator:checked { background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #3b82f6,stop:1 #2563EB); border-color:#2563EB; }

/* === Labels === */
QLabel { color:#374151; font-size:12px; background:transparent; }

/* === Scrollbars === */
QScrollArea { background:transparent; border:none; }
QScrollBar:vertical { background:#f1f5f9; width:7px; border-radius:3px; margin:0; }
QScrollBar::handle:vertical { background:#cbd5e1; border-radius:3px; min-height:22px; }
QScrollBar::handle:vertical:hover { background:#94a3b8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QScrollBar:horizontal { background:#f1f5f9; height:7px; border-radius:3px; margin:0; }
QScrollBar::handle:horizontal { background:#cbd5e1; border-radius:3px; min-width:22px; }
QScrollBar::handle:horizontal:hover { background:#94a3b8; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width:0; }

/* === Splitter === */
QSplitter::handle { background:#e2e8f0; }
QSplitter::handle:horizontal { width:1px; }
QSplitter::handle:vertical { height:1px; }

/* === Tables === */
QTableWidget, QTreeWidget, QListWidget { background:#ffffff; color:#1e293b; border:1px solid #e2e8f0; border-radius:6px; alternate-background-color:#f8fafc; gridline-color:#f1f5f9; font-size:12px; }
QTableWidget::item:selected, QTreeWidget::item:selected, QListWidget::item:selected { background:#dbeafe; color:#1e293b; }
QHeaderView::section { background:#f8fafc; color:#374151; border:none; border-bottom:1px solid #e2e8f0; padding:5px 8px; font-weight:600; font-size:11px; }

/* === Text/Progress === */
QTextEdit, QPlainTextEdit { background:#ffffff; color:#1e293b; border:1px solid #e2e8f0; border-radius:6px; padding:4px; selection-background-color:#dbeafe; }
QProgressBar { background:#f1f5f9; border:1px solid #e2e8f0; border-radius:4px; text-align:center; font-size:10px; color:#64748b; max-height:12px; }
QProgressBar::chunk { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #3b82f6,stop:1 #2563EB); border-radius:3px; }

/* === Tooltips === */
QToolTip { background:#1e293b; color:#f1f5f9; border:1px solid #334155; border-radius:6px; padding:5px 8px; font-size:11px; }

/* === Dialogs === */
QDialog { background:#f8fafc; }
QDialogButtonBox QPushButton { min-width:80px; }

/* === Sliders === */
QSlider::groove:horizontal { height:4px; background:#e2e8f0; border-radius:2px; }
QSlider::handle:horizontal { background:#2563EB; width:14px; height:14px; margin:-5px 0; border-radius:7px; }
QSlider::sub-page:horizontal { background:#2563EB; border-radius:2px; }
"""
