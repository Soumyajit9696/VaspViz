"""
VaspViz — gl_viewer.py
VESTA-style OpenGL POSCAR viewer using QOpenGLWidget.

BLACK SCREEN FIX: Uses QOpenGLExtraFunctions (available in PyQt6 ≥ 6.3)
which wraps the full OpenGL 3.x/4.x API correctly on all platforms.
Falls back to ctypes if unavailable.

SPEED: Uses instanced VAO upload + draw_idle (no redundant repaints).
"""

import sys, math, ctypes
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFileDialog, QMessageBox, QDialog, QDialogButtonBox, QTreeWidget,
    QTreeWidgetItem, QHeaderView, QTableWidget, QTableWidgetItem,
    QTextEdit, QFrame, QSlider, QColorDialog, QAbstractItemView,
    QSizePolicy, QApplication, QGridLayout,
)
from PyQt6.QtCore  import Qt, QTimer, pyqtSignal
from PyQt6.QtGui   import QColor, QFont, QSurfaceFormat, QMatrix4x4, QVector3D, QQuaternion
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram

from constants import (
    ATOM_COLORS, COVALENT_RADII, VDW_RADII, DISPLAY_RADII,
    VALENCE_ELECTRONS, ELEMENT_NUMBERS, VIEWER_SIDE_STYLE,
)
from parsers import PoscarParser

# ─────────────────────────────────────────────────────────────────────────────
#  OpenGL raw function loader (platform-independent, no PyOpenGL needed)
# ─────────────────────────────────────────────────────────────────────────────

def _load_gl():
    """
    Build a GL function wrapper that works on Windows / Linux / macOS with PyQt6.

    Strategy:
    1. Try QOpenGLFunctions_3_3_Core via QOpenGLVersionFunctionsFactory (PyQt6 ≥ 6.4)
    2. Try QOpenGLExtraFunctions (no-arg constructor in PyQt6)
    3. Fall back to ctypes — tries DLL directly then getProcAddress.
       On Windows, core functions (glClear etc.) live in opengl32.dll and are NOT
       returned by getProcAddress, so we must query the DLL directly.
    """
    from PyQt6.QtGui import QOpenGLContext

    ctx = QOpenGLContext.currentContext()
    if ctx is None:
        raise RuntimeError("No current OpenGL context")

    # ── Strategy 1: versioned functions factory ──────────────────────────────
    try:
        from PyQt6.QtOpenGL import QOpenGLVersionFunctionsFactory, QOpenGLVersionProfile
        vp = QOpenGLVersionProfile()
        vp.setVersion(3, 3)
        vp.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        funcs = QOpenGLVersionFunctionsFactory.get(vp, ctx)
        if funcs is not None:
            funcs.initializeOpenGLFunctions()

            class _V33:
                def __init__(self, f): self._f = f
                def glEnable(self,c):              self._f.glEnable(c)
                def glDisable(self,c):             self._f.glDisable(c)
                def glDepthFunc(self,f):           self._f.glDepthFunc(f)
                def glBlendFunc(self,s,d):         self._f.glBlendFunc(s,d)
                def glClearColor(self,r,g,b,a):    self._f.glClearColor(r,g,b,a)
                def glClear(self,m):               self._f.glClear(m)
                def glViewport(self,x,y,w,h):      self._f.glViewport(x,y,w,h)
                def glLineWidth(self,w):           self._f.glLineWidth(w)
                def glDrawArrays(self,m,f,c):      self._f.glDrawArrays(m,f,c)
                def glGenVertexArrays(self,n):     return self._f.glGenVertexArrays(n)
                def glBindVertexArray(self,v):     self._f.glBindVertexArray(v)
                def glDeleteVertexArrays(self,n,a):self._f.glDeleteVertexArrays(n,a)
                def glGenBuffers(self,n):          return self._f.glGenBuffers(n)
                def glBindBuffer(self,t,b):        self._f.glBindBuffer(t,b)
                def glBufferData(self,t,s,d,u):    self._f.glBufferData(t,s,d,u)
                def glEnableVertexAttribArray(self,i):        self._f.glEnableVertexAttribArray(i)
                def glVertexAttribPointer(self,i,n,t,nm,s,p):self._f.glVertexAttribPointer(i,n,t,nm,s,p)
            return _V33(funcs)
    except Exception as e:
        print(f"[GL] QOpenGLVersionFunctionsFactory: {e}")

    # ── Strategy 2: QOpenGLExtraFunctions (no-arg constructor) ──────────────
    try:
        from PyQt6.QtOpenGL import QOpenGLExtraFunctions
        ef = QOpenGLExtraFunctions()   # PyQt6: no argument!
        ef.initializeOpenGLFunctions()

        class _EF:
            def __init__(self, f): self._f = f
            def glEnable(self,c):              self._f.glEnable(c)
            def glDisable(self,c):             self._f.glDisable(c)
            def glDepthFunc(self,f):           self._f.glDepthFunc(f)
            def glBlendFunc(self,s,d):         self._f.glBlendFunc(s,d)
            def glClearColor(self,r,g,b,a):    self._f.glClearColor(r,g,b,a)
            def glClear(self,m):               self._f.glClear(m)
            def glViewport(self,x,y,w,h):      self._f.glViewport(x,y,w,h)
            def glLineWidth(self,w):           self._f.glLineWidth(w)
            def glDrawArrays(self,m,f,c):      self._f.glDrawArrays(m,f,c)
            def glGenVertexArrays(self,n):     return self._f.glGenVertexArrays(n)
            def glBindVertexArray(self,v):     self._f.glBindVertexArray(v)
            def glDeleteVertexArrays(self,n,a):self._f.glDeleteVertexArrays(n,a)
            def glGenBuffers(self,n):          return self._f.glGenBuffers(n)
            def glBindBuffer(self,t,b):        self._f.glBindBuffer(t,b)
            def glBufferData(self,t,s,d,u):    self._f.glBufferData(t,s,d,u)
            def glEnableVertexAttribArray(self,i):        self._f.glEnableVertexAttribArray(i)
            def glVertexAttribPointer(self,i,n,t,nm,s,p):self._f.glVertexAttribPointer(i,n,t,nm,s,p)
        return _EF(ef)
    except Exception as e:
        print(f"[GL] QOpenGLExtraFunctions: {e}")

    # ── Strategy 3: ctypes (works everywhere, handles Windows core funcs) ───
    # Load platform GL library
    _lib = None
    if sys.platform.startswith("win"):
        for dll in ("opengl32", "opengl32.dll"):
            try: _lib = ctypes.WinDLL(dll); break
            except: pass
    elif sys.platform.startswith("darwin"):
        for p in ("/System/Library/Frameworks/OpenGL.framework/OpenGL",
                  "/System/Library/Frameworks/OpenGL.framework/Versions/A/OpenGL"):
            try: _lib = ctypes.CDLL(p); break
            except: pass
    else:
        for name in ("libGL.so.1","libGL.so","libGL.so.0","libGL"):
            try: _lib = ctypes.CDLL(name); break
            except: pass

    if _lib is None:
        raise RuntimeError("Could not load OpenGL library on this platform")

    F    = ctypes.c_float;  U = ctypes.c_uint;  I = ctypes.c_int
    UB   = ctypes.c_ubyte;  SZ = ctypes.c_ssize_t;  VP = ctypes.c_void_p

    _SIGS = {
        "glEnable":                  (None,[U]),
        "glDisable":                 (None,[U]),
        "glDepthFunc":               (None,[U]),
        "glBlendFunc":               (None,[U,U]),
        "glClearColor":              (None,[F,F,F,F]),
        "glClear":                   (None,[U]),
        "glViewport":                (None,[I,I,I,I]),
        "glLineWidth":               (None,[F]),
        "glDrawArrays":              (None,[U,I,I]),
        "glGenVertexArrays":         (None,[I,ctypes.POINTER(U)]),
        "glBindVertexArray":         (None,[U]),
        "glDeleteVertexArrays":      (None,[I,ctypes.POINTER(U)]),
        "glGenBuffers":              (None,[I,ctypes.POINTER(U)]),
        "glBindBuffer":              (None,[U,U]),
        "glBufferData":              (None,[U,SZ,VP,U]),
        "glEnableVertexAttribArray": (None,[U]),
        "glVertexAttribPointer":     (None,[U,I,U,UB,I,VP]),
    }

    class _CT:
        pass
    gl = _CT()

    def _get_fn(name, restype, argtypes):
        # 1) try DLL directly (works for core on Windows)
        fn = getattr(_lib, name, None)
        if fn:
            fn.restype = restype
            fn.argtypes = argtypes
            return fn
        # 2) try getProcAddress (extension functions)
        addr = ctx.getProcAddress(name.encode())
        if addr:
            # PyQt6 returns sip.voidptr — must convert to int for ctypes
            addr_int = int(addr)
            if addr_int:
                proto = ctypes.CFUNCTYPE(restype, *argtypes)
                return proto(addr_int)
        return None

    for name, (restype, argtypes) in _SIGS.items():
        fn = _get_fn(name, restype, argtypes)
        if fn:
            setattr(gl, name, fn)
        else:
            print(f"[GL] WARNING: could not resolve {name}")
            setattr(gl, name, lambda *a, _n=name: None)

    # Wrap glGenVertexArrays/glGenBuffers to return int like the PyQt6 wrappers
    _orig_gva = gl.glGenVertexArrays
    _orig_gb  = gl.glGenBuffers

    def _gen_vao(n, _f=_orig_gva):
        ids = (U * n)()
        _f(n, ids)
        return ids[0] if n == 1 else list(ids)

    def _gen_buf(n, _f=_orig_gb):
        ids = (U * n)()
        _f(n, ids)
        return ids[0] if n == 1 else list(ids)

    gl.glGenVertexArrays = _gen_vao
    gl.glGenBuffers      = _gen_buf

    return gl


# ─────────────────────────────────────────────────────────────────────────────
#  GLSL shaders — sphere & cylinder impostors + wireframe lines
# ─────────────────────────────────────────────────────────────────────────────

_SPHERE_VERT = """
#version 330 core
layout(location=0) in vec3  a_center;
layout(location=1) in float a_radius;
layout(location=2) in vec3  a_color;
layout(location=3) in float a_selected;

uniform mat4 u_MV;
uniform mat4 u_P;

out vec3  v_center_eye;
out float v_radius;
out vec3  v_color;
out float v_selected;
out vec2  v_uv;

const vec2 C[4] = vec2[4](vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,1));

void main(){
    v_uv=C[gl_VertexID%4]; v_radius=a_radius; v_color=a_color; v_selected=a_selected;
    vec4 ce=u_MV*vec4(a_center,1.0);
    v_center_eye=ce.xyz;
    vec4 p=ce; p.xy+=v_uv*a_radius*1.05;
    gl_Position=u_P*p;
}
"""

_SPHERE_FRAG = """
#version 330 core
in vec3  v_center_eye;
in float v_radius;
in vec3  v_color;
in float v_selected;
in vec2  v_uv;

uniform mat4  u_P;
uniform vec3  u_light;
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_specular;
uniform float u_shininess;
uniform vec3  u_bg;
uniform bool  u_fog;
uniform float u_fog_near;
uniform float u_fog_far;

out vec4 fragColor;

void main(){
    float r2=v_radius*v_radius;
    float d2=dot(v_uv,v_uv);
    if(d2>1.0) discard;

    float zlift=sqrt(max(0.0, r2-d2*r2));
    vec3 hit=vec3(v_center_eye.xy + v_uv*v_radius, v_center_eye.z+zlift);
    vec3 normal=normalize(hit-v_center_eye);

    vec4 clip=u_P*vec4(hit,1.0);
    gl_FragDepth=(clip.z/clip.w+1.0)*0.5;

    vec3 L=normalize(u_light);
    vec3 V=vec3(0.0, 0.0, 1.0);
    vec3 H=normalize(L+V);
    float diff=max(dot(normal,L),0.0);
    float spec=pow(max(dot(normal,H),0.0),u_shininess);
    vec3 col=v_color*(u_ambient+u_diffuse*diff)
            +vec3(u_specular*spec);

    if(v_selected>0.5){
        float rv=sqrt(d2);
        float ring=smoothstep(0.0,0.04,abs(rv-0.88));
        col=mix(vec3(1.0,0.85,0.1),col,ring);
    }
    if(u_fog){
        float f=clamp((length(hit)-u_fog_near)/max(u_fog_far-u_fog_near,0.01),0.0,1.0);
        col=mix(col,u_bg,f*0.55);
    }
    fragColor=vec4(col,1.0);
}
"""

_BOND_VERT = """
#version 330 core
layout(location=0) in vec3  a_p1;
layout(location=1) in vec3  a_p2;
layout(location=2) in vec3  a_color;
layout(location=3) in float a_radius;

uniform mat4 u_MV;
uniform mat4 u_P;

out vec3  v_color;
out float v_u;

const vec2 C[4]=vec2[4](vec2(-1,0),vec2(1,0),vec2(1,1),vec2(-1,1));

void main(){
    vec2 c=C[gl_VertexID%4];
    v_color=a_color; v_u=c.x;
    vec3 e1=(u_MV*vec4(a_p1,1.0)).xyz;
    vec3 e2=(u_MV*vec4(a_p2,1.0)).xyz;
    vec3 axis=normalize(e2-e1);
    vec3 ref=abs(axis.z)<0.9?vec3(0,0,1):vec3(0,1,0);
    vec3 side=normalize(cross(axis,ref));
    vec3 base=mix(e1,e2,c.y)+c.x*side*a_radius;
    gl_Position=u_P*vec4(base,1.0);
}
"""

_BOND_FRAG = """
#version 330 core
in vec3  v_color;
in float v_u;

uniform vec3  u_light;
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_specular;
uniform float u_shininess;

out vec4 fragColor;

void main(){
    if(abs(v_u)>1.0) discard;
    float cosA=sqrt(max(0.0,1.0-v_u*v_u));
    vec3 normal=vec3(v_u,cosA,0.0);
    vec3 L=normalize(u_light);
    vec3 H=normalize(L+vec3(0,0,1));
    float diff=max(dot(normal,L),0.0);
    float spec=pow(max(dot(normal,H),0.0),u_shininess);
    vec3 col=v_color*(u_ambient+u_diffuse*diff)+vec3(u_specular*spec);
    fragColor=vec4(col,1.0);
}
"""

_LINE_VERT = """
#version 330 core
layout(location=0) in vec3 a_pos;
uniform mat4 u_MVP;
void main(){gl_Position=u_MVP*vec4(a_pos,1.0);}
"""
_LINE_FRAG = """
#version 330 core
uniform vec4 u_color;
out vec4 fragColor;
void main(){fragColor=u_color;}
"""

# GL constants
GL_DEPTH_TEST       = 0x0B71
GL_LESS             = 0x0201
GL_MULTISAMPLE      = 0x809D
GL_BLEND            = 0x0BE2
GL_SRC_ALPHA        = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_COLOR_BUFFER_BIT = 0x4000
GL_DEPTH_BUFFER_BIT = 0x0100
GL_FLOAT            = 0x1406
GL_ARRAY_BUFFER     = 0x8892
GL_DYNAMIC_DRAW     = 0x88E8
GL_TRIANGLE_FAN     = 0x0006
GL_LINES            = 0x0001


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16)/255.0 for i in (0,2,4))

def _detect_bonds(pos, labels, scale=1.15):
    bonds = []
    n = len(pos)
    for i in range(n):
        ri = COVALENT_RADII.get(labels[i], COVALENT_RADII["default"])
        for j in range(i+1,n):
            rj = COVALENT_RADII.get(labels[j], COVALENT_RADII["default"])
            d  = float(np.linalg.norm(pos[i]-pos[j]))
            if 0.4 < d < scale*(ri+rj):
                bonds.append((i,j,d))
    return bonds

def _cell_lines(lattice):
    lat = np.asarray(lattice, np.float32)
    a,b,c = lat[0],lat[1],lat[2]
    V = {(0,0,0):np.zeros(3,np.float32),(1,0,0):a,(0,1,0):b,(0,0,1):c,
         (1,1,0):a+b,(1,0,1):a+c,(0,1,1):b+c,(1,1,1):a+b+c}
    edges=[(0,0,0),(1,0,0),(0,0,0),(0,1,0),(0,0,0),(0,0,1),
           (1,0,0),(1,1,0),(1,0,0),(1,0,1),(0,1,0),(1,1,0),
           (0,1,0),(0,1,1),(0,0,1),(1,0,1),(0,0,1),(0,1,1),
           (1,1,0),(1,1,1),(1,0,1),(1,1,1),(0,1,1),(1,1,1)]
    pts = []
    for i in range(0,len(edges),2):
        pts.append(V[edges[i]]); pts.append(V[edges[i+1]])
    return np.vstack(pts).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Arcball camera
# ─────────────────────────────────────────────────────────────────────────────

class ArcballCamera:
    def __init__(self):
        self.rotation = QQuaternion()
        self.zoom     = 15.0
        self.pan      = QVector3D()
        self._last    = None
        self._mode    = None

    def reset(self, r=5.0):
        self.rotation = QQuaternion(); self.zoom = r*2.8; self.pan = QVector3D()

    def press(self, pos, btn):
        self._last = pos
        self._mode = ("rotate" if btn==Qt.MouseButton.LeftButton else
                      "pan"    if btn==Qt.MouseButton.MiddleButton else None)

    def move(self, pos, h):
        if not self._last: return
        dx,dy = pos.x()-self._last.x(), pos.y()-self._last.y()
        if self._mode=="rotate":
            q = QQuaternion.fromEulerAngles(dy*0.4, dx*0.4, 0)
            self.rotation = q*self.rotation; self.rotation.normalize()
        elif self._mode=="pan":
            s = self.zoom/max(h,1)
            self.pan += QVector3D(-dx*s, dy*s, 0)
        self._last = pos

    def release(self): self._last=None; self._mode=None

    def scroll(self, d):
        self.zoom = max(0.3, min(500.0, self.zoom*(0.88 if d>0 else 1.14)))

    def view(self):
        m=QMatrix4x4(); m.translate(self.pan); m.translate(0,0,-self.zoom); m.rotate(self.rotation); return m

    def preset(self, az, el):
        self.rotation = (QQuaternion.fromAxisAndAngle(0,1,0,az)*
                         QQuaternion.fromAxisAndAngle(1,0,0,el))
        self.rotation.normalize()


# ─────────────────────────────────────────────────────────────────────────────
#  GL Widget
# ─────────────────────────────────────────────────────────────────────────────

class StructureGLWidget(QOpenGLWidget):
    atom_picked = pyqtSignal(int)

    def __init__(self, parent=None):
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        fmt.setVersion(3,3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        QSurfaceFormat.setDefaultFormat(fmt)
        super().__init__(parent)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400,300)

        # scene state
        self._pos    = np.zeros((0,3), np.float32)
        self._labels = []
        self._radii  = np.zeros(0, np.float32)
        self._colors = np.zeros((0,3), np.float32)
        self._bonds  = []
        self._cell   = np.zeros((0,3), np.float32)
        self._center = np.zeros(3)
        self._scene_r= 5.0
        self._selected = set()

        # settings
        self.render_mode  = 0   # 0=Ball&Stick 1=SpaceFill 2=Stick 3=Wire
        self.show_cell    = True
        self.show_bonds   = True
        self.show_axes    = True
        self.fog_enabled  = False
        self.bg           = (0.08, 0.08, 0.14)
        self.ambient      = 0.25
        self.diffuse      = 0.70
        self.specular     = 0.45
        self.shininess    = 52.0
        self.radius_scale = 1.0
        self.bond_radius  = 0.10
        self.custom_colors= {}
        self.custom_radii = {}

        self.camera = ArcballCamera()
        self._timer = QTimer(self); self._timer.timeout.connect(self._spin)
        self._spin_angle = 0.0

        self._ready  = False
        self._gl     = None
        self._progs  = {}
        # Raw VAO/VBO ids (uint)
        self._vao_sphere = self._vao_bond = self._vao_cell = 0
        self._vbo_sphere = self._vbo_bond = self._vbo_cell = 0
        self._vao_axes = self._vbo_axes = 0
        self._n_sphere = self._n_bond = self._n_cell = 0
        self._pending_scene = None  # deferred upload when GL not ready yet

    # ── GL lifecycle ────────────────────────────────────────────────────
    def initializeGL(self):
        try:
            self._gl = _load_gl()
        except Exception as e:
            print(f"[GL] init failed: {e}"); return

        gl = self._gl
        gl.glEnable(GL_DEPTH_TEST)
        gl.glDepthFunc(GL_LESS)
        try: gl.glEnable(GL_MULTISAMPLE)
        except: pass
        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._compile()
        self._alloc_vaos()
        self._up_axes()
        self._ready = True

        # Upload any scene that was set before GL was ready
        if self._pending_scene is not None:
            pos, labels, lattice, bond_scale = self._pending_scene
            self._pending_scene = None
            self.set_scene(pos, labels, lattice, bond_scale)

    def _compile(self):
        def mk(vs,fs):
            p=QOpenGLShaderProgram(self)
            ok1=p.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex,   vs)
            ok2=p.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs)
            if not ok1 or not ok2: print("[shader]",p.log())
            if not p.link():       print("[link]",  p.log())
            return p
        self._progs = {
            "sphere": mk(_SPHERE_VERT, _SPHERE_FRAG),
            "bond":   mk(_BOND_VERT,   _BOND_FRAG),
            "line":   mk(_LINE_VERT,   _LINE_FRAG),
        }

    def _alloc_vaos(self):
        """Allocate VAO and VBO ids. Works with both PyQt6 wrappers and ctypes."""
        gl = self._gl
        try:
            self._vao_sphere = int(gl.glGenVertexArrays(1))
            self._vao_bond   = int(gl.glGenVertexArrays(1))
            self._vao_cell   = int(gl.glGenVertexArrays(1))
            self._vao_axes   = int(gl.glGenVertexArrays(1))
            self._vbo_sphere = int(gl.glGenBuffers(1))
            self._vbo_bond   = int(gl.glGenBuffers(1))
            self._vbo_cell   = int(gl.glGenBuffers(1))
            self._vbo_axes   = int(gl.glGenBuffers(1))
        except Exception as e:
            print(f"[GL] _alloc_vaos error: {e}")

    def _up_axes(self):
        """Upload XYZ axis lines for the gizmo (6 vertices = 3 line segments)."""
        L = 1.0  # axis length
        data = np.array([
            [0,0,0], [L,0,0],   # X axis
            [0,0,0], [0,L,0],   # Y axis
            [0,0,0], [0,0,L],   # Z axis
        ], np.float32)
        self._upload_vao(self._vao_axes, self._vbo_axes, data,
                         self._progs["line"], [(0,3,0)])

    def resizeGL(self,w,h):
        if self._gl: self._gl.glViewport(0,0,w,h)

    # ── scene upload ───────────────────────────────────────────────────
    def set_scene(self, pos, labels, lattice=None, bond_scale=1.15):
        if len(pos)==0:
            self._n_sphere=self._n_bond=self._n_cell=0; self.update(); return

        self._pos    = np.asarray(pos, np.float32)
        self._labels = list(labels)
        self._selected.clear()
        self._center = self._pos.mean(axis=0)
        d = np.linalg.norm(self._pos-self._center, axis=1)
        self._scene_r = max(float(d.max())*1.1, 3.0)
        self.camera.reset(self._scene_r)

        self._radii  = np.array([self._atom_r(l) for l in labels], np.float32)
        self._colors = np.array([self._atom_c(l) for l in labels], np.float32)
        self._bonds  = _detect_bonds(self._pos, self._labels, bond_scale)
        self._cell   = _cell_lines(lattice) if lattice is not None else np.zeros((0,3),np.float32)

        if self._ready:
            self.makeCurrent()
            self._up_spheres(); self._up_bonds(); self._up_cell()
            self.doneCurrent()
        self.update()

    def _atom_r(self,e):
        return self.custom_radii.get(e, DISPLAY_RADII.get(e,DISPLAY_RADII["default"]))*self.radius_scale

    def _atom_c(self,e):
        if e in self.custom_colors: return np.array(self.custom_colors[e],np.float32)
        return np.array(_hex_to_rgb(ATOM_COLORS.get(e,ATOM_COLORS["default"])),np.float32)

    def _eff_radii(self):
        if self.render_mode==1:
            return np.array([VDW_RADII.get(l,VDW_RADII["default"])*self.radius_scale
                             for l in self._labels],np.float32)
        if self.render_mode==2: return np.full(len(self._labels),0.06,np.float32)
        if self.render_mode==3: return np.zeros(len(self._labels),np.float32)
        return self._radii

    def _up_spheres(self):
        N = len(self._pos)
        if N==0: self._n_sphere=0; return
        sel   = np.array([1.0 if i in self._selected else 0.0 for i in range(N)],np.float32)
        radii = self._eff_radii()
        ctr   = self._center
        # layout per vertex: center(3) radius(1) color(3) sel(1) = 8f
        data = np.zeros((N,8),np.float32)
        data[:,0:3]=self._pos-ctr; data[:,3]=radii
        data[:,4:7]=self._colors; data[:,7]=sel
        quad = np.repeat(data,4,axis=0).astype(np.float32)
        self._upload_vao(self._vao_sphere, self._vbo_sphere, quad,
                         self._progs["sphere"],
                         [(0,3,0),(1,1,3),(2,3,4),(3,1,7)])
        self._n_sphere = N*4

    def _up_bonds(self):
        bonds=(self._bonds if (self.show_bonds and self.render_mode != 3) else [])
        if not bonds: self._n_bond=0; return
        ctr=self._center; br=self.bond_radius
        rows=[]
        for i,j,_ in bonds:
            p1=self._pos[i]-ctr; p2=self._pos[j]-ctr; mid=(p1+p2)*0.5
            c1=self._colors[i]; c2=self._colors[j]
            rows.append(np.concatenate([p1,mid,c1,[br]]))
            rows.append(np.concatenate([mid,p2,c2,[br]]))
        data=np.array(rows,np.float32)
        quad=np.repeat(data,4,axis=0).astype(np.float32)
        self._upload_vao(self._vao_bond,self._vbo_bond,quad,
                         self._progs["bond"],
                         [(0,3,0),(1,3,3),(2,3,6),(3,1,9)])
        self._n_bond=len(rows)*4

    def _up_cell(self):
        if not self.show_cell or len(self._cell)==0:
            self._n_cell=0; return
        data=(self._cell-self._center).astype(np.float32)
        self._upload_vao(self._vao_cell,self._vbo_cell,data,
                         self._progs["line"],[(0,3,0)])
        self._n_cell=len(data)

    def _upload_vao(self, vao_id, vbo_id, data, prog, attrs):
        """Upload float32 numpy array into VAO/VBO and set attrib pointers."""
        gl     = self._gl
        n_f    = data.shape[1] if data.ndim == 2 else 3
        stride = n_f * 4          # bytes per vertex
        gl.glBindVertexArray(vao_id)
        gl.glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        raw = data.tobytes()
        gl.glBufferData(GL_ARRAY_BUFFER, len(raw), raw, GL_DYNAMIC_DRAW)
        prog.bind()
        for loc, count, offset in attrs:
            gl.glEnableVertexAttribArray(loc)
            # normalized must be 0 (GL_FALSE), not Python bool False,
            # because ctypes / PyQt6 wrappers differ in how they handle it.
            gl.glVertexAttribPointer(
                loc, count, GL_FLOAT,
                0,                          # GL_FALSE — never normalise floats
                stride,
                ctypes.c_void_p(offset * 4)
            )
        prog.release()
        gl.glBindBuffer(GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    # ── render ─────────────────────────────────────────────────────────
    def paintGL(self):
        if not self._ready: return
        gl=self._gl
        bg=self.bg
        gl.glClearColor(bg[0],bg[1],bg[2],1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        w,h=self.width(),self.height()
        if w==0 or h==0: return

        half_h = self.camera.zoom * 0.3153
        half_w = half_h * w / max(h, 1)
        P=QMatrix4x4(); P.ortho(-half_w, half_w, -half_h, half_h, 0.01, 2000.0)
        MV=self.camera.view(); MVP=P*MV

        # light
        lx,ly,lz=0.5,0.8,1.0; n=math.sqrt(lx*lx+ly*ly+lz*lz)
        lx,ly,lz=lx/n,ly/n,lz/n
        fn=self._scene_r*0.8; ff=self._scene_r*4.0

        # bonds
        if self._n_bond>0:
            p=self._progs["bond"]; p.bind()
            p.setUniformValue("u_MV",MV); p.setUniformValue("u_P",P)
            p.setUniformValue("u_light",lx,ly,lz)
            p.setUniformValue("u_ambient",self.ambient*0.9)
            p.setUniformValue("u_diffuse",self.diffuse)
            p.setUniformValue("u_specular",self.specular*0.3)
            p.setUniformValue("u_shininess",self.shininess)
            gl.glBindVertexArray(self._vao_bond)
            for q in range(self._n_bond//4): gl.glDrawArrays(GL_TRIANGLE_FAN,q*4,4)
            gl.glBindVertexArray(0); p.release()

        # spheres
        if self._n_sphere>0 and self.render_mode!=3:
            p=self._progs["sphere"]; p.bind()
            p.setUniformValue("u_MV",MV); p.setUniformValue("u_P",P)
            p.setUniformValue("u_light",lx,ly,lz)
            p.setUniformValue("u_ambient",self.ambient)
            p.setUniformValue("u_diffuse",self.diffuse)
            p.setUniformValue("u_specular",self.specular)
            p.setUniformValue("u_shininess",self.shininess)
            p.setUniformValue("u_bg",bg[0],bg[1],bg[2])
            p.setUniformValue("u_fog",self.fog_enabled)
            p.setUniformValue("u_fog_near",fn); p.setUniformValue("u_fog_far",ff)
            gl.glBindVertexArray(self._vao_sphere)
            for q in range(self._n_sphere//4): gl.glDrawArrays(GL_TRIANGLE_FAN,q*4,4)
            gl.glBindVertexArray(0); p.release()

        # cell
        if self._n_cell>0 and self.show_cell:
            p=self._progs["line"]; p.bind()
            p.setUniformValue("u_MVP",MVP)
            r,g,b=(0.55,0.70,0.95) if bg[0]<0.5 else (0.3,0.4,0.6)
            p.setUniformValue("u_color",r,g,b,0.9)
            gl.glBindVertexArray(self._vao_cell)
            gl.glLineWidth(1.8)
            gl.glDrawArrays(GL_LINES,0,self._n_cell)
            gl.glBindVertexArray(0); p.release()

        # ── XYZ axes gizmo (bottom-left corner) ─────────────────────────
        if self.show_axes and self._vao_axes:
            gizmo_size = min(w, h) // 6   # pixel size of gizmo viewport
            margin = 10
            gl.glViewport(margin, margin, gizmo_size, gizmo_size)
            gl.glClear(GL_DEPTH_BUFFER_BIT)  # clear depth so axes are on top

            # Build MVP: rotation only (no pan/zoom), fixed distance
            Pa = QMatrix4x4(); Pa.ortho(-1.6, 1.6, -1.6, 1.6, 0.01, 100.0)
            MVa = QMatrix4x4(); MVa.translate(0, 0, -3.5)
            MVa.rotate(self.camera.rotation)
            MVPa = Pa * MVa

            p = self._progs["line"]; p.bind()
            gl.glBindVertexArray(self._vao_axes)
            gl.glLineWidth(2.5)
            # X axis — red
            p.setUniformValue("u_MVP", MVPa)
            p.setUniformValue("u_color", 0.95, 0.2, 0.2, 1.0)
            gl.glDrawArrays(GL_LINES, 0, 2)
            # Y axis — green
            p.setUniformValue("u_color", 0.2, 0.8, 0.2, 1.0)
            gl.glDrawArrays(GL_LINES, 2, 2)
            # Z axis — blue
            p.setUniformValue("u_color", 0.3, 0.4, 0.95, 1.0)
            gl.glDrawArrays(GL_LINES, 4, 2)
            gl.glBindVertexArray(0); p.release()

            # Restore main viewport
            gl.glViewport(0, 0, w, h)

            # ── QPainter overlay for axis labels ─────────────────────────
            from PyQt6.QtGui import QPainter as _QP, QPen as _QPen
            painter = _QP(self)
            painter.setRenderHint(_QP.RenderHint.Antialiasing)
            font = QFont("Segoe UI", 11, QFont.Weight.Bold)
            painter.setFont(font)

            # Project each axis tip through the gizmo MVP to screen coords
            tips = [(QVector3D(1,0,0), "a", QColor(230,50,50)),
                    (QVector3D(0,1,0), "b", QColor(50,190,50)),
                    (QVector3D(0,0,1), "c", QColor(70,100,230))]
            for tip3d, label, color in tips:
                ndc = MVPa.map(tip3d)
                # NDC → pixel in gizmo viewport
                sx = margin + (ndc.x() * 0.5 + 0.5) * gizmo_size
                sy = h - margin - (ndc.y() * 0.5 + 0.5) * gizmo_size
                painter.setPen(_QPen(color, 2))
                painter.drawText(int(sx) + 2, int(sy) - 2, label)
            painter.end()

    # ── interaction ─────────────────────────────────────────────────────
    def mousePressEvent(self,ev):
        self.camera.press(ev.pos(),ev.button())
        if ev.button()==Qt.MouseButton.LeftButton:
            idx=self._pick(ev.pos())
            if idx>=0:
                if idx in self._selected: self._selected.discard(idx)
                else: self._selected.add(idx)
                self.atom_picked.emit(idx)
                if self._ready:
                    self.makeCurrent(); self._up_spheres(); self.doneCurrent()
        self.update()

    def mouseMoveEvent(self,ev):
        self.camera.move(ev.pos(),self.height()); self.update()
    def mouseReleaseEvent(self,ev): self.camera.release()
    def wheelEvent(self,ev): self.camera.scroll(ev.angleDelta().y()); self.update()

    def keyPressEvent(self,ev):
        k=ev.key()
        if   k==Qt.Key.Key_R: self.camera.reset(self._scene_r)
        elif k==Qt.Key.Key_T: self.toggle_spin()
        elif k==Qt.Key.Key_1: self.camera.preset(0,90)
        elif k==Qt.Key.Key_2: self.camera.preset(0,0)
        elif k==Qt.Key.Key_3: self.camera.preset(90,0)
        elif k==Qt.Key.Key_4: self.camera.preset(-45,25)
        elif k==Qt.Key.Key_Escape:
            self._selected.clear()
            if self._ready: self.makeCurrent(); self._up_spheres(); self.doneCurrent()
        self.update()

    def _pick(self,screen_pos):
        if not len(self._pos): return -1
        w,h=self.width(),self.height()
        nx=(2.0*screen_pos.x()/w)-1.0; ny=-(2.0*screen_pos.y()/h)+1.0
        half_h = self.camera.zoom * 0.3153
        half_w = half_h * w / max(h, 1)
        P=QMatrix4x4(); P.ortho(-half_w, half_w, -half_h, half_h, 0.01, 2000.0)
        inv,ok=(P*self.camera.view()).inverted()
        if not ok: return -1
        near4=inv.map(QVector3D(nx,ny,-1)); far4=inv.map(QVector3D(nx,ny,1))
        ro=np.array([near4.x(),near4.y(),near4.z()])
        rd=np.array([far4.x()-ro[0],far4.y()-ro[1],far4.z()-ro[2]])
        rd/=np.linalg.norm(rd)+1e-12
        pos=self._pos-self._center; radii=self._eff_radii()
        bt,bi=np.inf,-1
        for i,(p,r) in enumerate(zip(pos,radii)):
            if r<0.01: continue
            oc=ro-p; b=2*np.dot(oc,rd); c=np.dot(oc,oc)-r*r; d=b*b-4*c
            if d<0: continue
            t=(-b-math.sqrt(d))*0.5
            if 0.01<t<bt: bt=t; bi=i
        return bi

    def toggle_spin(self):
        if self._timer.isActive(): self._timer.stop()
        else: self._timer.start(16)
    def stop_spin(self): self._timer.stop()
    def _spin(self): self._spin_angle+=0.4; self.camera.preset(self._spin_angle,20); self.update()

    def refresh(self):
        if not self._ready: return
        self.makeCurrent(); self._up_spheres(); self._up_bonds(); self._up_cell()
        self.doneCurrent(); self.update()

    def clear_sel(self):
        self._selected.clear()
        if self._ready: self.makeCurrent(); self._up_spheres(); self.doneCurrent()
        self.update()

    def set_bg(self,r,g,b): self.bg=(r,g,b); self.update()
    def screenshot(self): return self.grabFramebuffer()


# ─────────────────────────────────────────────────────────────────────────────
#  Full PoscarViewerWidget
# ─────────────────────────────────────────────────────────────────────────────

class PoscarViewerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.data=None; self._meas=[]; self._ecolors={}; self._eradii={}
        self._bscale=1.15
        self._build()

    def _build(self):
        root=QHBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # ── LEFT SCROLL PANEL ─────────────────────────────────────────────
        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(295)
        scroll.setStyleSheet("QScrollArea{background:#1a1a2e;border:none;}"
            "QScrollBar:vertical{background:#16213e;width:7px;border-radius:3px;}"
            "QScrollBar::handle:vertical{background:#3d3d6b;border-radius:3px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}")

        lw=QWidget(); lw.setStyleSheet(VIEWER_SIDE_STYLE)
        ll=QVBoxLayout(lw); ll.setContentsMargins(10,12,10,18); ll.setSpacing(6)

        btn=QPushButton("📂  Open Structure File"); btn.setObjectName("primary")
        btn.clicked.connect(self._open); ll.addWidget(btn)

        self.lbl_pkgs=QLabel(); self._upd_pkgs(); ll.addWidget(self.lbl_pkgs)

        # Info tree
        g=QGroupBox("Structure Info"); gl_=QVBoxLayout(g)
        self.info_tree=QTreeWidget(); self.info_tree.setHeaderLabels(["Property","Value"])
        self.info_tree.setAlternatingRowColors(True)
        self.info_tree.setMinimumHeight(165); self.info_tree.setMaximumHeight(195)
        self.info_tree.header().setSectionResizeMode(0,QHeaderView.ResizeMode.ResizeToContents)
        self.info_tree.header().setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch)
        gl_.addWidget(self.info_tree)
        self.lbl_elec=QLabel("—"); self.lbl_elec.setWordWrap(True)
        self.lbl_elec.setStyleSheet("font-size:11px;padding:5px;background:#1c3520;"
            "border-radius:5px;border:1px solid #3a6640;color:#86efac;font-family:monospace;")
        gl_.addWidget(self.lbl_elec); ll.addWidget(g)

        # Symmetry
        g2=QGroupBox("Symmetry (spglib)"); gl2=QVBoxLayout(g2)
        self.lbl_sg=QLabel("Space group: —")
        self.lbl_sg.setStyleSheet("font-size:12px;font-weight:700;color:#7eb8f7;")
        gl2.addWidget(self.lbl_sg)
        r=QHBoxLayout()
        for t,f in [("Find",self._find_sym),("Prim",self._prim),("Conv",self._conv)]:
            b=QPushButton(t); b.clicked.connect(f); r.addWidget(b)
        gl2.addLayout(r); ll.addWidget(g2)

        # Render
        g3=QGroupBox("Render Style"); gl3=QVBoxLayout(g3); gl3.setSpacing(5)
        row=QHBoxLayout(); row.addWidget(QLabel("Style:"))
        self.mode_cb=QComboBox()
        self.mode_cb.addItems(["Ball & Stick","Space Fill","Stick Only","Wireframe"])
        self.mode_cb.currentIndexChanged.connect(self._on_mode); row.addWidget(self.mode_cb)
        gl3.addLayout(row)
        row2=QHBoxLayout(); row2.addWidget(QLabel("Supercell:"))
        self.sp_a=QSpinBox(); self.sp_a.setRange(1,5); self.sp_a.setValue(1); self.sp_a.setFixedWidth(44)
        self.sp_b=QSpinBox(); self.sp_b.setRange(1,5); self.sp_b.setValue(1); self.sp_b.setFixedWidth(44)
        self.sp_c_=QSpinBox(); self.sp_c_.setRange(1,5); self.sp_c_.setValue(1); self.sp_c_.setFixedWidth(44)
        for s in [self.sp_a,self.sp_b,self.sp_c_]: s.valueChanged.connect(self._rebuild); row2.addWidget(s)
        gl3.addLayout(row2)
        row3=QHBoxLayout(); row3.addWidget(QLabel("Bond scale:"))
        self.sp_bond=QDoubleSpinBox(); self.sp_bond.setRange(0.5,2.0); self.sp_bond.setValue(1.15); self.sp_bond.setSingleStep(0.05)
        self.sp_bond.valueChanged.connect(self._on_bond); row3.addWidget(self.sp_bond); gl3.addLayout(row3)
        row4=QHBoxLayout(); row4.addWidget(QLabel("Atom scale:"))
        self.sl_rad=QSlider(Qt.Orientation.Horizontal); self.sl_rad.setRange(25,220); self.sl_rad.setValue(100)
        self.sl_rad.valueChanged.connect(self._on_rad); row4.addWidget(self.sl_rad); gl3.addLayout(row4)
        r2=QHBoxLayout()
        self.chk_cell=QCheckBox("Unit cell"); self.chk_cell.setChecked(True)
        self.chk_bonds=QCheckBox("Bonds");    self.chk_bonds.setChecked(True)
        self.chk_fog=QCheckBox("Fog")
        self.chk_axes=QCheckBox("Axes");      self.chk_axes.setChecked(True)
        for c in [self.chk_cell,self.chk_bonds,self.chk_fog,self.chk_axes]: c.stateChanged.connect(self._on_opts); r2.addWidget(c)
        gl3.addLayout(r2); ll.addWidget(g3)

        # Lighting
        g4=QGroupBox("Lighting"); gl4=QGridLayout(g4); gl4.setSpacing(4)
        for ri,(lbl,attr,lo,hi,val) in enumerate([("Ambient","sl_amb",0,100,25),("Diffuse","sl_dif",0,100,70),("Specular","sl_spc",0,100,45),("Shininess","sl_shn",2,128,52)]):
            sl=QSlider(Qt.Orientation.Horizontal); sl.setRange(lo,hi); sl.setValue(val)
            sl.valueChanged.connect(self._on_light); setattr(self,attr,sl)
            gl4.addWidget(QLabel(lbl),ri,0); gl4.addWidget(sl,ri,1)
        ll.addWidget(g4)

        # Camera presets
        g5=QGroupBox("Camera  (Keys: 1‑4 R T Esc)"); gl5=QVBoxLayout(g5)
        cr=QHBoxLayout()
        for lbl,az,el in [("Top",0,90),("Front",0,0),("Side",90,0),("Iso",-45,25)]:
            b=QPushButton(lbl); b.setFixedHeight(26)
            b.clicked.connect(lambda _,a=az,e=el:(self.gl.camera.preset(a,e),self.gl.update()))
            cr.addWidget(b)
        gl5.addLayout(cr)
        sr=QHBoxLayout()
        btn_sp=QPushButton("▶ Turntable"); btn_sp.clicked.connect(lambda:self.gl.toggle_spin())
        btn_st=QPushButton("⏹ Stop");     btn_st.clicked.connect(lambda:self.gl.stop_spin())
        sr.addWidget(btn_sp); sr.addWidget(btn_st); gl5.addLayout(sr); ll.addWidget(g5)

        # Background
        g6=QGroupBox("Background"); gl6=QHBoxLayout(g6)
        for lbl,rgb in [("VESTA",(0.08,0.08,0.14)),("White",(1,1,1)),("Slate",(0.12,0.14,0.18)),("Black",(0,0,0))]:
            b=QPushButton(lbl); b.clicked.connect(lambda _,c=rgb:(self.gl.set_bg(*c),self.gl.update())); gl6.addWidget(b)
        ll.addWidget(g6)

        # Element editor
        g7=QGroupBox("Element Editor  (double-click)"); gl7=QVBoxLayout(g7)
        self.elem_tbl=QTableWidget(0,3); self.elem_tbl.setHorizontalHeaderLabels(["Elem","Color","Radius Å"])
        self.elem_tbl.setMaximumHeight(130)
        self.elem_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.elem_tbl.itemDoubleClicked.connect(self._edit_elem); gl7.addWidget(self.elem_tbl); ll.addWidget(g7)

        # Measurement
        g8=QGroupBox("Measurement  (click atoms in 3D)"); gl8=QVBoxLayout(g8)
        mr=QHBoxLayout()
        for t,f in [("📏 Dist",self._meas_d),("∠ Angle",self._meas_a),("✕ Clear",self._meas_clr)]:
            b=QPushButton(t); b.clicked.connect(f); mr.addWidget(b)
        gl8.addLayout(mr)
        self.lbl_meas=QLabel("Click atoms in 3D view to select"); self.lbl_meas.setWordWrap(True)
        self.lbl_meas.setStyleSheet("font-size:11px;padding:5px;background:#2a2a1e;"
            "border-radius:5px;border:1px solid #f9e2af;color:#fde68a;")
        gl8.addWidget(self.lbl_meas); ll.addWidget(g8)

        # Export
        g9=QGroupBox("Export & Analysis"); gl9=QVBoxLayout(g9)
        for t,f in [("Screenshot",self._shot),("Raw POSCAR",self._raw),
                    ("Copy formula",self._formula),("Bond table",self._bond_tbl),
                    ("Coordination",self._coord_tbl),("Atom positions",self._atom_tbl)]:
            b=QPushButton(t); b.clicked.connect(f); gl9.addWidget(b)
        ll.addWidget(g9)
        ll.addStretch()
        scroll.setWidget(lw)

        # ── RIGHT (OpenGL) ─────────────────────────────────────────────────
        rw=QWidget(); rw.setStyleSheet("background:#0d0d1a;")
        rw.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        rl=QVBoxLayout(rw); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0)

        tb=QWidget(); tb.setFixedHeight(36); tb.setStyleSheet("background:#16213e;border-bottom:1px solid #2d2d50;")
        tl=QHBoxLayout(tb); tl.setContentsMargins(8,3,8,3); tl.setSpacing(8)
        tl.addWidget(QLabel("VaspViz OpenGL Viewer  —  VESTA-style")); tl.addStretch()
        for t,f in [("Reset (R)",lambda:(self.gl.camera.reset(self.gl._scene_r),self.gl.update())),
                    ("ESC Deselect",lambda:self.gl.clear_sel())]:
            b=QPushButton(t); b.setStyleSheet("background:#2d2d50;color:#c0c0d0;border:1px solid #3d3d6b;border-radius:4px;padding:2px 8px;font-size:11px;"); b.clicked.connect(f); tl.addWidget(b)

        self.lbl_st=QLabel("  Ready — open a POSCAR/CONTCAR/CIF/XYZ")
        self.lbl_st.setFixedHeight(24)
        self.lbl_st.setStyleSheet("background:#16213e;color:#7eb8f7;font-size:11px;border-top:1px solid #2d2d50;padding:2px 10px;")

        self.gl=StructureGLWidget(self)
        self.gl.atom_picked.connect(self._on_pick)
        self.gl.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        rl.addWidget(tb); rl.addWidget(self.gl,1); rl.addWidget(self.lbl_st)
        root.addWidget(scroll); root.addWidget(rw,1)

    def _upd_pkgs(self):
        spg=self._has_spg(); ase=self._has_ase()
        self.lbl_pkgs.setText(("spglib OK" if spg else "spglib missing")+"   "+("ASE OK" if ase else "ASE optional"))
        self.lbl_pkgs.setStyleSheet(f"font-size:10px;color:{'#86efac' if spg else '#fca5a5'};padding:3px 6px;background:#16213e;border-radius:4px;")

    @staticmethod
    def _has_spg():
        try: import spglib; return True
        except: return False
    @staticmethod
    def _has_ase():
        try: import ase; return True
        except: return False

    def _st(self,m): self.lbl_st.setText(f"  {m}")

    # ── file loading ────────────────────────────────────────────────────
    def _open(self):
        p,_=QFileDialog.getOpenFileName(self,"Open Structure","",
            "Structure files (POSCAR CONTCAR *.vasp *.poscar *.cif *.xyz *.xsf);;All (*)")
        if not p: return
        ext=p.rsplit(".",1)[-1].lower() if "." in p else ""
        self.data=None
        if self._has_ase() and ext in ("cif","xyz","xsf"):
            try:
                from ase.io import read as ar
                self.data=self._ase_dict(ar(p),p)
            except Exception as e: print(f"ASE: {e}")
        if not self.data:
            try: self.data=PoscarParser(p).parse()
            except Exception as e: QMessageBox.critical(self,"Load Error",str(e)); return
        self._populate(); self._rebuild()
        if self._has_spg():
            sg=self._get_sg()
            if sg: self.lbl_sg.setText(f"Space group: {sg}")

    def _ase_dict(self,atoms,fp):
        lat=atoms.cell.array; cart=atoms.positions; syms=atoms.get_chemical_symbols()
        from collections import OrderedDict as OD
        cnt=OD()
        for s in syms: cnt[s]=cnt.get(s,0)+1
        sp=list(cnt.keys()); co=list(cnt.values())
        try: frac=atoms.get_scaled_positions()
        except: frac=cart@np.linalg.pinv(lat)
        vol=abs(float(np.dot(lat[0],np.cross(lat[1],lat[2]))))
        lens=np.linalg.norm(lat,axis=1)
        def ang(a,b): return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12),-1,1))))
        return {"comment":Path(fp).name,"lattice":lat,"scale":1.0,"species":sp,"counts":co,
                "ion_labels":syms,"frac_positions":frac,"cart_positions":cart,
                "coord_type":"direct","filepath":fp,"total_atoms":len(atoms),"volume":vol,
                "a":float(lens[0]),"b":float(lens[1]),"c":float(lens[2]),
                "alpha":ang(lat[1],lat[2]),"beta":ang(lat[0],lat[2]),"gamma":ang(lat[0],lat[1]),
                "n_electrons":sum(VALENCE_ELECTRONS.get(s,0) for s in syms)}

    def _populate(self):
        d=self.data; self.info_tree.clear()
        formula="".join(f"{s}{c}" for s,c in zip(d["species"],d["counts"]))
        for k,v in [("File",Path(d["filepath"]).name),("Formula",formula),
                    ("Atoms",str(d["total_atoms"])),
                    ("Species","  ".join(f"{s}×{c}" for s,c in zip(d["species"],d["counts"]))),
                    ("a Å",f"{d['a']:.4f}"),("b Å",f"{d['b']:.4f}"),("c Å",f"{d['c']:.4f}"),
                    ("α°",f"{d.get('alpha',90.):.2f}"),("β°",f"{d.get('beta',90.):.2f}"),
                    ("γ°",f"{d.get('gamma',90.):.2f}"),("Vol Å³",f"{d['volume']:.3f}")]:
            self.info_tree.addTopLevelItem(QTreeWidgetItem([k,str(v)]))
        self.info_tree.resizeColumnToContents(0)
        ions=d["ion_labels"]; nelec=d.get("n_electrons",0); dens=len(ions)/max(d["volume"],1)*1e3
        self.lbl_elec.setText(f"Elements: {', '.join(dict.fromkeys(ions))}\n"
            f"Valence e⁻≈{nelec}    ρ={dens:.2f} at/nm³\nVol/atom={d['volume']/max(1,len(ions)):.2f} Å³")
        uniq=list(dict.fromkeys(ions)); self.elem_tbl.setRowCount(len(uniq))
        for row,elem in enumerate(uniq):
            hx=self._ecolors.get(elem,ATOM_COLORS.get(elem,"#888888"))
            rv=self._eradii.get(elem,DISPLAY_RADII.get(elem,DISPLAY_RADII["default"]))
            self.elem_tbl.setItem(row,0,QTableWidgetItem(elem))
            ci=QTableWidgetItem(hx); ci.setBackground(QColor(hx))
            bright=sum(int(hx[i:i+2],16) for i in (1,3,5))>400
            ci.setForeground(QColor("#000" if bright else "#fff"))
            self.elem_tbl.setItem(row,1,ci); self.elem_tbl.setItem(row,2,QTableWidgetItem(f"{rv:.3f}"))
        self._st(f"{Path(d['filepath']).name}  |  {formula}  |  {d['total_atoms']} atoms  |  V={d['volume']:.1f} Å³")

    def _rebuild(self):
        if not self.data: return
        d=self.data; na,nb,nc=self.sp_a.value(),self.sp_b.value(),self.sp_c_.value()
        lat=d["lattice"]; frac=d["frac_positions"]; ions=d["ion_labels"]
        sc_pos=[]; sc_lbl=[]
        for ia in range(na):
            for ib in range(nb):
                for ic in range(nc):
                    for fp,l in zip(frac,ions):
                        sc_pos.append((fp+np.array([ia,ib,ic]))@lat); sc_lbl.append(l)
        sc_lat=lat*np.array([[na],[nb],[nc]])
        self.gl.custom_colors={e:_hex_to_rgb(self._ecolors[e]) for e in self._ecolors}
        self.gl.custom_radii=dict(self._eradii)
        self.gl.render_mode=self.mode_cb.currentIndex()
        self.gl.show_cell=self.chk_cell.isChecked(); self.gl.show_bonds=self.chk_bonds.isChecked()
        self.gl.fog_enabled=self.chk_fog.isChecked(); self.gl.radius_scale=self.sl_rad.value()/100.0
        self.gl.bond_radius=0.16
        self.gl.set_scene(np.array(sc_pos),sc_lbl,sc_lat,self._bscale)
        self._st(f"{na}×{nb}×{nc} supercell  |  {len(sc_pos)} atoms")

    def _sc_data(self):
        if not self.data: return None,None
        d=self.data; na,nb,nc=self.sp_a.value(),self.sp_b.value(),self.sp_c_.value()
        lat=d["lattice"]; frac=d["frac_positions"]; ions=d["ion_labels"]
        pos=[]; lbl=[]
        for ia in range(na):
            for ib in range(nb):
                for ic in range(nc):
                    for fp,l in zip(frac,ions): pos.append((fp+np.array([ia,ib,ic]))@lat); lbl.append(l)
        return np.array(pos),lbl

    # ── signal handlers ─────────────────────────────────────────────────
    def _on_mode(self,i): self.gl.render_mode=i; self.gl.refresh()
    def _on_bond(self,v): self._bscale=float(v); self._rebuild()
    def _on_rad(self,v):  self.gl.radius_scale=v/100.0; self.gl.refresh()
    def _on_opts(self):
        self.gl.show_cell=self.chk_cell.isChecked()
        self.gl.show_bonds=self.chk_bonds.isChecked()
        self.gl.fog_enabled=self.chk_fog.isChecked()
        self.gl.show_axes=self.chk_axes.isChecked(); self.gl.refresh()
    def _on_light(self):
        self.gl.ambient=self.sl_amb.value()/100.0; self.gl.diffuse=self.sl_dif.value()/100.0
        self.gl.specular=self.sl_spc.value()/100.0; self.gl.shininess=float(self.sl_shn.value())
        self.gl.update()
    def _on_pick(self,idx):
        sc_pos,sc_lbl=self._sc_data()
        if sc_pos is None or idx>=len(sc_lbl): return
        lbl=sc_lbl[idx]; pos=sc_pos[idx]
        if idx in self._meas: self._meas.remove(idx)
        else:
            self._meas.append(idx)
            if len(self._meas)>3: self._meas=self._meas[-3:]
        self.lbl_meas.setText(f"Selected [{len(self._meas)}/3]: {lbl}[{idx+1}] ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) Å")
        self._st(f"Atom {idx+1}: {lbl}  ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) Å")

    # ── element editor ───────────────────────────────────────────────────
    def _edit_elem(self,item):
        row=item.row()
        elem=(self.elem_tbl.item(row,0) or QTableWidgetItem("")).text()
        if not elem: return
        cur=self._ecolors.get(elem,ATOM_COLORS.get(elem,"#888888"))
        cur_r=self._eradii.get(elem,DISPLAY_RADII.get(elem,1.25))
        dlg=QDialog(self); dlg.setWindowTitle(f"Edit: {elem}"); dlg.setStyleSheet("background:#1a1a2e;color:#e0e0e0;")
        v=QVBoxLayout(dlg)
        self._eh=cur
        btn_c=QPushButton(f"Color: {cur}"); btn_c.setStyleSheet(f"background:{cur};padding:8px;")
        def pick():
            c=QColorDialog.getColor(QColor(self._eh),dlg)
            if c.isValid(): self._eh=c.name(); btn_c.setStyleSheet(f"background:{c.name()};padding:8px;"); btn_c.setText(f"Color: {c.name()}")
        btn_c.clicked.connect(pick); v.addWidget(btn_c)
        v.addWidget(QLabel("Radius (Å):"))
        sp=QDoubleSpinBox(); sp.setRange(0.1,3.5); sp.setSingleStep(0.05); sp.setValue(cur_r)
        sp.setStyleSheet("background:#2d2d50;color:#e0e0e0;"); v.addWidget(sp)
        def rst(): self._ecolors.pop(elem,None); self._eradii.pop(elem,None); dlg.accept(); self._populate(); self._rebuild()
        br=QPushButton("↩ Reset"); br.setStyleSheet("background:#3d3d6b;"); br.clicked.connect(rst); v.addWidget(br)
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        bb.setStyleSheet("background:#2d2d50;"); bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); v.addWidget(bb)
        if dlg.exec()==QDialog.DialogCode.Accepted:
            self._ecolors[elem]=self._eh; self._eradii[elem]=sp.value()
            self._populate(); self._rebuild()

    # ── measurement ──────────────────────────────────────────────────────
    def _meas_d(self):
        if len(self._meas)<2: QMessageBox.warning(self,"","Select ≥2 atoms"); return
        pos,lbl=self._sc_data()
        if pos is None: return
        i,j=self._meas[0],self._meas[1]; d=float(np.linalg.norm(pos[i]-pos[j]))
        QMessageBox.information(self,"Distance",f"{lbl[i]}[{i+1}] — {lbl[j]}[{j+1}]\n\nd = {d:.4f} Å")
    def _meas_a(self):
        if len(self._meas)<3: QMessageBox.warning(self,"","Select 3 atoms"); return
        pos,lbl=self._sc_data()
        if pos is None: return
        i,j,k=self._meas[:3]; v1=pos[i]-pos[j]; v2=pos[k]-pos[j]
        ang=float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-12),-1,1))))
        QMessageBox.information(self,"Angle",f"∠ {lbl[i]}—{lbl[j]}—{lbl[k]}\n\n∠ = {ang:.4f}°")
    def _meas_clr(self): self._meas.clear(); self.gl.clear_sel(); self.lbl_meas.setText("Click atoms in 3D view")

    # ── symmetry ─────────────────────────────────────────────────────────
    def _spg_cell(self):
        d=self.data
        return (d["lattice"],d["frac_positions"],[ELEMENT_NUMBERS.get(s,1) for s in d["ion_labels"]])
    def _get_sg(self):
        try: import spglib; return spglib.get_spacegroup(self._spg_cell(),symprec=1e-3)
        except: return None
    def _find_sym(self):
        if not self._has_spg(): QMessageBox.warning(self,"","pip install spglib"); return
        if not self.data: return
        try:
            import spglib; cell=self._spg_cell()
            sg=spglib.get_spacegroup(cell,symprec=1e-3); ds=spglib.get_symmetry_dataset(cell,symprec=1e-3)
            n_ops=len(ds["rotations"]) if ds else "?"
            wyck=", ".join(dict.fromkeys(ds.get("wyckoffs",[]))) if ds else "?"
            self.lbl_sg.setText(f"Space group: {sg}")
            QMessageBox.information(self,"Symmetry",f"Space group: {sg}\nHall: {ds.get('hall','?') if ds else '?'}\nSym ops: {n_ops}\nWyckoff: {wyck}")
        except Exception as e: QMessageBox.warning(self,"Error",str(e))
    def _prim(self):
        if not self._has_spg() or not self.data: return
        try:
            import spglib; p=spglib.find_primitive(self._spg_cell(),symprec=1e-3)
            if p: lat,pos,_=p; QMessageBox.information(self,"Primitive",f"{self.data['total_atoms']}→{len(pos)} atoms  ({self.data['total_atoms']/len(pos):.1f}× reduction)")
            else: QMessageBox.information(self,"Primitive","Already primitive.")
        except Exception as e: QMessageBox.warning(self,"",str(e))
    def _conv(self):
        if not self._has_spg() or not self.data: return
        try:
            import spglib; c=spglib.standardize_cell(self._spg_cell(),to_primitive=False,symprec=1e-3)
            if c: lat,pos,_=c; lens=np.linalg.norm(lat,axis=1); QMessageBox.information(self,"Conventional",f"{len(pos)} atoms\na={lens[0]:.4f} b={lens[1]:.4f} c={lens[2]:.4f} Å")
        except Exception as e: QMessageBox.warning(self,"",str(e))

    # ── analysis tables ───────────────────────────────────────────────────
    def _mk_dlg(self,title,nr,nc,hdrs):
        dlg=QDialog(self); dlg.setWindowTitle(title); dlg.resize(560,400)
        dlg.setStyleSheet("background:#1a1a2e;color:#e0e0e0;")
        v=QVBoxLayout(dlg)
        tbl=QTableWidget(nr,nc); tbl.setHorizontalHeaderLabels(hdrs)
        tbl.setStyleSheet("background:#16213e;color:#e0e0e0;font-size:11px;")
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tbl.setAlternatingRowColors(True); v.addWidget(tbl)
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok); bb.setStyleSheet("background:#2d2d50;"); bb.accepted.connect(dlg.accept); v.addWidget(bb)
        return dlg,tbl

    def _bond_tbl(self):
        if not self.data: return
        pos,lbl=self._sc_data()
        if pos is None: return
        bonds=[]
        for i in range(len(pos)):
            ri=COVALENT_RADII.get(lbl[i],COVALENT_RADII["default"])
            for j in range(i+1,len(pos)):
                rj=COVALENT_RADII.get(lbl[j],COVALENT_RADII["default"])
                d=float(np.linalg.norm(pos[i]-pos[j]))
                if 0.4<d<self._bscale*(ri+rj): bonds.append((lbl[i],i+1,lbl[j],j+1,d))
        if not bonds: QMessageBox.information(self,"Bonds","No bonds detected."); return
        dlg,tbl=self._mk_dlg("Bond Lengths",len(bonds),4,["Atom A","Atom B","d (Å)","Type"])
        for r,(la,ia,lb,ib,d) in enumerate(sorted(bonds,key=lambda x:x[4])):
            for c,v in enumerate([f"{la}[{ia}]",f"{lb}[{ib}]",f"{d:.4f}",f"{la}-{lb}"]): tbl.setItem(r,c,QTableWidgetItem(v))
        dlg.exec()

    def _coord_tbl(self):
        if not self.data: return
        pos,lbl=self._sc_data()
        if pos is None: return
        n=len(pos); coord=[[] for _ in range(n)]
        for i in range(n):
            ri=COVALENT_RADII.get(lbl[i],COVALENT_RADII["default"])
            for j in range(n):
                if i==j: continue
                if float(np.linalg.norm(pos[i]-pos[j]))<self._bscale*(ri+COVALENT_RADII.get(lbl[j],COVALENT_RADII["default"])): coord[i].append(j)
        dlg,tbl=self._mk_dlg(f"Coordination ({n} atoms)",n,4,["#","Elem","CN","Neighbours"])
        for i in range(n):
            nb=coord[i]; nb_str=", ".join(f"{lbl[j]}[{j+1}]" for j in nb[:6])+("..." if len(nb)>6 else "")
            for c,v in enumerate([str(i+1),lbl[i],str(len(nb)),nb_str]): tbl.setItem(i,c,QTableWidgetItem(v))
        dlg.exec()

    def _atom_tbl(self):
        if not self.data: return
        d=self.data; ions=d["ion_labels"]; cart=d["cart_positions"]; frac=d["frac_positions"]
        dlg,tbl=self._mk_dlg("Per-atom Positions",len(ions),8,["#","Elem","x Å","y Å","z Å","fx","fy","fz"])
        for i,(l,cp,fp) in enumerate(zip(ions,cart,frac)):
            vals=[str(i+1),l]+[f"{cp[j]:.4f}" for j in range(3)]+[f"{fp[j]:.4f}" for j in range(3)]
            for c,v in enumerate(vals): tbl.setItem(i,c,QTableWidgetItem(v))
        dlg.exec()

    # ── export ────────────────────────────────────────────────────────────
    def _shot(self):
        p,_=QFileDialog.getSaveFileName(self,"Screenshot","structure","PNG (*.png);;JPEG (*.jpg)")
        if not p: return
        px=self.gl.screenshot()
        if px.save(p): self._st(f"Saved: {p}")
        else: QMessageBox.warning(self,"","Save failed.")
    def _raw(self):
        if not self.data: return
        fp=self.data["filepath"]
        try: txt=open(fp,errors="replace").read()
        except: txt="Cannot read"
        dlg=QDialog(self); dlg.setWindowTitle(f"Raw: {Path(fp).name}"); dlg.resize(560,480)
        dlg.setStyleSheet("background:#1a1a2e;color:#e0e0e0;"); v=QVBoxLayout(dlg)
        te=QTextEdit(); te.setFont(QFont("Courier New",10)); te.setStyleSheet("background:#16213e;color:#e0e0e0;"); te.setPlainText(txt); te.setReadOnly(True); v.addWidget(te)
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok); bb.setStyleSheet("background:#2d2d50;"); bb.accepted.connect(dlg.accept); v.addWidget(bb); dlg.exec()
    def _formula(self):
        if not self.data: return
        d=self.data; f="".join(f"{s}{c}" for s,c in zip(d["species"],d["counts"]))
        QApplication.clipboard().setText(f); self._st(f"Copied: {f}")
