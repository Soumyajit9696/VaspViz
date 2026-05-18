"""VaspViz — analysis.py: demo data, band-gap/mass/optical analysis, PlotEngine."""
import csv, math
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib
from constants import (ORBITAL_GROUPS, ORBITAL_COLORS, SPIN_COLORS, KNAME_MAP,
                       HBAR2_OVER_2M, EV_TO_CM1, AVAILABLE_FONTS)


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def find_band_gap(ev_shifted):
    vbm=-np.inf; cbm=np.inf; vbm_k=0; cbm_k=0; vbm_b=0; cbm_b=0
    for isp in range(ev_shifted.shape[0]):
        for ib in range(ev_shifted.shape[2]):
            e = ev_shifted[isp,:,ib]
            below = e[e<=0]
            if len(below) and below.max()>vbm:
                vbm=below.max(); vbm_k=int(np.argmax(np.where(e<=0,e,-np.inf))); vbm_b=ib
            above = e[e>0]
            if len(above) and above.min()<cbm:
                cbm=above.min(); cbm_k=int(np.argmin(np.where(e>0,e,np.inf))); cbm_b=ib
    if cbm==np.inf or vbm==-np.inf:
        return {"gap":0,"type":"metal","vbm":0,"cbm":0,"vbm_k":0,"cbm_k":0,"vbm_b":0,"cbm_b":0}
    return {"gap":max(0,cbm-vbm),"type":"direct" if vbm_k==cbm_k else "indirect",
            "vbm":vbm,"cbm":cbm,"vbm_k":vbm_k,"cbm_k":cbm_k,"vbm_b":vbm_b,"cbm_b":cbm_b}

def fit_effective_mass(kdist, energies, k_idx, n_pts=8):
    i0=k_idx; half=n_pts//2
    lo=max(0,i0-half); hi=min(len(kdist)-1,i0+half)
    k=kdist[lo:hi+1]-kdist[i0]; e=energies[lo:hi+1]
    if len(k)<3: return None
    try:
        popt,_=curve_fit(lambda k,e0,a: e0+a*k**2, k, e, p0=[energies[i0],1.0])
        a=popt[1]
        if abs(a)<1e-8: return None
        return HBAR2_OVER_2M/abs(a)*(1 if a>0 else -1)
    except: return None

def compute_jdos(energies, band_ev, broadening=0.1):
    nk,nb=band_ev.shape; jdos=np.zeros(len(energies))
    de=energies[1]-energies[0] if len(energies)>1 else 0.01
    for i in range(nb):
        for j in range(i+1,nb):
            trans=band_ev[:,j]-band_ev[:,i]; trans=trans[trans>0]
            for delta_e in trans:
                idx=int(round((delta_e-energies[0])/de))
                if 0<=idx<len(jdos): jdos[idx]+=1.0
    if broadening>0: jdos=gaussian_filter1d(jdos,sigma=max(1,broadening/de))
    return jdos

def compute_optical_spectrum(energies, band_ev, occ, broadening=0.1, volume_ang3=100.):
    """
    Simplified optical conductivity using Drude-Lorentz model from JDOS.
    Returns: sigma1, sigma2, epsilon1, epsilon2, n, k, absorption, EELS, reflectivity
    """
    jdos = compute_jdos(energies, band_ev, broadening)
    # Epsilon imaginary from JDOS (proportional)
    eps2 = np.zeros(len(energies))
    mask = energies > 0.01
    eps2[mask] = jdos[mask] / (energies[mask]**2 + 0.01)
    eps2 = eps2 / (eps2.max()+1e-10)  # normalize

    # Kramers-Kronig for eps1
    from scipy.signal import hilbert
    eps1 = 1.0 + np.real(hilbert(eps2))

    # Optical conductivity: σ = ω·ε₂/(4π) (in arb units)
    sigma1 = np.zeros(len(energies))
    sigma2 = np.zeros(len(energies))
    sigma1[mask] = energies[mask]*eps2[mask]
    sigma2[mask] = energies[mask]*(1-eps1[mask])

    # Refractive index n + ik
    n_idx = np.sqrt(np.maximum(0, (eps1+np.sqrt(eps1**2+eps2**2))/2))
    k_idx = np.where(n_idx>0, eps2/(2*n_idx), 0.)

    # Absorption α = 2ω·k/c (in cm⁻¹, ω in eV)
    absorption = 2*energies*EV_TO_CM1*k_idx/3e10 * 1e7  # arb
    absorption = gaussian_filter1d(absorption, sigma=max(1,broadening/((energies[1]-energies[0])+1e-10)))

    # EELS: -Im(1/ε)
    denom = eps1**2+eps2**2+1e-10
    eels = eps2/denom

    # Reflectivity
    reflect = ((n_idx-1)**2+k_idx**2)/((n_idx+1)**2+k_idx**2+1e-10)

    return {"sigma1":sigma1,"sigma2":sigma2,"eps1":eps1,"eps2":eps2,
            "n":n_idx,"k":k_idx,"absorption":absorption,"eels":eels,"reflectivity":reflect}

def moire_period(a, theta_deg):
    th = np.radians(theta_deg)
    if abs(th)<1e-6: return np.inf
    return a/(2*np.sin(th/2))

def reciprocal_lattice(lattice):
    """Compute reciprocal lattice vectors from real-space lattice (rows = a1,a2,a3)."""
    a1,a2,a3 = lattice
    vol = np.dot(a1, np.cross(a2,a3))
    b1 = 2*np.pi*np.cross(a2,a3)/vol
    b2 = 2*np.pi*np.cross(a3,a1)/vol
    b3 = 2*np.pi*np.cross(a1,a2)/vol
    return np.array([b1,b2,b3])

def detect_bonds(cart_pos, labels, cutoff_scale=1.2):
    """Detect bonds by covalent radius sum."""
    bonds = []
    n = len(cart_pos)
    for i in range(n):
        ri = COVALENT_RADII.get(labels[i], COVALENT_RADII["default"])
        for j in range(i+1, n):
            rj = COVALENT_RADII.get(labels[j], COVALENT_RADII["default"])
            dist = np.linalg.norm(cart_pos[i]-cart_pos[j])
            if dist < cutoff_scale*(ri+rj) and dist > 0.5:
                bonds.append((i, j, dist))
    return bonds

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class PlotEngine:
    def __init__(self, fig):
        self.fig = fig; self.data = None; self.data2 = None; self.S = {}
        self.spin_colors = list(SPIN_COLORS); self.band2_color = "#A855F7"
        self.selected_bands = None  # None = all, list = specific bands

    def set_data(self,d): self.data=d
    def set_data2(self,d): self.data2=d
    def set_settings(self,**kw): self.S.update(kw)

    def _se(self):
        ev = self.data["eigenvalues"].copy()
        if self.S.get("shift_efermi", True): ev -= self.data["efermi"]
        return ev

    def _ew(self): return self.S.get("emin", -6.), self.S.get("emax", 6.)
    def _efl(self): return 0. if self.S.get("shift_efermi", True) else self.data["efermi"]
    def _rast(self): return not self.S.get("vector_export", False)
    def _ef_ls(self):
        style = self.S.get("fermi_style", "dashed")
        if style == "solid": return "-"
        if style == "dotted": return ":"
        if style == "dashdot": return "-."
        return "--"
    def _ylbl(self):
        yl = self.S.get("ylabel", None)
        if yl: return yl
        return "E − $E_F$ (eV)" if self.S.get("shift_efermi", True) else "Energy (eV)"

    def _set_font(self):
        fam=self.S.get("font_family","DejaVu Sans")
        matplotlib.rcParams["font.family"]=fam
        matplotlib.rcParams["mathtext.fontset"]="stix" if "stix" in fam.lower() or "times" in fam.lower() else "dejavusans"
        matplotlib.rcParams["font.size"]=self.S.get("font_size",11)

    def _style(self,ax):
        dk=self.S.get("dark",False)
        bg="#0f172a" if dk else "#ffffff"; fg="#e2e8f0" if dk else "#1e293b"
        ax.set_facecolor(bg)

        # ── Spine customization ───────────────────────────────────────────────
        show_spines = self.S.get("show_spines", True)
        spine_sides = self.S.get("spine_sides", {"left":True,"right":False,"bottom":True,"top":False})
        spine_color = self.S.get("spine_color", fg)
        spine_lw    = self.S.get("spine_lw", 0.8)
        spine_ls_map = {"solid":"-", "dashed":"--", "dotted":":", "dashdot":"-."}
        spine_ls    = spine_ls_map.get(self.S.get("spine_style", "solid"), "-")
        for side, sp in ax.spines.items():
            visible = show_spines and spine_sides.get(side, True)
            sp.set_visible(visible)
            if visible:
                sp.set_color(spine_color)
                sp.set_linewidth(spine_lw)
                sp.set_linestyle(spine_ls)

        t_dir = self.S.get("tick_dir", "out")
        t_len = self.S.get("tick_len", 4.0)
        t_wid = self.S.get("tick_width", 0.8)
        t_pad = self.S.get("tick_pad", 3.0)

        ax.tick_params(labelsize=self.S.get("font_size",11)-1, colors=fg,
                       direction=t_dir, length=t_len, width=t_wid, pad=t_pad)
        ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg); ax.title.set_color(fg)
        if self.S.get("grid_major",True): ax.grid(True,which="major",alpha=0.12,lw=0.5,color=fg)
        else: ax.grid(False)
        if self.S.get("minor_ticks", True):
            ax.minorticks_on()
            ax.tick_params(which="minor", direction=t_dir, length=max(2.0, t_len/2.0), width=max(0.4, t_wid/2.0), colors=fg)
        else:
            ax.minorticks_off()

    def _klines(self,ax):
        if not self.S.get("show_klines",True): return
        for idx,_ in self.data["klabels"]:
            if idx<len(self.data["kdist"]):
                ax.axvline(self.data["kdist"][idx],color="#94A3B8",lw=0.8,ls="--",alpha=0.45,zorder=0)

    def _kticks(self,ax):
        kd=self.data["kdist"]; ticks=[]; labs=[]; px=None
        for idx,lbl in self.data["klabels"]:
            if idx>=len(kd): continue
            x=kd[idx]
            if px is not None and abs(x-px)<1e-8:
                if ticks:
                    old=labs[-1]; labs[-1]=f"{old}|{lbl}" if lbl and lbl!=old else old or lbl
                continue
            ticks.append(x); labs.append(lbl or ""); px=x
        ax.set_xticks(ticks)
        dk=self.S.get("dark",False); fg="#e2e8f0" if dk else "#1e293b"
        ax.set_xticklabels(labs,fontsize=self.S.get("font_size",11)+1,color=fg)
        ax.set_xlim(kd[0],kd[-1]); ax.tick_params(axis="x",bottom=False)

    def _band_iter(self, nb):
        """Yield 0-based band indices respecting selected_bands filter."""
        sel = self.S.get("selected_bands", None)
        if sel is not None and isinstance(sel, (list, tuple)) and len(sel) > 0:
            return [i for i in sel if 0 <= i < nb]
        return range(nb)

    def _spin_iter(self, ns):
        """Yield spin indices respecting spin_channel setting."""
        ch = self.S.get("spin_channel", "both")
        if ch == "up":   return [0]
        if ch == "down": return [1] if ns > 1 else [0]
        return range(ns)

    def _interpolate_bands(self, ev, kd):
        """Return interpolated (ev_new, kd_new) if interp_factor > 1."""
        factor = int(self.S.get("interp_factor", 1))
        if factor <= 1: return ev, kd
        from scipy.interpolate import interp1d
        ns, nk, nb = ev.shape
        nk_new = nk * factor
        kd_new = np.linspace(kd[0], kd[-1], nk_new)
        ev_new = np.zeros((ns, nk_new, nb))
        for isp in range(ns):
            for ib in range(nb):
                f = interp1d(kd, ev[isp, :, ib], kind="cubic", fill_value="extrapolate")
                ev_new[isp, :, ib] = f(kd_new)
        return ev_new, kd_new

    def _annotate_gap(self,ax,info,kd):
        if info["type"]=="metal" or info["gap"]<0.01: return
        vk=kd[info["vbm_k"]]; ck=kd[info["cbm_k"]]
        ax.plot(vk,info["vbm"],"v",color="#16A34A",ms=8,zorder=6,label=f"VBM {info['vbm']:.3f} eV")
        ax.plot(ck,info["cbm"],"^",color="#DC2626",ms=8,zorder=6,label=f"CBM {info['cbm']:.3f} eV")
        mx=(vk+ck)/2; my=(info["vbm"]+info["cbm"])/2
        ax.annotate(f"$E_g$={info['gap']:.3f} eV ({info['type']})",xy=(mx,my),
                    fontsize=9,ha="center",color="#DC2626",
                    bbox=dict(boxstyle="round,pad=0.35",fc="white",alpha=0.9,ec="#DC2626",lw=1.2))

    def plot_bands(self, ax, show_gap=False, show_mstar=False):
        if not self.data: return
        self._set_font()
        ev = self._se(); emin, emax = self._ew(); kd = self.data["kdist"]
        ev, kd = self._interpolate_bands(ev, kd)  # sumo interpolation
        ns, nk, nb = ev.shape; ax.clear(); self._klines(ax)
        lw = self.S.get("linewidth", 1.5)
        spin_colors = self.S.get("spin_colors", self.spin_colors)
        band_indices = list(self._band_iter(nb))
        spin_indices = list(self._spin_iter(ns))
        for isp in spin_indices:
            col = spin_colors[isp] if isp < len(spin_colors) else "#2563EB"
            for ib in band_indices:
                e = ev[isp, :, ib]
                if e.max() < emin-.5 or e.min() > emax+.5: continue
                ax.plot(kd, e, color=col, lw=lw, alpha=0.88, rasterized=self._rast(), zorder=2,
                        solid_capstyle="round", solid_joinstyle="round")
        # Highlight bands
        highlight = self.S.get("highlight_bands", None)
        hl_color  = self.S.get("highlight_color", "#F59E0B")
        if highlight and isinstance(highlight, list):
            for ib in highlight:
                if 0 <= ib < nb:
                    e = ev[0, :, ib]
                    if e.max() < emin-.5 or e.min() > emax+.5: continue
                    ax.plot(kd, e, color=hl_color, lw=lw*2.5, alpha=0.65,
                            rasterized=self._rast(), zorder=1, solid_capstyle="round")
                    mid_ik = len(kd)//3
                    ax.annotate(f" B{ib+1}", xy=(kd[mid_ik], e[mid_ik]), fontsize=8,
                                color=hl_color, va="center",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec=hl_color, lw=0.8))
        # Zero line / Fermi indicator
        show_fermi = self.S.get("show_fermi", True) or self.S.get("show_zero", True)
        if show_fermi:
            ax.axhline(self._efl(), color=self.S.get("fermi_color", "#EF4444"), lw=1.1, ls=self._ef_ls(), alpha=0.85, zorder=3)
        if self.data2 is not None:
            ev2 = self.data2["eigenvalues"].copy()
            if self.S.get("shift_efermi", True): ev2 -= self.data2["efermi"]
            kd2 = self.data2["kdist"]; c2 = self.S.get("band2_color", self.band2_color)
            for ib in range(ev2.shape[2]):
                e = ev2[0, :, ib]
                if e.max() < emin-.5 or e.min() > emax+.5: continue
                ax.plot(kd2, e, color=c2, lw=lw*0.8, alpha=0.6, ls="--", rasterized=self._rast(), zorder=2)
        if show_gap:
            gap = find_band_gap(ev); self._annotate_gap(ax, gap, kd)
        if show_mstar:
            gap = find_band_gap(ev)
            if gap["type"] != "metal":
                ms = fit_effective_mass(kd, ev[0, :, gap["cbm_b"]], gap["cbm_k"])
                if ms:
                    ax.text(0.02, 0.97, f"$m^*$ = {abs(ms):.3f} $m_e$", transform=ax.transAxes,
                            va="top", fontsize=9, color="#16A34A",
                            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.9))
        # Legend
        legend_on = self.S.get("legend_on", True)
        lpos = self.S.get("legend_pos", "upper right")
        if legend_on:
            handles = []
            if len(spin_indices) > 1:
                _up, _dn = "\u2191", "\u2193"
                handles = [mpatches.Patch(color=spin_colors[i], label=f"Spin {_up if i==0 else _dn}")
                           for i in spin_indices]
            if show_gap and not handles:
                handles = [mpatches.Patch(color="#16A34A", label=f"VBM {find_band_gap(ev)['vbm']:.3f} eV"),
                           mpatches.Patch(color="#DC2626", label=f"CBM {find_band_gap(ev)['cbm']:.3f} eV")]
            if handles:
                ax.legend(handles=handles, fontsize=9, framealpha=0.9, loc=lpos)
        # Band edges (sumo --band-edges)
        if self.S.get("band_edges", False):
            gap = find_band_gap(ev)
            if gap["type"] != "metal":
                ax.plot(kd[gap["vbm_k"]], gap["vbm"], "v", color="#16A34A", ms=9,
                        zorder=7, label="VBM", clip_on=False)
                ax.plot(kd[gap["cbm_k"]], gap["cbm"], "^", color="#DC2626", ms=9,
                        zorder=7, label="CBM", clip_on=False)
                if self.S.get("legend_on", True):
                    ax.legend(fontsize=9, framealpha=0.9, loc=self.S.get("legend_pos","upper right"))
        self._kticks(ax); ax.set_ylim(emin, emax)
        ax.set_ylabel(self._ylbl(), fontsize=self.S.get("font_size", 11))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(self.S.get("ytick_step", 2)))
        title = self.S.get("title", "")
        if title: ax.set_title(title, fontsize=self.S.get("font_size", 11)+2, fontweight="bold", pad=8)
        self._style(ax)

    def plot_fatbands(self,ax,orbital="s",atom_idx=None,scale=40):
        if not self.data: return
        self._set_font()
        ev=self._se(); emin,emax=self._ew(); kd=self.data["kdist"]
        _,nk,nb=ev.shape; proj=self.data.get("projections")
        ax.clear(); self._klines(ax)
        band_indices=list(self._band_iter(nb))
        for ib in band_indices:
            e=ev[0,:,ib]
            if e.max()<emin-.5 or e.min()>emax+.5: continue
            ax.plot(kd,e,color="#CBD5E1",lw=0.6,alpha=0.4,zorder=1,rasterized=self._rast())
        oi=ORBITAL_GROUPS.get(orbital,[0]); oc=ORBITAL_COLORS.get(orbital,"#2563EB")
        if proj is not None:
            p=proj[0]
            for ib in band_indices:
                e=ev[0,:,ib]
                if e.max()<emin-.5 or e.min()>emax+.5: continue
                if atom_idx is not None and 0<=atom_idx<p.shape[2]:
                    w=p[:,ib,atom_idx,:][:,oi].sum(axis=1)
                else:
                    w=p[:,ib,:,:][:,:,oi].sum(axis=(1,2))
                w=np.clip(w,0,None); mask=(e>=emin)&(e<=emax)
                ax.scatter(kd[mask],e[mask],s=w[mask]*scale,c=oc,alpha=0.6,linewidths=0,zorder=2,rasterized=self._rast())
        else:
            ax.text(.5,.5,"No projections.\nRun VASP with LORBIT=11",transform=ax.transAxes,
                    ha="center",va="center",fontsize=12,color="#888",style="italic")
        if self.S.get("show_fermi",True): ax.axhline(self._efl(),color=self.S.get("fermi_color", "#EF4444"),lw=1.1,ls=self._ef_ls(),zorder=3)
        self._kticks(ax); ax.set_ylim(emin,emax)
        ax.set_ylabel(self._ylbl(),fontsize=self.S.get("font_size",11))
        title=self.S.get("title","") or f"Fat bands — {orbital}" + (f", atom {atom_idx}" if atom_idx is not None else " (all)")
        ax.set_title(title,fontsize=self.S.get("font_size",11)+1,pad=8)
        self._style(ax)

    def plot_colormap_bands(self,ax,orbital="s",atom_idx=None):
        if not self.data: return
        self._set_font()
        proj=self.data.get("projections")
        if proj is None: self.plot_fatbands(ax,orbital,atom_idx); return
        ev=self._se(); emin,emax=self._ew(); kd=self.data["kdist"]
        _,nk,nb=ev.shape; oi=ORBITAL_GROUPS.get(orbital,[0])
        p=proj[0]; cmap=cm.get_cmap(self.S.get("cmap","plasma"))
        norm=Normalize(vmin=self.S.get("wmin",0.),vmax=self.S.get("wmax",1.))
        ax.clear(); self._klines(ax)
        band_indices=list(self._band_iter(nb))
        for ib in band_indices:
            e=ev[0,:,ib]
            if e.max()<emin-.5 or e.min()>emax+.5: continue
            if atom_idx is not None and 0<=atom_idx<p.shape[2]:
                w=p[:,ib,atom_idx,:][:,oi].sum(axis=1)
            else:
                w=p[:,ib,:,:][:,:,oi].sum(axis=(1,2))
            w=np.clip(w,0,1)
            pts=np.array([kd,e]).T.reshape(-1,1,2)
            segs=np.concatenate([pts[:-1],pts[1:]],axis=1)
            lc=LineCollection(segs,cmap=cmap,norm=norm,lw=2.0,rasterized=self._rast(),capstyle="round")
            lc.set_array((w[:-1]+w[1:])/2); ax.add_collection(lc)
        if self.S.get("show_fermi",True): ax.axhline(self._efl(),color=self.S.get("fermi_color", "#EF4444"),lw=1.1,ls=self._ef_ls(),zorder=3)
        self._kticks(ax); ax.set_ylim(emin,emax); ax.set_xlim(kd[0],kd[-1])
        ax.set_ylabel(self._ylbl(),fontsize=self.S.get("font_size",11))
        self._style(ax)
        cb=self.fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,pad=0.01,fraction=0.025,aspect=30)
        cb.set_label(f"Projection ({orbital})",fontsize=9); cb.ax.tick_params(labelsize=8)
        title=self.S.get("title","")
        if title: ax.set_title(title,fontsize=self.S.get("font_size",11)+2,fontweight="bold",pad=8)

    def plot_dos(self,ax,show_s=True,show_p=True,show_d=True,fill=True,
                 horizontal=False,ion_filter=None,spin_mirror=False):
        if not self.data: return
        self._set_font()
        dos=self.data.get("dos")
        if dos is None:
            ax.text(.5,.5,"No DOS data.\nRun VASP with ISTART=0, NSW=0",
                    transform=ax.transAxes,ha="center",va="center",fontsize=12,color="#888",style="italic")
            self._style(ax); return
        ax.clear(); emin,emax=self._ew()
        ef=self.data["efermi"]; shift=ef if self.S.get("shift_efermi",True) else 0.
        spins=list(dos["total"].keys())
        for si,spin_key in enumerate(spins):
            total=dos["total"].get(spin_key)
            if total is None: continue
            e=total[:,0]-shift; tdos=total[:,1]
            sign=-1 if(spin_mirror and si>0) else 1
            mask=(e>=emin)&(e<=emax); e_p=e[mask]
            partial=dos.get("partial",[])
            ions_use=ion_filter if ion_filter else list(range(len(partial)))
            s_sum=np.zeros(len(e)); p_sum=np.zeros(len(e)); d_sum=np.zeros(len(e))
            for ii in ions_use:
                if ii>=len(partial): continue
                pd=partial[ii].get(spin_key)
                if pd is None: continue
                nc=pd.shape[1]
                if nc>=2: s_sum+=pd[:,1]
                if nc>=5: p_sum+=pd[:,2]+pd[:,3]+pd[:,4]
                elif nc>=3: p_sum+=pd[:,2]
                if nc>=10: d_sum+=pd[:,5:10].sum(axis=1)
                elif nc>=6: d_sum+=pd[:,5]

            # Apply DOS smoothing (sumo-style sigma)
            sigma = self.S.get("dos_sigma", 0.0)
            if sigma > 0:
                s_sum = gaussian_filter1d(s_sum, sigma * len(e) / max(abs(e[-1]-e[0]), 1e-6))
                p_sum = gaussian_filter1d(p_sum, sigma * len(e) / max(abs(e[-1]-e[0]), 1e-6))
                d_sum = gaussian_filter1d(d_sum, sigma * len(e) / max(abs(e[-1]-e[0]), 1e-6))
                tdos  = gaussian_filter1d(tdos,  sigma * len(e) / max(abs(e[-1]-e[0]), 1e-6))

            def pc(y, color, label, ls="-"):
                ym = y[mask] * sign
                if horizontal:
                    ax.plot(ym, e_p, lw=1.4, color=color, label=label if si==0 else "", zorder=3, ls=ls)
                    if fill: ax.fill_betweenx(e_p, 0, ym, alpha=self.S.get("dos_fill_alpha", 0.18), color=color, zorder=2)
                else:
                    ax.plot(e_p, ym, lw=1.4, color=color, label=label if si==0 else "", zorder=3, ls=ls)
                    if fill: ax.fill_between(e_p, 0, ym, alpha=self.S.get("dos_fill_alpha", 0.18), color=color, zorder=2)

            total_only = self.S.get("total_only", False)
            no_total   = self.S.get("no_total", False)

            dos_colors = self.S.get("dos_colors", {"s":"#2563EB", "p":"#16A34A", "d":"#EA580C", "tot":None})
            if not total_only:
                if show_s and s_sum.any(): pc(s_sum, dos_colors.get("s", "#2563EB"), "s")
                if show_p and p_sum.any(): pc(p_sum, dos_colors.get("p", "#16A34A"), "p")
                if show_d and d_sum.any(): pc(d_sum, dos_colors.get("d", "#EA580C"), "d")
            if not no_total:
                dk = self.S.get("dark", False)
                tc = dos_colors.get("tot") if dos_colors.get("tot") else ("#cbd5e1" if dk else "#374151")
                if horizontal: ax.plot(tdos[mask]*sign, e_p, lw=1.8, color=tc, alpha=0.85, label="Total" if si==0 else "", zorder=4)
                else: ax.plot(e_p, tdos[mask]*sign, lw=1.8, color=tc, alpha=0.85, label="Total" if si==0 else "", zorder=4)

        dmax = self.S.get("dos_max", 0.0)
        yf=0. if self.S.get("shift_efermi",True) else ef
        if horizontal:
            ax.axhline(yf,color=self.S.get("fermi_color", "#EF4444"),lw=1.1,ls=self._ef_ls(),label="$E_F$",zorder=5)
            ax.set_ylim(emin,emax); ax.set_xlabel("DOS (states/eV)",fontsize=self.S.get("font_size",11))
            if dmax > 0: ax.set_xlim(-dmax if spin_mirror else 0, dmax)
            if spin_mirror: ax.axvline(0,color="#94A3B8",lw=0.5)
            ax.set_yticks([])
        else:
            ax.axvline(yf,color=self.S.get("fermi_color", "#EF4444"),lw=1.1,ls=self._ef_ls(),label="$E_F$",zorder=5)
            ax.set_xlim(emin,emax)
            if dmax > 0: ax.set_ylim(-dmax if spin_mirror else 0, dmax)
            ax.set_xlabel("E − $E_F$ (eV)" if self.S.get("shift_efermi",True) else "Energy (eV)",fontsize=self.S.get("font_size",11))
            ax.set_ylabel("DOS (states/eV)",fontsize=self.S.get("font_size",11))
        title=self.S.get("title","")
        if title: ax.set_title(title,fontsize=self.S.get("font_size",11)+2,fontweight="bold",pad=8)
        ax.legend(loc="upper right" if not horizontal else "upper left",fontsize=9,framealpha=0.8)
        self._style(ax)

    def plot_stacked_dos(self, ax, ion_labels=None):
        """Plot per-ion stacked DOS (waterfall style)."""
        if not self.data: return
        dos = self.data.get("dos"); ef = self.data["efermi"]
        if dos is None or not dos.get("partial"):
            ax.text(.5,.5,"No partial DOS (LORBIT=11 needed)",transform=ax.transAxes,
                    ha="center",va="center",fontsize=11,color="#888",style="italic"); return
        self._set_font(); ax.clear(); emin,emax=self._ew()
        shift = ef if self.S.get("shift_efermi",True) else 0.
        partial = dos["partial"]
        ions = ion_labels or [f"Ion {i}" for i in range(len(partial))]
        colors = matplotlib.cm.Set2(np.linspace(0,1,len(partial)))
        offset = 0.
        for ii,(ion_data,lbl,col) in enumerate(zip(partial,ions,colors)):
            pd = ion_data.get("spin 1")
            if pd is None: continue
            e = pd[:,0]-shift; mask=(e>=emin)&(e<=emax)
            tdos = pd[:,1:].sum(axis=1) if pd.shape[1]>1 else np.zeros(len(e))
            y = tdos[mask]; e_p = e[mask]
            ax.fill_between(e_p, offset, offset+y, alpha=0.6, color=col, label=lbl)
            ax.plot(e_p, offset+y, lw=0.8, color=col, alpha=0.8)
            offset += y.max()*1.1 if y.max()>0 else 1.
        yf = 0. if self.S.get("shift_efermi",True) else ef
        ax.axvline(yf,color=self.S.get("fermi_color", "#EF4444"),lw=1.,ls=self._ef_ls(),alpha=0.8,label="$E_F$")
        ax.set_xlim(emin,emax); ax.set_ylim(0, None)
        ax.set_xlabel("E − $E_F$ (eV)" if self.S.get("shift_efermi",True) else "Energy (eV)",fontsize=self.S.get("font_size",11))
        ax.set_ylabel("DOS (stacked, offset)",fontsize=self.S.get("font_size",11))
        ax.set_title("Per-ion DOS (stacked)",fontsize=self.S.get("font_size",11)+1,pad=8)
        ax.legend(loc="upper right",fontsize=8,framealpha=0.8); self._style(ax)

    def plot_band_dos(self):
        self.fig.clear()
        gs=gridspec.GridSpec(1,2,width_ratios=[3,1],wspace=0.04,figure=self.fig)
        axb=self.fig.add_subplot(gs[0]); axd=self.fig.add_subplot(gs[1])
        self.plot_bands(axb,show_gap=self.S.get("show_gap",False))
        self.plot_dos(axd,horizontal=True,show_s=self.S.get("dos_s",True),
                      show_p=self.S.get("dos_p",True),show_d=self.S.get("dos_d",True),
                      fill=self.S.get("dos_fill",True),spin_mirror=self.S.get("spin_mirror",False),
                      ion_filter=self.S.get("ion_filter"))
        emin,emax=self._ew(); axd.set_ylim(emin,emax); axd.tick_params(axis="y",labelleft=False); axd.set_ylabel("")
        try:
            self.fig.tight_layout(pad=1.5)
        except Exception:
            self.fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.10, wspace=0.04)

    def plot_bz_auto(self, ax):
        """Plot BZ from actual reciprocal lattice."""
        if not self.data: return
        self._set_font(); ax.clear()
        dk=self.S.get("dark",False); bg="#0f172a" if dk else "#f8fafc"; fg="#e2e8f0" if dk else "#1e293b"
        ax.set_facecolor(bg); self.fig.patch.set_facecolor(bg); ax.set_aspect("equal"); ax.set_axis_off()

        lattice=self.data.get("lattice",np.eye(3))
        rec=reciprocal_lattice(lattice)
        b1,b2,b3=rec
        # Project to 2D (xy plane)
        b1_2d=b1[:2]; b2_2d=b2[:2]
        # BZ boundary = Voronoi cell of reciprocal lattice
        # For hexagonal: draw hexagon
        angle=np.angle(complex(b1_2d[0],b1_2d[1]))
        r=np.linalg.norm(b1_2d)/np.sqrt(3)
        bz_angles=np.linspace(angle,angle+2*np.pi,7)
        bz_pts=np.array([[r*np.cos(a),r*np.sin(a)] for a in bz_angles])
        ax.fill(bz_pts[:,0],bz_pts[:,1],color="#2563EB",alpha=0.08)
        ax.plot(bz_pts[:,0],bz_pts[:,1],color="#2563EB",lw=1.8)

        # Draw k-path
        kd=self.data["kdist"]; kpts=self.data["kpoints"]
        if len(kpts):
            # Project kpoints to cartesian reciprocal coords
            k_cart=kpts@rec
            for ik in range(1,len(k_cart)):
                x0,y0=k_cart[ik-1,:2]; x1,y1=k_cart[ik,:2]
                ax.plot([x0,x1],[y0,y1],color="#EA580C",lw=2.0,alpha=0.8,zorder=4)

        # High-sym points
        for idx,lbl in self.data["klabels"]:
            if idx<len(kpts):
                kc=kpts[idx]@rec
                x,y=kc[:2]
                ax.plot(x,y,"o",color="#DC2626",ms=6,zorder=5)
                ax.text(x+r*0.07,y+r*0.07,lbl,fontsize=11,color=fg,fontweight="bold",zorder=6)

        ax.set_title("Brillouin Zone (auto)",fontsize=self.S.get("font_size",11),color=fg,pad=8)
        # Reciprocal vectors
        origin=np.zeros(2)
        for i,(bv,lbl) in enumerate([(b1_2d,"$b_1$"),(b2_2d,"$b_2$")]):
            ax.annotate("",xy=bv*0.45,xytext=origin,
                        arrowprops=dict(arrowstyle="->",color="#F59E0B",lw=1.8))
            ax.text(bv[0]*0.5,bv[1]*0.5,lbl,fontsize=10,color="#F59E0B",ha="center")
        lim=np.linalg.norm(b1_2d)*1.3
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)

    def plot_optical_props(self, ax_dict, broadening=0.1, which="all"):
        """Plot optical properties in provided axes dictionary."""
        if not self.data: return
        self._set_font()
        ev=self._se(); emin,emax=self._ew()
        energies=np.linspace(0.01,max(emax-emin,8),600)
        opt=compute_optical_spectrum(energies,ev[0],self.data.get("occupancies",None),broadening)
        dk=self.S.get("dark",False); fg="#e2e8f0" if dk else "#1e293b"

        props = [
            ("sigma1","Re[σ] (arb.)","Optical Conductivity","#2563EB","#2563EB"),
            ("sigma2","Im[σ] (arb.)","Optical Conductivity Im","#DC2626","#DC2626"),
            ("eps1","Re[ε]","Dielectric Function","#16A34A","#16A34A"),
            ("eps2","Im[ε]","Dielectric Function Im","#EA580C","#EA580C"),
            ("n","n","Refractive Index","#7C3AED","#7C3AED"),
            ("k","k (extinction)","Extinction Coefficient","#D97706","#D97706"),
            ("absorption","α (arb.)","Absorption Coefficient","#EF4444","#EF4444"),
            ("eels","EELS (-Im 1/ε)","Electron Energy Loss","#0891B2","#0891B2"),
            ("reflectivity","R","Reflectivity","#BE185D","#BE185D"),
        ]
        for key,ylabel,title,col,fill_col in props:
            if key not in ax_dict: continue
            ax=ax_dict[key]; ax.clear()
            y=opt[key]
            ax.plot(energies,y,lw=1.5,color=col); ax.fill_between(energies,0,y,alpha=0.2,color=fill_col)
            ax.set_xlabel("Photon energy (eV)",fontsize=self.S.get("font_size",11)-1)
            ax.set_ylabel(ylabel,fontsize=self.S.get("font_size",11)-1)
            ax.set_title(title,fontsize=self.S.get("font_size",11),pad=6)
            ax.set_xlim(0,energies[-1]); ax.grid(True,alpha=0.12,lw=0.5)
            self._style(ax)

    def plot_spin_texture(self, ax):
        """Plot band structure colored by spin expectation value (for SP calcs)."""
        if not self.data: return
        self._set_font()
        ev = self._se(); emin,emax = self._ew(); kd = self.data["kdist"]
        ns,nk,nb = ev.shape
        ax.clear(); self._klines(ax)
        if ns < 2:
            ax.text(.5,.5,"Spin texture requires spin-polarized calculation\n(ISPIN=2 in INCAR)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#888", style="italic")
            self._style(ax); return
        lw = self.S.get("linewidth",1.5)
        band_indices = list(self._band_iter(nb))
        # Color by (E_up - E_dn) / 2 — spin splitting
        cmap = cm.get_cmap("bwr"); norm = Normalize(vmin=-1, vmax=1)
        for ib in band_indices:
            e_up = ev[0,:,ib]; e_dn = ev[1,:,ib]
            e_avg = (e_up + e_dn) / 2
            if e_avg.max() < emin-.5 or e_avg.min() > emax+.5: continue
            splitting = e_up - e_dn  # positive = up higher, negative = dn higher
            split_norm = np.clip(splitting / (np.abs(splitting).max()+1e-10), -1, 1)
            pts = np.array([kd, e_avg]).T.reshape(-1,1,2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=norm, lw=lw+0.3, rasterized=self._rast())
            lc.set_array((split_norm[:-1]+split_norm[1:])/2)
            ax.add_collection(lc)
        if self.S.get("show_fermi",True):
            ax.axhline(self._efl(), color=self.S.get("fermi_color", "#EF4444"), lw=1., ls=self._ef_ls(), alpha=0.8, zorder=3)
        self._kticks(ax); ax.set_ylim(emin,emax); ax.set_xlim(kd[0],kd[-1])
        ax.set_ylabel(self._ylbl(), fontsize=self.S.get("font_size",11))
        cb = self.fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax, pad=0.01, fraction=0.025, aspect=30)
        cb.set_label("Spin splitting (↑−↓) / max", fontsize=9); cb.ax.tick_params(labelsize=8)
        title = self.S.get("title","") or "Spin-polarized band structure"
        ax.set_title(title, fontsize=self.S.get("font_size",11)+1, pad=8)
        self._style(ax)

    def plot_band_velocity(self, ax):
        """Plot band group velocity dE/dk as a heatmap overlay on bands."""
        if not self.data: return
        self._set_font()
        ev = self._se(); emin,emax = self._ew(); kd = self.data["kdist"]
        ns,nk,nb = ev.shape; ax.clear(); self._klines(ax)
        cmap = cm.get_cmap("RdYlGn"); band_indices = list(self._band_iter(nb))
        all_vels = []
        for ib in band_indices:
            e = ev[0,:,ib]
            if e.max()<emin-.5 or e.min()>emax+.5: continue
            v = np.gradient(e, kd)
            all_vels.extend(v.tolist())
        vmax = np.percentile(np.abs(all_vels), 95) if all_vels else 1.
        norm = Normalize(vmin=-vmax, vmax=vmax)
        for ib in band_indices:
            e = ev[0,:,ib]
            if e.max()<emin-.5 or e.min()>emax+.5: continue
            v = np.gradient(e, kd)
            pts = np.array([kd,e]).T.reshape(-1,1,2)
            segs = np.concatenate([pts[:-1],pts[1:]],axis=1)
            lc = LineCollection(segs,cmap=cmap,norm=norm,lw=2.0,rasterized=self._rast())
            lc.set_array((v[:-1]+v[1:])/2); ax.add_collection(lc)
        if self.S.get("show_fermi",True):
            ax.axhline(self._efl(),color=self.S.get("fermi_color", "#EF4444"),lw=1.,ls=self._ef_ls(),alpha=0.8,zorder=3)
        self._kticks(ax); ax.set_ylim(emin,emax); ax.set_xlim(kd[0],kd[-1])
        ax.set_ylabel(self._ylbl(),fontsize=self.S.get("font_size",11))
        cb=self.fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,pad=0.01,fraction=0.025,aspect=30)
        cb.set_label("Group velocity dE/dk (eV/Å⁻¹)",fontsize=9); cb.ax.tick_params(labelsize=8)
        ax.set_title(self.S.get("title","") or "Band group velocity",fontsize=self.S.get("font_size",11)+1,pad=8)
        self._style(ax)

    def zoom_to_gap(self, margin=1.5):
        """Return (emin,emax) centred around the band gap."""
        if not self.data: return None, None
        ev = self._se()
        info = find_band_gap(ev)
        if info["type"] == "metal": return None, None
        mid = (info["vbm"] + info["cbm"]) / 2
        half = max((info["cbm"] - info["vbm"]) * 2.5, margin)
        return round(mid - half, 2), round(mid + half, 2)

    def plot_fermi_surface(self,ax,band_idx=0):
        if not self.data: return
        ev=self._se(); ax.clear(); kpts=self.data["kpoints"]
        kx=np.unique(kpts[:,0]); ky=np.unique(kpts[:,1])
        if len(kx)<3 or len(ky)<3:
            ax.text(.5,.5,"Need 2D k-mesh\n(dense Monkhorst-Pack grid)",
                    transform=ax.transAxes,ha="center",va="center",fontsize=11,color="#888",style="italic")
            self._style(ax); return
        try:
            Z=ev[0,:,band_idx].reshape(len(ky),len(kx))
            cs=ax.contourf(kx,ky,Z,levels=25,cmap=self.S.get("cmap","RdBu_r"))
            ax.contour(kx,ky,Z,levels=[0.],colors="#EF4444",linewidths=2.0)
            self.fig.colorbar(cs,ax=ax,label="E−E_F (eV)",fraction=0.04,pad=0.02)
            ax.set_xlabel("$k_x$ (r.l.u.)"); ax.set_ylabel("$k_y$ (r.l.u.)")
            ax.set_title(f"Fermi surface — band {band_idx+1}",pad=8)
        except Exception as e:
            ax.text(.5,.5,f"Error: {e}",transform=ax.transAxes,ha="center",va="center",fontsize=10,color="#888",style="italic")
        self._style(ax)

    def plot_wannier_compare(self,ax,wannier_data):
        if not self.data or wannier_data is None: return
        self._set_font()
        ev=self._se(); emin,emax=self._ew()
        kd_vasp=self.data["kdist"]; kd_w=wannier_data["kdist"].copy()
        ev_w=wannier_data["eigenvalues"]
        if kd_w.max()>0: kd_w=kd_w/kd_w.max()*kd_vasp.max()
        ax.clear(); self._klines(ax)
        lw=self.S.get("linewidth",1.5); spin_colors=self.S.get("spin_colors",self.spin_colors)
        band_indices=list(self._band_iter(ev.shape[2]))
        for ib in band_indices:
            e=ev[0,:,ib]
            if e.max()<emin-.5 or e.min()>emax+.5: continue
            ax.plot(kd_vasp,e,color=spin_colors[0],lw=lw,alpha=0.7,rasterized=self._rast(),zorder=2,label="VASP DFT" if ib==band_indices[0] else "")
        w_col=self.S.get("wannier_color","#DC2626")
        for ib in range(ev_w.shape[1]):
            e=ev_w[:,ib]
            if e.max()<emin-.5 or e.min()>emax+.5: continue
            ax.plot(kd_w,e,color=w_col,lw=lw*0.85,alpha=0.9,ls="--",rasterized=self._rast(),zorder=3,label="Wannier90" if ib==0 else "")
        if self.S.get("show_fermi",True): ax.axhline(self._efl(),color=self.S.get("fermi_color", "#EF4444"),lw=1.,ls=self._ef_ls(),alpha=0.8,zorder=4)
        self._kticks(ax); ax.set_ylim(emin,emax)
        ax.set_ylabel(self._ylbl(),fontsize=self.S.get("font_size",11))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.legend(fontsize=10,framealpha=0.9,loc="upper right")
        title=self.S.get("title","") or "VASP vs Wannier90"
        ax.set_title(title,fontsize=self.S.get("font_size",11)+1,pad=8)
        self._style(ax)


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER BUILDER  (ZrX₂-inspired, full featured)
# ══════════════════════════════════════════════════════════════════════════════
