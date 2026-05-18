"""
VaspViz — parsers.py
Parsers: vasprun.xml, POSCAR/CONTCAR, Wannier90 bands, OUTCAR.
No GUI dependencies — pure Python + numpy.
"""
import re, csv
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

from constants import VALENCE_ELECTRONS, KNAME_MAP

class VasprunParser:
    def __init__(self, fp): self.fp = fp

    def parse(self):
        tree = ET.parse(self.fp); root = tree.getroot()
        efermi = self._efermi(root)
        eigs, occs = self._eigenvalues(root)
        kpts, kdist, klabels = self._kpoints(root, eigs.shape[1])
        ions = self._ions(root)
        lattice = self._lattice(root)
        incar   = self._incar(root)
        # Compute electron counts
        n_elec = self._count_electrons(ions, incar)
        return {
            "filepath":self.fp, "system":self._system(root),
            "efermi":efermi, "kpoints":kpts, "kdist":kdist, "klabels":klabels,
            "eigenvalues":eigs, "occupancies":occs,
            "projections":self._projections(root),
            "dos":self._dos(root), "ions":ions,
            "lattice":lattice, "incar":incar,
            "spin_polarized":eigs.shape[0]==2,
            "nbands":eigs.shape[2], "nkpoints":eigs.shape[1],
            "n_electrons":n_elec, "n_ions":len(ions),
        }

    def _system(self,r):
        for i in r.iter("i"):
            if i.get("name")=="SYSTEM": return (i.text or "").strip() or Path(self.fp).stem
        return Path(self.fp).stem

    def _efermi(self,r):
        for calc in reversed(list(r.iter("calculation"))):
            d=calc.find("dos")
            if d is not None:
                ef=d.find("i[@name='efermi']")
                if ef is not None: return float(ef.text)
        for i in r.iter("i"):
            if i.get("name")=="efermi": return float(i.text)
        return 0.0

    def _lattice(self,r):
        for calc in reversed(list(r.findall(".//calculation"))):
            sc=calc.find(".//structure/crystal/varray[@name='basis']")
            if sc is not None:
                return np.array([[float(x) for x in v.text.split()] for v in sc.findall("v")])
        return np.eye(3)*3.0

    def _incar(self,r):
        d={}
        for i in r.iter("i"):
            n=i.get("name","")
            if n and i.text: d[n]=i.text.strip()
        return d

    def _kpoints(self,r,nk_eig):
        sec=r.find(".//kpoints"); kpts=None
        if sec is not None:
            for va in sec.findall("varray"):
                if va.get("name")=="kpointlist":
                    kpts=np.array([[float(x) for x in v.text.split()] for v in va.findall("v")])
        if kpts is None or not len(kpts):
            return np.zeros((nk_eig,3)),np.linspace(0,1,nk_eig),[]
        diffs=np.diff(kpts,axis=0)
        sd=np.sqrt((diffs**2).sum(axis=1))
        kdist=np.concatenate([[0.],np.cumsum(sd)])
        return kpts,kdist,self._klabels(sec,kpts,kdist,sd)

    def _klabels(self,sec,kpts,kdist,sd):
        gen=sec.find("generation"); nk=len(kpts)
        if gen is not None:
            named=[(v.get("name","").strip(),v.text) for v in gen.findall("v") if v.get("name","").strip()]
            if named:
                out=[]
                for name,txt in named:
                    if not txt: continue
                    c=np.array([float(x) for x in txt.split()])
                    idx=int(np.argmin(np.linalg.norm(kpts-c,axis=1)))
                    lbl=KNAME_MAP.get(name,name)
                    if out and out[-1][0]==idx:
                        p=out[-1][1]; out[-1]=(idx,f"{p}|{lbl}" if lbl!=p else p)
                    else: out.append((idx,lbl))
                if out: return out
            div=gen.find("i[@name='divisions']")
            if div is not None:
                nd=int(div.text.strip()); ns=nk//nd
                return [(min(i*nd,nk-1),"") for i in range(ns+1)]
        out=[(0,"")]
        for i in range(1,len(sd)):
            if sd[i-1]<1e-7:
                pi=i-1
                if not(out and out[-1][0]==pi): out.append((pi,""))
                out.append((i,""))
        out.append((nk-1,""))
        seen=set(); unique=[]
        for idx,l in out:
            if idx not in seen: seen.add(idx); unique.append((idx,l))
        return unique

    def _eigenvalues(self,r):
        calc=self._last(r); blk=calc.find(".//eigenvalues/array/set")
        if blk is None: return np.zeros((1,1,1)),np.zeros((1,1,1))
        ae,ao=[],[]
        for ss in blk.findall("set"):
            ke,ko=[],[]
            for ks in ss.findall("set"):
                e,o=[],[]
                for rv in ks.findall("r"):
                    v=rv.text.split(); e.append(float(v[0])); o.append(float(v[1]) if len(v)>1 else 0.)
                ke.append(e); ko.append(o)
            ae.append(ke); ao.append(ko)
        return np.array(ae),np.array(ao)

    def _projections(self,r):
        calc=self._last(r); blk=calc.find(".//projected/array/set")
        if blk is None: return None
        sp=[]
        for ss in blk.findall("set"):
            kd=[]
            for ks in ss.findall("set"):
                bd=[]
                for bs in ks.findall("set"):
                    bd.append([[float(x) for x in rv.text.split()] for rv in bs.findall("r")])
                kd.append(bd)
            sp.append(kd)
        try: return np.array(sp)
        except: return None

    def _dos(self,r):
        calc=self._last(r); db=calc.find(".//dos")
        if db is None: return None
        res={"total":{},"partial":[]}
        ts=db.find("total/array/set")
        if ts:
            for ss in ts.findall("set"):
                spin=ss.get("comment","spin 1")
                res["total"][spin]=np.array([[float(x) for x in rv.text.split()] for rv in ss.findall("r")])
        ps=db.find("partial/array/set")
        if ps:
            for ion_set in ps.findall("set"):
                id_={}
                for ss in ion_set.findall("set"):
                    spin=ss.get("comment","spin 1")
                    id_[spin]=np.array([[float(x) for x in rv.text.split()] for rv in ss.findall("r")])
                res["partial"].append(id_)
        return res

    def _ions(self,r):
        for arr in r.iter("array"):
            if arr.get("name")=="atoms":
                return [rv.text.strip().split()[0] for rv in arr.findall(".//r") if rv.text and rv.text.strip()]
        return []

    def _count_electrons(self, ions, incar):
        """Estimate number of valence electrons."""
        total = 0
        for ion in ions:
            total += VALENCE_ELECTRONS.get(ion, 0)
        # Check NELECT in INCAR
        if "NELECT" in incar:
            try: return int(float(incar["NELECT"]))
            except: pass
        return total

    def _last(self,r):
        c=r.findall(".//calculation"); return c[-1] if c else r


class PoscarParser:
    def __init__(self, fp): self.fp = fp

    def parse(self):
        lines = Path(self.fp).read_text().splitlines()
        comment = lines[0].strip()
        scale   = float(lines[1].strip())
        a1 = np.array([float(x) for x in lines[2].split()])
        a2 = np.array([float(x) for x in lines[3].split()])
        a3 = np.array([float(x) for x in lines[4].split()])
        lattice = np.array([a1,a2,a3]) * scale
        idx = 5
        has_species = not lines[idx].strip()[0].isdigit()
        if has_species:
            species = lines[idx].split(); idx += 1
        else:
            species = []
        counts = [int(x) for x in lines[idx].split()]; idx += 1
        if not species: species = [f"X{i}" for i in range(len(counts))]
        coord_type = lines[idx].strip().lower(); idx += 1
        if coord_type.startswith("s"):
            coord_type = lines[idx].strip().lower(); idx += 1
        positions = []
        for line in lines[idx:]:
            parts = line.split()
            if len(parts) < 3: continue
            try: positions.append([float(x) for x in parts[:3]])
            except: break
        frac_pos = np.array(positions) if positions else np.zeros((0,3))
        cart_pos = frac_pos @ lattice if len(frac_pos) else np.zeros((0,3))
        if coord_type.startswith("c"):
            cart_pos = frac_pos.copy()
            try: frac_pos = cart_pos @ np.linalg.inv(lattice)
            except: pass
        ion_labels = []
        for sp, cnt in zip(species, counts):
            ion_labels.extend([sp]*cnt)
        vol = abs(np.dot(a1*scale, np.cross(a2*scale, a3*scale)))
        return {
            "comment":comment, "lattice":lattice, "scale":scale,
            "species":species, "counts":counts, "ion_labels":ion_labels,
            "frac_positions":frac_pos, "cart_positions":cart_pos,
            "coord_type":coord_type, "filepath":self.fp,
            "total_atoms":sum(counts), "volume":vol,
            "a":np.linalg.norm(a1*scale), "b":np.linalg.norm(a2*scale), "c":np.linalg.norm(a3*scale),
            "alpha":np.degrees(np.arccos(np.dot(a2,a3)/(np.linalg.norm(a2)*np.linalg.norm(a3)))),
            "beta": np.degrees(np.arccos(np.dot(a1,a3)/(np.linalg.norm(a1)*np.linalg.norm(a3)))),
            "gamma":np.degrees(np.arccos(np.dot(a1,a2)/(np.linalg.norm(a1)*np.linalg.norm(a2)))),
        }


class Wannier90Parser:
    def __init__(self, fp): self.fp = fp

    def parse(self):
        lines = [l for l in Path(self.fp).read_text().splitlines() if not l.strip().startswith("#")]
        bands_raw = []; current = []
        for l in lines:
            l = l.strip()
            if not l:
                if current: bands_raw.append(current); current = []
            else:
                parts = l.split()
                if len(parts) >= 2:
                    try: current.append((float(parts[0]), float(parts[1])))
                    except: pass
        if current: bands_raw.append(current)
        if not bands_raw: raise ValueError("No band data found")
        nk = len(bands_raw[0])
        kdist = np.array([pt[0] for pt in bands_raw[0]])
        nb = len(bands_raw)
        ev = np.zeros((nk, nb))
        for ib, band in enumerate(bands_raw):
            for ik, (k, e) in enumerate(band):
                if ik < nk: ev[ik, ib] = e
        return {"filepath":self.fp, "kdist":kdist, "eigenvalues":ev,
                "nkpoints":nk, "nbands":nb, "system":Path(self.fp).stem}


class ProcarParser:
    def __init__(self, fp): self.fp = fp
    
    def parse(self):
        """Parse VASP PROCAR for orbital weights."""
        with open(self.fp, 'r') as f:
            lines = f.readlines()
        
        # Header info
        # PROCAR lm decomposed
        # # of k-points:   100         # of bands:  20         # of ions:   6
        info_line = lines[1]
        try:
            m = re.findall(r'\d+', info_line)
            nkpts = int(m[0])
            nbands = int(m[1])
            nions = int(m[2])
        except:
            raise ValueError("Failed to parse PROCAR header")
            
        is_spin_polarized = False
        if len(lines) > 3 and "spin component 1" in lines[3].lower():
            is_spin_polarized = True
            
        data = []
        orbitals = []
        current_k = 0
        current_b = 0
        
        weights = np.zeros((1 if not is_spin_polarized else 2, nkpts, nbands, nions, 9)) # up to 9 orbitals (s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2)
        energies = np.zeros((1 if not is_spin_polarized else 2, nkpts, nbands))
        
        spin_idx = 0
        idx = 2
        while idx < len(lines):
            line = lines[idx].strip()
            if "spin component" in line:
                spin_idx = int(line.split()[-1]) - 1
                idx += 1; continue
            if "k-point" in line:
                # k-point    1 :    0.00000000 0.00000000 0.00000000     weight = 0.01000000
                current_k = int(line.split()[1]) - 1
                idx += 1; continue
            if "band" in line and "energy" in line:
                # band   1 # energy  -6.85239335 # occ.  1.00000000
                parts = line.split()
                current_b = int(parts[1]) - 1
                energies[spin_idx, current_k, current_b] = float(parts[4])
                
                idx += 2 # skip "ion      s     py     pz     px    dxy    dyz    dz2    dxz  dx2-y2    tot"
                if not orbitals:
                    orbitals = lines[idx-1].split()[1:-1] # exclude 'ion' and 'tot'
                
                for i in range(nions):
                    w_parts = lines[idx].split()[1:-1] # exclude ion index and total
                    for o in range(min(len(w_parts), 9)):
                        weights[spin_idx, current_k, current_b, i, o] = float(w_parts[o])
                    idx += 1
                idx += 1 # skip 'tot' line
                continue
            idx += 1
            
        return {
            "filepath": self.fp,
            "nkpoints": nkpts,
            "nbands": nbands,
            "nions": nions,
            "orbitals": orbitals,
            "weights": weights,  # shape: (nspin, nk, nb, nion, norb)
            "energies": energies, # shape: (nspin, nk, nb)
            "spin_polarized": is_spin_polarized
        }


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO DATA
# ══════════════════════════════════════════════════════════════════════════════


def read_outcar_info(filepath):
    """Extract useful info from VASP OUTCAR file."""
    info = {"nkpoints":None,"nbands":None,"nelect":None,"efermi":None,
            "total_energy":None,"forces":None,"stress":None,"timing":None,
            "ionic_steps":0,"warnings":[]}
    try:
        text = Path(filepath).read_text(errors="ignore")
        import re
        # NKPTS
        m = re.search(r"NKPTS\s*=\s*(\d+)", text)
        if m: info["nkpoints"] = int(m.group(1))
        # NBANDS
        m = re.search(r"NBANDS\s*=\s*(\d+)", text)
        if m: info["nbands"] = int(m.group(1))
        # NELECT
        m = re.search(r"NELECT\s*=\s*([\d.]+)", text)
        if m: info["nelect"] = float(m.group(1))
        # Fermi energy (last occurrence)
        for m in re.finditer(r"E-fermi\s*:\s*([-\d.]+)", text):
            info["efermi"] = float(m.group(1))
        # Total energy (last)
        for m in re.finditer(r"TOTEN\s*=\s*([-\d.]+)\s*eV", text):
            info["total_energy"] = float(m.group(1))
        # Ionic steps
        info["ionic_steps"] = len(re.findall(r"Iteration\s+\d+\s*\(\s*1\s*\)", text))
        # Warnings
        for m in re.finditer(r"WARNING:(.*)", text):
            w = m.group(1).strip()
            if w and w not in info["warnings"]: info["warnings"].append(w)
        # Timing
        m = re.search(r"LOOP\+.*?(\d+:\d+:\d+)", text[::-1])
        if not m: m = re.search(r"Total CPU time used.*?([\d.]+)\s*sec", text)
        if m: info["timing"] = m.group(1)
    except Exception as e:
        info["error"] = str(e)
    return info


# ══════════════════════════════════════════════════════════════════════════════
#  CHGCAR / PARCHG PARSER
# ══════════════════════════════════════════════════════════════════════════════

class ChgcarParser:
    def __init__(self, fp):
        self.fp = fp

    def parse(self):
        with open(self.fp, 'r') as f:
            lines = f.readlines()
            
        comment = lines[0].strip()
        scale = float(lines[1].strip())
        a1 = np.array([float(x) for x in lines[2].split()])
        a2 = np.array([float(x) for x in lines[3].split()])
        a3 = np.array([float(x) for x in lines[4].split()])
        lattice = np.array([a1, a2, a3]) * scale
        
        idx = 5
        has_species = not lines[idx].strip()[0].isdigit()
        species = lines[idx].split() if has_species else []
        if has_species: idx += 1
        counts = [int(x) for x in lines[idx].split()]; idx += 1
        total_atoms = sum(counts)
        if not species: species = [f"X{i}" for i in range(len(counts))]
        
        coord_type = lines[idx].strip().lower(); idx += 1
        if coord_type.startswith("s"):
            coord_type = lines[idx].strip().lower(); idx += 1
            
        positions = []
        for i in range(total_atoms):
            parts = lines[idx].split()
            if len(parts) >= 3:
                positions.append([float(x) for x in parts[:3]])
            idx += 1
            
        frac_pos = np.array(positions) if positions else np.zeros((0,3))
        if coord_type.startswith("c"):
            cart_pos = frac_pos.copy()
            try: frac_pos = cart_pos @ np.linalg.inv(lattice)
            except: pass
        else:
            cart_pos = frac_pos @ lattice if len(frac_pos) else np.zeros((0,3))
            
        ion_labels = []
        for sp, cnt in zip(species, counts):
            ion_labels.extend([sp]*cnt)
            
        # Skip empty lines to find grid dims
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
            
        grid_dims = [int(x) for x in lines[idx].split()]
        nx, ny, nz = grid_dims
        idx += 1
        total_pts = nx * ny * nz
        
        def read_block(start_idx, num_pts):
            vals = []
            curr_idx = start_idx
            while curr_idx < len(lines) and len(vals) < num_pts:
                line = lines[curr_idx].strip()
                curr_idx += 1
                if not line or "augmentation" in line.lower(): continue
                # VASP CHGCAR writes numbers separated by spaces
                vals.extend([float(x) for x in line.split()])
            if len(vals) > num_pts: vals = vals[:num_pts]
            return np.array(vals), curr_idx

        chg_total, idx = read_block(idx, total_pts)
        chg_total = chg_total.reshape((nx, ny, nz), order='F')
        
        chg_diff = None
        # Check for spin-polarized magnetization density block
        while idx < len(lines):
            line = lines[idx].strip()
            if line:
                parts = line.split()
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    chk_nx, chk_ny, chk_nz = [int(x) for x in parts]
                    if chk_nx == nx and chk_ny == ny and chk_nz == nz:
                        idx += 1
                        chg_diff, idx = read_block(idx, total_pts)
                        if len(chg_diff) == total_pts:
                            chg_diff = chg_diff.reshape((nx, ny, nz), order='F')
                        else:
                            chg_diff = None
                        break
            idx += 1
            
        return {
            "filepath": self.fp,
            "comment": comment,
            "lattice": lattice,
            "frac_positions": frac_pos,
            "cart_positions": cart_pos,
            "ion_labels": ion_labels,
            "grid": (nx, ny, nz),
            "chg_total": chg_total,
            "chg_diff": chg_diff,
            "is_spin_polarized": chg_diff is not None
        }


# ══════════════════════════════════════════════════════════════════════════════
