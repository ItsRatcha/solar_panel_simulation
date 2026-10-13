"""firmware_ref/mppt.c must take the same duty decisions as the Python MPPTs.

The C file is compiled in double precision (MPPT_USE_DOUBLE) into a shared
library and driven through ctypes with the measurements the Python algorithm
saw.  Skipped when no C compiler (gcc / clang / cc on PATH, or MSVC found via
vswhere) is available.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pvsim.converter import make_converter
from pvsim.load.electronic_load import ElectronicLoad
from pvsim.mppt import create_mppt
from pvsim.mppt.harness import run_fixed_conditions
from pvsim.pv.datasheet import PanelDatasheet
from pvsim.pv.panel import Panel
from pvsim.sensors.ina226 import INA226, INA226Config

SRC = Path(__file__).resolve().parents[1] / "firmware_ref" / "mppt.c"
ALGOS = {"po": 0, "po_variable": 1, "inccond": 2, "inccond_variable": 3, "global_scan": 4, "manual": 5}

_d, _i = ctypes.c_double, ctypes.c_int


class CParams(ctypes.Structure):
    _fields_ = [("d_init", _d), ("d_min", _d), ("d_max", _d), ("step", _d),
                ("n_gain", _d), ("step_min", _d), ("step_max", _d),
                ("eps_g", _d), ("eps_v", _d), ("eps_i", _d),
                ("scan_points", _i), ("scan_period", _i), ("dp_trigger", _d), ("ref_alpha", _d),
                ("scan_on_start", _i), ("duty", _d)]


class CState(ctypes.Structure):
    _fields_ = [("d", _d), ("v_prev", _d), ("i_prev", _d), ("p_prev", _d), ("n", _i),
                ("dir", _i), ("mode", _i), ("scan_idx", _i), ("best_p", _d), ("best_d", _d),
                ("since_scan", _i), ("p_ref", _d), ("n_scans", _i)]


class CMppt(ctypes.Structure):
    _fields_ = [("algo", _i), ("p", CParams), ("s", CState)]


def _msvc_vcvars() -> Path | None:
    if os.name != "nt":
        return None
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                   "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not vswhere.exists():
        return None
    out = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                          "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                          "-property", "installationPath"], capture_output=True, text=True)
    bits = "64" if struct.calcsize("P") == 8 else "32"
    for line in out.stdout.splitlines():
        bat = Path(line.strip(), "VC", "Auxiliary", "Build", f"vcvars{bits}.bat")
        if bat.exists():
            return bat
    return None


def _build(out_dir: Path) -> Path | None:
    ext = ".dll" if os.name == "nt" else (".dylib" if sys.platform == "darwin" else ".so")
    lib = out_dir / f"mppt_ref{ext}"
    defs = ["-DMPPT_USE_DOUBLE", "-DMPPT_BUILD_DLL"]
    for cc in ("gcc", "clang", "cc"):
        exe = shutil.which(cc)
        if exe:
            r = subprocess.run([exe, "-shared", "-O2", "-ffp-contract=off", *defs,
                                *([] if os.name == "nt" else ["-fPIC"]),
                                "-o", str(lib), str(SRC), *([] if os.name == "nt" else ["-lm"])],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return lib
    vcvars = _msvc_vcvars()
    if vcvars is not None:
        bat = out_dir / "build.bat"
        bat.write_text(f'@call "{vcvars}" >nul\r\n'
                       f'cl /nologo /LD /O2 /fp:strict /DMPPT_USE_DOUBLE /DMPPT_BUILD_DLL '
                       f'"{SRC}" /Fe:"{lib}" /Fo:"{out_dir}\\\\"\r\n')
        r = subprocess.run(["cmd", "/c", str(bat)], capture_output=True, text=True, cwd=out_dir)
        if r.returncode == 0 and lib.exists():
            return lib
    return None


@pytest.fixture(scope="module")
def clib(tmp_path_factory):
    lib_path = _build(tmp_path_factory.mktemp("mppt_ref"))
    if lib_path is None:
        pytest.skip("no C compiler found for firmware_ref/mppt.c")
    lib = ctypes.CDLL(str(lib_path))
    lib.mppt_default_params.argtypes = [ctypes.POINTER(CParams)]
    lib.mppt_init.argtypes = [ctypes.POINTER(CMppt), _i, ctypes.POINTER(CParams)]
    lib.mppt_init.restype = _i
    lib.mppt_reset.argtypes = [ctypes.POINTER(CMppt)]
    lib.mppt_step.argtypes = [ctypes.POINTER(CMppt), _d, _d]
    lib.mppt_step.restype = _d
    assert lib.mppt_sizeof_params() == ctypes.sizeof(CParams)
    assert lib.mppt_sizeof() == ctypes.sizeof(CMppt)
    return lib


def _c_algo(lib, name: str, py_params: dict) -> CMppt:
    p = CParams()
    lib.mppt_default_params(ctypes.byref(p))
    for k, v in py_params.items():
        setattr(p, k, v)
    m = CMppt()
    assert lib.mppt_init(ctypes.byref(m), ALGOS[name], ctypes.byref(p)) == 0
    return m


def _replay(lib, name: str, algo, v_seq, i_seq) -> tuple[np.ndarray, np.ndarray]:
    """Duty sequences (before each step) of the Python and C algorithms fed v_seq, i_seq."""
    algo.reset()
    m = _c_algo(lib, name, algo.params_dict())
    py, c = [algo.duty], [m.s.d]
    for v, i in zip(v_seq, i_seq):
        py.append(algo.step(float(v), float(i)))
        c.append(lib.mppt_step(ctypes.byref(m), float(v), float(i)))
    return np.array(py), np.array(c)


def test_c_defaults_match_python(clib):
    p = CParams()
    clib.mppt_default_params(ctypes.byref(p))
    for name in ALGOS:
        for k, v in create_mppt(name).params_dict().items():
            assert getattr(p, k) == pytest.approx(v, rel=0, abs=0), (name, k)


@pytest.fixture(scope="module")
def panel():
    return Panel(PanelDatasheet())


@pytest.mark.parametrize("name", list(ALGOS))
@pytest.mark.parametrize("case", ["uniform", "shaded"])
def test_c_matches_python_closed_loop(clib, panel, name, case):
    """Measurements from a closed-loop run (noisy INA226) replayed into both."""
    g = 900.0 if case == "uniform" else (1000.0, 1000.0, 250.0)
    curve = panel.curve(g, 35.0)
    kw = {"d_init": 0.3}
    if name == "global_scan":
        kw.update(scan_period=150, scan_points=25)
    algo = create_mppt(name, **kw)
    tr = run_fixed_conditions(curve, make_converter("buck"), ElectronicLoad("CR", 3.0), algo, 500,
                              sensor=INA226(INA226Config(), seed=11))
    py, c = _replay(clib, name, algo, tr.v_meas, tr.i_meas)
    np.testing.assert_array_equal(py[:-1], tr.duty)  # replay reproduces the run
    np.testing.assert_allclose(c, py, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", list(ALGOS))
def test_c_matches_python_random(clib, name):
    """Random measurements, including V = 0 (collapse) and repeated V (dV = 0) samples."""
    rng = np.random.default_rng(7)
    n = 3000
    v = rng.uniform(0.0, 40.0, n)
    i = rng.uniform(0.0, 9.0, n)
    v[rng.random(n) < 0.05] = 0.0
    rep = np.flatnonzero(rng.random(n) < 0.1)
    v[rep[rep > 0]] = v[rep[rep > 0] - 1]
    kw = dict(d_init=0.4, d_min=0.05, d_max=0.9, step=0.01)
    if name == "global_scan":
        kw.update(scan_period=60, scan_points=12, dp_trigger=0.3)
    if name == "inccond":
        kw.update(eps_g=0.01)
    py, c = _replay(clib, name, create_mppt(name, **kw), v, i)
    np.testing.assert_allclose(c, py, rtol=0, atol=1e-12)
