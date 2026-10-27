"""
lidar — point-cloud download / processing package.

On Windows, puts the running conda env's Library\\bin first on PATH before
any submodule imports PDAL.  PDAL resolves its plugins' DLLs through PATH, and
when the env's python.exe is run without `conda activate`, a PATH that also
lists another conda install's Library\\bin (base anaconda3, mamba) makes PDAL
load *that* install's older libprotobuf / abseil / libomp.  Anything imported
afterwards that needs the env's versions — PyTorch — then fails with
"DLL load failed ... The specified procedure could not be found".
No-op when the env is activated (its Library\\bin is already first).
"""

import os
import sys


def _prefer_env_dlls() -> None:
    if os.name != "nt":
        return
    env_bin = os.path.join(sys.prefix, "Library", "bin")
    if not os.path.isdir(env_bin):
        return
    key = os.path.normcase(env_bin)
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if parts and os.path.normcase(parts[0]) == key:
        return
    os.environ["PATH"] = os.pathsep.join([env_bin] + [p for p in parts if os.path.normcase(p) != key])


_prefer_env_dlls()
