
"""
split_video_audio.py
--------------------
End-to-end converter that

1. **Loads** a `.aedat` (v1 or v2) file   →  raw addresses + timestamps
2. **Splits** the stream into “retina” (vision) and “cochlea” (audio)
3. **Plots** both streams in 3-D for a quick visual check
4. **Re-writes** two new `.aedat` files:
      visual/<basename>_retina.aedat
      audio /<basename>_cochlea.aedat

Only standard-library + NumPy + Matplotlib are required.
"""

from __future__ import annotations
import os
import struct
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 – registers the “3d” proj


# --------------------------------------------------------------------------- #
#                               Low-level reader                              #
# --------------------------------------------------------------------------- #
def loadaerdat(path: str | os.PathLike,
               max_events: int = int(30e6)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure-Python re-implementation of the original MATLAB **loadaerdat**.

    Parameters
    ----------
    path        : str or Path
        Full path to a *.aedat*/*.dat* file (v1 or v2).
    max_events  : int, optional
        Soft upper bound to avoid reading absurdly large recordings.

    Returns
    -------
    addr : ndarray
        Raw event addresses  (dtype = uint16 for v1, uint32 for v2).
    ts   : ndarray
        Corresponding timestamps in micro-seconds (dtype = uint32).

    Notes
    -----
    *   The file is assumed to be **big-endian** (as per the
        `fread(...,'b')` in the original MATLAB).
    *   Header lines are recognised when they start with the
        character “#”.  If the data section accidentally begins with
        “#”, you must delete or rename that byte beforehand.
    """

    path = Path(path).expanduser().resolve()
    tok  = b'#!AER-DAT'
    version = 0          # default ⇢ v1 (16-bit addr)

    with path.open('rb') as f:
        # ---------- parse header ------------------------------------------------
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:                                     # EOF before data
                raise EOFError("No data section found")
            if line[0:1] != b'#':                            # first data byte
                bof = pos                                    # begin-of-frame
                break
            # line is header → check for version token
            if line.startswith(tok):
                try:
                    version = int(float(line[len(tok):].strip()))
                except ValueError:
                    pass                                     # leave default
            # echo header for info (strip '\r\n')

        if version == 0:
            version = 1      # “no header → assume v1”

        # ---------- compute number of events -----------------------------------
        bytes_per_event = 6 if version == 1 else 8
        f.seek(0, os.SEEK_END)
        file_bytes = f.tell() - bof
        num_events = file_bytes // bytes_per_event
        if num_events > max_events:
            num_events = max_events

        # ---------- load data section ------------------------------------------
        f.seek(bof)                                           # jump to data
        if version == 1:                                      # v1 = 16/32
            dtype = np.dtype([('addr', '>u2'), ('ts', '>u4')])
        else:                                                 # v2 = 32/32
            dtype = np.dtype([('addr', '>u4'), ('ts', '>u4')])

        events = np.fromfile(f, dtype=dtype, count=num_events)
        if events.size != num_events:
            raise IOError("File ended unexpectedly while reading")

    addr = events['addr']
    ts   = events['ts']

    return addr, ts

def read_cochlea(path,
                      t_init: int = 0,
                      t_end: int = int(1e11)):
    """Load the audio stream and return (channel, ts)."""
    inaddr, ints = loadaerdat(path)

    # ------- 2. TEMPORAL WINDOWING -------------------------------------------
    ints = ints - ints.min()                         # normalise to t=0
    keep = (ints > t_init) & (ints < t_end)
    inaddr = inaddr[keep]
    ints   =  ints[keep]

    # ------- 3. CHANNEL SPLIT -------------------------------------------------
    is_cochlea = inaddr > 0
    coch_addr  = inaddr[is_cochlea]
    ts_coch    =  ints[is_cochlea]
    coch       = (coch_addr % 256) + 128            # [0-255] → [128-383]
    y_coch     = np.zeros_like(coch)

    ret_addr   = inaddr[~is_cochlea]
    ts_ret     =  ints[~is_cochlea]
    sign       = ret_addr & 0x0001
    inx        = (ret_addr & 0x00FE) >> 1
    iny        = (ret_addr & 0x7F00) >> 8

    return coch.astype(np.uint16) - 128, ts_coch