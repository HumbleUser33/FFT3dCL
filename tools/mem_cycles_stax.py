# -*- coding: utf-8 -*-
# Create, render and destroy the filter several times in one process and report
# working set, private committed memory and TOTAL GPU memory after each cycle.
# Run it with StaxRip's own VapourSynth so the host matches:
#   "D:\StaxRip-v2.52.6-x64\Apps\FrameServer\VapourSynth\python.exe" mem_cycles_stax.py 4
# Pass --no-clear to keep the frame cache between cycles, like StaxRip does.
import ctypes
import gc
import subprocess
import sys

import vapoursynth as vs

core = vs.core


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("pf", ctypes.c_uint32)] + \
               [("q" + str(i), ctypes.c_uint64) for i in range(11)]


def proc_mem():
    p = PMC()
    p.cb = ctypes.sizeof(p)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(p), p.cb)
    return p.q1 / 1048576.0, p.q6 / 1048576.0


def vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True,
                         text=True).stdout
    try:
        return float(out.strip().splitlines()[0])
    except Exception:
        return -1.0


SRC = r"D:\Test Monkeys\Sample.mkv"
IDX = r"D:\Test Monkeys\Sample.mkv_temp\temp.ffindex"
PLUGIN = r"D:\StaxRip-v2.52.6-x64\Settings\Plugins\VapourSynth\fft3dcl.dll"
FFMS2 = r"D:\StaxRip-v2.52.6-x64\Apps\Plugins\Dual\ffms2\ffms2.dll"
PARAMS = dict(sigma=3, bt=4, bw=32, bh=32, ow=16, oh=16, sharpen=0.4)
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CLEAR = "--no-clear" not in sys.argv

core.std.LoadPlugin(PLUGIN)
core.std.LoadPlugin(FFMS2)

ws, pr = proc_mem()
g = vram_mib()
print(f"start        : working={ws:7.1f}  private={pr:7.1f}  vram={g:7.1f} MiB")
prev = (ws, pr, g)
for trial in range(1, TRIALS + 1):
    c = core.ffms2.Source(source=SRC, cachefile=IDX)
    c = c.std.Trim(750, 765)
    c = core.fft3dcl.FFT3DCL(c, **PARAMS)
    for i in range(c.num_frames):
        c.get_frame(i)
    ws, pr = proc_mem()
    g = vram_mib()
    print(f"  trial {trial} render : working={ws:7.1f}  private={pr:7.1f}  vram={g:7.1f} MiB")
    if CLEAR:
        core.clear_cache()
    del c
    gc.collect()
    ws, pr = proc_mem()
    g = vram_mib()
    print(f"  trial {trial} freed : working={ws:7.1f}  private={pr:7.1f}  vram={g:7.1f} MiB"
          f"   (working {ws - prev[0]:+.0f}, private {pr - prev[1]:+.0f}, vram {g - prev[2]:+.0f})")
    prev = (ws, pr, g)
