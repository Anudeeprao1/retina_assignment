#!/usr/bin/env python3
"""Compare opl_vr.py with the compiled Virtual Retina binary, frame by frame.

usage: python validate_opl.py <retina.xml> <binary_outD> <reps_per_frame> <save_every> <frame1.pgm> [frame2.pgm ...]
  set OPL_BLUR=fast to validate the scipy blur instead of the exact one
  (use the same -r and -nS values that were given to bin/Retina)
"""
import os, sys, glob, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opl_vr import opl_from_xml, set_blur_backend
from inr_view import read_inr

xml, outD, reps, every = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
frames = [np.asarray(Image.open(f).convert("L"), dtype=np.float64) for f in sys.argv[5:]]

set_blur_backend(os.environ.get('OPL_BLUR', 'exact'))
opl, lum = opl_from_xml(xml)
opl.allocate(frames[0].shape, lum / 2.0)             # uniform screen of 127.5 at start

names = {"opl": "oplFrames", "receptor": "receptorFrames", "horizontal": "horizontalFrames"}
ref = {k: sorted(glob.glob(os.path.join(outD, v, "*.inr"))) for k, v in names.items()}

worst = {k: 0.0 for k in names}            # largest absolute difference
peak = {k: 0.0 for k in names}             # largest |binary| value over the whole run
tim, t0, n_cmp = 0, time.time(), 0
for f in frames:
    for _ in range(reps):
        out = opl.step(f)
        if tim % every == 0:
            k = tim // every
            ours = {"opl": out, "receptor": opl.receptor.read(),
                    "horizontal": opl.inhib.read() if opl.inhib else None}
            for key in names:
                if ours[key] is None or k >= len(ref[key]):
                    continue
                theirs = read_inr(ref[key][k])
                worst[key] = max(worst[key], np.abs(ours[key] - theirs).max())
                peak[key] = max(peak[key], np.abs(theirs).max())
            n_cmp += 1
        tim += 1
el = time.time() - t0
print(f"{tim} steps simulated, {n_cmp} saved frames compared, {el:.1f} s ({1000*el/tim:.1f} ms/step)")
for key, v in worst.items():
    print(f"  {names[key]:17s} max |python - binary| = {v:.2e}   "
          f"(relative to the run's peak |binary| {peak[key]:.3f}: {v / (peak[key] or 1):.2e})")
