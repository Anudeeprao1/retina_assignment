#!/usr/bin/env python3
"""run_image.py -- run the Virtual Retina OPL on one image (any format).

Examples
  python run_image.py photo.jpg
  python run_image.py photo.png --config configs/opl_sharp.xml --duration 300
  python run_image.py photo.heic --pre noise --pre-ms 100          # like the paper's Fig. 10
  python run_image.py                                              # opens a file picker

What happens
  The retina first looks at a uniform gray screen (or noise, with --pre noise),
  then the image appears at t = 0 and stays on for --duration ms. The OPL is
  simulated step by step (dt from the XML) and the outputs are written to
  out/<image name>/ :
    stages.png       input L, center C, surround S, junction output I_OPL (+ sign map)
    onset.png        I_OPL at several times after the image appears
    timecourse.png   how C, S and I_OPL evolve at one pixel, and overall activity
    I_opl_final.png  the junction image at the end (mid-gray = 0)
    I_opl_final.npy  the same, as raw floats
    onset.gif        I_OPL over time (with --gif)
"""
import argparse, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from opl_vr import opl_from_xml, set_blur_backend
from retina_io import load_luminance, signed_to_u8, IMAGE_EXTS


def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", " ".join("*" + e for e in IMAGE_EXTS)), ("All files", "*")])
        root.destroy()
        return path
    except Exception as e:
        sys.exit(f"No image given and no file picker available ({e}). "
                 "Pass the path: python run_image.py <image>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="input image (jpg, png, bmp, tiff, webp, gif, pgm, heic...)")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "opl_fig10.xml"))
    ap.add_argument("--duration", type=float, default=300, help="ms the image stays on (default 300)")
    ap.add_argument("--pre", choices=["gray", "noise"], default="gray",
                    help="what the retina sees before the image (default: uniform gray)")
    ap.add_argument("--pre-ms", type=float, default=100, help="duration of --pre noise in ms")
    ap.add_argument("--max-size", type=int, default=512, help="long side in pixels (0 = full size)")
    ap.add_argument("--ppd", type=float, help="override pixels-per-degree from the XML")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--gif", action="store_true", help="also write onset.gif")
    ap.add_argument("--blur", default="auto", choices=["auto", "numba", "fast", "exact"],
                    help="blur backend: numba/exact are bit-identical to the C++ (default auto)")
    a = ap.parse_args()

    path = a.image or pick_file()
    if not path:
        sys.exit("No image chosen.")
    L = load_luminance(path, a.max_size)
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True)

    backend = set_blur_backend(a.blur)
    opl, lum = opl_from_xml(a.config)
    if a.ppd:
        for f in (opl.excit, opl.under, opl.inhib):
            if f: f.ppd = a.ppd
    dt_ms = opl.excit.dt * 1000
    ppd = opl.excit.ppd
    print(f"image   : {path}  ->  {L.shape[1]}x{L.shape[0]} px, mean luminance {L.mean():.1f}")
    print(f"config  : {os.path.basename(a.config)}  (dt = {dt_ms:g} ms, {ppd:g} px/deg, "
          f"image spans {L.shape[1]/ppd:.2f} x {L.shape[0]/ppd:.2f} deg, blur {backend})")

    opl.allocate(L.shape, lum / 2.0)                 # steady state with a gray screen of 127.5

    # ---- before the image: optional noise of matched luminance (paper Fig. 10) ----
    if a.pre == "noise":
        rng = np.random.default_rng(0)
        n_pre = int(round(a.pre_ms / dt_ms)); hold = max(1, int(round(5 / dt_ms)))
        for k in range(n_pre):
            if k % hold == 0:                         # a new noise frame every ~5 ms
                noise = np.clip(rng.normal(L.mean(), L.std(), L.shape), 0, 255)
            opl.step(noise)
        print(f"pre     : {n_pre} steps of noise (mean {L.mean():.1f}, std {L.std():.1f})")

    # ---- the image is on ----
    n_steps = int(round(a.duration / dt_ms))
    want = [t for t in (10, 20, 30, 50, 90, 150, 300) if t <= a.duration] or [a.duration]
    snaps, gif, trace = {}, [], []
    cy, cx = L.shape[0] // 2, L.shape[1] // 2
    t0 = time.time()
    for k in range(n_steps):
        I = opl.step(L)
        t = (k + 1) * dt_ms                           # ms since the image appeared
        C = opl.receptor.read(); S = opl.inhib.read() if opl.inhib else np.zeros_like(C)
        trace.append((t, C[cy, cx], S[cy, cx], I[cy, cx], np.abs(I).mean()))
        for w in want:
            if w not in snaps and t >= w - 1e-9:
                snaps[w] = (t, I.copy())
        if a.gif and (k % max(1, int(round(5 / dt_ms))) == 0):
            gif.append(I.copy())
        if (k + 1) % max(1, n_steps // 10) == 0:
            print(f"\r  simulating: {t:6.0f} / {a.duration:g} ms", end="", flush=True)
    el = time.time() - t0
    print(f"\nsimulated {n_steps} steps in {el:.1f} s ({1000*el/n_steps:.1f} ms per step)")
    print(f"I_OPL at {t:g} ms: min {I.min():+.4f}  mean {I.mean():+.4f}  max {I.max():+.4f}")

    # ---- outputs ----
    m = np.percentile(np.abs(I), 99.5) or 1.0
    np.save(os.path.join(out, "I_opl_final.npy"), I)
    Image.fromarray(signed_to_u8(I, m)).save(os.path.join(out, "I_opl_final.png"))

    fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))
    panels = [(L, "input  L", "gray", None),
              (C, "center  C  (receptors)", "gray", None),
              (S, "surround  S  (horizontal cells)", "gray", None),
              (I, f"I_OPL  at {t:g} ms  (gray = 0)", "gray", (-m, m)),
              (I, "I_OPL sign:  red +,  blue -", "coolwarm", (-m, m))]
    for axi, (img, title, cmap, lim) in zip(ax, panels):
        axi.imshow(img, cmap=cmap, **({} if lim is None else dict(vmin=lim[0], vmax=lim[1])))
        axi.set_title(title); axi.set_xticks([]); axi.set_yticks([])
    fig.suptitle(f"{name}  -  {os.path.basename(a.config)}", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(out, "stages.png"), dpi=100, bbox_inches="tight"); plt.close()

    keys = sorted(snaps)
    mm = max(np.percentile(np.abs(snaps[k][1]), 99.5) for k in keys) or 1.0
    cols = min(4, len(keys)); rows = int(np.ceil(len(keys) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.4 * rows), squeeze=False)
    for axi in ax.flat: axi.axis("off")
    for axi, k in zip(ax.flat, keys):
        axi.imshow(snaps[k][1], cmap="gray", vmin=-mm, vmax=mm)
        axi.set_title(f"I_OPL   t = {snaps[k][0]:g} ms")
    fig.suptitle(f"after onset (from {a.pre}), shared scale", y=1.0)
    plt.tight_layout(); plt.savefig(os.path.join(out, "onset.png"), dpi=90, bbox_inches="tight"); plt.close()

    tr = np.array(trace)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    ax[0].plot(tr[:, 0], tr[:, 1], label="C  center"); ax[0].plot(tr[:, 0], tr[:, 2], label="S  surround")
    ax[0].plot(tr[:, 0], tr[:, 3], label="I_OPL = C - S"); ax[0].axhline(0, color="gray", lw=0.6)
    ax[0].set_title(f"one pixel ({cx},{cy}) over time"); ax[0].set_xlabel("ms after onset"); ax[0].legend()
    ax[1].plot(tr[:, 0], tr[:, 4], color="k")
    ax[1].set_title("overall activity: mean |I_OPL|"); ax[1].set_xlabel("ms after onset")
    plt.tight_layout(); plt.savefig(os.path.join(out, "timecourse.png"), dpi=90, bbox_inches="tight"); plt.close()

    if a.gif and gif:
        g = max(np.percentile(np.abs(f), 99.5) for f in gif) or 1.0
        fr = [Image.fromarray(signed_to_u8(f, g)) for f in gif]
        fr[0].save(os.path.join(out, "onset.gif"), save_all=True, append_images=fr[1:], duration=60, loop=0)
    print(f"outputs : {out}/  (stages.png, onset.png, timecourse.png, I_opl_final.png/.npy"
          f"{', onset.gif' if a.gif else ''})")


if __name__ == "__main__":
    main()
