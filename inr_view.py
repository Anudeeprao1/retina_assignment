#!/usr/bin/env python3
"""Read Virtual Retina .inr maps and render the OPL stages as PNGs.

usage:  python3 inr_view.py <outD> <input_image.pgm> [dt_seconds]
  outD = the -outD directory given to bin/Retina (must be run with -savemap -nS 1)
"""
import sys, glob, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def read_inr(path):
    with open(path, "rb") as f:
        header = f.read(256).decode("ascii", "ignore")
        if not header.startswith("#INRIMAGE-4#{"):
            raise ValueError(f"{path}: not an INRIMAGE-4 file")
        get = lambda k, d=None: (re.search(rf"{k}=([^\n]+)", header) or [None, d])[1]
        x, y, z, v = (int(get(k, 1)) for k in ("XDIM", "YDIM", "ZDIM", "VDIM"))
        bits = int(re.search(r"PIXSIZE=(\d+)", header).group(1))
        typ = get("TYPE", "float").strip()
        endian = ">" if get("CPU", "decm").strip() in ("sun", "sgi") else "<"
        kind = "f" if "float" in typ else ("u" if "unsigned" in typ else "i")
        data = np.frombuffer(f.read(), dtype=f"{endian}{kind}{bits // 8}")
    return data[: x * y * z * v].reshape(z, y, x, v).squeeze().astype(np.float64)


def frames(outD, name):
    return sorted(glob.glob(os.path.join(outD, name, "*.inr")))


def main():
    outD, img = sys.argv[1], sys.argv[2]
    dt = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
    L = np.asarray(Image.open(img).convert("L"), dtype=np.float64)
    rec, hor, opl = (frames(outD, n) for n in ("receptorFrames", "horizontalFrames", "oplFrames"))
    if not opl:
        sys.exit("no oplFrames found: run Retina with -savemap -nS 1")
    C, S, I = read_inr(rec[-1]), read_inr(hor[-1]), read_inr(opl[-1])
    T = len(opl) * dt * 1000
    print(f"{len(opl)} saved steps, final t = {T:.0f} ms")
    for n, a in (("C receptors", C), ("S horizontal", S), ("I_OPL", I)):
        print(f"  {n:13s} min {a.min():+.4f}  mean {a.mean():+.4f}  max {a.max():+.4f}")

    m = np.percentile(np.abs(I), 99.5)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.9))
    ax[0].imshow(L, cmap="gray"); ax[0].set_title(f"INPUT  L  ({os.path.basename(img)})")
    ax[1].imshow(C, cmap="gray"); ax[1].set_title("receptorFrames  →  C  (center)")
    ax[2].imshow(S, cmap="gray"); ax[2].set_title("horizontalFrames  →  S  (surround)")
    ax[3].imshow(I, cmap="gray", vmin=-m, vmax=m)
    ax[3].set_title(f"oplFrames  →  I_OPL  at t={T:.0f} ms\n(mid-gray = 0)")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(outD, "opl_stages.png"), dpi=110, bbox_inches="tight"); plt.close()

    picks = [i for i in (1, 3, 5, 9, 19, len(opl) - 1) if i < len(opl)]
    maps = [read_inr(opl[i]) for i in picks]
    mm = max(np.percentile(np.abs(a), 99.5) for a in maps)
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for a, i, M in zip(ax.flat, picks, maps):
        a.imshow(M, cmap="gray", vmin=-mm, vmax=mm)
        a.set_title(f"I_OPL   t = {(i + 1) * dt * 1000:.0f} ms"); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Virtual Retina binary output after image onset (from mid-gray screen), shared scale", y=1.0)
    plt.tight_layout(); plt.savefig(os.path.join(outD, "opl_onset.png"), dpi=100, bbox_inches="tight"); plt.close()
    print("wrote", os.path.join(outD, "opl_stages.png"), "and opl_onset.png")


if __name__ == "__main__":
    main()
