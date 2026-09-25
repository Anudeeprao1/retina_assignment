#!/usr/bin/env python3
"""run_webcam.py -- run the OPL live on a webcam (or a video, an image sequence,
a still image, or a synthetic test pattern) and show the junction output as video.

Examples
  python run_webcam.py                              # default camera (index 0)
  python run_webcam.py --source 1                   # another camera
  python run_webcam.py --source pattern:flicker     # verification stimulus (see patterns.py)
  python run_webcam.py --source clip.mp4            # a video file
  python run_webcam.py --source "frames/*.png"      # an image sequence (quote the pattern)
  python run_webcam.py --source photo.jpg           # a still image shown as a stream
  python run_webcam.py --record out.mp4             # also save the mosaic video

Window: [ input | I_OPL (gray = 0) | I_OPL sign (red +, blue -) ]
Keys:   q or Esc quit,  space pause,  r reset the retina,  s save a snapshot

Time handling
  Each displayed frame advances the model by about frame_interval / dt steps
  (the fractional remainder is carried over), capped at --max-substeps, so model
  time keeps up with real time when the
  computer is fast enough. The overlay shows FPS, processing time, and the ratio
  model-time / real-time (1.00x = the retina runs in real time).
"""
import argparse, glob, os, sys, threading, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from opl_vr import opl_from_xml, set_blur_backend
from retina_io import load_luminance, signed_to_u8, IMAGE_EXTS
from patterns import Pattern


class LatestFrameReader(threading.Thread):
    """Reads the camera continuously in the background and keeps only the newest
    frame, so processing never waits on (or lags behind) buffered camera frames."""
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.cond = cap, threading.Condition()
        self.frame, self.seq, self.ok, self.running = None, 0, True, True
        self.start()

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            with self.cond:
                self.ok = ok
                if ok:
                    self.frame, self.seq = f, self.seq + 1
                self.cond.notify_all()
            if not ok:
                break

    def latest(self, last_seq):
        """Newest frame not yet processed (waits only if no new frame has arrived)."""
        with self.cond:
            while self.ok and self.seq == last_seq:
                self.cond.wait(timeout=1.0)
            return (self.frame, self.seq) if self.ok or self.seq != last_seq else (None, last_seq)

    def stop(self):
        self.running = False


def _sign_lut():
    """256-entry colour table: 0 -> blue (negative), 128 -> white (zero), 255 -> red (positive)."""
    v = (np.arange(256) - 127.5) / 127.5
    pos, neg = np.clip(v, 0, 1), np.clip(-v, 0, 1)
    bgr = np.stack([255 * (1 - pos), 255 * (1 - pos - neg), 255 * (1 - neg)], axis=-1)
    return bgr.clip(0, 255).astype(np.uint8).reshape(256, 1, 3)


SIGN_LUT = _sign_lut()


class Source:
    """Uniform interface over camera / video / image sequence / still image / pattern."""
    def __init__(self, spec, width, fps_hint):
        self.width, self.live, self.cap, self.frames, self.pattern = width, False, None, None, None
        if spec.startswith("pattern:"):
            h = int(round(width * 3 / 4))
            self.pattern = Pattern(spec.split(":", 1)[1], width, h, fps_hint)
            self.fps, self.label = fps_hint, spec
        elif spec.isdigit():
            idx = int(spec)
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(idx, backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap.isOpened():
                sys.exit(f"Could not open camera {idx}. On macOS, allow camera access for your terminal "
                         "app in System Settings > Privacy & Security > Camera, then restart the terminal.")
            self.live, self.fps = True, self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.label = f"camera {idx}"
            self.reader, self.seq = LatestFrameReader(self.cap), 0
        elif any(ch in spec for ch in "*?["):
            self.frames = sorted(glob.glob(spec))
            if not self.frames:
                sys.exit(f"No files match {spec}")
            self.k, self.fps, self.label = 0, fps_hint, f"{len(self.frames)} images"
        elif spec.lower().endswith(IMAGE_EXTS):
            self.still = load_luminance(spec, max_size=0)
            self.fps, self.label = fps_hint, os.path.basename(spec)
        else:
            self.cap = cv2.VideoCapture(spec)
            if not self.cap.isOpened():
                sys.exit(f"Could not open video {spec}")
            self.fps, self.label = self.cap.get(cv2.CAP_PROP_FPS) or fps_hint, os.path.basename(spec)

    def _fit(self, gray):
        h, w = gray.shape
        if w != self.width:
            gray = cv2.resize(gray, (self.width, int(round(h * self.width / w))), interpolation=cv2.INTER_AREA)
        return gray.astype(np.float64)

    def read(self):
        """Returns (luminance float64 0..255 at the working width, original BGR for display) or None."""
        if self.pattern is not None:
            f = self.pattern.read()
            return f, None
        if self.frames is not None:
            if self.k >= len(self.frames):
                return None
            f = load_luminance(self.frames[self.k], max_size=0); self.k += 1
            return self._fit(f), None
        if self.cap is None:                          # still image
            return self._fit(self.still), None
        if self.live:
            bgr, self.seq = self.reader.latest(self.seq)
            if bgr is None:
                return None
        else:
            ok, bgr = self.cap.read()
            if not ok:
                return None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # same 0.299/0.587/0.114 weights as Pillow 'L'
        return self._fit(gray), bgr

    def release(self):
        if self.live:
            self.reader.stop()
        if self.cap is not None:
            self.cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, video file, 'glob*', image, or pattern:NAME")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "opl_webcam.xml"))
    ap.add_argument("--width", type=int, default=320, help="processing width in pixels (default 320)")
    ap.add_argument("--max-substeps", type=int, default=6, help="max model steps per frame (default 6)")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate for patterns / image sequences")
    ap.add_argument("--zoom", type=float, default=1.5, help="display scale of the mosaic")
    ap.add_argument("--mirror", action="store_true", help="mirror the camera image (selfie view)")
    ap.add_argument("--blur", default="auto", choices=["auto", "numba", "fast", "exact"],
                    help="blur backend: numba/exact are bit-identical to the C++ (default auto)")
    ap.add_argument("--record", help="write the mosaic to this .mp4 file")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until quit)")
    ap.add_argument("--no-display", action="store_true", help="run without a window (for testing)")
    a = ap.parse_args()

    backend = set_blur_backend(a.blur)
    src = Source(a.source, a.width, a.fps)
    opl, lum = opl_from_xml(a.config)
    dt = opl.excit.dt
    print(f"source  : {src.label}  ({'live' if src.live else f'{src.fps:g} fps'})")
    print(f"config  : {os.path.basename(a.config)}  (dt = {dt*1000:g} ms, width {a.width} px, "
          f"blur {backend})")

    writer, allocated, paused = None, False, False
    scale = None                                      # display scale, smoothed over frames
    fps_ema = None
    model_t = real_t = 0.0
    debt = 0.0                                        # stream time not yet simulated
    stages = ("wait for frame", "convert", "OPL model", "draw mosaic", "show window")
    tot = dict.fromkeys(stages, 0.0)                  # accumulated seconds per stage
    n = 0
    t_prev = time.perf_counter()
    snap_dir = os.path.join(HERE, "out", "snapshots"); os.makedirs(snap_dir, exist_ok=True)
    try:
        while True:
            if paused and not a.no_display:
                k = cv2.waitKey(30) & 0xFF
                if k == ord(" "): paused = False; t_prev = time.perf_counter()
                elif k in (ord("q"), 27): break
                continue
            t0 = time.perf_counter()
            got = src.read()
            t1 = time.perf_counter()
            if got is None:
                break
            gray, bgr = got
            if a.mirror:
                gray = gray[:, ::-1].copy()
            t2 = time.perf_counter()
            wall_dt = t2 - t_prev                     # measured time between processed frames
            frame_dt = wall_dt if src.live else 1.0 / src.fps   # stream time this frame covers
            t_prev = t2

            if not allocated:
                opl.allocate(gray.shape, lum / 2.0)   # start from a uniform gray screen
                allocated = True
                frame_dt = wall_dt = 1.0 / src.fps if not src.live else 1.0 / 30

            # ---- advance the model so that model time follows real time ----
            # carry the fractional remainder to the next frame (3,3,4,3,3,4... steps at
            # 30 fps and dt = 10 ms) instead of rounding every frame, which would drift.
            debt += frame_dt
            sub = int(np.clip(debt // dt, 1, a.max_substeps))
            debt = min(debt - sub * dt, dt)           # if capped, drop the backlog
            for _ in range(sub):
                I = opl.step(gray)
            model_t += sub * dt; real_t += frame_dt; n += 1
            t3 = time.perf_counter()

            # ---- draw: gray input | signed I_OPL | sign colour map ----
            s_now = np.percentile(np.abs(I[::2, ::2]), 99.5) or 1.0
            scale = s_now if scale is None else 0.9 * scale + 0.1 * s_now
            u8 = signed_to_u8(I, scale)
            left = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            mid = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            right = cv2.LUT(mid, SIGN_LUT)
            mosaic = np.hstack([left, mid, right])
            if a.zoom != 1:
                mosaic = cv2.resize(mosaic, None, fx=a.zoom, fy=a.zoom, interpolation=cv2.INTER_NEAREST)
            fps_ema = 1 / max(wall_dt, 1e-6) if fps_ema is None else 0.9 * fps_ema + 0.1 / max(wall_dt, 1e-6)
            text = (f"{fps_ema:5.1f} fps | OPL {1000*(t3-t2):5.1f} ms ({sub} x {dt*1000:g} ms steps) | "
                    f"model/real {model_t/max(real_t,1e-9):.2f}x")
            cv2.putText(mosaic, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(mosaic, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            if a.record:
                if writer is None:
                    writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             src.fps if not src.live else 30.0, mosaic.shape[1::-1])
                writer.write(mosaic)
            t4 = time.perf_counter()

            if not a.no_display:
                cv2.imshow("Retina OPL  -  input | I_OPL | sign", mosaic)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27): break
                if k == ord(" "): paused = True
                if k == ord("r"): opl.allocate(gray.shape, lum / 2.0); scale = None
                if k == ord("s"):
                    p = os.path.join(snap_dir, f"snap_{time.strftime('%H%M%S')}_{n}.png")
                    cv2.imwrite(p, mosaic); np.save(p[:-4] + ".npy", I); print("saved", p)
            t5 = time.perf_counter()
            for name, d in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                tot[name] += d
            if a.frames and n >= a.frames:
                break
    finally:
        src.release()
        if writer: writer.release()
        if not a.no_display: cv2.destroyAllWindows()
    if n:
        print(f"frames  : {n}, display rate {fps_ema:.1f} fps, model/real time {model_t/max(real_t,1e-9):.2f}x")
        print("time per frame:")
        for name in stages:
            print(f"  {name:15s} {1000*tot[name]/n:6.1f} ms")
        print(f"  {'total':15s} {1000*sum(tot.values())/n:6.1f} ms")


if __name__ == "__main__":
    main()
