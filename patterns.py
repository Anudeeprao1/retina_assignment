"""patterns.py -- synthetic stimuli for verifying the OPL (usable in place of a camera).

Each pattern tests one property of the OPL (see the README for what to expect):
  flicker   uniform field, 2 Hz square wave     -> response with NO edges (non-separability)
  edge      static step edge                    -> Mach bands, then fade (undershoot)
  bar       bright bar moving left to right      -> strong response on the moving edges
  grating   drifting sine grating (2 Hz)        -> spatial + temporal band-pass
  onset     gray 1 s, then a test scene 1 s      -> luminance first, edges later
"""
import numpy as np

NAMES = ("flicker", "edge", "bar", "grating", "onset")


class Pattern:
    def __init__(self, name, width=320, height=240, fps=30.0):
        if name not in NAMES:
            raise ValueError(f"unknown pattern '{name}', choose from {NAMES}")
        self.name, self.w, self.h, self.fps = name, width, height, fps
        self.k = 0
        yy, xx = np.mgrid[0:height, 0:width]
        self.xx, self.yy = xx, yy
        scene = np.full((height, width), 70.0)                       # onset test scene
        scene[height // 5: 3 * height // 5, width // 6: width // 2] = 200.0
        scene[:, 2 * width // 3:] = np.linspace(60, 200, width - 2 * width // 3)
        r = min(width, height) // 14
        scene[(xx - width // 4) ** 2 + (yy - 4 * height // 5) ** 2 < r * r] = 240.0
        scene[(xx - width // 2) ** 2 + (yy - 4 * height // 5) ** 2 < r * r] = 15.0
        self.scene = scene

    def read(self):
        t = self.k / self.fps
        self.k += 1
        n, w, h, xx = self.name, self.w, self.h, self.xx
        if n == "flicker":
            f = np.full((h, w), 160.0 if int(t * 4) % 2 == 0 else 95.0)
        elif n == "edge":
            f = np.where(xx < w // 2, 70.0, 190.0)
        elif n == "bar":
            x0 = int((t * 0.5 % 1.0) * (w + 40)) - 20                  # crosses in 2 s
            f = np.full((h, w), 90.0); f[:, max(0, x0): max(0, x0 + 16)] = 220.0
        elif n == "grating":
            f = 127.5 + 80 * np.sin(2 * np.pi * (xx / 40.0 - 2.0 * t))
        else:  # onset
            f = self.scene if (t % 2.0) >= 1.0 else np.full((h, w), 127.5)
        return f.astype(np.float64)
