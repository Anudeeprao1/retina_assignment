"""retina_io.py -- shared helpers for loading inputs and displaying OPL maps."""
import numpy as np
from PIL import Image, ImageOps

try:                                  # optional: iPhone HEIC/HEIF photos
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp",
              ".pgm", ".ppm", ".pbm", ".pnm", ".heic", ".heif", ".jp2", ".ico")


def load_luminance(path, max_size=512, background=127.5):
    """Open any image Pillow can read and return luminance as float64 in 0..255.

    - phone photos are rotated upright (EXIF orientation)
    - animated GIF / multi-page TIFF: the first frame is used
    - 16-bit images are scaled to 0..255; 32-bit/float images are stretched to 0..255
    - transparent pixels are composited onto mid-gray (neutral for the retina)
    - colour is converted to luminance (Pillow 'L': 0.299 R + 0.587 G + 0.114 B)
    - the long side is limited to max_size pixels (0 = keep full size)
    """
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)
    if im.mode.startswith("I;16"):                  # 16-bit: keep brightness, just rescale
        a = np.asarray(im, dtype=np.float64) * (255.0 / 65535.0)
        im = Image.fromarray(np.round(a).astype(np.uint8))
    elif im.mode in ("I", "F"):                      # 32-bit / float: no fixed range, stretch
        a = np.asarray(im, dtype=np.float64)
        a = 255.0 * (a - a.min()) / (np.ptp(a) or 1.0)
        im = Image.fromarray(np.round(a).astype(np.uint8))
    if im.mode == "P":
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", im.size, (int(background),) * 3 + (255,))
        im = Image.alpha_composite(bg, im.convert("RGBA"))
    im = im.convert("L")
    if max_size:
        im.thumbnail((max_size, max_size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def signed_to_u8(x, scale):
    """Map a signed map to 0..255 with 0 -> mid-gray and +/-scale -> white/black."""
    return np.clip(127.5 + 127.5 * x / (scale or 1.0), 0, 255).astype(np.uint8)


def signed_to_bgr(x, scale):
    """Blue (negative, center darker) - white (0) - red (positive, center brighter)."""
    v = np.clip(x / (scale or 1.0), -1, 1)
    pos, neg = np.clip(v, 0, 1), np.clip(-v, 0, 1)
    b = 255 * (1 - pos)
    g = 255 * (1 - pos - neg)
    r = 255 * (1 - neg)
    return np.dstack([b, g, r]).clip(0, 255).astype(np.uint8)
