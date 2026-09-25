# Retina OPL emulation (Virtual Retina port)

Python implementation of the **Outer Plexiform Layer** of the Virtual Retina model
(Wohrer & Kornprobst 2009). `opl_vr.py` is a line-by-line port of the C++ `LinearOPL`
and reproduces the compiled simulator's output exactly (see *Verification*).

```
retina_opl/
  opl_vr.py          the OPL engine (RecFilter, deriche_blur, MapFilter, LinearOPL)
  opl_numba.py       the same Deriche blur compiled with Numba (bit-identical, ~5x faster)
  retina_io.py       load any image format as luminance; signed-map display helpers
  run_image.py       Stage 1: one image in -> OPL stages and junction image out
  run_webcam.py      Stage 2: webcam / video / pattern in -> live junction video out
  patterns.py        synthetic verification stimuli
  configs/           OPL parameter files (same XML format as Virtual Retina)
  validate_opl.py    compare opl_vr.py with the C++ binary's saved maps
  inr_view.py        read the binary's .inr maps (used by validate_opl.py)
```

## Setup (macOS)

```bash
cd retina_opl
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pillow-heif          # optional: iPhone .heic photos
```

For the webcam: the first run asks for camera permission. If it fails, allow your
terminal app in **System Settings > Privacy & Security > Camera** and restart the terminal.

## Stage 1: an image

```bash
python run_image.py path/to/photo.jpg                     # Fig. 10 parameters, 300 ms
python run_image.py photo.png --pre noise --pre-ms 100    # noise before onset, as in Fig. 10
python run_image.py photo.jpg --config configs/opl_sharp.xml
python run_image.py                                       # file picker
```

Any format Pillow reads works: jpg, png (transparency is put on mid-gray), bmp, tiff
(8/16-bit), webp, gif (first frame), pgm/ppm, and heic with `pillow-heif`. Colour is
converted to luminance. The long side is limited to 512 px (`--max-size`).

Outputs go to `out/<image name>/`:

| file | what it shows |
|---|---|
| `stages.png` | input L, center C, surround S, junction output I_OPL, and its sign |
| `onset.png` | I_OPL 10-300 ms after the image appears: luminance first, edges later |
| `timecourse.png` | C, S and I_OPL at the center pixel over time, and mean activity |
| `I_opl_final.png` / `.npy` | the junction image at the end (PNG: mid-gray = 0) |
| `onset.gif` | I_OPL over time (with `--gif`) |

The Fig. 10 config uses a 1 ms step, so 300 ms = 300 steps (about 15-60 s for a 512 px image).

## Stage 2: live video

```bash
python run_webcam.py                          # default camera
python run_webcam.py --mirror                 # selfie view
python run_webcam.py --source pattern:bar     # standard verification input
python run_webcam.py --source clip.mp4        # a video file
python run_webcam.py --record out.mp4         # save the output video
```

Window: `[ input | I_OPL (gray = 0) | sign: red +, blue - ]`, with FPS, OPL time per
frame and model/real time in the overlay. When you quit, the terminal prints the time
per frame spent waiting for the camera, converting, in the OPL model, drawing, and
showing the window. Keys: `q` quit, `space` pause, `r` reset,
`s` snapshot to `out/snapshots/`. Processing width is 320 px (`--width`).

Each frame advances the model by frame_interval / dt steps (remainder carried to the
next frame, capped by `--max-substeps`). `model/real 1.00x` means the retina runs in real time.

### Verification patterns (`--source pattern:NAME`)

| pattern | stimulus | expected OPL response |
|---|---|---|
| `flicker` | uniform field, 2 Hz | whole field responds (+ when brighter, - when darker) although there are no edges |
| `edge` | static step edge | bright/dark bands along the edge (Mach bands), fading as the undershoot adapts |
| `bar` | bar moving left to right | response on the moving edges with a trailing wake; static background stays at 0 |
| `grating` | drifting sine grating | stripes in the output; amplitude depends on spatial and temporal frequency |
| `onset` | gray 1 s, then a scene 1 s | first frames after onset look like the scene's brightness, then only edges remain |

### Speed: blur backends (`--blur`, both scripts)

The spatial blur dominates the cost. All backends compute the same Deriche filter:

| backend | vs C++ simulator | 2-D blur at 320x240* |
|---|---|---|
| `numba` (default when installed) | bit-identical | ~2 ms |
| `fast` (scipy `lfilter`) | ~1e-13 (float rounding) | ~3-5 ms |
| `exact` (pure Python, reference) | bit-identical | ~10 ms |

*measured on the development machine; your numbers will differ.

The live loop also reads the camera in a background thread (only the newest frame is
processed) and draws the colour map with a lookup table.

Real-time condition: the retina keeps up (`model/real 1.00x`) only if one model step
is computed faster than the time it represents (dt = 10 ms), with room left for
capture and display. With a fake 30 fps camera the numba build ran at 30 fps and 1.00x
(OPL ~22 ms per frame for 3-4 steps); the pure-Python blur ran at ~7 fps and 0.38x.

## Configs

| file | step | use |
|---|---|---|
| `opl_fig10.xml` | 1 ms | paper's Fig. 10 primate OPL; default for images |
| `opl_sharp.xml` | 5 ms | shipped primate-fovea sizes, leaky-heat filter |
| `opl_webcam.xml` | 10 ms | default for live video (3-4 steps per 30 fps frame) |
 
