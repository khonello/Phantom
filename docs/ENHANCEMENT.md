# Phantom — Face Enhancement & Upscaling

## Overview

The pipeline has two post-processing stages that sit on top of the core face swap:

1. **GFPGAN Face Restoration** — priority feature, always on
2. **Background Upscaling** — optional, currently disabled, for batch/offline use

---

## GFPGAN Face Restoration

### What it does

After the face swap, the swapped face region often has subtle artefacts — softness, lighting inconsistencies, loss of fine texture. GFPGAN (Generative Facial Prior GAN) is a restoration model that reconstructs those details. It brings back:

- Skin texture and pores
- Eye sharpness and clarity
- Hair strand detail around the face
- Consistent lighting on the face region

The result is a face swap that looks generated rather than composited. It's the difference between "that looks off" and "that looks real".

### How it runs in the pipeline

GFPGAN runs asynchronously on a dedicated thread so it never blocks the main frame loop. The main loop keeps producing frames at full speed; GFPGAN works on a copy in parallel and feeds enhanced frames back when ready.

The `enhance_interval` setting controls how often a frame is sent to GFPGAN:

```
enhance_interval = 1  → every frame gets enhanced (highest quality, most GPU)
enhance_interval = 5  → every 5th frame gets enhanced (default, balanced)
enhance_interval = 10 → every 10th frame gets enhanced (light touch, low GPU)
```

Between enhanced frames, the pipeline shows the direct swap output. At 30fps with interval 5, you get a fresh enhanced frame every ~167ms — imperceptible to the eye as a gap.

### Priority across presets

GFPGAN is enabled across all quality presets. It is never fully disabled:

| Preset | enhance_interval | Effective rate at 30fps |
|---|---|---|
| `fast` | 10 | ~1 enhanced frame every 333ms |
| `optimal` | 5 | ~1 enhanced frame every 167ms |
| `production` | 1 | Every frame enhanced |

The global default (before any preset is applied) is interval 5.

### Model file

GFPGAN requires `GFPGANv1.4.pth` placed in `pipeline/models/`:

```
pipeline/
  models/
    GFPGANv1.4.pth     ← required
    inswapper_128.onnx  ← face swap model
```

Download: https://github.com/TencentARC/GFPGAN/releases

If the file is missing, the pipeline logs a warning and continues without enhancement — the swap still works, just without restoration.

### Relevant code

```python
# pipeline/stream.py — _load_gfpgan()
return GFPGANer(
    model_path=model_path,
    upscale=1,           # 1 = restore only, no resolution change
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None,   # upscaler disabled — see section below
)
```

```python
# pipeline/stream.py — inside _enhancement_worker()
_, _, restored = gfpganer.enhance(
    frame,
    has_aligned=False,
    only_center_face=False,
    paste_back=True,     # pastes restored face back onto full frame
)
```

---

## Background Upscaling

### What it is

A separate capability that increases the actual pixel resolution of the output frame. For example, 960×540 → 1920×1080. It uses a Real-ESRGAN model as a background upsampler passed into GFPGAN.

### Current status: disabled

```python
# pipeline/stream.py
GFPGANer(
    upscale=1,          # 1 = no upscale
    bg_upsampler=None,  # no background upscaler loaded
)
```

### Why it's disabled for real-time

| Mode | Approx time per frame | Feasible at 30fps |
|---|---|---|
| Restoration only (`upscale=1`) | ~20–40ms | Yes |
| Restoration + 2× upscale | ~150–300ms | No |

At 30fps you have ~33ms per frame budget. Upscaling blows that budget — it's designed for offline/batch processing where time per frame doesn't matter.

### When it makes sense

- Exporting a recorded video at higher resolution (batch mode, not stream mode)
- Post-processing a saved output file
- Running on a very powerful GPU (A100, H100) where 150ms/frame is acceptable at lower fps

### How to enable it (when appropriate)

```python
# pipeline/stream.py — _load_gfpgan()
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

bg_model = RealESRGANer(
    scale=2,
    model_path='pipeline/models/RealESRGAN_x2plus.pth',
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                  num_block=23, num_grow_ch=32, scale=2),
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True,
)

return GFPGANer(
    model_path=model_path,
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=bg_model,
)
```

Additional model needed: `RealESRGAN_x2plus.pth` in `pipeline/models/`.

---

## VRAM Requirements

| Configuration | Approx VRAM |
|---|---|
| Swap only | ~2–3GB |
| Swap + GFPGAN restore | ~5–6GB |
| Swap + GFPGAN + Real-ESRGAN 2× | ~8–10GB |

RTX 3080 (10GB) handles the first two comfortably. The third is tight on a 3080 and comfortable on a 3090 (24GB).

---

## Summary

| Feature | Status | Default interval | GPU cost |
|---|---|---|---|
| GFPGAN face restoration | Always on | 5 frames | Moderate |
| Background upscaling | Disabled | — | High (offline only) |

GFPGAN is the primary enhancement layer and should remain enabled at all times. The interval is the only tuning knob — lower for higher quality, higher to reduce GPU load.
