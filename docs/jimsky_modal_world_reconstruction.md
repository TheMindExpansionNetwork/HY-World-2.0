# Jimsky HY-World 2.0 Modal Reconstruction Lane

Status: Modal GPU smoke-tested successfully on 2026-04-27 with a tiny synthetic five-frame capture. The batch result produced `gaussians.ply`, `points.ply`, depth/normal maps, camera JSON, timing JSON, and `rendered/rendered_rgb.mp4`. Continue to require explicit user approval for new cost-bearing GPU runs.

## What this gives Jimsky

This fork turns Tencent Hunyuan **HY-World 2.0 / WorldMirror 2.0** into a practical capture lane:

```text
phone photos or short video
→ Modal serverless GPU
→ WorldMirror reconstruction
→ GLB / PLY / camera JSON / depth+normal maps
→ rendered MP4 preview when rendering succeeds
→ ZIP back to Telegram / Google Drive / SIGIL / MemPalace
```

This is the foundation for “universe building”: capture real places, reconstruct them into portable 3D artifacts, then reuse them in Jimsky scenes, projections, spatial memory, game worlds, and future live agent environments.

## Upstream repo facts

- Upstream: `https://github.com/Tencent-Hunyuan/HY-World-2.0`
- Fork: `https://github.com/TheMindExpansionNetwork/HY-World-2.0`
- Model repo: `tencent/HY-World-2.0`
- Main open component currently documented in repo: **WorldMirror 2.0** world reconstruction.
- Panorama generation and world generation sections are marked “coming soon” upstream.

## Input guidance

Best results:

- 5–30 photos of the same room/object/place, overlapping views.
- Or a short slow walkaround video.
- Good lighting, minimal blur, avoid wildly different zoom levels.
- For video, default sampling caps at 32 frames.

## Outputs

HY-World/WorldMirror can save:

- `gaussians.ply` — 3D Gaussian splatting representation.
- `points.ply` — colored point cloud.
- `camera_params.json` — camera extrinsics/intrinsics.
- `depth/*.png` + `depth/*.npy` — per-view depth maps.
- `normal/*.png` — per-view normal maps.
- `sparse/0/*` — optional COLMAP sparse output when `--save_colmap` is used.
- `rendered/rendered_rgb.mp4` — preview flythrough when `--save_rendered` succeeds.
- `rendered/rendered_depth.mp4` — optional depth preview when `--render_depth` is used.

## Modal scaffold

File:

```text
modal/hyworld_modal_app.py
```

It defines:

- Modal app: `jimsky-hyworld-2-world-reconstruction`
- Cache volume: `jimsky-hyworld2-cache`
- GPU function: `reconstruct_archive(...)`
- Local entrypoint: `reconstruct_local(...)`

The image follows upstream install expectations:

- Python 3.10
- PyTorch 2.4.0 / torchvision 0.19.0, CUDA 12.4 wheel
- `gsplat` pt24/cu124 wheel
- `ffmpeg`, OpenCV libs, Open3D, pycolmap, Gradio dependencies
- FlashAttention-2 from a prebuilt Python 3.10 / Torch 2.4 / CUDA 12 wheel, because upstream imports `flash_attn` during pipeline startup
- Hugging Face cache mounted in a Modal Volume

The first live smoke test showed FlashAttention is required at import time. Avoid source-building `flash-attn` in Modal's slim image unless a CUDA-devel base is used; the prebuilt wheel path is faster and avoids `CUDA_HOME`/nvcc failures.

## Run command when approved

From repo root:

```bash
modal run modal/hyworld_modal_app.py::reconstruct_local \
  --input-path /path/to/photos_or_video_or_zip \
  --output-path /opt/data/hyworld-results/jimsky-world.zip \
  --target-size 952 \
  --video-max-frames 32
```

Then unpack and deliver preview:

```bash
mkdir -p /opt/data/hyworld-results/jimsky-world
unzip -o /opt/data/hyworld-results/jimsky-world.zip -d /opt/data/hyworld-results/jimsky-world
python scripts/package_hyworld_result.py /opt/data/hyworld-results/jimsky-world/result \
  --zip /opt/data/hyworld-results/jimsky-world-delivery.zip
```

If `rendered/rendered_rgb.mp4` exists, send that first as a Telegram preview, then send/upload the ZIP.

## Modal cost safety

- Use Modal because it scales to zero when the function exits.
- First model download/build can be slow; cache in the Modal Volume.
- Start with `A100-80GB` for compatibility.
- Try lower-cost GPUs only after a known-good A100 run establishes memory use.
- Do not leave persistent GPU services running unless user asks for a Gradio live demo.
- After each test, verify no long-running Modal app/container remains beyond normal function completion.

## Future upgrades

1. Add a FastAPI upload endpoint for direct Telegram/webhook submissions.
2. Add Google Drive upload after Google auth is connected.
3. Add SIGIL/MemPalace import: store `camera_params.json`, preview MP4, and PLY/GLB metadata as world-memory artifacts.
4. Add a Gradio-on-Modal live demo mode only when user wants an interactive public preview.
5. Add post-processing converters for Gaussian splats / Three.js / WebXR / Unreal / Unity lanes.
