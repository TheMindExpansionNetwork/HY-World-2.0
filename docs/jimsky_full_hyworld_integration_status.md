# Jimsky / HY-World 2.0 Full Integration Status

Updated: 2026-05-18

## Goal

Create a Hermes/Jimsky workbench that contains the whole HY-World 2.0 project surface, with a working Modal lane for practical reconstruction and a documented path for the full world-generation pipeline.

## Current source of truth

- Upstream: https://github.com/Tencent-Hunyuan/HY-World-2.0
- Jimsky fork/workbench: `/opt/data/workspace/projects/HY-World-2.0`
- Branch: `jimsky/universe-builder`
- Primary Modal runner: `modal/hyworld_modal_app.py`
- Packaging helper: `scripts/package_hyworld_result.py`
- Existing reusable skill: `hyworld-modal-reconstruction`

## What is actually implemented

### 1. WorldMirror reconstruction lane — implemented

The Modal runner `modal/hyworld_modal_app.py` is a real batch runner for WorldMirror 2.0 reconstruction:

- Accepts local directory, video/image file, or zip through `reconstruct_local`.
- Zips directories before upload.
- Runs `python -m hyworld2.worldrecon.pipeline` remotely on Modal.
- Uses a Modal Volume for Hugging Face cache: `jimsky-hyworld2-cache`.
- Targets `A100-80GB` by default.
- Exposes practical quality knobs:
  - `--target-size`
  - `--video-min-frames`
  - `--video-max-frames`
  - `--render-interp-per-pair`
  - `--render-depth`
  - `--save-colmap`
  - `--save-conf`
  - `--no-compress-pts`
  - `--compress-pts-max-points`
  - `--compress-gs-max-points`
  - `--max-resolution`
  - `--prior-cam-path`
- Writes result ZIP locally.

Verification performed this session:

```bash
cd /opt/data/workspace/projects/HY-World-2.0
python3 -m py_compile modal/hyworld_modal_app.py scripts/package_hyworld_result.py
```

Result: compile passed.

### 2. Packaging lane — implemented

`scripts/package_hyworld_result.py` summarizes key HY-World artifacts and creates a delivery zip.

Key expected artifacts:

- `rendered/rendered_rgb.mp4`
- `rendered/rendered_depth.mp4`
- `gaussians.ply`
- `points.ply`
- `camera_params.json`
- `pipeline_timing.json`

### 3. Full world-generation lane — now synced into Jimsky workbench; Modal automation scaffolded

The main Jimsky workbench was missing most of the actual upstream `hyworld2/worldgen` implementation and only had the README. This session synced the complete upstream worldgen tree from `/opt/hermes/hyworld-2.0/hyworld2/worldgen` into `/opt/data/workspace/projects/HY-World-2.0/hyworld2/worldgen`.

Synced worldgen code now includes the 5-stage pipeline:

1. `traj_generate.py` — trajectory planning / WorldNav
2. `traj_render.py` — trajectory rendering + VLM captions
3. `video_gen.py` — WorldStereo expansion
4. `gen_gs_data.py` — GS training data prep
5. `world_gs_trainer.py` — Gaussian splatting training / export

Important constraints from upstream docs:

- Python 3.11+ for worldgen.
- CUDA 12.8 expected.
- >=4 GPUs recommended; upstream tested on 8x H20.
- Requires a running VLM server, usually Qwen3-VL-8B-Instruct via vLLM, for stages 1 and 2.
- Requires source builds / third-party components such as modified `gsplat_maskgaussian` and `navmesh`.

This is **not** the same as the single-GPU WorldMirror reconstruction lane. It needs a separate Modal architecture, likely multi-GPU, with either:

- one large Modal app that starts/uses a VLM service plus worldgen stages, or
- a two-service design: persistent/temporary vLLM endpoint + batch worldgen jobs.

## Recommended next implementation steps

1. Keep using the existing WorldMirror Modal runner for immediate working artifacts.
2. Add a `modal/hyworld_worldgen_modal_app.py` scaffold that packages the 5-stage worldgen pipeline without claiming it is tested.
3. Add a preflight command that checks a scene folder has:
   - `panorama.png`
   - `meta_info.json`
   - expected worldgen subfolders after each stage
   - VLM endpoint env vars: `LLM_ADDR`, `LLM_PORT`, `LLM_NAME`
4. Build a dependency image for worldgen separately from reconstruction.
5. Run only syntax/preflight locally first; do not launch multi-GPU Modal spend without explicit approval.

## Honest status

- Reconstruction: real, implemented, previously proven by existing workflow notes, and compile-verified now.
- Full world-generation code: now present in the Jimsky workbench (`hyworld2/worldgen`, 114 files synced from upstream) and core entrypoint scripts compile.
- Full world-generation Modal automation: scaffold/preflight exists in `modal/hyworld_worldgen_modal_app.py`, but no paid multi-GPU end-to-end worldgen run has been launched yet.
- Immediate action: continue hardening the worldgen scaffold, add dependency/preflight docs, then run only when GPU/VLM setup is approved.
