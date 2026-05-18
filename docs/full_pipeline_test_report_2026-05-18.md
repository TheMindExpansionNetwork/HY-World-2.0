# HY-World 2.0 Full Pipeline Test Report

Updated: 2026-05-18

## Direct answer

The full HY-World 2.0 world-generation pipeline is **not end-to-end working yet**.

What works now:

- Jimsky workbench has the upstream `hyworld2/worldgen` code tree synced.
- Local no-GPU scene preflight passes.
- Modal no-GPU scene preflight passes.
- Modal no-GPU 5-stage command planner passes and returns a concrete planned command list.

What does not work yet:

- The actual paid GPU execution path has not completed a 5-stage worldgen run.
- The current worldgen Modal image is only a scaffold and lacks the Python/CUDA dependencies required by stage execution.
- No reachable VLM/vLLM endpoint is configured for WorldNav / trajectory planning stages.
- Upstream README still says full world-generation usage/model components are partly "Coming soon"; reconstruction is the stable released lane.

## Tests run

### 1. Local syntax/compile check

Command:

```bash
python3 -m py_compile modal/hyworld_worldgen_modal_app.py scripts/preflight_worldgen_scene.py
```

Result: pass.

### 2. Synthetic scene fixture

Created:

```text
/opt/data/workspace/hyworld-inputs/worldgen-smoke-scene/
  panorama.png
  panorama.ppm
  meta_info.json
```

The fixture is a small synthetic 1024x512 equirectangular neon/cyber room panorama.

### 3. Local full-worldgen preflight

Command:

```bash
python3 scripts/preflight_worldgen_scene.py \
  /opt/data/workspace/hyworld-inputs/worldgen-smoke-scene \
  --repo-root . \
  --json
```

Result: pass for scene readiness.

Before VLM env variables were set, preflight correctly warned:

```text
VLM env not fully set; stages 1 and 2 need LLM_ADDR, LLM_PORT, LLM_NAME
```

### 4. Modal preflight

Command:

```bash
modal run modal/hyworld_worldgen_modal_app.py::preflight_local \
  --input-path /opt/data/workspace/hyworld-inputs/worldgen-smoke-scene \
  --output-path /opt/data/hyworld-results/worldgen-smoke-preflight.json
```

Result: pass.

Modal run URL:

```text
https://modal.com/apps/m1ndb0t-2045/main/ap-fYJQcAxtalOWWVqEwmxbxA
```

Output file:

```text
/opt/data/hyworld-results/worldgen-smoke-preflight.json
```

### 5. Modal full-pipeline command planning

Command:

```bash
modal run modal/hyworld_worldgen_modal_app.py::plan_local \
  --input-path /opt/data/workspace/hyworld-inputs/worldgen-smoke-scene \
  --output-path /opt/data/hyworld-results/worldgen-smoke-plan.zip \
  --llm-addr 127.0.0.1 \
  --llm-port 8000 \
  --llm-name Qwen/Qwen3-VL-8B-Instruct
```

Result: pass.

Modal run URL:

```text
https://modal.com/apps/m1ndb0t-2045/main/ap-941o6pvU9XG2oHVZm646JF
```

Output:

```text
/opt/data/hyworld-results/worldgen-smoke-plan.zip
/opt/data/hyworld-results/worldgen-smoke-plan/STATUS.txt
/opt/data/hyworld-results/worldgen-smoke-plan/preflight.json
/opt/data/hyworld-results/worldgen-smoke-plan/planned_commands.json
```

The planned commands cover all five intended stages:

1. `hyworld2/worldgen/traj_generate.py`
2. `hyworld2/worldgen/traj_render.py`
3. `hyworld2/worldgen/video_gen.py`
4. `hyworld2/worldgen/gen_gs_data.py`
5. `hyworld2.worldgen.world_gs_trainer`

### 6. Modal dependency probe

Command:

```bash
modal run modal/hyworld_worldgen_modal_app.py::probe_env_local \
  --output-path /opt/data/hyworld-results/worldgen-env-probe.json
```

Result: the scaffold image starts, repo is present, but worldgen dependencies are missing.

Modal run URL:

```text
https://modal.com/apps/m1ndb0t-2045/main/ap-8E1JRtwZj19twKDz6YTFtE
```

Output file:

```text
/opt/data/hyworld-results/worldgen-env-probe.json
```

Missing modules reported:

```text
torch, torchvision, diffusers, transformers, accelerate, open3d, cv2, numpy, PIL, gsplat, trimesh, omegaconf, openai
```

## Exact blocker for paid full execution

A `dry_run=False` paid H100 run would currently fail at stage 1 before producing a world because:

1. The Modal worldgen image does not yet install the required dependencies.
2. The repo requirements expect Python 3.10 + CUDA 12.4 PyTorch + a cp310 gsplat wheel, while the scaffold image used Python 3.11.
3. Stages 1 and 2 need a reachable VLM/vLLM endpoint (`LLM_ADDR`, `LLM_PORT`, `LLM_NAME`). `127.0.0.1` only validates command shaping; it is not a real endpoint inside the Modal worker unless we launch one in the same app/network.
4. Upstream says HY-Pano-2 / WorldStereo-2 / full generation usage remains partly unreleased/coming soon, so the stable production lane is still WorldMirror reconstruction.

## Next fix

Build a real full-worldgen Modal image separately from the working reconstruction image:

- Python 3.10
- PyTorch 2.4.0 + torchvision 0.19.0 CUDA 12.4
- `requirements.txt`
- cp310 `gsplat` wheel
- FlashAttention 2 or 3
- OpenAI client + VLM/vLLM endpoint wiring
- optional VLM sidecar app serving Qwen3-VL

Then rerun:

1. dependency probe
2. stage 1 only
3. stages 1-2
4. full 5-stage run with short/bounded settings

## Bottom line

The full pipeline is **not working end-to-end yet**, but the test was useful: it proved the project packaging, scene upload, preflight, and 5-stage command planning are wired, and it identified the exact missing runtime layer before spending H100 time.
