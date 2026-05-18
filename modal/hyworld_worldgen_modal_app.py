"""HY-World 2.0 full world-generation Modal scaffold.

This is intentionally a *scaffold/preflight* for the heavier 5-stage worldgen lane,
not a claimed production runner yet. The stable production lane remains
`modal/hyworld_modal_app.py::reconstruct_local` for WorldMirror reconstruction.

Worldgen upstream requires Python 3.11+, CUDA 12.8, multiple GPUs, a VLM/vLLM
endpoint for stages 1-2, and source-built third-party pieces. Keep this separate
from the reconstruction image so the working reconstruction lane stays reliable.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

APP_NAME = "jimsky-hyworld-2-full-worldgen-scaffold"
REMOTE_REPO = Path("/workspace/HY-World-2.0")
HF_CACHE = Path("/cache/huggingface")

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("jimsky-hyworld2-worldgen-cache", create_if_missing=True)

# Scaffold image only. This is not fully dependency-complete until navmesh,
# modified gsplat_maskgaussian, and CUDA 12.8 details are validated.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "build-essential", "cmake", "ninja-build", "libgl1", "libglib2.0-0", "libgomp1")
    .run_commands("python -m pip install --upgrade pip setuptools wheel packaging ninja")
    .env({"HF_HOME": str(HF_CACHE), "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(".", remote_path=str(REMOTE_REPO))
)


def _zip_dir(src: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src))
    return buf.getvalue()


def _extract_zip(payload: bytes, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(dest)
    return dest


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=merged, check=True)


@app.function(image=image, timeout=20 * 60, volumes={str(HF_CACHE): cache_volume})
def preflight_worldgen_archive(scene_zip: bytes) -> dict:
    """No-GPU preflight: unpack a scene zip and validate worldgen readiness."""
    work = Path(tempfile.mkdtemp(prefix="hyworld_worldgen_preflight_", dir="/tmp"))
    scene_root = _extract_zip(scene_zip, work / "scene")
    proc = subprocess.run(
        ["python", "scripts/preflight_worldgen_scene.py", str(scene_root), "--repo-root", str(REMOTE_REPO), "--json"],
        cwd=str(REMOTE_REPO),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _planned_worldgen_commands(scene_root: Path, out: Path, llm_addr: str, llm_port: int, llm_name: str, nproc: int = 4) -> list[list[str]]:
    return [
        ["python", "hyworld2/worldgen/traj_generate.py", "--target_path", str(scene_root), "--llm_addr", llm_addr, "--llm_port", str(llm_port), "--llm_name", llm_name, "--apply_nav_traj", "--apply_up_route", "--apply_recon_iteration", "--force_vlm"],
        ["torchrun", "--nproc_per_node", str(nproc), "hyworld2/worldgen/traj_render.py", "--target_path", str(scene_root), "--llm_addr", llm_addr, "--llm_port", str(llm_port), "--llm_name", llm_name],
        ["torchrun", "--nproc_per_node", str(nproc), "hyworld2/worldgen/video_gen.py", "--target_path", str(scene_root), "--fsdp"],
        ["torchrun", "--nproc_per_node", str(nproc), "hyworld2/worldgen/gen_gs_data.py", "--root_path", str(scene_root), "--save_normal", "--split_sky"],
        ["python", "-m", "hyworld2.worldgen.world_gs_trainer", "default", "--data_dir", str(scene_root / "gs_data"), "--result_dir", str(out / "gs_result"), "--max_steps", "2000", "--save_steps", "2000", "--eval_steps", "2000", "--ply_steps", "2000", "--save_ply", "--convert_to_spz", "--disable_video", "--depth_loss", "--normal_loss", "--use_mask_gaussian"],
    ]


@app.function(image=image, timeout=20 * 60, volumes={str(HF_CACHE): cache_volume})
def plan_worldgen_archive(
    scene_zip: bytes,
    llm_addr: str,
    llm_port: int = 8000,
    llm_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    nproc: int = 4,
) -> bytes:
    """No-GPU planner: produce preflight + exact five-stage command plan."""
    work = Path(tempfile.mkdtemp(prefix="hyworld_worldgen_plan_", dir="/tmp"))
    scene_root = _extract_zip(scene_zip, work / "scene")
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    env = {"LLM_ADDR": llm_addr, "LLM_PORT": str(llm_port), "LLM_NAME": llm_name}
    preflight = subprocess.run(
        ["python", "scripts/preflight_worldgen_scene.py", str(scene_root), "--repo-root", str(REMOTE_REPO), "--json"],
        cwd=str(REMOTE_REPO), text=True, capture_output=True, check=False, env={**os.environ, **env}
    )
    (out / "preflight.json").write_text(preflight.stdout or preflight.stderr, encoding="utf-8")
    commands = _planned_worldgen_commands(scene_root, out, llm_addr, llm_port, llm_name, nproc=nproc)
    (out / "planned_commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    (out / "STATUS.txt").write_text(
        "No-GPU worldgen plan succeeded. This validates scene upload, repo sync, preflight, and exact 5-stage command planning.\n"
        "It does not execute GPU/VLM stages.\n",
        encoding="utf-8",
    )
    return _zip_dir(out)


@app.function(image=image, gpu="H100:4", timeout=12 * 60 * 60, volumes={str(HF_CACHE): cache_volume})
def generate_world_archive(
    scene_zip: bytes,
    llm_addr: str,
    llm_port: int = 8000,
    llm_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    dry_run: bool = True,
) -> bytes:
    """Scaffold for the full five-stage HY-World worldgen pipeline.

    Set dry_run=False only after the image/dependencies/VLM endpoint are validated.
    """
    work = Path(tempfile.mkdtemp(prefix="hyworld_worldgen_", dir="/tmp"))
    scene_root = _extract_zip(scene_zip, work / "scene")
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)

    env = {"LLM_ADDR": llm_addr, "LLM_PORT": str(llm_port), "LLM_NAME": llm_name}
    preflight = subprocess.run(
        ["python", "scripts/preflight_worldgen_scene.py", str(scene_root), "--repo-root", str(REMOTE_REPO), "--json"],
        cwd=str(REMOTE_REPO), text=True, capture_output=True, check=False, env={**os.environ, **env}
    )
    (out / "preflight.json").write_text(preflight.stdout or preflight.stderr, encoding="utf-8")

    commands = [
        ["python", "hyworld2/worldgen/traj_generate.py", "--target_path", str(scene_root), "--llm_addr", llm_addr, "--llm_port", str(llm_port), "--llm_name", llm_name, "--apply_nav_traj", "--apply_up_route", "--apply_recon_iteration", "--force_vlm"],
        ["torchrun", "--nproc_per_node", "4", "hyworld2/worldgen/traj_render.py", "--target_path", str(scene_root), "--llm_addr", llm_addr, "--llm_port", str(llm_port), "--llm_name", llm_name],
        ["torchrun", "--nproc_per_node", "4", "hyworld2/worldgen/video_gen.py", "--target_path", str(scene_root), "--fsdp"],
        ["torchrun", "--nproc_per_node", "4", "hyworld2/worldgen/gen_gs_data.py", "--root_path", str(scene_root), "--save_normal", "--split_sky"],
        ["python", "-m", "hyworld2.worldgen.world_gs_trainer", "default", "--data_dir", str(scene_root / "gs_data"), "--result_dir", str(out / "gs_result"), "--max_steps", "2000", "--save_steps", "2000", "--eval_steps", "2000", "--ply_steps", "2000", "--save_ply", "--convert_to_spz", "--disable_video", "--depth_loss", "--normal_loss", "--use_mask_gaussian"],
    ]
    (out / "planned_commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")

    if dry_run:
        (out / "DRY_RUN.txt").write_text("Dry run only: generated planned_commands.json and preflight.json.\n", encoding="utf-8")
        return _zip_dir(out)

    for cmd in commands:
        _run(cmd, cwd=REMOTE_REPO, env=env)
    return _zip_dir(out)


@app.function(image=image, timeout=20 * 60, volumes={str(HF_CACHE): cache_volume})
def probe_worldgen_env() -> dict:
    """No-GPU dependency probe for the worldgen Modal image."""
    import importlib.util
    import sys

    modules = [
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "open3d",
        "cv2",
        "numpy",
        "PIL",
        "gsplat",
        "trimesh",
        "omegaconf",
        "openai",
    ]
    found = {name: importlib.util.find_spec(name) is not None for name in modules}
    return {
        "python": sys.version,
        "repo_exists": REMOTE_REPO.exists(),
        "worldgen_exists": (REMOTE_REPO / "hyworld2/worldgen/traj_generate.py").exists(),
        "modules": found,
        "missing": [name for name, ok in found.items() if not ok],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


@app.local_entrypoint()
def probe_env_local(output_path: str = "hyworld_worldgen_env_probe.json"):
    """Run the no-GPU dependency probe and save JSON locally."""
    result = probe_worldgen_env.remote()
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved env probe: {out}")


@app.local_entrypoint()
def preflight_local(input_path: str, output_path: str = "hyworld_worldgen_preflight_result.json"):
    """Upload a local scene folder/zip and run the no-GPU Modal preflight."""
    src = Path(input_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    if src.is_dir():
        payload = _zip_dir(src)
    else:
        payload = src.read_bytes()
    result = preflight_worldgen_archive.remote(payload)
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved preflight result: {out}")


@app.local_entrypoint()
def plan_local(
    input_path: str,
    output_path: str = "hyworld_worldgen_plan_result.zip",
    llm_addr: str = "127.0.0.1",
    llm_port: int = 8000,
    llm_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    nproc: int = 4,
):
    """Upload a scene and return preflight + exact 5-stage command plan, no GPU."""
    src = Path(input_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    if src.is_dir():
        payload = _zip_dir(src)
    else:
        payload = src.read_bytes()
    result = plan_worldgen_archive.remote(
        payload,
        llm_addr=llm_addr,
        llm_port=llm_port,
        llm_name=llm_name,
        nproc=nproc,
    )
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(result)
    print(f"Saved worldgen plan ZIP: {out}")


@app.local_entrypoint()
def generate_world_local(
    input_path: str,
    output_path: str = "hyworld_worldgen_result.zip",
    llm_addr: str = "127.0.0.1",
    llm_port: int = 8000,
    llm_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    dry_run: bool = True,
):
    """Run the full-worldgen Modal scaffold.

    Default `dry_run=True` still allocates the configured Modal function image/GPU
    if invoked; use `preflight_local` first for no-GPU validation.
    """
    src = Path(input_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    if src.is_dir():
        payload = _zip_dir(src)
    else:
        payload = src.read_bytes()
    result = generate_world_archive.remote(
        payload,
        llm_addr=llm_addr,
        llm_port=llm_port,
        llm_name=llm_name,
        dry_run=dry_run,
    )
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(result)
    print(f"Saved worldgen result ZIP: {out}")
