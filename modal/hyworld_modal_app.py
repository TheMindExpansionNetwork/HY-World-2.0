"""
Jimsky / HY-World 2.0 Modal runner.

Purpose:
  phone photos/video -> Modal GPU -> HY-World/WorldMirror reconstruction -> ZIP artifact

This file is intentionally safe to commit: no secrets, no tokens, no baked credentials.
Run only when the user explicitly approves GPU spend.

Typical use from the repo root after `modal setup`:
  modal run modal/hyworld_modal_app.py::reconstruct_local \
    --input-path /path/to/photos_or_video_or_zip \
    --output-path /tmp/hyworld_result.zip \
    --target-size 952 \
    --video-max-frames 32 \
    --gpu A100-80GB

Notes:
  - First run downloads model weights from Hugging Face into the Modal Volume.
  - Outputs include GLB/PLY/depth/normal/camera JSON and, when rendering succeeds,
    rendered/rendered_rgb.mp4 for preview.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

APP_NAME = "jimsky-hyworld-2-world-reconstruction"
REMOTE_REPO = Path("/workspace/HY-World-2.0")
REMOTE_INPUT = Path("/workspace/input")
REMOTE_OUTPUT = Path("/workspace/output")
HF_CACHE = Path("/cache/huggingface")

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("jimsky-hyworld2-cache", create_if_missing=True)

# Keep this close to upstream's README: Python 3.10, PyTorch 2.4.0 + CUDA 12.4,
# and gsplat's matching pt24/cu124 wheel. FlashAttention is deliberately omitted
# from the baseline image because it can be slow/brittle to build; add it later
# after the first successful smoke test if throughput requires it.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "mesa-utils",
        "curl",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja",
        "pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "gradio==5.49.1",
        "moviepy==1.0.3",
        "jaxtyping",
        "typeguard",
        "tqdm",
        "omegaconf",
        "plyfile",
        "colorspacious",
        "pydantic",
        "opencv-python",
        "scipy",
        "requests",
        "trimesh",
        "matplotlib",
        "spaces",
        "pillow_heif",
        "onnxruntime",
        "einops",
        "torchmetrics",
        "uniception",
        "safetensors",
        "numpy<2.0.0",
        "open3d==0.18.0",
        "pycolmap==3.10.0",
        "hf_transfer",
        "huggingface_hub[hf_transfer]",
        "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl",
    )
    .pip_install(
        # Upstream imports flash_attn at module import time. Use a prebuilt wheel
        # matching Python 3.10 + Torch 2.4 + CUDA 12 + PyTorch's default cxx11 ABI
        # because Modal's slim image does not include CUDA_HOME/nvcc for source builds.
        "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
    )
    .env(
        {
            "HF_HOME": str(HF_CACHE),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(".", remote_path=str(REMOTE_REPO))
)


def _zip_dir(src: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src))
    return buf.getvalue()


def _extract_archive(archive_bytes: bytes, filename: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    lower = filename.lower()
    raw_path = dest / filename
    raw_path.write_bytes(archive_bytes)

    if lower.endswith(".zip"):
        unpack_dir = dest / "unzipped"
        unpack_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw_path) as zf:
            zf.extractall(unpack_dir)
        return unpack_dir

    # For a single video/image input, pass the file directly to HY-World. The pipeline
    # can ingest a video file and sample frames itself.
    return raw_path


def _run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={str(HF_CACHE): cache_volume},
)
def reconstruct_archive(
    archive_bytes: bytes,
    filename: str = "input.zip",
    target_size: int = 952,
    fps: int = 1,
    video_min_frames: int = 4,
    video_max_frames: int = 32,
    save_rendered: bool = True,
    render_depth: bool = False,
    render_interp_per_pair: int = 12,
    save_colmap: bool = False,
    save_conf: bool = False,
    apply_confidence_mask: bool = False,
    confidence_percentile: float = 10.0,
    no_compress_pts: bool = False,
    compress_pts_max_points: int = 2_000_000,
    compress_pts_voxel_size: float = 0.002,
    compress_gs_max_points: int = 5_000_000,
    max_resolution: int = 1920,
    prior_cam_relpath: str | None = None,
) -> bytes:
    """Run HY-World reconstruction on a zip/images/video payload and return a ZIP."""
    work = Path(tempfile.mkdtemp(prefix="hyworld_job_", dir="/tmp"))
    input_root = work / "input"
    output_root = work / "output"
    strict_out = output_root / "result"
    strict_out.mkdir(parents=True, exist_ok=True)

    input_path = _extract_archive(archive_bytes, filename, input_root)

    cmd = [
        "python",
        "-m",
        "hyworld2.worldrecon.pipeline",
        "--input_path",
        str(input_path),
        "--strict_output_path",
        str(strict_out),
        "--target_size",
        str(target_size),
        "--fps",
        str(fps),
        "--video_min_frames",
        str(video_min_frames),
        "--video_max_frames",
        str(video_max_frames),
        "--no_interactive",
    ]
    if prior_cam_relpath:
        prior_path = (input_path / prior_cam_relpath) if input_path.is_dir() else (input_root / prior_cam_relpath)
        if prior_path.exists():
            cmd += ["--prior_cam_path", str(prior_path)]
        else:
            print(f"[Modal] Warning: requested prior_cam_relpath not found: {prior_path}")
    if save_rendered:
        cmd += ["--save_rendered", "--render_interp_per_pair", str(render_interp_per_pair)]
    if render_depth:
        cmd += ["--render_depth"]
    if save_colmap:
        cmd += ["--save_colmap"]
    if save_conf:
        cmd += ["--save_conf"]
    if apply_confidence_mask:
        cmd += ["--apply_confidence_mask", "--confidence_percentile", str(confidence_percentile)]
    if no_compress_pts:
        cmd += ["--no_compress_pts"]
    else:
        cmd += [
            "--compress_pts_max_points",
            str(compress_pts_max_points),
            "--compress_pts_voxel_size",
            str(compress_pts_voxel_size),
        ]
    cmd += ["--compress_gs_max_points", str(compress_gs_max_points), "--max_resolution", str(max_resolution)]

    _run(cmd, cwd=REMOTE_REPO)

    manifest = strict_out / "JIMSKY_RESULT_MANIFEST.txt"
    manifest.write_text(
        "HY-World 2.0 / WorldMirror reconstruction output\n"
        "Key artifacts:\n"
        "- gaussians.ply: 3D Gaussian splat point representation\n"
        "- points.ply: colored point cloud\n"
        "- camera_params.json: camera poses/intrinsics\n"
        "- depth/ and normal/: per-view maps\n"
        "- rendered/rendered_rgb.mp4: preview video, if render succeeded\n",
        encoding="utf-8",
    )
    return _zip_dir(strict_out)


@app.local_entrypoint()
def reconstruct_local(
    input_path: str,
    output_path: str = "hyworld_result.zip",
    target_size: int = 952,
    fps: int = 1,
    video_min_frames: int = 4,
    video_max_frames: int = 32,
    render_interp_per_pair: int = 12,
    gpu: str = "A100-80GB",
    render_depth: bool = False,
    save_colmap: bool = False,
    save_conf: bool = False,
    apply_confidence_mask: bool = False,
    confidence_percentile: float = 10.0,
    no_compress_pts: bool = False,
    compress_pts_max_points: int = 2_000_000,
    compress_pts_voxel_size: float = 0.002,
    compress_gs_max_points: int = 5_000_000,
    max_resolution: int = 1920,
    prior_cam_path: str | None = None,
):
    """Upload a local file/folder/zip to Modal and save returned ZIP locally.

    If input_path is a directory, it is zipped first. If it is a photo/video/zip,
    it is uploaded directly.
    """
    src = Path(input_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    temp_zip = None
    prior_cam_relpath = None
    prior_src = Path(prior_cam_path).expanduser().resolve() if prior_cam_path else None
    if prior_src and not prior_src.exists():
        raise FileNotFoundError(prior_src)
    if src.is_dir():
        if prior_src:
            try:
                prior_cam_relpath = str(prior_src.relative_to(src))
            except ValueError:
                raise ValueError("prior_cam_path must be inside input_path directory when uploading a folder")
        temp_zip = Path(tempfile.mktemp(suffix=".zip"))
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(src.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(src))
        upload_path = temp_zip
        filename = src.name + ".zip"
    else:
        if prior_src:
            raise ValueError("prior_cam_path is currently supported only when input_path is a directory containing the prior JSON")
        upload_path = src
        filename = src.name

    try:
        result = reconstruct_archive.remote(
            upload_path.read_bytes(),
            filename=filename,
            target_size=target_size,
            fps=fps,
            video_min_frames=video_min_frames,
            video_max_frames=video_max_frames,
            render_depth=render_depth,
            render_interp_per_pair=render_interp_per_pair,
            save_colmap=save_colmap,
            save_conf=save_conf,
            apply_confidence_mask=apply_confidence_mask,
            confidence_percentile=confidence_percentile,
            no_compress_pts=no_compress_pts,
            compress_pts_max_points=compress_pts_max_points,
            compress_pts_voxel_size=compress_pts_voxel_size,
            compress_gs_max_points=compress_gs_max_points,
            max_resolution=max_resolution,
            prior_cam_relpath=prior_cam_relpath,
        )
    finally:
        if temp_zip and temp_zip.exists():
            temp_zip.unlink()

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(result)
    print(f"Saved HY-World result ZIP: {out}")
