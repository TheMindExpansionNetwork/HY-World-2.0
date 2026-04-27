#!/usr/bin/env python3
"""Package and summarize a HY-World / WorldMirror result directory.

This helper is intentionally local/no-secrets. It finds likely preview/artifact files,
writes a manifest, and creates a ZIP suitable for Telegram/Drive delivery.
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

KEY_PATTERNS = [
    "rendered/rendered_rgb.mp4",
    "rendered/rendered_depth.mp4",
    "gaussians.ply",
    "points.ply",
    "camera_params.json",
    "pipeline_timing.json",
]


def sizeof(path: Path) -> str:
    n = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", help="HY-World output/result directory")
    ap.add_argument("--zip", dest="zip_path", default=None, help="Output zip path")
    args = ap.parse_args()

    root = Path(args.result_dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    key = []
    for pat in KEY_PATTERNS:
        p = root / pat
        if p.exists():
            key.append({"path": pat, "size": sizeof(p)})

    manifest = {
        "root": str(root),
        "total_files": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "key_artifacts": key,
        "preview_video": next((x["path"] for x in key if x["path"].endswith("rendered_rgb.mp4")), None),
    }
    manifest_path = root / "JIMSKY_PACKAGE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = Path(args.zip_path).expanduser().resolve() if args.zip_path else root.with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root))

    print(json.dumps({"zip": str(zip_path), "zip_size": sizeof(zip_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
