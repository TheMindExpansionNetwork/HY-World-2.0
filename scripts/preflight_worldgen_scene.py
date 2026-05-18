#!/usr/bin/env python3
"""Preflight a HY-World 2.0 world-generation scene folder.

This is a no-GPU/no-network helper. It validates that a target scene folder has the
minimum files/structure needed before launching the expensive five-stage worldgen
pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


STAGE_MARKERS = {
    "stage0_panorama": ["panorama.png"],
    "stage0_meta": ["meta_info.json"],
    "stage1_trajectory_plan": ["render_results"],
    "stage2_rendered_trajectories": ["render_results"],
    "stage3_worldstereo_bank": ["render_results"],
    "stage4_gs_data": ["gs_data"],
}

WORLDGEN_SCRIPTS = [
    "hyworld2/worldgen/traj_generate.py",
    "hyworld2/worldgen/traj_render.py",
    "hyworld2/worldgen/video_gen.py",
    "hyworld2/worldgen/gen_gs_data.py",
    "hyworld2/worldgen/world_gs_trainer.py",
]


def exists_any(root: Path, rels: list[str]) -> bool:
    return any((root / rel).exists() for rel in rels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target_path", help="HY-World worldgen scene directory or parent directory")
    ap.add_argument("--repo-root", default=".", help="HY-World repo root")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    target = Path(args.target_path).expanduser().resolve()
    repo = Path(args.repo_root).expanduser().resolve()

    report = {
        "target_path": str(target),
        "target_exists": target.exists(),
        "target_is_dir": target.is_dir(),
        "repo_root": str(repo),
        "worldgen_scripts": {rel: (repo / rel).exists() for rel in WORLDGEN_SCRIPTS},
        "env": {
            "LLM_ADDR": bool(os.environ.get("LLM_ADDR")),
            "LLM_PORT": bool(os.environ.get("LLM_PORT")),
            "LLM_NAME": bool(os.environ.get("LLM_NAME")),
        },
        "scenes": [],
        "ready_for_stage1": False,
        "warnings": [],
    }

    if not target.exists() or not target.is_dir():
        report["warnings"].append("target_path does not exist or is not a directory")
    else:
        if (target / "panorama.png").exists() or (target / "meta_info.json").exists():
            candidates = [target]
        else:
            candidates = [p for p in sorted(target.iterdir()) if p.is_dir()]

        for scene in candidates:
            scene_report = {
                "path": str(scene),
                "name": scene.name,
                "markers": {name: exists_any(scene, rels) for name, rels in STAGE_MARKERS.items()},
                "render_result_dirs": len(list((scene / "render_results").glob("*"))) if (scene / "render_results").exists() else 0,
            }
            scene_report["ready_for_stage1"] = scene_report["markers"]["stage0_panorama"] and scene_report["markers"]["stage0_meta"]
            report["scenes"].append(scene_report)

    report["ready_for_stage1"] = any(s.get("ready_for_stage1") for s in report["scenes"])

    missing_scripts = [rel for rel, ok in report["worldgen_scripts"].items() if not ok]
    if missing_scripts:
        report["warnings"].append(f"missing worldgen scripts: {missing_scripts}")
    if not all(report["env"].values()):
        report["warnings"].append("VLM env not fully set; stages 1 and 2 need LLM_ADDR, LLM_PORT, LLM_NAME")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
