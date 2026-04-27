# Jimsky HY-World → Unreal Import Notes

HY-World / WorldMirror outputs are best treated as a portable reconstruction bundle, not a finished Unreal scene. The first useful target is a point-cloud/splat preview, then optional conversion into meshes/collision later.

## Output bundle shape

A successful Modal batch result contains:

```text
JIMSKY_RESULT_MANIFEST.txt
camera_params.json
depth/*.npy + depth/*.png
gaussians.ply
normal/*.png
pipeline_timing.json
points.ply
rendered/rendered_rgb.mp4
```

## Unreal lane 1: point cloud proof-of-scene

1. Enable Unreal's **LiDAR Point Cloud Support** plugin.
2. Restart Unreal.
3. Import `points.ply`.
4. Drop the point cloud into a level.
5. Adjust scale/origin/material settings.
6. Use `rendered/rendered_rgb.mp4` beside the asset as the visual reference.

This is the fastest path for a first “yes, this capture became a navigable spatial artifact” demo.

## Unreal lane 2: Gaussian splats

`gaussians.ply` is not a standard mesh. Use one of these paths:

- Import with an Unreal Gaussian Splatting plugin that supports `.ply` splats.
- Convert the splat to a viewer/runtime format used by Three.js/WebXR if the demo is web-first.
- Convert or reconstruct a mesh from `points.ply` if collision or static mesh workflows are required.

## Camera/depth/normal reuse

- `camera_params.json` stores reconstruction camera parameters.
- `depth/*.png` and `normal/*.png` can drive shader/material/post-process experiments.
- `rendered/rendered_rgb.mp4` is the phone-friendly preview to share immediately.

## Recommended project layout

```text
Content/JimskyWorlds/<capture-name>/
  SourceHYWorldZip/
  PointCloud/points.ply
  GaussianSplat/gaussians.ply
  Cameras/camera_params.json
  Preview/rendered_rgb.mp4
  Maps/<capture-name>_Preview.umap
```

## Demo verification

On 2026-04-27, a bounded Modal A100 smoke test with five synthetic overlapping frames succeeded after adding FlashAttention support to the Modal image. The result ZIP included `gaussians.ply`, `points.ply`, depth/normal maps, camera JSON, timing JSON, and `rendered/rendered_rgb.mp4`.
