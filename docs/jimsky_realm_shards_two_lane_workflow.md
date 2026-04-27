# Jimsky Realm Shards — Two-Lane Reconstruction Workflow

Date: 2026-04-27
Status: working plan for the next HY-World / Unreal test run

## Core idea

A **Realm Shard** is a reusable piece of a real or imagined world: a room, venue, stage, installation, portal scene, or synthetic environment that can be reconstructed, previewed, packaged, and eventually loaded into Unreal / browser viewers / maps.

We now have two complementary capture/generation lanes.

---

## Lane A — Panel / Multi-View Shard

Use multiple overlapping panels/stills/images, like the previous hacker-room attempts.

### Best for

- Generated concept rooms where we can create several consistent angle panels.
- Real venue/stage captures from phone photos.
- Testing how many consistent views are needed before HY-World gives usable point clouds / splats.

### Input recipe

Create or capture **8–18 panels** instead of only 3–5 when possible.

Recommended view order:

1. Front-left wide
2. Front-center wide
3. Front-right wide
4. Left wall / side detail
5. Right wall / side detail
6. Rear-left
7. Rear-center
8. Rear-right
9. Ceiling/upper lighting if important
10. Floor/ground plane if important
11–18. Detail passes with overlap, not random unrelated designs

### Prompting rule for generated panels

Every panel should repeat a fixed spatial contract:

- same room dimensions
- same anchor objects
- same wall/floor/ceiling materials
- same lighting direction
- same camera height
- same lens feeling
- specify the camera position per panel

Example prompt skeleton:

```text
Create one panel from a consistent multi-view set for a Realm Shard reconstruction.
Room: [fixed description].
Anchor objects that must remain consistent: [list].
Camera position: [front-left / rear-right / etc.], eye height 1.6m, 24mm wide lens.
Preserve the same room layout, object placement, wall colors, floor pattern, lighting direction, and scale.
No text labels, no UI overlays, no impossible geometry, no redesigned room.
```

### HY-World settings to test

Budget/quick:

```bash
--target-size 384 \
--video-max-frames 12 \
--video-min-frames 8 \
--fps 1 \
--render-interp-per-pair 12
```

Better quality:

```bash
--target-size 512 \
--video-max-frames 18 \
--video-min-frames 10 \
--fps 1 \
--render-interp-per-pair 17 \
--save_colmap \
--save_conf
```

Potential sharper/larger experiment, use only after the 512 run looks promising:

```bash
--target-size 768 \
--video-max-frames 24 \
--video-min-frames 12 \
--fps 1 \
--render-interp-per-pair 17 \
--save_colmap \
--save_conf \
--compress_pts_max_points 3000000 \
--compress_gs_max_points 7000000
```

Notes:

- There is no obvious exposed `training steps` / `fine-tune gaussian splat steps` CLI flag in the current WorldMirror pipeline. This is primarily a feed-forward model that predicts splats/points/depth/cameras.
- The knobs we can actually tune are: resolution, number/quality/consistency of views, masking/confidence/edge settings, and compression limits.
- If output is blurry, first improve input consistency and target size before assuming more Gaussian optimization is available.

---

## Lane B — Equirectangular / 360 Reference Shard

Generate one full **equirectangular 360 image** as the spatial master reference, then derive directional views from it.

The user called this “echo rectangular”; the technical term is **equirectangular panorama**.

### Best for

- Creating a complete imagined room/world where every direction is covered.
- Reducing contradictions between separate generated panels.
- Making a map/atlas-style shard with spatial coordinates.
- Unreal tests where a panoramic skybox/reference can guide layout.

### Input recipe

1. Generate one high-resolution equirectangular panorama.
2. Convert/slice it into perspective views around a compass/time dial.
3. Feed those perspective views to HY-World, not just the panorama directly.
4. Package both:
   - master panorama
   - derived perspective panels
   - camera coordinate manifest
   - reconstruction outputs
   - Unreal notes

### Panorama prompt skeleton

```text
Create a seamless equirectangular 360-degree panorama for a Realm Shard.
Format: 2:1 equirectangular panorama, full spherical environment, continuous left/right edges.
Scene: [fixed room/venue/world].
Spatial anchors:
- North / 12 o'clock: [object/feature]
- East / 3 o'clock: [object/feature]
- South / 6 o'clock: [object/feature]
- West / 9 o'clock: [object/feature]
Ceiling/upper half: [lighting/rigging/sky]
Floor/lower half: [floor/ground details]
Lighting: consistent single environment lighting.
No text, no logos, no UI, no people unless requested, no duplicated impossible objects.
```

### Coordinate / time-dial convention

Use a simple clock/compass coordinate system for derived views:

| Label | Yaw | Meaning |
|---|---:|---|
| 12 o'clock / North | 0° | main hero view |
| 1:30 | 45° | front-right |
| 3 o'clock / East | 90° | right side |
| 4:30 | 135° | rear-right |
| 6 o'clock / South | 180° | rear |
| 7:30 | 225° | rear-left |
| 9 o'clock / West | 270° | left side |
| 10:30 | 315° | front-left |

Pitch passes:

- 0° eye-level
- +25° upper walls / ceiling / lighting
- -25° floor / ground / stage deck

For HY-World, start with eye-level 8 views, then add upper/lower views if needed.

### Derived perspective view settings

Recommended extraction target:

```text
FOV: 75–90 degrees
aspect: 16:9
size: 1536x864 or 1920x1080
views: 8 yaw views at pitch 0 first
optional: pitch +25 and -25 for vertical coverage
```

### HY-World settings to test

Quick panorama-derived run:

```bash
--target-size 512 \
--video-max-frames 8 \
--video-min-frames 8 \
--fps 1 \
--render-interp-per-pair 17 \
--save_colmap \
--save_conf
```

Expanded 360 run:

```bash
--target-size 512 \
--video-max-frames 16 \
--video-min-frames 8 \
--fps 1 \
--render-interp-per-pair 12 \
--save_colmap \
--save_conf
```

---

## Quality checklist before spending GPU

- Are all panels/perspective views from the same room/world?
- Do anchor objects remain stable across views?
- Are there enough overlapping surfaces between adjacent views?
- Are images sharp, not painterly/noisy/blurred?
- Are there no text labels or UI overlays inside the images?
- Is there a manifest mapping each view to yaw/pitch/clock direction?

---

## Packaging checklist for Unreal test

Each Realm Shard run should package:

```text
README_UNREAL_IMPORT.md
realm_shard_manifest.json
source_panorama/                 # Lane B only
source_panels/ or derived_views/
input_contact_sheet.jpg
rendered/rendered_rgb.mp4
screenshots/contact_sheet.jpg
points.ply
gaussians.ply
camera_params.json
depth/
normal/
pipeline_timing.json
```

Unreal note:

- `points.ply` is simplest for initial inspection / conversion tests.
- `gaussians.ply` is the higher-fidelity target but needs a real Gaussian splat renderer/import path, not generic PLY point-cloud loading.
- If Unreal import is rough, also test the browser Three.js point-cloud viewer and/or dedicated Gaussian splat viewers for comparison.

---

## Next recommended experiment

Run two tiny but structured tests on the same concept:

1. **Realm Shards Panel Test 002**
   - 12 consistent panels
   - target size 512
   - save confidence + COLMAP

2. **Realm Shards Panorama Test 001**
   - one 2:1 equirectangular master
   - 8 derived yaw views
   - target size 512
   - same packaging

Compare:

- blur/sharpness
- geometry consistency
- holes
- camera path stability
- Unreal import usefulness
- cost/time

Winner becomes the default workflow; loser becomes a fallback/reference lane.
