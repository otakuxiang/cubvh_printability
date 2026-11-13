# cubvh-printability

`cubvh-printability` contains the Python printability orchestration migrated out of
`cubvh`. It uses the installed [`cubvh`](../) package for cuBVH and sparse CUDA
backend primitives; the OptiX helper sources are packaged here so this feature can
be installed and invoked independently. Import printability APIs from
`cubvh_printability`; the `cubvh` package itself exposes only its core backend.

## Install

Install `cubvh` and its CUDA dependencies first, then install this client:

```bash
python -m pip install -e ./cubvh_printability
```

For high-resolution repairs, build `cubvh` with the sparse-MC memory patches in
`patches/cubvh-sparse-mc-memory.patch` and
`patches/cubvh-sparse-mc-consistency-memory.patch` before installing its wheel
(for example, `cd ../cubvh && git apply ../cubvh_printability/patches/cubvh-sparse-mc-memory.patch \
../cubvh_printability/patches/cubvh-sparse-mc-consistency-memory.patch`). The first patch sorts
compact int32 edge keys and keeps only one representative per shared edge;
the second applies the same bounded-record design to the optional corner
consistency pass.
the Python reconstruction path also stages crossing cells outside the GPU.
Use `CUBVH_MC_PROFILE=1` to log native driver-memory checkpoints. The measured
RTX 5090 build completed 8M, 10M, and 20M-cell inputs; see
`docs/sparse-mc-memory.md` for the allocation model and reproduction commands.

## Python API

```python
from cubvh_printability import PrintabilityConfig, PrintabilityClient

config = PrintabilityConfig(
    min_thickness=0.05,
    size=50.0,
    resolution=1536,
    max_iso_scale=2.0,
    # Kept for compatibility; geometry is bounded by max_iso_scale.
    max_geometry_change_ratio=None,
    min_silhouette_iou=0.98,
)
result = PrintabilityClient().check_printability(
    "input.glb", config, output_path="repaired.ply"
)
print(result["printable"], result["failure_reason"])
```

A function-style API is also available:

```python
from cubvh_printability import check_printability
result = check_printability(
    "input.glb", min_thickness=0.05, size=50.0,
    max_iso_scale=2.0, output_path="repaired.ply",
)
```

For an already constructed configuration, `repair_mesh(mesh, config,
output_path=...)` is an equivalent named helper.

## CLI

```bash
cubvh-printability input.glb \
  --min-thickness 0.05 --size 50 \
  --max-iso-scale 2 --output repaired.ply
```

`max_geometry_change_ratio` and `min_silhouette_iou` remain accepted for
configuration compatibility. Repair iteration does not render camera views;
`max_iso_scale` is the geometry-growth limit.

The experimental `parts` strategy measures face thickness, ranks thin faces by
increasing `max(min_thickness - thickness, 0)` until their cumulative area meets
the repair budget, and builds a spatially varying signed
field. Missing area-uniform face samples receive three interior samples; valid
samples are aggregated with the p10 quantile. Unknown thickness and unpaired
thin faces remain unrepaired and consume the thin-area budget. If that area
already exhausts the budget, the strategy requests global fallback. Final
measurements continue counting unknown samples as thin.

A diagnosed inconsistency between local material directions and shell volume
can still permit a local proposal when all shell directions are resolved and
valid thickness queries exist. This does not mark the source measurement
reliable: the reconstructed output must pass a new reliable thickness/volume
measurement. Other reliability failures retain the fallback path.

```bash
cubvh-printability input.glb \
  --min-thickness 0.5 --size 50 --threshold 0.1 \
  --repair-strategy parts --output repaired.ply
```

The thickness deficit is shared symmetrically between opposing material
surfaces, without a camera or visibility pass. This keeps the local path
deterministic and avoids ten-direction ray tracing before SDF reconstruction.
Offsets spread along mesh edges with a compact smooth transition. The local
field retains original material signs (including cavities) and uses the union
of the original material SDF with variable-radius surface balls. This avoids
the discontinuity caused by switching to a nearest sheet with a different
radius. Outside the support of all growing balls the material is unchanged;
a zero-radius face can still be reached by growth from a nearby face.
Marching cubes can introduce sampling error. Local candidates are checked
again for topology and thickness; their radius is always capped by
`min_thickness * max_iso_scale / 2`.
Rejected repairs export the original scaled mesh and report `printable=false`.
Nonmeasurable inputs receive one global bootstrap attempt with normalized
offset `1 / resolution` (`size / resolution` mm), independent of the target
thickness and subject to the shared offset cap. Per-face profile
statistics are reported in `local_profile`; `local_fallback_reason` explains a
fallback decision.

The local path makes at most three thickness-driven SDF proposals
(`strength = 1, 1.25, 1.5`). Each candidate is reconstructed, then reduced toward
the input mesh's face count before topology, shell, and thickness validation.
There is no face-count parameter. QEM preserves topology and rejects face flips;
link-checked edge contractions resolve float32 vertex coincidences. If the
reduced candidate fails validation, the raw reconstruction is checked as a
fallback. The accepted mesh is the mesh exported to the caller.
There is no render-feedback loop or camera-dependent visibility pass.
The default `global` path measures per-face thickness before choosing growth.
Unmeasurable or unreliable inputs first receive a `size / resolution` mm
bootstrap and are measured again. Global offset is half an area-weighted
quantile of `max(min_thickness - face_thickness, 0)`, reserving unknown face
area and targeting 90% of the allowed thin-area ratio when possible.
The first proposal adds 10% radial slack or a quarter nominal voxel (whichever
is larger), capped at 5% of `min_thickness`. A rejected proposal gets at most
one retry with 20% extra radius or a quarter voxel. Both proposals use the
same original/bootstrap mesh. Reconstructed candidates retain their physical
growth instead of being resized back down before thickness validation.
Results include `global_offset_plan` and per-attempt offsets; the prediction
does not replace reliable thickness validation of the exported mesh.
Both paths cap growth using `min_thickness * max_iso_scale / 2` and record
`max_iso_scale` as the geometry constraint.
This bounds the requested offset field, not silhouette or normal changes.
Bootstrap and subsequent local growth share the same total offset budget.

Local reconstruction reuses shared voxel corners within bounded batches for
distance, material parity, and variable-radius field queries. It preserves the
requested grid resolution and gathers the resulting values back into the
original Marching Cubes corner order. Envelope bounds prune inactive blocks;
vertex-ball upper bounds and native triangle-distance lower bounds reduce
analytic distance solves. Per-attempt `reconstruction_seconds`,
`simplification_seconds`, `validation_seconds`, and option-level
`measurement_seconds` expose runtime cost. Large meshes use GPU topology and
shell connectivity when CuMesh is available, with cached float64 reductions.
`geometry_metrics.render_evaluated=false` identifies the camera-free policy.

Tests requiring CUDA can be enabled with `PRINTABILITY_GPU_TESTS=1 python -m
pytest`. `scripts/batch_cli.py INPUT OUTPUT` runs the requested 50 mm / 0.5 mm /
10% CLI batch and retains individual JSON results, logs, and output meshes.

The client imports `cubvh` lazily, so importing this package does not initialize
CUDA. A clear runtime error is raised if cubvh or its native dependencies are missing.
