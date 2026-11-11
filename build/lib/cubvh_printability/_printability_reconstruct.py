"""Hierarchical UDF offset reconstruction using a complete packed corner sign grid."""
import itertools
import logging
import math
import numpy as np
import trimesh

from .core import PrintabilityDependencyError

# Same corner order as cuBVH sparse marching cubes.
CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
CHUNK = 262144
_LOG = logging.getLogger(__name__)


def _bitmap_get(bitmap, positions, torch):
    # newbit owns a private stream. Its methods wait for that stream, but not
    # for input producers on the current PyTorch stream: bridge explicitly.
    positions = positions.to(torch.int16).contiguous()
    torch.cuda.current_stream(positions.device).synchronize()
    return bitmap.get_values(positions)


def _bitmap_set(bitmap, positions, torch):
    positions = positions.to(torch.int16).contiguous()
    torch.cuda.current_stream(positions.device).synchronize()
    bitmap.set_positions(positions)


def mark_offset_corners(bvh, bitmap, resolution, half_extent, radius, torch):
    """Distance is 1-Lipschitz: reject a block only if every corner is > r.

    Blocks partition the (resolution+1)^3 lattice, including its last plane.
    Leaf queries are bounded, and every f<=0 corner is marked. No dense float
    volume is allocated. Coordinates and linear indices stay int64.
    """
    device = bitmap.device
    step = 2 * half_extent / resolution
    blocks = [(torch.zeros((1, 3), device=device, dtype=torch.int64),
               torch.full((1, 3), resolution + 1, device=device, dtype=torch.int64))]
    bits = torch.tensor(list(itertools.product((False, True), repeat=3)), device=device)
    leaf_offsets = torch.cartesian_prod(*(torch.arange(8, device=device) for _ in range(3)))
    while blocks:
        low, high = blocks.pop()
        if len(low) > 4096:
            blocks.append((low[4096:], high[4096:]))
            low, high = low[:4096], high[:4096]
        centers = (low + high - 1).float() * (.5 * step) - half_extent
        distance = bvh.unsigned_distance(centers)[0]
        bound = torch.linalg.vector_norm((high - low - 1).float(), dim=1) * (.5 * step)
        active = distance <= radius + bound + 8 * np.finfo(np.float32).eps * half_extent
        low, high = low[active], high[active]
        leaf = (high - low).amax(dim=1) <= 8
        lo, hi = low[~leaf], high[~leaf]
        if len(lo):
            mid = (lo + hi) // 2
            child_low = torch.where(bits[None], mid[:, None], lo[:, None]).reshape(-1, 3)
            child_high = torch.where(bits[None], hi[:, None], mid[:, None]).reshape(-1, 3)
            valid = (child_high > child_low).all(dim=1)
            blocks.append((child_low[valid], child_high[valid]))
        lo, hi = low[leaf], high[leaf]
        if len(lo):
            # Enumerate up to 4096*512 corners directly on the GPU, avoiding
            # millions of Python/NumPy leaf-block allocations at 1536/2048.
            points = lo[:, None] + leaf_offsets[None]
            points = points[(points < hi[:, None]).all(dim=2)]
            for coords in points.split(CHUNK):
                field = bvh.unsigned_distance(coords.float() * step - half_extent)[0] - radius
                _bitmap_set(bitmap, coords[field <= 0], torch)


def filled_corner_field(bvh, bitmap, corners, step, half_extent, radius, torch):
    """Evaluate the MC field using floodfilled signs, including deep interior.

    UDF-r can remain positive inside a closed shell. The filled bitmap, not
    that raw value, determines the sign. Unqueried interior corners have the
    same implicit negative sign; they need no stored floating-point value.
    """
    magnitude = (bvh.unsigned_distance(corners.float() * step - half_extent)[0] - radius).abs()
    # MC uses < iso: keep exact-zero occupied corners negative and avoid
    # coincident edge vertices with a deterministic perturbation.
    magnitude = torch.where(magnitude == 0, step * 1e-6, magnitude)
    inside = _bitmap_get(bitmap, corners, torch)
    return torch.where(inside, -magnitude, magnitude)


def crossing_cells(bitmap, resolution, torch):
    """Scan complete filled signs in two-plane slabs, not a sparse support band."""
    device = bitmap.device
    n = resolution + 1
    yz = torch.arange(n * n, device=device, dtype=torch.int64)
    base = torch.stack((torch.zeros_like(yz), yz // n, yz % n), dim=1)

    def plane(x):
        base[:, 0] = x
        return torch.cat([_bitmap_get(bitmap, c, torch) for c in base.split(CHUNK)]).reshape(n, n)

    previous = plane(0)
    for x in range(resolution):
        current = plane(x + 1)
        signs = [p[dy:dy + resolution, dz:dz + resolution]
                 for p in (previous, current) for dy, dz in itertools.product((0, 1), repeat=2)]
        any_inside = signs[0].clone()
        all_inside = signs[0].clone()
        for sign in signs[1:]:
            any_inside |= sign
            all_inside &= sign
        yz_active = torch.nonzero(any_inside & ~all_inside)
        if len(yz_active):
            yield torch.cat((torch.full((len(yz_active), 1), x, device=device, dtype=torch.int64), yz_active), dim=1)
        previous = current


class _ScaledUDF:
    def __init__(self, bvh, scale):
        self.bvh, self.scale = bvh, scale

    def unsigned_distance(self, positions):
        distance, face, uvw = self.bvh.unsigned_distance(positions / self.scale)
        return distance * self.scale, face, uvw


class Reconstructor:
    def __init__(self, mesh, resolution):
        try:
            import torch
            from newbit import Bitmap3D
            from cumesh import CuMesh
            from cubvh.api import cuBVH, sparse_marching_cubes
        except ImportError as exc:
            raise PrintabilityDependencyError('Repair requires CUDA PyTorch, compiled cuBVH, newbit and ../CuMesh') from exc
        if not torch.cuda.is_available():
            raise PrintabilityDependencyError('Repair requires an NVIDIA CUDA device')
        # Ensure OptiX is available before spending time on reconstruction.
        from .build_optix import build
        build()
        self.torch, self.Bitmap3D, self.CuMesh = torch, Bitmap3D, CuMesh
        self.mc = sparse_marching_cubes
        self.resolution = resolution
        self.scale = float(np.max(mesh.extents))
        # Normalize UDF coordinates to avoid the legacy cuBVH distance bound.
        udf_mesh = mesh.copy()
        udf_mesh.vertices = np.asarray(udf_mesh.vertices) / self.scale
        udf_mesh.update_faces(udf_mesh.area_faces > 0)
        while len(udf_mesh.faces) <= 8:
            udf_mesh = udf_mesh.subdivide()
        self.BVH = cuBVH
        self.udf_vertices = np.asarray(udf_mesh.vertices)
        self.udf_faces = np.asarray(udf_mesh.faces)
        self.device = torch.device('cuda', torch.cuda.current_device())

    def reconstruct(self, offset):
        torch = self.torch
        res = self.resolution
        radius = offset / self.scale
        # half_extent - .5 - r >= 2*step; add a numerical safety cell.
        half_extent = (.5 + radius) / (1 - 6 / res)
        if not math.isfinite(half_extent) or half_extent > np.finfo(np.float32).max / 16:
            return None  # No representable offset field for this attempt.
        bvh = _ScaledUDF(self.BVH(self.udf_vertices / half_extent, self.udf_faces), half_extent)
        step = 2 * half_extent / res
        bitmap = self.Bitmap3D(resolution=res + 1, device=self.device)
        bitmap.clear()
        _LOG.info("Marking offset corners: resolution=%d offset=%g mm", res, offset)
        mark_offset_corners(bvh, bitmap, res, half_extent, radius, torch)
        _LOG.info("Filling interior on %d cubed corner grid", res + 1)
        bitmap.fill_interior()
        _LOG.info("Enumerating complete crossing cells")
        chunks = list(crossing_cells(bitmap, res, torch))
        if not chunks:
            return None
        cells = torch.cat(chunks)
        del chunks
        _LOG.info("Crossing cells: %d; querying shared UDF corners", len(cells))
        # Existing sparse MC uses local int32 array offsets; never permit overflow.
        if len(cells) >= (2**31 - 1) // 12:
            raise MemoryError('sparse marching cubes exceeds its local index capacity')
        offsets = torch.tensor(CORNERS, device=self.device, dtype=torch.int64)
        n = res + 1
        keys = []
        for chunk in cells.split(CHUNK):
            corners = chunk[:, None] + offsets
            keys.append(((corners[..., 0] * n + corners[..., 1]) * n + corners[..., 2]).reshape(-1))
        keys = torch.unique(torch.cat(keys), sorted=True)
        values = torch.empty(len(keys), device=self.device, dtype=torch.float32)
        # Shared corners queried exactly once: preserve |UDF-r|, change only sign.
        for start in range(0, len(keys), CHUNK):
            key = keys[start:start + CHUNK]
            corners = torch.stack((key // (n * n), key // n % n, key % n), dim=1)
            values[start:start + len(key)] = filled_corner_field(
                bvh, bitmap, corners, step, half_extent, radius, torch)
        del bitmap
        field = torch.empty((len(cells), 8), device=self.device, dtype=torch.float32)
        for start in range(0, len(cells), CHUNK):
            corners = cells[start:start + CHUNK, None] + offsets
            key = (corners[..., 0] * n + corners[..., 1]) * n + corners[..., 2]
            field[start:start + len(corners)] = values[torch.searchsorted(keys, key)]
        del keys, values, bvh
        # MC/CuMesh allocate outside PyTorch. Return unused sort/query buffers
        # to CUDA rather than holding them in PyTorch's caching allocator.
        torch.cuda.empty_cache()
        _LOG.info("Extracting sparse marching cubes")
        vertices, faces = self.mc(cells, field, 0.0, ensure_consistency=False, high_precision=True)
        _LOG.info("MC result: %d vertices, %d faces", len(vertices), len(faces))
        if not len(faces):
            return None
        return trimesh.Trimesh(vertices=(vertices.cpu().numpy().astype(np.float64) * step - half_extent) * self.scale,
                               faces=faces.cpu().numpy(), process=False)

    def simplify(self, mesh, target):
        torch = self.torch
        cm = self.CuMesh()
        cm.init(torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=self.device),
                torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=self.device))
        torch.cuda.empty_cache()
        _LOG.info("Simplifying %d faces toward %d", len(mesh.faces), target)
        cm.simplify_quadric(target, verbose=_LOG.isEnabledFor(logging.INFO), options={'PreserveTopology': True, 'HardNormalCheck': True,
                                            'NormalCheck': True, 'NormalThrRad': math.pi / 2})
        vertices, faces = cm.read()
        # CuMesh stores float32 coordinates. Distinct high-precision MC vertices
        # can coincide on conversion even if the original candidate is valid.
        # Use link-checked, no-flip edge collapse, never area-based face deletion.
        for chunk in faces.split(CHUNK):
            triangles = vertices[chunk.long()].double()
            cross = torch.linalg.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0])
            if bool((cross == 0).all(dim=1).any().item()):
                _LOG.info('Collapsing float32 degeneracies with topology protection')
                cm.collapse_skinny_faces(min_angle_deg=1e-3, max_iterations=10)
                vertices, faces = cm.read()
                break
        return trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy(), process=False)
