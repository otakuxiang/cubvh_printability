"""Face-centroid material thickness visualization of the unchanged input surface."""
import logging
import math
from pathlib import Path
import time

import numpy as np
import trimesh

from ._printability_geometry import face_areas
from ._printability_thickness import (MaterialQuery, _material_side, thickness_directions,
                                      DEFAULT_MAX_NORMAL_ANGLE_DEG, DEFAULT_THICKNESS_BATCH_SIZE,
                                      WallExitFilter)
from .optix import OptixRayTracer

# Face classes: measured >= threshold, measured < threshold, unknown.
COLORS = np.array([[170, 180, 190, 255], [230, 45, 45, 255],
                   [255, 190, 35, 255]], dtype=np.uint8)
_LOG = logging.getLogger(__name__)


def load_input(path, size=100.):
    """Apply scene instances and uniform scale; do not weld, repair or simplify."""
    if not math.isfinite(size) or size <= 0:
        raise ValueError('size must be positive and finite')
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        parts = []
        for node in loaded.graph.nodes_geometry:
            transform, name = loaded.graph[node]
            part = loaded.geometry[name]
            if isinstance(part, trimesh.Trimesh) and len(part.faces):
                part = part.copy()
                part.apply_transform(transform)
                parts.append(part)
        if not parts:
            raise ValueError('scene contains no triangle meshes')
        loaded = trimesh.util.concatenate(parts)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError('input contains no triangles')
    if not np.isfinite(loaded.vertices).all():
        raise ValueError('vertices must be finite')
    bounds = loaded.bounds
    extent = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError('input extent must be positive and finite')
    loaded.vertices = (np.asarray(loaded.vertices) - bounds.mean(axis=0)) * (size / extent)
    return loaded


def measure_faces(mesh, threshold, *, batch_size=DEFAULT_THICKNESS_BATCH_SIZE,
                  max_normal_angle_deg=DEFAULT_MAX_NORMAL_ANGLE_DEG, require_opposing_surfaces=True):
    """One centroid per face, normal span with consensus for a thin decision.

    Uses the printability detector's side classification and its deterministic
    default 30-degree ray filter. Unknowns remain separate from measured thin faces.
    This local diagnostic does not certify the input as a closed valid solid.
    """
    import torch
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold must be positive and finite')
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    scale = float(np.max(mesh.extents))
    tracer = OptixRayTracer(vertices / scale, faces)
    device = tracer.device
    epsilon = float(4 * np.finfo(np.float32).eps)
    query = MaterialQuery(tracer, epsilon)
    exit_filter = WallExitFilter(mesh, device, torch) if require_opposing_surfaces else None
    thickness = np.full(len(faces), np.nan, dtype=np.float32)
    classes = np.full(len(faces), 2, dtype=np.uint8)
    if threshold <= 4 * epsilon * scale:
        return thickness, classes
    last_log = time.monotonic()
    for start in range(0, len(faces), batch_size):
        triangles = vertices[faces[start:start + batch_size]] / scale
        vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(vectors, axis=1)
        usable = np.isfinite(lengths) & (lengths > 0)
        ids = np.arange(start, start + len(triangles))[usable]
        if not len(ids):
            continue
        points = torch.tensor(triangles[usable].mean(axis=1), device=device, dtype=torch.float32)
        normals = torch.tensor(vectors[usable] / lengths[usable, None], device=device, dtype=torch.float32)
        inward, classified, _ = _material_side(query, points, normals, epsilon)
        directions = thickness_directions(inward, torch, max_normal_angle_deg)
        ray_count = directions.shape[1]
        origins = points[:, None] + epsilon * directions
        source = torch.tensor(ids, device=device, dtype=torch.int64)[:, None].expand(-1, ray_count).reshape(-1)
        hits, distances = tracer._trace_trusted(origins.reshape(-1, 3), directions.reshape(-1, 3),
                                       epsilon / 8, 8., source)
        if exit_filter is not None:
            values, exits = exit_filter.normal_supported(
                hits, distances, directions, inward, epsilon, threshold / scale)
            values = values * scale
        else:
            minimum = distances.reshape(-1, ray_count).min(dim=1).values
            exits = (hits.reshape(-1, ray_count) >= 0).all(dim=1)
            values = (minimum + epsilon) * scale
        valid = classified & exits & torch.isfinite(values)
        valid_cpu = valid.cpu().numpy()
        values_cpu = values.cpu().numpy()
        thickness[ids[valid_cpu]] = values_cpu[valid_cpu]
        classes[ids[valid_cpu]] = (values_cpu[valid_cpu] < threshold).astype(np.uint8)
        if time.monotonic() - last_log >= 15:
            _LOG.info('Measured %d/%d face centroids', start + len(triangles), len(faces))
            last_log = time.monotonic()
    return thickness, classes


def export_colored_ply(mesh, thickness, classes, path, threshold, chunk_size=262144,
                       max_normal_angle_deg=DEFAULT_MAX_NORMAL_ANGLE_DEG, require_opposing_surfaces=True):
    """Binary PLY face colors plus thickness/class attributes; retain connectivity."""
    path = Path(path)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if len(vertices) >= 2**31 or len(faces) >= 2**31:
        raise ValueError('PLY face/vertex IDs exceed int32')
    if thickness.shape != (len(faces),) or classes.shape != (len(faces),) or not np.isin(classes, [0, 1, 2]).all():
        raise ValueError('one thickness and class per input face is required')
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'comment threshold_mm {threshold:.17g}\n'
        'comment face_class 0=gray_measured_ge_threshold 1=red_measured_lt_threshold 2=yellow_unknown\n'
        f'comment max_normal_angle_deg {max_normal_angle_deg:g}\n'
        f'comment opposing_surface_filter {int(require_opposing_surfaces)} hit_incidence_and_normal_alignment_max_deg 45\n'
        f'comment thickness_estimator {"normal_supported_majority" if require_opposing_surfaces else "minimum_chord"}\n'
        'comment thickness_mm is NaN for unknown; one centroid; 32 candidate rays before angle filtering\n'
        f'element vertex {len(vertices)}\nproperty double x\nproperty double y\nproperty double z\n'
        f'element face {len(faces)}\nproperty list uchar int vertex_indices\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n'
        'property float thickness_mm\nproperty uchar thickness_class\nproperty int source_face_id\nend_header\n')
    dtype = np.dtype([('count', 'u1'), ('indices', '<i4', (3,)),
                      ('rgba', 'u1', (4,)), ('thickness_mm', '<f4'),
                      ('thickness_class', 'u1'), ('source_face_id', '<i4')])
    with path.open('wb') as f:
        f.write(header.encode('ascii'))
        for start in range(0, len(vertices), chunk_size):
            f.write(np.asarray(vertices[start:start + chunk_size], dtype='<f8').tobytes())
        for start in range(0, len(faces), chunk_size):
            stop = min(start + chunk_size, len(faces))
            block = np.empty(stop - start, dtype=dtype)
            block['count'] = 3
            block['indices'] = faces[start:stop]
            block['rgba'] = COLORS[classes[start:stop]]
            block['thickness_mm'] = thickness[start:stop]
            block['thickness_class'] = classes[start:stop]
            block['source_face_id'] = np.arange(start, stop)
            f.write(block.tobytes())


def validate_export(mesh, thickness, classes, path):
    """Reload the exported geometry, colors and scalar attributes independently."""
    reloaded = trimesh.load(path, process=False)
    if reloaded.vertices.shape != mesh.vertices.shape or reloaded.faces.shape != mesh.faces.shape:
        raise AssertionError('export changed mesh counts')
    for start in range(0, len(mesh.vertices), 262144):
        np.testing.assert_array_equal(reloaded.vertices[start:start + 262144], mesh.vertices[start:start + 262144])
    raw = reloaded.metadata['_ply_raw']['face']['data']
    for start in range(0, len(mesh.faces), 262144):
        sl = slice(start, start + 262144)
        np.testing.assert_array_equal(reloaded.faces[sl], mesh.faces[sl])
        np.testing.assert_array_equal(reloaded.visual.face_colors[sl], COLORS[classes[sl]])
        np.testing.assert_array_equal(np.asarray(raw['thickness_class']).reshape(-1)[sl], classes[sl])
        np.testing.assert_allclose(np.asarray(raw['thickness_mm']).reshape(-1)[sl], thickness[sl], rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(np.asarray(raw['source_face_id']).reshape(-1)[sl],
                                      np.arange(start, min(start + 262144, len(mesh.faces))))
    return True


def summary(mesh, thickness, classes):
    areas = face_areas(mesh)
    total = float(areas.sum())
    counts = np.bincount(classes, minlength=3)
    area_by_class = np.bincount(classes, weights=areas, minlength=3)
    return dict(num_faces=len(mesh.faces), num_vertices=len(mesh.vertices), surface_area_mm2=total,
                face_counts=dict(zip(['normal', 'thin', 'unknown'], map(int, counts))),
                area_ratios=dict(zip(['normal', 'thin', 'unknown'], map(float, area_by_class / total))))
