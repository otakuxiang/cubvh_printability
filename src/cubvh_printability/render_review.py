"""Fixed-camera geometry renders and close-ups for comparing repair candidates."""
import json
from pathlib import Path

import numpy as np

from ._printability_fidelity import VIEW_DIRECTIONS


def render_view(mesh, tracer, center, width, direction, resolution=512, *, return_face_ids=False):
    """Ray-traced clay shading, depth and normals, with a shared world frame."""
    z = np.asarray(direction, dtype=float)
    z /= np.linalg.norm(z)
    up = np.array([0., 1., 0.]) if abs(z[1]) < .95 else np.array([0., 0., 1.])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    axis = (np.arange(resolution) + .5) / resolution - .5
    xx, yy = np.meshgrid(axis, axis[::-1])
    distance = max(float(np.max(mesh.extents)) * 2, width * 2)
    origins = np.asarray(center) + width * (xx[..., None] * x + yy[..., None] * y) + distance * z
    ids, depth = tracer.trace(origins.reshape(-1, 3), np.broadcast_to(-z, (resolution**2, 3)).copy())
    ids, depth = ids.cpu().numpy(), depth.cpu().numpy()
    mask = ids >= 0
    normal = np.zeros((len(ids), 3))
    normal[mask] = np.asarray(mesh.face_normals)[ids[mask]]
    normal *= np.where((normal @ z) < 0, -1., 1.)[:, None]
    light = z + .5 * y - .4 * x
    light /= np.linalg.norm(light)
    intensity = .25 + .75 * np.maximum(0., normal @ light)
    rgb = np.full((len(ids), 3), 245., dtype=float)
    rgb[mask] = intensity[mask, None] * np.array([185., 192., 205.])
    # Height in the shared world frame, independent of camera distance.
    height = distance - depth
    height[~mask] = np.nan
    result = (rgb.reshape(resolution, resolution, 3).astype(np.uint8),
              height.reshape(resolution, resolution), normal.reshape(resolution, resolution, 3))
    return result + (ids.reshape(resolution, resolution),) if return_face_ids else result


def render_damage_cost(original, candidate, views, resolution=384, size=50.):
    """Map measured render violations onto original faces for diagnostic feedback.

    Candidate hit points also map through the original BVH so a newly expanded
    foreground patch receives blame instead of only its occluded background.
    A zero value means no observed violation, not proof of fidelity elsewhere.
    """
    from cubvh import cuBVH
    from .optix import OptixRayTracer
    import torch
    a = OptixRayTracer(np.asarray(original.vertices), np.asarray(original.faces))
    b = OptixRayTracer(np.asarray(candidate.vertices), np.asarray(candidate.faces))
    original_bvh = cuBVH(np.asarray(original.vertices) / size, np.asarray(original.faces))
    damage = np.zeros(len(original.faces), dtype=np.float64)
    stats = []
    for view in views:
        center, width = np.asarray(view['center']), view['width_mm']
        direction = VIEW_DIRECTIONS[view['view']]
        before = render_view(original, a, center, width, direction, resolution, return_face_ids=True)
        after = render_view(candidate, b, center, width, direction, resolution, return_face_ids=True)
        ma, mb = before[3] >= 0, after[3] >= 0
        common = ma & mb
        error = np.zeros(ma.shape)
        error[ma ^ mb] = 4.
        depth_error = np.abs(before[1][common] - after[1][common]) / .25
        angles = np.degrees(np.arccos(np.clip(np.sum(
            before[2][common] * after[2][common], axis=1), -1, 1))) / 30.
        error[common] = np.maximum(depth_error, angles)
        # Only violated local tolerances contribute a feedback penalty.
        error = np.where(error > 1., np.minimum(error, 10.), 0.)
        observed = ma & (error > 0)
        np.maximum.at(damage, before[3][observed], error[observed])
        changed = mb & (error > 0)
        if changed.any():
            z = np.asarray(direction, dtype=float)
            z /= np.linalg.norm(z)
            up = np.array([0., 1., 0.]) if abs(z[1]) < .95 else np.array([0., 0., 1.])
            x = np.cross(up, z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            axis = (np.arange(resolution) + .5) / resolution - .5
            xx, yy = np.meshgrid(axis, axis[::-1])
            points = (center + width * (xx[changed, None] * x + yy[changed, None] * y)
                      + after[1][changed, None] * z)
            _, source_ids, _ = original_bvh.unsigned_distance(torch.as_tensor(
                points / size, device='cuda', dtype=torch.float32), return_uvw=True)
            np.maximum.at(damage, source_ids.cpu().numpy(), error[changed])
        stats.append({'region': view['region'], 'view': view['view'],
                      'violating_pixels': int(np.count_nonzero(error)),
                      'foreground_pixels': int(np.count_nonzero(ma | mb))})
    # Compact intrinsic spreading covers neighboring triangles missed by pixels.
    from ._local_sdf import smooth_vertex_offsets
    vertices = smooth_vertex_offsets(original, damage, .2)
    return vertices[np.asarray(original.faces)].max(axis=1), stats


def compare_views(before, after):
    a, b = before[1], after[1]
    ma, mb = np.isfinite(a), np.isfinite(b)
    common, union = ma & mb, ma | mb
    depth = np.abs(a[common] - b[common])
    angle = np.degrees(np.arccos(np.clip(np.sum(before[2][common] * after[2][common], axis=1), -1, 1)))
    return {'silhouette_iou': float(common.sum() / max(1, union.sum())),
            'depth_p95_mm': float(np.percentile(depth, 95)) if len(depth) else None,
            'depth_max_mm': float(depth.max()) if len(depth) else None,
            'normal_angle_p95_deg': float(np.percentile(angle, 95)) if len(angle) else None}


def review_pair(original, candidate, directory, centers=(), size=50., resolution=512):
    """Save original/candidate/difference strips for ten whole and local views."""
    from PIL import Image, ImageDraw
    from .optix import OptixRayTracer
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    a = OptixRayTracer(np.asarray(original.vertices), np.asarray(original.faces))
    b = OptixRayTracer(np.asarray(candidate.vertices), np.asarray(candidate.faces))
    cameras = [('whole', np.zeros(3), size * 1.3)]
    cameras += [(f'local_{i:02d}', c, max(4., size * .12)) for i, c in enumerate(centers)]
    metrics = []
    for name, center, width in cameras:
        for j, direction in enumerate(VIEW_DIRECTIONS):
            before = render_view(original, a, center, width, direction, resolution)
            after = render_view(candidate, b, center, width, direction, resolution)
            difference = np.minimum(np.abs(before[0].astype(int) - after[0].astype(int)) * 4, 255).astype(np.uint8)
            strip = Image.fromarray(np.concatenate((before[0], after[0], difference), axis=1))
            draw = ImageDraw.Draw(strip)
            draw.text((8, 8), 'Original', fill=(10, 10, 10))
            draw.text((resolution + 8, 8), 'Candidate', fill=(10, 10, 10))
            draw.text((resolution * 2 + 8, 8), 'Difference x4', fill=(255, 255, 255))
            strip.save(directory / f'{name}_view_{j:02d}.png')
            metrics.append(dict(region=name, view=j, center=np.asarray(center).tolist(),
                                width_mm=width, **compare_views(before, after)))
    (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics


def select_review_centers(mesh, offsets, size=50., count=3):
    """Choose separated high-area displacement regions for deterministic close-ups."""
    ids = np.flatnonzero(np.asarray(offsets) > 0)
    if not len(ids):
        return []
    score = np.asarray(mesh.area_faces)[ids] * np.asarray(offsets)[ids]
    order = ids[np.argsort(-score, kind='stable')]
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    centers = []
    for i in order:
        center = vertices[faces[i]].mean(axis=0)
        if all(np.linalg.norm(center - c) >= size * .12 for c in centers):
            centers.append(center)
        if len(centers) >= count:
            break
    return centers


def displaced_centers(original, candidate, size, count=12, sample_count=100_000):
    """Use deterministic area-uniform probes across all newly changed regions.

    Three probes can miss a second ornament or an attachment junction. Keep a
    larger, spatially separated set so local fidelity feedback cannot accept
    one repaired patch while another displaced patch remains unobserved.
    """
    from cubvh import cuBVH
    import torch
    if (not isinstance(count, int) or isinstance(count, bool) or count < 1
            or not isinstance(sample_count, int) or isinstance(sample_count, bool)
            or sample_count < count):
        raise ValueError('count and sample_count must be positive integers with sample_count >= count')
    rng = np.random.default_rng(20261112)
    ids = rng.choice(len(original.faces), sample_count,
                     p=original.area_faces / original.area)
    uv = rng.random((sample_count, 2))
    root = np.sqrt(uv[:, 0])
    bary = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    points = np.einsum('ni,nij->nj', bary, original.vertices[original.faces[ids]])
    bvh = cuBVH(candidate.vertices / size, candidate.faces)
    distances = bvh.unsigned_distance(torch.as_tensor(points / size,
        dtype=torch.float32, device='cuda'))[0].cpu().numpy()
    centers = []
    for i in np.argsort(-distances, kind='stable'):
        if all(np.linalg.norm(points[i] - c) >= .08 * size for c in centers):
            centers.append(points[i])
        if len(centers) == count:
            break
    return centers

