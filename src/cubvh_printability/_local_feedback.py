"""Source-point upper bounds from visible damage of an extracted local field.

Cuts are necessary constraints at sampled envelope owners, not a certificate
for other pixels or owners. Reconstruct and review after every field change.
"""
import logging

import numpy as np

from ._local_field_solver import validate_point_caps
from ._printability_fidelity import VIEW_DIRECTIONS
from .render_review import compare_views, displaced_centers, render_view

_LOG = logging.getLogger(__name__)


def render_feedback_progress(previous, current):
    """Allow extra corrections only for improving renders or newly found regions."""
    def scores(result):
        limit = result['max_allowed_depth_p95_mm']
        whole_iou = result.get('min_allowed_silhouette_iou', .995)
        values = {}
        for view in result['views']:
            required = whole_iou if view['region'] == 'whole' else .95
            depth, normal = view['depth_p95_mm'], view['normal_angle_p95_deg']
            if depth is None or normal is None:
                return None
            key = (view['region'], view['view'], tuple(view.get('center', ())),
                   view.get('width_mm'))
            values[key] = max(0., (required-view['silhouette_iou'])/max(1-required, 1e-9),
                              depth/limit-1., normal/30.-1.)
        return values

    if previous is None:
        return False
    before, after = scores(previous), scores(current)
    if not before or not after or not before.keys() <= after.keys():
        return False
    if after.keys() != before.keys():
        return True  # Newly discovered damage deserves its own correction.
    old, new = sum(before.values()), sum(after.values())
    # Require measurable improvement on the identical retained camera set.
    return new < old-max(1e-6, .001*old)


class FieldFidelityGuard:
    """Stream whole and cumulative local cameras, exposing them for feedback.

    Rendering is evaluated at the same resolution used for pixel constraints.
    Reference images are rendered one at a time instead of retaining gigabytes
    of full-resolution depth and normal arrays across many local regions.
    """

    def __init__(self, original, size, centers=(), resolution=1024):
        self.original, self.size = original, size
        self.resolution = resolution
        self.centers = [np.asarray(c, dtype=float) for c in centers]

    def evaluate(self, candidate, max_depth_ratio=.005, min_silhouette_iou=.995):
        from .optix import OptixRayTracer
        for center in displaced_centers(self.original, candidate, self.size):
            if all(np.linalg.norm(np.asarray(center)-c) >= .02*self.size for c in self.centers):
                self.centers.append(np.asarray(center))
        original = OptixRayTracer(self.original.vertices, self.original.faces)
        repaired = OptixRayTracer(candidate.vertices, candidate.faces)
        cameras = [('whole', np.asarray(self.original.bounds).mean(axis=0), self.size*1.3)]
        cameras += [(f'local_{i:02d}', c, max(4., .12*self.size))
                    for i, c in enumerate(self.centers)]
        depth_limit = min(.25, self.size*max_depth_ratio)
        whole_iou = max(.995, min_silhouette_iou)
        views = []
        for region, center, width in cameras:
            for index, direction in enumerate(VIEW_DIRECTIONS):
                before = render_view(self.original, original, center, width, direction, self.resolution)
                after = render_view(candidate, repaired, center, width, direction, self.resolution)
                metric = compare_views(before, after)
                passed = (metric['silhouette_iou'] >= (whole_iou if region == 'whole' else .95)
                          and metric['depth_p95_mm'] is not None
                          and metric['depth_p95_mm'] <= depth_limit
                          and metric['normal_angle_p95_deg'] is not None
                          and metric['normal_angle_p95_deg'] <= 30.)
                views.append(dict(region=region, view=index, center=center.tolist(),
                                  width_mm=width, passed=bool(passed), **metric))
            region_views = views[-len(VIEW_DIRECTIONS):]
            _LOG.info('Render region %s: failed_views=%s min_iou=%.6f '
                      'max_depth_p95_mm=%s max_normal_p95_deg=%s', region,
                      [v['view'] for v in region_views if not v['passed']],
                      min(v['silhouette_iou'] for v in region_views),
                      max((v['depth_p95_mm'] for v in region_views
                           if v['depth_p95_mm'] is not None), default=None),
                      max((v['normal_angle_p95_deg'] for v in region_views
                           if v['normal_angle_p95_deg'] is not None), default=None))
        whole = [v for v in views if v['region'] == 'whole']
        local = [v for v in views if v['region'] != 'whole']
        return {'passed': all(v['passed'] for v in views),
                'local_passed': all(v['passed'] for v in local),
                'min_silhouette_iou': min(v['silhouette_iou'] for v in whole),
                'max_depth_p95_mm': max((v['depth_p95_mm'] or 0) for v in views),
                'max_allowed_depth_p95_mm': depth_limit,
                'min_allowed_silhouette_iou': whole_iou,
                'max_allowed_normal_p95_deg': 30.,
                'resolution': self.resolution, 'views': views, 'local_views': local,
                'method': 'whole and cumulative local full-triangle OptiX renders'}


def ray_caps(points, directions, starts, centers):
    """Distance from source centers to half-rays at/above the start depth."""
    depth = np.sum((centers-points)*directions, axis=-1)
    projected = points+np.maximum(starts, depth)[..., None]*directions
    return np.linalg.norm(centers-projected, axis=-1), projected


def choose_cuts(cost, count):
    """Choose the cheapest positive witness corrections; report infeasibility."""
    cost = np.asarray(cost)
    eligible = np.flatnonzero(cost > 1e-6)
    return eligible[np.argsort(cost[eligible], kind='stable')[:max(0, count)]]


def silhouette_corrections(source, candidate, *, whole, min_silhouette_iou=.995,
                           only_failing=True):
    """Count silhouette corrections using the same gate as render acceptance."""
    difference = int(np.count_nonzero(source ^ candidate))
    union = int(np.count_nonzero(source | candidate))
    gate = 1-max(.995, min_silhouette_iou) if whole else .05
    # Keep a proposal margin below the final allowed mismatch. At IoU=1,
    # every extra/missing foreground pixel needs attention.
    budget = min(.003, .6*gate) if whole else .04
    if only_failing and difference <= gate*union:
        return 0
    return max(0, difference-int(budget*union))


def generate_pixel_caps(mesh, candidate, values, views, *, size=50.,
                        resolution=1024, previous=None, normal_budget=.04,
                        depth_margin=.01, depth_only_margin=.24,
                        depth_limit=.25, only_failing=True, min_silhouette_iou=.995):
    """Return cumulative affine radius caps and per-camera correction counts.

    All lengths use the input mesh frame (millimetres). Unknown or invalid
    radius arrays are rejected before loading GPU dependencies. `previous`
    refers to the same source mesh's face ordering and is always retained.
    """
    values = np.asarray(values, dtype=float)
    if (values.shape != (len(mesh.vertices),) or not np.isfinite(values).all()
            or np.any(values < 0)):
        raise ValueError('radii must be finite nonnegative values per source vertex')
    if (not np.isfinite(size) or size <= 0 or not np.isfinite(depth_limit)
            or depth_limit <= 0 or not 0 <= normal_budget < .05
            or not 0 < depth_margin < depth_limit
            or not 0 < depth_only_margin < depth_limit):
        raise ValueError('invalid scale, proposal budget or depth margin')
    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution < 32:
        raise ValueError('resolution must be an integer at least 32')
    if not np.isfinite(min_silhouette_iou) or not 0 <= min_silhouette_iou <= 1:
        raise ValueError('min_silhouette_iou must be finite and in [0, 1]')
    if previous is not None:
        validate_point_caps(previous, len(mesh.faces))
    import torch
    from cubvh import cuBVH
    from .optix import OptixRayTracer
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    bvh = cuBVH(vertices/size, faces)
    bvh.set_offset_radii(values[faces]/size)
    original_tracer = OptixRayTracer(vertices, faces)
    candidate_tracer = OptixRayTracer(candidate.vertices, candidate.faces)
    rows, weights, limits, records = [], [], [], []
    if previous is not None:
        ids, bary, caps = validate_point_caps(previous, len(faces))
        rows.append(ids)
        weights.append(bary)
        limits.append(caps)
    for view in views:
        z = np.array(VIEW_DIRECTIONS[view['view']], dtype=float)
        z /= np.linalg.norm(z)
        center, width = np.asarray(view['center']), view['width_mm']
        before = render_view(mesh, original_tracer, center, width, z, resolution, return_face_ids=True)
        after = render_view(candidate, candidate_tracer, center, width, z, resolution, return_face_ids=True)
        ma, mb = before[3] >= 0, after[3] >= 0
        common, union = ma & mb, ma | mb
        angles = np.degrees(np.arccos(np.clip((before[2]*after[2]).sum(axis=-1), -1, 1)))
        normal_bad = common & (angles > 30.)
        depth_bad = np.zeros_like(common)
        depth_bad[common] = np.abs(after[1][common]-before[1][common]) > depth_limit
        silhouette_bad = mb & ~ma
        n_needed = max(0, int(normal_bad.sum())-int(normal_budget*common.sum()))
        d_needed = max(0, int(depth_bad.sum())-int(.04*common.sum()))
        s_needed = silhouette_corrections(ma, mb, whole=view['region'] == 'whole',
            min_silhouette_iou=min_silhouette_iou, only_failing=only_failing)
        if only_failing:
            if normal_bad.sum() <= .05*common.sum():
                n_needed = 0
            if depth_bad.sum() <= .05*common.sum():
                d_needed = 0
        if not n_needed and not s_needed and not d_needed:
            continue
        tested = (normal_bad if n_needed else np.zeros_like(mb)) | (
            silhouette_bad if s_needed else np.zeros_like(mb)) | (
            depth_bad if d_needed else np.zeros_like(mb))
        up = np.array([0., 1., 0.]) if abs(z[1]) < .95 else np.array([0., 0., 1.])
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        axis = (np.arange(resolution)+.5)/resolution-.5
        xx, yy = np.meshgrid(axis, axis[::-1])
        origins = center+width*(xx[tested, None]*x+yy[tested, None]*y)
        # Normal failures require restoring the visible source surface. A
        # depth-only failure permits a larger displacement below depth_limit.
        # These proposal margins never change the final acceptance gate.
        restore_normal = normal_bad[tested] & bool(n_needed)
        margin = np.where(restore_normal, depth_margin, depth_only_margin)
        starts = np.where(ma[tested], before[1][tested]+margin, -2*size)
        point = origins+after[1][tested, None]*z
        # Alternating minimization finds witnesses along the forbidden ray.
        # Every witness remains a valid necessary inequality even when this
        # search does not find the globally most violated source point.
        best_cost = np.zeros(len(point))
        best_id = np.zeros(len(point), dtype=np.int64)
        best_bary = np.zeros((len(point), 3))
        best_cap = np.zeros(len(point))
        for _ in range(4):
            _, ids, bary = bvh.offset_distance(torch.as_tensor(point/size, device='cuda', dtype=torch.float32))
            ids, bary = ids.cpu().numpy(), bary.cpu().numpy()
            if np.any(ids < 0):
                raise RuntimeError('query returned no envelope owner')
            q = (vertices[faces[ids]]*bary[..., None]).sum(axis=1)
            cap, point = ray_caps(origins, z, starts, q)
            violation = (values[faces[ids]]*bary).sum(axis=1)-cap
            improved = violation > best_cost
            best_cost[improved], best_id[improved] = violation[improved], ids[improved]
            best_bary[improved], best_cap[improved] = bary[improved], cap[improved]
        selected = np.zeros(len(point), dtype=bool)
        counts = {}
        for name, subset, needed in [('normal', normal_bad[tested], n_needed),
                                     ('depth', depth_bad[tested], d_needed),
                                     ('silhouette', silhouette_bad[tested], s_needed)]:
            indices = np.flatnonzero(subset)
            chosen = indices[choose_cuts(best_cost[indices], needed)]
            selected[chosen] = True
            counts[name+'_needed'] = int(needed)
            counts[name+'_cut_pixels'] = len(chosen)
        rows.append(best_id[selected])
        weights.append(best_bary[selected])
        limits.append(best_cap[selected])
        records.append({'region':view['region'], 'view':view['view'], **counts,
                        'normal_bad_fraction':float(normal_bad.sum()/max(1,common.sum())),
                        'max_witness_correction_mm':float(best_cost[selected].max()) if selected.any() else 0.})
        _LOG.info("Pixel cuts: %s", records[-1])
    constraints = {
        'face_ids': np.concatenate(rows) if rows else np.empty(0, dtype=np.int64),
        'bary': np.concatenate(weights) if weights else np.empty((0, 3)),
        'caps': np.concatenate(limits) if limits else np.empty(0),
    }
    return constraints, records
