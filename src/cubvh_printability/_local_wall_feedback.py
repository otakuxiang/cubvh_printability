"""Local growth proposals from measured reconstructed walls and envelope owners."""
import numpy as np

from ._local_field_solver import validate_wall_witnesses


def linearize_wall_witnesses(mesh, owners, minimum):
    """Linearize small wall deficits at fixed, nongrazing envelope owners.

    The capacity envelope is only a necessary upper bound. These targets do
    not certify thickness after reconstruction or preserve owner identity.
    Lengths use the mesh frame; proposal margins scale with the minimum.
    """
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError('minimum thickness must be finite and positive')
    ids = np.asarray(owners['face_ids'])
    bary = np.asarray(owners['bary'], dtype=float)
    positions = np.asarray(owners['positions'], dtype=float)
    thickness = np.asarray(owners['thickness'], dtype=float)
    radius, bounds, distance = (np.asarray(owners[k], dtype=float)
                                for k in ('radius', 'bounds', 'distance'))
    area = np.asarray(owners['sample_area'], dtype=float)
    count = len(thickness) if thickness.ndim == 1 else -1
    if (count < 0 or ids.shape != (count, 2) or ids.dtype.kind not in 'iu'
            or np.any((ids < 0) | (ids >= len(mesh.faces)))
            or bary.shape != (count, 2, 3) or positions.shape != (count, 2, 3)
            or any(a.shape != (count, 2) for a in (radius, bounds, distance))
            or area.shape != (count,) or np.any(area <= 0)
            or any(not np.isfinite(a).all() for a in
                   (bary, positions, thickness, radius, bounds, distance, area))
            or np.any(thickness <= 0)):
        raise ValueError('invalid measured wall owners')
    # Native closest-point queries can yield slightly negative barycentrics
    # on tiny triangles. Exclude those rows instead of moving their owners by
    # clamping or letting one invalid row discard every valid measured wall.
    valid_owner = ((bary >= 0).all(axis=(1, 2))
                   & np.isclose(bary.sum(axis=2), 1., atol=1e-6, rtol=0).all(axis=1)
                   & (radius >= 0).all(axis=1) & (bounds >= 0).all(axis=1))
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[ids]]
    q = (triangles * bary[..., None]).sum(axis=2)
    normal = positions - q
    length = np.linalg.norm(normal, axis=2)
    normal /= np.maximum(length[..., None], 1e-12)
    zero = radius < 2e-5 * minimum
    normal[zero] = np.asarray(mesh.face_normals)[ids[zero]]
    ray = positions[:, 1] - positions[:, 0]
    ray_length = np.linalg.norm(ray, axis=1)
    ray /= np.maximum(ray_length[:, None], 1e-12)
    incidence = np.abs((normal * ray[:, None]).sum(axis=2))
    factors = 1 / np.clip(incidence, .5, 1.)
    gap = minimum - thickness
    margin, maximum_step, owner_tolerance = .01 * minimum, .1 * minimum, .04 * minimum
    targets = (radius * factors).sum(axis=1) + gap + margin
    capacity = (bounds * factors).sum(axis=1)
    selected = (valid_owner & (incidence >= .5).all(axis=1) & (ray_length > 1e-12)
                & (gap <= maximum_step) & (gap > 0) & (capacity >= targets)
                & (np.abs(distance) <= owner_tolerance).all(axis=1))
    witnesses = {'face_ids': ids[selected], 'bary': bary[selected],
                 'targets': targets[selected], 'areas': area[selected],
                 'factors': factors[selected]}
    validate_wall_witnesses(witnesses, len(mesh.faces))
    report = {'witness_count': int(selected.sum()), 'known_thin_samples': count,
              'invalid_owner_samples': int((~valid_owner).sum()),
              'target_margin_mm': margin, 'max_linearized_step_mm': maximum_step,
              'owner_tolerance_mm': owner_tolerance, 'linearization_only': True}
    return witnesses, report


def measured_wall_witnesses(source, candidate, values, envelope, minimum, *,
                            size=50., sample_count=200_000, seed=20261113):
    """Sample a candidate and attribute its remaining thin spans to source radii.

    Feedback uses a separate draw from default acceptance. Unreliable sampled
    geometry produces no targets. Callers must still remeasure the solved and
    reconstructed mesh; this function never decides acceptance.
    """
    values, envelope = np.asarray(values, dtype=float), np.asarray(envelope, dtype=float)
    if (values.shape != (len(source.vertices),) or envelope.shape != values.shape
            or not np.isfinite(values).all() or not np.isfinite(envelope).all()
            or np.any(values < 0) or np.any(envelope < 0)
            or not np.isfinite(size) or size <= 0
            or not np.isfinite(minimum) or minimum <= 0
            or not isinstance(sample_count, int) or isinstance(sample_count, bool)
            or sample_count <= 0):
        raise ValueError('invalid measured wall feedback inputs')
    from ._printability_thickness import measure
    measurement = measure(candidate, minimum, sample_count=sample_count, seed=seed,
                          return_samples=True, diagnostics=True)
    samples = measurement.pop('samples')
    eligible = (samples['valid'] & np.isfinite(samples['thickness'])
                & (samples['thickness'] > 0) & (samples['thickness'] < minimum))
    report = {'seed': seed, 'samples': sample_count, 'measurement': measurement,
              'known_thin_area_ratio': float(eligible.sum() / sample_count),
              'linearization_only': True}
    empty = {'face_ids': np.empty((0, 2), dtype=np.int64), 'bary': np.empty((0, 2, 3)),
             'targets': np.empty(0), 'areas': np.empty(0), 'factors': np.empty((0, 2))}
    if not measurement['reliable'] or not eligible.any():
        report.update(witness_count=0, sampled_area_ratio=0.,
                      reason='unreliable_measurement' if not measurement['reliable'] else 'no_known_thin_samples')
        return empty, report
    import torch
    from cubvh import cuBVH
    ids = samples['face_ids'][eligible]
    bary = samples['bary'][eligible]
    points = (np.asarray(candidate.vertices)[np.asarray(candidate.faces)[ids]]
              * bary[..., None]).sum(axis=1)
    thickness = samples['thickness'][eligible]
    positions = np.stack((points, points + samples['inward'][eligible] * thickness[:, None]), axis=1)
    vertices, faces = np.asarray(source.vertices), np.asarray(source.faces)
    bvh = cuBVH(vertices / size, faces)
    bvh.set_offset_radii(values[faces] / size)
    distance, owners, bary = bvh.offset_distance(torch.as_tensor(
        positions.reshape(-1, 3) / size, dtype=torch.float32, device='cuda'))
    owners = owners.cpu().numpy().reshape(-1, 2)
    bary = bary.cpu().numpy().reshape(-1, 2, 3)
    distance = distance.cpu().numpy().reshape(-1, 2) * size
    if np.any((owners < 0) | (owners >= len(faces))):
        raise RuntimeError('measured wall query returned no envelope owner')
    data = {'face_ids': owners, 'bary': bary, 'positions': positions,
            'radius': (values[faces[owners]] * bary).sum(axis=2),
            'bounds': (envelope[faces[owners]] * bary).sum(axis=2),
            'distance': distance, 'thickness': thickness,
            'sample_area': np.full(len(owners), candidate.area / sample_count)}
    witnesses, proposal = linearize_wall_witnesses(source, data, minimum)
    report.update(proposal, sampled_area_ratio=len(witnesses['targets']) / sample_count)
    return witnesses, report
