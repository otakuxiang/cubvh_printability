"""Area-uniform material thickness estimates using only OptiX closest hits."""
import numpy as np
import logging
import time
import math

from .optix import OptixRayTracer
from ._printability_geometry import face_areas


def _component_geometry(mesh, chunk_size=262144):
    """Closed edge-connected shells, signed volumes and representative faces.

    Scalar edge sorting and a three-neighbor CSR avoid Trimesh's full edge,
    adjacency, triangle and submesh caches. Point-touching shells stay separate.
    """
    cached = mesh._cache['printability_components']
    if cached is not None:
        return cached
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    faces, vertices = np.asarray(mesh.faces), np.asarray(mesh.vertices)
    nf, nv = len(faces), len(vertices)
    if not nf or nv * nv > np.iinfo(np.int64).max:
        raise ValueError('component analysis requires a nonempty indexable mesh')
    keys = np.empty(nf * 3, dtype=np.int64)
    for start in range(0, nf, chunk_size):
        part = faces[start:start + chunk_size]
        for edge, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
            u, v = part[:, a], part[:, b]
            keys[start * 3 + edge:(start + len(part)) * 3:3] = np.minimum(u, v) * nv + np.maximum(u, v)
    order = np.argsort(keys)
    if len(order) % 2:
        raise ValueError('component analysis requires closed manifold edges')
    dtype = np.int32 if nf * 3 < np.iinfo(np.int32).max else np.int64
    neighbors = np.empty(nf * 3, dtype=dtype)
    for start in range(0, len(order), chunk_size * 2):
        pairs = order[start:start + chunk_size * 2].reshape(-1, 2)
        left, right = keys[pairs[:, 0]], keys[pairs[:, 1]]
        if (np.any(left != right) or np.any(left[1:] == left[:-1])
                or (start and left[0] == keys[order[start - 1]])):
            raise ValueError('component analysis requires closed manifold edges')
        neighbors[pairs[:, 0]] = pairs[:, 1] // 3
        neighbors[pairs[:, 1]] = pairs[:, 0] // 3
    del keys, order
    graph = csr_matrix((np.ones(nf * 3, dtype=np.bool_), neighbors,
                        np.arange(0, nf * 3 + 1, 3, dtype=dtype)), shape=(nf, nf))
    # The graph is symmetric; directed=True avoids constructing its transpose.
    count, labels = connected_components(graph, directed=True, connection='strong')
    del graph, neighbors
    volumes = np.zeros(count, dtype=np.float64)
    areas = face_areas(mesh)
    maxima = np.full(count, -np.inf)
    np.maximum.at(maxima, labels, areas)
    representatives = np.full(count, nf, dtype=np.int64)
    # Translation reduces cancellation for shells far from the origin.
    reference = vertices.mean(axis=0)
    for start in range(0, nf, chunk_size):
        stop = min(start + chunk_size, nf)
        component = labels[start:stop]
        triangles = vertices[faces[start:stop]] - reference
        signed = np.einsum('ij,ij->i', triangles[:, 0],
                           np.cross(triangles[:, 1], triangles[:, 2])) / 6
        volumes += np.bincount(component, weights=signed, minlength=count)
        selected = areas[start:stop] == maxima[component]
        np.minimum.at(representatives, component[selected], np.arange(start, stop)[selected])
    result = labels, volumes, representatives
    mesh._cache['printability_components'] = result
    return result


class MaterialQuery:
    def __init__(self, tracer, epsilon):
        self.tracer = tracer
        self.torch = tracer._torch
        self.epsilon = epsilon
        directions = []
        for vector in [(1., .37139, .52917), (-.21731, 1., .41327), (.31973, -.61129, 1.)]:
            direction = self.torch.tensor(vector, device=tracer.device, dtype=self.torch.float32)
            directions.append(direction / self.torch.linalg.vector_norm(direction))
        self.step_directions = self.torch.stack(directions)
        self.trace_directions = (self.step_directions / self.torch.linalg.vector_norm(
            self.step_directions, dim=1, keepdim=True)).contiguous()

    def contains(self, points):
        """Same three-direction parity test, stepped entirely inside OptiX.

        Closest hits, source exclusion, float32 stepping and the 256-crossing
        limit match _contains_reference. No any-hit parity shortcuts are used.
        """
        decisions = self.tracer._impl.contains_closest(
            points.contiguous(), self.trace_directions, self.step_directions, self.epsilon)
        known = (decisions < 2).all(dim=1) & (decisions == decisions[:, :1]).all(dim=1)
        return (decisions[:, 0] & 1) == 1, known

    def _contains_reference(self, points):
        """Odd/even crossing test in three oblique directions; disagreement is unknown.

        A 256-crossing cap bounds pathological layered meshes. Exhaustion is
        unknown, never silently outside. Repeated nearest-hit queries preserve
        both distant components and nested cavities, independent of winding.
        """
        torch = self.torch
        decisions, knowns = [], []
        for vector in [(1., .37139, .52917), (-.21731, 1., .41327), (.31973, -.61129, 1.)]:
            direction = torch.tensor(vector, device=points.device, dtype=torch.float32)
            direction /= torch.linalg.vector_norm(direction)
            active = torch.arange(len(points), device=points.device)
            origins = points.clone()
            source = torch.full((len(points),), -1, device=points.device, dtype=torch.int64)
            parity = torch.zeros(len(points), device=points.device, dtype=torch.bool)
            known = torch.zeros_like(parity)
            for _ in range(256):
                if not len(active):
                    break
                faces, distance = self.tracer.trace(origins, direction.expand_as(origins),
                                                    0.0, 8.0, source)
                hit = faces >= 0
                known[active[~hit]] = True
                active = active[hit]
                parity[active] = ~parity[active]
                origins = origins[hit] + (distance[hit, None] + self.epsilon) * direction
                source = faces[hit]
            decisions.append(parity)
            knowns.append(known)
        decisions = torch.stack(decisions)
        known = torch.stack(knowns).all(dim=0) & (decisions == decisions[0]).all(dim=0)
        return decisions[0], known


def _material_side(query, points, normals, epsilon):
    sides, classified = query.contains(query.torch.cat(
        (points - epsilon * normals, points + epsilon * normals)))
    minus, plus = sides.chunk(2)
    km, kp = classified.chunk(2)
    known = km & kp & (minus != plus)
    inward = query.torch.where(minus[:, None], -normals, normals)
    return inward, known, minus


DEFAULT_MAX_NORMAL_ANGLE_DEG = 30.
DEFAULT_THICKNESS_BATCH_SIZE = 65536


def thickness_directions(inward, torch, max_normal_angle_deg=DEFAULT_MAX_NORMAL_ANGLE_DEG):
    """Filter the original 32 ray candidates by angle to the material normal.

    Keep the normal unconditionally. Default 30 degrees retains nine rays
    (normal plus eight spiral samples), excluding near-tangent side-wall hits.
    Filtering, rather than resampling, keeps comparisons to the old 60-degree
    policy reproducible. The three parity rays are only for inside/outside
    classification and do not contribute thickness distances.
    """
    if (not math.isfinite(max_normal_angle_deg)
            or not 0 <= max_normal_angle_deg <= 60):
        raise ValueError('max_normal_angle_deg must be finite and in [0, 60]')
    k = torch.arange(31, device=inward.device, dtype=inward.dtype)
    cos = 1 - .5 * (k + .5) / 31
    keep = cos >= math.cos(math.radians(max_normal_angle_deg))
    k, cos = k[keep], cos[keep]
    sin = torch.sqrt(1 - cos**2)
    angle = k * (np.pi * (3 - np.sqrt(5)))
    axis = torch.zeros_like(inward)
    axis[:, 0] = 1
    axis[torch.abs(inward[:, 0]) > .9] = torch.tensor([0., 1., 0.], device=inward.device, dtype=inward.dtype)
    tangent = torch.linalg.cross(inward, axis)
    tangent /= torch.linalg.vector_norm(tangent, dim=1, keepdim=True).clamp_min(1e-30)
    bitangent = torch.linalg.cross(inward, tangent)
    cone = (inward[:, None] * cos[None, :, None]
            + tangent[:, None] * (sin * torch.cos(angle))[None, :, None]
            + bitangent[:, None] * (sin * torch.sin(angle))[None, :, None])
    return torch.cat((inward[:, None], cone), dim=1)


class WallExitFilter:
    """Reject grazing/side-wall nearest exits; never trace beyond an air exit.

    Source-side classification supplies the material direction. For a nearest
    exit, the hit normal is oriented toward the ray so inverted mesh winding
    does not change this test. Require <=45 degrees to both the ray and the
    source inward normal. This is a local wall-pair heuristic, not proof of a
    valid solid or exact manufacturing thickness.
    """
    def __init__(self, mesh, device, torch):
        self.torch = torch
        # Construct normals on the query device in float64, preserving the
        # CPU geometry precision before the existing float32 ray predicate.
        # Avoid rebuilding millions of face normals on the CPU each round.
        vertices = torch.as_tensor(np.asarray(mesh.vertices), device=device, dtype=torch.float64)
        faces = torch.as_tensor(np.asarray(mesh.faces), device=device, dtype=torch.int64)
        self.normals = torch.empty((len(faces), 3), device=device, dtype=torch.float32)
        for start in range(0, len(faces), 262144):
            tri = vertices[faces[start:start + 262144]]
            normals = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
            length = torch.linalg.vector_norm(normals, dim=1, keepdim=True)
            normals = torch.where(length > 0, normals / length, 0.)
            self.normals[start:start + len(tri)] = normals.float()

    def accepted_exits(self, hits, distances, directions, inward):
        torch = self.torch
        hits = hits.reshape(directions.shape[:2])
        distances = distances.reshape(hits.shape)
        normals = self.normals[hits.clamp_min(0).long()]
        signed_incidence = (normals * directions).sum(dim=2)
        outward = torch.where((signed_incidence >= 0)[..., None], normals, -normals)
        alignment = (outward * inward[:, None]).sum(dim=2)
        cutoff = math.sqrt(.5)
        return ((hits >= 0) & torch.isfinite(distances)
                & (signed_incidence.abs() >= cutoff) & (alignment >= cutoff))

    def reduce(self, hits, distances, directions, inward):
        """Legacy minimum-chord reducer, retained for diagnostic comparisons."""
        hits = hits.reshape(directions.shape[:2])
        distances = distances.reshape(hits.shape)
        accepted = self.accepted_exits(hits, distances, directions, inward)
        minimum = distances.masked_fill(~accepted, float('inf')).min(dim=1).values
        # Keep missing exits conservative even if another direction qualified.
        valid = accepted.any(dim=1) & (hits >= 0).all(dim=1)
        return minimum, valid

    def normal_supported(self, hits, distances, directions, inward, epsilon, threshold):
        """Normal wall span, with a strict majority supporting a thin decision.

        The normal ray is direction zero. An isolated short oblique chord must
        never make a thick normal span red. For a short normal span require a
        qualifying opposing normal hit AND >half of the sampled directions to
        have qualifying short exits projected onto the source normal. Use all
        sampled directions in the denominator, including rejected/missing hits.
        Projection preserves planar slab thickness under oblique rays. A short
        but unsupported normal span is unknown, not a thick-wall assertion.

        This defines local normal thickness, not exact shortest material chord
        or a guarantee of manufacturability. No rays continue beyond an exit.
        """
        torch = self.torch
        hits = hits.reshape(directions.shape[:2])
        distances = distances.reshape(hits.shape)
        accepted = self.accepted_exits(hits, distances, directions, inward)
        lengths = distances + epsilon
        normal_span = lengths[:, 0]
        projection = (directions * inward[:, None]).sum(dim=2).clamp(0., 1.)
        votes = accepted & (lengths * projection < threshold)
        supported = accepted[:, 0] & (votes.sum(dim=1) > directions.shape[1] // 2)
        all_exits = (hits >= 0).all(dim=1) & torch.isfinite(distances).all(dim=1)
        valid = all_exits & ((normal_span >= threshold) | supported)
        return normal_span, valid


def _aggregate_profile_samples(face_count, face_ids, sample_thickness, sample_valid,
                               sample_opposite, sample_count, missing):
    """Exact per-face 10th percentiles without globally sorting extra triplets."""
    face_thickness = np.full(face_count, np.nan, dtype=np.float32)
    valid_samples = sample_valid & np.isfinite(sample_thickness)
    # Extra samples are already grouped as three consecutive samples per
    # missing face. Sort only the area-uniform draw globally, not tens of
    # millions of already-grouped samples.
    uniform_valid = np.flatnonzero(valid_samples[:sample_count])
    order = uniform_valid[np.lexsort((sample_thickness[uniform_valid],
                                     face_ids[uniform_valid]))]
    ids = face_ids[order]
    values = sample_thickness[order]
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1] if len(ids) else np.empty(0, dtype=np.int64)
    ends = np.r_[starts[1:], len(ids)] if len(starts) else np.empty(0, dtype=np.int64)
    rank = .1 * (ends - starts - 1)
    low = np.floor(rank).astype(np.int64)
    fraction = rank - low
    if len(starts):
        face_thickness[ids[starts]] = (values[starts + low] * (1 - fraction)
            + values[np.minimum(starts + low + 1, ends - 1)] * fraction)
    opposite = np.full(face_count, -1, dtype=np.int64)
    if len(starts):
        opposite[ids[starts]] = sample_opposite[order][starts + low]
    if len(missing):
        values = sample_thickness[sample_count:].reshape(-1, 3)
        valid = valid_samples[sample_count:].reshape(-1, 3)
        local_order = np.argsort(np.where(valid, values, np.inf), axis=1, kind='stable')
        sorted_values = np.take_along_axis(values, local_order, axis=1)
        counts = valid.sum(axis=1)
        rows = np.flatnonzero(counts)
        fraction = .1 * (counts[rows] - 1)
        # For 1..3 samples the lower 10th-percentile rank is always zero.
        high = np.minimum(1, counts[rows] - 1)
        face_thickness[missing[rows]] = (sorted_values[rows, 0] * (1 - fraction)
            + sorted_values[rows, high] * fraction)
        exits = sample_opposite[sample_count:].reshape(-1, 3)
        opposite[missing[rows]] = exits[rows, local_order[rows, 0]]
    return face_thickness, opposite, int(valid_samples.sum())


def measure(mesh, min_thickness, *, sample_count=200_000, seed=20261110,
            batch_size=DEFAULT_THICKNESS_BATCH_SIZE,
            max_normal_angle_deg=DEFAULT_MAX_NORMAL_ANGLE_DEG, require_opposing_surfaces=True,
            return_profile=False, diagnostics=False, return_samples=False):
    """Normal wall span with multi-direction support for thin; unknown is thin."""
    import torch
    scale = float(np.max(mesh.extents))
    tracer = OptixRayTracer(np.asarray(mesh.vertices) / scale, np.asarray(mesh.faces))
    device = tracer.device
    # Keep scalar predicates native Python bool: numpy.bool_ promotes a CUDA
    # boolean tensor to float in PyTorch's bitwise operator dispatch.
    epsilon = float(4 * np.finfo(np.float32).eps)
    query = MaterialQuery(tracer, epsilon)
    exit_filter = WallExitFilter(mesh, device, torch) if require_opposing_surfaces else None
    # Material-oriented component volumes handle inverted shells and cavities.
    volume = 0.0
    volume_known = True
    unknown_component_orientations = 0
    orientation_mismatch_samples = 0
    started = time.perf_counter()
    labels, volumes, representatives = _component_geometry(mesh)
    logging.getLogger(__name__).info('Component geometry: %d shells in %.2fs', len(volumes), time.perf_counter() - started)
    expected_minus = np.empty(len(volumes), dtype=bool)
    component_known = np.zeros(len(volumes), dtype=bool)
    vertices, mesh_faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    areas = face_areas(mesh)
    for start in range(0, len(representatives), batch_size):
        triangles = vertices[mesh_faces[representatives[start:start + batch_size]]]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        point = torch.tensor(triangles.mean(axis=1) / scale, device=device, dtype=torch.float32)
        normal = torch.tensor(normals, device=device, dtype=torch.float32)
        _, known, minus = _material_side(query, point, normal, epsilon)
        component_known[start:start + len(triangles)] = known.cpu().numpy()
        expected_minus[start:start + len(triangles)] = minus.cpu().numpy()
    initial_unknown_component_orientations = int((~component_known).sum())
    # A single representative can lie at a numerically ambiguous contact.
    # Retry other large faces of that same closed shell; never discard the
    # shell or infer its sign from winding alone. Conflicting known probes
    # remain unresolved, and all ordinary area-uniform checks still run below.
    for component in np.flatnonzero(~component_known):
        candidates = np.flatnonzero(labels == component)
        candidates = candidates[candidates != representatives[component]]
        candidates = candidates[np.argsort(-areas[candidates], kind='stable')[:8]]
        if not len(candidates):
            continue
        triangles = vertices[mesh_faces[candidates]]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        points = torch.tensor(triangles.mean(axis=1) / scale, device=device, dtype=torch.float32)
        normals = torch.tensor(normals, device=device, dtype=torch.float32)
        _, known, minus = _material_side(query, points, normals, epsilon)
        observed = minus[known]
        if len(observed) and bool((observed == observed[0]).all().item()):
            component_known[component] = True
            expected_minus[component] = bool(observed[0].item())
    unknown_component_orientations = int((~component_known).sum())
    volume_known &= unknown_component_orientations == 0
    volume = float(np.dot(volumes, np.where(expected_minus, 1., -1.)))
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(len(mesh.faces), sample_count, p=areas / areas.sum())
    bary = rng.random((sample_count, 2))
    root = np.sqrt(bary[:, 0])
    weights = np.column_stack((1 - root, root * (1 - bary[:, 1]), root * bary[:, 1]))
    # Keep the first sample_count area-uniform samples for unbiased acceptance.
    # Add three interior samples on every face absent from that draw so small
    # triangles cannot silently be treated as thick. Profile-only samples do
    # not enter the area-uniform estimator.
    if return_profile:
        missing = np.flatnonzero(np.bincount(face_ids, minlength=len(mesh.faces)) == 0)
        face_ids = np.r_[face_ids, np.repeat(missing, 3)]
        weights = np.concatenate((weights, np.tile(
            [[.6, .2, .2], [.2, .6, .2], [.2, .2, .6]], (len(missing), 1))))
    query_count = len(face_ids)
    # Reuse source triangles on the GPU for the millions of profile samples.
    device_vertices = torch.as_tensor(vertices, device=device, dtype=torch.float64)
    device_faces = torch.as_tensor(mesh_faces, device=device, dtype=torch.int64)
    device_expected = torch.as_tensor(expected_minus[labels], device=device)
    thin_count = 0
    measured_count = 0
    retain_samples = return_profile or return_samples
    sample_thickness = np.full(query_count, np.nan, dtype=np.float32) if retain_samples else None
    sample_valid = np.zeros(query_count, dtype=bool) if retain_samples else None
    sample_opposite = np.full(query_count, -1, dtype=np.int64) if retain_samples else None
    sample_inward = np.empty((sample_count, 3), dtype=np.float32) if return_samples else None
    for start in range(0, query_count, batch_size):
        ids = face_ids[start:start + batch_size]
        device_ids = torch.as_tensor(ids, device=device, dtype=torch.int64)
        triangles = device_vertices[device_faces[device_ids]] / scale
        barycentric = torch.as_tensor(weights[start:start + batch_size], device=device)
        points = (barycentric[..., None] * triangles).sum(dim=1).float()
        normal_vectors = torch.linalg.cross(triangles[:, 1] - triangles[:, 0],
                                           triangles[:, 2] - triangles[:, 0])
        normals = (normal_vectors / torch.linalg.vector_norm(
            normal_vectors, dim=1, keepdim=True)).float()
        inward, classified, minus = _material_side(query, points, normals, epsilon)
        expected = device_expected[device_ids]
        # Overlapping/self-intersecting shells can change material orientation
        # within one connected component. Such samples invalidate its volume.
        mismatch_count = int(((minus != expected) & classified).sum().item())
        orientation_mismatch_samples += mismatch_count
        volume_known &= mismatch_count == 0
        directions = thickness_directions(inward, torch, max_normal_angle_deg)
        ray_count = directions.shape[1]
        origins = points[:, None] + epsilon * directions
        source = device_ids[:, None].expand(-1, ray_count).reshape(-1)
        faces, distances = tracer._trace_trusted(origins.reshape(-1, 3), directions.reshape(-1, 3),
                                        epsilon / 8, 8.0, source)
        if exit_filter is not None:
            thickness, exits = exit_filter.normal_supported(
                faces, distances, directions, inward, epsilon, min_thickness / scale)
            thickness = thickness * scale
        else:
            exits = (faces.reshape(-1, ray_count) >= 0).all(dim=1)
            minimum = distances.reshape(-1, ray_count).min(dim=1).values
            thickness = (minimum + epsilon) * scale
        valid = classified & exits & torch.isfinite(thickness)
        if min_thickness <= 4 * epsilon * scale:
            valid = torch.zeros_like(valid)
        uniform_count = max(0, min(len(valid), sample_count - start))
        thin_count += int(((thickness[:uniform_count] < min_thickness) | ~valid[:uniform_count]).sum().item())
        measured_count += int(valid[:uniform_count].sum().item())
        if retain_samples:
            stop = start + len(thickness)
            sample_thickness[start:stop] = thickness.detach().cpu().numpy()
            sample_valid[start:stop] = valid.detach().cpu().numpy()
            sample_opposite[start:stop] = faces.reshape(-1, ray_count)[:, 0].detach().cpu().numpy()
        if return_samples and uniform_count:
            sample_inward[start:start+uniform_count] = inward[:uniform_count].detach().cpu().numpy()
    result = {'thin_wall_area_ratio': thin_count / sample_count, 'volume': volume,
              'reliable': bool(volume_known and np.isfinite(volume) and volume > 0 and measured_count > 0)}
    if diagnostics:
        result['reliability_details'] = {
            'initial_unknown_component_orientations': initial_unknown_component_orientations,
            'unknown_component_orientations': unknown_component_orientations,
            'orientation_mismatch_samples': orientation_mismatch_samples,
            'queried_samples': query_count,
            'measured_uniform_samples': measured_count,
            'uniform_samples': sample_count,
            'positive_finite_volume': bool(np.isfinite(volume) and volume > 0),
        }
    if return_samples:
        # Preserve the exact area-uniform acceptance draw. No additional
        # per-face coverage queries are needed for a reconstruction audit.
        result['samples'] = {
            'face_ids': face_ids[:sample_count].copy(),
            'bary': weights[:sample_count].copy(),
            'thickness': sample_thickness[:sample_count].copy(),
            'valid': sample_valid[:sample_count].copy(),
            'opposite_faces': sample_opposite[:sample_count].copy(),
            'inward': sample_inward,
        }
    if return_profile:
        face_thickness, opposite, valid_count = _aggregate_profile_samples(
            len(mesh.faces), face_ids, sample_thickness, sample_valid,
            sample_opposite, sample_count, missing)
        face_area = np.asarray(areas, dtype=np.float64)
        unknown = ~np.isfinite(face_thickness)
        thin = (~unknown) & (face_thickness < min_thickness)
        result['face_profile'] = {
            'thickness': face_thickness,
            'opposite_faces': opposite,
            'gap': np.maximum(min_thickness - np.nan_to_num(face_thickness, nan=0.0), 0.0),
            'unknown': unknown,
            'thin': thin,
            'areas': face_area,
            'thin_area_ratio': float(face_area[thin].sum() / face_area.sum()),
            'unknown_area_ratio': float(face_area[unknown].sum() / face_area.sum()),
            'sample_count': valid_count,
        }
    return result


def measure_profile(mesh, min_thickness, **kwargs):
    """Return the normal thickness measurement plus a sampled per-face profile."""
    kwargs['return_profile'] = True
    return measure(mesh, min_thickness, **kwargs)
