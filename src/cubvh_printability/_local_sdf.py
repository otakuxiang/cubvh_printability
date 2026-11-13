"""Signed, spatially varying offsets on a conservative sparse cell grid."""
import itertools
import logging
import time
import numpy as np
import trimesh

_LOG = logging.getLogger(__name__)


def _cuda_memory(torch, stage, **details):
    """Log allocator and driver memory without making CUDA profiling mandatory.

    ``memory_allocated`` only accounts for tensors owned by PyTorch.  The
    sparse marching-cubes implementation uses thrust allocations, so the
    driver free-memory value is the useful number at the MC boundary.
    """
    if not torch.cuda.is_available():
        return
    try:
        free, total = torch.cuda.mem_get_info()
        fields = {
            'free': int(free),
            'total': int(total),
            'allocated': int(torch.cuda.memory_allocated()),
            'reserved': int(torch.cuda.memory_reserved()),
            'max_allocated': int(torch.cuda.max_memory_allocated()),
        }
        fields.update(details)
        _LOG.info('CUDA memory [%s]: %s', stage, fields)
    except (RuntimeError, AttributeError):
        # Profiling must never turn a valid reconstruction into a failure.
        _LOG.debug('CUDA memory query failed at %s', stage, exc_info=True)


def select_offsets(thickness, areas, minimum, threshold, cap, *, priority='area_deficit',
                   visibility=None):
    """Select by area * deficit until remaining thin area is strictly below target."""
    thickness, areas = np.asarray(thickness), np.asarray(areas)
    if not np.isfinite(thickness).all():
        raise ValueError('unknown face thickness requires global fallback')
    gap = np.maximum(minimum - thickness, 0.)
    candidates = np.flatnonzero(gap > 0)
    if priority == 'area_deficit':
        score = -(areas[candidates] * gap[candidates])
    elif priority == 'minimum_displacement':
        # Minimize the maximum requested offset needed to meet the area budget.
        # Each repaired unit of area costs its thickness deficit, independent
        # of how that surface happens to be tessellated into triangles.
        score = gap[candidates]
    elif priority == 'visible_displacement':
        exposure = np.asarray(visibility, dtype=float)
        if (exposure.shape != thickness.shape or not np.isfinite(exposure).all()
                or np.any(exposure < 0)):
            raise ValueError('visibility must contain one finite nonnegative value per face')
        positive = exposure[exposure > 0]
        floor = float(np.median(positive) * .1) if len(positive) else 1.
        # Cost per repaired area: thickness deficit times projected external
        # exposure. The floor prevents missed samples from having zero cost.
        score = gap[candidates] * (exposure[candidates] + floor)
    elif priority == 'projected_cost':
        cost = np.asarray(visibility, dtype=float)
        if cost.shape != thickness.shape or not np.isfinite(cost).all() or np.any(cost < 0):
            raise ValueError('projected costs must be finite nonnegative face values')
        score = cost[candidates]
    else:
        raise ValueError('unknown thin-face selection priority')
    order = candidates[np.argsort(score, kind='stable')]
    selected = np.zeros(len(areas), dtype=bool)
    remaining = float(areas[candidates].sum())
    target = float(threshold * areas.sum())
    if remaining >= target and len(order):
        cumulative = np.cumsum(areas[order], dtype=np.float64)
        count = min(len(order), int(np.searchsorted(cumulative, remaining - target, side='right')) + 1)
        selected[order[:count]] = True
    offsets = np.where(selected, np.minimum(gap * .5, cap), 0.)
    return offsets, selected


def smooth_vertex_offsets(mesh, offsets, width, *, protected_faces=None):
    """Extend nonnegative offsets along mesh edges with compact smoothstep decay.

    Distances are intrinsic to the mesh: separate nearby sheets cannot spread
    offsets to each other. Selected vertices retain their requested amplitude.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    faces, vertices = np.asarray(mesh.faces), np.asarray(mesh.vertices)
    values = np.zeros(len(vertices), dtype=np.float64)
    for k in range(3):
        np.maximum.at(values, faces[:, k], offsets)
    seeds = np.flatnonzero(values > 0)
    if not len(seeds):
        return values
    # Unique edges keep duplicate COO entries from doubling lengths.
    graph = mesh._cache['printability_intrinsic_graph']
    if graph is None:
        nv = len(vertices)
        if nv * nv < np.iinfo(np.int64).max:
            keys = np.empty(len(faces) * 3, dtype=np.int64)
            for start in range(0, len(faces), 262144):
                part = faces[start:start + 262144]
                for edge, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
                    u, v = part[:, a], part[:, b]
                    keys[start*3+edge:(start+len(part))*3:3] = np.minimum(u, v)*nv+np.maximum(u, v)
            keys = np.unique(keys)
            edges = np.column_stack((keys//nv, keys%nv))
        else:
            edges = np.asarray(mesh.edges_unique)
        lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        graph = coo_matrix((np.r_[lengths, lengths],
                        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                       shape=(len(vertices), len(vertices))).tocsr()
        mesh._cache['printability_intrinsic_graph'] = graph
    # Multi-source weighted distance creates the upper envelope of cones.
    # Unlike nearest-seed interpolation, this bounds spatial slope even when
    # adjacent seeds have very different thickness deficits.
    maximum = float(values.max())
    slope = maximum / width
    n = len(vertices)
    inverse = .5 - np.sin(np.arcsin(1 - 2 * np.clip(values[seeds] / maximum, 0., 1.)) / 3)
    virtual_cost = width * (1 - inverse) + 1e-12
    rows, cols = graph.nonzero()
    augmented = coo_matrix((np.r_[graph.data, virtual_cost],
        (np.r_[rows, np.full(len(seeds), n)], np.r_[cols, seeds])),
        shape=(n + 1, n + 1)).tocsr()
    distances = dijkstra(augmented, directed=True, indices=n, limit=width)[:n]
    envelope = np.maximum(0., maximum - slope * distances)
    # A smoothstep profile has zero slope at support boundary and at peaks.
    # Inverting smoothstep at source costs preserves requested seed amplitudes.
    t = np.clip(envelope / maximum, 0., 1.)
    values = maximum * t * t * (3 - 2 * t)

    if protected_faces is not None:
        protected = np.asarray(protected_faces, dtype=bool)
        if protected.shape != (len(faces),):
            raise ValueError('protected_faces must contain one flag per face')
        anchors = np.unique(faces[protected])
        if len(anchors):
            # Enforce zero displacement on observed surface vertices, with a
            # C1 taper into hidden regions. Visibility weighting at seeds alone
            # cannot prevent a geodesic envelope from leaking onto the exterior.
            clearance = dijkstra(graph, directed=True, indices=anchors,
                                 min_only=True, limit=width)
            u = np.clip(clearance / width, 0., 1.)
            values *= u * u * (3 - 2 * u)

    return values


def outward_ball_surface(vertices, faces, radii, query, torch):
    """Build affine centers/radii of balls tangent on the material's air side.

    A radius increment r uses center q + r*n/2 and ball radius r/2. On
    each planar triangle, both center and radius are affine. Every ball
    touches its source plane and remains in its outward half-space.
    Unknown or conflicting orientation of a growing triangle is rejected.
    """
    from ._printability_thickness import _material_side
    active = np.flatnonzero(np.max(radii[faces], axis=1) > 0)
    centers = np.empty((len(active), 3, 3), dtype=np.float32)
    ball_radii = np.asarray(radii[faces[active]] * .5, dtype=np.float32)
    for start in range(0, len(active), 65536):
        ids = active[start:start+65536]
        triangles = vertices[faces[ids]]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        if np.any(lengths <= 0):
            raise ValueError('outward growth requires nondegenerate source triangles')
        normals /= lengths
        normal = torch.as_tensor(normals, dtype=torch.float32, device=query.tracer.device)
        points = torch.as_tensor(triangles.mean(axis=1), dtype=torch.float32, device=normal.device)
        inward, known, minus = _material_side(query, points, normal, query.epsilon)
        # Two independent interior probes resolve numerically ambiguous
        # centroids, but conflicting material sides cannot be averaged away.
        missing = torch.nonzero(~known).flatten()
        if len(missing):
            retry_tri = triangles[missing.cpu().numpy()]
            observed = torch.zeros(len(missing), dtype=torch.bool, device=normal.device)
            side = torch.zeros_like(observed)
            conflict = torch.zeros_like(observed)
            for bary in ([.6, .2, .2], [.2, .6, .2], [.2, .2, .6]):
                points = torch.as_tensor(np.einsum('j,ijk->ik', bary, retry_tri),
                                         dtype=torch.float32, device=normal.device)
                _, valid, probe_side = _material_side(query, points, normal[missing], query.epsilon)
                conflict |= observed & valid & (side != probe_side)
                side = torch.where(valid, probe_side, side)
                observed |= valid
            known[missing] = observed & ~conflict
            minus[missing] = side
            inward[missing] = torch.where(side[:, None], -normal[missing], normal[missing])
        if not bool(known.all().item()):
            raise ValueError(f'unknown material orientation on {int((~known).sum().item())} growing faces')
        outward = -inward.cpu().numpy()
        centers[start:start+len(ids)] = (triangles +
            ball_radii[start:start+len(ids), :, None] * outward[:, None])
    if not len(active):
        raise ValueError('outward growth requires a nonzero radius')
    # cuBVH requires more than eight triangles; duplicates leave this minimum
    # exactly unchanged for small diagnostic meshes.
    if len(active) <= 8:
        repeats = (9 + len(active)-1) // len(active)
        centers = np.tile(centers, (repeats, 1, 1))
        ball_radii = np.tile(ball_radii, (repeats, 1))
    return centers.reshape(-1, 3), np.arange(centers.size//3).reshape(-1, 3), ball_radii


def shared_grid_corners(cells, corners, resolution, torch):
    """Return unique int64 grid positions and cell-corner gather indices."""
    grid = cells.to(dtype=torch.int64)[:, None] + corners
    n = resolution + 1
    keys = (grid[..., 0] * n + grid[..., 1]) * n + grid[..., 2]
    unique, inverse = torch.unique(keys.reshape(-1), return_inverse=True)
    positions = torch.stack((unique // (n*n), unique // n % n, unique % n), dim=1)
    return positions, inverse


def reconstruct_local(backend, mesh, face_offsets):
    """Extract a local offset field using original material parity.

    Cell subdivision uses the global maximum radius and the 1-Lipschitz UDF
    bound, never assumes the varying radius field itself is 1-Lipschitz.
    Original material parity defines signs even where local radius is zero.
    Experimental continuous variants union original material with surface
    balls, optionally translated into the local outward half-space.
    """
    started = time.perf_counter()
    from ._printability_reconstruct import CORNERS, CHUNK
    from ._printability_thickness import MaterialQuery
    from .optix import OptixRayTracer
    torch = backend.torch
    res, scale, device = backend.resolution, backend.scale, backend.device
    center = np.asarray(mesh.bounds).mean(axis=0)
    vertices = (np.asarray(mesh.vertices) - center) / scale
    faces = np.asarray(mesh.faces)
    max_offset = float(np.max(face_offsets))
    # A zero field has an exact and useful answer.  Besides avoiding a costly
    # no-op MC pass, this prevents the sparse pipeline from allocating a full
    # shell merely to reproduce the source mesh.
    if max_offset == 0 and not np.any(np.asarray(face_offsets) != 0):
        return mesh.copy()
    max_radius = max_offset / scale
    half = (.5 + max_radius) / (1 - 6 / res)
    step = 2 * half / res
    transition_factor = float(getattr(backend, 'local_transition_factor', 4.))
    if not np.isfinite(transition_factor) or transition_factor <= 0:
        raise ValueError('local transition factor must be finite and positive')
    transition_width = max(2 * step * scale, transition_factor * max_offset)
    solved_values = getattr(backend, 'local_vertex_offsets', None)
    if solved_values is None:
        field_values = smooth_vertex_offsets(mesh, face_offsets, transition_width,
            protected_faces=getattr(backend, 'local_protected_faces', None)) / scale
    else:
        solved_values = np.asarray(solved_values, dtype=float)
        if (solved_values.shape != (len(vertices),) or not np.isfinite(solved_values).all()
                or np.any(solved_values < 0) or np.any(solved_values > max_offset + 1e-12)):
            raise ValueError('vertex offsets must be finite, nonnegative and bounded by the face offset maximum')
        field_values = solved_values / scale
    # cuBVH requires >8 faces: carry face/vertex scalar data through subdivision.
    if len(faces) <= 8:
        from trimesh.remesh import subdivide
        vertices, faces, attributes = subdivide(vertices, faces,
            vertex_attributes={'radius': field_values[:, None]})
        field_values = attributes['radius'].reshape(-1)
    bvh = backend.BVH(vertices / half, faces)
    directional_distance = bool(getattr(backend, 'local_directional_distance', False))
    weighted_distance = bool(getattr(backend, 'local_weighted_distance', False)) or directional_distance
    tracer = OptixRayTracer(vertices, faces)
    query = MaterialQuery(tracer, float(4 * np.finfo(np.float32).eps))
    offset_bvh = bvh
    if directional_distance:
        shifted, shifted_faces, ball_radii = outward_ball_surface(vertices, faces, field_values, query, torch)
        offset_bvh = backend.BVH(shifted / half, shifted_faces)
        offset_bvh.set_offset_radii(ball_radii / half)
        del shifted, shifted_faces, ball_radii
    elif weighted_distance:
        bvh.set_offset_radii(np.asarray(field_values[faces] / half, dtype=np.float32))
    seed_bound = bool(getattr(backend, 'local_seed_bound', False)) and weighted_distance and not directional_distance
    seed_vertices = seed_radii = seed_faces = None
    if seed_bound:
        seed_vertices = torch.as_tensor(vertices / half, dtype=torch.float32, device=device)
        seed_radii = torch.as_tensor(field_values / half, dtype=torch.float32, device=device)
        seed_faces = torch.as_tensor(faces, dtype=torch.long, device=device)

    def initial_bound(points, ids, initial):
        # Every vertex ball is a member of the affine triangle envelope.
        # Its value is an upper bound on the minimum, with a conservative
        # float32 allowance in these normalized coordinates. The native
        # query still searches all triangles that can improve this bound.
        indices = seed_faces[ids.long()]
        distances = torch.linalg.vector_norm(points[:, None]-seed_vertices[indices], dim=2)
        upper = (distances-seed_radii[indices]).amin(dim=1)
        return torch.minimum(initial, upper + 2e-6*(1+upper.abs()))
    vertex_radius = face_tensor = None
    if not weighted_distance:
        vertex_radius = torch.as_tensor(field_values, dtype=torch.float32, device=device)
        face_tensor = torch.as_tensor(faces, dtype=torch.long, device=device)
    corners = torch.tensor(CORNERS, dtype=torch.int64, device=device)
    bits = torch.tensor(list(itertools.product((False, True), repeat=3)), device=device)
    leaf_offsets = torch.cartesian_prod(*(torch.arange(4, device=device) for _ in range(3)))
    blocks = [(torch.zeros((1, 3), dtype=torch.int64, device=device),
               torch.full((1, 3), res, dtype=torch.int64, device=device))]
    # Stage compact crossing records on the host. This keeps GPU residency
    # bounded during queries and avoids list + concatenation GPU copies.
    kept_cells, kept_fields = [], []
    _cuda_memory(torch, 'local_reconstruct_start', resolution=int(res))
    tested = queried_corners = 0
    prune_envelope = bool(getattr(backend, 'local_prune_envelope', False)) and weighted_distance and not directional_distance
    while blocks:
        low, high = blocks.pop()
        if len(low) > 2048:
            blocks.append((low[2048:], high[2048:]))
            low, high = low[:2048], high[:2048]
        centers = (low + high).float() * (.5 * step) - half
        distance, nearest, _ = bvh.unsigned_distance(centers / half)
        distance = distance * half
        bound = torch.linalg.vector_norm((high - low).float(), dim=1) * (.5 * step)
        active = distance <= max_radius + bound + 8 * np.finfo(np.float32).eps * half
        if prune_envelope and bool(active.any().item()):
            # W(x)=min_q(|x-q|-r(q)) is 1-Lipschitz even when r varies.
            # W<-bound means the entire block is inside growing balls.
            # W>bound means no ball or source surface enters the block
            # (r>=0), so original-material parity is constant there too.
            # Neither case can contain a zero of min(material_sdf, W).
            ids = torch.nonzero(active).flatten()
            initial = distance[ids] / half
            if seed_bound:
                initial = initial_bound(centers[ids] / half, nearest[ids], initial)
            envelope = bvh.min_offset_distance(centers[ids] / half, initial) * half
            active[ids] = envelope.abs() <= bound[ids] + 2e-6 * half
        low, high = low[active], high[active]
        leaf = (high - low).amax(dim=1) <= 4
        lo, hi = low[~leaf], high[~leaf]
        if len(lo):
            mid = (lo + hi) // 2
            cl = torch.where(bits[None], mid[:, None], lo[:, None]).reshape(-1, 3)
            ch = torch.where(bits[None], hi[:, None], mid[:, None]).reshape(-1, 3)
            valid = (ch > cl).all(dim=1)
            blocks.append((cl[valid], ch[valid]))
        lo, hi = low[leaf], high[leaf]
        if not len(lo):
            continue
        cells = lo[:, None] + leaf_offsets[None]
        cells = cells[(cells < hi[:, None]).all(dim=2)]
        for batch in cells.split(CHUNK // 8):
            centers = (batch.float() + .5) * step - half
            d = bvh.unsigned_distance(centers / half)[0] * half
            batch = batch[d <= max_radius + np.sqrt(3) * .5 * step + 1e-6]
            if not len(batch):
                continue
            # Adjacent cells share up to eight corners. Query each grid
            # position once per bounded batch, then gather its scalar field.
            # int64 keys preserve uniqueness above a 1290-cubed grid.
            grid, inverse = shared_grid_corners(batch, corners, res, torch)
            xyz = grid.float() * step - half
            queried_corners += len(grid)
            if weighted_distance:
                distance, nearest, _ = bvh.unsigned_distance(xyz / half)
            else:
                distance, ids, weights = bvh.unsigned_distance(xyz / half, return_uvw=True)
                radius = (vertex_radius[face_tensor[ids.long()]] * weights).sum(dim=1)
            distance = distance * half
            decisions = tracer._impl.contains_closest(
                xyz.contiguous(), query.trace_directions, query.step_directions, query.epsilon)
            valid = decisions < 2
            inside_votes = ((decisions == 1) & valid).sum(dim=1)
            outside_votes = ((decisions == 0) & valid).sum(dim=1)
            inside = inside_votes >= 2
            known = inside | (outside_votes >= 2)
            uncertain = ~known & (distance > 8 * query.epsilon)
            if bool(uncertain.any().item()):
                raise ValueError('ambiguous material parity in local SDF reconstruction')
            base = torch.where(inside, -distance, distance)
            if weighted_distance:
                # Union of original material and variable-radius balls.
                # Searching all potential contributors prevents jumps
                # when the nearest unmodified surface switches between sheets.
                initial = initial_bound(xyz / half, nearest, base / half) if seed_bound else base / half
                sdf = offset_bvh.min_offset_distance(xyz / half, initial) * half
            else:
                sdf = base - radius
            # Exact surface corners use a deterministic negative sign.
            sdf = torch.where(sdf == 0, -step * 1e-6, sdf)[inverse].reshape(-1, 8)
            crossing = (sdf < 0).any(dim=1) & (sdf >= 0).any(dim=1)
            if bool(crossing.any().item()):
                kept_cells.append(batch[crossing].to(torch.int32).cpu().numpy())
                kept_fields.append(sdf[crossing].cpu().numpy())
            tested += len(batch)
            del xyz, distance, decisions, valid, grid, inverse
            if not weighted_distance:
                del ids, weights, radius
            del inside_votes, outside_votes, inside, known, uncertain, sdf, base
        del cells
    if not kept_cells:
        return None
    total_crossing = sum(len(c) for c in kept_cells)
    _cuda_memory(torch, 'crossing_accumulated', cells=total_crossing)
    del bvh, offset_bvh, tracer, query, face_tensor, vertex_radius
    del seed_vertices, seed_radii, seed_faces
    cells = torch.empty((total_crossing, 3), dtype=torch.int32, device=device)
    field = torch.empty((total_crossing, 8), dtype=torch.float32, device=device)
    start = 0
    for cell_part, field_part in zip(kept_cells, kept_fields):
        stop = start + len(cell_part)
        cells[start:stop].copy_(torch.from_numpy(cell_part))
        field[start:stop].copy_(torch.from_numpy(field_part))
        start = stop
    del kept_cells, kept_fields, cell_part, field_part
    field_seconds = time.perf_counter() - started
    _LOG.info('Local SDF: %d tested cells, %d crossing cells; %d queried corners (%.2fx reuse)',
              tested, len(cells), queried_corners, 8*tested/max(1, queried_corners))
    if len(cells) >= (2**31 - 1) // 12:
        raise MemoryError('local marching cubes index capacity exceeded')
    torch.cuda.empty_cache()
    _cuda_memory(torch, 'before_sparse_mc', cells=len(cells),
                 cell_bytes=cells.numel() * cells.element_size(),
                 field_bytes=field.numel() * field.element_size())
    v, f = backend.mc(cells, field, 0., ensure_consistency=False, high_precision=True)
    _cuda_memory(torch, 'after_sparse_mc', vertices=len(v), faces=len(f))
    del cells, field
    vertices = v.cpu().numpy()
    # MC high_precision=True returns float64; transform it in place to avoid
    # several simultaneous full host copies of a multi-million-vertex mesh.
    vertices *= step
    vertices -= half
    vertices *= scale
    vertices += center
    backend.local_reconstruction_stats = {
        'resolution': int(res), 'tested_cells': tested, 'crossing_cells': total_crossing,
        'queried_corners': queried_corners, 'field_seconds': field_seconds,
        'total_seconds': time.perf_counter() - started,
    }
    return trimesh.Trimesh(vertices=vertices, faces=f.cpu().numpy(), process=False)


def visibility_weights(mesh, size, resolution=384):
    """Estimate external exposure with three probes per face in ten directions.

    Image hit counts miss small visible faces when tessellation exceeds pixel
    count. Trace from each face toward each camera instead. A face is observed
    when any interior probe has an unobstructed ray. This remains a finite
    sampling estimate, not proof that an unsampled surface is hidden.
    ``resolution`` is retained for callers of the former raster estimator.
    """
    from ._printability_fidelity import VIEW_DIRECTIONS
    from .optix import OptixRayTracer
    tracer = OptixRayTracer(np.asarray(mesh.vertices) / size, np.asarray(mesh.faces))
    torch = tracer._torch
    epsilon = float(4 * np.finfo(np.float32).eps)
    directions = np.asarray(VIEW_DIRECTIONS, dtype=float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions = torch.as_tensor(directions, device=tracer.device, dtype=torch.float32)
    barycentric = np.array([[.6, .2, .2], [.2, .6, .2], [.2, .2, .6]])
    exposure = np.zeros(len(mesh.faces), dtype=np.float32)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    for start in range(0, len(faces), 65536):
        triangles = vertices[faces[start:start + 65536]] / size
        points = np.einsum('sk,nkj->nsj', barycentric, triangles)
        points = torch.as_tensor(points, device=tracer.device, dtype=torch.float32)
        source = torch.arange(start, start + len(triangles), device=tracer.device)
        source = source[:, None].expand(-1, 3).reshape(-1)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
        normal = torch.as_tensor(normals, device=tracer.device, dtype=torch.float32)
        scores = torch.zeros(len(triangles), device=tracer.device)
        for direction in directions:
            origins = points.reshape(-1, 3) + epsilon * direction
            hits, _ = tracer._trace_trusted(origins, direction.expand_as(origins),
                                            epsilon / 8, 4., source)
            visible = (hits.reshape(-1, 3) < 0).any(dim=1)
            scores += visible * (normal * direction).sum(dim=1).abs()
        exposure[start:start + len(triangles)] = scores.cpu().numpy()
    return exposure


def allocate_paired_offsets(offsets, opposite, visibility, cap, *, protected_faces=None):
    """Share required wall growth between paired faces, favoring less visible sides."""
    offsets = np.asarray(offsets)
    opposite, visibility = np.asarray(opposite), np.asarray(visibility)
    if np.any((offsets > 0) & ((opposite < 0) | (opposite >= len(offsets)))):
        raise ValueError('selected thin face has no valid opposing surface')
    result = np.zeros_like(offsets)
    ids = np.flatnonzero((offsets > 0) & (opposite >= 0))
    peers = opposite[ids]
    # A floor keeps missing screen samples from claiming perfect invisibility.
    positive = visibility[visibility > 0]
    floor = float(np.median(positive) * .1) if len(positive) else 1.
    own, other = visibility[ids] + floor, visibility[peers] + floor
    total = 2 * offsets[ids]
    own_cap = np.full(len(ids), cap)
    other_cap = np.full(len(ids), cap)
    if protected_faces is not None:
        protected = np.asarray(protected_faces, dtype=bool)
        if protected.shape != offsets.shape:
            raise ValueError('protected_faces must contain one flag per face')
        own_cap[protected[ids]] = 0.
        other_cap[protected[peers]] = 0.
        if np.any(total > own_cap + other_cap + 1e-12):
            raise ValueError('protected wall pair cannot accommodate requested growth')
    own_growth = np.minimum(total * other / (own + other), own_cap)
    other_growth = np.minimum(total - own_growth, other_cap)
    own_growth = np.minimum(total - other_growth, own_cap)
    np.maximum.at(result, ids, own_growth)
    np.maximum.at(result, peers, other_growth)
    return result
