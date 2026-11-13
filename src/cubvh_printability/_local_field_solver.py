"""Experimental convex wall-growth field with coupled surface regularization."""
import logging
import numpy as np


def projected_wall_cost(thickness, opposite, minimum, caps, weights, penalty=100.,
                        *, render_loss='quadratic', linear_scale=.025):
    """Rank paired walls by projected growth cost, minimizing over side splits.

    This is a seed-selection surrogate; it does not certify image fidelity.
    View weights are projected visible fractions, not normalized per face.
    """
    thickness, opposite = np.asarray(thickness), np.asarray(opposite)
    caps, weights = np.asarray(caps), np.asarray(weights)
    if (caps.ndim != 2 or caps.shape != weights.shape or len(caps) != len(thickness)
            or opposite.shape != thickness.shape or not np.isfinite(caps).all()
            or not np.isfinite(weights).all() or np.any(caps < 0) or np.any(weights < 0)
            or not np.isfinite(penalty) or penalty <= 0
            or render_loss not in ('quadratic', 'linear')
            or not np.isfinite(linear_scale) or linear_scale <= 0):
        raise ValueError('invalid projected wall cost data')
    result = np.full(len(thickness), np.finfo(np.float64).max)
    eligible = np.flatnonzero(np.isfinite(thickness) & (thickness < minimum)
                             & (opposite >= 0) & (opposite < len(thickness)))
    for start in range(0, len(eligible), 32768):
        ids = eligible[start:start+32768]
        other = opposite[ids]
        gap = minimum-thickness[ids]
        best = np.full(len(ids), np.inf)
        for fraction in np.linspace(0, 1, 17):
            r, s = gap*fraction, gap*(1-fraction)
            excess_r = np.maximum(r[:, None]-caps[ids], 0)
            excess_s = np.maximum(s[:, None]-caps[other], 0)
            if render_loss == 'quadratic':
                excess_r, excess_s = excess_r**2, excess_s**2
            else:
                # Twice the linear hinge matches the quadratic objective's
                # common factor 1/2, omitted throughout this ranking cost.
                excess_r, excess_s = 2*linear_scale*excess_r, 2*linear_scale*excess_s
            damage = ((weights[ids]*excess_r).mean(axis=1)
                      + (weights[other]*excess_s).mean(axis=1))
            best = np.minimum(best, .1*(r*r+s*s)+penalty*damage)
        result[ids] = best
    return result


def surface_gradient_operator(mesh):
    """Return affine barycentric gradients on each nondegenerate triangle."""
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    u, v = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    normal = np.cross(u, v)
    determinant = np.einsum('ij,ij->i', normal, normal)
    if np.any(determinant <= 0):
        raise ValueError('surface slope constraints require nondegenerate triangles')
    g1 = np.cross(v, normal) / determinant[:, None]
    g2 = np.cross(normal, u) / determinant[:, None]
    return np.stack((-g1-g2, g1, g2), axis=1)


def validate_point_caps(point_caps, face_count):
    """Validate sparse affine-radius upper bounds at source triangle points."""
    ids = np.asarray(point_caps['face_ids'])
    bary, caps = np.asarray(point_caps['bary']), np.asarray(point_caps['caps'])
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any((ids < 0) | (ids >= face_count))
            or bary.shape != (len(ids), 3) or caps.shape != (len(ids),)
            or not np.isfinite(bary).all() or not np.isfinite(caps).all()
            or np.any(bary < 0) or np.any(caps < 0)
            or not np.allclose(bary.sum(axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError('invalid affine point caps')
    return ids, bary, caps


def validate_wall_witnesses(witnesses, face_count):
    """Validate paired affine growth targets obtained from measured wall spans."""
    ids = np.asarray(witnesses['face_ids'])
    bary = np.asarray(witnesses['bary'], dtype=float)
    targets = np.asarray(witnesses['targets'], dtype=float)
    areas = np.asarray(witnesses['areas'], dtype=float)
    factors = np.asarray(witnesses['factors'], dtype=float)
    count = ids.shape[0] if ids.ndim else 0
    if (ids.shape != (count, 2) or ids.dtype.kind not in 'iu'
            or np.any((ids < 0) | (ids >= face_count))
            or bary.shape != (count, 2, 3) or not np.isfinite(bary).all()
            or np.any(bary < 0) or not np.allclose(bary.sum(axis=2), 1., atol=1e-6, rtol=0)
            or targets.shape != (count,) or not np.isfinite(targets).all() or np.any(targets < 0)
            or areas.shape != (count,) or not np.isfinite(areas).all() or np.any(areas <= 0)
            or factors.shape != (count, 2) or not np.isfinite(factors).all()
            or np.any((factors < 1) | (factors > 2))):
        raise ValueError('invalid measured wall witnesses')
    return ids, bary, targets, areas, factors


def radius_capacity_envelope(mesh, point_caps, maximum, slope=.5):
    """Necessary vertex bounds implied by point caps and surface slope.

    Graph propagation supplies upper bounds, not a sufficient condition for
    triangle-interior slope or image fidelity. Keep the point constraints in
    the solve and project/reconstruct/review its result separately.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    if not np.isfinite(maximum) or maximum <= 0 or not np.isfinite(slope) or not 0 < slope < 1:
        raise ValueError('invalid capacity bound scales')
    faces, vertices = np.asarray(mesh.faces), np.asarray(mesh.vertices)
    ids, bary, limits = validate_point_caps(point_caps, len(faces))
    bounds = np.full(len(vertices), maximum, dtype=np.float64)
    for start in range(0, len(ids), 131072):
        f = faces[ids[start:start+131072]]
        triangles = vertices[f]
        q = (triangles*bary[start:start+len(f), :, None]).sum(axis=1)
        for k in range(3):
            local = limits[start:start+len(f)] + slope*np.linalg.norm(triangles[:, k]-q, axis=1)
            np.minimum.at(bounds, f[:, k], local)
    edges = np.asarray(mesh.edges_unique)
    costs = slope*np.linalg.norm(vertices[edges[:, 0]]-vertices[edges[:, 1]], axis=1)
    n = len(vertices)
    graph = coo_matrix((np.r_[costs, costs, bounds],
        (np.r_[edges[:, 0], edges[:, 1], np.full(n, n)],
         np.r_[edges[:, 1], edges[:, 0], np.arange(n)])), shape=(n+1, n+1)).tocsr()
    return dijkstra(graph, directed=True, indices=n)[:n]


def select_capacity_walls(mesh, thickness, opposite, areas, minimum, threshold,
                          envelope, strength=1., *, visibility=None):
    """Reselect source walls that can meet the affine target under current caps.

    Unknown/unpaired area must already be reserved by the caller. If this
    model cannot cover the requested area, propose every eligible wall and
    report the shortfall; only reconstructed measurement decides acceptance.
    """
    thickness, opposite, areas = map(np.asarray, (thickness, opposite, areas))
    envelope = np.asarray(envelope)
    n = len(mesh.faces)
    if (thickness.shape != (n,) or opposite.shape != (n,) or areas.shape != (n,)
            or opposite.dtype.kind not in 'iu'
            or not np.isfinite(thickness).all() or not np.isfinite(areas).all()
            or np.any(areas < 0) or areas.sum() <= 0
            or envelope.shape != (len(mesh.vertices),) or not np.isfinite(envelope).all()
            or np.any(envelope < 0) or not np.isfinite(minimum) or minimum <= 0
            or not np.isfinite(threshold) or not 0 <= threshold <= 1
            or not np.isfinite(strength) or strength < 1):
        raise ValueError('invalid capacity selection inputs')
    capacity = envelope[np.asarray(mesh.faces)].mean(axis=1)
    paired = (opposite >= 0) & (opposite < n)
    gap = np.maximum(minimum-thickness, 0.)
    available = capacity + capacity[np.clip(opposite, 0, n-1)]
    eligible = paired & (gap > 0) & (available >= strength*gap)
    order = np.flatnonzero(eligible)
    if visibility is None:
        score = gap[order]
    else:
        visibility = np.asarray(visibility, dtype=float)
        if (visibility.shape != (n,) or not np.isfinite(visibility).all()
                or np.any(visibility < 0)):
            raise ValueError('visibility must be finite nonnegative face values')
        positive = visibility[visibility > 0]
        floor = float(np.median(positive) * .1) if len(positive) else 1.
        # Prefer feasible walls with a small thickness deficit and low camera
        # exposure.  The area target is unchanged; this only determines which
        # eligible wall is allowed to spend the appearance budget first.
        score = gap[order] * (visibility[order] + floor)
    order = order[np.argsort(score, kind='stable')]
    selected = np.zeros(n, dtype=bool)
    thin_area = float(areas[gap > 0].sum())
    target = float(threshold*areas.sum())
    if thin_area >= target and len(order):
        count = min(len(order), int(np.searchsorted(np.cumsum(areas[order]),
                    thin_area-target, side='right'))+1)
        selected[order[:count]] = True
    selected_area = float(areas[selected].sum())
    return selected, {'eligible_source_area_ratio': float(areas[eligible].sum()/areas.sum()),
        'selected_source_area_ratio': selected_area/float(areas.sum()),
        'model_budget_met': bool(thin_area-selected_area < target),
        'model_only': True}


def project_radius_caps(mesh, values, point_caps, *, max_surface_slope=.5,
                        iterations=128, tolerance=1e-5, device='cuda',
                        floor_values=None):
    """Monotonically reduce a field to satisfy point caps and restore smoothness.

    Point projection scales the three incident radii together. Triangle
    projection contracts values towards a lower field, bounding its affine
    gradient. When ``floor_values`` is supplied, that lower field is preserved
    exactly; this lets render-reviewed growth receive a measured-wall
    increment without retracting the reviewed geometry. Simultaneous
    contractions can activate adjacent triangle constraints, so finite
    iterations expose residuals instead of certifying convergence.
    Nonnegative point upper bounds survive every contraction.
    Reconstructed thickness and rendering must be checked after projection.
    """
    import torch
    faces = np.asarray(mesh.faces)
    initial = np.asarray(values, dtype=float)
    if (initial.shape != (len(mesh.vertices),) or not np.isfinite(initial).all()
            or np.any(initial < 0) or not np.isfinite(max_surface_slope)
            or not 0 < max_surface_slope < 1 or iterations < 1
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('invalid radius projection inputs')
    if floor_values is None:
        floor = np.zeros_like(initial)
    else:
        floor = np.asarray(floor_values, dtype=float)
        if (floor.shape != initial.shape or not np.isfinite(floor).all()
                or np.any(floor < 0) or np.any(floor > initial + tolerance)):
            raise ValueError('floor values must be finite, nonnegative and at most the field')
        # Permit only harmless ulp-level overshoot from a previous projection.
        floor = np.minimum(floor, initial)
    ids, bary, caps = validate_point_caps(point_caps, len(faces))
    radius = torch.tensor(initial, dtype=torch.float64, device=device)
    lower = torch.tensor(floor, dtype=torch.float64, device=device)
    triangle_ids = torch.tensor(faces, dtype=torch.long, device=device)
    gradient = torch.tensor(surface_gradient_operator(mesh), dtype=torch.float64, device=device)
    if len(ids):
        point_ids = triangle_ids[torch.tensor(ids, dtype=torch.long, device=device)]
        point_bary = torch.tensor(bary, dtype=torch.float64, device=device)
        limits = torch.tensor(caps, dtype=torch.float64, device=device)
        growth = (radius[point_ids]*point_bary).sum(dim=1)
        floor_growth = (lower[point_ids]*point_bary).sum(dim=1)
        increment = (growth-floor_growth).clamp_min(0.)
        allowed = (limits-floor_growth).clamp_min(0.)
        ratio = (allowed/increment.clamp_min(1e-30)).clamp_max(1.)
        vertex_ratio = torch.ones_like(radius)
        vertex_ratio.scatter_reduce_(0, point_ids.reshape(-1),
            ratio[:, None].expand(-1, 3).reshape(-1), reduce='amin', include_self=True)
        radius = lower + (radius-lower).clamp_min(0.)*vertex_ratio
    used = 0
    for iteration in range(iterations):
        local = radius[triangle_ids]
        slope = torch.linalg.vector_norm((gradient*local[..., None]).sum(dim=1), dim=1)
        if float(slope.max().item()) <= max_surface_slope+tolerance:
            break
        ratio = (max_surface_slope/slope.clamp_min(1e-30)).clamp_max(1.)
        minimum = local.min(dim=1).values
        contracted = minimum[:, None]+ratio[:, None]*(local-minimum[:, None])
        contracted = torch.maximum(contracted, lower[triangle_ids])
        radius.scatter_reduce_(0, triangle_ids.reshape(-1), contracted.reshape(-1),
                               reduce='amin', include_self=True)
        radius = torch.maximum(radius, lower)
        used = iteration+1
    result = radius.cpu().numpy()
    slope = np.linalg.norm(np.einsum('fkc,fk->fc', surface_gradient_operator(mesh), result[faces]), axis=1)
    excess = np.maximum((result[faces[ids]]*bary).sum(axis=1)-caps, 0.)
    changed = np.max(initial[faces]-result[faces], axis=1) > tolerance
    floor_violation = np.maximum(floor-result, 0.)
    stats = {'iterations':used, 'max_point_cap_excess_mm':float(excess.max()) if len(excess) else 0.,
             'max_surface_slope':float(slope.max()), 'surface_slope_limit':max_surface_slope,
             'max_radius_reduction_mm':float((initial-result).max()),
             'max_floor_violation_mm':float(floor_violation.max()) if len(floor_violation) else 0.,
             'floor_preserved': bool(floor_values is not None),
             'changed_area_ratio':float(mesh.area_faces[changed].sum()/mesh.area),
             'converged':bool(slope.max() <= max_surface_slope+tolerance
                              and (not len(excess) or excess.max() <= tolerance)
                              and (not len(floor_violation)
                                   or floor_violation.max() <= tolerance))}
    return result, stats


def solve_wall_offsets(mesh, thickness, opposite, selected, minimum, cap, width,
                       *, visibility=None, cost_multiplier=None,
                       iterations=600, penalty=100., device='cuda',
                       max_surface_slope=None, vertex_caps=None,
                       render_caps=None, render_weights=None, render_penalty=100.,
                       checkpoint_interval=0, render_loss='quadratic', linear_scale=.025,
                       face_caps=None, point_caps=None,
                       wall_targets=None, wall_importance=None,
                       wall_loss='quadratic', wall_linear_scale=.01, wall_witnesses=None,
                       initial_values=None):
    """Minimize displacement, surface gradients and wall-deficit residual.

    Each selected face and its normal-ray partner impose a linearized growth
    constraint on their mean vertex radii. A soft constraint allows the solver
    to report infeasible/unfinished growth instead of spreading a hard maximum.
    Actual thickness and visibility still require post-reconstruction checks.
    Diagonally preconditioned primal-dual iterations avoid a dense Hessian.
    Optional triangle gradient constraints bound radius variation in the
    optimization problem. Finite iterations need not satisfy them exactly;
    returned statistics expose the maximum slope and violating area.
    Per-face targets and positive importance weights can prioritize marginal
    walls without changing the minimum used for final thickness measurement.
    A linear deficit loss bounds the pressure of infeasible walls and can
    reach feasible targets exactly, unlike a squared soft deficit penalty.
    Optional measured wall witnesses add affine constraints at actual surface
    owners, with ray-incidence factors. They are local linearizations only.
    ``initial_values`` supplies a field that already passed a render review.
    In that mode the optimization solves only a nonnegative increment and
    evaluates every constraint on ``initial_values + increment``. This keeps
    a successful appearance field intact while measured wall feedback adds
    the smallest possible extra growth.
    """
    import torch
    faces = np.asarray(mesh.faces)
    thickness, opposite = np.asarray(thickness), np.asarray(opposite)
    selected = np.asarray(selected, dtype=bool)
    if any(a.shape != (len(faces),) for a in (thickness, opposite, selected)):
        raise ValueError('wall data must have one entry per face')
    ids = np.flatnonzero(selected)
    if np.any(~np.isfinite(thickness[ids])) or np.any((opposite[ids] < 0) | (opposite[ids] >= len(faces))):
        raise ValueError('selected walls require finite thickness and valid opposing faces')
    if not all(np.isfinite(v) and v > 0 for v in (minimum, cap, width, penalty)) or iterations < 1:
        raise ValueError('solver scales and iteration count must be positive')
    if (wall_loss not in ('quadratic', 'linear') or not np.isfinite(wall_linear_scale)
            or wall_linear_scale <= 0):
        raise ValueError('invalid wall loss or linear scale')
    targets = np.full(len(faces), minimum) if wall_targets is None else np.asarray(wall_targets, dtype=float)
    importance = np.ones(len(faces)) if wall_importance is None else np.asarray(wall_importance, dtype=float)
    if (targets.shape != thickness.shape or not np.isfinite(targets).all()
            or np.any(targets < minimum)):
        raise ValueError('wall targets must be finite face values at least the minimum')
    if (importance.shape != thickness.shape or not np.isfinite(importance).all()
            or np.any(importance <= 0)):
        raise ValueError('wall importance must be finite positive face values')
    if render_loss not in ('quadratic', 'linear') or not np.isfinite(linear_scale) or linear_scale <= 0:
        raise ValueError('invalid render loss or linear scale')
    if max_surface_slope is not None and (not np.isfinite(max_surface_slope)
                                          or not 0 < max_surface_slope < 1):
        raise ValueError('surface slope must be finite and between zero and one')
    n = len(mesh.vertices)
    caps = np.full(n, cap) if vertex_caps is None else np.asarray(vertex_caps, dtype=float)
    if caps.shape != (n,) or not np.isfinite(caps).all() or np.any((caps < 0) | (caps > cap)):
        raise ValueError('vertex caps must be finite, nonnegative and at most the global cap')
    base = np.zeros(n, dtype=np.float64) if initial_values is None else np.asarray(initial_values, dtype=np.float64)
    if (base.shape != (n,) or not np.isfinite(base).all() or np.any(base < 0)
            or np.any(base > caps + 1e-8)):
        raise ValueError('initial values must be finite, nonnegative and within vertex caps')
    # A projected field can differ from a cap by a few floating-point ulps;
    # do not turn that harmless residual into a negative increment bound.
    increment_caps = np.maximum(caps - base, 0.)
    witness_data = (validate_wall_witnesses(wall_witnesses, len(faces))
                    if wall_witnesses is not None else None)
    witness_count = len(witness_data[0]) if witness_data is not None else 0
    if not len(ids) and not witness_count:
        return base.copy(), {'iterations': 0, 'max_deficit_mm': 0.,
                             'initial_values': bool(initial_values is not None)}
    area = np.asarray(mesh.area_faces)
    exposure = np.ones(len(faces)) if visibility is None else np.asarray(visibility, dtype=float)
    if exposure.shape != area.shape or np.any(~np.isfinite(exposure)) or np.any(exposure < 0):
        raise ValueError('visibility must be finite nonnegative face values')
    positive = exposure[exposure > 0]
    exposure = .1 + exposure / (np.median(positive) if len(positive) else 1.)
    if cost_multiplier is not None:
        multiplier = np.asarray(cost_multiplier, dtype=float)
        if (multiplier.shape != area.shape or not np.isfinite(multiplier).all()
                or np.any(multiplier <= 0)):
            raise ValueError('cost multipliers must be finite positive face values')
        exposure *= multiplier
    mass = np.zeros(n)
    cost = np.zeros(n)
    for k in range(3):
        np.add.at(mass, faces[:, k], area / 3)
        np.add.at(cost, faces[:, k], area * exposure / 3)
    # Normalize objective weights without changing their relative importance.
    unit = np.median(mass[mass > 0])
    edges = np.asarray(mesh.edges_unique)
    lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    degree = np.bincount(edges.reshape(-1), minlength=n)
    edge_area = (mass[edges[:, 0]] / np.maximum(degree[edges[:, 0]], 1)
                 + mass[edges[:, 1]] / np.maximum(degree[edges[:, 1]], 1))
    coefficients = width * np.sqrt(edge_area / unit) / np.maximum(lengths, 1e-10)
    paired = np.column_stack((faces[ids], faces[opposite[ids]]))
    pair_coefficients = np.full(paired.shape, 1/3)
    gaps = np.maximum(targets[ids] - thickness[ids], 0.)
    wall_weights = penalty * importance[ids] * area[ids] / unit
    if witness_count:
        wi, wb, wt, wa, wf = witness_data
        paired = np.concatenate((paired, faces[wi].reshape(-1, 6)))
        pair_coefficients = np.concatenate((pair_coefficients, (wb*wf[..., None]).reshape(-1, 6)))
        gaps = np.r_[gaps, wt]
        wall_weights = np.r_[wall_weights, penalty*wa/unit]

    tensor = lambda a, dtype=torch.float32: torch.tensor(a, dtype=dtype, device=device)
    # The primal variable is an increment. All geometric constraints below
    # add the fixed, already-reviewed base field before measuring growth.
    base_tensor = tensor(base)
    upper = tensor(increment_caps)
    if point_caps is not None:
        point_ids, point_bary, point_upper = validate_point_caps(point_caps, len(faces))
        point_faces = tensor(faces[point_ids], torch.long)
        point_weights, point_limits = tensor(point_bary), tensor(point_upper)
        point_dual = torch.zeros_like(point_limits)
    if face_caps is not None:
        fc = np.asarray(face_caps, dtype=float)
        if fc.shape != (len(faces),) or not np.isfinite(fc).all() or np.any(fc < 0):
            raise ValueError('face caps must be finite nonnegative face means')
        bounded = np.flatnonzero(fc < cap)
        bounded_faces = tensor(faces[bounded], torch.long)
        mean_upper = tensor(fc[bounded])
        cap_dual = torch.zeros_like(mean_upper)
    if (render_caps is None) != (render_weights is None):
        raise ValueError('render caps and weights must be supplied together')
    if render_caps is not None:
        rc, rw = np.asarray(render_caps), np.asarray(render_weights)
        if (rc.ndim != 2 or rc.shape != rw.shape or len(rc) != len(faces)
                or not np.isfinite(rc).all() or not np.isfinite(rw).all()
                or np.any(rc < 0) or np.any(rw < 0)
                or not np.isfinite(render_penalty) or render_penalty <= 0):
            raise ValueError('invalid projected render constraints')
        render_ids, views = np.nonzero((rw > 0) & (rc < cap))
        render_faces = tensor(faces[render_ids], torch.long)
        render_limits = tensor(rc[render_ids, views])
        render_cost = tensor(render_penalty * area[render_ids]/unit
                             * rw[render_ids, views]/rc.shape[1])
        render_dual = torch.zeros_like(render_limits)
    left, right = tensor(edges[:, 0], torch.long), tensor(edges[:, 1], torch.long)
    c, cost = tensor(coefficients), tensor(cost / unit)
    pair, gaps, weights = tensor(paired, torch.long), tensor(gaps), tensor(wall_weights)
    pair_coefficients = tensor(pair_coefficients)
    # K consists of weighted edge differences and six-vertex wall averages.
    column_sum = torch.zeros(n, device=device)
    column_sum.index_add_(0, left, c)
    column_sum.index_add_(0, right, c)
    column_sum.index_add_(0, pair.reshape(-1), pair_coefficients.reshape(-1))
    if point_caps is not None:
        column_sum.index_add_(0, point_faces.reshape(-1), point_weights.reshape(-1))
    if face_caps is not None:
        column_sum.index_add_(0, bounded_faces.reshape(-1),
                             torch.full((bounded_faces.numel(),), 1/3, device=device))
    if render_caps is not None:
        column_sum.index_add_(0, render_faces.reshape(-1),
                             torch.full((render_faces.numel(),), 1/3, device=device))
    if max_surface_slope is not None:
        # Full triangle gradients, rather than edge differences, also constrain
        # narrow triangles. The block dual proximal is vector soft thresholding
        # for the support function of a ball of radius max_surface_slope.
        surface = tensor(surface_gradient_operator(mesh))
        face_ids = tensor(faces, torch.long)
        norms = torch.linalg.vector_norm(surface, dim=2)
        column_sum.index_add_(0, face_ids.reshape(-1), norms.reshape(-1))
        sigma_surface = .9 / norms.sum(dim=1).clamp_min(1e-12)
        surface_limit = torch.full((len(faces),), float(max_surface_slope), device=device)
        if initial_values is not None:
            base_surface = torch.linalg.vector_norm(
                (surface * base_tensor[face_ids, None]).sum(dim=1), dim=1)
            # Triangle inequality gives a conservative remaining budget for
            # the increment while preserving the absolute full-field bound.
            surface_limit = (surface_limit - base_surface).clamp_min(0.)
        surface_dual = torch.zeros((len(faces), 3), device=device)
    tau = .9 / column_sum.clamp_min(1e-12)
    sigma_edge = .9 / (2 * c).clamp_min(1e-12)
    sigma_wall = .9/pair_coefficients.sum(dim=1)
    # The primal variable below is only the additional growth.  Keeping it
    # separate from ``base_tensor`` means the displacement objective does not
    # charge again for a field that has already passed render review.
    radius = torch.zeros(n, device=device)
    extrapolated = radius.clone()
    edge_dual = torch.zeros_like(c)
    wall_dual = torch.zeros_like(gaps)
    gradient = torch.empty_like(radius)
    history = []
    previous_checkpoint = radius.clone() if checkpoint_interval else None
    for iteration in range(iterations):
        total_extrapolated = base_tensor + extrapolated
        # A reviewed base may legitimately contain a nonzero gradient.  Do
        # not equalize it by adding growth to its low side; regularize only
        # the new increment and let the final full-field projection enforce
        # the absolute slope bound.
        edge_dual = (edge_dual + sigma_edge * c *
                     (extrapolated[left] - extrapolated[right])) / (1 + sigma_edge)
        wall_growth = (total_extrapolated[pair]*pair_coefficients).sum(dim=1)
        proposed_wall_dual = wall_dual + sigma_wall * (wall_growth - gaps)
        if wall_loss == 'linear':
            # Conjugate of lambda*max(gap-growth, 0) restricts the dual to
            # [-lambda, 0]. Large unresolved deficits cannot dominate small
            # feasible ones solely because their residual is larger.
            wall_dual = torch.maximum(proposed_wall_dual.clamp_max(0),
                                      -weights*wall_linear_scale)
        else:
            wall_dual = (proposed_wall_dual
                         / (1 + sigma_wall / weights.clamp_min(1e-12))).clamp_max(0)
        gradient.zero_()
        gradient.index_add_(0, left, c * edge_dual)
        gradient.index_add_(0, right, -c * edge_dual)
        gradient.index_add_(0, pair.reshape(-1), (wall_dual[:, None]*pair_coefficients).reshape(-1))
        if point_caps is not None:
            point_growth = ((base_tensor[point_faces] + extrapolated[point_faces])
                            * point_weights).sum(dim=1)
            point_dual = (point_dual+.9*(point_growth-point_limits)).clamp_min(0)
            gradient.index_add_(0, point_faces.reshape(-1), (point_dual[:, None]*point_weights).reshape(-1))
        if face_caps is not None:
            cap_dual = (cap_dual + .9*((base_tensor[bounded_faces] + extrapolated[bounded_faces]).mean(dim=1)
                                       -mean_upper)).clamp_min(0)
            gradient.index_add_(0, bounded_faces.reshape(-1),
                               (cap_dual[:, None].expand(-1, 3)/3).reshape(-1))
        if render_caps is not None:
            growth = (base_tensor[render_faces] + extrapolated[render_faces]).mean(dim=1)
            proposed_dual = render_dual + .9*(growth-render_limits)
            if render_loss == 'linear':
                # Conjugate of lambda * positive_part(x-cap) is the
                # interval indicator 0 <= dual <= lambda, plus cap*dual.
                render_dual = torch.minimum(proposed_dual.clamp_min(0), render_cost*linear_scale)
            else:
                render_dual = (proposed_dual / (1 + .9/render_cost.clamp_min(1e-12))).clamp_min(0)
            gradient.index_add_(0, render_faces.reshape(-1),
                               (render_dual[:, None].expand(-1, 3)/3).reshape(-1))
        if max_surface_slope is not None:
            value = (surface * extrapolated[face_ids, None]).sum(dim=1)
            surface_dual += sigma_surface[:, None] * value
            magnitude = torch.linalg.vector_norm(surface_dual, dim=1)
            shrink = (1 - sigma_surface * surface_limit / magnitude.clamp_min(1e-30)).clamp_min(0)
            surface_dual *= shrink[:, None]
            adjoint = (surface * surface_dual[:, None]).sum(dim=2)
            gradient.index_add_(0, face_ids.reshape(-1), adjoint.reshape(-1))
        updated = torch.minimum(((radius - tau * gradient) / (1 + tau * cost)).clamp_min(0), upper)
        extrapolated = 2 * updated - radius
        radius = updated
        if checkpoint_interval and ((iteration+1) % checkpoint_interval == 0 or iteration+1 == iterations):
            checkpoint = {'iteration':iteration+1,
                          'max_radius_change_mm':float((radius-previous_checkpoint).abs().max().item()),
                          'mean_deficit_mm':float(((gaps-((base_tensor[pair] + radius[pair])*pair_coefficients).sum(dim=1)).clamp_min(0)
                                                  * weights).sum().item()/weights.sum().item())}
            if max_surface_slope is not None:
                measured = torch.linalg.vector_norm(
                    (surface*(base_tensor[face_ids, None] + radius[face_ids, None])).sum(dim=1), dim=1)
                checkpoint['max_surface_slope'] = float(measured.max().item())
            if face_caps is not None:
                checkpoint['max_face_cap_excess_mm'] = float(
                    ((base_tensor[bounded_faces] + radius[bounded_faces]).mean(dim=1)-mean_upper)
                    .clamp_min(0).max().item()) if len(bounded) else 0.
            if point_caps is not None:
                checkpoint['max_point_cap_excess_mm'] = float(
                    (((base_tensor[point_faces] + radius[point_faces])*point_weights).sum(dim=1)-point_limits)
                    .clamp_min(0).max().item()
                ) if len(point_ids) else 0.
            history.append(checkpoint)
            logging.getLogger(__name__).info('Field solver checkpoint: %s', checkpoint)
            previous_checkpoint.copy_(radius)
    residual = (gaps - ((base_tensor[pair] + radius[pair])*pair_coefficients).sum(dim=1)).clamp_min(0)
    result = (base_tensor + radius).detach().cpu().numpy().astype(np.float64)
    stats = {'iterations': iterations, 'penalty': penalty, 'width_mm': width,
             'wall_loss': wall_loss,
             'wall_linear_scale_mm': float(wall_linear_scale) if wall_loss == 'linear' else None,
             'measured_wall_witnesses': witness_count,
             'initial_values': bool(initial_values is not None),
             'max_radius_mm': float(result.max()),
             'max_deficit_mm': float(residual.max().item()),
             'mean_deficit_mm': float((residual * weights).sum().item() / weights.sum().item())}
    if max_surface_slope is not None:
        slope = torch.linalg.vector_norm(
            (surface * (base_tensor[face_ids, None] + radius[face_ids, None])).sum(dim=1), dim=1)
        measured = slope.cpu().numpy()
        stats.update(surface_slope_limit=float(max_surface_slope),
                     max_surface_slope=float(measured.max()),
                     slope_violation_area_ratio=float(area[measured > max_surface_slope + 1e-3].sum() / area.sum()))
    if render_caps is not None:
        excess = ((base_tensor[render_faces] + radius[render_faces]).mean(dim=1)-render_limits).clamp_min(0)
        stats.update(render_penalty=float(render_penalty), render_loss=render_loss,
                     linear_scale_mm=float(linear_scale) if render_loss == 'linear' else None,
                     render_constraints=len(render_ids),
                     render_weighted_excess_mm=float((excess*render_cost).sum().item()
                         / max(render_cost.sum().item(), 1e-30)))
    if checkpoint_interval:
        stats['checkpoints'] = history
    if face_caps is not None:
        excess = np.maximum(result[faces].mean(axis=1)-fc, 0.)
        stats.update(max_face_cap_excess_mm=float(excess.max()),
                     face_cap_violation_area_ratio=float(area[excess > 1e-4].sum()/area.sum()))
    if point_caps is not None:
        excess = np.maximum((result[faces[point_ids]]*point_bary).sum(axis=1)-point_upper, 0.)
        stats.update(point_caps=len(point_ids), max_point_cap_excess_mm=float(excess.max()) if len(excess) else 0.,
                     point_cap_violation_fraction=float((excess > 1e-4).mean()) if len(excess) else 0.)
    return result, stats
