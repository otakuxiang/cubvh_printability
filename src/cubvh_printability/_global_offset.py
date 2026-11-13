"""Area-weighted global wall-growth planning; all lengths are millimeters."""
import numpy as np


def plan_global_offset(profile, minimum, threshold, voxel_size, cap):
    """Choose a gap quantile, reserving unknown area and modest safety margins."""
    face = profile['face_profile']
    thickness = np.asarray(face['thickness'], dtype=float)
    areas = np.asarray(face['areas'], dtype=float)
    if (thickness.shape != areas.shape or thickness.ndim != 1
            or not np.all(np.isfinite(areas)) or np.any(areas < 0) or areas.sum() <= 0):
        raise ValueError('global profile requires finite nonnegative face areas')
    unknown = ~np.isfinite(thickness) | (thickness < 0) | np.asarray(face['unknown'])
    total = float(areas.sum())
    reserved = float(areas[unknown].sum() / total)
    report = {'method': 'area_weighted_gap_quantile', 'unknown_area_ratio': reserved}
    if not profile['reliable'] or reserved >= threshold:
        return None, {**report, 'reason': 'unreliable_global_profile'}
    # Aim 10% below the allowed area, unless unknown faces consume that margin.
    target = max(.9 * threshold, .5 * (reserved + threshold))
    known = ~unknown & (areas > 0)
    gaps = np.maximum(minimum - thickness[known], 0.)
    weights = areas[known]
    order = np.argsort(gaps, kind='stable')
    gaps, weights = gaps[order], weights[order]
    required_area = (1 - target) * total
    index = min(int(np.searchsorted(np.cumsum(weights), required_area, side='right')),
                len(gaps) - 1)
    gap = float(gaps[index])
    base = .5 * gap  # Both opposing surfaces expand.
    # 10% radial slack or a quarter nominal voxel, at most 5% of minimum.
    margin = min(max(.1 * base, .25 * voxel_size), .05 * minimum)
    requested = base + margin
    offset = min(requested, cap)
    predicted = reserved + float(weights[gaps > 2 * offset].sum() / total)
    report.update(target_thin_area_ratio=target, weighted_gap_mm=gap,
                  area_weighted_mean_gap_mm=float(np.dot(gaps, weights) / total),
                  base_offset_mm=base, margin_mm=margin, requested_offset_mm=requested,
                  offset_mm=offset, cap_limited=requested > cap,
                  predicted_thin_area_ratio=predicted)
    return offset, report
