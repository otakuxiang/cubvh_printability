"""Automatic printability checking and repair. Lengths are in millimetres."""
import argparse
import json
import logging
import math
import numbers
import time
from pathlib import Path

import numpy as np

from ._printability_geometry import export_mesh, load_mesh, measurable, resize, topology

_LOG = logging.getLogger(__name__)


class PrintabilityDependencyError(ImportError):
    """A required GPU dependency is unavailable; this is not a geometry failure."""


def _measure(mesh, min_thickness):
    from .build_optix import build
    build()
    from ._printability_thickness import measure
    return measure(mesh, min_thickness)


def _repair_backend(mesh, resolution):
    from ._printability_reconstruct import Reconstructor
    return Reconstructor(mesh, resolution)


def _prepare_geometry(mesh):
    """Use CUDA for large meshes when the optional geometry backend exists."""
    if len(mesh.faces) < 250000:
        return
    try:
        import torch
        from ._gpu_geometry import prepare_geometry
        import cumesh
    except ImportError:
        return
    if torch.cuda.is_available():
        prepare_geometry(mesh)


def _evaluate(mesh, min_thickness, threshold, diagnostics=None):
    _prepare_geometry(mesh)
    stats = topology(mesh)
    if diagnostics is not None:
        diagnostics['topology'] = stats
    if not measurable(mesh, stats):
        if diagnostics is not None:
            diagnostics['reason'] = 'invalid_topology'
        _LOG.info('Topology prevents thickness measurement: %s', stats)
        return None
    _LOG.info('Measuring material thickness: %d faces', len(mesh.faces))
    measurement = _measure(mesh, min_thickness)
    _LOG.info('Thickness result: %s', measurement)
    if diagnostics is not None:
        diagnostics['measurement'] = measurement
        diagnostics['reason'] = ('unreliable_measurement' if not measurement['reliable'] else
                                 'thin_wall_ratio_exceeded' if measurement['thin_wall_area_ratio'] >= threshold else 'accepted')
    if not measurement['reliable'] or measurement['thin_wall_area_ratio'] >= threshold:
        return None
    return {**stats, 'volume': measurement['volume'],
            'thin_wall_area_ratio': measurement['thin_wall_area_ratio'],
            'thin_wall_area': measurement['thin_wall_area_ratio'] * stats['surface_area']}


def _profile(mesh, min_thickness):
    from ._printability_thickness import measure_profile
    return measure_profile(mesh, min_thickness, diagnostics=True)


def _evaluate_reconstruction(backend, candidate, target_faces, min_thickness,
                             threshold, entry):
    """Validate a reduced output first, falling back to the raw reconstruction."""
    if candidate is None:
        entry['reason'] = 'empty_reconstruction'
        return candidate, None
    options = entry.setdefault('options', [])
    if len(candidate.faces) > target_faces:
        started = time.perf_counter()
        reduced = backend.simplify(candidate, target_faces)
        entry['simplification_seconds'] = time.perf_counter() - started
        entry['simplification_target_faces'] = target_faces
        option = {'kind': 'simplified', 'num_faces': len(reduced.faces)}
        options.append(option)
        started = time.perf_counter()
        stats = _evaluate(reduced, min_thickness, threshold, option)
        option['measurement_seconds'] = time.perf_counter() - started
        if stats is not None:
            entry.update(reason='accepted', selected_kind='simplified')
            entry['measurement'] = option.get('measurement')
            return reduced, stats
        entry['simplification_fallback'] = option.get('reason', 'printability_check_failed')
    option = {'kind': 'raw', 'num_faces': len(candidate.faces)}
    options.append(option)
    started = time.perf_counter()
    stats = _evaluate(candidate, min_thickness, threshold, option)
    option['measurement_seconds'] = time.perf_counter() - started
    entry['reason'] = option.get('reason', 'accepted' if stats is not None else 'printability_check_failed')
    entry['measurement'] = option.get('measurement')
    if stats is not None:
        entry['selected_kind'] = 'raw'
    return candidate, stats


def _profile_summary(profile, min_thickness):
    face = profile['face_profile']
    thickness = face['thickness']
    gap = face['gap']
    finite = np.isfinite(thickness)
    values = thickness[finite]
    gaps = gap[finite]
    percentile = lambda values, q: float(np.percentile(values, q)) if len(values) else None
    return {'reliable': bool(profile['reliable']),
            'reliability_details': profile.get('reliability_details'),
            'num_faces': int(len(thickness)), 'measured_faces': int(finite.sum()),
            'unknown_faces': int((~finite).sum()), 'thin_faces': int(face['thin'].sum()),
            'thin_area_ratio': float(face['thin_area_ratio']),
            'unknown_area_ratio': float(face['unknown_area_ratio']),
            'thickness_mm': {'min': float(values.min()) if len(values) else None,
                             'mean': float(values.mean()) if len(values) else None,
                             'p10': percentile(values, 10), 'p90': percentile(values, 90)},
            'gap_mm': {'max': float(gaps.max()) if len(gaps) else None,
                       'mean': float(gaps.mean()) if len(gaps) else None,
                       'p90': percentile(gaps, 90)},
            'sample_count': int(face['sample_count']), 'threshold': float(min_thickness)}


def _check_parts(original, mesh_path, min_thickness, threshold, size,
                 fix_printability, resolution, output_path, max_iso_scale,
                 max_geometry_change_ratio, min_silhouette_iou):
    """Select measured thin patches, reconstruct, and validate in the input frame."""
    from ._local_sdf import (select_offsets, allocate_paired_offsets,
                             smooth_vertex_offsets)
    history, initial = [], {}
    reference = original
    current = original
    bootstrap_offset = 0.
    total_cap = .5 * min_thickness * max_iso_scale
    profile_stats = None
    # Camera-independent local repair: select faces by measured thickness and
    # area only.  Geometry is bounded by the radius implied by max_iso_scale.
    selection_priority = 'minimum_displacement'

    def finish(mesh, stats, reason, geometry=None, offset=None):
        return {
            'printable': stats is not None,
            **(stats if stats is not None else {**topology(mesh), 'volume': None,
                'thin_wall_area': None, 'thin_wall_area_ratio': None}),
            'output_mesh_path': export_mesh(mesh, mesh_path, output_path) if fix_printability else None,
            'repair_attempts': len(history), 'repair_offset': offset,
            'iso_scale': 2 * offset / min_thickness if offset is not None else None,
            'max_iso_scale': max_iso_scale, 'geometry_constraint': 'max_iso_scale',
            'iteration_policy': 'thickness_only',
            'max_geometry_change_ratio': max_geometry_change_ratio,
            'min_silhouette_iou': min_silhouette_iou, 'geometry_metrics': geometry,
            'repair_history': history, 'initial_diagnostics': initial,
            'failure_reason': reason, 'num_faces': len(mesh.faces),
            'face_count_deviation': len(mesh.faces) / len(reference.faces) - 1,
            'repair_strategy': 'parts',
            'local_profile': profile_stats,
            'local_selection_priority': selection_priority,
        }

    def fallback(reason):
        _LOG.warning('Local SDF fallback to global repair: %s', reason)
        result = _check_global(reference, mesh_path, min_thickness, threshold, size,
            fix_printability, resolution, output_path, max_iso_scale,
            max_geometry_change_ratio,
            min_silhouette_iou)
        result['repair_strategy'] = 'global_fallback'
        result['local_fallback_reason'] = reason
        result['local_profile'] = profile_stats
        result['repair_history'] = history + result['repair_history']
        result['repair_attempts'] += len(history)
        return result

    if threshold == 0:
        return finish(reference, None, 'zero_thin_wall_threshold')
    initial_stats = _evaluate(reference, min_thickness, threshold, initial)
    if initial_stats is not None:
        return finish(reference, initial_stats, None)
    if not measurable(reference):
        # Bootstrap topology at normalized radius 1 / resolution; the backend
        # takes millimeters. Leave thickness-driven growth to local repair.
        offset = size / resolution
        if offset > .5 * min_thickness * max_iso_scale:
            return finish(reference, None, 'max_iso_scale_exceeded')
        bootstrap_backend = _repair_backend(reference, resolution)
        current = bootstrap_backend.reconstruct(offset)
        history.append({'kind': 'bootstrap', 'offset_mm': offset})
        current, stats = _evaluate_reconstruction(bootstrap_backend, current, len(reference.faces),
            min_thickness, threshold, history[-1])
        if current is None or not measurable(current):
            return fallback('bootstrap_reconstruction_failed')
        bootstrap_offset = offset
        history[-1]['geometry'] = {
            'passed': True, 'method': 'max_iso_scale',
            'iso_scale': 2 * offset / min_thickness,
            'max_iso_scale': max_iso_scale,
            'views': [], 'render_evaluated': False,
        }
        if stats is not None:
            return finish(current, stats, None, history[-1]['geometry'], offset)
        if bootstrap_offset >= total_cap:
            return finish(reference, None, 'max_iso_scale_exceeded')

    profile = _profile(current, min_thickness)
    profile_stats = _profile_summary(profile, min_thickness)
    details = profile.get('reliability_details') or {}
    # Local ray spans do not require a globally consistent signed-shell
    # volume. A few resolved material-side reversals can invalidate that
    # volume on intersecting input shells without invalidating every local
    # query. Permit a proposal only for this diagnosed failure; unknown
    # thickness still consumes the area budget, SDF parity must resolve,
    # and _evaluate requires a reliable volume on the reconstructed output.
    volume_only_failure = (
        details.get('orientation_mismatch_samples', 0) > 0
        and details.get('unknown_component_orientations') == 0
        and details.get('positive_finite_volume') is True
        and details.get('measured_uniform_samples', 0) > 0)
    profile_stats['planning_with_uncertain_volume'] = bool(
        not profile['reliable'] and volume_only_failure)
    _LOG.info('Local thickness profile: %s', profile_stats)
    if not profile['reliable'] and not volume_only_failure:
        return fallback('unreliable_local_measurement')
    face = profile['face_profile']
    areas = np.asarray(face.get('areas', current.area_faces))
    known = np.asarray(face['thickness']).copy()
    unknown = ~np.isfinite(known) | np.asarray(face['unknown'])
    if float(areas[unknown].sum()/areas.sum()) >= threshold:
        return fallback('unknown_face_thickness')
    opposite = np.asarray(face['opposite_faces'])
    unpaired = (known < min_thickness) & ((opposite < 0) | (opposite >= len(known)))
    unresolved = unknown | unpaired
    reserved = float(areas[unresolved].sum()/areas.sum())
    profile_stats['reserved_unresolved_area_ratio'] = reserved
    if reserved >= threshold:
        return fallback('unresolved_wall_area_exceeds_budget')
    # These faces remain unrepaired and spend the thin-area budget. Replace
    # their selector values only after subtracting their full area; the final
    # unbiased measurement still counts unknown samples as thin.
    known[unresolved] = min_thickness
    cap = total_cap - bootstrap_offset
    selector_args = (known, areas, min_thickness, threshold-reserved,
                     cap)
    offsets, selected = select_offsets(*selector_args, priority=selection_priority)
    if not selected.any():
        return finish(reference, None, 'profile_disagrees_with_area_measurement')
    # Split paired walls evenly; this avoids the former multi-view exposure
    # pass while preserving the requested total wall growth.
    offsets = allocate_paired_offsets(
        offsets, face['opposite_faces'], np.ones(len(offsets), dtype=float),
        cap)
    local_resolution = int(resolution)
    backend = _repair_backend(current, local_resolution)
    # Retain the original material SDF and union it with variable-radius
    # surface balls. No render feedback is needed in this fast path.
    backend.local_weighted_distance = True
    backend.local_prune_envelope = True
    backend.local_seed_bound = True
    reason = 'printability_check_failed'
    previous_values = None

    for strength in (1., 1.25, 1.5):
        radii = np.minimum(offsets * strength, cap)
        maximum = float(radii.max())
        step_mm = 2 * (.5 * size + maximum) / max(1, local_resolution - 6)
        values = smooth_vertex_offsets(current, radii, max(2 * step_mm, 4 * maximum))
        values = np.clip(values, 0., cap)
        if previous_values is not None:
            values = np.maximum(values, previous_values)
            if np.allclose(values, previous_values, rtol=0., atol=1e-12):
                reason = 'local_field_stalled'
                break
        previous_values = values.copy()
        backend.local_vertex_offsets = values
        radii = values[np.asarray(current.faces)].max(axis=1)
        maximum = float(radii.max())
        entry = {'kind': 'local_sdf', 'strength': strength, 'offset_mm': maximum,
                 'iso_scale': 2 * (bootstrap_offset + maximum) / min_thickness,
                 'cumulative_offset_mm': bootstrap_offset + maximum,
                 'geometry_method': 'max_iso_scale'}
        history.append(entry)
        _LOG.info('Local attempt %d: strength=%g radius=%g mm iso_scale=%g',
                  len(history), strength, maximum, entry['iso_scale'])
        try:
            started = time.perf_counter()
            candidate = backend.reconstruct(maximum, face_offsets=radii)
            entry['reconstruction_seconds'] = time.perf_counter() - started
            entry['reconstruction'] = getattr(backend, 'local_reconstruction_stats', None)
        except ValueError as exc:
            entry['reason'] = str(exc)
            return finish(reference, None, 'local_field_unreliable')
        started = time.perf_counter()
        candidate, stats = _evaluate_reconstruction(backend, candidate, len(reference.faces),
            min_thickness, threshold, entry)
        entry['validation_seconds'] = time.perf_counter() - started
        entry['thickness_passed'] = stats is not None
        entry['geometry'] = {
            'passed': entry['iso_scale'] <= max_iso_scale * (1 + 1e-12),
            'method': 'max_iso_scale', 'iso_scale': entry['iso_scale'],
            'max_iso_scale': max_iso_scale,
            'views': [], 'render_evaluated': False,
        }
        if stats is not None:
            entry['reason'] = 'accepted'
            return finish(candidate, stats, None, entry['geometry'], bootstrap_offset + maximum)
        reason = entry.get('reason', 'printability_check_failed')
        del candidate
    return finish(reference, None, reason,
                  history[-1].get('geometry') if history else None)


def check_printability(mesh_path: str, min_thickness: float,
                       thin_wall_area_ratio_threshold: float, size: float,
                       fix_printability: bool, *, resolution: int = 1536,
                       output_path: str | None = None,
                       max_iso_scale: float = 3.0,
                       max_geometry_change_ratio: float | None = None,
                       min_silhouette_iou: float = 0.98,
                       repair_strategy: str = 'global') -> dict:
    """Repair thin walls without view-dependent iteration; lengths are millimetres.

    iso_scale = cumulative_offset / (min_thickness / 2). Global proposals
    start from the original mesh or its one-voxel bootstrap. Stop at the first candidate that passes the thickness-area
    criterion. Geometry growth is bounded by ``max_iso_scale``. The legacy
    view-related arguments are accepted for configuration compatibility but do
    not trigger rendering or affect iteration.
    """
    for name, value in [('min_thickness', min_thickness), ('size', size)]:
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a positive finite number')
    threshold = thin_wall_area_ratio_threshold
    for name, value in [('thin_wall_area_ratio_threshold', threshold),
                        ('min_silhouette_iou', min_silhouette_iou)]:
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be in [0, 1]')
    if (isinstance(max_iso_scale, bool)
            or not isinstance(max_iso_scale, numbers.Real)
            or not math.isfinite(max_iso_scale) or max_iso_scale <= 0):
        raise ValueError('max_iso_scale must be positive and finite')
    if max_geometry_change_ratio is not None and (isinstance(max_geometry_change_ratio, bool)
            or not isinstance(max_geometry_change_ratio, numbers.Real)
            or not math.isfinite(max_geometry_change_ratio) or max_geometry_change_ratio <= 0):
        raise ValueError('max_geometry_change_ratio must be positive and finite, or None')
    if not isinstance(fix_printability, bool):
        raise TypeError('fix_printability must be bool')
    if isinstance(resolution, bool) or not isinstance(resolution, numbers.Integral) or not 8 <= resolution <= 4095:
        raise ValueError('resolution must be an integer in [8, 4095]')
    if repair_strategy not in ('global', 'parts'):
        raise ValueError("repair_strategy must be 'global' or 'parts'")
    if fix_printability and output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        source = Path(mesh_path).expanduser().resolve()
        if destination == source or (destination.exists() and source.exists() and destination.samefile(source)):
            raise ValueError('output_path must not overwrite the input')
        if destination.suffix.lower() != '.ply':
            raise ValueError('output_path must end in .ply')
    original = load_mesh(mesh_path, size)
    _prepare_geometry(original)
    original_stats = topology(original)
    _LOG.info('Loaded %s: %d faces, %s', mesh_path, len(original.faces), original_stats)
    if repair_strategy == 'parts':
        return _check_parts(original, mesh_path, min_thickness, threshold, size,
                            fix_printability, int(resolution), output_path,
                            max_iso_scale, max_geometry_change_ratio,
                            min_silhouette_iou)
    return _check_global(original, mesh_path, min_thickness, threshold, size,
                         fix_printability, resolution, output_path, max_iso_scale,
                         max_geometry_change_ratio, min_silhouette_iou)


def _check_global(original, mesh_path, min_thickness, threshold, size,
                  fix_printability, resolution, output_path, max_iso_scale,
                  max_geometry_change_ratio, min_silhouette_iou, *, local_review=False):
    """Run global repair with thickness acceptance and an iso-scale budget."""
    original_stats = topology(original)
    final_mesh = original
    stats = None
    attempts = 0
    selected_offset = None
    selected_geometry = None
    history = []
    initial_diagnostics = {}
    stop_reason = 'zero_thin_wall_threshold' if threshold == 0 else None
    from ._global_offset import plan_global_offset
    profile_report = None
    if threshold > 0:
        stats = _evaluate(original, min_thickness, threshold, initial_diagnostics)
        if stats is None:
            source = original
            bootstrap_offset = 0.
            cap = .5 * min_thickness * max_iso_scale
            voxel_size = size / resolution
            profile = None
            needs_bootstrap = (not measurable(original) or
                initial_diagnostics.get('measurement', {}).get('reliable') is False)
            if not needs_bootstrap:
                profile = _profile(source, min_thickness)
                _, profile_report = plan_global_offset(
                    profile, min_thickness, threshold, voxel_size, cap)
                needs_bootstrap = profile_report.get('reason') is not None
            stop_reason = 'max_iso_scale_exceeded'
            if needs_bootstrap and voxel_size <= cap:
                profile = None
                bootstrap_offset = voxel_size
                backend = _repair_backend(original, int(resolution))
                entry = {'kind': 'bootstrap', 'source': 'input_mesh',
                         'offset_mm': bootstrap_offset,
                         'iso_scale': 2 * bootstrap_offset / min_thickness}
                history.append(entry)
                source, evaluated = _evaluate_reconstruction(backend,
                    backend.reconstruct(bootstrap_offset), len(original.faces),
                    min_thickness, threshold, entry)
                if evaluated is not None:
                    final_mesh, stats = source, evaluated
                    selected_offset = bootstrap_offset
                elif source is None or not measurable(source):
                    stop_reason = 'bootstrap_reconstruction_failed'
                else:
                    profile = _profile(source, min_thickness)
            if stats is None and profile is not None and (not needs_bootstrap or bootstrap_offset):
                offset, profile_report = plan_global_offset(profile, min_thickness,
                    threshold, voxel_size, max(0., cap - bootstrap_offset))
                if offset is None:
                    stop_reason = profile_report['reason']
                elif offset <= 0:
                    stop_reason = 'max_iso_scale_exceeded'
                else:
                    backend = _repair_backend(source, int(resolution))
                    # Reuse a fixed input/bootstrap field, never the last candidate.
                    for attempt in range(2):
                        cumulative = bootstrap_offset + offset
                        entry = {'kind': 'global', 'attempt': attempt + 1,
                                 'offset_mm': offset, 'cumulative_offset_mm': cumulative,
                                 'iso_scale': 2 * cumulative / min_thickness,
                                 'source': 'bootstrap_mesh' if bootstrap_offset else 'input_mesh',
                                 'offset_plan': profile_report}
                        history.append(entry)
                        _LOG.info('Global area-based attempt %d: offset=%g mm cumulative=%g mm',
                                  attempt + 1, offset, cumulative)
                        candidate, evaluated = _evaluate_reconstruction(backend,
                            backend.reconstruct(offset), len(original.faces),
                            min_thickness, threshold, entry)
                        if evaluated is not None:
                            final_mesh, stats = candidate, evaluated
                            selected_offset = cumulative
                            break
                        stop_reason = entry['reason']
                        # A single modest retry; the cap remains cumulative.
                        increased = min(offset + max(.2 * offset, .25 * voxel_size),
                                        cap - bootstrap_offset)
                        if increased <= offset * (1 + 1e-12):
                            break
                        offset = increased
            if stats is not None:
                stop_reason = None
                selected_geometry = {
                    'passed': True, 'method': 'max_iso_scale',
                    'iso_scale': 2 * selected_offset / min_thickness,
                    'max_iso_scale': max_iso_scale, 'views': [], 'render_evaluated': False}
                history[-1]['geometry'] = selected_geometry
    attempts = len(history)
    printable = stats is not None
    if not printable:
        stats = {**original_stats, 'volume': None, 'thin_wall_area': None,
                 'thin_wall_area_ratio': None}
    result = {'printable': printable, **stats, 'output_mesh_path': None,
              'repair_attempts': attempts, 'repair_offset': selected_offset,
              'iso_scale': 2*selected_offset/min_thickness if selected_offset is not None else None,
              'max_iso_scale': max_iso_scale,
              'geometry_constraint': 'max_iso_scale',
              'iteration_policy': 'thickness_only',
              'max_geometry_change_ratio': max_geometry_change_ratio,
              'min_silhouette_iou': min_silhouette_iou,
              'geometry_metrics': selected_geometry,
              'repair_history': history,
              'initial_diagnostics': initial_diagnostics,
              'failure_reason': None if printable else stop_reason,
              'repair_strategy': 'global',
              'global_offset_plan': profile_report,
              'num_faces': len(final_mesh.faces),
              'face_count_deviation': len(final_mesh.faces) / len(original.faces) - 1}
    if fix_printability:
        result['output_mesh_path'] = export_mesh(final_mesh, mesh_path, output_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh_path')
    parser.add_argument('--min-thickness', type=float, required=True)
    parser.add_argument('--thin-wall-area-ratio-threshold', type=float, required=True)
    parser.add_argument('--size', type=float, required=True)
    parser.add_argument('--fix-printability', action='store_true', help='export the resulting mesh; repair is always automatic')
    parser.add_argument('--resolution', type=int, default=1536)
    parser.add_argument('--output-path')
    parser.add_argument('--max-iso-scale', type=float, default=3.0)
    parser.add_argument('--max-geometry-change-ratio', type=float, default=None,
                    help='legacy compatibility option; view checks are disabled')
    parser.add_argument('--min-silhouette-iou', type=float, default=.98)
    parser.add_argument('--repair-strategy', choices=('global', 'parts'), default='global')
    args = parser.parse_args()
    print(json.dumps(check_printability(**vars(args)), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
