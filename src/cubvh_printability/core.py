"""Automatic printability checking and repair. Lengths are in millimetres."""
import argparse
import json
import logging
import math
import numbers
from pathlib import Path

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


def _fidelity_guard(original, size):
    from ._printability_fidelity import FidelityGuard
    return FidelityGuard(original, size)


def _evaluate(mesh, min_thickness, threshold, diagnostics=None):
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


def check_printability(mesh_path: str, min_thickness: float,
                       thin_wall_area_ratio_threshold: float, size: float,
                       fix_printability: bool, *, resolution: int = 1536,
                       output_path: str | None = None,
                       max_iso_scale: float = 3.0,
                       max_geometry_change_ratio: float | None = None,
                       min_silhouette_iou: float = 0.98) -> dict:
    """Repair while preserving visible shape; lengths are millimetres.

    iso_scale = offset / (min_thickness / 2). Each offset starts from the
    original mesh. Stop at the first printable, fidelity-preserving candidate;
    face count is secondary. Failure exports the resized original if requested.
    The geometry limit is worst-view p95 depth difference / size (10 views).
    None explicitly disables the visibility gate for diagnostic comparisons.
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
    if fix_printability and output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        source = Path(mesh_path).expanduser().resolve()
        if destination == source or (destination.exists() and source.exists() and destination.samefile(source)):
            raise ValueError('output_path must not overwrite the input')
        if destination.suffix.lower() != '.ply':
            raise ValueError('output_path must end in .ply')
    original = load_mesh(mesh_path, size)
    original_stats = topology(original)
    _LOG.info('Loaded %s: %d faces, %s', mesh_path, len(original.faces), original_stats)
    final_mesh = original
    stats = None
    attempts = 0
    selected_offset = None
    selected_geometry = None
    history = []
    initial_diagnostics = {}
    stop_reason = 'zero_thin_wall_threshold' if threshold == 0 else None
    if threshold > 0:
        stats = _evaluate(original, min_thickness, threshold, initial_diagnostics)
        if stats is None:
            # One backend, whose UDF always refers to the original geometry.
            offset = max(0.5 * min_thickness, 0.5 * size / resolution)
            cap = 0.5 * min_thickness * max_iso_scale
            backend = None
            guard = None
            stop_reason = 'max_iso_scale_exceeded'
            while attempts < 2 and offset <= cap * (1 + 1e-12):
                if backend is None:
                    backend = _repair_backend(original, int(resolution))
                    if max_geometry_change_ratio is not None:
                        guard = _fidelity_guard(original, size)
                attempts += 1
                entry = dict(attempt=attempts, offset_mm=offset,
                             iso_scale=2*offset/min_thickness, source='input_mesh', options=[])
                history.append(entry)
                _LOG.info('Repair attempt %d: offset=%g mm iso_scale=%g', attempts, offset, entry['iso_scale'])
                candidate = backend.reconstruct(offset)
                if candidate is None:
                    entry['reason'] = 'empty_reconstruction'
                else:
                    candidate = resize(candidate, size)
                    if not measurable(candidate):
                        entry['reason'] = 'invalid_reconstruction_topology'
                    else:
                        geometry = guard.evaluate(candidate, max_geometry_change_ratio, min_silhouette_iou) if guard else None
                        entry['raw_geometry'] = geometry
                        if geometry is not None and not geometry['passed']:
                            entry['reason'] = 'geometry_change_exceeded'
                            # Larger offsets cannot be relied upon to preserve
                            # erased features; do not enlarge a destructive result.
                            stop_reason = 'geometry_change_exceeded'
                            break
                        simplified = backend.simplify(candidate, len(original.faces))
                        options = ([('simplified', resize(simplified, size))] if measurable(simplified) else [])
                        options.append(('raw', candidate))
                        for kind, model in options:
                            change = (geometry if kind == 'raw' else
                                      guard.evaluate(model, max_geometry_change_ratio, min_silhouette_iou) if guard else None)
                            option = dict(kind=kind, num_faces=len(model.faces), geometry=change)
                            entry['options'].append(option)
                            if change is not None and not change['passed']:
                                option['reason'] = 'geometry_change_exceeded'
                                continue
                            evaluated = _evaluate(model, min_thickness, threshold, option)
                            if evaluated is None:
                                option['reason'] = 'printability_check_failed'
                                continue
                            option['reason'] = 'accepted'
                            final_mesh, stats, selected_offset, selected_geometry = model, evaluated, offset, change
                            break
                        if stats is not None:
                            stop_reason = None
                            break
                        entry['reason'] = 'printability_check_failed'
                if offset >= cap * (1 - 1e-12):
                    break
                offset = min(offset * 2.0, cap)
            if attempts == 2 and stats is None and stop_reason != 'geometry_change_exceeded':
                stop_reason = 'repair_budget_exhausted'
    printable = stats is not None
    if not printable:
        stats = {**original_stats, 'volume': None, 'thin_wall_area': None,
                 'thin_wall_area_ratio': None}
    result = {'printable': printable, **stats, 'output_mesh_path': None,
              'repair_attempts': attempts, 'repair_offset': selected_offset,
              'iso_scale': 2*selected_offset/min_thickness if selected_offset is not None else None,
              'max_iso_scale': max_iso_scale,
              'max_geometry_change_ratio': max_geometry_change_ratio,
              'min_silhouette_iou': min_silhouette_iou,
              'geometry_metrics': selected_geometry,
              'repair_history': history,
              'initial_diagnostics': initial_diagnostics,
              'failure_reason': None if printable else stop_reason,
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
                    help='enable the geometry-change guard with a relative depth limit')
    parser.add_argument('--min-silhouette-iou', type=float, default=.98)
    args = parser.parse_args()
    print(json.dumps(check_printability(**vars(args)), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
