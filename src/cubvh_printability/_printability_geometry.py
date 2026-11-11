"""CPU geometry and lossless output helpers for printability (units: mm)."""
from pathlib import Path
import os
import tempfile
import uuid
import logging

import numpy as np
import trimesh


def face_areas(mesh):
    """Populate Trimesh's area cache without retaining all triangle coordinates."""
    cached = mesh._cache['area_faces']
    if cached is None:
        cached = np.empty(len(mesh.faces), dtype=np.float64)
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        for start in range(0, len(faces), 262144):
            triangles = vertices[faces[start:start + 262144]]
            cross = np.cross(triangles[:, 1] - triangles[:, 0],
                             triangles[:, 2] - triangles[:, 0])
            cached[start:start + len(triangles)] = np.linalg.norm(cross, axis=1) * .5
        mesh._cache['area_faces'] = cached
    return cached


def resize(mesh, size):
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    bounds = mesh.bounds
    extent = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("mesh must have a finite, nonzero extent")
    mesh.vertices = (np.asarray(mesh.vertices) - bounds.mean(axis=0)) * (size / extent)
    return mesh


def load_mesh(path, size):
    # process=True welds identical vertices (e.g. STL triangle soup).
    loaded = trimesh.load(path, process=True)
    if isinstance(loaded, trimesh.Scene):
        parts = []
        for node in loaded.graph.nodes_geometry:
            transform, name = loaded.graph[node]
            part = loaded.geometry[name]
            if not isinstance(part, trimesh.Trimesh):
                continue
            part = part.copy()
            part.apply_transform(transform)
            parts.append(part)
        if not parts:
            raise ValueError("scene must contain a nonempty triangle mesh")
        loaded = trimesh.util.concatenate(parts)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError("input must contain a nonempty triangle mesh")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError("mesh vertices must be finite")
    if np.any(loaded.faces < 0) or np.any(loaded.faces >= len(loaded.vertices)):
        raise ValueError("mesh contains invalid vertex indices")
    loaded = resize(loaded, size)
    area = float(face_areas(loaded).sum())
    if not np.isfinite(area) or area <= 0:
        raise ValueError("mesh must have positive finite surface area")
    return loaded


def topology(mesh):
    face_areas(mesh)
    cached = mesh._cache['printability_edge_topology']
    if cached is not None:
        return {**cached, 'surface_area': float(mesh.area)}
    # Sorting scalar edge keys is substantially faster and leaner than sorting
    # 3F structured (vertex, vertex) rows on multi-million-face MC outputs.
    n = len(mesh.vertices)
    if n and n * n * 2 <= np.iinfo(np.int64).max:
        encoded = np.empty(len(mesh.faces) * 3, dtype=np.int64)
        for start in range(0, len(mesh.faces), 1_000_000):
            faces = np.asarray(mesh.faces[start:start + 1_000_000])
            for edge, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
                u, v = faces[:, a], faces[:, b]
                encoded[start * 3 + edge:(start + len(faces)) * 3:3] = (
                    (np.minimum(u, v) * n + np.maximum(u, v)) * 2 + (u > v))
        encoded.sort()
        keys = encoded // 2
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else np.empty(0, np.int64)
        count = np.diff(np.r_[starts, len(keys)])
        boundary_keys = keys[starts[count == 1]]
        boundary = np.column_stack((boundary_keys // n, boundary_keys % n))
        if len(mesh.faces) and np.all(count == 2):
            mesh._cache['printability_winding'] = bool(np.all((encoded[starts] & 1) != (encoded[starts + 1] & 1)))
    else:
        edges, count = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
        boundary = edges[count == 1]
    parent = {}

    def root(v):
        parent.setdefault(v, v)
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in boundary:
        parent[root(int(a))] = root(int(b))
    holes = len({root(v) for v in parent})
    result = dict(watertight=bool(len(mesh.faces) and np.all(count == 2)),
                  num_holes=holes, num_nonmanifold_edges=int(np.sum(count > 2)))
    mesh._cache['printability_edge_topology'] = result
    return {**result, 'surface_area': float(mesh.area)}


def measurable(mesh, stats=None):
    cached = mesh._cache['printability_measurable']
    if cached is not None:
        return cached
    stats = topology(mesh) if stats is None else stats
    if not stats['watertight']:
        logging.getLogger(__name__).info('Unmeasurable topology: %s', stats)
        return False
    winding = mesh._cache['printability_winding']
    if not (mesh.is_winding_consistent if winding is None else winding):
        logging.getLogger(__name__).info('Unmeasurable: inconsistent winding')
        return False
    invalid = ~np.isfinite(mesh.area_faces) | (mesh.area_faces <= 0)
    if invalid.any():
        logging.getLogger(__name__).info('Unmeasurable: %d degenerate faces', invalid.sum())
        return False
    faces = np.sort(np.asarray(mesh.faces), axis=1)
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    for start in range(1, len(order), 262144):
        end = min(start + 262144, len(order))
        if np.any(np.all(faces[order[start:end]] == faces[order[start - 1:end - 1]], axis=1)):
            logging.getLogger(__name__).info('Unmeasurable: duplicate faces')
            return False
    mesh._cache['printability_measurable'] = True
    return True


def export_mesh(mesh, source, output_path):
    source = Path(source).resolve()
    destination = (Path(output_path).expanduser().resolve() if output_path is not None
                   else source.with_name(f'{source.stem}_printability_{uuid.uuid4().hex}.ply'))
    if destination == source or (destination.exists() and os.path.samefile(source, destination)):
        raise ValueError("output_path must not overwrite the input")
    if destination.suffix.lower() != '.ply':
        raise ValueError("output_path must end in .ply (double precision export)")
    # Trimesh's standard PLY writer downcasts to float32. Binary doubles retain
    # resize precision and avoid ASCII parsing's large temporary strings/arrays.
    fd, temporary = tempfile.mkstemp(prefix='.printability-', suffix='.ply', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(mesh.vertices)}\n'
                         'property double x\nproperty double y\nproperty double z\n'
                         f'element face {len(mesh.faces)}\nproperty list uchar int vertex_indices\nend_header\n').encode('ascii'))
            for start in range(0, len(mesh.vertices), 262144):
                stream.write(np.asarray(mesh.vertices[start:start + 262144], dtype='<f8').tobytes())
            face_dtype = np.dtype([('count', 'u1'), ('indices', '<i4', (3,))])
            for start in range(0, len(mesh.faces), 262144):
                faces = np.asarray(mesh.faces[start:start + 262144])
                if faces.min() < 0 or faces.max() > np.iinfo(np.int32).max:
                    raise ValueError('PLY export requires nonnegative int32 vertex indices')
                records = np.empty(len(faces), dtype=face_dtype)
                records['count'] = 3
                records['indices'] = faces
                stream.write(records.tobytes())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(destination)
