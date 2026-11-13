"""Populate validated geometry caches using CUDA for large closed meshes."""
import numpy as np


def prepare_geometry(mesh, device='cuda'):
    """Check actual edge incidence and duplicate faces; retain CPU fallback.

    Component labels use edge adjacency, so point-touching shells remain
    distinct. Source vertices are never rounded or replaced in the mesh.
    """
    import torch
    from cumesh import CuMesh
    if mesh._cache['printability_components'] is not None:
        return
    nv, nf = len(mesh.vertices), len(mesh.faces)
    if not nf or nv*nv*2 >= np.iinfo(np.int64).max:
        return
    faces = torch.as_tensor(np.asarray(mesh.faces), device=device, dtype=torch.int64)
    encoded = torch.empty((nf, 3), device=device, dtype=torch.int64)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        u, v = faces[:, i], faces[:, j]
        encoded[:, i] = 2*(torch.minimum(u, v)*nv+torch.maximum(u, v))+(u > v)
    encoded = encoded.flatten().sort().values
    _, counts = torch.unique_consecutive(encoded//2, return_counts=True)
    closed = bool((counts == 2).all().item())
    if not closed:
        # Existing CPU path reports boundary loops and nonmanifold counts.
        return
    winding = bool(((encoded[::2] & 1) != (encoded[1::2] & 1)).all().item())
    del encoded, counts
    sorted_faces = faces.sort(dim=1).values
    # A pair key fits int64 even when a full triple key would overflow.
    pair = sorted_faces[:, 0]*nv+sorted_faces[:, 1]
    order = sorted_faces[:, 2].argsort(stable=True)
    order = order[pair[order].argsort(stable=True)]
    triples = sorted_faces[order]
    duplicate = bool((triples[1:] == triples[:-1]).all(dim=1).any().item())
    del sorted_faces, pair, order, triples
    from ._printability_geometry import face_areas
    areas = face_areas(mesh)
    mesh._cache['printability_edge_topology'] = {
        'watertight': True, 'num_holes': 0, 'num_nonmanifold_edges': 0}
    mesh._cache['printability_winding'] = winding
    mesh._cache['printability_measurable'] = bool(
        winding and not duplicate and np.isfinite(areas).all() and (areas > 0).all())
    if not mesh._cache['printability_measurable']:
        return
    cm = CuMesh()
    cm.init(torch.as_tensor(np.asarray(mesh.vertices), device=device, dtype=torch.float32), faces.int())
    cm.get_connected_components()
    count, labels = cm.read_connected_components()
    labels = labels.cpu().numpy()
    del cm, faces
    # Keep the original float64 reductions and deterministic representative
    # choice, so orientation probes are chosen identically to the CPU path.
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    maxima = np.full(count, -np.inf)
    np.maximum.at(maxima, labels, areas)
    representatives = np.full(count, nf, dtype=np.int64)
    volumes = np.zeros(count, dtype=np.float64)
    reference = vertices.mean(axis=0)
    for start in range(0, nf, 262144):
        stop = min(start+262144, nf)
        component = labels[start:stop]
        tri = vertices[faces[start:stop]]-reference
        signed = np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))/6
        volumes += np.bincount(component, weights=signed, minlength=count)
        selected = areas[start:stop] == maxima[component]
        np.minimum.at(representatives, component[selected], np.arange(start, stop)[selected])
    mesh._cache['printability_components'] = labels, volumes, representatives
