"""Persistent OptiX closest-hit tracer; no alternative intersection backend."""
import math


class OptixRayTracer:
    def __init__(self, vertices, faces):
        from .build_optix import build
        module, ptx = build()
        import torch
        self._torch = torch
        device = vertices.device if torch.is_tensor(vertices) and vertices.is_cuda else torch.device('cuda', torch.cuda.current_device())
        vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device).clone().contiguous()
        indices = torch.as_tensor(faces, dtype=torch.int64, device=device).contiguous()
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not torch.isfinite(vertices).all():
            raise ValueError('vertices must be finite, nonempty [V,3]')
        if indices.ndim != 2 or indices.shape[1] != 3 or not len(indices) or indices.min() < 0 or indices.max() >= len(vertices):
            raise ValueError('faces must be nonempty [F,3] with valid vertex indices')
        if max(len(indices), len(vertices)) >= 2**31:
            raise ValueError('OptiX mesh exceeds 32-bit primitive/index limit')
        self.device, self.num_faces = device, len(indices)
        self._impl = module.OptixRayTracer(vertices, indices.int().contiguous(), ptx)

    def trace(self, origins, directions, t_min=0.0, t_max=float('inf'), source_face_ids=None):
        """Return (original face IDs, metric distances); misses are (-1, inf).

        Directions are normalized. t_min excludes numerical self intersections;
        source_face_ids additionally ignores a surface's own primitive.
        Launches use and synchronize the current PyTorch stream.
        """
        torch = self._torch
        if not math.isfinite(t_min) or t_min < 0 or math.isnan(t_max) or t_max <= t_min:
            raise ValueError('require finite 0 <= t_min < t_max')
        origins = torch.as_tensor(origins, device=self.device, dtype=torch.float32).contiguous()
        directions = torch.as_tensor(directions, device=self.device, dtype=torch.float32)
        if origins.ndim != 2 or origins.shape[1] != 3 or directions.shape != origins.shape:
            raise ValueError('origins and directions must be [N,3]')
        norm = torch.linalg.vector_norm(directions, dim=1, keepdim=True)
        if not torch.isfinite(origins).all() or not torch.isfinite(directions).all() or not torch.isfinite(norm).all() or (norm <= 0).any():
            raise ValueError('rays must be finite with nonzero directions')
        source = (torch.full((len(origins),), -1, device=self.device, dtype=torch.int32) if source_face_ids is None
                  else torch.as_tensor(source_face_ids, device=self.device, dtype=torch.int64))
        if source.shape != (len(origins),) or (source < -1).any() or (source >= self.num_faces).any():
            raise ValueError('source_face_ids must contain -1 or valid face IDs')
        return self._impl.trace(origins, (directions / norm).contiguous(), float(t_min), float(t_max), source.int().contiguous())

    def _trace_trusted(self, origins, directions, t_min, t_max, source):
        """Internal finite CUDA rays and valid IDs, retaining normalization.

        Public trace validates external inputs. Thickness code constructs these
        tensors itself, so repeated scalar validation would only synchronize
        the GPU. Keep normalization identical to trace for reproducibility.
        """
        norm = self._torch.linalg.vector_norm(directions, dim=1, keepdim=True)
        return self._impl.trace(origins.contiguous(), (directions / norm).contiguous(),
                                float(t_min), float(t_max), source.int().contiguous())
