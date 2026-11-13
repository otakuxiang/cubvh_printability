"""Deterministic external-surface fidelity gate, in the final resized frame.

This is a finite-view visibility check, not a Hausdorff bound. It deliberately
uses full triangles and the same camera rays for input and candidate, with no
ICP alignment, independent rescaling, mesh decimation, or point rendering.
"""
import gc
import numpy as np

VIEW_DIRECTIONS = [(0,0,1),(0,0,-1),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),
                   (1,.65,1.3),(-1,.65,1.3),(1,.65,-1.3),(-1,.65,-1.3)]


def compare_depths(reference, candidate, size):
    """Summarize each view; background is +inf and cannot count as a match."""
    views=[]
    for a,b in zip(reference,candidate):
        ma=np.isfinite(a);mb=np.isfinite(b);common=ma&mb;union=ma|mb
        distance=np.abs(a[common]-b[common])
        views.append(dict(silhouette_iou=float(common.sum()/max(1,union.sum())),
            depth_p95_mm=float(np.quantile(distance,.95)) if len(distance) else float(size),
            depth_mean_mm=float(distance.mean()) if len(distance) else float(size),
            depth_changed_area_ratio=float(np.count_nonzero(distance > .005*size)/max(1,union.sum()))))
    return dict(views=views,min_silhouette_iou=min(v['silhouette_iou'] for v in views),
                max_depth_p95_mm=max(v['depth_p95_mm'] for v in views),
                max_depth_p95_ratio=max(v['depth_p95_mm'] for v in views)/size)


class FidelityGuard:
    def __init__(self, original, size, resolution=384):
        self.size=float(size)
        self.resolution=resolution
        self.reference_area=float(original.area)
        self.depths=self.render_depths(original)

    def render_depths(self, mesh):
        from .optix import OptixRayTracer
        import torch
        tracer=OptixRayTracer(np.asarray(mesh.vertices)/self.size,np.asarray(mesh.faces))
        axis=(np.arange(self.resolution)+.5)/self.resolution*1.4-.7
        xx,yy=np.meshgrid(axis,axis[::-1]);depths=[]
        for direction in VIEW_DIRECTIONS:
            z=np.array(direction,dtype=float);z/=np.linalg.norm(z)
            up=np.array([0.,1.,0.]) if abs(z[1])<.95 else np.array([0.,0.,-1.])
            x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x)
            origins=(xx[...,None]*x+yy[...,None]*y+2*z).reshape(-1,3)
            _,dist=tracer.trace(origins,np.broadcast_to(-z,origins.shape).copy(),t_max=4.)
            depths.append(dist.cpu().numpy().reshape(self.resolution,self.resolution)*self.size)
        del tracer
        gc.collect()
        torch.cuda.empty_cache()
        return np.array(depths)

    def evaluate(self, candidate, max_depth_ratio, min_silhouette_iou):
        metrics=compare_depths(self.depths,self.render_depths(candidate),self.size)
        metrics.update(area_change_ratio=abs(float(candidate.area)/self.reference_area-1),
                       method='10 fixed orthographic views, full triangles, OptiX closest hit',
                       resolution=self.resolution,max_allowed_depth_p95_ratio=max_depth_ratio,
                       min_allowed_silhouette_iou=min_silhouette_iou)
        metrics['passed']=bool(metrics['max_depth_p95_ratio'] <= max_depth_ratio
                               and metrics['min_silhouette_iou'] >= min_silhouette_iou)
        return metrics


class LocalFidelityGuard:
    """Whole-object depths plus cumulative close-ups of proposed and actual changes."""
    def __init__(self, original, size, centers, resolution=256):
        from .render_review import render_view
        from .optix import OptixRayTracer
        self.whole = FidelityGuard(original, size)
        self.size, self.resolution = size, resolution
        self.original = original
        self.centers = [np.asarray(c) for c in centers]
        self.reference = []
        tracer = OptixRayTracer(np.asarray(original.vertices), np.asarray(original.faces))
        for center in centers:
            for direction in VIEW_DIRECTIONS:
                self.reference.append(render_view(original, tracer, center,
                    max(4., .12 * size), direction, resolution))

    def evaluate(self, candidate, max_depth_ratio, min_silhouette_iou):
        from .render_review import render_view, compare_views, displaced_centers
        from .optix import OptixRayTracer
        # Seed area can underweight small ornaments with large distortion.
        # Retain old cameras and add close-ups from measured surface movement.
        added = []
        for center in displaced_centers(self.original, candidate, self.size):
            if all(np.linalg.norm(np.asarray(center)-c) >= .02*self.size for c in self.centers):
                self.centers.append(np.asarray(center))
                added.append(center)
        if added:
            reference_tracer = OptixRayTracer(np.asarray(self.original.vertices),
                                             np.asarray(self.original.faces))
            for center in added:
                for direction in VIEW_DIRECTIONS:
                    self.reference.append(render_view(self.original, reference_tracer, center,
                        max(4., .12*self.size), direction, self.resolution))
            del reference_tracer
        result = self.whole.evaluate(candidate, max_depth_ratio, min_silhouette_iou)
        tracer = OptixRayTracer(np.asarray(candidate.vertices), np.asarray(candidate.faces))
        local = []
        for center in self.centers:
            for direction in VIEW_DIRECTIONS:
                after = render_view(candidate, tracer, center, max(4., .12 * self.size),
                                    direction, self.resolution)
                before = self.reference[len(local)]
                metric = compare_views(before, after)
                # A close-up containing no object in either image has no evidence.
                if not (np.isfinite(before[1]) | np.isfinite(after[1])).any():
                    metric['passed'] = True
                else:
                    metric['passed'] = bool(metric['silhouette_iou'] >= .95
                        and metric['depth_p95_mm'] is not None
                        and metric['depth_p95_mm'] <= min(.25, self.size * max_depth_ratio)
                        and metric['normal_angle_p95_deg'] <= 30.)
                local.append(metric)
        result['local_views'] = local
        result['local_passed'] = all(m['passed'] for m in local)
        result['passed'] = bool(result['passed'] and result['local_passed'])
        return result


class FallbackFidelityGuard:
    """Keep the parts strategy's render gates and cameras across fallback attempts."""
    def __init__(self, original, size):
        # Lazy import: the feedback module shares VIEW_DIRECTIONS above.
        from ._local_feedback import FieldFidelityGuard
        self.guard = FieldFidelityGuard(original, size)

    def evaluate(self, candidate, max_depth_ratio, min_silhouette_iou):
        return self.guard.evaluate(candidate, max_depth_ratio, min_silhouette_iou)
