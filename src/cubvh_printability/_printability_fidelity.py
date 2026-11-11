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
