"""Independent Python client for the migrated Printability implementation."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True, slots=True)
class PrintabilityConfig:
    """Options passed to the migrated Printability implementation.

    Lengths are millimetres.  The defaults mirror the current cubvh API while
    keeping all policy choices explicit at the wrapper boundary.
    Repair acceptance uses measured thickness; max_iso_scale bounds growth.
    max_geometry_change_ratio and min_silhouette_iou are legacy options and
    no longer enable camera checks.
    """

    min_thickness: float
    size: float
    thin_wall_area_ratio_threshold: float = 0.1
    fix_printability: bool = True
    resolution: int = 1536
    max_iso_scale: float = 3.0
    max_geometry_change_ratio: float | None = None
    min_silhouette_iou: float = 0.98
    repair_strategy: str = "global"

    def __post_init__(self) -> None:
        _positive("min_thickness", self.min_thickness)
        _positive("size", self.size)
        _unit_interval("thin_wall_area_ratio_threshold", self.thin_wall_area_ratio_threshold)
        _unit_interval("min_silhouette_iou", self.min_silhouette_iou)
        if not isinstance(self.fix_printability, bool):
            raise TypeError("fix_printability must be bool")
        if isinstance(self.resolution, bool) or not isinstance(self.resolution, int):
            raise TypeError("resolution must be an integer")
        if not 8 <= self.resolution <= 4095:
            raise ValueError("resolution must be in [8, 4095]")
        _positive("max_iso_scale", self.max_iso_scale)
        if self.max_geometry_change_ratio is not None:
            _positive("max_geometry_change_ratio", self.max_geometry_change_ratio)
        if self.repair_strategy not in ("global", "parts"):
            raise ValueError("repair_strategy must be 'global' or 'parts'")

    def as_kwargs(self, output_path: str | Path | None = None) -> dict[str, Any]:
        """Return keyword arguments accepted by the migrated implementation."""
        return {
            "min_thickness": self.min_thickness,
            "thin_wall_area_ratio_threshold": self.thin_wall_area_ratio_threshold,
            "size": self.size,
            "fix_printability": self.fix_printability,
            "resolution": self.resolution,
            "output_path": None if output_path is None else str(Path(output_path)),
            "max_iso_scale": self.max_iso_scale,
            "max_geometry_change_ratio": self.max_geometry_change_ratio,
            "min_silhouette_iou": self.min_silhouette_iou,
            "repair_strategy": self.repair_strategy,
        }


class PrintabilityClient:
    """Call the migrated printability implementation without importing it eagerly.

    ``check_fn`` is primarily useful for tests or applications that provide a
    compatible backend. In normal use it is resolved from this package on first use.
    """

    def __init__(self, check_fn: Callable[..., Mapping[str, Any]] | None = None):
        self._check_fn = check_fn

    def _backend(self) -> Callable[..., Mapping[str, Any]]:
        if self._check_fn is None:
            try:
                module = importlib.import_module("cubvh_printability.core")
                self._check_fn = module.check_printability
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "cubvh with its CUDA dependencies must be installed before "
                    "using cubvh-printability"
                ) from exc
        return self._check_fn

    def check_printability(
        self,
        mesh_path: str | Path,
        config: PrintabilityConfig,
        *,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run the cubvh check and return its result dictionary."""
        if not isinstance(config, PrintabilityConfig):
            raise TypeError("config must be PrintabilityConfig")
        source = Path(mesh_path).expanduser()
        if not source.exists():
            raise FileNotFoundError(source)
        result = self._backend()(str(source), **config.as_kwargs(output_path))
        if not isinstance(result, Mapping):
            raise TypeError("the Printability implementation must return a mapping")
        return dict(result)


def check_printability(
    mesh_path: str | Path,
    *,
    min_thickness: float,
    size: float,
    thin_wall_area_ratio_threshold: float = 0.1,
    fix_printability: bool = True,
    resolution: int = 1536,
    output_path: str | Path | None = None,
    max_iso_scale: float = 3.0,
    max_geometry_change_ratio: float | None = None,
    min_silhouette_iou: float = 0.98,
    repair_strategy: str = "global",
) -> dict[str, Any]:
    """Convenience function with the same policy parameters as cubvh."""
    config = PrintabilityConfig(
        min_thickness=min_thickness,
        size=size,
        thin_wall_area_ratio_threshold=thin_wall_area_ratio_threshold,
        fix_printability=fix_printability,
        resolution=resolution,
        max_iso_scale=max_iso_scale,
        max_geometry_change_ratio=max_geometry_change_ratio,
        min_silhouette_iou=min_silhouette_iou,
        repair_strategy=repair_strategy,
    )
    return PrintabilityClient().check_printability(mesh_path, config, output_path=output_path)


def repair_mesh(
    mesh_path: str | Path,
    config: PrintabilityConfig,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a configured repair and return cubvh's result dictionary.

    This name is convenient for applications that always enable export; the
    ``fix_printability`` flag in ``config`` still controls whether cubvh writes
    an output mesh.
    """
    return PrintabilityClient().check_printability(
        mesh_path, config, output_path=output_path
    )


def _positive(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _unit_interval(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be in [0, 1]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
