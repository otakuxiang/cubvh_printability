"""Command-line entry point for the standalone client."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import PrintabilityConfig, PrintabilityClient


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Call cubvh's printability checker")
    p.add_argument("mesh", type=Path, help="input mesh (.glb/.ply/.obj/...)" )
    p.add_argument("--min-thickness", type=float, required=True)
    p.add_argument("--size", type=float, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--threshold", type=float, default=0.1,
                   dest="thin_wall_area_ratio_threshold")
    p.add_argument("--resolution", type=int, default=1536)
    p.add_argument("--max-iso-scale", type=float, default=3.0,
                   help="cap growth in mm at min_thickness * max_iso_scale / 2")
    p.add_argument("--max-geometry-change-ratio", type=float, default=None,
                   help="legacy compatibility option; view checks are disabled")
    p.add_argument("--min-silhouette-iou", type=float, default=0.98,
                   help="legacy compatibility option; view checks are disabled")
    p.add_argument("--repair-strategy", choices=("global", "parts"), default="global",
                   help="repair globally or thicken selected thin patches with a local signed field")
    p.add_argument("--no-export", action="store_true",
                   help="run the check without exporting a repaired mesh")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = PrintabilityConfig(
        min_thickness=args.min_thickness,
        size=args.size,
        thin_wall_area_ratio_threshold=args.thin_wall_area_ratio_threshold,
        fix_printability=not args.no_export,
        resolution=args.resolution,
        max_iso_scale=args.max_iso_scale,
        max_geometry_change_ratio=args.max_geometry_change_ratio,
        min_silhouette_iou=args.min_silhouette_iou,
        repair_strategy=args.repair_strategy,
    )
    result = PrintabilityClient().check_printability(
        args.mesh, config, output_path=args.output
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
