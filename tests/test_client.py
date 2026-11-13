from pathlib import Path

import pytest

from cubvh_printability import PrintabilityClient, PrintabilityConfig


def test_config_validation_and_kwargs():
    cfg = PrintabilityConfig(min_thickness=0.05, size=50, max_iso_scale=2)
    kwargs = cfg.as_kwargs("out.ply")
    assert kwargs["max_iso_scale"] == 2
    assert kwargs["output_path"] == "out.ply"
    assert kwargs["max_geometry_change_ratio"] is None
    assert kwargs["repair_strategy"] == "global"


def test_parts_repair_strategy_is_validated_and_forwarded():
    cfg = PrintabilityConfig(min_thickness=0.05, size=50, repair_strategy="parts")
    assert cfg.as_kwargs()["repair_strategy"] == "parts"
    with pytest.raises(ValueError):
        PrintabilityConfig(min_thickness=0.05, size=50, repair_strategy="unknown")


def test_client_delegates_without_importing_cuda(tmp_path: Path):
    mesh = tmp_path / "input.ply"
    mesh.write_text("placeholder")
    calls = {}

    def fake(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return {"printable": True, "output_mesh_path": "out.ply"}

    cfg = PrintabilityConfig(min_thickness=0.5, size=50, fix_printability=False)
    result = PrintabilityClient(fake).check_printability(mesh, cfg)
    assert result["printable"] is True
    assert calls["path"] == str(mesh)
    assert calls["kwargs"]["fix_printability"] is False


def test_missing_mesh_is_reported():
    cfg = PrintabilityConfig(min_thickness=0.05, size=50)
    with pytest.raises(FileNotFoundError):
        PrintabilityClient(lambda *_args, **_kwargs: {}).check_printability("missing.ply", cfg)
