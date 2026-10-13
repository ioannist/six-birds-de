import json
from pathlib import Path
from experiments import run_des_y6_3x2pt_smoke as smoke


def test_y6_smoke_blocked_mode_writes_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "dataset_exists", lambda key: False)
    monkeypatch.setattr(smoke, "run_availability_check", lambda: tmp_path / "availability.json")
    monkeypatch.setattr(smoke.manifest, "create_run_dir", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(smoke.manifest, "_repo_root", lambda: tmp_path)
    smoke.main()
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics.get("available") is False
    assert metrics.get("availability_scope") == "local_registry"
    blocked_reason = metrics.get("blocked_reason")
    assert isinstance(blocked_reason, str) and blocked_reason
