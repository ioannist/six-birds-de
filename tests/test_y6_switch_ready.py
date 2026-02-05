import json
from pathlib import Path
import subprocess


def test_y6_smoke_blocked_mode_writes_metrics():
    subprocess.run(["python", "experiments/run_des_y6_3x2pt_smoke.py"], check=True)
    results_dir = Path("results/des_y6_3x2pt_smoke")
    assert results_dir.exists()
    run_dirs = [p for p in results_dir.iterdir() if p.is_dir()]
    assert run_dirs
    latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    metrics_path = latest / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics.get("available") is False
    blocked_reason = metrics.get("blocked_reason")
    assert isinstance(blocked_reason, str) and blocked_reason
