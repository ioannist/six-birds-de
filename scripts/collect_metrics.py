import json
from pathlib import Path


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _json_cell(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main() -> int:
    root = Path("results")
    aggregate_dir = root / "_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    metrics_paths = [p for p in root.rglob("metrics.json") if aggregate_dir not in p.parents]

    rows = []
    metric_keys = set()

    for metrics_path in metrics_paths:
        run_dir = metrics_path.parent
        metrics = _load_json(metrics_path)
        if not isinstance(metrics, dict):
            metrics = {}

        prov = _load_json(run_dir / "provenance.json")
        exp_name = None
        timestamp = None
        git_sha = None
        if isinstance(prov, dict):
            exp_name = prov.get("exp_name")
            timestamp = prov.get("timestamp_utc")
            git_sha = prov.get("git_sha")

        rel_run_dir = str(run_dir)
        row = {
            "run_dir": rel_run_dir,
            "exp_name": exp_name or run_dir.parent.name,
            "timestamp_utc": timestamp,
            "git_sha": git_sha,
        }

        for k, v in metrics.items():
            metric_keys.add(k)
            row[k] = _json_cell(v)

        rows.append(row)

    fixed_cols = ["run_dir", "exp_name", "timestamp_utc", "git_sha"]
    metric_cols = sorted(metric_keys)

    # sort rows
    def sort_key(r):
        return (r.get("timestamp_utc") or "", r.get("run_dir"))

    rows_sorted = sorted(rows, key=sort_key)

    out_path = aggregate_dir / "metrics_table.csv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(fixed_cols + metric_cols) + "\n")
        for row in rows_sorted:
            vals = []
            for col in fixed_cols + metric_cols:
                v = row.get(col, "")
                if isinstance(v, bool):
                    v = "true" if v else "false"
                vals.append(str(v) if v is not None else "")
            f.write(",".join(vals) + "\n")

    print(f"Wrote {out_path}")
    print(f"Rows: {len(rows_sorted)}")
    if metric_cols:
        print(f"Metric keys: {len(metric_cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
