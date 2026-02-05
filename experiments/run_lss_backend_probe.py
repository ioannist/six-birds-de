import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sixbirds_cosmo import manifest
from sixbirds_cosmo.reporting import write_run_markdown_stub


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _record_error(errors: list, stage: str, exc: Exception) -> None:
    tb = traceback.format_exc()
    errors.append(
        {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": tb[-4000:] if tb else None,
        }
    )


def _probe_pyccl(force_no_backend: bool) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    probe = {
        "timestamp_utc": _utc_now_iso(),
        "backend_attempted": "none" if force_no_backend else "pyccl",
        "backend_available": False,
        "backend_version": None,
        "success_stage": "none",
        "errors": [],
    }

    if force_no_backend:
        return probe, None, None

    try:
        import pyccl as ccl  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        _record_error(probe["errors"], "import", exc)
        return probe, None, None

    probe["backend_version"] = getattr(ccl, "__version__", None)
    probe["success_stage"] = "import"

    try:
        cosmo = ccl.Cosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.67,
            sigma8=0.8,
            n_s=0.965,
        )
        k = np.logspace(-3, 0, 50)
        pk = ccl.linear_matter_power(cosmo, k, 1.0)
        if not np.all(np.isfinite(pk)):
            raise ValueError("Non-finite P(k) values returned")
        probe["backend_available"] = True
        probe["success_stage"] = "pk"
        pk_at_k0p1 = float(np.interp(0.1, k, pk)) if (k.min() <= 0.1 <= k.max()) else None
        probe["pk_summary"] = {
            "k_min": float(k.min()),
            "k_max": float(k.max()),
            "pk_min": float(pk.min()),
            "pk_max": float(pk.max()),
            "pk_at_k0p1": pk_at_k0p1,
        }
        return probe, k, pk
    except Exception as exc:  # pragma: no cover - depends on environment
        _record_error(probe["errors"], "pk", exc)
        return probe, None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    force_no_backend = os.environ.get("SIXBIRDS_FORCE_NO_PYCCL") == "1"

    run_dir = manifest.create_run_dir("lss_backend_probe", seed=seed)

    config = {
        "seed": seed,
        "force_no_backend": force_no_backend,
        "backend_preference": "pyccl",
        "pk_k_min": 1e-3,
        "pk_k_max": 1.0,
        "pk_n": 50,
    }
    manifest.write_config(run_dir, config)

    probe, k, pk = _probe_pyccl(force_no_backend)

    plot_files: list[str] = []
    if probe.get("backend_available") and k is not None and pk is not None:
        fig, ax = plt.subplots()
        ax.loglog(k, pk)
        ax.set_xlabel("k [1/Mpc]")
        ax.set_ylabel("P(k)")
        plot_path = run_dir / "pk_linear.png"
        manifest.save_fig(fig, plot_path)
        plt.close(fig)
        plot_files.append(plot_path.name)

    backend_probe_path = run_dir / "backend_probe.json"
    with backend_probe_path.open("w", encoding="utf-8") as f:
        json.dump(probe, f, indent=2, sort_keys=True)

    note = "ok" if probe.get("backend_available") else (
        probe["errors"][0]["message"] if probe.get("errors") else "backend unavailable"
    )
    metrics = {
        "backend_attempted": probe.get("backend_attempted"),
        "backend_available": probe.get("backend_available"),
        "success_stage": probe.get("success_stage"),
        "plot_files": plot_files,
        "note": note,
    }
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "lss_backend_probe", seed=seed)
    write_run_markdown_stub(run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
