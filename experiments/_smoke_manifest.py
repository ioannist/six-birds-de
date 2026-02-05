import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sixbirds_cosmo import manifest
from sixbirds_cosmo.reporting import write_run_markdown_stub


def main() -> None:
    exp_name = "smoke_manifest"
    seed = 123
    run_dir = manifest.create_run_dir(exp_name, seed=seed)

    config = {
        "exp_name": exp_name,
        "seed": seed,
        "params": {
            "alpha": 0.1,
            "steps": 5,
            "path": Path("configs/example.yaml"),
        },
    }
    metrics = {
        "loss": 0.123,
        "accuracy": 0.987,
        "steps": 5,
        "status": "ok",
    }

    manifest.write_config(run_dir, config)
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, exp_name, seed, extra={"note": "smoke"})

    x = np.linspace(0, 1, 50)
    y = x**2
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Smoke manifest plot")
    manifest.save_fig(fig, run_dir / "plot.png")
    plt.close(fig)

    table = pd.DataFrame({"x": x, "y": y})
    manifest.save_table(table, run_dir / "table.csv")

    write_run_markdown_stub(run_dir)

    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
