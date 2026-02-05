import matplotlib

matplotlib.use("Agg")

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sixbirds_cosmo import manifest
from sixbirds_cosmo.core import Completion, Lens, PackagingOperator, idempotence_defect, route_mismatch
from sixbirds_cosmo.reporting import write_run_markdown_stub


def rms_distance(u: np.ndarray, v: np.ndarray) -> float:
    diff = u - v
    return float(np.linalg.norm(diff) / np.sqrt(diff.size))


def rk4_step(a: np.ndarray, adot: np.ndarray, rho0: np.ndarray, dt: float, g4pi: float) -> tuple[np.ndarray, np.ndarray]:
    def accel(a_in: np.ndarray) -> np.ndarray:
        return -(g4pi / 3.0) * rho0 / (a_in ** 2)

    k1a = adot
    k1v = accel(a)

    k2a = adot + 0.5 * dt * k1v
    k2v = accel(a + 0.5 * dt * k1a)

    k3a = adot + 0.5 * dt * k2v
    k3v = accel(a + 0.5 * dt * k2a)

    k4a = adot + dt * k3v
    k4v = accel(a + dt * k3a)

    a_next = a + (dt / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
    adot_next = adot + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

    eps = 1e-12
    a_next = np.clip(a_next, eps, None)
    return a_next, adot_next


def build_packaging(N: int, L: int, idx_sorted: np.ndarray, inv_sorted: np.ndarray) -> PackagingOperator:
    G = N // L

    def lens_fn(x: np.ndarray) -> np.ndarray:
        a = x[:N]
        adot = x[N:]
        a_s = a[idx_sorted].reshape(G, L)
        adot_s = adot[idx_sorted].reshape(G, L)
        a_mean = np.mean(a_s, axis=1)
        adot_mean = np.mean(adot_s, axis=1)
        return np.concatenate([a_mean, adot_mean])

    def completion_fn(y: np.ndarray) -> np.ndarray:
        a_mean = y[:G]
        adot_mean = y[G:]
        a_s = np.repeat(a_mean, L)
        adot_s = np.repeat(adot_mean, L)
        a = a_s[inv_sorted]
        adot = adot_s[inv_sorted]
        return np.concatenate([a, adot])

    lens = Lens(lens_fn)
    completion = Completion(completion_fn)
    return PackagingOperator(lens, completion, distance_x=rms_distance)


def main() -> None:
    seed = 123
    N = 500
    steps = 250
    dt = 0.02
    g4pi = 1.0
    rho_mean = 1.0
    delta_rho = 0.8
    kappa_scale = 5.0
    L_list = [1, 2, 4, 5, 10, 20, 25, 50, 100, 125, 250, 500]

    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=N)
    rho0 = rho_mean * (1.0 + delta_rho * (2.0 * u - 1.0))
    rho0 = np.clip(rho0, 1e-6, None)
    kappa = kappa_scale * np.maximum(0.0, (rho_mean - rho0) / rho_mean)

    a0 = np.ones(N, dtype=float)
    H0 = np.sqrt((2.0 / 3.0) * g4pi * rho0 + kappa)
    adot0 = H0.copy()
    x0 = np.concatenate([a0, adot0])

    idx_sorted = np.argsort(rho0)
    inv_sorted = np.empty_like(idx_sorted)
    inv_sorted[idx_sorted] = np.arange(N)

    summary_rows = []
    rm_mean_late_list = []
    delta_max_list = []

    for L in L_list:
        if N % L != 0:
            raise ValueError(f"L={L} must divide N={N}")
        G = N // L
        E = build_packaging(N, L, idx_sorted, inv_sorted)
        x = x0.copy()
        rm_t = np.zeros(steps, dtype=float)
        delta_t = np.zeros(steps, dtype=float)
        def T(state: np.ndarray) -> np.ndarray:
            a_state = state[:N]
            adot_state = state[N:]
            a_next, adot_next = rk4_step(a_state, adot_state, rho0, dt, g4pi)
            return np.concatenate([a_next, adot_next])

        for t in range(steps):
            rm_t[t] = route_mismatch(T, E, x, space="x")
            delta_t[t] = idempotence_defect(E, x, space="x")
            x = T(x)

        rm_mean = float(np.mean(rm_t))
        rm_mean_late = float(np.mean(rm_t[int(0.8 * steps) :]))
        rm_max = float(np.max(rm_t))
        delta_mean = float(np.mean(delta_t))
        delta_max = float(np.max(delta_t))

        summary_rows.append(
            {
                "L": L,
                "G": G,
                "rm_mean": rm_mean,
                "rm_mean_late": rm_mean_late,
                "rm_max": rm_max,
                "delta_mean": delta_mean,
                "delta_max": delta_max,
            }
        )
        rm_mean_late_list.append(rm_mean_late)
        delta_max_list.append(delta_max)

    rm_L1 = rm_mean_late_list[L_list.index(1)]
    rm_Lmax = rm_mean_late_list[L_list.index(500)]

    delta_max_arr = np.array(delta_max_list)
    L_min_delta_idx = int(np.argmin(delta_max_arr))
    L_min_delta = L_list[L_min_delta_idx]
    RM_at_L_min_delta = rm_mean_late_list[L_min_delta_idx]

    # Assertions
    assert rm_L1 < 1e-12
    assert rm_Lmax > 1e-4
    assert float(np.max(delta_max_arr)) < 1e-10

    run_dir = manifest.create_run_dir("scale_sweep", seed=seed)

    # Plots
    fig1, ax1 = plt.subplots()
    ax1.plot(L_list, rm_mean_late_list, marker="o")
    ax1.set_xlabel("L")
    ax1.set_ylabel("rm_mean_late")
    ax1.set_xscale("log")
    manifest.save_fig(fig1, run_dir / "rm_mean_late_vs_L.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(L_list, delta_max_list, marker="o")
    ax2.set_xlabel("L")
    ax2.set_ylabel("delta_max")
    ax2.set_xscale("log")
    manifest.save_fig(fig2, run_dir / "delta_max_vs_L.png")
    plt.close(fig2)

    summary_df = pd.DataFrame(summary_rows)
    manifest.save_table(summary_df, run_dir / "summary.csv")

    metrics = {
        "L_list": L_list,
        "rm_mean_late_list": rm_mean_late_list,
        "delta_max_list": delta_max_list,
        "L_min_delta": L_min_delta,
        "RM_at_L_min_delta": RM_at_L_min_delta,
        "rm_L1": rm_L1,
        "rm_Lmax": rm_Lmax,
        "params": {
            "N": N,
            "steps": steps,
            "dt": dt,
            "g4pi": g4pi,
            "rho_mean": rho_mean,
            "delta_rho": delta_rho,
            "kappa_scale": kappa_scale,
            "seed": seed,
        },
        "delta_summary": "delta_max",
    }

    manifest.write_config(run_dir, metrics["params"])
    manifest.write_metrics(run_dir, metrics)
    manifest.write_provenance(run_dir, "scale_sweep", seed)
    write_run_markdown_stub(run_dir)

    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
