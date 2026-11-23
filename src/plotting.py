"""Shared plotting helpers reproducing the figures of Section 4
("Numerical experiments"). Each function saves one PNG into ``results/``
and returns the output path, so every experiment script uses one plotting
implementation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_exact_vs_estimate(p_grid, exact, series: dict, title: str, out_path: Path) -> Path:
    """Figure 1 style plot: exact value v(p) = p vs one or more estimate series."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(p_grid, exact, "k--", label="exact v(p) = p", lw=1.5)
    for label, values in series.items():
        ax.plot(p_grid, values, marker="o", label=label)
    ax.set_xlabel("p")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_projection_bias(grids_out: list, out_path: Path) -> Path:
    """Figure 2 style plot: fixed vs adaptive bound at two grid resolutions."""
    fig, axes = plt.subplots(1, len(grids_out), figsize=(6 * len(grids_out), 5), sharey=True)
    if len(grids_out) == 1:
        axes = [axes]
    for ax, grid in zip(axes, grids_out):
        p = [row["p"] for row in grid["rows"]]
        exact = [row["exact"] for row in grid["rows"]]
        fixed = [row["fixed_estimate"] for row in grid["rows"]]
        adaptive = [row["adaptive_estimate"] for row in grid["rows"]]
        ax.plot(p, exact, "k--", label="exact v(p) = p")
        ax.plot(p, fixed, marker="o", label="fixed bound M")
        ax.plot(p, adaptive, marker="s", label="adaptive bound")
        ax.set_title(f"N = {grid['N']} (M = {grid['M']:.3f})")
        ax.set_xlabel("p")
        ax.legend()
    axes[0].set_ylabel("value")
    fig.suptitle("Toy example: projection bias, fixed vs adaptive bound (Section 4.2, Figure 2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_fixed_vs_adaptive(rows: list, title: str, out_path: Path) -> Path:
    """Figure 3 style plot: reference curve vs fixed/adaptive bound estimates."""
    p = [row["p"] for row in rows]
    reference = [row["reference"] for row in rows]
    fixed = [row["fixed_estimate"] for row in rows]
    adaptive = [row["adaptive_estimate"] for row in rows]
    fixed_se = [row["fixed_stderr"] for row in rows]
    adaptive_se = [row["adaptive_stderr"] for row in rows]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(p, reference, "k--", marker="x", label="Monte Carlo / closed-form reference")
    ax.errorbar(p, fixed, yerr=fixed_se, marker="o", capsize=3, label="fixed bound M")
    ax.errorbar(p, adaptive, yerr=adaptive_se, marker="s", capsize=3, label="adaptive bound")
    ax.set_xlabel("p")
    ax.set_ylabel("quantile hedging value v(0, x, p)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_convergence(grids, mean_err, std_err, slope, intercept, title: str, out_path: Path) -> Path:
    """Figure 4 style plot: mean absolute error vs N (log-log), with the
    fitted slope and the theoretical N^(-1/8) PCPT reference line."""
    grids = np.asarray(grids, dtype=float)
    mean_err = np.asarray(mean_err, dtype=float)
    std_err = np.asarray(std_err, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(grids, mean_err, yerr=std_err, fmt="o", capsize=3, label="mean |error| (3 chains)")
    fit_line = np.exp(intercept) * grids ** slope
    ax.plot(grids, fit_line, "-", label=f"log-log fit, slope={slope:.3f}")
    theo_const = mean_err[0] * grids[0] ** (1.0 / 8.0)
    theo_line = theo_const * grids ** (-1.0 / 8.0)
    ax.plot(grids, theo_line, "--", label="theoretical N^(-1/8) (Corollary 3.16)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("mean absolute pricing error")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
