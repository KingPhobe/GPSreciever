"""Plotting helpers for demo runs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_solution_errors(times_s: np.ndarray, errors_m: np.ndarray) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    labels = ["X", "Y", "Z"]
    for idx, ax in enumerate(axes):
        ax.plot(times_s, errors_m[:, idx])
        ax.set_ylabel(f"{labels[idx]} error (m)")
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Position Error")
    plt.tight_layout()


def plot_residuals(times_s: np.ndarray, residuals_m: np.ndarray) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(times_s, residuals_m)
    plt.xlabel("Time (s)")
    plt.ylabel("Max residual (m)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title("Residual Magnitude")
    plt.tight_layout()
