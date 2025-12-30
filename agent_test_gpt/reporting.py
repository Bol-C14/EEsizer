"""Reporting helpers for optimization runs."""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent_test_gpt.logging_utils import get_logger

_logger = get_logger(__name__)


def run_multiple_optimizations(target_values, sim_netlist, tool_chain, extract_number, optimization, num_runs=1):
    results = []

    for i in range(num_runs):
        result, _opti_netlist = optimization(tool_chain, target_values, sim_netlist, extract_number)
        results.append(result)
    return results


def copy_netlist(source_file: str, destination_file: str):
    """Copy a netlist to a caller-defined destination; no shared defaults to avoid collisions."""
    shutil.copyfile(source_file, destination_file)


def plot_subplot(ax, df, colors, x, y, xlabel, ylabel, ylim_min, ylim_max, fill_range=None, fill_label=None, log_scale=False):
    for batch, group in df.groupby("batch"):
        ax.plot(
            group[x],
            group[y],
            marker="o",
            color=colors[batch],
            label=f"Attempt {batch}",
            markersize=4,
            linewidth=1,
        )
    ax.set_xlim(df[x].min() - 1, df[x].max() + 1)
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(False)
    if fill_range:
        ax.fill_between(ax.get_xlim(), fill_range[0], fill_range[1], color="blue", alpha=0.1, label=fill_label)
    if log_scale:
        ax.set_yscale("log")


def _validate_history_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "iteration",
        "pm_output",
        "ubw_output",
        "tr_gain_output",
        "pr_output",
        "cmrr_output",
        "output_swing_output",
        "thd_output",
        "input_offset_output",
        "icmr_output",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in history csv: {sorted(missing)}")
    return df


def plot_optimization_history(csv_path: str, output_pdf: str, max_batches=5):
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        raise FileNotFoundError(f"History CSV not found or empty: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read history CSV {csv_path}: {exc}") from exc
    df = _validate_history_dataframe(df)
    df["batch"] = (df["iteration"] == 0).cumsum()
    df["bw_output_dB"] = 20 * np.log10(df["ubw_output"] + 1e-9)

    # Filter the DataFrame for the first batches
    filtered_df = df[df["batch"] <= max_batches]

    # Create a 4x2 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.3)

    # Generate a list of colors
    colors = plt.cm.viridis(np.linspace(0, 0.9, filtered_df["batch"].max() + 1))

    # Plot each metric in a subplot
    plot_subplot(axes[0, 1], filtered_df, colors, "iteration", "pm_output", "Iterations", "Phase Margin (°)", -5, 115, fill_range=(50, 150), fill_label="Target Range")
    plot_subplot(axes[0, 2], filtered_df, colors, "iteration", "bw_output_dB", "Iterations", "UGBW (dB)", 90, 20 * np.log10(1e7) + 35, fill_range=(20 * np.log10(1e7), 20 * np.log10(1e7) + 35), fill_label="Target Range")
    plot_subplot(axes[1, 1], filtered_df, colors, "iteration", "tr_gain_output", "Iterations", "Gain (dB)", 15, 85, fill_range=(61.75, 90), fill_label="Target Range")
    plot_subplot(axes[2, 1], filtered_df, colors, "iteration", "pr_output", "Iterations", "Power (W)", -0.0005, 0.015, fill_range=(-0.0005, 0.0105), fill_label="Target Range")
    plot_subplot(axes[1, 2], filtered_df, colors, "iteration", "cmrr_output", "Iterations", "CMRR (dB)", 0, 130, fill_range=(100 * 0.95, 160), fill_label="Target Range")
    plot_subplot(axes[1, 0], filtered_df, colors, "iteration", "output_swing_output", "Iterations", "Output Swing (V)", 0, 1.3, fill_range=(1.2 * 0.95, 1.3), fill_label="Target Range")
    plot_subplot(axes[2, 2], filtered_df, colors, "iteration", "thd_output", "Iterations", "THD (dB)", -50, 0, fill_range=(-60, -24.7), fill_label="Target Range")
    plot_subplot(axes[2, 0], filtered_df, colors, "iteration", "input_offset_output", "Iterations", "Offset (V)", -0.05, 0.2, fill_range=(-0.1, 0.001), fill_label="Target Range")
    plot_subplot(axes[0, 0], filtered_df, colors, "iteration", "icmr_output", "Iterations", "Input Range", 0, 1.3, fill_range=(1.2 * 0.95, 1.3), fill_label="Target Range")

    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
    for i, ax in enumerate(axes.flat):
        ax.text(1, 0.1, labels[i], transform=ax.transAxes, fontsize=14, va="top", ha="right")

    for ax in axes[:-1, :].flat:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    # Create a common legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=6, fontsize=13)

    # Save the figure
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.show()
