"""Timing, cost, accuracy and charts for the guardrail benchmark."""

import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# USD per 1M tokens, standard tier, checked October 2026.
# If you switch models in the notebook, add its prices here.
PRICES = {
    "Jev": {"input": 0.042, "output": 0.0},
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
}

COLORS = ["#2a78d6", "#eb6834"]  # one per system, in the order they were run
INK = "#52514e"
GRID = "#e4e3df"


def run(check, dataset: pd.DataFrame, system: str) -> pd.DataFrame:
    """Call `check` on every message, one at a time, recording latency, tokens and attack scores.

    `check(message)` must return {"scores": {attack: 0-1}, "input_tokens": int, "output_tokens": int}.
    """
    rows = []
    for message, label in zip(dataset.text, dataset.label):
        start = time.perf_counter()
        result = check(message)
        latency_ms = (time.perf_counter() - start) * 1000
        rows.append({
            "system": system,
            "label": label,
            "text": message,
            "latency_ms": latency_ms,
            "risk": max(result["scores"].values()),
            **result["scores"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
        })
    return pd.DataFrame(rows)


def summary(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per system: latency, cost, and how well it separates attacks from harmless messages."""
    rows = []
    for system, df in results.groupby("system", sort=False):
        price = PRICES[system]
        cost = (df.input_tokens * price["input"] + df.output_tokens * price["output"]) / 1e6
        blocked = df.risk >= threshold
        is_attack = df.label != "benign"
        rows.append({
            "system": system,
            "median latency (ms)": df.latency_ms.median(),
            "p95 latency (ms)": df.latency_ms.quantile(0.95),
            "cost per 1M requests ($)": cost.mean() * 1e6,
            "attacks caught (%)": 100 * blocked[is_attack].mean(),
            "harmless blocked (%)": 100 * blocked[~is_attack].mean(),
            "accuracy (%)": 100 * (blocked == is_attack).mean(),
        })
    return pd.DataFrame(rows).set_index("system").round(1)


def blocked_by_label(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Percentage of messages blocked, per label and system."""
    return (
        results.assign(blocked=results.risk >= threshold)
        .pivot_table(index="label", columns="system", values="blocked", aggfunc="mean", sort=False)
        .mul(100)
        .round(0)
    )


def plot_latency(results: pd.DataFrame):
    """Every request as a dot, one row per system, with the median marked."""
    systems = list(results.system.unique())
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 1 + 1.1 * len(systems)))

    for i, system in enumerate(systems):
        latency = results.loc[results.system == system, "latency_ms"]
        jitter = rng.uniform(-0.15, 0.15, len(latency))
        ax.scatter(latency, i + jitter, s=36, alpha=0.5, color=COLORS[i], linewidths=0)
        median = latency.median()
        ax.plot([median, median], [i - 0.3, i + 0.3], color=INK, linewidth=2)
        ax.annotate(f"median {median:,.0f} ms", (median, i + 0.32), ha="center", va="bottom", fontsize=9, color=INK)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_yticks(range(len(systems)), systems)
    ax.set_ylim(-0.6, len(systems) - 0.2)
    ax.set_xlabel("Latency per request (ms, log scale)")
    ax.set_title("Guardrail latency", loc="left")
    _style(ax)
    ax.grid(axis="y", visible=False)
    plt.show()


def plot_tradeoff(results: pd.DataFrame, threshold: float):
    """Attacks caught vs. harmless messages blocked as the block cutoff moves."""
    cutoffs = np.linspace(0.01, 0.99, 99)
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, (system, df) in enumerate(results.groupby("system", sort=False)):
        attacks, harmless = df.risk[df.label != "benign"], df.risk[df.label == "benign"]
        # Probabilities give a curve; true/false answers give a single point.
        if df.risk.nunique() > 2:
            ax.plot(
                [100 * (harmless >= c).mean() for c in cutoffs],
                [100 * (attacks >= c).mean() for c in cutoffs],
                color=COLORS[i], linewidth=2, label=f"{system} (any cutoff)",
            )
        x, y = 100 * (harmless >= threshold).mean(), 100 * (attacks >= threshold).mean()
        ax.scatter(x, y, s=80, color=COLORS[i], edgecolors="white", linewidths=2, zorder=3, label=f"{system} at {threshold}")
        ax.annotate(system, (x, y), xytext=(8, -12), textcoords="offset points", fontsize=9, color=INK)

    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Harmless messages blocked (%)")
    ax.set_ylabel("Attacks caught (%)")
    ax.set_title("Catching attacks vs. blocking real customers", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _style(ax)
    plt.show()


def _style(ax):
    """Light grid, no box, muted text."""
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK, length=0)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color("#0b0b0b")
