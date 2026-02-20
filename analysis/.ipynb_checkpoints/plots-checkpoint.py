"""
analysis/plots.py
-----------------
Publication-quality figures for all research questions.

Figure catalogue
----------------
Fig 1  — JSD time series: epistemic drift over episode steps (RQ1)
Fig 2  — Alignment time series: belief accuracy over steps (RQ1, RQ2)
Fig 3  — Final-state scatter: JSD vs alignment by condition (RQ1, RQ2)
Fig 4  — Silent failure heatmap: rate across loss × latency grid (RQ1)
Fig 5  — Bandwidth vs alignment: efficiency frontier (RQ2, RQ3)
Fig 6  — Message volume: C2 vs C3 across conditions (RQ3)
Fig 7  — Task success & time-to-success: summary panel (all RQs)

All functions accept a dict[str, ConditionSummary] — the direct output of
ExperimentRunner.run() — and an optional output path for saving.

Aesthetic direction: scientific monograph — ink-efficient, high-contrast,
structured for print and screen both. Dark slate background panels with
precise coloration, using a hand-tuned discrete palette tied to comm type.
No chartjunk. Every ink mark earns its place.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from experiment.metrics import ConditionSummary


# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------

# Colour palette — one accent per comm type, consistent across all figures
COMM_COLORS: dict[str, str] = {
    "C0": "#8ecae6",   # pale blue  — no comms baseline
    "C1": "#f4a261",   # amber      — semantic
    "C2": "#e63946",   # crimson    — epistemic
    "C3": "#57cc99",   # mint       — confidence-gated
}

COMM_LABELS: dict[str, str] = {
    "C0": "C₀ None",
    "C1": "C₁ Semantic",
    "C2": "C₂ Epistemic",
    "C3": "C₃ Gated",
}

COMM_MARKERS: dict[str, str] = {
    "C0": "o",
    "C1": "s",
    "C2": "^",
    "C3": "D",
}

# Background and grid colours
BG       = "#0f1117"
PANEL_BG = "#1a1d27"
GRID_CLR = "#2a2d3a"
TEXT_CLR = "#e8eaf0"
TICK_CLR = "#6b7280"
SPINE_CLR = "#3a3d4a"

# Typography
FONT_FAMILY = "monospace"   # clean, technical, no web-font dependency


def _apply_style() -> None:
    """Apply the global matplotlib style."""
    mpl.rcParams.update({
        "figure.facecolor":    BG,
        "axes.facecolor":      PANEL_BG,
        "axes.edgecolor":      SPINE_CLR,
        "axes.labelcolor":     TEXT_CLR,
        "axes.titlecolor":     TEXT_CLR,
        "axes.grid":           True,
        "axes.grid.which":     "major",
        "grid.color":          GRID_CLR,
        "grid.linewidth":      0.6,
        "grid.alpha":          1.0,
        "xtick.color":         TICK_CLR,
        "ytick.color":         TICK_CLR,
        "xtick.labelcolor":    TEXT_CLR,
        "ytick.labelcolor":    TEXT_CLR,
        "text.color":          TEXT_CLR,
        "legend.facecolor":    PANEL_BG,
        "legend.edgecolor":    SPINE_CLR,
        "legend.labelcolor":   TEXT_CLR,
        "figure.dpi":          120,
        "savefig.dpi":         180,
        "savefig.facecolor":   BG,
        "savefig.bbox":        "tight",
        "font.family":         FONT_FAMILY,
        "font.size":           9,
        "axes.titlesize":      10,
        "axes.labelsize":      9,
        "xtick.labelsize":     8,
        "ytick.labelsize":     8,
        "legend.fontsize":     8,
        "lines.linewidth":     1.6,
        "lines.solid_capstyle": "round",
        "patch.linewidth":     0.8,
    })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _comm_type(name: str) -> str:
    """Extract 'C0'/'C1'/'C2'/'C3' from a condition name like 'C2_loss10_lat1'."""
    m = re.match(r"^(C[0-3])", name)
    return m.group(1) if m else "C0"


def _loss_from_name(name: str) -> float:
    """Extract packet loss rate float from condition name."""
    m = re.search(r"loss(\d+)", name)
    return int(m.group(1)) / 100.0 if m else 0.0


def _latency_from_name(name: str) -> int:
    """Extract latency int from condition name."""
    m = re.search(r"lat(\d+)", name)
    return int(m.group(1)) if m else 0


def _comm_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=COMM_COLORS[k], lw=2,
               marker=COMM_MARKERS[k], markersize=6, label=COMM_LABELS[k])
        for k in ["C0", "C1", "C2", "C3"]
    ]


def _save(fig: plt.Figure, path: str | None, suffix: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, f"{suffix}.png"))
    plt.close(fig)


def _stderr_band(ax: plt.Axes, x: np.ndarray, matrix: np.ndarray,
                 color: str, alpha_band: float = 0.15) -> None:
    """Plot mean ± 1 SE band from a (n_seeds × T) matrix."""
    mean = np.nanmean(matrix, axis=0)
    se   = stats.sem(matrix, axis=0, nan_policy="omit")
    ax.plot(x, mean, color=color, linewidth=1.8)
    ax.fill_between(x, mean - se, mean + se, color=color, alpha=alpha_band)


# ---------------------------------------------------------------------------
# Figure 1 — JSD time series   (RQ1: epistemic drift over episode)
# ---------------------------------------------------------------------------

def fig_jsd_timeseries(
    results: dict[str, ConditionSummary],
    output_path: str | None = None,
    loss_filter: float | None = 0.0,
    latency_filter: int | None = 0,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Mean ± SE inter-agent Jensen-Shannon divergence over episode steps,
    one line per communication type.

    Parameters
    ----------
    loss_filter    : if set, only include conditions with this loss rate.
    latency_filter : if set, only include conditions with this latency.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    plotted: set[str] = set()
    for name, summary in results.items():
        ct = _comm_type(name)
        if loss_filter is not None and abs(_loss_from_name(name) - loss_filter) > 1e-6:
            continue
        if latency_filter is not None and _latency_from_name(name) != latency_filter:
            continue
        if summary.jsd_matrix.shape[0] == 0:
            continue

        T = summary.jsd_matrix.shape[1]
        x = np.arange(T)
        _stderr_band(ax, x, summary.jsd_matrix, COMM_COLORS[ct])
        plotted.add(ct)

    ax.set_xlabel("Episode step")
    ax.set_ylabel("Mean pairwise JSD (nats)")
    cond_str = ""
    if loss_filter is not None:
        cond_str += f"  loss={int(loss_filter*100)}%"
    if latency_filter is not None:
        cond_str += f"  latency={latency_filter}"
    ax.set_title(f"Inter-agent Epistemic Divergence over Time{cond_str}{title_suffix}")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    legend_handles = [h for h in _comm_legend_handles() if h.get_label().split()[0].replace("₀","0").replace("₁","1").replace("₂","2").replace("₃","3") in [f"C{ct[-1]}" for ct in plotted] or True]
    ax.legend(handles=_comm_legend_handles(), loc="upper right", framealpha=0.9)

    fig.tight_layout()
    _save(fig, output_path, "fig1_jsd_timeseries")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Alignment time series   (RQ1, RQ2)
# ---------------------------------------------------------------------------

def fig_alignment_timeseries(
    results: dict[str, ConditionSummary],
    output_path: str | None = None,
    loss_filter: float | None = 0.0,
    latency_filter: int | None = 0,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Mean ± SE alignment-to-truth over episode steps.
    Higher = agents' beliefs are better calibrated to ground truth.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    for name, summary in results.items():
        ct = _comm_type(name)
        if loss_filter is not None and abs(_loss_from_name(name) - loss_filter) > 1e-6:
            continue
        if latency_filter is not None and _latency_from_name(name) != latency_filter:
            continue
        if summary.alignment_matrix.shape[0] == 0:
            continue

        T = summary.alignment_matrix.shape[1]
        x = np.arange(T)
        _stderr_band(ax, x, summary.alignment_matrix, COMM_COLORS[ct])

    ax.set_xlabel("Episode step")
    ax.set_ylabel("Mean alignment to truth  P(target | belief)")
    cond_str = ""
    if loss_filter is not None:
        cond_str += f"  loss={int(loss_filter*100)}%"
    if latency_filter is not None:
        cond_str += f"  latency={latency_filter}"
    ax.set_title(f"Belief Alignment to Ground Truth{cond_str}{title_suffix}")
    ax.set_ylim(bottom=0)
    ax.legend(handles=_comm_legend_handles(), loc="upper left", framealpha=0.9)

    fig.tight_layout()
    _save(fig, output_path, "fig2_alignment_timeseries")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — JSD vs Alignment scatter   (RQ1, RQ2)
# ---------------------------------------------------------------------------

def fig_jsd_vs_alignment(
    results: dict[str, ConditionSummary],
    output_path: str | None = None,
) -> plt.Figure:
    """
    Final-state scatter: x = mean pairwise JSD, y = mean alignment to truth.
    Each point is one condition; marker encodes comm type, opacity encodes loss rate.

    Quadrant annotations highlight the key outcome zones:
      - Low JSD + High Alignment  → ideal (agents agree AND are right)
      - Low JSD + Low Alignment   → silent failure (agents agree but are wrong)
      - High JSD + High Alignment → divergent but individually accurate
      - High JSD + Low Alignment  → divergent and wrong
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    all_jsd = [s.mean_final_jsd for s in results.values() if not np.isnan(s.mean_final_jsd)]
    all_align = [s.mean_final_alignment for s in results.values() if not np.isnan(s.mean_final_alignment)]

    if not all_jsd:
        ax.set_title("No data")
        _save(fig, output_path, "fig3_jsd_vs_alignment")
        return fig

    x_max = max(all_jsd) * 1.15 if all_jsd else 0.7
    y_max = max(all_align) * 1.15 if all_align else 1.0
    x_mid = x_max / 2
    y_mid = y_max / 2

    # Quadrant shading
    quad_alpha = 0.06
    ax.axhspan(y_mid, y_max * 1.1, xmin=0, xmax=x_mid / (x_max * 1.1),
               color=COMM_COLORS["C3"], alpha=quad_alpha)   # low JSD, high align (good)
    ax.axhspan(0, y_mid, xmin=0, xmax=x_mid / (x_max * 1.1),
               color=COMM_COLORS["C2"], alpha=quad_alpha)   # silent failure zone

    # Quadrant labels
    label_kw = dict(fontsize=7, alpha=0.45, va="center", ha="center",
                    style="italic", color=TEXT_CLR)
    ax.text(x_mid * 0.5, y_max * 0.88, "aligned\nconsensus", **label_kw)
    ax.text(x_mid * 0.5, y_max * 0.12, "silent\nfailure", **label_kw)
    ax.text(x_mid * 1.5, y_max * 0.88, "accurate\nindividuals", **label_kw)
    ax.text(x_mid * 1.5, y_max * 0.12, "lost\ncollective", **label_kw)

    # Divider lines
    ax.axvline(x_mid, color=SPINE_CLR, lw=0.8, ls="--", alpha=0.5)
    ax.axhline(y_mid, color=SPINE_CLR, lw=0.8, ls="--", alpha=0.5)

    # Plot each condition
    for name, summary in results.items():
        if np.isnan(summary.mean_final_jsd) or np.isnan(summary.mean_final_alignment):
            continue
        ct = _comm_type(name)
        loss = _loss_from_name(name)
        # Opacity scales with packet loss: lossless is fully opaque
        alpha = 1.0 - loss * 0.5

        ax.scatter(
            summary.mean_final_jsd,
            summary.mean_final_alignment,
            color=COMM_COLORS[ct],
            marker=COMM_MARKERS[ct],
            s=70,
            alpha=alpha,
            edgecolors="none",
            zorder=3,
        )

        # Error bars (std)
        ax.errorbar(
            summary.mean_final_jsd,
            summary.mean_final_alignment,
            xerr=summary.std_final_jsd,
            yerr=summary.std_final_alignment,
            fmt="none",
            color=COMM_COLORS[ct],
            alpha=alpha * 0.4,
            linewidth=0.8,
            capsize=2,
        )

    ax.set_xlabel("Mean pairwise JSD (nats)  ←  more agreement")
    ax.set_ylabel("Mean alignment to truth  →  more accurate")
    ax.set_title("Epistemic State at Episode End: Divergence vs Accuracy")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Legend: comm types + loss note
    handles = _comm_legend_handles()
    handles.append(Line2D([0], [0], color="none", label=""))
    handles.append(Patch(facecolor="none", edgecolor=TICK_CLR,
                         label="opacity = 1 − loss_rate"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    _save(fig, output_path, "fig3_jsd_vs_alignment")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Silent failure heatmap   (RQ1)
# ---------------------------------------------------------------------------

def fig_silent_failure_heatmap(
    results: dict[str, ConditionSummary],
    output_path: str | None = None,
) -> plt.Figure:
    """
    2×2 grid of heatmaps (one per comm type): silent failure rate as a
    function of packet loss rate × latency.  Reveals where hidden
    coordination failures cluster.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    axes = axes.flatten()

    loss_vals = sorted({_loss_from_name(n) for n in results})
    lat_vals  = sorted({_latency_from_name(n) for n in results})

    for idx, ct in enumerate(["C0", "C1", "C2", "C3"]):
        ax = axes[idx]

        # Build rate matrix: rows=latency, cols=loss
        rate_matrix = np.full((len(lat_vals), len(loss_vals)), np.nan)
        for name, summary in results.items():
            if _comm_type(name) != ct:
                continue
            li = loss_vals.index(_loss_from_name(name))
            lati = lat_vals.index(_latency_from_name(name))
            rate_matrix[lati, li] = summary.silent_failure_rate

        # Custom colormap: PANEL_BG → accent colour
        base_color = mpl.colors.to_rgb(COMM_COLORS[ct])
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            f"cmap_{ct}",
            [mpl.colors.to_rgb(PANEL_BG), base_color],
        )

        im = ax.imshow(
            rate_matrix,
            aspect="auto",
            vmin=0, vmax=1,
            cmap=cmap,
            origin="lower",
        )

        # Annotate cells
        for r in range(len(lat_vals)):
            for c in range(len(loss_vals)):
                v = rate_matrix[r, c]
                if not np.isnan(v):
                    txt_color = BG if v > 0.5 else TEXT_CLR
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=8.5, color=txt_color, fontweight="bold")

        ax.set_xticks(range(len(loss_vals)))
        ax.set_xticklabels([f"{int(l*100)}%" for l in loss_vals])
        ax.set_yticks(range(len(lat_vals)))
        ax.set_yticklabels([f"{lat}" for lat in lat_vals])
        ax.set_xlabel("Packet loss rate")
        ax.set_ylabel("Latency (steps)")
        ax.set_title(f"{COMM_LABELS[ct]}", color=COMM_COLORS[ct], fontweight="bold")
        ax.grid(False)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Silent failure rate", fontsize=7)
        cb.ax.yaxis.set_tick_params(color=TICK_CLR, labelcolor=TEXT_CLR)

    fig.suptitle("Silent Failure Rate  (agents converge but on wrong target)",
                 fontsize=11)
    _save(fig, output_path, "fig4_silent_failure_heatmap")
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Bandwidth efficiency frontier   (RQ2, RQ3)
# ---------------------------------------------------------------------------

def fig_bandwidth_efficiency(
    results: dict[str, ConditionSummary],
    output_path: str | None = None,
) -> plt.Figure:
    """
    Alignment-per-byte vs bytes transmitted: the efficiency frontier.

    Each point is one condition.  The Pareto-optimal frontier (highest
    alignment per byte at each byte budget) is annotated.  Reveals
    whether epistemic encoding buys more alignment per bit than semantic.
    """
    _apply_style()
    fig, (ax_main, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 5), constrained_layout=True,
        gridspec_kw={"width_ratios": [2, 1]}
    )

    # Collect per-condition data
    ct_data: dict[str, list[tuple[float, float, str]]] = {k: [] for k in ["C1","C2","C3"]}

    for name, summary in results.items():
        ct = _comm_type(name)
        if ct == "C0":
            continue
        if np.isnan(summary.mean_bytes_transmitted) or np.isnan(summary.mean_alignment_per_byte):
            continue
        if summary.mean_bytes_transmitted == 0:
            continue
        ct_data[ct].append((
            summary.mean_bytes_transmitted,
            summary.mean_alignment_per_byte,
            name,
        ))

    for ct, points in ct_data.items():
        if not points:
            continue
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        ax_main.scatter(xs, ys, color=COMM_COLORS[ct], marker=COMM_MARKERS[ct],
                        s=60, alpha=0.85, zorder=3, label=COMM_LABELS[ct])

    # Pareto frontier across all points
    all_pts = [
        (summary.mean_bytes_transmitted, summary.mean_alignment_per_byte)
        for summary in results.values()
        if not np.isnan(summary.mean_alignment_per_byte)
        and summary.mean_bytes_transmitted > 0
    ]
    if all_pts:
        all_pts_sorted = sorted(all_pts, key=lambda p: p[0])
        pareto = []
        best_y = -np.inf
        for x, y in all_pts_sorted:
            if y > best_y:
                pareto.append((x, y))
                best_y = y
        if len(pareto) > 1:
            px, py = zip(*pareto)
            ax_main.step(px, py, where="post", color=TEXT_CLR, lw=1.0,
                         ls="--", alpha=0.35, label="Pareto frontier")

    ax_main.set_xlabel("Mean bytes transmitted per episode")
    ax_main.set_ylabel("Alignment per byte  (alignment / bytes)")
    ax_main.set_title("Communication Efficiency Frontier")
    ax_main.legend(framealpha=0.9)
    ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))

    # Bar chart: mean alignment-per-byte by comm type (no-loss, no-latency)
    bar_vals, bar_errs, bar_colors, bar_labels = [], [], [], []
    for ct in ["C1", "C2", "C3"]:
        matching = [
            s for n, s in results.items()
            if _comm_type(n) == ct
            and abs(_loss_from_name(n)) < 1e-6
            and _latency_from_name(n) == 0
            and not np.isnan(s.mean_alignment_per_byte)
        ]
        if matching:
            vals = [s.mean_alignment_per_byte for s in matching]
            bar_vals.append(np.mean(vals))
            bar_errs.append(np.std(vals))
            bar_colors.append(COMM_COLORS[ct])
            bar_labels.append(COMM_LABELS[ct])

    if bar_vals:
        bars = ax_bar.bar(range(len(bar_vals)), bar_vals, color=bar_colors,
                          width=0.6, alpha=0.85, zorder=3)
        ax_bar.errorbar(range(len(bar_vals)), bar_vals, yerr=bar_errs,
                        fmt="none", color=TEXT_CLR, capsize=4, lw=1.2)
        ax_bar.set_xticks(range(len(bar_labels)))
        ax_bar.set_xticklabels(bar_labels, rotation=15, ha="right")
        ax_bar.set_ylabel("Alignment per byte")
        ax_bar.set_title("Efficiency (no loss, no latency)")
        ax_bar.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
        ax_bar.set_ylim(bottom=0)

    _save(fig, output_path, "fig5_bandwidth_efficiency")
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Message volume: C2 vs C3   (RQ3)
# ---------------------------------------------------------------------------

def fig_message_volume(
    results: dict[str, ConditionSummary],
    output_path: str | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: mean messages sent per episode for C2 and C3,
    faceted by packet loss rate.  Also shows mean JSD as a line overlay
    to reveal whether reduced message volume comes at a cost to alignment.

    This is the core RQ3 figure: does C3 achieve similar JSD to C2
    while sending substantially fewer messages?
    """
    _apply_style()
    loss_vals = sorted({_loss_from_name(n) for n in results})
    n_loss = len(loss_vals)

    fig, (ax_msgs, ax_jsd) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True,
        gridspec_kw={"hspace": 0.08}
    )

    bar_width = 0.35
    x = np.arange(n_loss)

    for ct_offset, ct in [(-0.5, "C2"), (0.5, "C3")]:
        msgs_means, msgs_errs = [], []
        jsd_means, jsd_errs = [], []

        for loss in loss_vals:
            matching = [
                s for n, s in results.items()
                if _comm_type(n) == ct
                and abs(_loss_from_name(n) - loss) < 1e-6
            ]
            if matching:
                msg_vals = [s.mean_messages_sent for s in matching]
                jsd_vals_l = [s.mean_final_jsd for s in matching if not np.isnan(s.mean_final_jsd)]
                msgs_means.append(np.mean(msg_vals))
                msgs_errs.append(np.std(msg_vals))
                jsd_means.append(np.mean(jsd_vals_l) if jsd_vals_l else np.nan)
                jsd_errs.append(np.std(jsd_vals_l) if len(jsd_vals_l) > 1 else 0.0)
            else:
                msgs_means.append(np.nan); msgs_errs.append(0)
                jsd_means.append(np.nan); jsd_errs.append(0)

        xs = x + ct_offset * bar_width
        color = COMM_COLORS[ct]

        ax_msgs.bar(xs, msgs_means, width=bar_width, color=color,
                    alpha=0.85, label=COMM_LABELS[ct], zorder=3)
        ax_msgs.errorbar(xs, msgs_means, yerr=msgs_errs,
                         fmt="none", color=TEXT_CLR, capsize=3, lw=1.0)

        ax_jsd.plot(xs, jsd_means, color=color, marker=COMM_MARKERS[ct],
                    markersize=7, lw=1.6, label=COMM_LABELS[ct], zorder=3)
        ax_jsd.errorbar(xs, jsd_means, yerr=jsd_errs,
                        fmt="none", color=color, alpha=0.4, capsize=3, lw=0.9)

    ax_msgs.set_ylabel("Mean messages sent")
    ax_msgs.set_title("C₂ vs C₃: Message Volume and Epistemic Divergence")
    ax_msgs.legend(framealpha=0.9)
    ax_msgs.set_ylim(bottom=0)

    ax_jsd.set_ylabel("Final mean pairwise JSD (nats)")
    ax_jsd.set_xlabel("Packet loss rate")
    ax_jsd.set_xticks(x)
    ax_jsd.set_xticklabels([f"{int(l*100)}%" for l in loss_vals])
    ax_jsd.set_ylim(bottom=0)
    ax_jsd.legend(framealpha=0.9)

    # Annotation: reduction ratio
    for i, loss in enumerate(loss_vals):
        c2_match = [s for n, s in results.items()
                    if _comm_type(n) == "C2" and abs(_loss_from_name(n) - loss) < 1e-6]
        c3_match = [s for n, s in results.items()
                    if _comm_type(n) == "C3" and abs(_loss_from_name(n) - loss) < 1e-6]
        if c2_match and c3_match:
            c2_msgs = np.mean([s.mean_messages_sent for s in c2_match])
            c3_msgs = np.mean([s.mean_messages_sent for s in c3_match])
            if c2_msgs > 0 and not np.isnan(c3_msgs):
                reduction = (1 - c3_msgs / c2_msgs) * 100
                ax_msgs.annotate(
                    f"−{reduction:.0f}%",
                    xy=(i, max(c2_msgs, c3_msgs)),
                    xytext=(i, max(c2_msgs, c3_msgs) * 1.05),
                    ha="center", fontsize=7.5, color=TEXT_CLR, alpha=0.8,
                )

    fig.tight_layout()
    _save(fig, output_path, "fig6_message_volume")
    return fig


# ---------------------------------------------------------------------------
# Figure 7 — Summary panel: task outcomes   (all RQs)
# ---------------------------------------------------------------------------

def fig_task_outcomes(
    results: dict[str, ConditionSummary],
    output_path: str | None = None,
) -> plt.Figure:
    """
    Three-panel summary of task-level outcomes:

    Left:   Task success rate by comm type and loss rate (grouped bars)
    Centre: Time to success CDF by comm type (no-loss, no-latency)
    Right:  Task success rate vs silent failure rate scatter
    """
    _apply_style()
    fig, (ax_bar, ax_cdf, ax_scatter) = plt.subplots(
        1, 3, figsize=(14, 5), constrained_layout=True,
        gridspec_kw={"wspace": 0.35}
    )

    loss_vals = sorted({_loss_from_name(n) for n in results})
    n_loss = len(loss_vals)
    comm_types = ["C0", "C1", "C2", "C3"]
    bar_width = 0.8 / len(comm_types)

    # --- Left: success rate grouped bars ---
    for ci, ct in enumerate(comm_types):
        offset = (ci - (len(comm_types) - 1) / 2) * bar_width
        success_by_loss = []
        for loss in loss_vals:
            matching = [
                s for n, s in results.items()
                if _comm_type(n) == ct and abs(_loss_from_name(n) - loss) < 1e-6
            ]
            sr = np.mean([s.task_success_rate for s in matching]) if matching else np.nan
            success_by_loss.append(sr)

        x = np.arange(n_loss) + offset
        ax_bar.bar(x, success_by_loss, width=bar_width, color=COMM_COLORS[ct],
                   alpha=0.85, label=COMM_LABELS[ct], zorder=3)

    ax_bar.set_xlabel("Packet loss rate")
    ax_bar.set_ylabel("Task success rate")
    ax_bar.set_title("Success Rate by\nComm Type & Loss")
    ax_bar.set_xticks(np.arange(n_loss))
    ax_bar.set_xticklabels([f"{int(l*100)}%" for l in loss_vals])
    ax_bar.set_ylim(0, 1.12)
    ax_bar.axhline(1.0, color=SPINE_CLR, lw=0.6, ls=":")
    ax_bar.legend(framealpha=0.9, fontsize=7.5)

    # --- Centre: TTS CDF (zero-loss, zero-latency) ---
    # Note: We only have ONE condition per comm type with (loss=0, latency=0),
    # so this will show at most 4 points (one per comm type).
    # If you have multiple seeds, each provides one mean_time_to_success value.
    for ct in comm_types:
        matching = {
            n: s for n, s in results.items()
            if _comm_type(n) == ct
            and abs(_loss_from_name(n)) < 1e-6
            and _latency_from_name(n) == 0
        }
        # Collect TTS values (mean across seeds for each matching condition)
        tts_vals = []
        for n, s in matching.items():
            tts = s.mean_time_to_success
            if not np.isnan(tts):
                tts_vals.append(tts)

        if len(tts_vals) > 1:  # Need at least 2 points for a meaningful CDF
            sorted_tts = np.sort(tts_vals)
            cdf = np.arange(1, len(sorted_tts) + 1) / len(sorted_tts)
            ax_cdf.step(sorted_tts, cdf, color=COMM_COLORS[ct],
                        where="post", lw=2.0, label=COMM_LABELS[ct])
        elif len(tts_vals) == 1:
            # Only one point - show as a single marker
            ax_cdf.plot(tts_vals[0], 1.0, marker='o', markersize=8,
                       color=COMM_COLORS[ct], label=COMM_LABELS[ct])

    ax_cdf.set_xlabel("Time to first success (steps)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("Time-to-Success CDF\n(no loss, no latency)")
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.set_xlim(left=0)
    
    # Only show legend if we actually plotted something
    handles, labels = ax_cdf.get_legend_handles_labels()
    if handles:
        ax_cdf.legend(framealpha=0.9)
    else:
        ax_cdf.text(0.5, 0.5, "Insufficient data\n(need multiple conditions per comm type)",
                   ha='center', va='center', transform=ax_cdf.transAxes,
                   fontsize=10, color='#6b7280')

    # --- Right: success vs silent failure scatter ---
    for name, summary in results.items():
        ct = _comm_type(name)
        if np.isnan(summary.task_success_rate) or np.isnan(summary.silent_failure_rate):
            continue
        loss = _loss_from_name(name)
        alpha = max(0.25, 1.0 - loss * 0.7)
        ax_scatter.scatter(
            summary.task_success_rate,
            summary.silent_failure_rate,
            color=COMM_COLORS[ct],
            marker=COMM_MARKERS[ct],
            s=55, alpha=alpha, zorder=3,
        )

    # Ideal corner annotation
    ax_scatter.annotate("← ideal →",
                        xy=(1.0, 0.0), xytext=(0.72, 0.07),
                        fontsize=7.5, color=COMM_COLORS["C3"],
                        arrowprops=dict(arrowstyle="->", color=COMM_COLORS["C3"],
                                        lw=0.9, alpha=0.6),
                        alpha=0.7)

    ax_scatter.set_xlabel("Task success rate  (higher = better)")
    ax_scatter.set_ylabel("Silent failure rate  (lower = better)")
    ax_scatter.set_title("Success vs Silent Failure\nTrade-off by Condition")
    ax_scatter.set_xlim(-0.05, 1.1)
    ax_scatter.set_ylim(-0.05, 1.1)
    ax_scatter.legend(handles=_comm_legend_handles(), framealpha=0.9, fontsize=7.5)

    fig.suptitle("Task Outcome Summary Across All Experimental Conditions",
                 fontsize=11)
    _save(fig, output_path, "fig7_task_outcomes")
    return fig


# ---------------------------------------------------------------------------
# Convenience: render all figures
# ---------------------------------------------------------------------------

def plot_all(
    results: dict[str, ConditionSummary],
    output_path: str | None = None,
    show: bool = True,
) -> dict[str, plt.Figure]:
    """
    Generate all seven figures from an experiment results dict.

    Parameters
    ----------
    results     : output of ExperimentRunner.run()
    output_path : directory to save PNGs; None = don't save
    show        : if True, call plt.show() after generating all figures

    Returns
    -------
    dict mapping figure name → Figure object
    """
    figs: dict[str, plt.Figure] = {}

    print("Generating figures...")
    figs["jsd_timeseries"]       = fig_jsd_timeseries(results, output_path)
    figs["alignment_timeseries"] = fig_alignment_timeseries(results, output_path)
    figs["jsd_vs_alignment"]     = fig_jsd_vs_alignment(results, output_path)
    figs["silent_failure"]       = fig_silent_failure_heatmap(results, output_path)
    figs["bandwidth_efficiency"] = fig_bandwidth_efficiency(results, output_path)
    figs["message_volume"]       = fig_message_volume(results, output_path)
    figs["task_outcomes"]        = fig_task_outcomes(results, output_path)
    print(f"  7 figures generated{' and saved to ' + output_path if output_path else ''}.")

    if show:
        plt.show()

    return figs