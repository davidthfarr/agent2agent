"""
experiment/metrics.py
---------------------
All metric collection for a single episode and across conditions.

Per-step metrics (recorded every step):
  - mean_pairwise_jsd        : inter-agent epistemic divergence
  - mean_alignment_to_truth  : average belief mass on true target cell(s)
  - any_agent_success        : whether any agent is at a target this step

Per-episode summary metrics:
  - task_success             : did any agent reach the target within T steps?
  - time_to_success          : step at which first success occurred (nan if none)
  - silent_failure           : agents converged but on wrong cell
  - messages_sent            : total recipient-messages sent
  - bytes_transmitted        : total payload bytes sent
  - messages_dropped         : total messages lost to packet loss
  - final_mean_jsd           : mean pairwise JSD at episode end
  - final_mean_alignment     : mean alignment to truth at episode end
  - alignment_per_byte       : final_mean_alignment / bytes_transmitted
    (np.nan if bytes_transmitted == 0)
  - jsd_time_series          : array of per-step mean pairwise JSD  (shape T,)
  - alignment_time_series    : array of per-step mean alignment     (shape T,)

Design note: EpisodeMetrics is a plain dataclass — easy to serialise to
JSON / CSV / numpy without any framework dependency.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from core.belief import mean_pairwise_jsd, alignment_to_truth, silent_failure
from core.agent import Agent


# ---------------------------------------------------------------------------
# Per-episode container
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    """All metrics for a single episode run."""

    # --- task outcome ---
    task_success: bool = False
    time_to_success: float = np.nan      # step index (0-based); nan = failure

    # --- epistemic health ---
    silent_failure: bool = False
    final_mean_jsd: float = np.nan
    final_mean_alignment: float = np.nan

    # --- communication cost ---
    messages_sent: int = 0
    bytes_transmitted: int = 0
    messages_dropped: int = 0

    # --- efficiency ---
    alignment_per_byte: float = np.nan   # final_mean_alignment / bytes_transmitted

    # --- time series (length = actual steps taken) ---
    jsd_time_series: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0)
    )
    alignment_time_series: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0)
    )

    def to_dict(self) -> dict:
        """Flat dictionary for DataFrame / CSV export."""
        return {
            "task_success": int(self.task_success),
            "time_to_success": self.time_to_success,
            "silent_failure": int(self.silent_failure),
            "final_mean_jsd": self.final_mean_jsd,
            "final_mean_alignment": self.final_mean_alignment,
            "messages_sent": self.messages_sent,
            "bytes_transmitted": self.bytes_transmitted,
            "messages_dropped": self.messages_dropped,
            "alignment_per_byte": self.alignment_per_byte,
        }


# ---------------------------------------------------------------------------
# Step-level recorder (called once per step inside the runner)
# ---------------------------------------------------------------------------

class StepRecorder:
    """
    Accumulates per-step snapshots during an episode.
    Call record() once per step, then finalise() at episode end.
    """

    def __init__(self, episode_length: int, min_agents_for_success: int = 1) -> None:
        self._T = episode_length
        self._min_agents = min_agents_for_success
        self._jsd: list[float] = []
        self._alignment: list[float] = []
        self._first_success_step: int | None = None

    def record(
        self,
        step: int,
        agents: list[Agent],
        true_target_cells: list[int],
    ) -> None:
        """
        Record one step's epistemic state.

        Parameters
        ----------
        step       : current step index (0-based)
        agents     : all agents (with current beliefs)
        true_target_cells : ground-truth target locations
        """
        belief_maps = [a.belief for a in agents]

        # Inter-agent divergence
        jsd = mean_pairwise_jsd(belief_maps)
        self._jsd.append(jsd)

        # Alignment to truth
        n_cells = agents[0].belief.n_cells
        alignments = [
            alignment_to_truth(a.belief, true_target_cells, n_cells)
            for a in agents
        ]
        self._alignment.append(float(np.mean(alignments)))

        # First success (check if min_agents have reached target)
        if self._first_success_step is None:
            agents_at_target = sum(1 for a in agents if a.reached_target)
            if agents_at_target >= self._min_agents:
                self._first_success_step = step

    def finalise(
        self,
        agents: list[Agent],
        true_target_cells: list[int],
        network,       # Network instance (for comm stats)
        config,        # CommConfig
        silent_failure_divergence_threshold: float = 0.1,
    ) -> EpisodeMetrics:
        """
        Build the final EpisodeMetrics from accumulated data.
        Call once after the episode loop ends.
        """
        jsd_arr = np.array(self._jsd, dtype=np.float64)
        align_arr = np.array(self._alignment, dtype=np.float64)

        task_success = self._first_success_step is not None
        time_to_success = float(self._first_success_step) if task_success else np.nan

        # Final-step epistemic state
        final_jsd = float(jsd_arr[-1]) if len(jsd_arr) else np.nan
        final_alignment = float(align_arr[-1]) if len(align_arr) else np.nan

        # Silent failure: agents converged on same location but it's wrong
        is_silent_failure = silent_failure(
            belief_maps=[a.belief for a in agents],
            true_target_cells=true_target_cells,
            divergence_threshold=silent_failure_divergence_threshold,
        )

        # Communication cost (from network stats)
        msgs_sent = network.messages_sent_this_episode
        bytes_tx = network.bytes_sent_this_episode
        msgs_dropped = network.messages_dropped_this_episode

        # Efficiency
        if bytes_tx > 0:
            apb = final_alignment / bytes_tx
        else:
            apb = np.nan   # C0: no bytes sent → undefined, not zero

        return EpisodeMetrics(
            task_success=task_success,
            time_to_success=time_to_success,
            silent_failure=is_silent_failure,
            final_mean_jsd=final_jsd,
            final_mean_alignment=final_alignment,
            messages_sent=msgs_sent,
            bytes_transmitted=bytes_tx,
            messages_dropped=msgs_dropped,
            alignment_per_byte=apb,
            jsd_time_series=jsd_arr,
            alignment_time_series=align_arr,
        )


# ---------------------------------------------------------------------------
# Condition-level aggregation
# ---------------------------------------------------------------------------

@dataclass
class ConditionSummary:
    """
    Aggregated statistics across all seeds of one experimental condition.
    All mean/std fields are over the seed distribution.
    """
    condition_name: str

    # Task
    task_success_rate: float = np.nan
    mean_time_to_success: float = np.nan
    std_time_to_success: float = np.nan

    # Epistemic health
    silent_failure_rate: float = np.nan
    silent_failure_rate_given_failure: float = np.nan
    mean_final_jsd: float = np.nan
    std_final_jsd: float = np.nan
    mean_final_alignment: float = np.nan
    std_final_alignment: float = np.nan

    # Communication cost
    mean_messages_sent: float = np.nan
    mean_bytes_transmitted: float = np.nan
    mean_drop_rate: float = np.nan

    # Efficiency
    mean_alignment_per_byte: float = np.nan

    # Full time series: shape (n_seeds, T) — for plotting
    jsd_matrix: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0))
    )
    alignment_matrix: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0))
    )

    def to_dict(self) -> dict:
        return {
            "condition": self.condition_name,
            "task_success_rate": self.task_success_rate,
            "mean_time_to_success": self.mean_time_to_success,
            "std_time_to_success": self.std_time_to_success,
            "silent_failure_rate": self.silent_failure_rate,
            "silent_failure_rate_given_failure": self.silent_failure_rate_given_failure,
            "mean_final_jsd": self.mean_final_jsd,
            "std_final_jsd": self.std_final_jsd,
            "mean_final_alignment": self.mean_final_alignment,
            "std_final_alignment": self.std_final_alignment,
            "mean_messages_sent": self.mean_messages_sent,
            "mean_bytes_transmitted": self.mean_bytes_transmitted,
            "mean_drop_rate": self.mean_drop_rate,
            "mean_alignment_per_byte": self.mean_alignment_per_byte,
        }


# ---------------------------------------------------------------------------
# Condition-level aggregation  (module-level function, NOT inside the class)
# ---------------------------------------------------------------------------

def aggregate_episodes(
    condition_name: str,
    episode_metrics: list[EpisodeMetrics],
    episode_length: int,
    n_cells: int,
) -> ConditionSummary:
    """
    Aggregate a list of EpisodeMetrics into a ConditionSummary.

    Time series are padded / truncated to episode_length so the matrix
    shape is consistent across seeds regardless of early termination.

    Silent failure is re-evaluated here to ensure consistency:
    agents converged on same location (low JSD) but it's wrong (low alignment).
    The alignment threshold scales with grid size: 5/n_cells.
    """
    n = len(episode_metrics)
    if n == 0:
        return ConditionSummary(condition_name=condition_name)

    # Silent failure threshold: 5x the uniform prior, scales with grid size
    _SILENT_FAILURE_ALIGNMENT_MULTIPLIER = 5.0
    alignment_threshold = _SILENT_FAILURE_ALIGNMENT_MULTIPLIER / n_cells

    # --- re-evaluate silent failure: low JSD (agree) + low alignment (wrong) ---
    for m in episode_metrics:
        if not m.task_success:
            m.silent_failure = (
                m.final_mean_jsd < 0.1 and
                m.final_mean_alignment < alignment_threshold
            )
        else:
            m.silent_failure = False

    # --- scalar aggregations (must come AFTER re-evaluation above) ---
    successes = np.array([m.task_success for m in episode_metrics], dtype=float)
    tts = np.array([
        m.time_to_success if m.task_success else np.nan
        for m in episode_metrics
    ])
    sf_flags = np.array(
        [1.0 if (not m.task_success and m.final_mean_jsd < 0.1) else 0.0
         for m in episode_metrics]
    )
    final_jsd = np.array([m.final_mean_jsd for m in episode_metrics])
    final_align = np.array([m.final_mean_alignment for m in episode_metrics])
    msgs = np.array([m.messages_sent for m in episode_metrics], dtype=float)
    bytes_ = np.array([m.bytes_transmitted for m in episode_metrics], dtype=float)
    dropped = np.array([m.messages_dropped for m in episode_metrics], dtype=float)
    # ratio of means (not mean of ratios)
    valid_episodes = [
        m for m in episode_metrics
        if m.bytes_transmitted > 0 and not np.isnan(m.final_mean_alignment)
    ]
    mean_alignment_per_byte = (
        float(np.mean([m.final_mean_alignment for m in valid_episodes])) /
        float(np.mean([m.bytes_transmitted for m in valid_episodes]))
    ) if valid_episodes else np.nan

    # Drop rate per episode (avoid divide-by-zero)
    with np.errstate(invalid="ignore", divide="ignore"):
        drop_rates = np.where(msgs > 0, dropped / msgs, np.nan)

    # --- silent failure: failed episodes where agents converged (low JSD) ---
    failed_episodes = [m for m in episode_metrics if not m.task_success]
    if failed_episodes:
        silent_failures = sum(
            1 for m in failed_episodes if m.final_mean_jsd < 0.1
        )
        sf_rate_given_failure = silent_failures / len(failed_episodes)
    else:
        sf_rate_given_failure = np.nan

    # --- time series (pad shorter episodes with final value) ---
    def _pad_series(arr: NDArray, T: int) -> NDArray:
        if len(arr) == 0:
            return np.full(T, np.nan)
        if len(arr) >= T:
            return arr[:T]
        pad = np.full(T - len(arr), arr[-1])
        return np.concatenate([arr, pad])

    jsd_mat = np.stack([
        _pad_series(m.jsd_time_series, episode_length)
        for m in episode_metrics
    ])  # shape (n_seeds, T)
    align_mat = np.stack([
        _pad_series(m.alignment_time_series, episode_length)
        for m in episode_metrics
    ])

    return ConditionSummary(
        condition_name=condition_name,
        task_success_rate=float(np.mean(successes)),
        mean_time_to_success=float(np.nanmean(tts)) if np.any(~np.isnan(tts)) else np.nan,
        std_time_to_success=float(np.nanstd(tts)) if np.any(~np.isnan(tts)) else np.nan,
        silent_failure_rate=float(np.mean(sf_flags)),
        silent_failure_rate_given_failure=sf_rate_given_failure,
        mean_final_jsd=float(np.nanmean(final_jsd)),
        std_final_jsd=float(np.nanstd(final_jsd)),
        mean_final_alignment=float(np.nanmean(final_align)),
        std_final_alignment=float(np.nanstd(final_align)),
        mean_messages_sent=float(np.mean(msgs)),
        mean_bytes_transmitted=float(np.mean(bytes_)),
        mean_drop_rate=float(np.nanmean(drop_rates)) if np.any(~np.isnan(drop_rates)) else np.nan,
        mean_alignment_per_byte=mean_alignment_per_byte,
        jsd_matrix=jsd_mat,
        alignment_matrix=align_mat,
    )
