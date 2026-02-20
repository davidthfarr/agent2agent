"""
experiment/runner.py
--------------------
Episode runner and condition-level orchestration.

Architecture
------------
EpisodeRunner.run(seed)
  → constructs world, agents, sensor, network fresh each episode
  → runs the step loop: observe → communicate → fuse → move
  → records metrics via StepRecorder
  → returns EpisodeMetrics

ConditionRunner.run(condition)
  → runs EpisodeRunner for each seed in range(num_seeds)
  → aggregates into ConditionSummary
  → optionally prints progress

ExperimentRunner.run(conditions)
  → runs ConditionRunner for each condition
  → returns dict[condition_name → ConditionSummary]

Step loop order (per agent, per step):
  1. observe_and_update   — Bayesian update from noisy FOV
  2. build_message        — construct outgoing message (comm type determines content)
  3. network.send         — register message in channel
  4. network.deliver      — resolve packet loss + latency; get inbox
  5. apply_all_deliveries — fuse received beliefs
  6. move                 — greedy step toward belief argmax
  7. check_success        — did any agent reach the target?
  8. StepRecorder.record  — snapshot metrics

Note: all agents observe and send before anyone fuses. This models
synchronous communication — beliefs are updated on the *previous*
step's messages, not the current one. This is standard for discrete-time
multi-agent sims and avoids ordering artifacts.
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass

from config import SimConfig
from core.world import make_world
from core.agent import make_agents, Agent
from core.observation import NoisySensor
from comms.message import build_message, CommType
from comms.network import make_network
from comms.fusion import apply_all_deliveries
from experiment.conditions import Condition
from experiment.metrics import (
    EpisodeMetrics,
    StepRecorder,
    ConditionSummary,
    aggregate_episodes,
)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

class EpisodeRunner:
    """
    Runs a single episode of the simulation for a given seed and condition.

    A new World, Agent list, NoisySensor, and Network are constructed from
    scratch each call, ensuring complete seed isolation between episodes.
    """

    def __init__(self, config: SimConfig, comm_type: CommType) -> None:
        self.config = config
        self.comm_type = comm_type

    def run(self, seed: int) -> EpisodeMetrics:
        """
        Execute one episode.

        Parameters
        ----------
        seed : int
            Master seed for this episode.  Derived sub-seeds are used for
            world generation, sensor noise, and network packet loss so that
            each component is independently reproducible.

        Returns
        -------
        EpisodeMetrics for this episode.
        """
        cfg = self.config
        T = cfg.experiment.episode_length

        # --- Seed splitting: each component gets its own derived seed ---
        rng_master = np.random.default_rng(seed)
        seed_world, seed_sensor, seed_network = (
            int(rng_master.integers(0, 2**31)) for _ in range(3)
        )

        # --- Construct world ---
        world = make_world(cfg.world, seed=seed_world,
                           num_agents=cfg.agents.num_agents)

        # --- Construct agents ---
        agents = make_agents(cfg.agents, world)

        # --- Construct sensor ---
        sensor_rng = np.random.default_rng(seed_sensor)
        sensor = NoisySensor(cfg.obs, sensor_rng)

        # --- Construct network ---
        network = make_network(cfg.comms, cfg.agents.num_agents, seed=seed_network)

        # --- Pre-compute max entropy for fusion weighting ---
        n_cells = world.N ** 2
        max_entropy = float(np.log(n_cells))

        # --- Metrics recorder ---
        recorder = StepRecorder(
            episode_length=T,
            min_agents_for_success=cfg.experiment.min_agents_for_success,
        )

        # --- Episode loop ---
        done = False
        for step in range(T):
            # 1. All agents observe and update their own beliefs
            for agent in agents:
                agent.observe_and_update(world, sensor, cfg.obs)

            # 2. All agents build and send their outgoing messages
            #    (synchronous: everyone sends based on post-observation beliefs)
            for agent in agents:
                msg = build_message(
                    agent,
                    comm_type=self.comm_type,
                    top_k=cfg.comms.top_k,
                    entropy_delta_threshold=cfg.comms.entropy_delta_threshold,
                )
                if msg is not None:
                    network.send(msg, sender_id=agent.agent_id, current_step=step)

            # 3. Deliver messages (apply packet loss + latency)
            inbox = network.deliver(current_step=step)

            # 4. Fuse received messages into each agent's belief
            apply_all_deliveries(agents, inbox, cfg.comms, max_entropy=max_entropy)

            # 5. All agents move
            for agent in agents:
                agent.move(world)

            # 6. Check for success (require min_agents_for_success at target)
            agents_at_target = sum(1 for agent in agents if agent.check_success(world))
            if agents_at_target >= cfg.experiment.min_agents_for_success:
                done = True

            # 7. Record this step's metrics
            recorder.record(step, agents, world.target_cells)

            if done:
                break

        # --- Finalise metrics ---
        metrics = recorder.finalise(
            agents=agents,
            true_target_cells=world.target_cells,
            network=network,
            config=cfg.comms,
        )
        return metrics


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------

class ConditionRunner:
    """
    Runs all seeds for a single experimental condition.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def run(self, condition: Condition) -> ConditionSummary:
        """
        Execute all seeds for `condition` and return aggregated metrics.

        Seeds are drawn from range(num_seeds) for reproducibility.
        """
        cfg = condition.config
        n_seeds = cfg.experiment.num_seeds
        T = cfg.experiment.episode_length

        runner = EpisodeRunner(config=cfg, comm_type=condition.comm_type)
        episode_results: list[EpisodeMetrics] = []

        t_start = time.perf_counter()

        for seed in range(n_seeds):
            metrics = runner.run(seed=seed)
            episode_results.append(metrics)

            if self.verbose and (seed + 1) % max(1, n_seeds // 5) == 0:
                elapsed = time.perf_counter() - t_start
                success_so_far = np.mean([m.task_success for m in episode_results])
                print(
                    f"  [{condition.name}] seed {seed+1:>3}/{n_seeds} | "
                    f"success_rate={success_so_far:.2f} | "
                    f"elapsed={elapsed:.1f}s"
                )

        summary = aggregate_episodes(
            condition_name=condition.name,
            episode_metrics=episode_results,
            episode_length=T,
        )

        if self.verbose:
            _print_condition_summary(summary)

        return summary


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Top-level runner: iterates over a list of Conditions and collects all results.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self._condition_runner = ConditionRunner(verbose=verbose)

    def run(self, conditions: list[Condition]) -> dict[str, ConditionSummary]:
        """
        Run all conditions and return a name → ConditionSummary mapping.

        Parameters
        ----------
        conditions : list of Condition objects from conditions.py

        Returns
        -------
        dict mapping condition.name → ConditionSummary
        """
        results: dict[str, ConditionSummary] = {}
        t_total = time.perf_counter()

        print(f"\n{'='*60}")
        print(f"Running {len(conditions)} conditions")
        print(f"{'='*60}")

        for i, condition in enumerate(conditions):
            print(f"\n[{i+1}/{len(conditions)}] {condition.name}")
            summary = self._condition_runner.run(condition)
            results[condition.name] = summary

        elapsed = time.perf_counter() - t_total
        print(f"\n{'='*60}")
        print(f"Experiment complete — {len(conditions)} conditions, "
              f"{elapsed:.1f}s total")
        print(f"{'='*60}\n")

        return results

    def run_and_save(
        self,
        conditions: list[Condition],
        output_path: str,
    ) -> dict[str, ConditionSummary]:
        """
        Run all conditions and save full results to a pickle file.

        Parameters
        ----------
        output_path : path stem for .pkl file (e.g. 'results/experiment')
                     Will create 'results/experiment.pkl'
        """
        results = self.run(conditions)
        save_results(results, output_path)
        return results


# ---------------------------------------------------------------------------
# Results serialisation
# ---------------------------------------------------------------------------

def save_results(results: dict[str, ConditionSummary], path: str) -> None:
    """
    Save full ConditionSummary objects to a pickle file.
    
    Parameters
    ----------
    path : file path WITHOUT extension (e.g. 'results/experiment')
           Will create 'results/experiment.pkl'
    """
    import pickle
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    pkl_path = path if path.endswith(".pkl") else f"{path}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Results saved to {pkl_path}")


def load_results(path: str) -> dict[str, ConditionSummary]:
    """
    Load full ConditionSummary objects from a pickle file.
    
    Parameters
    ----------
    path : file path (with or without .pkl extension)
    
    Returns
    -------
    dict mapping condition_name → ConditionSummary
    """
    import pickle
    
    pkl_path = path if path.endswith(".pkl") else f"{path}.pkl"
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results)} conditions from {pkl_path}")
    return results


def _save_results_npz(results: dict[str, ConditionSummary], path: str) -> None:
    """
    Legacy: Save scalar summaries to a numpy .npz archive.
    Use save_results() instead for full object persistence.
    """
    arrays: dict[str, np.ndarray] = {}

    for name, summary in results.items():
        safe_name = name.replace("/", "_").replace(" ", "_")
        d = summary.to_dict()
        for metric, value in d.items():
            key = f"{safe_name}__{metric}"
            arrays[key] = np.array(value)
        # Time series
        arrays[f"{safe_name}__jsd_matrix"] = summary.jsd_matrix
        arrays[f"{safe_name}__alignment_matrix"] = summary.alignment_matrix

    np.savez(path, **arrays)
    print(f"Scalar summaries saved to {path}.npz")


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def _print_condition_summary(s: ConditionSummary) -> None:
    """Print a compact summary of one condition's results."""
    print(
        f"  → success_rate={s.task_success_rate:.3f} | "
        f"silent_fail={s.silent_failure_rate:.3f} | "
        f"jsd={s.mean_final_jsd:.4f} | "
        f"alignment={s.mean_final_alignment:.4f} | "
        f"msgs={s.mean_messages_sent:.1f} | "
        f"bytes={s.mean_bytes_transmitted:.1f} | "
        f"apb={s.mean_alignment_per_byte:.6f}"
    )


def print_results_table(results: dict[str, ConditionSummary]) -> None:
    """Print a formatted comparison table of all conditions."""
    header = (
        f"{'Condition':<25} {'Success':>8} {'SF|All':>8} {'SF|Fail':>8} {'CoordFail':>10} "
        f"{'JSD':>8} {'Align':>8} {'Msgs':>8} {'A/B':>12}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for name, s in results.items():
        apb_str = f"{s.mean_alignment_per_byte:.2e}" if not np.isnan(s.mean_alignment_per_byte) else "    N/A"
        sf_fail_str = f"{s.silent_failure_rate_given_failure:.3f}" if not np.isnan(s.silent_failure_rate_given_failure) else "  N/A"
        coord_fail_str = f"{s.coordinated_failure_rate:.3f}" if not np.isnan(s.coordinated_failure_rate) else "    N/A"
        
        print(
            f"{name:<25} {s.task_success_rate:>8.3f} {s.silent_failure_rate:>8.3f} "
            f"{sf_fail_str:>8} {coord_fail_str:>10} "
            f"{s.mean_final_jsd:>8.4f} {s.mean_final_alignment:>8.4f} "
            f"{s.mean_messages_sent:>8.1f} {apb_str:>12}"
        )

    print("=" * len(header))
    print("\nColumn Legend:")
    print("  Success    = Task success rate (any agent reached target)")
    print("  SF|All     = Silent failure rate across all episodes")
    print("  SF|Fail    = Silent failure rate among FAILED episodes only")
    print("  CoordFail  = Coordinated failure rate (low JSD among failures)")
    print("  JSD        = Mean final pairwise JSD")
    print("  Align      = Mean final alignment to truth")
    print("  Msgs       = Mean messages sent per episode")
    print("  A/B        = Alignment per byte (efficiency)")
    print()