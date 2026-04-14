"""
main.py
-------
Command-line entry point for running experiments.

Usage examples
--------------
# Quick sanity check: 4 comm types, no noise, 5 seeds
python main.py --mode baseline --seeds 5

# Full RQ1 analysis
python main.py --mode rq1 --seeds 50

# Full factorial experiment (36 conditions × 50 seeds)
python main.py --mode full --seeds 50 --output results/full_experiment

# Single condition debug run
python main.py --mode single --comm C2 --loss 0.1 --latency 1 --seeds 3
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import SimConfig, WorldConfig, ObservationConfig, AgentConfig, CommConfig, ExperimentConfig
from comms.message import CommType
from experiment.conditions import (
    all_conditions, baseline_conditions,
    rq1_conditions, rq2_conditions, rq3_conditions,
    make_condition,
)
from experiment.runner import ExperimentRunner, print_results_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Epistemic Alignment in Decentralised Agent Systems"
    )
    parser.add_argument(
        "--mode", choices=["baseline", "rq1", "rq2", "rq3", "full", "single"],
        default="baseline",
        help="Which condition set to run (default: baseline)"
    )
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of random seeds per condition (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path stem for .npz output (optional)")
    parser.add_argument("--grid", type=int, default=20,
                        help="Grid size N (default: 20)")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Episode length T (default: 200)")
    parser.add_argument("--min-agents", type=int, default=2,
                        help="Min agents required at target for success (default: 1)")
    # Single condition args
    parser.add_argument("--comm", choices=["C0","C1","C2","C3"], default="C2")
    parser.add_argument("--loss", type=float, default=0.0)
    parser.add_argument("--latency", type=int, default=0)
    parser.add_argument("--entropy-delta", type=float, default=0.20,
                    help="C3 entropy delta gate threshold in nats (default: 0.15)")
    return parser.parse_args()


def build_base_config(args) -> SimConfig:
    return SimConfig(
        world=WorldConfig(grid_size=args.grid),
        obs=ObservationConfig(),
        agents=AgentConfig(num_agents=4),
        comms=CommConfig(entropy_delta_threshold=args.entropy_delta),
        experiment=ExperimentConfig(
            episode_length=args.episodes,
            num_seeds=args.seeds,
            min_agents_for_success=args.min_agents,
        ),
    )


def main():
    args = parse_args()
    base_cfg = build_base_config(args)

    # Select condition set
    if args.mode == "baseline":
        conditions = baseline_conditions(base_cfg)
    elif args.mode == "rq1":
        conditions = rq1_conditions(base_cfg)
    elif args.mode == "rq2":
        conditions = rq2_conditions(base_cfg)
    elif args.mode == "rq3":
        conditions = rq3_conditions(base_cfg)
    elif args.mode == "full":
        conditions = all_conditions(base_cfg)
    elif args.mode == "single":
        ct_map = {"C0": CommType.C0_NONE, "C1": CommType.C1_SEMANTIC,
                  "C2": CommType.C2_EPISTEMIC, "C3": CommType.C3_GATED}
        conditions = [make_condition(
            ct_map[args.comm], args.loss, args.latency, base_cfg
        )]

    runner = ExperimentRunner(verbose=True)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        results = runner.run_and_save(conditions, args.output)
    else:
        results = runner.run(conditions)

    print_results_table(results)


if __name__ == "__main__":
    main()
