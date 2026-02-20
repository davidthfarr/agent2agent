# Epistemic Alignment in Decentralized Agent Systems

Research code for evaluating how communication constraints influence shared situational understanding in multi-agent systems.

## Quick Start

### 1. Run an experiment

```bash
# Small baseline test (4 conditions × 10 seeds, ~30 seconds)
python main.py --mode baseline --seeds 10 --output results/baseline_test

# Full RQ1 analysis (4 comm types, no noise, 50 seeds)
python main.py --mode rq1 --seeds 50 --output results/rq1

# Full factorial experiment (36 conditions × 50 seeds, ~10 minutes)
python main.py --mode full --seeds 50 --output results/full_experiment

# Single condition debug
python main.py --mode single --comm C2 --loss 0.1 --latency 1 --seeds 5
```

**Output**: Creates `results/baseline_test.pkl` containing full `ConditionSummary` objects.

### 2. Analyze results

```bash
# Generate all 7 figures
python analyze.py results/baseline_test.pkl --figures results/baseline_figures

# View specific figures interactively
python analyze.py results/baseline_test.pkl --only jsd_timeseries,alignment_timeseries --show

# Print summary table
python analyze.py results/baseline_test.pkl --table
```

**Output**: Creates PNG files in `results/baseline_figures/`:
- `fig1_jsd_timeseries.png`
- `fig2_alignment_timeseries.png`
- `fig3_jsd_vs_alignment.png`
- `fig4_silent_failure_heatmap.png`
- `fig5_bandwidth_efficiency.png`
- `fig6_message_volume.png`
- `fig7_task_outcomes.png`

---

## Project Structure

```
epistemic_alignment/
├── main.py              # Experiment entry point
├── analyze.py           # Analysis entry point
├── config.py            # All simulation parameters
│
├── core/                # Simulation core
│   ├── world.py         # Grid world, obstacles, LOS
│   ├── agent.py         # Agent belief maps, movement
│   ├── observation.py   # Noisy sensor model
│   └── belief.py        # Bayesian updates, JSD, fusion
│
├── comms/               # Communication layer
│   ├── message.py       # C0-C3 message types
│   ├── network.py       # Packet loss, latency
│   └── fusion.py        # Belief integration
│
├── experiment/          # Experiment orchestration
│   ├── conditions.py    # 36-condition factorial
│   ├── metrics.py       # All metrics collection
│   └── runner.py        # Episode/condition runners
│
└── analysis/            # Visualization
    └── plots.py         # All 7 publication figures
```

---

## Research Questions

**RQ1**: Can agents complete tasks while holding divergent beliefs?  
→ Compare C0 (no comms) vs C1/C2/C3. Measure JSD, alignment, silent failure rate.

**RQ2**: Does epistemic communication (C2) preserve alignment better than semantic (C1)?  
→ Compare C1 vs C2 across loss/latency. Key metric: alignment per byte.

**RQ3**: Can confidence-gated comms (C3) reduce bandwidth without increasing drift?  
→ Compare C2 vs C3. Key metrics: messages sent, final JSD.

---

## Communication Types

| Type | Name | Payload | Size | Gate |
|------|------|---------|------|------|
| C0 | None | — | 0 bytes | — |
| C1 | Semantic | argmax cell | 7 bytes | always |
| C2 | Epistemic | top-k cells + probs + entropy | 35 bytes | always |
| C3 | Gated | same as C2 | 35 bytes | only if ΔH ≥ threshold |

---

## Configuration

Edit `config.py` to change:

```python
# World
grid_size = 20           # N×N grid
num_targets = 1          # 1 or 2
num_obstacles = 15       # static occlusion

# Observation
fov_radius = 3           # Chebyshev distance
false_negative_rate = 0.1
false_positive_rate = 0.05

# Communication
base_packet_loss_rate = 0.0
latency_steps = 0
top_k = 5                # C2/C3 message sparsity
entropy_delta_threshold = 0.3  # C3 gate (nats)
max_fusion_weight = 0.8  # inverse-entropy fusion cap

# Episode
episode_length = 200     # T steps
num_seeds = 50           # per condition
```

---

## Workflow Examples

### Example 1: Reproduce paper results

```bash
# Run full 36-condition factorial
python main.py --mode full --seeds 50 --grid 20 --episodes 200 \
    --output results/paper_replication

# Generate all figures
python analyze.py results/paper_replication.pkl \
    --figures results/paper_figures \
    --table

# Figures are in results/paper_figures/*.png
```

### Example 2: Sweep a single parameter

```python
# In Python REPL or script:
from config import SimConfig, WorldConfig, CommConfig
from experiment.conditions import sweep_loss_rate
from experiment.runner import ExperimentRunner, save_results
from comms.message import CommType

# Configure base
base = SimConfig(
    world=WorldConfig(grid_size=15),
    comms=CommConfig(entropy_delta_threshold=0.5),  # custom threshold
)

# Sweep packet loss for C3 only
conditions = sweep_loss_rate(
    CommType.C3_GATED, 
    loss_rates=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    base_config=base
)

# Run
runner = ExperimentRunner(verbose=True)
results = runner.run(conditions)
save_results(results, "results/c3_loss_sweep")
```

### Example 3: Single-condition deep dive

```bash
# Run one condition with many seeds for high statistical power
python main.py --mode single --comm C2 --loss 0.1 --latency 0 \
    --seeds 200 --output results/c2_loss10_deep

# Analyze
python analyze.py results/c2_loss10_deep.pkl --show
```

---

## Metrics Reference

**Per-step metrics** (recorded every step):
- `mean_pairwise_jsd`: Inter-agent epistemic divergence
- `mean_alignment_to_truth`: Average P(target | belief)

**Per-episode metrics** (computed at end):
- `task_success`: Did any agent reach target?
- `time_to_success`: Step of first success (NaN if failed)
- `silent_failure`: Agents agree but are wrong
- `messages_sent`: Total recipient-messages
- `bytes_transmitted`: Total payload bytes
- `alignment_per_byte`: Efficiency metric

**Aggregated metrics** (across seeds):
- All of the above as `mean_*` and `std_*`
- `jsd_matrix`: (n_seeds × T) for plotting
- `alignment_matrix`: (n_seeds × T) for plotting

---

## Advanced Usage

### Custom condition sets

```python
from experiment.conditions import make_condition, Condition
from comms.message import CommType

# Build arbitrary conditions
my_conditions = [
    make_condition(CommType.C2_EPISTEMIC, base_packet_loss_rate=0.05, 
                   latency_steps=2, name="C2_custom"),
    make_condition(CommType.C3_GATED, base_packet_loss_rate=0.05, 
                   latency_steps=2, name="C3_custom"),
]

# Run
from experiment.runner import ExperimentRunner
runner = ExperimentRunner()
results = runner.run(my_conditions)
```

### Programmatic analysis

```python
from experiment.runner import load_results
from analysis.plots import fig_jsd_timeseries

# Load
results = load_results("results/my_experiment.pkl")

# Generate specific figure with custom filtering
fig = fig_jsd_timeseries(
    results,
    output_path="custom_figures",
    loss_filter=0.1,        # only 10% loss conditions
    latency_filter=0,       # only zero latency
    title_suffix=" (filtered)"
)

# Access raw data
for name, summary in results.items():
    print(f"{name}: success_rate = {summary.task_success_rate:.2f}")
    # summary.jsd_matrix is (n_seeds, T) numpy array
    # summary.alignment_matrix is (n_seeds, T) numpy array
```

---

## Testing

```bash
# All unit tests (~2 seconds)
python tests/test_core.py

# Quick integration check
python main.py --mode baseline --seeds 2 --episodes 20
```

---

## Dependencies

- Python 3.12+
- NumPy 2.4+
- Matplotlib 3.10+
- SciPy 1.17+

All are available in the base environment.

---

## Citation

If you use this code, please cite:

> []

---

## License

[]