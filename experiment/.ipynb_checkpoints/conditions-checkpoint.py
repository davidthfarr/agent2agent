"""
experiment/conditions.py
------------------------
Defines the full experimental condition matrix.

Each Condition is a named (SimConfig, CommType) pair.  The runner iterates
over conditions × seeds to produce the results matrix.

Condition axes (from the spec):
  - Communication type:   C0, C1, C2, C3
  - Packet loss rate:     0%, 10%, 30%
  - Latency:              0, 1, 3 steps

The full factorial crossing is 4 × 3 × 3 = 36 conditions.
Helper functions also produce subsets for targeted RQ analysis.

Usage
-----
    from experiment.conditions import all_conditions, rq1_conditions

    for condition in all_conditions():
        results = runner.run_condition(condition)
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from itertools import product

from config import SimConfig, CommConfig, DEFAULT_CONFIG
from comms.message import CommType


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    """
    A single experimental condition: a fully-specified SimConfig
    paired with the communication type to use.

    The name is used as a key in results dictionaries and plot labels.
    """
    name: str
    config: SimConfig
    comm_type: CommType

    def __repr__(self) -> str:
        return f"Condition({self.name!r})"


# ---------------------------------------------------------------------------
# Condition construction helpers
# ---------------------------------------------------------------------------

def make_condition(
    comm_type: CommType,
    base_packet_loss_rate: float = 0.0,
    latency_steps: int = 0,
    base_config: SimConfig | None = None,
    name: str | None = None,
) -> Condition:
    """
    Build a single Condition by overriding CommConfig fields on a base SimConfig.
    """
    cfg = base_config or DEFAULT_CONFIG

    # Replace CommConfig fields (dataclass replace is immutable-safe)
    new_comm = replace(
        cfg.comms,
        base_packet_loss_rate=base_packet_loss_rate,
        latency_steps=latency_steps,
    )
    new_cfg = replace(cfg, comms=new_comm)

    if name is None:
        loss_str = f"loss{int(base_packet_loss_rate*100):02d}"
        lat_str = f"lat{latency_steps}"
        ct_str = f"C{int(comm_type)}"
        name = f"{ct_str}_{loss_str}_{lat_str}"

    return Condition(name=name, config=new_cfg, comm_type=comm_type)


# ---------------------------------------------------------------------------
# Full factorial condition matrix
# ---------------------------------------------------------------------------

# Parameter grid (from spec)
_COMM_TYPES = [
    CommType.C0_NONE,
    CommType.C1_SEMANTIC,
    CommType.C2_EPISTEMIC,
    CommType.C3_GATED,
]

_PACKET_LOSS_RATES = [0.0, 0.10, 0.30, .40, .50]
_LATENCY_STEPS     = [0, 1, 3, 5]


def all_conditions(base_config: SimConfig | None = None) -> list[Condition]:
    """
    Full 4 × 3 × 3 = 36-condition factorial design.
    """
    conditions = []
    for ct, loss, lat in product(_COMM_TYPES, _PACKET_LOSS_RATES, _LATENCY_STEPS):
        conditions.append(make_condition(
            comm_type=ct,
            base_packet_loss_rate=loss,
            latency_steps=lat,
            base_config=base_config,
        ))
    return conditions


# ---------------------------------------------------------------------------
# Per-RQ condition subsets
# ---------------------------------------------------------------------------

def rq1_conditions(base_config: SimConfig | None = None) -> list[Condition]:
    """
    RQ1: Can agents complete tasks while holding divergent beliefs?
    Compares C0 (no comms) vs C1/C2/C3 with no packet loss, no latency.
    Focuses on epistemic divergence and task success.
    """
    return [
        make_condition(ct, base_packet_loss_rate=0.0, latency_steps=0,
                       base_config=base_config)
        for ct in _COMM_TYPES
    ]


def rq2_conditions(base_config: SimConfig | None = None) -> list[Condition]:
    """
    RQ2: Does epistemic (C2) communication preserve alignment better than
    semantic (C1) at equal bandwidth?

    Pairs C1 and C2 across all loss/latency levels.  The bandwidth
    constraint is enforced by setting C1's top_k=1 (one cell asserted)
    vs C2's top_k=1 (one cell + probability + entropy).  Since per-spec
    we compare at equal *packet* bandwidth (same message count), this
    isolates the information content difference.
    """
    cfg = base_config or DEFAULT_CONFIG

    conditions = []
    for loss, lat in product(_PACKET_LOSS_RATES, _LATENCY_STEPS):
        for ct in [CommType.C1_SEMANTIC, CommType.C2_EPISTEMIC]:
            conditions.append(make_condition(
                ct, base_packet_loss_rate=loss,
                latency_steps=lat, base_config=cfg,
            ))
    return conditions


def rq3_conditions(base_config: SimConfig | None = None) -> list[Condition]:
    """
    RQ3: Can confidence-gated comms (C3) reduce message volume without
    increasing epistemic drift?

    Compares C2 (always send) vs C3 (gate) across loss/latency.
    """
    conditions = []
    for loss, lat in product(_PACKET_LOSS_RATES, _LATENCY_STEPS):
        for ct in [CommType.C2_EPISTEMIC, CommType.C3_GATED]:
            conditions.append(make_condition(
                ct, base_packet_loss_rate=loss,
                latency_steps=lat, base_config=base_config,
            ))
    return conditions


def baseline_conditions(base_config: SimConfig | None = None) -> list[Condition]:
    """
    Minimal set for quick sanity checks: one of each comm type,
    no packet loss, no latency.
    """
    return rq1_conditions(base_config)


# ---------------------------------------------------------------------------
# Custom condition builder (for targeted parameter sweeps)
# ---------------------------------------------------------------------------

def sweep_loss_rate(
    comm_type: CommType,
    loss_rates: list[float],
    latency_steps: int = 0,
    base_config: SimConfig | None = None,
) -> list[Condition]:
    """Sweep packet loss rates for a fixed comm type and latency."""
    return [
        make_condition(comm_type, r, latency_steps, base_config)
        for r in loss_rates
    ]


def sweep_latency(
    comm_type: CommType,
    latency_values: list[int],
    base_packet_loss_rate: float = 0.0,
    base_config: SimConfig | None = None,
) -> list[Condition]:
    """Sweep latency values for a fixed comm type and loss rate."""
    return [
        make_condition(comm_type, base_packet_loss_rate, lat, base_config)
        for lat in latency_values
    ]
