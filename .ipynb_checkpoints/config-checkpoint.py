"""
config.py
---------
Central configuration for all simulation parameters.
Edit this file to change grid size, agent count, noise rates, etc.
"""

from dataclasses import dataclass, field


@dataclass
class WorldConfig:
    grid_size: int = 50         # N: world is N x N cells
    num_targets: int = 1         # 1 or 2 stationary targets per episode
    num_obstacles: int = 0     # number of static obstacle cells
    obstacle_seed: int | None = None  # if set, obstacles are fixed across episodes


@dataclass
class ObservationConfig:
    fov_radius: int = 2          # agent sees cells within Chebyshev distance r
    false_negative_rate: float = 0.0  # P(miss | target present in FOV)
    false_positive_rate: float = 0.0 # P(detect | target absent in FOV)


@dataclass
class AgentConfig:
    num_agents: int = 4
    prior: float | None = None   # None → uniform; float → used for all cells


@dataclass
class CommConfig:
    # Network
    base_packet_loss_rate: float = 0.0
    # Packet loss scales with channel congestion: each simultaneous message in
    # the channel adds pressure. P(drop) = 1 - (1 - base)^n_concurrent_messages
    # Set base_packet_loss_rate=0 and congestion_factor=0 for lossless channel.
    packet_loss_congestion_factor: float = 0.00
    latency_steps: int = 0          # number of time steps a message is delayed

    # C2 / C3 parameters
    top_k: int = 5                  # number of top-belief cells transmitted in C2/C3

    # C3: agent transmits only if |H(now) - H(at_last_send)| >= threshold (nats).
    # Gates on *change* in entropy since last transmission, not on absolute level.
    # Sends when belief has meaningfully shifted (concentrated or diffused).
    entropy_delta_threshold: float = 0.25  # nats; tune per experiment

    # Fusion: weight is derived from sender confidence (inverse entropy).
    # This caps the maximum weight any single message can contribute.
    max_fusion_weight: float = .8


@dataclass
class ExperimentConfig:
    episode_length: int = 150    # T: max steps per episode
    num_seeds: int = 10          # random seeds per experimental condition
    min_agents_for_success: int = 2  # number of agents that must reach target
    # 1 = any single agent (individual search)
    # 2+ = coordinated arrival (true collaboration required)


@dataclass
class SimConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    obs: ObservationConfig = field(default_factory=ObservationConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    comms: CommConfig = field(default_factory=CommConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# ---------------------------------------------------------------------------
# Default config — import and mutate for experiments
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = SimConfig()

