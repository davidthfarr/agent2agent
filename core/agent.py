"""
core/agent.py
-------------
Agent class: maintains position, belief map, and Bayesian update logic.

An agent:
  1. Occupies a single grid cell.
  2. At each step, receives a noisy Observation from NoisySensor.
  3. Updates its BeliefMap via Bayesian inference.
  4. Moves according to a simple greedy policy (toward argmax cell).
  5. Exposes its belief for communication (C0–C3 message construction
     lives in comms/message.py, not here).

Design note: Agents are stateful objects mutated in-place each step.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from config import AgentConfig, ObservationConfig, CommConfig
from core.world import World, CellIndex
from core.belief import BeliefMap
from core.observation import Observation, NoisySensor


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    A single decentralised agent in the grid world.

    Attributes
    ----------
    agent_id : int
        Unique identifier (0-indexed).
    pos : CellIndex
        Current cell (flat index).
    belief : BeliefMap
        Agent's probabilistic belief about target location.
    reached_target : bool
        True once the agent steps onto a true target cell.
    step_count : int
        Number of steps taken this episode.
    """

    def __init__(
        self,
        agent_id: int,
        start_pos: CellIndex,
        n_cells: int,
        prior: float | None,
    ) -> None:
        self.agent_id = agent_id
        self.pos = start_pos
        self.belief = BeliefMap(n_cells=n_cells, prior=prior)
        self.reached_target = False
        self.step_count = 0

        # C3 gating: track entropy at the time of last transmission.
        # Initialised to current entropy (uniform prior) so the first
        # delta is measured relative to the starting state.
        self._entropy_at_last_send: float = self.belief.entropy

    # ------------------------------------------------------------------
    # Core step logic
    # ------------------------------------------------------------------

    def observe_and_update(
        self,
        world: World,
        sensor: NoisySensor,
        obs_config: ObservationConfig,
    ) -> Observation:
        """
        Sample a noisy observation from the current position and update belief.

        Returns the Observation (callers may log it for metrics).
        """
        obs = sensor.observe(
            agent_pos=self.pos,
            world=world,
            true_target_cells=world.target_cells,
        )
        self.belief.update(obs, obs_config, world.obstacle_mask)
        return obs

    def move(self, world: World) -> None:
        """
        Greedy movement: take one step toward the cell with highest belief.

        Strategy:
          - Compute Manhattan distance from each 4-connected neighbour to
            the current argmax cell.
          - Move to the neighbour that minimises that distance.
          - If already at argmax, stay put (agent "searches" the cell).
          - Tie-break randomly.

        This is intentionally simple — the experiment measures belief quality,
        not planning sophistication.
        """
        target_cell = self.belief.argmax_cell
        if self.pos == target_cell:
            # Already at believed target — do not move
            self.step_count += 1
            return

        neighbours = world.neighbors(self.pos)
        if not neighbours:
            self.step_count += 1
            return

        N = world.N
        tr, tc = world.coord(target_cell)

        def manhattan_to_target(idx: CellIndex) -> int:
            r, c = world.coord(idx)
            return abs(r - tr) + abs(c - tc)

        distances = np.array([manhattan_to_target(n) for n in neighbours])
        min_dist = distances.min()
        best = [n for n, d in zip(neighbours, distances) if d == min_dist]

        # Tie-break: pick the one closest to current argmax (stable ordering)
        self.pos = best[0]
        self.step_count += 1

    def check_success(self, world: World) -> bool:
        """
        Returns True if the agent is currently on a true target cell.
        Sets self.reached_target = True on first success.
        """
        if self.pos in world.target_cells:
            self.reached_target = True
        return self.reached_target

    # ------------------------------------------------------------------
    # Belief inspection helpers (used by comms layer)
    # ------------------------------------------------------------------

    @property
    def belief_entropy(self) -> float:
        return self.belief.entropy

    @property
    def belief_argmax(self) -> CellIndex:
        return self.belief.argmax_cell

    def belief_top_k(self, k: int) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        return self.belief.top_k(k)

    def belief_probs(self) -> NDArray[np.float64]:
        return self.belief.probs

    # ------------------------------------------------------------------
    # C3 gating: entropy-delta trigger
    # ------------------------------------------------------------------

    def should_send_c3(self, entropy_delta_threshold: float) -> bool:
        """
        C3 gate: return True if the agent's belief has changed enough since
        its last transmission to justify sending a new message.

        Gate condition:  |H(now) - H(at_last_send)| >= threshold

        This fires when the belief meaningfully concentrates (agent learned
        something — delta < 0) OR diffuses (agent became more uncertain —
        delta > 0, e.g. after moving into a new region).

        The absolute value means *both* directions of change trigger sends,
        capturing information gain in either direction.
        """
        current_entropy = self.belief.entropy
        return abs(current_entropy - self._entropy_at_last_send) >= entropy_delta_threshold

    def record_send(self) -> None:
        """
        Call this after a C3 message is sent to update the entropy baseline.
        Also valid to call for C2 messages if tracking is desired.
        """
        self._entropy_at_last_send = self.belief.entropy

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, start_pos: CellIndex, n_cells: int, prior: float | None) -> None:
        """Reinitialise for a new episode."""
        self.pos = start_pos
        self.belief = BeliefMap(n_cells=n_cells, prior=prior)
        self.reached_target = False
        self.step_count = 0
        self._entropy_at_last_send = self.belief.entropy

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.agent_id}, pos={self.pos}, "
            f"belief={self.belief}, success={self.reached_target})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_agents(
    config: AgentConfig,
    world: World,
) -> list[Agent]:
    """
    Instantiate all agents at their start positions.
    """
    n_cells = world.N ** 2
    agents = []
    for i in range(config.num_agents):
        start = world.agent_starts[i]
        agents.append(
            Agent(
                agent_id=i,
                start_pos=start,
                n_cells=n_cells,
                prior=config.prior,
            )
        )
    return agents
