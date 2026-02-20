"""
core/world.py
-------------
2D grid world with static obstacles and stationary targets.

Key design choices:
  - Cells are indexed as (row, col) tuples; row 0 is the top row.
  - The grid is stored as a flat NumPy array of shape (N*N,) wherever
    possible to keep belief-map operations vectorized.
  - "Cell index" refers to the scalar index row*N + col.
  - Obstacles fully occlude visibility (ray-cast occlusion check).
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Sequence

from config import WorldConfig, AgentConfig


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
CellIndex = int                          # flat index into N*N grid
Coord = tuple[int, int]                  # (row, col)


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------
@dataclass
class World:
    config: WorldConfig
    rng: np.random.Generator
    num_agents: int = 4          # passed in from AgentConfig at construction

    # Populated by reset()
    obstacle_mask: NDArray[np.bool_] = field(init=False)   # shape (N*N,)
    target_cells: list[CellIndex] = field(init=False)
    agent_starts: list[CellIndex] = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Place obstacles, targets, and agent start positions."""
        N = self.config.grid_size
        n_cells = N * N

        # --- obstacles ---
        if self.config.obstacle_seed is not None:
            obs_rng = np.random.default_rng(self.config.obstacle_seed)
        else:
            obs_rng = self.rng

        passable = list(range(n_cells))
        # Reserve corners for agents, never block them
        reserved = {0, N - 1, N * (N - 1), n_cells - 1}
        candidates = [c for c in passable if c not in reserved]
        obs_count = min(self.config.num_obstacles, len(candidates))
        obs_indices = obs_rng.choice(candidates, size=obs_count, replace=False)
        self.obstacle_mask = np.zeros(n_cells, dtype=bool)
        self.obstacle_mask[obs_indices] = True

        # --- targets ---
        free = self._free_cells()
        target_count = min(self.config.num_targets, len(free))
        target_indices = self.rng.choice(free, size=target_count, replace=False)
        self.target_cells = target_indices.tolist()

        # --- agent starts (corners, guaranteed obstacle-free) ---
        corner_indices = [0, N - 1, N * (N - 1), n_cells - 1]
        # Clear any obstacles that landed on corners (should not happen, but safety)
        self.obstacle_mask[corner_indices] = False
        self.agent_starts = corner_indices[: self.num_agents]
        # Re-seed targets away from agent starts
        occupied = set(self.agent_starts)
        self.target_cells = [t for t in self.target_cells if t not in occupied]
        if not self.target_cells:
            free2 = [c for c in self._free_cells() if c not in occupied]
            self.target_cells = [int(self.rng.choice(free2))]

    @property
    def N(self) -> int:
        return self.config.grid_size

    def coord(self, idx: CellIndex) -> Coord:
        """Convert flat cell index to (row, col)."""
        return divmod(idx, self.N)

    def index(self, row: int, col: int) -> CellIndex:
        """Convert (row, col) to flat cell index."""
        return row * self.N + col

    def is_obstacle(self, idx: CellIndex) -> bool:
        return bool(self.obstacle_mask[idx])

    def is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self.N and 0 <= col < self.N

    def neighbors(self, idx: CellIndex) -> list[CellIndex]:
        """
        4-connected passable neighbors (no diagonals).
        Used by agents for movement planning.
        """
        r, c = self.coord(idx)
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [
            self.index(nr, nc)
            for nr, nc in candidates
            if self.is_valid(nr, nc) and not self.obstacle_mask[self.index(nr, nc)]
        ]

    def cells_in_fov(self, agent_idx: CellIndex, fov_radius: int) -> NDArray[np.intp]:
        """
        Return flat indices of all cells within Chebyshev distance `fov_radius`
        of `agent_idx` that are not occluded by obstacles.

        Occlusion: a cell is visible only if the straight-line path from the
        agent to the cell centre does not pass through an obstacle cell.
        We use a simple discrete ray-march (Bresenham-style).
        """
        N = self.N
        ar, ac = self.coord(agent_idx)

        # Bounding box of FOV (clipped to grid)
        r_lo = max(0, ar - fov_radius)
        r_hi = min(N - 1, ar + fov_radius)
        c_lo = max(0, ac - fov_radius)
        c_hi = min(N - 1, ac + fov_radius)

        rows = np.arange(r_lo, r_hi + 1)
        cols = np.arange(c_lo, c_hi + 1)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")  # (rows, cols)
        flat = (rr * N + cc).ravel()

        # Filter to Chebyshev radius
        dr = np.abs(rr.ravel() - ar)
        dc = np.abs(cc.ravel() - ac)
        in_radius = (dr <= fov_radius) & (dc <= fov_radius)
        flat = flat[in_radius]

        # Occlusion filter
        visible = np.array(
            [self._has_los(ar, ac, *self.coord(int(cell))) for cell in flat],
            dtype=bool,
        )
        return flat[visible]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _free_cells(self) -> NDArray[np.intp]:
        """Flat indices of all non-obstacle cells."""
        return np.where(~self.obstacle_mask)[0]

    def _has_los(self, r0: int, c0: int, r1: int, c1: int) -> bool:
        """
        Bresenham line-of-sight check.
        Returns True if no obstacle cell lies strictly between (r0,c0)
        and (r1,c1).  The start cell (agent) is never counted as blocking.
        """
        if r0 == r1 and c0 == c1:
            return True

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1

        r, c = r0, c0
        err = dr - dc

        while True:
            if (r, c) != (r0, c0) and (r, c) != (r1, c1):
                if self.obstacle_mask[self.index(r, c)]:
                    return False
            if r == r1 and c == c1:
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_world(config: WorldConfig, seed: int, num_agents: int = 4) -> World:
    rng = np.random.default_rng(seed)
    return World(config=config, rng=rng, num_agents=num_agents)
