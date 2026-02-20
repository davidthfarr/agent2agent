"""
core/observation.py
-------------------
Noisy observation model for agent field-of-view.

An agent observes each cell in its FOV as either "signal" (target detected)
or "no signal".  Two independent error processes apply:

    False Negative (miss):   target IS present, agent does NOT detect it.
                             P(no signal | target present) = fnr
    False Positive (ghost):  target is ABSENT, agent DOES detect it.
                             P(signal | target absent)     = fpr

These parameters define the likelihood model used in Bayesian updates.
"""

import numpy as np
from numpy.typing import NDArray

from config import ObservationConfig
from core.world import World, CellIndex


# ---------------------------------------------------------------------------
# Observation result
# ---------------------------------------------------------------------------

class Observation:
    """
    The noisy observation an agent receives at a single time step.

    Attributes
    ----------
    visible_cells : NDArray[np.intp]
        Flat indices of all cells currently in the agent's FOV.
    detections : NDArray[np.bool_]
        Parallel boolean array; True means "signal detected" in that cell.
    """

    def __init__(
        self,
        visible_cells: NDArray[np.intp],
        detections: NDArray[np.bool_],
    ) -> None:
        assert len(visible_cells) == len(detections)
        self.visible_cells = visible_cells
        self.detections = detections

    @property
    def detected_cells(self) -> NDArray[np.intp]:
        """Cells where a target signal was observed (may be false positives)."""
        return self.visible_cells[self.detections]

    @property
    def empty_cells(self) -> NDArray[np.intp]:
        """Cells where no signal was observed (may be false negatives)."""
        return self.visible_cells[~self.detections]

    def __repr__(self) -> str:
        return (
            f"Observation(visible={len(self.visible_cells)}, "
            f"detections={self.detections.sum()})"
        )


# ---------------------------------------------------------------------------
# Sensor model
# ---------------------------------------------------------------------------

class NoisySensor:
    """
    Generates noisy observations and provides the likelihood functions
    needed for Bayesian belief updates.
    """

    def __init__(self, config: ObservationConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    # ------------------------------------------------------------------
    # Observation generation
    # ------------------------------------------------------------------

    def observe(
        self,
        agent_pos: CellIndex,
        world: World,
        true_target_cells: list[CellIndex],
    ) -> Observation:
        """
        Sample a noisy observation for an agent at `agent_pos`.

        Steps:
          1. Compute the visible cells (FOV + LOS).
          2. For each visible cell, determine the true target presence.
          3. Apply FN / FP noise to produce the detected signal.
        """
        visible = world.cells_in_fov(agent_pos, self.config.fov_radius)

        # Ground truth presence for each visible cell
        target_set = set(true_target_cells)
        true_presence = np.array(
            [cell in target_set for cell in visible], dtype=bool
        )

        # Noise process
        detections = self._apply_noise(true_presence)

        return Observation(visible_cells=visible, detections=detections)

    # ------------------------------------------------------------------
    # Likelihood functions (for Bayesian update in agent.py)
    # ------------------------------------------------------------------

    def likelihood_detection(self, target_present: bool) -> float:
        """
        P(signal detected | target_present).

        P(detect | present)  = 1 - FNR   (true positive rate)
        P(detect | absent)   = FPR       (false positive rate)
        """
        if target_present:
            return 1.0 - self.config.false_negative_rate
        else:
            return self.config.false_positive_rate

    def likelihood_no_detection(self, target_present: bool) -> float:
        """
        P(no signal | target_present).

        P(no detect | present) = FNR
        P(no detect | absent)  = 1 - FPR
        """
        if target_present:
            return self.config.false_negative_rate
        else:
            return 1.0 - self.config.false_positive_rate

    def likelihood_arrays(self) -> tuple[float, float, float, float]:
        """
        Return the four likelihood scalars as a named tuple for use in
        vectorized Bayesian updates:

            (ll_det_present, ll_det_absent, ll_no_det_present, ll_no_det_absent)

        i.e. (TPR, FPR, FNR, TNR)
        """
        fnr = self.config.false_negative_rate
        fpr = self.config.false_positive_rate
        return (1.0 - fnr, fpr, fnr, 1.0 - fpr)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_noise(self, true_presence: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """
        Vectorized noise application.

        For cells with target: flip to "no detection" with prob FNR.
        For cells without target: flip to "detection" with prob FPR.
        """
        n = len(true_presence)
        rand = self.rng.random(n)

        detections = np.zeros(n, dtype=bool)
        present = true_presence
        absent = ~true_presence

        # True positives: detect if rand > FNR
        detections[present] = rand[present] > self.config.false_negative_rate
        # False positives: detect if rand < FPR
        detections[absent] = rand[absent] < self.config.false_positive_rate

        return detections
