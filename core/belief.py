"""
core/belief.py
--------------
Belief map operations: the shared mathematical core of the simulation.

A belief map b is a 1D NumPy array of shape (N*N,) representing a probability
distribution over grid cells — b[x] = P(target is at cell x).

This module provides:
  - BeliefMap class: maintains, updates, and queries a single agent's belief
  - Divergence metrics: Jensen-Shannon divergence, alignment to truth
  - Belief fusion: weighted average for message integration
"""

import numpy as np
from numpy.typing import NDArray

from config import WorldConfig, ObservationConfig
from core.observation import Observation


# ---------------------------------------------------------------------------
# Epsilon guard against log(0)
# ---------------------------------------------------------------------------
_EPS = 1e-12


# ---------------------------------------------------------------------------
# BeliefMap
# ---------------------------------------------------------------------------

class BeliefMap:
    """
    An agent's probabilistic belief about where the target is located.

    Internally stored as a normalised probability vector over all N*N cells.
    Bayesian updates are performed in log-space for numerical stability, then
    exponentiated and renormalised.
    """

    def __init__(self, n_cells: int, prior: float | None = None) -> None:
        """
        Parameters
        ----------
        n_cells : int
            Total number of grid cells (N*N).
        prior : float | None
            If None, initialise with a uniform distribution (with tiny random noise
            to break ties in argmax).
            If a float, use that value for all cells (then normalise).
        """
        self.n_cells = n_cells
        if prior is None:
            self._log_b = np.full(n_cells, -np.log(n_cells))  # log(1/N²)
            # Add tiny noise to break ties in argmax (prevents all agents from 
            # initially targeting cell 0)
            self._log_b += np.random.randn(n_cells) * 1e-10  # ← NEW LINE
        else:
            raw = np.full(n_cells, prior, dtype=float)
            self._log_b = np.log(raw / raw.sum())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def probs(self) -> NDArray[np.float64]:
        """Return the current belief as a normalised probability array."""
        # Numerically stable softmax-style: subtract max before exp
        log_b = self._log_b - self._log_b.max()
        p = np.exp(log_b)
        return p / p.sum()

    @property
    def argmax_cell(self) -> int:
        """The cell with highest belief probability (semantic estimate)."""
        return int(np.argmax(self._log_b))

    @property
    def entropy(self) -> float:
        """Shannon entropy H(b) in nats."""
        return _entropy(self.probs)

    # ------------------------------------------------------------------
    # Bayesian update
    # ------------------------------------------------------------------

    def update(
        self,
        obs: Observation,
        obs_config: ObservationConfig,
        obstacle_mask: NDArray[np.bool_],
    ) -> None:
        """
        Update the belief map given a noisy observation.

        Proper Bayesian update in log space:
        For each visible cell x:
          - If signal detected:   log_b[x] += log(P(detect | present) / P(detect | absent))
          - If no signal:         log_b[x] += log(P(no detect | present) / P(no detect | absent))

        This is the log-likelihood ratio update. Cells not in FOV get no update (likelihood ratio = 1).

        Obstacle cells are zeroed out (target cannot be there).

        The update is performed in log space for numerical stability.
        The belief is implicitly renormalised when .probs is accessed.
        """
        tpr, fpr, fnr, tnr = _likelihood_arrays(obs_config)

        # Detection update: for cells where signal was observed
        # log(P(detect|present) / P(detect|absent)) = log(TPR / FPR)
        if len(obs.detected_cells) > 0:
            likelihood_ratio = tpr / (fpr + _EPS)
            self._log_b[obs.detected_cells] += np.log(likelihood_ratio + _EPS)

        # No-detection update: for cells in FOV where no signal was observed
        # log(P(no detect|present) / P(no detect|absent)) = log(FNR / TNR)
        if len(obs.empty_cells) > 0:
            likelihood_ratio = fnr / (tnr + _EPS)
            self._log_b[obs.empty_cells] += np.log(likelihood_ratio + _EPS)

        # Cells NOT in FOV: likelihood ratio = 1 → no update (log(1) = 0)

        # Zero out obstacles (can never contain a target)
        self._log_b[obstacle_mask] = -np.inf

        # Numerical stabilisation: keep log values from drifting too large
        finite_mask = np.isfinite(self._log_b)
        if finite_mask.any():
            self._log_b[finite_mask] -= self._log_b[finite_mask].max()

    # ------------------------------------------------------------------
    # Top-k extraction (for C2 / C3 messages)
    # ------------------------------------------------------------------

    def top_k(self, k: int) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """
        Return the top-k cells by probability and their probabilities.

        Returns
        -------
        indices : NDArray[np.intp]  shape (k,)
        probs   : NDArray[np.float64] shape (k,)
        """
        p = self.probs
        k = min(k, self.n_cells)
        # argpartition is O(n) vs O(n log n) for full sort
        idx = np.argpartition(p, -k)[-k:]
        idx = idx[np.argsort(p[idx])[::-1]]  # sort descending within top-k
        return idx, p[idx]

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def set_probs(self, p: NDArray[np.float64]) -> None:
        """
        Directly set the belief from a probability array.
        Useful after fusion operations.
        """
        assert len(p) == self.n_cells
        p = np.clip(p, _EPS, None)
        p = p / p.sum()
        self._log_b = np.log(p)

    def clone(self) -> "BeliefMap":
        bm = BeliefMap(self.n_cells)
        bm._log_b = self._log_b.copy()
        return bm

    def __repr__(self) -> str:
        p = self.probs
        return (
            f"BeliefMap(argmax={self.argmax_cell}, "
            f"max_prob={p.max():.3f}, H={self.entropy:.3f} nats)"
        )


# ---------------------------------------------------------------------------
# Divergence and alignment metrics
# ---------------------------------------------------------------------------

def jensen_shannon_divergence(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """
    Symmetric Jensen-Shannon divergence between two probability vectors.

    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m),  m = 0.5*(p+q)

    Returns a value in [0, ln(2)] nats (≈ [0, 0.693]).
    """
    p = p / (p.sum() + _EPS)
    q = q / (q.sum() + _EPS)
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def mean_pairwise_jsd(belief_maps: list[BeliefMap]) -> float:
    """
    Mean pairwise Jensen-Shannon divergence across all agent pairs.

    This is the primary "inter-agent divergence" metric (RQ1, RQ2, RQ3).

    Returns 0.0 for a single agent.
    """
    n = len(belief_maps)
    if n < 2:
        return 0.0

    probs = [bm.probs for bm in belief_maps]
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += jensen_shannon_divergence(probs[i], probs[j])
            count += 1
    return total / count


def alignment_to_truth(
    belief: BeliefMap,
    true_target_cells: list[int],
    n_cells: int,
) -> float:
    """
    How much probability mass the agent places on the true target location(s).

    For single target: P(target at true cell).
    For multiple targets: max P(target at any true cell).

    Returns a value in [0, 1].  Higher = better aligned.
    """
    p = belief.probs
    if not true_target_cells:
        return 0.0
    return float(p[true_target_cells].max())


def silent_failure(
    belief_maps: list[BeliefMap],
    true_target_cells: list[int],
    alignment_threshold: float = 0.1,
    divergence_threshold: float = 0.2,
) -> bool:
    """
    Detect a "silent failure": agents appear to coordinate (similar beliefs)
    but all beliefs are misaligned with the true target.

    A silent failure occurs when:
      - Mean pairwise JSD < divergence_threshold  (agents agree with each other)
      - Mean alignment to truth < alignment_threshold  (but they are all wrong)

    This operationalises the hidden coordination failure the paper targets.
    """
    mpjsd = mean_pairwise_jsd(belief_maps)
    mean_alignment = np.mean(
        [alignment_to_truth(bm, true_target_cells, bm.n_cells) for bm in belief_maps]
    )
    return bool(mpjsd < divergence_threshold and mean_alignment < alignment_threshold)


# ---------------------------------------------------------------------------
# Belief fusion
# ---------------------------------------------------------------------------

def _confidence_weight(entropy: float, max_weight: float, max_entropy: float) -> float:
    """
    Map belief entropy to a fusion weight via inverse-entropy scaling.

    An agent with entropy 0 (perfectly certain) contributes weight = max_weight.
    An agent with entropy = max_entropy (maximally uncertain) contributes weight → 0.

    Weight = max_weight * (1 - H / H_max)

    Clamped to [0, max_weight].

    Parameters
    ----------
    entropy     : Shannon entropy of the sender's belief (nats).
    max_weight  : ceiling on the returned weight (from CommConfig.max_fusion_weight).
    max_entropy : ln(N²) — entropy of the uniform prior.  Computed once per episode.
    """
    if max_entropy <= _EPS:
        return max_weight
    normalised_uncertainty = min(entropy / max_entropy, 1.0)
    return float(np.clip(max_weight * (1.0 - normalised_uncertainty), 0.0, max_weight))


def fuse_beliefs(
    own_belief: BeliefMap,
    incoming_probs: NDArray[np.float64],
    sender_entropy: float,
    max_fusion_weight: float = 0.8,
    max_entropy: float | None = None,
) -> BeliefMap:
    """
    Fuse own belief with an incoming belief using inverse-entropy weighting.

    More confident senders (lower entropy) exert greater influence.

    fused[x] = (1 - w) * own[x]  +  w * incoming[x]

    where w = max_fusion_weight * (1 - H_sender / H_max)

    This means:
      - A sender with entropy → 0 (very confident) gets weight ≈ max_fusion_weight.
      - A sender with entropy → H_max (uniform, knows nothing) gets weight ≈ 0.
      - The receiver's own belief is always preserved with weight (1 - w) ≥ 0.

    Parameters
    ----------
    own_belief         : the receiving agent's current BeliefMap.
    incoming_probs     : probability array from the sender (must sum to ~1).
                         Reconstruct from message before calling:
                           - C1: use reconstruct_from_semantic()
                           - C2/C3: use reconstruct_from_topk()
    sender_entropy     : Shannon entropy of the *sender's* belief at send time (nats).
                         Transmitted as part of C2/C3 messages; inferred for C1.
    max_fusion_weight  : ceiling on sender influence (CommConfig.max_fusion_weight).
    max_entropy        : ln(n_cells) — uniform prior entropy.  Auto-computed if None.

    Returns a new BeliefMap (does not mutate own_belief).
    """
    n_cells = own_belief.n_cells
    if max_entropy is None:
        max_entropy = float(np.log(n_cells))

    w = _confidence_weight(sender_entropy, max_fusion_weight, max_entropy)

    p_own = own_belief.probs
    p_in = incoming_probs / (incoming_probs.sum() + _EPS)

    p_fused = (1.0 - w) * p_own + w * p_in
    result = own_belief.clone()
    result.set_probs(p_fused)
    return result


def fuse_beliefs_multi(
    own_belief: BeliefMap,
    incoming: list[tuple[NDArray[np.float64], float]],
    max_fusion_weight: float = 0.8,
    max_entropy: float | None = None,
) -> BeliefMap:
    """
    Fuse own belief with multiple incoming messages simultaneously.

    Applies normalised inverse-entropy weights across all senders so that
    total incoming weight ≤ max_fusion_weight regardless of how many
    messages arrive in one step.

    Parameters
    ----------
    own_belief  : receiving agent's current BeliefMap.
    incoming    : list of (prob_array, sender_entropy) pairs.
    max_fusion_weight : ceiling on total incoming influence combined.
    max_entropy : ln(n_cells) if None.

    Returns a new BeliefMap (does not mutate own_belief).
    """
    if not incoming:
        return own_belief.clone()

    n_cells = own_belief.n_cells
    if max_entropy is None:
        max_entropy = float(np.log(n_cells))

    # Compute raw confidence weight for each sender
    raw_weights = np.array([
        _confidence_weight(h, max_fusion_weight, max_entropy)
        for _, h in incoming
    ])

    # Normalise so total incoming weight ≤ max_fusion_weight
    total_raw = raw_weights.sum()
    if total_raw > _EPS:
        # Scale so they sum to at most max_fusion_weight
        normalised = raw_weights / total_raw * min(total_raw, max_fusion_weight)
    else:
        # All senders are maximally uncertain — ignore incoming messages
        return own_belief.clone()

    w_own = 1.0 - normalised.sum()
    p_fused = w_own * own_belief.probs

    for (p_in_raw, _), w in zip(incoming, normalised):
        p_in = p_in_raw / (p_in_raw.sum() + _EPS)
        p_fused += w * p_in

    result = own_belief.clone()
    result.set_probs(p_fused)
    return result


def reconstruct_from_topk(
    indices: NDArray[np.intp],
    values: NDArray[np.float64],
    n_cells: int,
    background: float | None = None,
) -> NDArray[np.float64]:
    """
    Reconstruct a full probability vector from a sparse top-k message.

    The remaining mass (1 - sum(values)) is spread uniformly over
    the non-top-k cells (or set to `background` if provided).

    Parameters
    ----------
    indices : top-k cell indices
    values  : corresponding probabilities
    n_cells : total number of cells
    background : if None, remaining mass is spread uniformly

    Returns a normalised probability array of shape (n_cells,).
    """
    p = np.zeros(n_cells, dtype=float)
    top_k_mass = values.sum()

    if background is None:
        remaining = max(0.0, 1.0 - top_k_mass)
        n_remaining = n_cells - len(indices)
        bg = remaining / max(n_remaining, 1)
    else:
        bg = background

    p[:] = bg
    p[indices] = values

    # Renormalise to ensure valid distribution
    total = p.sum()
    if total > _EPS:
        p /= total
    else:
        p[:] = 1.0 / n_cells

    return p


def reconstruct_from_semantic(
    argmax_cell: int,
    n_cells: int,
    confidence: float = 0.9,
) -> NDArray[np.float64]:
    """
    Reconstruct a full probability vector from a C1 semantic message
    (a single argmax assertion).

    Places `confidence` mass on the asserted cell, spreads the rest uniformly.
    """
    p = np.full(n_cells, (1.0 - confidence) / max(n_cells - 1, 1))
    p[argmax_cell] = confidence
    return p


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _entropy(p: NDArray[np.float64]) -> float:
    """Shannon entropy in nats, safe against log(0)."""
    p = p[p > _EPS]
    return float(-np.sum(p * np.log(p)))


def _kl(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """KL divergence KL(p || q) in nats, safe against 0/0."""
    mask = p > _EPS
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + _EPS))))


def _likelihood_arrays(
    obs_config: ObservationConfig,
) -> tuple[float, float, float, float]:
    """TPR, FPR, FNR, TNR from observation config."""
    fnr = obs_config.false_negative_rate
    fpr = obs_config.false_positive_rate
    return (1.0 - fnr, fpr, fnr, 1.0 - fpr)