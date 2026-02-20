"""
comms/fusion.py
---------------
Wires received messages to belief updates on each agent.

This module bridges the network delivery layer and the belief math layer.
It is responsible for:
  1. Reconstructing a full probability distribution from each message type.
  2. Collecting all messages received by an agent in one step.
  3. Calling fuse_beliefs_multi() with inverse-entropy weights.

Separation of concerns
-----------------------
- comms/message.py   → message construction (sender side)
- comms/network.py   → delivery, packet loss, latency (channel)
- comms/fusion.py    → belief update on receipt (receiver side)
- core/belief.py     → the actual math (fuse_beliefs_multi, reconstruct_*)
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from comms.message import Message, SemanticMessage, EpistemicMessage, CommType
from core.belief import (
    BeliefMap,
    fuse_beliefs,
    fuse_beliefs_multi,
    reconstruct_from_topk,
    reconstruct_from_semantic,
)
from config import CommConfig


# ---------------------------------------------------------------------------
# Single-message reconstruction
# ---------------------------------------------------------------------------

def reconstruct_from_message(
    message: Message,
    n_cells: int,
) -> tuple[NDArray[np.float64], float]:
    """
    Convert any message type into a (probability_array, sender_entropy) pair
    ready for fusion.

    Returns
    -------
    probs          : NDArray[np.float64] shape (n_cells,), normalised
    sender_entropy : float — used to compute inverse-entropy fusion weight
    """
    if isinstance(message, SemanticMessage):
        # C1: reconstruct a peaked distribution from the argmax assertion.
        # Confidence is set to (1 - FPR-like spread); using 0.9 as a
        # reasonable default since C1 doesn't transmit uncertainty.
        probs = reconstruct_from_semantic(
            argmax_cell=message.argmax_cell,
            n_cells=n_cells,
            confidence=0.9,
        )
        return probs, message.sender_entropy

    elif isinstance(message, EpistemicMessage):
        # C2 / C3: reconstruct from sparse top-k representation
        probs = reconstruct_from_topk(
            indices=message.top_k_indices,
            values=message.top_k_probs,
            n_cells=n_cells,
        )
        return probs, message.sender_entropy

    else:
        raise TypeError(f"Unknown message type: {type(message)}")


# ---------------------------------------------------------------------------
# Apply all incoming messages to an agent's belief
# ---------------------------------------------------------------------------

def apply_messages(
    agent,                          # Agent — avoid circular import
    messages: list[Message],
    config: CommConfig,
    max_entropy: float | None = None,
) -> None:
    """
    Fuse all messages received by an agent this step into its belief.

    Uses fuse_beliefs_multi() so that:
      - Multiple simultaneous messages are handled in one operation.
      - The total incoming weight is bounded by max_fusion_weight.
      - Each sender's contribution is scaled by its inverse entropy.

    Mutates agent.belief in place.

    Parameters
    ----------
    agent    : receiving Agent.
    messages : list of Message objects delivered to this agent this step.
    config   : CommConfig (provides max_fusion_weight).
    max_entropy : ln(n_cells) for the current world; auto-computed if None.
    """
    if not messages:
        return

    n_cells = agent.belief.n_cells
    if max_entropy is None:
        max_entropy = float(np.log(n_cells))

    # Reconstruct distributions + extract entropy for each incoming message
    incoming: list[tuple[NDArray[np.float64], float]] = []
    for msg in messages:
        probs, sender_entropy = reconstruct_from_message(msg, n_cells)
        incoming.append((probs, sender_entropy))

    agent.belief = fuse_beliefs_multi(
        own_belief=agent.belief,
        incoming=incoming,
        max_fusion_weight=config.max_fusion_weight,
        max_entropy=max_entropy,
    )


# ---------------------------------------------------------------------------
# Step-level fusion: apply all delivered messages across all agents
# ---------------------------------------------------------------------------

def apply_all_deliveries(
    agents: list,                   # list[Agent]
    inbox: dict[int, list[Message]],
    config: CommConfig,
    max_entropy: float | None = None,
) -> None:
    """
    Apply one round of message deliveries to all agents.

    Parameters
    ----------
    agents   : all agents in the simulation.
    inbox    : {agent_id: [messages]} from Network.deliver().
    config   : CommConfig.
    max_entropy : ln(n_cells); auto-computed per agent if None.
    """
    for agent in agents:
        messages = inbox.get(agent.agent_id, [])
        if messages:
            apply_messages(agent, messages, config, max_entropy)
