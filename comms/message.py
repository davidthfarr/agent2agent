"""
comms/message.py
----------------
Message type definitions and construction for all four communication regimes.

Communication types
-------------------
C0 — None:        No messages sent.  Baseline condition.

C1 — Semantic:    Transmit arg max of belief map only.
                  Payload: (argmax_cell: int)
                  Receiver reconstructs a peaked distribution.

C2 — Epistemic:   Transmit top-k cells with probabilities + sender entropy.
                  Payload: (top_k_indices, top_k_probs, sender_entropy)
                  Receiver reconstructs a sparse distribution and uses
                  sender_entropy to compute inverse-entropy fusion weight.

C3 — Confidence-  Same payload as C2, but the message is only created if
     Gated:       |H(now) - H(at_last_send)| >= entropy_delta_threshold.
                  Gate evaluation lives in Agent.should_send_c3(); message
                  construction here is identical to C2 once the gate fires.

Byte cost accounting
--------------------
Each message type has a .byte_size property so the experiment runner can
accumulate total bytes transmitted per episode for the bandwidth metric.

Sizes assume:
  - cell index: 2 bytes (uint16, supports grids up to 256×256)
  - probability: 4 bytes (float32)
  - entropy:     4 bytes (float32)
  - sender_id:   1 byte
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Comm type enum
# ---------------------------------------------------------------------------

class CommType(IntEnum):
    C0_NONE      = 0
    C1_SEMANTIC  = 1
    C2_EPISTEMIC = 2
    C3_GATED     = 3   # same payload as C2; gate evaluated before construction


# ---------------------------------------------------------------------------
# Message dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticMessage:
    """C1: assert a single cell as the most likely target location."""
    sender_id: int
    argmax_cell: int
    sender_entropy: float   # included so fusion can weight correctly

    @property
    def byte_size(self) -> int:
        # sender_id(1) + cell_index(2) + entropy(4)
        return 7

    @property
    def comm_type(self) -> CommType:
        return CommType.C1_SEMANTIC


@dataclass(frozen=True)
class EpistemicMessage:
    """
    C2 / C3: transmit top-k cells with probabilities and sender entropy.

    The comm_type field distinguishes C2 from C3 for logging purposes;
    the payload is identical.
    """
    sender_id: int
    top_k_indices: NDArray[np.intp]    # shape (k,)
    top_k_probs: NDArray[np.float64]   # shape (k,)
    sender_entropy: float
    _comm_type: CommType = field(default=CommType.C2_EPISTEMIC)

    # NDArray fields are not hashable; override __hash__ to use id
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def k(self) -> int:
        return len(self.top_k_indices)

    @property
    def byte_size(self) -> int:
        # sender_id(1) + entropy(4) + k*(cell_index(2) + prob(4))
        return 1 + 4 + self.k * 6

    @property
    def comm_type(self) -> CommType:
        return self._comm_type


# Union type for type hints
Message = SemanticMessage | EpistemicMessage


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def make_semantic_message(
    sender_id: int,
    argmax_cell: int,
    sender_entropy: float,
) -> SemanticMessage:
    """Construct a C1 semantic message."""
    return SemanticMessage(
        sender_id=sender_id,
        argmax_cell=argmax_cell,
        sender_entropy=sender_entropy,
    )


def make_epistemic_message(
    sender_id: int,
    top_k_indices: NDArray[np.intp],
    top_k_probs: NDArray[np.float64],
    sender_entropy: float,
    gated: bool = False,
) -> EpistemicMessage:
    """
    Construct a C2 or C3 epistemic message.

    Parameters
    ----------
    gated : if True, marks the message as C3 (confidence-gated).
            The gate check (should_send_c3) must be performed by the caller
            before invoking this function.
    """
    return EpistemicMessage(
        sender_id=sender_id,
        top_k_indices=top_k_indices.copy(),
        top_k_probs=top_k_probs.copy(),
        sender_entropy=sender_entropy,
        _comm_type=CommType.C3_GATED if gated else CommType.C2_EPISTEMIC,
    )


# ---------------------------------------------------------------------------
# Message construction from Agent (convenience wrappers)
# ---------------------------------------------------------------------------

def build_message(
    agent,          # Agent — avoid circular import by not type-hinting
    comm_type: CommType,
    top_k: int,
    entropy_delta_threshold: float = 0.05,
) -> Message | None:
    """
    Build the appropriate outgoing message for an agent.

    Returns None if:
      - comm_type is C0 (no comms), or
      - comm_type is C3 and the entropy-delta gate does not fire.

    Parameters
    ----------
    agent                   : the sending Agent instance.
    comm_type               : which communication regime to use.
    top_k                   : number of cells to include in C2/C3 messages.
    entropy_delta_threshold : C3 gate threshold (nats); ignored for C0–C2.
    """
    if comm_type == CommType.C0_NONE:
        return None

    if comm_type == CommType.C1_SEMANTIC:
        return make_semantic_message(
            sender_id=agent.agent_id,
            argmax_cell=agent.belief_argmax,
            sender_entropy=agent.belief_entropy,
        )

    # C2 or C3
    if comm_type == CommType.C3_GATED:
        if not agent.should_send_c3(entropy_delta_threshold):
            return None   # gate did not fire

    indices, probs = agent.belief_top_k(top_k)
    msg = make_epistemic_message(
        sender_id=agent.agent_id,
        top_k_indices=indices,
        top_k_probs=probs,
        sender_entropy=agent.belief_entropy,
        gated=(comm_type == CommType.C3_GATED),
    )

    # Record the send so C3 resets its entropy baseline
    if comm_type in (CommType.C2_EPISTEMIC, CommType.C3_GATED):
        agent.record_send()

    return msg
