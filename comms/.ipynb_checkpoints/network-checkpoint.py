"""
comms/network.py
----------------
Communication network with congestion-dependent packet loss and latency.

Packet loss model
-----------------
Loss probability is a function of the number of messages simultaneously
in the channel at the current time step:

    P(message i is dropped) = 1 - (1 - base_rate)^n

where n = total messages sent this step (including message i itself).

This models a shared channel: the more agents transmit at once, the more
likely any individual message is lost.  A base_rate of 0 gives a lossless
channel; a base_rate of ~0.05 with 4 agents sending simultaneously gives
P(drop) ≈ 1 - 0.95^4 ≈ 0.19.

Latency model
-------------
Messages sent at step t arrive at the recipient's inbox at step t + latency.
Latency = 0 means same-step delivery (synchronous comms).
Latency ≥ 1 introduces a delay; very old messages are discarded if they
arrive after episode end (the runner handles this gracefully).

The network is a logical fully-connected graph: every agent can send to
every other agent.  Point-to-point addressing is not used — each message
is broadcast to all other agents, and packet loss is evaluated independently
for each recipient.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np

from config import CommConfig
from comms.message import Message


# ---------------------------------------------------------------------------
# Pending message (wrapper that adds scheduled delivery time)
# ---------------------------------------------------------------------------

@dataclass
class PendingMessage:
    message: Message
    deliver_at_step: int          # step at which the message enters recipient's inbox
    recipient_id: int


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network:
    """
    Simulates the logical communication channel between agents.

    Usage per time step
    -------------------
    1. Agents call network.send(msg, sender_id, current_step) for each
       outgoing message.  All sends in a step should be registered before
       calling deliver().
    2. Call network.deliver(current_step) to get {agent_id: [messages]}
       of messages that are ready to be received this step.
    3. Recipients fuse delivered messages into their beliefs.
    """

    def __init__(
        self,
        config: CommConfig,
        num_agents: int,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.num_agents = num_agents
        self.rng = rng

        # Queue of (PendingMessage) not yet delivered
        self._queue: list[PendingMessage] = []

        # Stats for metrics
        self.messages_sent_this_episode: int = 0
        self.messages_dropped_this_episode: int = 0
        self.bytes_sent_this_episode: int = 0

        # Track how many messages are in-flight per step for congestion calc
        # Maps step -> count of messages registered for that step
        self._messages_in_step: dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear queues and stats for a new episode."""
        self._queue.clear()
        self._messages_in_step.clear()
        self.messages_sent_this_episode = 0
        self.messages_dropped_this_episode = 0
        self.bytes_sent_this_episode = 0

    def send(
        self,
        message: Message,
        sender_id: int,
        current_step: int,
    ) -> None:
        """
        Register an outgoing broadcast message from sender_id.

        The message will be sent to all agents except the sender.
        Packet loss is evaluated per-recipient after all sends in this
        step have been registered (see _apply_loss).

        Call this for every message an agent wants to send.
        Then call flush_step() once all agents have submitted their
        messages for this step.
        """
        self._messages_in_step[current_step] += 1

        deliver_at = current_step + self.config.latency_steps

        for recipient_id in range(self.num_agents):
            if recipient_id == sender_id:
                continue
            # Count each recipient-message as one sent unit for accurate drop_rate
            self.messages_sent_this_episode += 1
            self.bytes_sent_this_episode += message.byte_size
            self._queue.append(
                PendingMessage(
                    message=message,
                    deliver_at_step=deliver_at,
                    recipient_id=recipient_id,
                )
            )

    def deliver(self, current_step: int) -> dict[int, list[Message]]:
        """
        Apply packet loss and return messages ready for delivery this step.

        Returns
        -------
        dict mapping recipient_id → list of Message objects that survived
        packet loss and are scheduled to arrive at current_step.

        Call once per step, after all sends have been registered.
        """
        inbox: dict[int, list[Message]] = defaultdict(list)
        still_pending: list[PendingMessage] = []

        # Congestion: how many messages were sent at the source step
        # (i.e. when each message was originally transmitted)
        for pm in self._queue:
            source_step = pm.deliver_at_step - self.config.latency_steps
            n_concurrent = self._messages_in_step.get(source_step, 1)
            drop_prob = self._congestion_loss_prob(n_concurrent)

            if pm.deliver_at_step == current_step:
                # Message is due — apply loss
                if self.rng.random() < drop_prob:
                    self.messages_dropped_this_episode += 1
                    # Dropped: do not deliver, do not re-queue
                else:
                    inbox[pm.recipient_id].append(pm.message)
            else:
                # Not yet due — keep in queue
                still_pending.append(pm)

        self._queue = still_pending
        return dict(inbox)

    def drop_rate_this_episode(self) -> float:
        """Fraction of sent messages that were dropped."""
        if self.messages_sent_this_episode == 0:
            return 0.0
        return self.messages_dropped_this_episode / self.messages_sent_this_episode

    # ------------------------------------------------------------------
    # Congestion model
    # ------------------------------------------------------------------

    def _congestion_loss_prob(self, n_concurrent: int) -> float:
        """
        P(drop) = 1 - (1 - base_rate)^n_concurrent

        Properties:
          n=0  → P=0       (no traffic, no loss)
          n=1  → P=base    (single message, base loss rate)
          n=4  → P≈4*base  for small base (linear approximation)
          n→∞  → P→1       (saturated channel)
        """
        base = self.config.base_packet_loss_rate
        if base <= 0.0:
            return 0.0
        if base >= 1.0:
            return 1.0
        return 1.0 - (1.0 - base) ** n_concurrent


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_network(config: CommConfig, num_agents: int, seed: int) -> Network:
    rng = np.random.default_rng(seed)
    return Network(config=config, num_agents=num_agents, rng=rng)
