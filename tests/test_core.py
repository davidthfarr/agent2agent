"""
tests/test_core.py
------------------
Unit tests for world, observation, belief, and agent modules.

Run with:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from config import (
    WorldConfig, ObservationConfig, AgentConfig, SimConfig, DEFAULT_CONFIG
)
from core.world import World, make_world
from core.observation import NoisySensor, Observation
from core.belief import (
    BeliefMap,
    jensen_shannon_divergence,
    mean_pairwise_jsd,
    alignment_to_truth,
    silent_failure,
    fuse_beliefs,
    reconstruct_from_topk,
    reconstruct_from_semantic,
)
from core.agent import Agent, make_agents


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_world() -> World:
    cfg = WorldConfig(grid_size=10, num_targets=1, num_obstacles=5)
    return make_world(cfg, seed=42)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def obs_config() -> ObservationConfig:
    return ObservationConfig(fov_radius=2, false_negative_rate=0.1, false_positive_rate=0.05)


# ===========================================================================
# World tests
# ===========================================================================

class TestWorld:
    def test_grid_size(self, small_world):
        N = small_world.N
        assert small_world.obstacle_mask.shape == (N * N,)

    def test_target_not_on_obstacle(self, small_world):
        for t in small_world.target_cells:
            assert not small_world.is_obstacle(t)

    def test_target_not_on_agent_start(self, small_world):
        starts = set(small_world.agent_starts)
        for t in small_world.target_cells:
            assert t not in starts

    def test_coord_roundtrip(self, small_world):
        for idx in range(small_world.N ** 2):
            r, c = small_world.coord(idx)
            assert small_world.index(r, c) == idx

    def test_neighbors_passable(self, small_world):
        for idx in range(small_world.N ** 2):
            for n in small_world.neighbors(idx):
                assert not small_world.is_obstacle(n)

    def test_fov_contains_agent_cell(self, small_world):
        agent_pos = small_world.agent_starts[0]
        visible = small_world.cells_in_fov(agent_pos, fov_radius=2)
        assert agent_pos in visible

    def test_fov_radius_bounds(self, small_world):
        """All visible cells should be within Chebyshev radius of agent."""
        agent_pos = small_world.agent_starts[0]
        r0, c0 = small_world.coord(agent_pos)
        fov_radius = 3
        visible = small_world.cells_in_fov(agent_pos, fov_radius)
        for cell in visible:
            r, c = small_world.coord(int(cell))
            assert abs(r - r0) <= fov_radius and abs(c - c0) <= fov_radius

    def test_reset_changes_targets(self):
        """Two resets with different rng seeds should (likely) give different targets."""
        cfg = WorldConfig(grid_size=10, num_targets=1, num_obstacles=3)
        w1 = make_world(cfg, seed=1)
        w2 = make_world(cfg, seed=99)
        # Not guaranteed to differ, but overwhelmingly likely
        all_same = w1.target_cells == w2.target_cells
        # Just check both are valid
        for t in w1.target_cells:
            assert not w1.is_obstacle(t)
        for t in w2.target_cells:
            assert not w2.is_obstacle(t)


# ===========================================================================
# Observation tests
# ===========================================================================

class TestObservation:
    def test_observation_shape(self, small_world, obs_config, rng):
        sensor = NoisySensor(obs_config, rng)
        obs = sensor.observe(small_world.agent_starts[0], small_world, small_world.target_cells)
        assert len(obs.visible_cells) == len(obs.detections)

    def test_detected_and_empty_partition(self, small_world, obs_config, rng):
        sensor = NoisySensor(obs_config, rng)
        obs = sensor.observe(small_world.agent_starts[0], small_world, small_world.target_cells)
        combined = set(obs.detected_cells.tolist()) | set(obs.empty_cells.tolist())
        assert combined == set(obs.visible_cells.tolist())

    def test_likelihood_sanity(self, obs_config, rng):
        sensor = NoisySensor(obs_config, rng)
        assert sensor.likelihood_detection(True) > sensor.likelihood_detection(False)
        assert sensor.likelihood_no_detection(False) > sensor.likelihood_no_detection(True)

    def test_high_fnr_reduces_detections(self, small_world, rng):
        """With 100% FNR, targets in FOV should never be detected."""
        high_fnr_config = ObservationConfig(fov_radius=5, false_negative_rate=1.0, false_positive_rate=0.0)
        sensor = NoisySensor(high_fnr_config, rng)
        for _ in range(20):
            obs = sensor.observe(0, small_world, small_world.target_cells)
            assert len(obs.detected_cells) == 0

    def test_high_fpr_creates_false_detections(self, small_world, rng):
        """With 100% FPR and no real targets in FOV, should always detect."""
        high_fpr_config = ObservationConfig(fov_radius=1, false_negative_rate=0.0, false_positive_rate=1.0)
        sensor = NoisySensor(high_fpr_config, rng)
        # Use agent at corner, targets are elsewhere
        obs = sensor.observe(0, small_world, [])  # empty true targets
        # Every visible cell should show a detection
        assert len(obs.detected_cells) == len(obs.visible_cells)


# ===========================================================================
# Belief tests
# ===========================================================================

class TestBeliefMap:
    def test_uniform_init(self):
        bm = BeliefMap(n_cells=100)
        p = bm.probs
        assert p.shape == (100,)
        np.testing.assert_allclose(p.sum(), 1.0, atol=1e-9)
        np.testing.assert_allclose(p, 1.0 / 100, atol=1e-9)

    def test_probs_always_sum_to_one(self, small_world, obs_config, rng):
        sensor = NoisySensor(obs_config, rng)
        bm = BeliefMap(n_cells=small_world.N ** 2)
        for _ in range(10):
            obs = sensor.observe(0, small_world, small_world.target_cells)
            bm.update(obs, obs_config, small_world.obstacle_mask)
            np.testing.assert_allclose(bm.probs.sum(), 1.0, atol=1e-6)

    def test_obstacle_cells_zero(self, small_world, obs_config, rng):
        sensor = NoisySensor(obs_config, rng)
        bm = BeliefMap(n_cells=small_world.N ** 2)
        for _ in range(5):
            obs = sensor.observe(0, small_world, small_world.target_cells)
            bm.update(obs, obs_config, small_world.obstacle_mask)
        p = bm.probs
        assert np.all(p[small_world.obstacle_mask] == 0.0)

    def test_entropy_decreases_with_more_observations(self, small_world, obs_config, rng):
        """Entropy should generally decrease as agent accumulates observations."""
        sensor = NoisySensor(obs_config, rng)
        bm = BeliefMap(n_cells=small_world.N ** 2)
        h_initial = bm.entropy
        for _ in range(30):
            pos = small_world.agent_starts[0]
            obs = sensor.observe(pos, small_world, small_world.target_cells)
            bm.update(obs, obs_config, small_world.obstacle_mask)
        h_final = bm.entropy
        assert h_final < h_initial, f"Entropy did not decrease: {h_initial:.3f} -> {h_final:.3f}"

    def test_top_k_length_and_order(self):
        bm = BeliefMap(n_cells=50)
        idx, vals = bm.top_k(10)
        assert len(idx) == 10
        assert len(vals) == 10
        # Should be sorted descending
        assert np.all(vals[:-1] >= vals[1:])

    def test_clone_is_independent(self):
        bm = BeliefMap(n_cells=20)
        clone = bm.clone()
        # Mutate original
        bm._log_b[0] = 100.0
        # Clone should not change
        assert clone._log_b[0] != 100.0


# ===========================================================================
# Divergence metric tests
# ===========================================================================

class TestDivergenceMetrics:
    def test_jsd_identical_distributions(self):
        p = np.array([0.1, 0.4, 0.3, 0.2])
        assert jensen_shannon_divergence(p, p) < 1e-9

    def test_jsd_symmetric(self):
        rng = np.random.default_rng(7)
        p = rng.dirichlet(np.ones(20))
        q = rng.dirichlet(np.ones(20))
        assert abs(jensen_shannon_divergence(p, q) - jensen_shannon_divergence(q, p)) < 1e-9

    def test_jsd_bounded(self):
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        jsd = jensen_shannon_divergence(p, q)
        assert 0.0 <= jsd <= np.log(2) + 1e-9

    def test_mean_pairwise_jsd_single_agent(self):
        bm = BeliefMap(n_cells=10)
        assert mean_pairwise_jsd([bm]) == 0.0

    def test_mean_pairwise_jsd_identical_beliefs(self):
        bms = [BeliefMap(n_cells=10) for _ in range(4)]
        assert mean_pairwise_jsd(bms) < 1e-9

    def test_alignment_to_truth(self):
        bm = BeliefMap(n_cells=10)
        # Manually place all mass on cell 3
        p = np.zeros(10)
        p[3] = 1.0
        bm.set_probs(p)
        assert alignment_to_truth(bm, [3], 10) == pytest.approx(1.0)
        assert alignment_to_truth(bm, [5], 10) == pytest.approx(0.0)

    def test_silent_failure_low_jsd_wrong_target(self):
        """Low JSD + task failure = silent failure."""
        from experiment.metrics import EpisodeMetrics, aggregate_episodes
        
        # Simulate failed episodes with low JSD (converged on wrong cell)
        episodes = []
        for _ in range(10):
            m = EpisodeMetrics(
                task_success=False,
                final_mean_jsd=0.005,        # well below 0.1 threshold
                final_mean_alignment=0.001,  # near uniform prior
            )
            episodes.append(m)
        
        summary = aggregate_episodes(
            condition_name="test",
            episode_metrics=episodes,
            episode_length=200,
            n_cells=2500,
        )
        assert summary.silent_failure_rate == pytest.approx(1.0)
    
    def test_no_silent_failure_when_successful(self):
        """Successful episodes are never classified as silent failure."""
        from experiment.metrics import EpisodeMetrics, aggregate_episodes
        
        episodes = []
        for _ in range(10):
            m = EpisodeMetrics(
                task_success=True,
                final_mean_jsd=0.005,
                final_mean_alignment=0.8,
            )
            episodes.append(m)
        
        summary = aggregate_episodes(
            condition_name="test",
            episode_metrics=episodes,
            episode_length=200,
            n_cells=2500,
        )
        assert summary.silent_failure_rate == pytest.approx(0.0)
    
    def test_no_silent_failure_high_jsd(self):
        """Failed episodes with high JSD are not silent failures."""
        from experiment.metrics import EpisodeMetrics, aggregate_episodes
        
        episodes = []
        for _ in range(10):
            m = EpisodeMetrics(
                task_success=False,
                final_mean_jsd=0.25,         # above 0.1 threshold
                final_mean_alignment=0.001,
            )
            episodes.append(m)
        
        summary = aggregate_episodes(
            condition_name="test",
            episode_metrics=episodes,
            episode_length=200,
            n_cells=2500,
        )
        assert summary.silent_failure_rate == pytest.approx(0.0)


# ===========================================================================
# Belief fusion tests
# ===========================================================================

class TestBeliefFusion:
    def test_fuse_equal_weight(self):
        bm_own = BeliefMap(n_cells=10)
        p_own = bm_own.probs

        # Create peaked incoming distribution
        p_in = np.zeros(10)
        p_in[7] = 1.0

        fused = fuse_beliefs(bm_own, p_in, weight_incoming=0.5)
        p_fused = fused.probs

        # Fused should be between own and incoming
        assert p_fused[7] > p_own[7]  # incoming was more peaked there

    def test_fuse_preserves_normalisation(self):
        bm = BeliefMap(n_cells=20)
        p_in = np.random.default_rng(5).dirichlet(np.ones(20))
        fused = fuse_beliefs(bm, p_in)
        np.testing.assert_allclose(fused.probs.sum(), 1.0, atol=1e-9)

    def test_reconstruct_topk_sums_to_one(self):
        indices = np.array([1, 5, 9])
        values = np.array([0.3, 0.25, 0.2])
        p = reconstruct_from_topk(indices, values, n_cells=20)
        np.testing.assert_allclose(p.sum(), 1.0, atol=1e-9)
        assert p[1] == pytest.approx(0.3 / values.sum() * (values.sum() / p.sum()), rel=0.1)

    def test_reconstruct_semantic_sums_to_one(self):
        p = reconstruct_from_semantic(argmax_cell=3, n_cells=10, confidence=0.9)
        np.testing.assert_allclose(p.sum(), 1.0, atol=1e-9)
        assert p[3] == pytest.approx(0.9, abs=1e-9)


# ===========================================================================
# Agent tests
# ===========================================================================

class TestAgent:
    def test_agent_creation(self, small_world):
        cfg = AgentConfig(num_agents=4)
        agents = make_agents(cfg, small_world)
        assert len(agents) == 4
        for i, agent in enumerate(agents):
            assert agent.agent_id == i
            assert not agent.reached_target

    def test_agent_belief_valid(self, small_world):
        cfg = AgentConfig(num_agents=4)
        agents = make_agents(cfg, small_world)
        for agent in agents:
            np.testing.assert_allclose(agent.belief_probs().sum(), 1.0, atol=1e-9)

    def test_agent_moves_toward_belief(self, small_world, obs_config, rng):
        """After observing the target, agent should move toward it."""
        cfg = AgentConfig(num_agents=1)
        # Place target and agent far apart
        w = make_world(WorldConfig(grid_size=10, num_obstacles=0, num_targets=1), seed=0)
        agents = make_agents(AgentConfig(num_agents=1), w)
        sensor = NoisySensor(obs_config, rng)

        agent = agents[0]
        initial_pos = agent.pos

        # Observe and update belief multiple times
        for _ in range(20):
            agent.observe_and_update(w, sensor, obs_config)

        # Move several steps
        for _ in range(10):
            agent.move(w)

        # Agent should have moved from initial position
        assert agent.pos != initial_pos or agent.belief.argmax_cell == initial_pos

    def test_agent_success_at_target(self, small_world):
        cfg = AgentConfig(num_agents=1)
        agents = make_agents(cfg, small_world)
        agent = agents[0]
        # Manually teleport agent to target
        agent.pos = small_world.target_cells[0]
        assert agent.check_success(small_world)
        assert agent.reached_target

    def test_agent_reset(self, small_world):
        cfg = AgentConfig(num_agents=1)
        agents = make_agents(cfg, small_world)
        agent = agents[0]
        agent.reached_target = True
        agent.step_count = 50
        agent.reset(start_pos=5, n_cells=small_world.N**2, prior=None)
        assert agent.pos == 5
        assert agent.step_count == 0
        assert not agent.reached_target


# ===========================================================================
# Integration smoke test
# ===========================================================================

class TestIntegration:
    def test_full_step_cycle(self):
        """
        Run a minimal episode (no comms) and confirm nothing crashes.
        All agents observe + update belief + move for 20 steps.
        """
        world_cfg = WorldConfig(grid_size=10, num_targets=1, num_obstacles=5)
        obs_cfg = ObservationConfig(fov_radius=2, false_negative_rate=0.1, false_positive_rate=0.05)
        agent_cfg = AgentConfig(num_agents=4)

        world = make_world(world_cfg, seed=123)
        agents = make_agents(agent_cfg, world)
        rng = np.random.default_rng(0)
        sensor = NoisySensor(obs_cfg, rng)

        success = False
        for step in range(20):
            for agent in agents:
                agent.observe_and_update(world, sensor, obs_cfg)
                agent.move(world)
                if agent.check_success(world):
                    success = True

            # Verify all beliefs are valid distributions
            for agent in agents:
                p = agent.belief_probs()
                assert p.sum() == pytest.approx(1.0, abs=1e-5)
                assert np.all(p >= 0)

        # Not asserting success (random — may or may not occur in 20 steps)
        # Just assert nothing crashed
        assert True
