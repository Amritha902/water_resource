"""Tests for the aeration MDP, the reward functions and both RL agents."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wwtp.bsm1 import influent
from wwtp.bsm1.plant import KLA5_BOUNDS
from wwtp.rl.baselines import FuzzyAeration, PIDAeration
from wwtp.rl.ddpg import DDPGAgent, DDPGConfig, ReplayBuffer
from wwtp.rl.env import (NH_LIMIT, TN_LIMIT, AerationEnv, AerationEnvConfig)
from wwtp.rl.panda_rl import Constraint, PandaRLAgent, PandaRLConfig
from wwtp.rl.rewards import reward_ae_eq, reward_eq


@pytest.fixture(scope="module")
def series():
    return influent.canonical_scenario("dry", days=2.0)


# -- environment -------------------------------------------------------------

def test_paper_observation_has_three_channels(series):
    env = AerationEnv(series, AerationEnvConfig(), days=0.2)
    obs = env.reset()
    assert obs.shape == (3,) and env.n_obs == 3
    assert np.all(obs >= 0.0)


def test_augmented_observation_appends_the_actuator_position(series):
    cfg = AerationEnvConfig(obs_mode="augmented")
    env = AerationEnv(series, cfg, days=0.2)
    obs = env.reset()
    assert obs.shape == (4,) and env.n_obs == 4
    assert obs[3] == pytest.approx(84.0 / KLA5_BOUNDS[1])


def test_incremental_and_absolute_actions_differ_as_documented(series):
    inc = AerationEnv(series, AerationEnvConfig(da_max=10.0), days=0.2)
    inc.reset()
    inc.step(1.0)
    assert inc.kla == pytest.approx(94.0)          # 84 + 1.0 * 10

    absolute = AerationEnv(series, AerationEnvConfig(action_mode="absolute"),
                           days=0.2)
    absolute.reset()
    absolute.step(0.0)
    assert absolute.kla == pytest.approx(120.0)    # midpoint of 0..240


def test_actuator_stays_inside_the_bsm1_bounds(series):
    env = AerationEnv(series, AerationEnvConfig(da_max=200.0), days=0.3)
    env.reset()
    for action in (1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0):
        env.step(action)
        assert KLA5_BOUNDS[0] <= env.kla <= KLA5_BOUNDS[1]


def test_action_interval_advances_several_control_periods(series):
    env = AerationEnv(series, AerationEnvConfig(action_interval=20), days=0.5)
    env.reset()
    _, info, _ = env.step(0.0)
    assert info["step"] == 20
    assert len(info["eff_samples"]) == 20
    # the reported values are means over the interval
    assert info["S_NH"] == pytest.approx(
        float(np.mean([z[9] for z in info["eff_samples"]])), rel=1e-6)
    assert info["NH_peak"] >= info["S_NH"]


def test_episode_terminates_at_the_requested_length(series):
    env = AerationEnv(series, AerationEnvConfig(action_interval=20), days=0.5)
    env.reset()
    done, steps = False, 0
    while not done and steps < 10_000:
        _, _, done = env.step(0.0)
        steps += 1
    assert done and env.k >= env.n_steps


# -- rewards -----------------------------------------------------------------

def test_reward_eq_only_charges_for_exceedances():
    clean = {"S_NH": 2.0, "N_tot": 15.0, "aeration_fraction": 0.9}
    assert reward_eq(clean) == 0.0
    over = {"S_NH": NH_LIMIT + 1.0, "N_tot": TN_LIMIT + 2.0,
            "aeration_fraction": 0.0}
    assert reward_eq(over) == pytest.approx(-(1.0 + 0.38 * 2.0))


def test_reward_ae_eq_pays_for_air_and_for_ammonium():
    info = {"S_NH": NH_LIMIT + 2.0, "N_tot": 10.0, "aeration_fraction": 0.5}
    assert reward_ae_eq(info) == pytest.approx(-(0.5 + 0.42 * 2.0))
    # the published coefficients must prefer sensible operation to no air
    pid_like = {"S_NH": 2.0, "N_tot": 16.0, "aeration_fraction": 0.62}
    starved = {"S_NH": 9.9, "N_tot": 15.9, "aeration_fraction": 0.0}
    assert reward_ae_eq(pid_like) > reward_ae_eq(starved)


# -- fixed-set-point baselines ----------------------------------------------

@pytest.mark.parametrize("ctrl", [PIDAeration, FuzzyAeration])
def test_baselines_hold_the_do_set_point(series, ctrl):
    controller = ctrl()
    env = AerationEnv(series, AerationEnvConfig(), days=1.5)
    obs = env.reset()
    controller.reset()
    do = []
    while True:
        obs, info, done = env.step(controller.act(obs))
        do.append(info["S_O5"])
        if done:
            break
    assert abs(float(np.mean(do[-200:])) - 2.0) < 0.15


# -- DDPG --------------------------------------------------------------------

def test_replay_buffer_is_a_ring():
    buf = ReplayBuffer(4, 2, 1)
    for i in range(6):
        buf.add(np.full(2, i), np.array([i]), float(i), np.full(2, i), False)
    assert buf.size == 4 and buf.ptr == 2
    o, a, r, o2, d = buf.sample(4, np.random.default_rng(0))
    assert o.shape == (4, 2) and r.shape == (4, 1)


def test_rewards_are_scaled_by_one_minus_gamma():
    agent = DDPGAgent(DDPGConfig(n_obs=3, gamma=0.99, warmup_steps=1))
    assert agent.reward_scale == pytest.approx(0.01)
    agent.observe(np.zeros(3), 0.0, -2.5, np.zeros(3), False)
    assert float(agent.buffer.rew[0, 0]) == pytest.approx(-0.025)


def test_initial_policy_holds_position():
    """The small final layer makes the untrained increment ~0 (paper 1's idea)."""
    agent = DDPGAgent(DDPGConfig(n_obs=4))
    actions = [abs(agent.act(np.random.default_rng(i).random(4)))
               for i in range(20)]
    assert max(actions) < 0.1


def test_soft_update_moves_targets_towards_the_online_nets():
    agent = DDPGAgent(DDPGConfig(n_obs=3, tau=0.5, warmup_steps=1))
    with torch.no_grad():
        for p in agent.critic.parameters():
            p.add_(1.0)
    before = [p.clone() for p in agent.critic_target.parameters()]
    agent._soft_update(agent.critic, agent.critic_target)
    after = list(agent.critic_target.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_pre_activation_penalty_pulls_a_saturated_actor_back():
    """A saturated tanh has no gradient; the penalty must still move it."""
    agent = DDPGAgent(DDPGConfig(n_obs=3, warmup_steps=1, batch_size=8,
                                 pre_act_penalty=1.0, lr_actor=1e-2))
    with torch.no_grad():
        last = [m for m in agent.actor.trunk if isinstance(m, torch.nn.Linear)][-1]
        last.bias.add_(20.0)                     # force deep saturation
    obs = np.random.default_rng(0).random(3)
    assert abs(agent.act(obs)) > 0.99
    for _ in range(64):
        agent.observe(obs, 1.0, -1.0, obs, False)
    with torch.no_grad():
        pre_before = float(agent.actor.pre_activation(
            torch.as_tensor(obs, dtype=torch.float32)[None]).abs())
    for _ in range(200):
        agent.update()
    with torch.no_grad():
        pre_after = float(agent.actor.pre_activation(
            torch.as_tensor(obs, dtype=torch.float32)[None]).abs())
    assert pre_after < pre_before


# -- PANDA-RL ----------------------------------------------------------------

def _panda(**kwargs):
    cfg = PandaRLConfig(n_obs=4, warmup_steps=1, batch_size=8, **kwargs)
    return PandaRLAgent(cfg)


def test_distributional_critic_emits_one_output_per_quantile():
    agent = _panda(n_quantiles=16)
    assert agent.critic[-1].out_features == 16


def test_cvar_is_the_mean_of_the_worst_tail():
    agent = _panda(n_quantiles=10, cvar_alpha=0.3)
    z = torch.arange(10, dtype=torch.float32)[None]      # 0..9
    assert agent.n_tail == 3
    assert float(agent._cvar(z)) == pytest.approx(1.0)   # mean of 0, 1, 2


def test_cvar_with_alpha_one_is_the_mean():
    agent = _panda(n_quantiles=8, cvar_alpha=1.0)
    z = torch.randn(4, 8)
    assert torch.allclose(agent._cvar(z), z.mean(-1), atol=1e-6)


def test_quantile_huber_loss_is_zero_on_an_exact_match():
    agent = _panda(n_quantiles=8)
    z = torch.zeros(2, 8)
    assert float(agent._quantile_huber(z, z)) == pytest.approx(0.0)
    assert float(agent._quantile_huber(z, z + 1.0)) > 0.0


def test_multiplier_rises_when_the_limit_is_exceeded():
    agent = _panda(constraints=(Constraint("NH_peak", 4.0, 0.05, "NH"),),
                   lambda_init=1.0, lambda_lr=2.0)
    out = agent.end_episode({"rate_NH_peak": 0.25})
    assert out["lambda_NH"] > 1.0                        # 1 + 2 * (0.25 - 0.05)
    assert out["lambda_NH"] == pytest.approx(1.4)


def test_multiplier_falls_when_the_limit_is_respected_and_stays_non_negative():
    agent = _panda(constraints=(Constraint("NH_peak", 4.0, 0.20, "NH"),),
                   lambda_init=0.1, lambda_lr=2.0)
    out = agent.end_episode({"rate_NH_peak": 0.0})
    assert out["lambda_NH"] == 0.0                       # clipped at zero


def test_constraint_costs_are_violation_indicators():
    agent = _panda()
    costs = agent.constraint_costs({"NH_peak": 5.0, "TN_peak": 10.0})
    assert costs.tolist() == [1.0, 0.0]


def test_panda_reduces_to_ddpg_when_every_component_is_ablated():
    agent = _panda(n_quantiles=1, cvar_alpha=1.0, constraints=(),
                   use_forecast=False)
    assert agent.n_context == 0
    assert agent.critic[-1].out_features == 1
    assert agent.n_constraints == 0
    assert agent.context().shape == (0,)


def test_panda_update_runs_and_reports_the_risk_measure():
    agent = _panda(n_quantiles=8)
    x = np.random.default_rng(0).random(4)
    for _ in range(32):
        agent.observe(x, 0.0, -0.4, x, False, cost=np.array([1.0, 0.0]))
    stats = agent.update()
    assert {"loss_critic", "loss_actor", "cvar", "loss_cost"} <= set(stats)
    assert np.isfinite(list(stats.values())).all()


# -- the aeration twin and Dyna updates --------------------------------------

def test_aeration_twin_is_a_residual_model():
    from wwtp.rl.twin import AerationTwin
    twin = AerationTwin()
    obs = torch.rand(6, 3)
    with torch.no_grad():
        nxt, eff = twin(obs, torch.rand(6, 1), torch.rand(6, 3))
    assert nxt.shape == (6, 3) and eff.shape == (6, 2)
    # an untrained residual head leaves the state close to where it started
    assert float((nxt - obs).abs().max()) < 2.0


def test_aeration_corpus_excites_the_actuator_open_loop():
    """A closed-loop corpus would leave KLa and DO almost uncorrelated."""
    from wwtp.rl.twin import collect_corpus
    corpus = collect_corpus(n_rollouts=1, seed=0, days=1.5)
    u = corpus["u"][:, 0]
    assert u.min() < 0.15 and u.max() > 0.85      # the whole range is visited
    assert len(np.unique(np.round(u, 3))) > 5     # piecewise constant, not fixed
    assert corpus["obs"].shape[1] == 3
    assert np.all(np.isfinite(corpus["eff"]))


def test_dyna_produces_well_formed_synthetic_transitions():
    from wwtp.rl.twin import AerationTwin
    agent = PandaRLAgent(PandaRLConfig(n_obs=4, n_env_obs=4, warmup_steps=1,
                                       batch_size=8, dyna_ratio=2),
                         twin=AerationTwin())
    x = np.random.default_rng(0).random(4)
    for _ in range(32):
        agent.observe(x, 0.0, -0.4, x, False, cost=np.array([0.0, 0.0]),
                      inlet=np.array([18000.0, 30.0, 70.0]))
    o, a, r, o2, d, cost = agent._dyna_batch(7)
    assert o.shape == (7, 4) and o2.shape == (7, 4)
    assert cost.shape == (7, 2)
    assert bool(((a >= -1.0) & (a <= 1.0)).all())
    assert bool((r <= 0.0).all())                 # energy reward is never positive
    assert bool((o2 >= 0.0).all())
    assert torch.isfinite(torch.cat([o, a, r, o2, d, cost], dim=-1)).all()


def test_dyna_updates_run_alongside_the_real_ones():
    from wwtp.rl.twin import AerationTwin
    agent = PandaRLAgent(PandaRLConfig(n_obs=4, n_env_obs=4, warmup_steps=1,
                                       batch_size=8, dyna_ratio=3),
                         twin=AerationTwin())
    x = np.random.default_rng(1).random(4)
    for _ in range(32):
        agent.observe(x, 0.0, -0.4, x, False, cost=np.array([1.0, 0.0]),
                      inlet=np.array([18000.0, 30.0, 70.0]))
    stats = agent.update()
    assert "loss_critic_dyna" in stats
    assert np.isfinite(list(stats.values())).all()


def test_dyna_is_off_without_a_twin():
    agent = PandaRLAgent(PandaRLConfig(n_obs=4, warmup_steps=1, batch_size=8,
                                       dyna_ratio=4))
    x = np.random.default_rng(2).random(4)
    for _ in range(32):
        agent.observe(x, 0.0, -0.4, x, False, cost=np.array([0.0, 0.0]))
    assert "loss_critic_dyna" not in agent.update()
