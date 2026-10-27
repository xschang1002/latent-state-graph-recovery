"""Tests for the revision baselines (PCMCI+ and MSVAR).

Heavy third-party fits (tigramite ``run_pcmciplus``, statsmodels
``MarkovAutoregression.fit``) are replaced with monkeypatched fakes so the
tests check shape, direction, and self-edge conventions in milliseconds.
"""

import numpy as np
import pytest

from stcg import msvar_scores, pcmciplus_scores, rpcmci_scores


def _fake_pcmciplus_result(n_nodes: int, tau_max: int):
    val = np.zeros((n_nodes, n_nodes, tau_max + 1))
    val[1, 2, 1] = 0.7  # source 1 -> target 2 at lag 1
    val[2, 1, 1] = 0.1  # weaker reverse direction: checks orientation
    val[1, 1, 2] = 0.9  # self-edge must be zeroed
    val[3, 0, 3] = 0.5  # source 3 -> target 0 at lag 3
    val[0, 3, 0] = 5.0  # contemporaneous tau=0 must be dropped
    val[0, 2, 2] = np.nan  # NaN must be cleaned
    p = np.ones((n_nodes, n_nodes, tau_max + 1))  # untested pairs -> zero score
    p[0, 1, 1] = 0.01  # -log10 -> 2
    p[1, 0, 2] = 1e-5  # -log10 -> 5
    graph = np.full((n_nodes, n_nodes, tau_max + 1), "", dtype="<U3")
    return {"graph": graph, "p_matrix": p, "val_matrix": val}


@pytest.fixture
def fake_run_pcmciplus(monkeypatch):
    from tigramite.pcmci import PCMCI

    captured = {}

    def fake_run(self, **kwargs):
        captured.update(kwargs)
        return _fake_pcmciplus_result(n_nodes=4, tau_max=3)

    monkeypatch.setattr(PCMCI, "run_pcmciplus", fake_run)
    return captured


def test_pcmciplus_scores_shape_direction_and_self_edges(fake_run_pcmciplus):
    x = np.random.default_rng(0).normal(size=(60, 4))
    scores = pcmciplus_scores(x, max_lag=3, pc_alpha=0.05)

    assert fake_run_pcmciplus["tau_min"] == 1
    assert fake_run_pcmciplus["tau_max"] == 3
    assert fake_run_pcmciplus["pc_alpha"] == 0.05
    assert scores.shape == (3, 4, 4)
    assert scores[0, 1, 2] == pytest.approx(0.7)
    assert scores[0, 2, 1] == pytest.approx(0.1)
    assert scores[2, 3, 0] == pytest.approx(0.5)
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)
    assert np.all(np.isfinite(scores))
    assert scores.max() == pytest.approx(0.7)  # tau=0 value 5.0 dropped


def test_pcmciplus_scores_p_value_transform(fake_run_pcmciplus):
    x = np.random.default_rng(1).normal(size=(60, 4))
    scores = pcmciplus_scores(x, max_lag=3, score="p_value")

    assert scores[0, 0, 1] == pytest.approx(2.0)
    assert scores[1, 1, 0] == pytest.approx(5.0)
    # Untested pairs carry p=1 in the fake, giving a zero score.
    assert scores[0, 3, 1] == pytest.approx(0.0)
    assert np.all(scores >= 0.0)
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)


def _msvar_param_table():
    """(target, exog column, regime) -> coefficient for the MSVAR fake."""

    return {
        (0, 1): (0.9, 0.1),
        (0, 2): (0.2, 0.7),
        (0, 3): (0.3, 0.0),
        (0, 4): (0.0, 0.4),
        (1, 1): (0.15, 0.55),
        (1, 2): (0.05, 0.25),
        (2, 1): (0.45, 0.35),
    }


def _install_fake_msvar(monkeypatch, failing_targets=(), retry_targets=()):
    """Patch statsmodels MarkovAutoregression with a deterministic fake.

    The fake mimics the real parameter naming (``x{k}[regime]`` switching
    exog, ``ar.L{lag}[regime]`` own-lag AR, ``const``/``sigma2``/``p[->]``)
    and identifies the target equation by creation order, which matches the
    sequential per-target fits in ``msvar_scores``.
    """

    instances: list = []

    class FakeResults:
        def __init__(self, model, converged):
            self.model = model
            self.params = model.values
            self.mle_retvals = {"converged": converged}
            probs = np.tile(np.array([0.6, 0.4]), (model.nobs_effective, 1))
            self.smoothed_marginal_probabilities = probs

    class FakeMarkovAutoregression:
        def __init__(
            self,
            endog,
            k_regimes,
            order,
            trend="c",
            exog=None,
            switching_ar=True,
            switching_trend=True,
            switching_exog=False,
            switching_variance=False,
        ):
            self.target_idx = len(instances)
            self.k_regimes = k_regimes
            self.order = order
            self.nobs_effective = len(endog) - order
            n_exog = exog.shape[1]
            table = _msvar_param_table()
            names = []
            values = []
            for regime in range(k_regimes):
                names.append(f"const[{regime}]")
                values.append(0.01)
            for column in range(1, n_exog + 1):
                for regime in range(k_regimes):
                    names.append(f"x{column}[{regime}]")
                    values.append(table.get((self.target_idx, column), (0.0, 0.0))[regime])
            for regime in range(k_regimes):
                names.append(f"sigma2[{regime}]")
                values.append(1.0)
            for lag in range(1, order + 1):
                for regime in range(k_regimes):
                    names.append(f"ar.L{lag}[{regime}]")
                    values.append(99.0)  # must not leak into self-edge scores
            for regime in range(k_regimes):
                names.append(f"p[{regime}->0]")
                values.append(0.5)
            self.param_names = names
            self.values = np.asarray(values, dtype=float)
            self.fit_calls = 0
            instances.append(self)

        def fit(self, **kwargs):
            self.fit_calls += 1
            if self.target_idx in failing_targets:
                raise RuntimeError("synthetic optimizer failure")
            converged = True
            if self.target_idx in retry_targets and self.fit_calls == 1:
                converged = False
            return FakeResults(self, converged=converged)

    monkeypatch.setattr(
        "statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression",
        FakeMarkovAutoregression,
    )
    return instances


def test_msvar_scores_shape_direction_self_edges_and_regime_max(monkeypatch):
    _install_fake_msvar(monkeypatch)
    x = np.random.default_rng(2).normal(size=(60, 3))
    scores = msvar_scores(x, max_lag=2, n_regimes=2)

    assert scores.shape == (2, 3, 3)
    assert np.all(np.isfinite(scores))
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)
    # Target 0 (others [1, 2]): x1/x3 -> source 1, x2/x4 -> source 2.
    assert scores[0, 1, 0] == pytest.approx(0.9)  # max(|0.9|, |0.1|)
    assert scores[0, 2, 0] == pytest.approx(0.7)  # max across regimes
    assert scores[1, 1, 0] == pytest.approx(0.3)  # lag 2
    assert scores[1, 2, 0] == pytest.approx(0.4)
    # Direction: target 1 (others [0, 2]) gives distinct values.
    assert scores[0, 0, 1] == pytest.approx(0.55)
    assert scores[0, 1, 0] != pytest.approx(scores[0, 0, 1])
    assert scores[0, 0, 2] == pytest.approx(0.45)


def test_msvar_scores_info_reports_regimes_and_offset(monkeypatch):
    _install_fake_msvar(monkeypatch)
    x = np.random.default_rng(3).normal(size=(60, 3))
    scores, info = msvar_scores(x, max_lag=2, n_regimes=2, return_info=True)

    assert info["effective_offset"] == 4
    assert info["regime_probabilities"].shape == (56, 2)
    assert info["regime_sequence"].shape == (56,)
    assert info["failed_targets"] == []
    assert info["converged"] == [True, True, True]
    assert np.allclose(np.asarray(info["regime_probabilities"]).sum(axis=1), 1.0)


def test_msvar_scores_handles_failed_target(monkeypatch):
    _install_fake_msvar(monkeypatch, failing_targets=(1,))
    x = np.random.default_rng(4).normal(size=(60, 3))
    scores, info = msvar_scores(x, max_lag=2, n_regimes=2, return_info=True)

    assert info["failed_targets"] == [1]
    assert info["converged"] == [True, False, True]
    assert np.all(scores[:, :, 1] == 0.0)  # failed target column stays zero
    assert scores[0, 1, 0] == pytest.approx(0.9)  # other targets still scored
    assert np.all(np.isfinite(scores))


def test_msvar_scores_retries_non_converged_fit(monkeypatch):
    instances = _install_fake_msvar(monkeypatch, retry_targets=(0,))
    x = np.random.default_rng(5).normal(size=(60, 3))
    _, info = msvar_scores(x, max_lag=2, n_regimes=2, return_info=True)

    assert instances[0].fit_calls == 2  # initial attempt + retry
    assert info["converged"][0] is True
    assert info["failed_targets"] == []


def test_msvar_scores_rejects_short_series():
    with pytest.raises(ValueError, match="timesteps"):
        msvar_scores(np.zeros((6, 3)), max_lag=2)


def _install_fake_rpcmci(monkeypatch, error_free: int = 7):
    """Patch tigramite RPCMCI with a deterministic fake.

    The fake mimics the real ``run_rpcmci`` return contract: a one-hot-ish
    ``gamma`` of shape ``(num_regimes, T)`` (first half regime 0, second half
    regime 1), per-regime ``run_pcmci`` results with ``val_matrix`` /
    ``p_matrix`` indexed ``source, target, tau``, and the
    ``error_free_annealings`` counter. With ``error_free=0`` it returns
    ``None`` like the real implementation does when nothing converges.
    """

    captured: dict = {}

    class FakeRPCMCI:
        def __init__(
            self,
            dataframe,
            cond_ind_test=None,
            prediction_model=None,
            seed=None,
            verbosity=-1,
        ):
            self.dataframe = dataframe
            self.seed = seed

        def run_rpcmci(self, **kwargs):
            captured.update(kwargs)
            if error_free == 0:
                return None
            tau_max = kwargs["tau_max"]
            num_regimes = kwargs["num_regimes"]
            timesteps = int(self.dataframe.values[0].shape[0])
            gamma = np.zeros((num_regimes, timesteps))
            gamma[0, : timesteps // 2] = 1.0
            gamma[1 % num_regimes, timesteps // 2 :] = 1.0

            val0 = np.zeros((4, 4, tau_max + 1))
            val0[1, 2, 1] = 0.7  # source 1 -> target 2 at lag 1
            val0[2, 1, 1] = 0.1  # weaker reverse direction: checks orientation
            val0[1, 1, 2] = 0.9  # self-edge must be zeroed
            val0[0, 3, 0] = 5.0  # contemporaneous tau=0 must be dropped
            val1 = np.zeros((4, 4, tau_max + 1))
            val1[1, 2, 1] = 0.3  # max over regimes keeps the regime-0 value
            val1[3, 0, 3] = 0.9  # edge active only in regime 1
            val1[2, 2, 1] = 0.8  # self-edge must be zeroed
            val1[0, 2, 2] = np.nan  # NaN must be cleaned

            p0 = np.ones((4, 4, tau_max + 1))  # untested pairs -> zero score
            p0[1, 2, 1] = 0.01  # -log10 -> 2
            p1 = np.ones((4, 4, tau_max + 1))
            p1[1, 2, 1] = 1e-5  # min over regimes -> -log10 = 5
            p1[3, 0, 3] = 0.001  # only tested in regime 1 -> 3

            return {
                "regimes": gamma,
                "causal_results": {
                    0: {"val_matrix": val0, "p_matrix": p0},
                    1: {"val_matrix": val1, "p_matrix": p1},
                },
                "diff_g_f": ([], []),
                "error_free_annealings": error_free,
            }

    monkeypatch.setattr("tigramite.rpcmci.RPCMCI", FakeRPCMCI)
    return captured


def test_rpcmci_scores_shape_direction_and_regime_max(monkeypatch):
    captured = _install_fake_rpcmci(monkeypatch)
    x = np.random.default_rng(6).normal(size=(60, 4))
    scores = rpcmci_scores(
        x,
        max_lag=3,
        num_regimes=2,
        max_transitions=10,
        num_iterations=20,
        max_anneal=10,
        pc_alpha=0.05,
        alpha_level=0.01,
    )

    assert captured["num_regimes"] == 2
    assert captured["tau_min"] == 1
    assert captured["tau_max"] == 3
    assert captured["pc_alpha"] == 0.05
    assert captured["alpha_level"] == 0.01
    assert captured["num_iterations"] == 20
    assert captured["max_anneal"] == 10
    assert captured["max_transitions"] == 10
    assert scores.shape == (3, 4, 4)
    assert scores[0, 1, 2] == pytest.approx(0.7)  # max over regimes
    assert scores[0, 2, 1] == pytest.approx(0.1)  # direction preserved
    assert scores[2, 3, 0] == pytest.approx(0.9)  # edge only active in regime 1
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)
    assert np.all(np.isfinite(scores))
    assert scores.max() == pytest.approx(0.9)  # tau=0 value 5.0 dropped


def test_rpcmci_scores_p_value_transform(monkeypatch):
    _install_fake_rpcmci(monkeypatch)
    x = np.random.default_rng(7).normal(size=(60, 4))
    scores = rpcmci_scores(x, max_lag=3, score="p_value")

    assert scores[0, 1, 2] == pytest.approx(5.0)  # min over regimes: 1e-5
    assert scores[2, 3, 0] == pytest.approx(3.0)  # only tested in regime 1
    assert scores[0, 0, 1] == pytest.approx(0.0)  # untested pairs carry p=1
    assert np.all(scores >= 0.0)
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)


def test_rpcmci_scores_return_info_regime_passthrough(monkeypatch):
    _install_fake_rpcmci(monkeypatch)
    x = np.random.default_rng(8).normal(size=(60, 4))
    scores, info = rpcmci_scores(
        x,
        max_lag=3,
        num_iterations=9,
        max_anneal=4,
        max_transitions=6,
        return_info=True,
    )

    assert scores.shape == (3, 4, 4)
    assert info["effective_offset"] == 3  # == max_lag: early gamma unconstrained
    assert info["error_free_annealings"] == 7
    assert info["num_iterations"] == 9
    assert info["max_anneal"] == 4
    assert info["max_transitions"] == 6
    assert info["regime_probabilities"].shape == (60, 2)
    expected = np.concatenate([np.zeros(30, dtype=int), np.ones(30, dtype=int)])
    assert np.array_equal(info["regime_sequence"], expected)


def test_rpcmci_scores_raises_when_no_annealing_converges(monkeypatch):
    _install_fake_rpcmci(monkeypatch, error_free=0)
    x = np.random.default_rng(9).normal(size=(60, 4))
    with pytest.raises(RuntimeError, match="no annealing"):
        rpcmci_scores(x, max_lag=3)


def test_rpcmci_scores_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="num_regimes"):
        rpcmci_scores(np.zeros((60, 3)), max_lag=2, num_regimes=1)
    with pytest.raises(ValueError, match="timesteps"):
        rpcmci_scores(np.zeros((6, 3)), max_lag=2)
