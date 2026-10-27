"""Random MSVAR retry starts are explicit and independent of caller RNG state."""

from types import SimpleNamespace

import numpy as np
import pytest

from stcg.baselines import msvar_scores


@pytest.mark.parametrize("retry_fails", [False, True])
@pytest.mark.parametrize("rng_api", ["legacy", "local"])
def test_msvar_retry_seed_is_reproducible_and_preserves_caller_rng(monkeypatch, retry_fails, rng_api):
    draws = []

    class RandomRetryModel:
        def __init__(self, endog, **kwargs):
            self.param_names = ["x1[0]", "x1[1]"]
            self.length = len(endog) - kwargs["order"]

        def fit(self, **kwargs):
            return self._fit(np.random, kwargs)

        def _fit(self, rng, kwargs):
            if "search_reps" not in kwargs:
                return SimpleNamespace(mle_retvals={"converged": False})
            value = rng.uniform(size=2)
            draws.append(value)
            if retry_fails:
                raise RuntimeError("Retry failed after drawing starts")
            return SimpleNamespace(
                model=self,
                params=value,
                mle_retvals={"converged": True},
                smoothed_marginal_probabilities=np.tile([0.7, 0.3], (self.length, 1)),
            )

    class LocalRandomRetryModel(RandomRetryModel):
        def fit(self, rng=None, **kwargs):
            return self._fit(np.random.default_rng(rng), kwargs)

    monkeypatch.setattr(
        "statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression",
        LocalRandomRetryModel if rng_api == "local" else RandomRetryModel,
    )
    x = np.random.default_rng(8).normal(size=(40, 2))
    caller_state = np.random.get_state()
    try:
        np.random.seed(111)
        expected_next = np.random.random(4)
        np.random.seed(111)
        first = msvar_scores(x, 1, seed=42)
        np.testing.assert_array_equal(np.random.random(4), expected_next)
        first_draws = np.array(draws)
        draws.clear()
        np.random.seed(999)
        second = msvar_scores(x, 1, seed=42)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_draws, np.array(draws))
        assert not np.array_equal(first_draws[0], first_draws[1])
        draws.clear()
        msvar_scores(x, 1, seed=43)
        assert not np.array_equal(first_draws, np.array(draws))
    finally:
        np.random.set_state(caller_state)


def test_installed_statsmodels_low_budget_retry_is_reproducible():
    """Exercise actual statsmodels, whose retry RNG API varies by version."""
    x = np.random.default_rng(5).normal(size=(80, 2))
    caller_state = np.random.get_state()
    kwargs = dict(max_lag=1, maxiter=1, retry_maxiter=5, search_reps=1,
                  seed=29, return_info=True)
    try:
        np.random.seed(123)
        first, info_first = msvar_scores(x, **kwargs)
        np.random.seed(987)
        second, info_second = msvar_scores(x, **kwargs)
        assert info_first["failed_targets"] == []
        assert info_second["failed_targets"] == []
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(info_first["regime_probabilities"],
                                      info_second["regime_probabilities"])
    finally:
        np.random.set_state(caller_state)
