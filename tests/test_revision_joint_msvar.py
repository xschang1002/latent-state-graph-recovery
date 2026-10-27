"""Small mathematical checks; no benchmark-series fitting or true-label tuning."""
import importlib.util
import itertools
from pathlib import Path
import sys
import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "_joint_msvar.py"
spec = importlib.util.spec_from_file_location("joint_msvar_under_test", MODULE)
joint = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = joint
spec.loader.exec_module(joint)


def test_forward_backward_matches_all_paths():
    transition = np.array([[.81, .19], [.27, .73]])
    initial = np.array([.4, .6])
    emissions = np.array([[.3, .8], [.9, .1], [.4, .7], [.8, .2]])
    expected_gamma = np.zeros_like(emissions)
    expected_xi = np.zeros((3, 2, 2))
    total = 0.
    for path in itertools.product(range(2), repeat=4):
        probability = initial[path[0]] * emissions[0, path[0]]
        for t in range(1, 4):
            probability *= transition[path[t - 1], path[t]] * emissions[t, path[t]]
        total += probability
        for t in range(4):
            expected_gamma[t, path[t]] += probability
        for t in range(3):
            expected_xi[t, path[t], path[t + 1]] += probability
    loglik, gamma, xi = joint.forward_backward(np.log(emissions), transition, initial)
    np.testing.assert_allclose(loglik, np.log(total))
    np.testing.assert_allclose(gamma, expected_gamma / total)
    np.testing.assert_allclose(xi, expected_xi / total)


def test_direct_var_mean_has_exact_lags_and_score_orientation():
    x = np.zeros((10, 2))
    beta = np.zeros((2, 5, 2))
    beta[0, 2, 0] = 2.  # source1 -> target0 at lag1
    beta[0, 4, 0] = 3.  # source1 -> target0 at lag2
    beta[1, 3, 1] = -4.  # source0 -> target1 at lag2
    beta[:, 1, 0] = 99.  # self edge fitted but excluded from recovery score
    influence = []
    for lag in range(1, 5):
        changed = x.copy()
        changed[6 - lag, 1] = 1.
        design, _ = joint.lag_design(changed, 2)
        influence.append((design @ beta[0])[6 - 2, 0])
    np.testing.assert_allclose(influence, [2., 3., 0., 0.])
    scores = joint.scores_from_coefficients(beta, 2)
    assert scores[0, 1, 0] == 2.
    assert scores[1, 1, 0] == 3.
    assert scores[1, 0, 1] == 4.
    assert scores[:, 0, 0].sum() == 0.


def test_joint_emission_uses_all_targets_and_full_covariance():
    design = np.ones((1, 1))
    response = np.array([[1., -1.]])
    beta = np.array([[[1., -1.]], [[-1., 1.]]])
    covariance = np.array([[[1., .3], [.3, 1.]], [[1., .3], [.3, 1.]]])
    emissions = joint.emission_log_probabilities(design, response, beta, covariance)
    assert emissions.shape == (1, 2)
    assert emissions[0, 0] > emissions[0, 1]
    _, gamma, xi = joint.forward_backward(emissions, np.array([[.8, .2], [.2, .8]]),
                                         np.array([.5, .5]))
    assert gamma.shape == (1, 2)  # no target-specific posterior axis
    assert xi.shape == (0, 2, 2)


def test_seed_reproducibility_and_observed_likelihood_progress():
    data = np.random.default_rng(391).normal(size=(95, 2))
    config = joint.JointMSVARConfig(n_starts=2, max_iter=12)
    scores, info = joint.joint_msvar_scores(data, 1, seed=123, config=config)
    scores2, info2 = joint.joint_msvar_scores(data, 1, seed=123, config=config)
    np.testing.assert_array_equal(scores, scores2)
    np.testing.assert_array_equal(info['gamma'], info2['gamma'])
    assert info['starts'] == info2['starts']
    for start in info['starts']:
        if start['status'] != 'failed':
            history = np.asarray(start['loglik_history'])
            assert np.all(np.diff(history) >= -1e-8 * (1 + np.abs(history[:-1])))
    z = (data - data.mean(axis=0)) / data.std(axis=0)
    design, response = joint.lag_design(z, 1)
    loglik, gamma, _ = joint.forward_backward(joint.emission_log_probabilities(
        design, response, info['coefficients'], info['covariance']),
        info['transition'], info['initial'])
    np.testing.assert_allclose(loglik, info['loglik'])
    np.testing.assert_allclose(gamma, info['gamma'])


def test_probability_floor_is_constrained_mle_not_clip_normalize():
    estimate = joint.probability_mle(np.array([0., 1., 9.]), .1)
    np.testing.assert_allclose(estimate, [.1, .1, .8])
    assert estimate.min() >= .1
