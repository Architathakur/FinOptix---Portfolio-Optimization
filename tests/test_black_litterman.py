import numpy as np
import pandas as pd
import pytest

from src.black_litterman import (
    implied_equilibrium_returns,
    build_ml_views,
    omega_from_confidence,
    black_litterman_posterior,
)

ASSETS = ["A", "B", "C", "D"]


@pytest.fixture
def synthetic_cov():
    np.random.seed(0)
    rand = np.random.randn(4, 4) * 0.01
    cov = rand @ rand.T + np.eye(4) * 1e-4  # guarantee positive semi-definite
    return pd.DataFrame(cov, index=ASSETS, columns=ASSETS)


def test_implied_returns_shape_and_sign(synthetic_cov):
    w_mkt = np.ones(4) / 4
    pi = implied_equilibrium_returns(synthetic_cov, w_mkt, risk_aversion=2.5)
    assert list(pi.index) == ASSETS
    assert len(pi) == 4
    # equilibrium returns should be finite and non-degenerate
    assert np.all(np.isfinite(pi.values))


def test_build_ml_views_identity_and_means():
    expected_returns = pd.DataFrame(
        {"A": [0.01, 0.02, 0.03], "B": [-0.01, 0.0, 0.01], "C": [0.0, 0.0, 0.0], "D": [0.02, 0.02, 0.02]}
    )
    P, Q = build_ml_views(expected_returns, ASSETS)
    assert P.shape == (4, 4)
    assert np.allclose(P, np.eye(4))
    assert np.allclose(Q, [0.02, 0.0, 0.0, 0.02])


def test_omega_scales_with_confidence(synthetic_cov):
    P = np.eye(4)
    omega_low_conf = omega_from_confidence(P, synthetic_cov, tau=0.025, confidence=0.1)
    omega_high_conf = omega_from_confidence(P, synthetic_cov, tau=0.025, confidence=1.0)
    # lower confidence -> larger uncertainty (bigger Omega diagonal)
    assert np.all(np.diag(omega_low_conf) > np.diag(omega_high_conf))


def test_posterior_matches_prior_when_omega_huge(synthetic_cov):
    """If view uncertainty -> infinity, the posterior should collapse to the prior pi."""
    w_mkt = np.ones(4) / 4
    pi = implied_equilibrium_returns(synthetic_cov, w_mkt, risk_aversion=2.5)
    P = np.eye(4)
    Q = np.array([10.0, 10.0, 10.0, 10.0])  # deliberately extreme, should be ignored
    Omega = np.diag([1e6, 1e6, 1e6, 1e6])   # near-infinite uncertainty

    mu_bl, _ = black_litterman_posterior(synthetic_cov, pi, P, Q, Omega, tau=0.025)
    assert np.allclose(mu_bl.values, pi.values, atol=1e-3)


def test_posterior_pulls_toward_views_when_confident(synthetic_cov):
    """With very tight Omega, posterior should move sharply toward Q."""
    w_mkt = np.ones(4) / 4
    pi = implied_equilibrium_returns(synthetic_cov, w_mkt, risk_aversion=2.5)
    P = np.eye(4)
    Q = np.array([0.05, 0.05, 0.05, 0.05])
    Omega = np.diag([1e-8, 1e-8, 1e-8, 1e-8])  # near-zero uncertainty = full confidence

    mu_bl, _ = black_litterman_posterior(synthetic_cov, pi, P, Q, Omega, tau=0.025)
    assert np.allclose(mu_bl.values, Q, atol=1e-3)
