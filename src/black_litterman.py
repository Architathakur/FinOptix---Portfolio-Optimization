"""
Black-Litterman model.

Key difference from the original notebook: views (P, Q) are no longer three
arbitrary hand-picked pairs disconnected from the ML step. Instead every
selected stock gets an absolute view equal to its XGBoost-predicted mean
return, with view uncertainty (Omega) set by the standard He-Litterman
proportional rule and scaled by a configurable confidence level. This
directly wires the ML model into the optimizer, closing the gap in the
original implementation.
"""

import numpy as np
import pandas as pd


def regularize_covariance(cov_matrix: pd.DataFrame, ridge: float = 1e-6) -> pd.DataFrame:
    """Add a small diagonal ridge to improve conditioning before matrix inversion."""
    cov_values = cov_matrix.to_numpy(dtype=float, copy=True)
    diag_scale = float(np.nanmean(np.diag(cov_values))) if len(cov_values) else 1.0
    if not np.isfinite(diag_scale) or diag_scale <= 0:
        diag_scale = 1.0
    cov_values[np.diag_indices_from(cov_values)] += ridge * diag_scale
    return pd.DataFrame(cov_values, index=cov_matrix.index, columns=cov_matrix.columns)


def implied_equilibrium_returns(cov_matrix: pd.DataFrame, market_weights: np.ndarray,
                                 risk_aversion: float) -> pd.Series:
    """pi = delta * Sigma * w_mkt"""
    pi = risk_aversion * cov_matrix.values @ market_weights
    return pd.Series(pi, index=cov_matrix.columns)


def build_ml_views(expected_returns: pd.DataFrame, assets: list):
    """
    Build absolute views: view_i = mean ML-predicted return for asset i.

    Returns
    -------
    P : identity matrix (n_assets x n_assets) - one view per asset
    Q : array of mean predicted returns, in `assets` order
    """
    n = len(assets)
    P = np.eye(n)
    Q = np.array([expected_returns[a].mean() for a in assets])
    return P, Q


def omega_from_confidence(P: np.ndarray, cov_matrix: pd.DataFrame, tau: float,
                           confidence: float) -> np.ndarray:
    """
    He-Litterman proportional uncertainty: Omega = tau * diag(P Sigma P').
    `confidence` in (0, 1] scales Omega down (higher confidence -> tighter
    views -> more weight on the ML predictions vs. the market prior).
    A confidence near 0 makes Omega huge, which pulls the posterior back
    toward the market-implied prior (i.e. "ignore the views").
    """
    confidence = max(confidence, 1e-6)  # avoid division by zero
    diag_vals = np.diag(P @ cov_matrix.values @ P.T) * tau
    return np.diag(diag_vals / confidence)


def black_litterman_posterior(cov_matrix: pd.DataFrame, pi: pd.Series,
                               P: np.ndarray, Q: np.ndarray, Omega: np.ndarray,
                               tau: float = 0.025, ridge: float = 1e-6):
    """
    Standard Black-Litterman posterior:
        mu_bl  = M [ (tau*Sigma)^-1 pi + P' Omega^-1 Q ]
        Sigma_bl = M
        M = [ (tau*Sigma)^-1 + P' Omega^-1 P ]^-1
    """
    cov_matrix = regularize_covariance(cov_matrix, ridge=ridge)
    Omega = Omega.copy()
    omega_diag = np.diag_indices_from(Omega)
    Omega[omega_diag] = np.where(Omega[omega_diag] <= 0, ridge, Omega[omega_diag])

    tau_cov = tau * cov_matrix.values
    inv_tau_cov = np.linalg.pinv(tau_cov)
    inv_omega = np.linalg.pinv(Omega)

    M = np.linalg.pinv(inv_tau_cov + P.T @ inv_omega @ P)
    mu_bl = M @ (inv_tau_cov @ pi.values + P.T @ inv_omega @ Q)

    mu_bl = pd.Series(mu_bl, index=cov_matrix.columns)
    cov_bl = pd.DataFrame(M, index=cov_matrix.columns, columns=cov_matrix.columns)
    return mu_bl, cov_bl
