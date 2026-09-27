"""
shap_sim — shared utilities for the SHAP simulation study.

Every notebook in this repository imports from here, so the data-generating
processes, the exact-SHAP benchmark and the replication loop are defined once.

Contents
--------
make_sigma                          AR(1)-style covariance, Sigma_jk = rho^|j-k|
simulate_design                     draw a fixed design matrix X
f1, f2, f3, true_gam_function       additive non-linear regression function
true_tree_function                  piecewise-constant regression function
marginal_shap_linear                SHAP for a linear model with independent X
coalition_value_conditional_linear  E[f(X) | X_S = x_S] - E[f(X)] for a linear f
true_conditional_shap_linear        exact conditional SHAP, Gaussian X, linear f
aggregate_dependence_curve          bin x and summarise SHAP within bins
run_replications                    fit a model over replications, average SHAP
"""

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
import shap

# --------------------------------------------------------------------------
# Covariance and design
# --------------------------------------------------------------------------

def make_sigma(p, rho):
    """AR(1)-style covariance matrix with entries rho^|j-k|."""
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def simulate_design(n, p, rho, seed=123):
    """Draw the design matrix once; it is held fixed across replications."""
    mu = np.zeros(p)
    Sigma = make_sigma(p, rho)
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=n)
    names = [f"X{j+1}" for j in range(p)]
    return X, pd.DataFrame(X, columns=names), mu, Sigma, names


# --------------------------------------------------------------------------
# True regression functions
# --------------------------------------------------------------------------

def f1(x):
    return 2.0 * np.log(np.abs(x))


def f2(x):
    return 1.5 * np.sqrt(np.abs(x))


def f3(x):
    return -2.0 * np.log(np.abs(x))


def true_gam_function(X, beta0=1.5):
    """Additive non-linear function of X1, X2, X3. X4, X5 are inactive."""
    return beta0 + f1(X[:, 0]) + f2(X[:, 1]) + f3(X[:, 2])


def true_tree_function(X):
    """
    Piecewise-constant function of X1, X2, X3. X4, X5 are inactive.

        X1 <= 0 and X2 <= 0    ->  4.0
        X1 <= 0 and X2 >  0    ->  1.0
        X1 >  0 and X3 <= 0.5  -> -2.0
        X1 >  0 and X3 >  0.5  ->  3.0
    """
    f = np.zeros(X.shape[0])
    left, right = X[:, 0] <= 0, X[:, 0] > 0
    f[left & (X[:, 1] <= 0)] = 4.0
    f[left & (X[:, 1] > 0)] = 1.0
    f[right & (X[:, 2] <= 0.5)] = -2.0
    f[right & (X[:, 2] > 0.5)] = 3.0
    return f


# --------------------------------------------------------------------------
# Exact SHAP benchmarks for a linear model
# --------------------------------------------------------------------------

def marginal_shap_linear(X, beta, mu=None):
    """
    SHAP values for f(x) = beta0 + beta'x under INDEPENDENT covariates:
    phi_j(x) = beta_j (x_j - mu_j). Only valid when rho = 0.
    """
    mu = np.zeros(X.shape[1]) if mu is None else mu
    return (X - mu.reshape(1, -1)) * beta.reshape(1, -1)


def coalition_value_conditional_linear(x, beta, Sigma, S, mu=None):
    """
    Conditional coalition value for a linear model:
        v_x(S) = E[f(X) | X_S = x_S] - E[f(X)]
    The intercept cancels after centring.
    """
    p = len(beta)
    mu = np.zeros(p) if mu is None else mu

    S = list(S)
    C = [j for j in range(p) if j not in S]
    dx = x - mu

    if len(S) == 0:
        return 0.0
    if len(S) == p:
        return beta @ dx

    Sigma_SS = Sigma[np.ix_(S, S)]
    Sigma_CS = Sigma[np.ix_(C, S)]
    cond_mean_C = mu[C] + Sigma_CS @ np.linalg.solve(Sigma_SS, dx[S])

    return beta[S] @ dx[S] + beta[C] @ (cond_mean_C - mu[C])


def true_conditional_shap_linear(x, beta, Sigma, mu=None):
    """
    Exact conditional SHAP values for a linear model with Gaussian covariates,
    by direct enumeration of all 2^(p-1) coalitions per feature.

    Exact but exponential in p: practical for p ~ 5, not for p = 500.
    """
    p = len(beta)
    mu = np.zeros(p) if mu is None else mu
    phi = np.zeros(p)

    for j in range(p):
        others = [k for k in range(p) if k != j]
        for s in range(len(others) + 1):
            for S in combinations(others, s):
                S = set(S)
                w = factorial(len(S)) * factorial(p - len(S) - 1) / factorial(p)
                phi[j] += w * (
                    coalition_value_conditional_linear(x, beta, Sigma, S | {j}, mu)
                    - coalition_value_conditional_linear(x, beta, Sigma, S, mu)
                )
    return phi


def true_conditional_shap_matrix(X, beta, Sigma, mu=None):
    """Exact conditional SHAP for every row of X."""
    return np.vstack([
        true_conditional_shap_linear(row, beta, Sigma, mu) for row in X
    ])


# --------------------------------------------------------------------------
# Diagnostics and the replication loop
# --------------------------------------------------------------------------

def aggregate_dependence_curve(x_vals, shap_vals, nbins=30, min_count=5):
    """Bin x and return (centres, mean SHAP, sd SHAP, counts) per bin."""
    bins = np.linspace(np.min(x_vals), np.max(x_vals), nbins + 1)
    bin_ids = np.digitize(x_vals, bins) - 1

    centers, means, sds, counts = [], [], [], []
    for b in range(nbins):
        mask = bin_ids == b
        if np.sum(mask) > min_count:
            centers.append(np.mean(x_vals[mask]))
            means.append(np.mean(shap_vals[mask]))
            sds.append(np.std(shap_vals[mask], ddof=1))
            counts.append(np.sum(mask))

    return (np.array(centers), np.array(means),
            np.array(sds), np.array(counts))


def run_replications(X_df, f_true, model_factory, n_reps=100,
                     sigma_eps=1.0, seed_base=1000, predict_fn=False):
    """
    Refit a model over `n_reps` noise replications on a FIXED design and
    average the SHAP values.

    Parameters
    ----------
    X_df          fixed design (n x p)
    f_true        true signal, length n (noise is added per replication)
    model_factory callable returning a fresh unfitted estimator
    predict_fn    True to explain `model.predict` (needed for models the
                  shap package cannot introspect, e.g. the EBM booster)

    Returns
    -------
    avg_shap, sd_shap, avg_base
    """
    n, p = X_df.shape
    shap_store = np.zeros((n_reps, n, p))
    base_store = np.zeros((n_reps, n))

    for rep in range(n_reps):
        rng_rep = np.random.default_rng(seed_base + rep)
        y = f_true + rng_rep.normal(0.0, sigma_eps, size=n)

        model = model_factory()
        model.fit(X_df, y)

        target = model.predict if predict_fn else model
        explainer = shap.Explainer(target, X_df)
        sv = explainer(X_df)

        shap_store[rep] = sv.values
        base_vals = np.atleast_1d(sv.base_values)
        base_store[rep] = base_vals[0] if len(base_vals) == 1 else base_vals

    return (shap_store.mean(axis=0),
            shap_store.std(axis=0, ddof=1),
            base_store.mean(axis=0))
