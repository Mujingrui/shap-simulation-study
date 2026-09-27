# What Do SHAP Values Actually Explain?

A simulation study of SHAP feature attributions across model classes, sparsity levels and
covariate correlation, with an exact analytical benchmark. 

**The question.** SHAP is routinely read as "which variables drive the outcome." But SHAP
values are computed from a *fitted model*, not from the data-generating process. When the
fitted model is wrong, what do the attributions describe — the world, or the model's view of
it? And how close are the `shap` package's estimates to the SHAP values the definition
actually implies?

## Design

Every scenario uses the same skeleton: a design matrix `X ~ N(0, Σ)` with `Σ_jk = ρ^|j−k|`
drawn **once** and held fixed (seed 123), `n = 500`, and **100 replications** in which only
the noise `ε ~ N(0, 1)` is redrawn (seed 1000 + rep). A model is refit on each replication and
the SHAP values are averaged, so the reported attributions reflect the estimator rather than
one lucky sample.

| Notebook | True mechanism | Fitted model | p | ρ |
|---|---|---|---:|---|
| `sim_linear.ipynb` | linear, β = (0.5, 2, −1, 1.5, −1) | OLS | 5 | 0, 0.1, 0.5 |
| `sim_linear_sparse.ipynb` | linear, β = (0, 2, −1, 1.5, 0) | OLS | 5 | 0, 0.5 |
| `sim_linear_sparse_high.ipynb` | linear, 5 active of 500 | OLS | 500 | 0, 0.5 |
| `sim_linear_sparse_high_lasso.ipynb` | linear, 5 active of 500 | Lasso (α = 0.05) | 500 | 0 |
| `sim_GAM.ipynb` | additive non-linear in X1–X3 | OLS, then explainable boosting | 5 | 0 |
| `sim_tree.ipynb` | piecewise constant in X1–X3 | OLS, then depth-2 regression tree | 5 | 0 |

The GAM and tree notebooks each fit **two** models to the same data — one misspecified (linear)
and one able to represent the true mechanism. That contrast is the core of the study.

The non-linear mechanism is `1.5 + 2log|X₁| + 1.5√|X₂| − 2log|X₃|`; the tree mechanism is a
depth-2 partition on X₁, X₂, X₃ with leaf values 4, 1, −2, 3. In both, **X₄ and X₅ are inactive
by construction** — any attribution they receive is spurious.

## The exact benchmark

For a linear model with Gaussian covariates, the conditional SHAP values can be computed
analytically. `src/shap_sim.py` implements this from the definition: for each feature, it
enumerates all 2^(p−1) coalitions, evaluates the conditional coalition value

    v_x(S) = E[f(X) | X_S = x_S] − E[f(X)]

using the Gaussian conditional mean, and combines them with the Shapley weights. Estimated
SHAP values are plotted against this ground truth in the dependence plots, so the package's
output is validated rather than assumed correct.

When the covariates are independent this reduces to `φ_j(x) = β_j(x_j − μ_j)`; once ρ > 0 it
does not, and the full conditional computation is used.

## Findings

**SHAP explains the fitted model, not the mechanism.** In the non-linear scenarios, a linear
fit produces attributions that misrepresent the truth in both directions: features that are
genuinely inactive receive non-negligible attribution, and an active feature whose effect the
linear model cannot represent is suppressed to near zero. Refitting with a model flexible
enough to capture the mechanism — explainable boosting for the additive case, a depth-2 tree
for the piecewise-constant case — restores attribution to the truly active features and pushes
the inactive ones back toward zero. The same data, two models, two different stories about
"what matters."

**Sparse recovery in high dimensions works.** With 500 candidate predictors and only five
active, the averaged attributions concentrate on the five active features; the remaining 495
sit tightly around zero, individually and in aggregate.

**Attribution patterns under correlation are not the independent-case formula.** With ρ > 0 the
product form `β_j(x_j − μ_j)` is no longer the SHAP value, which is why the exact conditional
computation exists in this repository.

The write-up also reviews LIME, KernelSHAP and DeepSHAP, summarises the de-biased U-statistic
inference of Whitehouse, Sawarni & Syrgkanis (2026) for global SHAP summaries, and proposes a
bootstrap scheme for pointwise uncertainty in the SHAP curve.

Full write-up: [`docs/course-project-shap.pdf`](docs/course-project-shap.pdf) ·
slides: [`docs/slides.pdf`](docs/slides.pdf)

## Layout

```
src/shap_sim.py   shared DGPs, exact-SHAP benchmark, replication loop
notebooks/        one notebook per scenario
figures/          beeswarm and dependence plots, by scenario
docs/             report and slides
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Notebooks import the shared code with:

```python
import sys; sys.path.append("../src")
from shap_sim import *
```

All seeds are fixed, so figures reproduce run to run.

## Reference

Whitehouse, J., Sawarni, A. & Syrgkanis, V. (2026). Statistical Inference and Learning for
Shapley Additive Explanations (SHAP). arXiv:2602.10532.
