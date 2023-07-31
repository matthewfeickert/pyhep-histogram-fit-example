# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: all,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.11.3
# ---

# %% [markdown]
# # Fit of Nomral model to histogram

# %%
from numbers import Number
from typing import Protocol

import hist
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

# %matplotlib widget


# %%
class Model(Protocol):
    def cdf(self, params: list[Number], x: Number) -> Number:
        ...

    def init(self) -> list[Number]:
        ...


def fit(model: Model, data: hist.Hist, extended: bool = False):
    assert len(data.axes) == 1
    axis = data.axes[0]
    # assert data.axes[0] is numeric

    if extended:
        raise NotImplementedError("Extended fit")

    def expectation(params: list[Number]) -> list[Number]:
        return data.sum() * np.diff([model.cdf(params, x) for x in axis.edges])

    def nll(params: list[Number]) -> Number:
        return -sum(
            stats.poisson.logpmf(n, rate)
            for n, rate in zip(data.values(), expectation(params))
        )

    result = minimize(nll, model.init())
    return result.x, expectation(result.x)


# %%
class Norm(Model):
    def cdf(self, params: list[Number], x: Number) -> Number:
        return stats.norm.cdf(x, loc=params[0], scale=params[1])

    def init(self) -> list[Number]:
        return [0, 1]


# %%
observed_hist = hist.Hist.new.Reg(40, -3, 3).Double().fill(stats.norm.rvs(size=1000))

params, expectation = fit(Norm(), observed_hist)

# %%
fig, ax = plt.subplots()

observed_hist.plot1d(ax=ax, histtype="errorbar", color="black", label="Observations")
ax.stairs(
    expectation,
    edges=observed_hist.axes[0].edges,
    label=f"Best fit of Normal model\n $\mu=${params[0]:.3f}, $\sigma=${params[1]:.3f}",
)

ax.legend()
ax.set_xlabel("Observable", fontsize=14)
ax.set_ylabel("Count", fontsize=14)

fig.savefig("normal_fit_to_hist.png");
