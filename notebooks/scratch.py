# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction
#
# - A spatial model is definde in the following way:
# $$
# y_{it} = \beta X_{it} + \rho \sum_{j=1}^{N} w_{it}
# $$

# %%
import numpy as np
import pymc as pm
import arviz as az
import bambi as bmb
import statsmodels.api as sm

from src.data.data_reg import SpatialReg
az.style.use("arviz-darkgrid")


num = 10
rho = .8
sr = SpatialReg(n=num)

# %%
gdf = sr.shape
n_obs = len(gdf)
gdf.plot()

# Construct X matrix (with a constant term)
X = np.ones((n_obs, 2))
X[:, 1] = np.random.normal(size=n_obs)  # Random variable
print(X)

# Define beta coefficients
beta = np.array([1, 2])
print(beta)

# Compute xb
xb = X @ beta
xb = xb.reshape(-1, 1)
u = np.random.normal(size=n_obs).reshape(-1, 1)

wr = weights.contiguity.Queen.from_dataframe(gdf, use_index=False)
wr.transform = "r"

y_d = dgp_lag(u, xb, wr, rho=rho)

gdf["y_d"] = y_d

gdf["X1"] = X[:, 1]
gdf["w_d"] = weights.lag_spatial(wr, y_d)
gdf["time"] = t


# %%
gdf = sr.spatial_panel(n=num,time=50,rho=rho)
gdf

# %%
gdf[gdf["time"]==0].plot("X1")

# %%
gdf[gdf["time"]==1].plot("y_d")

# %%
xb = gdf[["X1", "w_d"]].values.reshape(-1,2)
y_d = gdf["y_d"].values.reshape(-1,1)
X = sm.add_constant(xb)
results = sm.OLS(y_d, X).fit()
print(results.summary())

# %%
df = gdf.drop("geometry", axis=1)
priors = {
    "w_d": bmb.Prior("Normal", mu=0, sigma=2),
}
model = bmb.Model(
    "y_d ~ 1 + X1 + w_d",
    priors=priors,
    data=df, 
    dropna=True
)
results = model.fit()

# %%
az.plot_trace(results)
az.summary(results)

# %%
gdf["centroid"] = gdf.centroid
gdf["lat"] = gdf["centroid"].x
gdf["lon"] = gdf["centroid"].y
df = gdf.drop("geometry", axis=1)
df

# %%
X = df[["X1","lat","lon"]].values.reshape(-1,3)
y = df["y_d"].values.reshape(-1,1)
X


# %%
# Sort and extract variables
gdf = gdf.sort_values(["time", "id"]).reset_index(drop=True)

# Encode spatial unit ids as integers 0..N-1
gdf["unit_id"] = gdf["id"].astype("category").cat.codes
N = gdf["unit_id"].nunique()
T = gdf["time"].nunique()

y = gdf["y_d"].values
X1 = gdf["X1"].values
Wy = gdf["w_d"].values
unit_idx = gdf["unit_id"].values


# %%
with pm.Model() as model:
    # Hyperpriors
    sigma = pm.HalfNormal("sigma", 2.0)
    tau_rho = pm.HalfNormal("tau_rho", 1.0)
    tau_mu = pm.HalfNormal("tau_mu", 1.0)

    # Priors
    beta = pm.Normal("beta", mu=0, sigma=5)
    rho_i = pm.Normal("rho", mu=0, sigma=tau_rho, shape=N)     # one rho per unit
    mu_i = pm.Normal("mu", mu=0, sigma=tau_mu, shape=N)         # one intercept per unit

    # Create shared inputs
    X_data = pm.Data("X1", X1)
    Wy_data = pm.Data("Wy", Wy)
    unit_idx_data = pm.Data("unit_idx", unit_idx)

    # Compute mu_y
    mu_y = rho_i[unit_idx_data] * Wy_data + beta * X_data + mu_i[unit_idx_data]

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu_y, sigma=sigma, observed=y)

    trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)


# %%
# az.plot_trace(trace, var_names=["rho", "beta", "sigma"])
# az.summary(trace, var_names=["rho", "beta", "sigma"])


# %%
rho_true = .8
summary = az.summary(trace, var_names=["rho"], hdi_prob=0.94)
within_hdi = (rho_true >= summary["hdi_3%"]) & (rho_true <= summary["hdi_97%"])

# Report results
all_contain = within_hdi.all()
num_pass = within_hdi.sum()
num_total = len(within_hdi)

print(f"True rho = {rho_true}")
print(f"{num_pass}/{num_total} HDIs contain true rho.")

# Optionally, list which units failed
if not all_contain:
    failed_units = np.where(~within_hdi)[0]
    print(f"Units failing HDI test: {failed_units}")
