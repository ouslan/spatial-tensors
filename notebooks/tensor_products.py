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

# %%
import os 
os.chdir("..")

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from itertools import product
from patsy import dmatrix
from src.data.data_reg import SpatialReg

sr = SpatialReg()


# %%
gdf = sr.spatial_panel(time=100,rho=0.7, seed=787)
gdf

# %%
formula = "X_1 + X_2 + X_3 + te(cr(lat, df=6), cr(lon, df=6), constraints='center')"
design = dmatrix(formula, gdf)

model = Ridge(alpha=1e-3)
model.fit(design, gdf["y_true"])

gdf['y_tensor'] = model.predict(design)


# %%

column_names = design.design_info.column_names

coef = model.coef_

coef_dict = dict(zip(column_names, coef))

X_vars = ['X_1', 'X_2', 'X_3']
X_coefs = {var: coef_dict[var] for var in X_vars}

intercept = model.intercept_

print("Intercept:", intercept)
print("Linear coefficients:")
for var, val in X_coefs.items():
    print(f"  {var}: {val:.4f}")


# %%
column_names = design.design_info.column_names

coef = model.coef_

coef_dict = dict(zip(column_names, coef))
coef_dict["X_1"]

# %%
fig, ax = plt.subplots(figsize=(10, 8))

gdf[gdf["time"] == 0].plot(column='y_tensor', cmap='viridis', legend=True,
         ax=ax, markersize=10, edgecolor='black', linewidth=0.2)

ax.set_title("Smoothed Predictions: te(lat, lon)")
ax.set_axis_off()
plt.tight_layout()
plt.show()

# %%
print(((gdf["y_true"] - gdf["y_tensor"])**2).sum()/ len(gdf))
plt.scatter(gdf["y_true"], gdf["y_tensor"]);

# %%
xb = gdf[["X_1","X_2","X_3","w_rook"]].values.reshape(-1,4)
y_d = gdf["y_true"].values.reshape(-1,1)
X = sm.add_constant(xb)
results = sm.OLS(y_d, X).fit()
print(results.summary())

# %%
gdf["ols_rook"] = results.predict(X)

# %%
print(((gdf["y_true"] - gdf["ols_rook"])**2).sum()/ len(gdf))
plt.scatter(gdf["y_true"], gdf["ols_rook"]);

# %%
xb = gdf[["X_1","X_2","X_3","w_queen"]].values.reshape(-1,4)
y_d = gdf["y_true"].values.reshape(-1,1)
X = sm.add_constant(xb)
results = sm.OLS(y_d, X).fit()
print(results.summary())

gdf["ols_queen"] = results.predict(X)

# %%
print(((gdf["y_true"] - gdf["ols_queen"])**2).sum()/ len(gdf))
plt.scatter(gdf["y_true"], gdf["ols_queen"]);

# %%
xb = gdf[["X_1","X_2","X_3","w_knn6"]].values.reshape(-1,4)
y_d = gdf["y_true"].values.reshape(-1,1)
X = sm.add_constant(xb)
results = sm.OLS(y_d, X).fit()
print(results.summary())

gdf["ols_knn6"] = results.predict(X)

# %%
print(((gdf["y_true"] - gdf["ols_knn6"])**2).sum()/ len(gdf))
plt.scatter(gdf["y_true"], gdf["ols_knn6"]);
