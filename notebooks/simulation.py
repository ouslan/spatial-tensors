from src.data.data_reg import SpatialReg

sr = SpatialReg(n=10)

df = sr.spatial_simulation(n=10, time=20, rho=0.7, simulations=100)

df.write_csv("sim_results.csv")
print(sr.results)
