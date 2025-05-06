from src.data.data_reg import SpatialReg

sr = SpatialReg()

df = sr.spatial_simulation(n=100, time=20, rho=0.8, simulations=1000)
print(sr.results)
