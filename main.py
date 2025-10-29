from src.data.data_reg import SpatialReg
import json


sr = SpatialReg()


def main():
    master = sr.spatial_simulation(time=10, rho=0.7, simulations=100, start_seed=787)
    master.write_csv("results_raw.csv")
    file_path = "bayesian_results.json"

    # Save the dictionary as a JSON file
    with open(file_path, "w") as json_file:
        json.dump(sr.results, json_file, indent=4)


if __name__ == "__main__":
    main()
