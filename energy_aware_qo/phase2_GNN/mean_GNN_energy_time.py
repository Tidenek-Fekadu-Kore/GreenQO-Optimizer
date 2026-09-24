import csv
import numpy as np

# 1. Your collected run data
time_mae_runs = [0.9621803363926001, 1.1916602863832308, 1.3364500118198,
                 2.856335611946853, 0.7598042835451015]
time_mape_runs = [6.7661069018388655, 8.387506284553572, 10.683522204536457,
                  21.670816064255995, 6.759375886345243]

energy_mae_runs = [17.443527896582967, 36.70967890531016, 26.74158800535847,
                   22.073482174729538, 12.932994853759414]
energy_mape_runs = [8.531928132862378, 18.487967028148972, 18.603595590554587,
                    12.223228590395925, 9.04926701265897]

# 2. Dictionary to hold the datasets for batch processing and CSV export
results_data = {
    "MLP Execution Time": (time_mae_runs, time_mape_runs),
    "MLP Energy Consumption": (energy_mae_runs, energy_mape_runs)
}

# 3. Function to print and save summary results to a CSV file
def export_summary_to_csv(filename, data_dict):
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header matching your table structure
        writer.writerow(["Model / Metric", "Mean", "Standard Deviation (±)"])
        
        for model_name, (mae_list, mape_list) in data_dict.items():
            mae_mean = np.mean(mae_list)
            mae_std = np.std(mae_list, ddof=1)
            mape_mean = np.mean(mape_list)
            mape_std = np.std(mape_list, ddof=1)
            
            # Print to console (matching your original format)
            print(f"=== {model_name} ===")
            print(f"MAE:  {mae_mean:.2f} ± {mae_std:.2f}")
            print(f"MAPE: {mape_mean:.2f}% ± {mape_std:.2f}%\n")
            
            # Write rows to CSV
            writer.writerow([f"{model_name} - MAE", f"{mae_mean:.2f}", f"{mae_std:.2f}"])
            writer.writerow([f"{model_name} - MAPE (%)", f"{mape_mean:.2f}%", f"{mape_std:.2f}%"])

# Execute export
export_summary_to_csv("performance_GreenQO.csv", results_data)
print("Results saved to performance_GreenQO.csv")