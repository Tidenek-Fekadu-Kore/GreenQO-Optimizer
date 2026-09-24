import csv
import numpy as np

# 1. MLP Baseline collected run data (5 runs)
time_mae_runs = [5.279743702809016, 5.850385673551849, 5.867152279507031,
                 5.851717340873949, 5.780497373523135]
time_mape_runs = [423.50927681391124, 449.94246843673267, 449.75265967540577,
                  458.4506784744496, 443.3636957147245]

energy_mae_runs = [76.2553047353571, 77.05694432114109, 76.32631159695711,
                   76.49785044121019, 76.43531165845465]
energy_mape_runs = [373.3435582234106, 382.23077613188485, 378.6591820221642,
                    379.0288671289889, 376.61078143389824]

# 2. Dictionary to hold the datasets for batch processing and CSV export
results_data = {
    "MLP Baseline Execution Time": (time_mae_runs, time_mape_runs),
    "MLP Baseline Energy Consumption": (energy_mae_runs, energy_mape_runs)
}

# 3. Function to print and save summary results to a CSV file
def export_summary_to_csv(filename, data_dict):
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write CSV header
        writer.writerow(["Model / Metric", "Mean", "Standard Deviation (±)"])
        
        for model_name, (mae_list, mape_list) in data_dict.items():
            mae_mean = np.mean(mae_list)
            mae_std = np.std(mae_list, ddof=1)
            mape_mean = np.mean(mape_list)
            mape_std = np.std(mape_list, ddof=1)
            
            # Print to console
            print(f"=== {model_name} ===")
            print(f"MAE:  {mae_mean:.2f} ± {mae_std:.2f}")
            print(f"MAPE: {mape_mean:.2f}% ± {mape_std:.2f}%\n")
            
            # Write rows to CSV
            writer.writerow([f"{model_name} - MAE", f"{mae_mean:.2f}", f"{mae_std:.2f}"])
            writer.writerow([f"{model_name} - MAPE (%)", f"{mape_mean:.2f}%", f"{mape_std:.2f}%"])

# Execute export
export_summary_to_csv("perfornace_MLP_baseline.csv", results_data)
print("Results saved to perfornace_MLP_baseline.csv")