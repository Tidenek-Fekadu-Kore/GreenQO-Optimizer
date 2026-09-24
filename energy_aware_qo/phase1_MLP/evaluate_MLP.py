#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
from model_MLP import MLP

# Step1. Load Data and Features
df = pd.read_csv("../datasets/tpch_multi_scale_final_dataset.csv")

features = [
    "jit_enabled", "total_startup_cost", "total_cost", "max_plan_rows", 
    "total_plan_width", "Seq Scan", "Hash Join", "Nested Loop", 
    "Aggregate", "Sort", "Other_Operators"
]

# Extract feature tensors
X_raw = torch.tensor(df[features].values, dtype=torch.float32)

# Calculate scale constants exactly as done during training
X_mean = X_raw.mean(dim=0, keepdim=True)
X_std = X_raw.std(dim=0, keepdim=True)
X_std[X_std == 0] = 1.0 
X_scaled = (X_raw - X_mean) / X_std

# Extract target metrics
actual_time = df["Execution_Time"].values
actual_energy = df["total_energy_consumption_memory_plus_cpu"].values

# Step 2. Load Models
model_time = MLP(len(features))
model_energy = MLP(len(features))

model_time.load_state_dict(torch.load("mlp_time_model.pth"))
model_energy.load_state_dict(torch.load("mlp_energy_model.pth"))

model_time.eval()
model_energy.eval()

# Step3. Compute Predictions
with torch.no_grad():
    pred_time = model_time(X_scaled).numpy().flatten()
    pred_energy = model_energy(X_scaled).numpy().flatten()

# Save individual prediction .csv 
output_df = pd.DataFrame({
    "Query_Identifier": df["query_id"] if "query_id" in df.columns else df.index + 1,
    "Dataset_Scale_Gigabytes": df["dataset_scale_gb"] if "dataset_scale_gb" in df.columns else 1.0,
    "Actual_Execution_Time_Seconds": actual_time,
    "Predicted_Execution_Time_Seconds": pred_time,
    "Absolute_Execution_Time_Error": np.abs(actual_time - pred_time),
    "Actual_Energy_Consumption_Joules": actual_energy,
    "Predicted_Energy_Consumption_Joules": pred_energy,
    "Absolute_Energy_Consumption_Error": np.abs(actual_energy - pred_energy)
})
output_df.to_csv("evaluation_predictions.csv", index=False)


# Step4. Define Evaluation Metrics 
summary_rows = []

def calculate_metrics(actual, predicted, label):
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1e-5, actual))) * 100
    
    print(f"=== {label} Evaluation Metrics ===")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("-" * 40)
    
    # Store figures for summary sheet using long-hand descriptions
    summary_rows.append({
        "Model_Target_Metric": label,
        "Mean_Absolute_Error": mae,
        "Mean_Absolute_Percentage_Error": mape
    })

# Print final summaries
print("\n" + "="*15 + " EVALUATION START " + "="*15)
calculate_metrics(actual_time, pred_time, "Execution Time Model")
calculate_metrics(actual_energy, pred_energy, "Energy Consumption Model")

#  Save metrics summary CSV ---
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("evaluation_summary_metrics.csv", index=False)

print("Done. files saved :")
print("  └─ evaluation_predictions.csv   (Detailed row-by-row tracking)")
print("  └─ evaluation_summary_metrics.csv  (Error Metric summaries)")