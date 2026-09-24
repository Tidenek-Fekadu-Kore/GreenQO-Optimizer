#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from graph_builder import plan_to_graph
from model_GNN import TrueGNN

CSV_PATH = "../datasets/tpch_multi_scale_final_dataset.csv"
df = pd.read_csv(CSV_PATH)
graph_dataset = []
metadata_records = []

print("Assembling Query Graphs for Energy Evaluation")
for idx, row in df.iterrows():
    qid = int(row["query_id"])
    scale = int(row["dataset_scale_gb"])
    json_path = os.path.abspath(os.path.join("..", f"tpch_{scale}gb_results_without_execution_time", f"q{qid}.json"))
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            raw_plan_data = json.load(f)
        actual_time = float(row["Execution_Time"])
        actual_energy = float(row["total_energy_consumption_memory_plus_cpu"])
        
        graph_data = plan_to_graph(raw_plan_data, actual_time, actual_energy, float(scale), float(qid))
        graph_dataset.append(graph_data)
        metadata_records.append({"Query_Identifier": qid, "Dataset_Scale_Gigabytes": scale, "Actual_Energy": actual_energy})

eval_loader = DataLoader(graph_dataset, batch_size=1, shuffle=False)

model_energy = TrueGNN(node_dim=4)
model_energy.load_state_dict(torch.load("gnn_energy_model.pth"))
model_energy.eval()

predicted_energies_log_space = []
with torch.no_grad():
    for batch in eval_loader:
        pred_tensor = model_energy(batch)
        predicted_energies_log_space.append(float(pred_tensor.view(-1).item()))

# Invert log-space predictions back to raw Joules/mWh scales
predicted_energies = np.expm1(np.array(predicted_energies_log_space)).tolist()

predictions_log = []
for i, meta in enumerate(metadata_records):
    actual = meta["Actual_Energy"]
    predicted = predicted_energies[i]
    predictions_log.append({
        "Query_Identifier": meta["Query_Identifier"],
        "Dataset_Scale_Gigabytes": meta["Dataset_Scale_Gigabytes"],
        "Actual_Energy_Joules": actual,
        "Predicted_Energy_Joules": predicted,
        "Absolute_Energy_Error": abs(actual - predicted)
    })

pd.DataFrame(predictions_log).to_csv("gnn_energy_evaluation_predictions.csv", index=False)

actual_arr, pred_arr = np.array([m["Actual_Energy"] for m in metadata_records]), np.array(predicted_energies)
mae = float(np.mean(np.abs(actual_arr - pred_arr)))
mape = float(np.mean(np.abs((actual_arr - pred_arr) / np.where(actual_arr == 0, 1e-5, actual_arr))) * 100)

summary_log = [{"Model_Target_Metric": "GNN Energy Consumption Model", "Mean_Absolute_Error": mae, "Mean_Absolute_Percentage_Error": mape}]
pd.DataFrame(summary_log).to_csv("gnn_energy_evaluation_summary_metrics.csv", index=False)

print("\n ENERGY MODEL SUMMARY PERFORMANCE ")
print(f"Mean Absolute Error            : {mae:.4f} Joules")
print(f"Mean Absolute Percentage Error : {mape:.2f}%")