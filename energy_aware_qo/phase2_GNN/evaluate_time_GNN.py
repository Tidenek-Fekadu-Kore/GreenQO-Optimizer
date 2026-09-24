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
if not os.path.exists(CSV_PATH):
    print(f"CRITICAL ERROR: The dataset file was not found at: {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)
graph_dataset = []
metadata_records = []

print("Assembling Query Graphs for Evaluation")
for idx, row in df.iterrows():
    qid = int(row["query_id"])
    scale = int(row["dataset_scale_gb"])
    
    scale_str = f"{scale}gb"
    json_dir = os.path.abspath(os.path.join("..", f"tpch_{scale_str}_results_without_execution_time"))
    json_path = os.path.join(json_dir, f"q{qid}.json")
    
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                raw_plan_data = json.load(f)
            actual_time = float(row["Execution_Time"])
            actual_energy = float(row["total_energy_consumption_memory_plus_cpu"])
            
            graph_data = plan_to_graph(raw_plan_data, actual_time, actual_energy, float(scale), float(qid))
            graph_dataset.append(graph_data)
            metadata_records.append({
                "Query_Identifier": qid, "Dataset_Scale_Gigabytes": scale, "Actual_Execution_Time_Seconds": actual_time
            })
        except Exception as e:
            print(f"Error parsing Query {qid} at scale {scale}GB: {str(e)}")

eval_loader = DataLoader(graph_dataset, batch_size=1, shuffle=False)

try:
    model_time = TrueGNN(node_dim=4)
    model_time.load_state_dict(torch.load("gnn_time_model.pth"))
    model_time.eval()
    print(" GNN Model weights loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR loading model: {str(e)}")
    exit(1)

predicted_times_log_space = []
with torch.no_grad():
    for batch in eval_loader:
        try:
            pred_tensor = model_time(batch)
            predicted_times_log_space.append(float(pred_tensor.view(-1).item()))
        except Exception:
            predicted_times_log_space.append(0.0)

# Invert log-space predictions back to raw seconds scales
predicted_times = np.expm1(np.array(predicted_times_log_space)).tolist()

predictions_log = []
for i, meta in enumerate(metadata_records):
    actual = meta["Actual_Execution_Time_Seconds"]
    predicted = predicted_times[i]
    predictions_log.append({
        "Query_Identifier": meta["Query_Identifier"],
        "Dataset_Scale_Gigabytes": meta["Dataset_Scale_Gigabytes"],
        "Actual_Execution_Time_Seconds": actual,
        "Predicted_Execution_Time_Seconds": predicted,
        "Absolute_Execution_Time_Error": abs(actual - predicted)
    })

pd.DataFrame(predictions_log).to_csv("gnn_evaluation_predictions.csv", index=False)

actual_arr, pred_arr = np.array([m["Actual_Execution_Time_Seconds"] for m in metadata_records]), np.array(predicted_times)
mae = float(np.mean(np.abs(actual_arr - pred_arr)))
mape = float(np.mean(np.abs((actual_arr - pred_arr) / np.where(actual_arr == 0, 1e-5, actual_arr))) * 100)

summary_log = [{"Model_Target_Metric": "GNN Execution Time Model", "Mean_Absolute_Error": mae, "Mean_Absolute_Percentage_Error": mape}]
pd.DataFrame(summary_log).to_csv("gnn_time_evaluation_summary_metrics.csv", index=False)
print("\n Evaluation complete. Metrics stored successfully.")