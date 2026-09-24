#!/usr/bin/env python3
import os
import json
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from graph_builder import plan_to_graph
from model_GNN import TrueGNN

# Helper function to compute pure MAPE percentage safely on raw scales
def compute_mape(predictions, targets):
    epsilon = 1e-8
    absolute_percentage_errors = torch.abs((targets - predictions) / (targets + epsilon))
    return torch.mean(absolute_percentage_errors).item() * 100.0

CSV_PATH = "../datasets/tpch_multi_scale_final_dataset.csv"
if not os.path.exists(CSV_PATH):
    print(f"CRITICAL ERROR: The dataset file was not found at: {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)
graph_dataset = []

print("Assembling Query Graphs with Scale & One-Hot Identity for Energy...")
for idx, row in df.iterrows():
    qid = int(row["query_id"])
    scale = int(row["dataset_scale_gb"])
    
    scale_str = f"{scale}gb"
    json_dir = os.path.abspath(os.path.join("..", f"tpch_{scale_str}_results_without_execution_time"))
    json_path = os.path.join(json_dir, f"q{qid}.json")
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            raw_plan_data = json.load(f)
            
        actual_time = row["Execution_Time"]
        actual_energy = row["total_energy_consumption_memory_plus_cpu"]
        
        # Build the graph matching the updated 5-argument builder signature
        graph_data = plan_to_graph(raw_plan_data, actual_time, actual_energy, float(scale), float(qid))
        graph_dataset.append(graph_data)

loader = DataLoader(graph_dataset, batch_size=4, shuffle=True)

# Initialize with 4 input node features [Rows, Cost, Width, Op_ID]
model_energy = TrueGNN(node_dim=4)
optimizer = torch.optim.Adam(model_energy.parameters(), lr=0.001)
loss_fn = torch.nn.HuberLoss(delta=1.0)

training_history_log = []
model_energy.train()


# Training loop with LOG target & MAPE traking
print("\n Training Upgraded GNN Energy Model (Log-Stabilized Target Space)")
for epoch in range(350):
    total_loss = 0
    total_mape = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        # Forward pass now interprets outputs as log-space estimations
        log_predictions = model_energy(batch)
        
        # Compress target values into log-space to balance gradients
        log_energy_targets = torch.log1p(batch.y_energy)
        
        # Compute loss natively in the compressed log space
        loss = loss_fn(log_predictions, log_energy_targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model_energy.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        
        # Invert log-space estimations back to raw units to track real MAPE %
        true_scale_predictions = torch.expm1(log_predictions)
        total_mape += compute_mape(true_scale_predictions, batch.y_energy) * batch.num_graphs
        
    avg_loss = total_loss / len(graph_dataset)
    avg_mape = total_mape / len(graph_dataset)
    
    # Track both raw Huber Loss index and True Percentage Error
    training_history_log.append({
        "Training_Epoch_Number": epoch + 1, 
        "Huber_Loss_Value": avg_loss,
        "MAPE_Percentage": avg_mape
    })
    
    if epoch % 25 == 0 or epoch == 349:
        print(f"Epoch {epoch:03d} | Log Huber Loss: {avg_loss:.4f} | Real Error: {avg_mape:.2f}% MAPE")

# Save to a distinct weights file so we don't overwrite our time model
torch.save(model_energy.state_dict(), "gnn_energy_model.pth")
pd.DataFrame(training_history_log).to_csv("gnn_energy_training_loss_history.csv", index=False)
print("\n Model trained and weights saved to gnn_energy_model.pth")