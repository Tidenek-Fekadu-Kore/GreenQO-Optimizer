#!/usr/bin/env python3
import os
import sys
import torch
import pandas as pd

# Phase 2 for module code
PHASE2_PATH = "/home/admin_tidenek/energy_aware_qo/phase2_gnn"
if PHASE2_PATH not in sys.path:
    sys.path.append(PHASE2_PATH)

from graph_builder import plan_to_graph
from model_GNN import TrueGNN

class EnergyAwareOptimizer:
    def __init__(self, time_model_path, energy_model_path):
        """Loads both upgraded GAT models into memory."""
        print("Initializing Upgraded GAT Energy-Aware Plan Selector")
        
        # Node_dim remains 4 [Rows, Cost, Width, Op_ID]
        self.time_model = TrueGNN(node_dim=4)
        self.time_model.load_state_dict(torch.load(time_model_path, map_location=torch.device('cpu')))
        self.time_model.eval()
        
        self.energy_model = TrueGNN(node_dim=4)
        self.energy_model.load_state_dict(torch.load(energy_model_path, map_location=torch.device('cpu')))
        self.energy_model.eval()
        
        self.selection_history = []
        print("Both Graph Attention Networks loaded successfully.")

   # User can choose optimization strategy : ECO or Performance 
   # def select_optimal_plan(self, alternative_plans, scale_gb, query_id, optimization_strategy="performance", file_pointer=None):
    def select_optimal_plan(self, alternative_plans, scale_gb, query_id, optimization_strategy="eco", file_pointer=None):
        """Evaluates alternative query plans using updated data signatures."""
        if not alternative_plans:
            raise ValueError(" No alternative plans provided.")

        best_plan = None
        best_index = -1
        lowest_cost = float("inf")
        
        active_model = self.time_model if optimization_strategy == "performance" else self.energy_model
        metric_unit = "Seconds" if optimization_strategy == "performance" else "Joules"

        print(f" -> Processing Query {query_id} ({scale_gb}GB) [{optimization_strategy.upper()}]")

        all_option_predictions = {}

        with torch.no_grad():
            for idx, plan in enumerate(alternative_plans):
                # Convert plan object into graph data structure
                graph_data = plan_to_graph(
                    plan=plan, 
                    target_time=0.0, 
                    target_energy=0.0, 
                    scale_gb=float(scale_gb), 
                    query_id=float(query_id)
                )
                
                # Append content to the choosen scale file 
                if file_pointer is not None:
                    file_pointer.write(f"=== TOPOLOGICAL MAPPING LOG FOR QUERY {int(query_id)} | OPTION {idx} ===\n")
                    file_pointer.write(f"Strategy: {optimization_strategy.upper()} | Scale: {scale_gb}GB\n\n")
                    
                    file_pointer.write("--- EDGE INDEX (Parent-to-Child Topology Matrix) ---\n")
                    file_pointer.write(f"{graph_data.edge_index.tolist()}\n\n")
                    
                    file_pointer.write("--- NODE FEATURE MATRIX (x Tensor) ---\n")
                    for node_id, features in enumerate(graph_data.x):
                        file_pointer.write(f"Node {node_id}: {features.tolist()}\n")
                    file_pointer.write("\n" + "="*50 + "\n\n")
                # --------------------------------------------------
                
                predicted_score = float(active_model(graph_data).view(-1).item())
                all_option_predictions[f"Plan_Option_{idx}_Cost_{metric_unit}"] = predicted_score
                
                if predicted_score < lowest_cost:
                    lowest_cost = predicted_score
                    best_plan = plan
                    best_index = idx

        log_entry = {
            "Query_Identifier": int(query_id),
            "Dataset_Scale_Gigabytes": float(scale_gb),
            "Optimization_Strategy": optimization_strategy.upper(),
            "Total_Alternatives_Evaluated": len(alternative_plans),
            "Selected_Plan_Index": best_index,
            "Winning_Plan_Estimated_Cost": lowest_cost,
            "Metric_Unit": metric_unit
        }
        log_entry.update(all_option_predictions)
        self.selection_history.append(log_entry)
        
        return best_plan

    def save_history_to_csv(self, filename="optimizer_selection_history.csv"):
        """Saves selection decisions to a local CSV file."""
        if not self.selection_history:
            print("No decision history found to save.")
            return
        df_history = pd.DataFrame(self.selection_history)
        df_history.to_csv(filename, index=False)
        print(f"\n CSV Export Success! Decisions written to: {filename}")


if __name__ == "__main__":
    TIME_WEIGHTS = os.path.join(PHASE2_PATH, "gnn_time_model.pth")
    ENERGY_WEIGHTS = os.path.join(PHASE2_PATH, "gnn_energy_model.pth")
    
    plan_option_0 = {
        "Plan": {
            "Node Type": "Hash Join", "Plan Rows": 5000, "Total Cost": 950.0, "Plan Width": 32,
            "Plans": [{"Node Type": "Seq Scan", "Plan Rows": 5000, "Total Cost": 250.0, "Plan Width": 32}]
        }
    }
    plan_option_1 = {
        "Plan": {
            "Node Type": "Nested Loop", "Plan Rows": 150, "Total Cost": 120.0, "Plan Width": 32,
            "Plans": [{"Node Type": "Index Scan", "Plan Rows": 150, "Total Cost": 15.0, "Plan Width": 32}]
        }
    }
    
    alternatives = [plan_option_0, plan_option_1]
    
    try:
        optimizer = EnergyAwareOptimizer(time_model_path=TIME_WEIGHTS, energy_model_path=ENERGY_WEIGHTS)
        
        print("\n  Generating topologies and separating outputs into 3 scale files...")
        print("=" * 65)
        
        with open("scale_1gb_topologies.txt", "w") as f1, \
             open("scale_3gb_topologies.txt", "w") as f3, \
             open("scale_5gb_topologies.txt", "w") as f5:
            
            f1.write("==================================================\n  TOTAL TOPOLOGY REPOSITORY FOR ALL 1GB QUERIES \n==================================================\n\n")
            f3.write("==================================================\n  TOTAL TOPOLOGY REPOSITORY FOR ALL 3GB QUERIES \n==================================================\n\n")
            f5.write("==================================================\n  TOTAL TOPOLOGY REPOSITORY FOR ALL 5GB QUERIES \n==================================================\n\n")
            
            for target_scale, active_file in [(1.0, f1), (3.0, f3), (5.0, f5)]:
                for q_id in range(1, 23):
                    
                    # Run Performance Strategy
                    optimizer.select_optimal_plan(
                        alternative_plans=alternatives, 
                        scale_gb=target_scale, 
                        query_id=q_id, 
                        optimization_strategy="performance",
                        file_pointer=active_file
                    )
                    
                    # Run Eco Strategy
                    optimizer.select_optimal_plan(
                        alternative_plans=alternatives, 
                        scale_gb=target_scale, 
                        query_id=q_id, 
                        optimization_strategy="eco",
                        file_pointer=active_file
                    )
                    
        optimizer.save_history_to_csv("optimizer_selection_history.csv")
        print("\n Run Completed Successfully! You now have exactly 3 populated scale text files.")
        
    except Exception as e:
        print(f"\n Test failed: {str(e)}")