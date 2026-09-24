import torch
import pandas as pd
import os
from model_MLP import MLP

# Step1. Define models, paths, and features
features = [
    "jit_enabled", "total_startup_cost", "total_cost", "max_plan_rows", 
    "total_plan_width", "Seq Scan", "Hash Join", "Nested Loop", 
    "Aggregate", "Sort", "Other_Operators"
]

energy_model_path = "mlp_energy_model.pth"
time_model_path = "mlp_time_model.pth"
output_excel = "extracted_model_weights_comparison.xlsx"

# Initialize a clean Excel writer wrapper
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    
    # Step2. Process energy model
    if os.path.exists(energy_model_path):
        model_energy = MLP(dim=11)
        model_energy.load_state_dict(torch.load(energy_model_path))
        model_energy.eval()
        
        energy_weights = model_energy.net[0].weight.detach().cpu().numpy()
        df_energy = pd.DataFrame(energy_weights, columns=features)
        df_energy.index.name = "Hidden Neuron ID"
        
        # Save to the energy weights tab
        df_energy.to_excel(writer, sheet_name="Energy Weights")
        print(f" Extracted and bundled Energy weights.")
    else:
        print(f"Warning: Could not find '{energy_model_path}'. Skipping Energy tab.")

    # Step3. PROCESS TIME MODEL 
    if os.path.exists(time_model_path):
        model_time = MLP(dim=11)
        model_time.load_state_dict(torch.load(time_model_path))
        model_time.eval()
        
        time_weights = model_time.net[0].weight.detach().cpu().numpy()
        df_time = pd.DataFrame(time_weights, columns=features)
        df_time.index.name = "Hidden Neuron ID"
        
        # Save to the "Time Baseline Weights" tab
        df_time.to_excel(writer, sheet_name="Time Baseline Weights")
        print(f"Extracted and bundled Time Baseline weights.")
    else:
        print(f"Warning: Could not find '{time_model_path}'. Skipping Time tab.")

print(f"\nSuccess! Unified matrix workbook saved to: {output_excel}")