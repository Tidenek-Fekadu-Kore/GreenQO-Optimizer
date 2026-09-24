#!/usr/bin/env python3
import sys
import os
# Force Python to look one level up so it can find the config.py module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import json

def extract_node_metrics(node, metrics_summary):
    """Recursively walks the PostgreSQL execution tree to extract operational features."""
    if not node:
        return
        
    # Aggregate continuous variables
    metrics_summary["total_startup_cost"] += float(node.get("Startup Cost", 0.0))
    metrics_summary["total_cost"] += float(node.get("Total Cost", 0.0)) # ce-calculated internally by potgresql using ce formula
    metrics_summary["total_plan_width"] += int(node.get("Plan Width", 0))
    
    current_rows = int(node.get("Plan Rows", 0))
    if current_rows > metrics_summary["max_plan_rows"]:
        metrics_summary["max_plan_rows"] = current_rows

    # Track operational choices 
    node_type = node.get("Node Type")
    if node_type in metrics_summary:
        metrics_summary[node_type] += 1
    else:
        metrics_summary["Other_Operators"] += 1

    # Traverse child plans
    if "Plans" in node:
        for child in node["Plans"]:
            extract_node_metrics(child, metrics_summary)

def process_raw_plan(json_path):
    """Loads a single query JSON file and returns a flat dictionary of structural features."""
    with open(json_path, "r") as file:
        raw_data = json.load(file)
        # Safe JSON wrapper verification
        if isinstance(raw_data, list):
            raw_data = raw_data[0]
        
    metrics_summary = {
        "jit_enabled": 1.0 if "JIT" in raw_data else 0.0,
        "total_startup_cost": 0.0,
        "total_cost": 0.0,
        "max_plan_rows": 0,
        "total_plan_width": 0,
        "Seq Scan": 0,
        "Hash Join": 0,
        "Nested Loop": 0,
        "Aggregate": 0,
        "Sort": 0,
        "Other_Operators": 0
    }
    
    if "Plan" in raw_data:
        extract_node_metrics(raw_data["Plan"], metrics_summary)
    return metrics_summary

def build_dataset():
    """Loops through 1GB, 3GB, and 5GB benchmarks, compiling them into one master dataset."""
    
    # Define our targeted data scales
    scales = ["1gb", "3gb", "5gb"]
    all_scale_dataframes = []
    
    print("\n================= UNIFIED BUILD START =================")
    
    for scale in scales:
        print(f"\n--- Processing {scale.upper()} Scale Factor ---")
        
        # Paths dynamic to the current scale factor loop
        # Load json file exracted from EXPLAIN (FORMAT JSON) with out execution time
        DATA_DIR = os.path.join(os.path.dirname(__file__), "..", f"tpch_{scale}_query_execution_plan_without_execution_time")
        # Load energy consumption and execution time captured using our framework for each query
        EXCEL_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", f"tpch_{scale}_energy.xlsx")
        
        print(f"Targeting Excel: {os.path.abspath(EXCEL_PATH)}")
        
        if not os.path.exists(EXCEL_PATH):
            print(f" SKIPPING: Excel file for {scale} does not exist at that path location")
            continue
            
        energy = pd.read_excel(EXCEL_PATH)
        
        # Clean typo on loading: Dynamically locate the 3rd column (Index 2) and force rename it to 'Execution_Time' regardless of  tracking typo
        if len(energy.columns) >= 3:
            old_time_col = energy.columns[2]
            energy.rename(columns={old_time_col: "Execution_Time"}, inplace=True)
            print(f" SUCCESS: Renamed column '{old_time_col}' to 'Execution_Time'")
            
        # Parse query JSON files for this specific scale
        rows = []
        found_files = 0
        for qid in range(1, 23):
            json_file = os.path.join(DATA_DIR, f"q{qid}.json")
            if os.path.exists(json_file):
                found_files += 1
                feats = process_raw_plan(json_file)
            else:
                feats = {k: 0.0 for k in ["jit_enabled", "total_startup_cost", "total_cost", "max_plan_rows", "total_plan_width", "Seq Scan", "Hash Join", "Nested Loop", "Aggregate", "Sort", "Other_Operators"]}
            feats["query_id"] = qid
            rows.append(feats)

        print(f"Found {found_files} out of 22 query plans in: {f'tpch_{scale}_query_execution_plan_without_execution_time'}")
        
        X = pd.DataFrame(rows)
        
        # Merge features with energy/time targets for this specific scale size
        scale_dataset = pd.merge(X, energy, on="query_id", how="inner")
        
        # Add scale context : Add an explicit feature telling the NN what dataset size this came from
        scale_dataset["dataset_scale_gb"] = float(scale.replace("gb", ""))
        
        all_scale_dataframes.append(scale_dataset)
        print(f"Scale {scale.upper()} Matrix Built successfully. Shape: {scale_dataset.shape}")

    # Merge all scales together 
    if not all_scale_dataframes:
        print(" ERROR: No scale files were successfully compiled. Matrix is completely empty")
        print("================== BUILD END ==================\n")
        return
        
    final_master_dataset = pd.concat(all_scale_dataframes, ignore_index=True)
    print(f"\n4. Complete Multi-Scale Matrix Stacking Complete. Unified Shape: {final_master_dataset.shape}")
    
    # Save unified output 
    OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "datasets", "tpch_multi_scale_final_dataset.csv")
    
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        
    final_master_dataset.to_csv(OUTPUT_CSV, index=False)
    print(f"\nTarget File Output Destination: {os.path.abspath(OUTPUT_CSV)}")
    print(f"5. Saved Dataset File Size on Disk: {os.path.getsize(OUTPUT_CSV)} bytes")
    print("================== BUILD END ==================\n")

if __name__ == "__main__":
    build_dataset()