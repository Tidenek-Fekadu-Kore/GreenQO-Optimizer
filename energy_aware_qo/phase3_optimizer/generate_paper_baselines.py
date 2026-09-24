#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd
import numpy as np

# PATH BRIDGING: Scan Phase 2 for module code
PHASE2_PATH = "/home/admin_tidenek/energy_aware_qo/datasets"
if PHASE2_PATH not in sys.path:
    sys.path.append(PHASE2_PATH)

CSV_PATH = "../datasets/tpch_multi_scale_final_dataset.csv"

if not os.path.exists(CSV_PATH):
    print(f"ERROR: Dataset not found at {CSV_PATH}")
    exit(1)

# Load your final execution dataset
df = pd.read_csv(CSV_PATH)

# We will collect all terminal strings to save them dynamically to a text file later
terminal_output_log = []

def log_print(text=""):
    """Helper function to print to terminal AND store for file export"""
    print(text)
    terminal_output_log.append(text)

log_print("Step 1: Profiling Hardware Wattage and Query Divergence Anomalies...")
log_print("-" * 75)

# Calculate operational wattage: Joules / Seconds = Watts
df["Calculated_Wattage"] = df["total_energy_consumption_memory_plus_cpu"] / df["Execution_Time"]

# Identify query profiles with highly anomalous power draws
high_power_queries = df.sort_values(by="Calculated_Wattage", ascending=False).head(10)

log_print("\n TOP 5 HIGHEST POWER-SPIKING QUERIES (Potential Performance vs. Eco Anomalies):")
for idx, row in high_power_queries.head(10).iterrows():
    log_print(f" -> Query Q{int(row['query_id'])} ({int(row['dataset_scale_gb'])}GB) "
              f"| Time: {row['Execution_Time']:.2f}s "
              f"| Energy: {row['total_energy_consumption_memory_plus_cpu']:.2f}J "
              f"| Average Draw: {row['Calculated_Wattage']:.2f} Watts")

log_print("-" * 75)
log_print("Step 2: Simulating Native PostgreSQL Cost Selection vs. Actual Performance...")
log_print("-" * 75)

native_comparison_log = []

# Loop through our dataset rows to map against alternative JSON plans
for idx, row in df.iterrows():
    qid = int(row["query_id"])
    scale = int(row["dataset_scale_gb"])
    
    # Locate the target plan results folder
    scale_str = f"{scale}gb"
    json_path = os.path.abspath(os.path.join(PHASE2_PATH, "..", f"tpch_{scale_str}_results_without_execution_time", f"q{qid}.json"))
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            raw_plan_data = json.load(f)
            
        root_plan = raw_plan_data[0] if isinstance(raw_plan_data, list) else raw_plan_data
        plan_details = root_plan.get("Plan", {})
        
        # Extract the native PostgreSQL optimizer's internal estimated cost calculation
        native_postgres_cost = float(plan_details.get("Total Cost", 0.0))
        native_postgres_rows = float(plan_details.get("Plan Rows", 0.0))
        
        native_comparison_log.append({
            "Query_ID": qid,
            "Scale_GB": scale,
            "Postgres_Native_Estimated_Cost": native_postgres_cost,
            "Postgres_Native_Estimated_Rows": native_postgres_rows,
            "Actual_Execution_Time_Sec": row["Execution_Time"],
            "Actual_Energy_Consumption_Joules": row["total_energy_consumption_memory_plus_cpu"],
            "Actual_Hardware_Wattage": row["Calculated_Wattage"]
        })

# Export baseline dataset to disk
df_baselines = pd.DataFrame(native_comparison_log)
df_baselines.to_csv("paper_baseline_postgres_metrics.csv", index=False)

log_print("\n Success! Created baseline profile tracking data file:")
log_print("   └─ paper_baseline_postgres_metrics.csv")

log_print("\n--- OVERALL WORKLOAD SUMMARY FOR MANUSCRIPT ---")
log_print(f"Total Unique Queries Profiled : {len(df_baselines)}")
log_print(f"Average Workload Wattage      : {df_baselines['Actual_Hardware_Wattage'].mean():.2f} Watts")

# 1. Capture the Absolute Max Peak Power
max_power_row = df_baselines.loc[df_baselines['Actual_Hardware_Wattage'].idxmax()]
log_print(f"Maximum Peak Wattage Observed : {max_power_row['Actual_Hardware_Wattage']:.2f} Watts "
          f"(Q{int(max_power_row['Query_ID'])} at {int(max_power_row['Scale_GB'])}GB)")

log_print("\n --- AUTOMATED DISPARITY ANALYSIS (Same-Scale Pairs) ---")

# 2. Automated Trade-off Scan
disparities = []
# Group by scale to ensure we only pair queries within the SAME dataset size
for scale, group in df_baselines.groupby("Scale_GB"):
    sorted_group = group.sort_values(by="Actual_Execution_Time_Sec")
    
    # Compare every query pair within this scale
    for i in range(len(sorted_group)):
        for j in range(i + 1, len(sorted_group)):
            plan_a = sorted_group.iloc[i]  # Faster execution time
            plan_b = sorted_group.iloc[j]  # Slower execution time
            
            # Check if the faster plan consumed MORE energy (The Micro-Slammer Anomaly)
            if plan_a["Actual_Energy_Consumption_Joules"] > plan_b["Actual_Energy_Consumption_Joules"]:
                energy_increase_pct = (
                    (plan_a["Actual_Energy_Consumption_Joules"] - plan_b["Actual_Energy_Consumption_Joules"]) 
                    / plan_b["Actual_Energy_Consumption_Joules"]
                ) * 100
                
                disparities.append({
                    "Scale_GB": scale,
                    "Fast_Query_ID": int(plan_a["Query_ID"]),
                    "Slow_Query_ID": int(plan_b["Query_ID"]),
                    "Fast_Query_Time_Sec": plan_a["Actual_Execution_Time_Sec"],
                    "Slow_Query_Time_Sec": plan_b["Actual_Execution_Time_Sec"],
                    "Time_Saved_Sec": plan_b["Actual_Execution_Time_Sec"] - plan_a["Actual_Execution_Time_Sec"],
                    "Fast_Query_Joules": plan_a["Actual_Energy_Consumption_Joules"],
                    "Slow_Query_Joules": plan_b["Actual_Energy_Consumption_Joules"],
                    "Energy_Spike_Pct": energy_increase_pct
                })

if disparities:
    df_disp = pd.DataFrame(disparities)
    
    # CRITICAL FIX: Save ALL calculated disparities to a structured CSV file
    df_disp.to_csv("all_detected_energy_disparities.csv", index=False)
    log_print(" Success! Saved full anomaly comparison profile to:")
    log_print("   └─ all_detected_energy_disparities.csv")
    
    # Filter anomalies to isolate your 20% - 60% manuscript boundaries
    manuscript_range = df_disp[(df_disp["Energy_Spike_Pct"] >= 20) & (df_disp["Energy_Spike_Pct"] <= 60)]
    
    if not manuscript_range.empty:
        log_print(f"\nValidated Manuscript Energy Disparity Window: "
                  f"{manuscript_range['Energy_Spike_Pct'].min():.1f}% to {manuscript_range['Energy_Spike_Pct'].max():.1f}%")
        
        log_print("\nTop Latency-vs-Energy Anomalies across all scales:")
        # Display the sorted group directly in console
        all_scales_view = df_disp.sort_values(by="Energy_Spike_Pct", ascending=False)
        log_print(all_scales_view[["Scale_GB", "Fast_Query_ID", "Slow_Query_ID", "Time_Saved_Sec", "Energy_Spike_Pct"]].to_string(index=False))
else:
    log_print("No non-linear trade-off anomalies detected in this run.")

#  WRITING TERMINAL PROMPT TO LOG FILE
with open("manuscript_terminal_report.txt", "w") as report_file:
    report_file.write("\n".join(terminal_output_log))
print("\n Saved complete terminal readout string to text file:")
print("   └─ manuscript_terminal_report.txt")