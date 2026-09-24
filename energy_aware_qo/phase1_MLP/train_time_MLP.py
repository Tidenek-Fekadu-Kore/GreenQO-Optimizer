import torch
import pandas as pd
from model_MLP import MLP

df = pd.read_csv("../datasets/tpch_multi_scale_final_dataset.csv")

features = [
    "jit_enabled", "total_startup_cost", "total_cost", "max_plan_rows", 
    "total_plan_width", "Seq Scan", "Hash Join", "Nested Loop", 
    "Aggregate", "Sort", "Other_Operators"
]

X_raw = torch.tensor(df[features].values, dtype=torch.float32)
Y = torch.tensor(df["Execution_Time"].values, dtype=torch.float32).view(-1, 1)

# Apply Standard Scaling
X_mean = X_raw.mean(dim=0, keepdim=True)
X_std = X_raw.std(dim=0, keepdim=True)
X_std[X_std == 0] = 1.0 
X = (X_raw - X_mean) / X_std

# SAVE THE FLAT FEATURES ---
print("\n=== SAMPLE FLAT FEATURE VECTOR (MLP INPUT) ===")
print(f"Dataset Dimensions: {X.shape[0]} queries, {X.shape[1]} features each.")
print("First query's normalized flat features:\n", X[0])
print("================================================\n")

with open("inspected_time_flat_features.txt", "w") as f:
    f.write("=== FLAT FEATURE MATRIX LOG ===\n")
    f.write(f"Feature names order: {features}\n\n")
    for idx, row in enumerate(X[:67]): 
        f.write(f"Query {idx}: {row.tolist()}\n")
print("Saved sample feature arrays to 'inspected_flat_features.txt'\n")
# -------------------------------------------------------------

# Initialize model ONCE and link it to the optimizer
model = MLP(len(features))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

print("Starting MLP Network Time-Baseline Training Loop...")
for e in range(100):
    opt.zero_grad()
    predictions = model(X)
    loss = loss_fn(predictions, Y)
    loss.backward()
    opt.step()
    
    if e % 10 == 0 or e == 99:
        print(f"MLP Baseline Epoch {e:02d} | Training Loss (MSE): {loss.item():.4f}")

# Save trained time model weights
torch.save(model.state_dict(), "mlp_time_model.pth")
print("\nSuccess! Trained baseline model weights saved to mlp_time_model.pth")