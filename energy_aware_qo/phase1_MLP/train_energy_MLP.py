import torch
import pandas as pd
from model_MLP import MLP

# Open time_train_baseline.py and train.py and update line 5/6:
# Load final dataset for MLP baseline include flat vector extracted from JSON file with out exection time
# and energy and execution time metrics captured using our framwork 
df = pd.read_csv("../datasets/tpch_multi_scale_final_dataset.csv")

features = [
    "jit_enabled", "total_startup_cost", "total_cost", "max_plan_rows", 
    "total_plan_width", "Seq Scan", "Hash Join", "Nested Loop", 
    "Aggregate", "Sort", "Other_Operators"
]

X_raw = torch.tensor(df[features].values, dtype=torch.float32)
Y = torch.tensor(df["total_energy_consumption_memory_plus_cpu"].values, dtype=torch.float32).view(-1, 1)

# Apply Standard Scaling to stabilize the neural layers :xnorm
#Standard Z-score scaling: Converts X_raw -> Xnorm
X_mean = X_raw.mean(dim=0, keepdim=True)#mean
X_std = X_raw.std(dim=0, keepdim=True)#sigma
X_std[X_std == 0] = 1.0  
# Applies the Z-score normalization formula
X = (X_raw - X_mean) / X_std

# VERIFY AND SAVE FOR ENERGY TRAINING ---
print("\n=== SAMPLE FLAT FEATURE VECTOR (ENERGY INPUT) ===")
print(f"Dataset Dimensions: {X.shape[0]} queries, {X.shape[1]} features each.")
print("First query's normalized flat features:\n", X[0])
print("================================================\n")

# Save to a distinct file name so we don't mix it up with the time baseline
with open("inspected_energy_flat_features.txt", "w") as f:
    f.write("=== ENERGY INPUT FLAT FEATURE MATRIX ===\n")
    f.write(f"Feature order: {features}\n\n")
    for idx, row in enumerate(X[:5]): # Saves the first 5 as a sample
        f.write(f"Query {idx}: {row.tolist()}\n")
print("Saved sample arrays to 'inspected_energy_flat_features.txt'\n")



model = MLP(len(features))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()#define the loss function

print("Starting MLP Network Training Loop...")
for e in range(100):
    opt.zero_grad()
    predictions = model(X) #batch size Type of training: Full-batch gradient descent and Batch size value: len(df) (the total number of samples).
    loss = loss_fn(predictions, Y)#it evaluates L_MSE between ground truth Y (y_i) and predictions (y_hat_i)
    loss.backward()
    opt.step()
    
    if e % 10 == 0 or e == 99:
        print(f"MLP Epoch {e:02d} | Training Loss (MSE): {loss.item():.4f}")

# save energy learned metrics 
torch.save(model.state_dict(), "mlp_energy_model.pth")
print("\nSuccess! Trained model weights saved to mlp_energy_model.pth")