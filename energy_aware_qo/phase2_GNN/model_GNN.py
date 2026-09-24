import torch
import torch.nn as nn
import torch.nn.functional as F
# Step 1 : Swapped GCNConv for GATConv to capture attention paths
from torch_geometric.nn import GATConv, global_mean_pool

class TrueGNN(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        # Multi-head Graph Attention: 4 input features -> 32 features out (using 2 heads)
        # Heads concatenate by default, making the output of conv1 = 32 * 2 = 64 dimensions.
        # Wk-Instantiated as an internal trainable PyTorch parameter tensor
        self.conv1 = GATConv(node_dim, 32, heads=2, concat=True)
        
        # taks 64 in ,Second layer brings it back to a clean 32-dimensional graph representation embedding
        self.conv2 = GATConv(64, 32, heads=1, concat=False)
        

        #  - 32 (Graph representation embedding vector)
        #  - 1  (Dataset Scale in GB)
        #  - 22 (One-Hot Encoded Query ID Category)
        #  - 3  (Physical base sizes for lineitem, orders, and partsupp tables)
        #  Total input dimensions = 32 + 1 + 22 + 3 = 58
        self.fc = nn.Sequential(
            nn.Linear(58, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Attention-Driven Message Passing over tree nodes
        # αkij: Normalized attention weight from head k for neighbor node j relative to target node i.
        # computed during self.conv1(x, edge_index) inside forward().
        # Calculated automatically in the backend by PyTorch Geometric.
        x = F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x, edge_index), negative_slope=0.01)
        
        # 2. Pool node structures into a unique representation vector
        graph_vector = global_mean_pool(x, batch)
        
        #  Step 2 Check: We extract data.table_stats alongside data.scale and data.query_id
        # This gives our pooling network an immediate structural look at base row configurations.
        combined_features = torch.cat([
            graph_vector, 
            data.scale, 
            data.query_id, 
            data.table_stats
        ], dim=1)
        
        # 4. Final multi-layer regression tracking
        raw_prediction = self.fc(combined_features)
        
        # 5. Smooth non-zero boundary condition enforcement
        return F.softplus(raw_prediction) + 1e-3