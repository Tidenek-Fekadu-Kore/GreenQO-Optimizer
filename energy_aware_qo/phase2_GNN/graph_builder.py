import torch
import pandas as pd  
from torch_geometric.data import Data

def plan_to_graph(plan, target_time, target_energy, scale_gb, query_id):
    """Converts a raw JSON PostgreSQL plan into an augmented PyG Data object."""
    nodes = []
    edges = []

    # Operational Node Map (Categorizes node types into distinct float boundaries)
    OP_MAPPING = {
        "Seq Scan": 1.0, 
        "Index Scan": 2.0, 
        "Index Only Scan": 2.5,
        "Hash Join": 3.0, 
        "Merge Join": 4.0, 
        "Nested Loop": 5.0,
        "Hash": 6.0,
        "Sort": 7.0,
        "Aggregate": 8.0
    }

    def walk(node, parent=None):
        idx = len(nodes)
        
        node_type = node.get("Node Type", "Other")
        op_id = float(OP_MAPPING.get(node_type, 0.0))
        
        # Node features: [Plan Rows, Total Cost, Plan Width, Operational ID]
        nodes.append([
            float(node.get("Plan Rows", 0.0)) / 10000.0,   # Scaled down row count
            float(node.get("Total Cost", 0.0)) / 10000.0,   # Scaled down cost calculation "ce" calculated internally by potgresql
            float(node.get("Plan Width", 0.0)) / 100.0,     # Scaled down data column width
            op_id                                           # Operator Classification Feature
        ])
        
        if parent is not None:
            edges.append([idx, parent])
            
        for c in node.get("Plans", []):
            walk(c, idx)

    # Safe JSON root wrapper extraction
    root_plan = plan[0] if isinstance(plan, list) else plan
    if "Plan" in root_plan:
        walk(root_plan["Plan"])
    else:
        nodes.append([0.0, 0.0, 0.0, 0.0])

    x = torch.tensor(nodes, dtype=torch.float32)
    
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Convert Query ID into an unbiased 22-dimensional one-hot vector (TPC-H Q1 to Q22)
    qid_idx = max(0, min(int(query_id) - 1, 21))
    qid_one_hot = torch.nn.functional.one_hot(torch.tensor(qid_idx), num_classes=22).float().unsqueeze(0)

    # STEP 2  Calculate official standard TPC-H cardinalities relative to scale factor
    sf = float(scale_gb)
    lineitem_rows = sf * 6000000.0
    orders_rows = sf * 1500000.0
    partsupp_rows = sf * 800000.0

    # Scale cardinalities down by 10^6 (Millions of Rows) to ensure numeric training stability
    stats_tensor = torch.tensor(
        [[lineitem_rows / 1000000.0, orders_rows / 1000000.0, partsupp_rows / 1000000.0]], 
        dtype=torch.float32
    )

    # Build the complete PyG Data Object with metrics and table statistics
    data = Data(
        x=x, 
        edge_index=edge_index,
        y_time=torch.tensor([[target_time]], dtype=torch.float32),
        y_energy=torch.tensor([[target_energy]], dtype=torch.float32),
        scale=torch.tensor([[scale_gb]], dtype=torch.float32),
        query_id=qid_one_hot,
        table_stats=stats_tensor  # Attached to combat structural optimization blind spots
    )
    return data

#  TEST
if __name__ == "__main__":
    print("Testing  graph_builder.py conversion logic")
    
    mock_plan = {
        "Plan": {
            "Node Type": "Hash Join", 
            "Plan Rows": 100, 
            "Total Cost": 450.5, 
            "Plan Width": 64,
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Plan Rows": 1000,
                    "Total Cost": 120.0,
                    "Plan Width": 32
                }
            ]
        }
    }
    
    try:
        # Evaluate at a 3GB Scale for testing
        test_graph = plan_to_graph(mock_plan, target_time=1.5, target_energy=22.3, scale_gb=3.0, query_id=20)
        
        print("\nSUCCESS! Everything is working correctly inside the graphs.")
        print("-" * 50)
        print(f"Node Shapes Matrix (x)      : {test_graph.x.shape} -> Expected: [2, 4]")
        print(f"One-Hot Identity vector     : {test_graph.query_id.shape} -> Expected: [1, 22]")
        print(f"Table Stats Vector Shape    : {test_graph.table_stats.shape} -> Expected: [1, 3]")
        print(f"Injected TPC-H Base Stats   : {test_graph.table_stats.tolist()} -> (Lineitem, Orders, Partsupp in Millions)")
        print("-" * 50)
        
        diagnostic_data = [{
            "Test_Execution_Status": "SUCCESS",
            "Total_Nodes_Found": int(test_graph.x.shape[0]),
            "Features_Per_Node": int(test_graph.x.shape[1]),
            "One_Hot_Vector_Size": int(test_graph.query_id.shape[1]),
            "Table_Stats_Size": int(test_graph.table_stats.shape[1]),
            "Edge_Connection_Matrix": str(test_graph.edge_index.tolist())
        }]
        
        df_out = pd.DataFrame(diagnostic_data)
        df_out.to_csv("graph_builder_test_output.csv", index=False)
        print("Success! Created diagnostic verification test output:")
        print("  └─ graph_builder_test_output.csv")
        
    except Exception as e:
        print(f"\n ERROR: Something failed in your builder code:\n{str(e)}")