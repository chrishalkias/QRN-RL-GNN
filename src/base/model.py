from torch_geometric.nn import GATv2Conv, global_max_pool
import torch
import torch.nn as nn

'''
Using a trained model:

model_state_dict = torch.load("/Users/chrischalkias/GitHub/QRN-RL-GNN/assets/gnn_model.pth")
net = RepeaterNetwork(n=4) # for example
state = net.tensorState()
model = GNN()
model.load_state_dict(model_state_dict)
model(state)
'''

class GNN(nn.Module):
    """
    Enhanced GNN model with more capacity for complex features.
    """
    def __init__(self, 
                 node_dim,  
                 edge_dim=1,
                 embedding_dim=16, 
                 num_heads=2,    
                 hidden_dim=32,  
                 output_dim=2):
        super().__init__()
        
        # # One layer GAT
        # self.encoder = nn.SequentiaGATv2Conv(node_dim, embedding_dim, heads=num_heads, edge_dim=edge_dim)

        self.encoder = nn.ModuleList([
            GATv2Conv(node_dim, embedding_dim, heads=num_heads, edge_dim=edge_dim),
            GATv2Conv(embedding_dim * num_heads, embedding_dim, heads=num_heads, edge_dim=edge_dim),
])
        
        # 2 layer decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        for layer in self.encoder:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = layer(x)
        # # uncomment for global pooling
        # batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        # # global graph embedding
        # global_x = global_max_pool(x, batch)
        # # broadcast global embedding back to all nodes
        # global_x_expanded = global_x[batch]
        # # concat local and global contexts
        # combined_x = torch.cat([x, global_x_expanded], dim=-1)
        
        q_values = self.decoder(x)  # Shape: [num_nodes, 2]
        return q_values