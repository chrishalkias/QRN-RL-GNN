from torch_geometric.nn import GATv2Conv
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
    """Defines the GNN model used"""
    def __init__(self, node_dim=2,
                 embedding_dim=16,
                 num_heads = 2,
                 hidden_dim=32,
                 output_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
           GATv2Conv(node_dim, embedding_dim, heads=num_heads, edge_dim=2),
          #  GATConv(embedding_dim*num_heads, embedding_dim, heads=num_heads, edge_dim=1),
        )
        self.decoder = nn.Sequential(
          nn.Linear(embedding_dim * num_heads, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, output_dim),
          )
    def forward(self, data):
      x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

      for layer in self.encoder:
          x = layer(x, edge_index, edge_attr=edge_attr)
      q_values = self.decoder(x)  # Shape: [num_nodes, 2]

      return q_values