# -*- coding: utf-8 -*-
"""
Created on Thu 6 Mar 2025

Here are the implementations of the GNNPolicy and GATPolicy classes.
These classes are used to define custom feature extractors for the
Stable-Baselines3 library. The GNNPolicy uses a simple GNN to extract
features from the environment graph, while the GATPolicy uses a Graph
Attention Network (GAT) to extract features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.nn import GlobalAttention
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GNNPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, graph=None):
      """
                        ███████  ███    ██ ███    ██
                        ██       ████   ██ ████   ██
                        ██   ███ ██ ██  ██ ██ ██  ██
                        ██    ██ ██  ██ ██ ██  ██ ██
                        ██████   ██   ████ ██   ████

       This class implements a custom GNN policy for Stable-Baselines3.
       -------------------------------------------------------------------------
       This implementation uses the graph structure in a very general way, by
       aggregating to graph level features. While this coarse graining can be
       seen as a big information loss it might enable the learning agent to
       learn more eficciently by encoding more 'big picture' ideas about the system.

       Args:
        observation_space (gym.Space) : The observation space of the environment.
        features_dim      (int)       : The feature dim extracted by the GNN.
        graph             (dict)      : The graph structure of the environment.

       Attributes:
        gnn               (nn.Module) : The GNN model.
        num_edges         (int)       : The number of edges in the graph.
        graph             (dict)      : The graph structure of the environment.

       Methods:
        forward()           (nn.Module) : The forward pass of the GNN.
       """
      super().__init__(observation_space, features_dim)
      self.graph = graph
      self.gnn = nn.Sequential(
          nn.Linear(2, 64),  # 2 bcs of len(L_ij, E_ij])=2
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, features_dim))

      if len(observation_space.shape) == 1:  # If observation is 1D
        self.num_edges = observation_space.shape[0] // 2
      else:
        raise ValueError("Observation space must be a 1D array for this implementation.")

    def forward(self, observations):
        """
        The forward pass of the neural network. It processes batches of
        flattened observations with GNN then aggregates the edge embeddings and
        stacks features back into a single batch
        """
        batch_data = []
        for obs in observations:
            edge_attr = obs.view(self.num_edges, 2) #-> [num_edges, 2]
            edges = [[i, j] for i, j in sorted(self.graph.keys())]
            edge_index = torch.tensor(edges).t().contiguous() #-> [2, num_edges]
            data = Data(edge_index=edge_index, edge_attr=edge_attr)
            batch_data.append(data)

        features = []
        for data in batch_data: # o through each graph in the batch
            edge_emb = self.gnn(data.edge_attr)
            # Aggregate edge embeddings to get global features (mean pooling)
            graph_feature = torch.mean(edge_emb, dim=0, keepdim=True) #-> [1, features_dim]
            features.append(graph_feature)

        # Stack features into a batch, -> [batch_size, features_dim]
        return torch.cat(features, dim=0)


class GATPolicy(BaseFeaturesExtractor):
  def __init__(self,
                observation_space,
                features_dim=64,
                graph=None,
                num_heads=4,
                pooling = 'mean'):
      """
                           ██████   █████  ████████
                          ██       ██   ██    ██
                          ██   ███ ███████    ██
                          ██    ██ ██   ██    ██
                           ██████  ██   ██    ██


      GNNPolicy using a Graph Attention Network (GAT) for feature extraction.

      Args:
        observation_space (gym.Space) : The observation space of the env
        features_dim      (int)       : The output feature dimension
        graph             (dict)      : Graph adjacency structure
        num_heads         (int)       : Number of attention heads

      Attributes:
        gnn               (nn.Module) : GAT-based graph neural network
        num_edges         (int)       : Number of edges in the graph
        graph             (dict)      : The graph structure of the environment
      """
      super().__init__(observation_space, features_dim)
      self.graph = graph
      self.pooling = pooling
      # define the 2 layer GATConv network
      self.gnn = nn.ModuleList([
        GATConv(in_channels=2, out_channels=32, heads=num_heads, concat=True),
        GATConv(in_channels=32 * num_heads, out_channels=features_dim, heads=1, concat=False)])
      self.num_edges = observation_space.shape[0] // 2  # edges encoded in obs

  def forward(self, observations):
    """
    Forward pass of the GAT-based GNN policy.
    """
    batch_data = []
    for obs in observations:
      # # maybe get unique node id's and then reate a mapping
      # unique_nodes = np.unique([i for i, j in self.graph.keys()] + [j for i, j in self.graph.keys()])
      # node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
      # # remap edge indices
      # mapped = [[node_mapping[i], node_mapping[j]] for i, j in sorted(self.graph.keys())]
      # edge_index = torch.tensor(mapped).t().contiguous()

      edge_list = [[i, j] for i, j in sorted(self.graph.keys())]
      edge_tensor = torch.tensor(edge_list).t() # Shape: [2, num_edges]
      edge_index = edge_tensor.contiguous()  #make sure its contiguous

      edge_attr = obs.view(self.num_edges, 2)  # shape: [num_edges, 2]
      data = Data(edge_index=edge_index, edge_attr=edge_attr)
      batch_data.append(data)

    features = []
    for data in batch_data:  #Process each graph in the batch
      x = torch.randn(data.num_nodes, 2)  # random node features since no info
      for gat_layer in self.gnn: #attention layers
        x = gat_layer(x, data.edge_index)
        x = F.relu(x)
      #many choices here with differemt pros (P) and cons (C)
      if self.pooling == 'mean': #Shape: [1, features_dim] (for all)
        #(P) Works well if all nodes contribute equally to the graph representation
        #(P) Often used in graph classification tasks
        #(C) Loses per-node specificity
        graph_feature = torch.mean(x, dim=0, keepdim=True)
      elif self.pooling == 'max':
        #(P) Captures the most important (strongest activation) features
        #(P) Useful in tasks where only certain nodes dominate the decision
        #(C) Can be sensitive to outliers
        graph_feature = torch.max(x, dim=0, keepdim=True)
      elif self.pooling == 'sum':
        #(P) Preserves graph size information
        #(P) Works well when features represent presence of properties (E_ij)
        #(C) May lead to very large values for big graphs
        graph_feature = torch.sum(x, dim=0, keepdim=True)
      elif self.pooling == 'attention':
        #(P) Lets the model learn which nodes are important
        #(P) More expressive than simple pooling methods
        #(C) Computationally expensive
        gate_nn = nn.Linear(x.shape[1], 1)
        attn_pool = GlobalAttention(gate_nn)
      elif self.pooling == 'weighted_sum':
        #(P) Lets the model dynamically learn importance weights for each node.
        #(P) More flexible than fixed pooling functions.
        #(C) Needs extra parameters.
        weights = nn.Parameter(torch.ones(x.shape[0], 1))  # Learnable weights
      features.append(graph_feature)
    return torch.cat(features, dim=0)