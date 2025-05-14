# -*- coding: utf-8 -*-
# src/models.py

'''
    ███    ██ ███████ ██    ██ ██████   █████  ██          ███    ██ ███████ ████████ ███████ 
    ████   ██ ██      ██    ██ ██   ██ ██   ██ ██          ████   ██ ██         ██    ██      
    ██ ██  ██ █████   ██    ██ ██████  ███████ ██          ██ ██  ██ █████      ██    ███████ 
    ██  ██ ██ ██      ██    ██ ██   ██ ██   ██ ██          ██  ██ ██ ██         ██         ██ 
    ██   ████ ███████  ██████  ██   ██ ██   ██ ███████     ██   ████ ███████    ██    ███████ 
                                                                   
                            Created Wed 02 Apr 2025
                        Contains the NN models for the RL agent.
'''

import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class CNN(nn.Module):
    def __init__(self,
                 convolutions = 5,
                 pooling_dim = 32,
                 embeding_dim = 64,
                 hidden_dim = 64,
                 unembeding_dim = 8,
                 ):
      """
      Description:                                            
        CNN with fixed parameters for any input:

      Attributes:
        convolutions   (int) : Number of convolutional layers,
        pooling_dim    (int) : Dimension of the pooling layer,
        embeding_dim   (int) : Dimension of the embedding layer,
        hidden_dim     (int) : Dimension of the hidden layer,
        unembeding_dim (int) : Dimension of the unembedding layer,

      Methods:
        forward  (x:Tensor)  : Forward pass through the network
      """
      super().__init__()
      self.convolutions = convolutions
      self.pooling_dim = pooling_dim
      self.embeding_dim = embeding_dim
      self.hidden_dim = hidden_dim
      self.unembeding_dim = unembeding_dim

      self.conv = nn.Conv2d(2, 2, kernel_size=2, padding=1)
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.spatial_compress = nn.Sequential(
        nn.AdaptiveAvgPool2d((pooling_dim, pooling_dim)),  # Force output to (4, 4)
        nn.Flatten(),
        nn.Linear(pooling_dim*pooling_dim*2, embeding_dim)  # Fixed input size (target_dim)
      )
      
      self.fcn = nn.Sequential(
          nn.Linear(embeding_dim, 64),
          nn.ReLU(),
          nn.Linear(64, unembeding_dim)
      )

    def forward(self, x):
      """
      Args:
          x (torch.Tensor): Input tensor of shape (1, 2, H, W)
      Returns:
          torch.Tensor: Output tensor of shape (1, 4)
      """
      x = x.float()  # Ensure float32
      original_input = x
      
      # Convolution + Pooling
      for _ in range(self.convolutions):
        x = self.conv(x)       # Shape: (1, 2, H, W) → (1, 2, H, W) (padding=1 preserves size)
        x = self.pool(x)       # Shape: (1, 2, H//2, W//2)
      
      # Spatial compression (flatten + project)
      batch_size, C, H, W = x.shape
      x = self.spatial_compress(x)  # Shape: (1, target_dim)
      
      # FCN
      x = self.fcn(x)              # Shape: (1, 8)
      x = x.view(batch_size, 2, self.unembeding_dim // 2)  # Shape: (1, 2, 4)
      
      # Restore original dimensionality
      original_flat = original_input.view(batch_size, 2, -1).permute(0, 2, 1)
      decoded = torch.bmm(original_flat, x)  # Shape: (1, H*W, 4)
      output = F.softmax(decoded, dim=-1)
      return output.squeeze(0)  # Shape: (H*W, 4)
    

class GNN(nn.Module):
    def __init__(self, node_dim=1, 
                 embedding_dim=16,
                 num_layers = 2,
                 num_heads = 4,  
                 hidden_dim=64, 
                 unembedding_dim = 16, 
                 output_dim=4):
        """
      Description:                        
        The GNN model with fixed parameters for any input:

      Attributes:
        embedding_dim   (int),
        num_layers      (int),
        num_heads       (int),  
        hidden_dim      (int), 
        unembedding_dim (int), 
        output_dim      (int),
      Methods:
        -forward      (x:Tensor)  : Forward pass through the network
      """
        super().__init__()

        self.output_dim = output_dim
        # self.gnn_layers = nn.ModuleList()
        # self.ingat = GATConv(node_dim, embedding_dim, heads=num_heads)
        # self.gatconv = GATConv(embedding_dim * num_heads, embedding_dim, heads=num_heads)

        # self.gnn_layers.append(self.ingat)
        # for _ in range(num_layers):
        #     self.gnn_layers.append(self.gatconv)

        self.encoder = nn.Sequential(
           GATConv(node_dim, embedding_dim, heads=num_heads),
        )
        for _ in range(num_layers):
           self.encoder.append(GATConv(embedding_dim * num_heads, embedding_dim, heads=num_heads),)

        self.latent = nn.Sequential(
          nn.Linear(embedding_dim * num_heads, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, unembedding_dim)
      )
        self.decoder = nn.Sequential(
            nn.Linear(unembedding_dim, output_dim),
            nn.Softmax(dim =-1)
        )

    def forward(self, data: torch.tensor):
        """
        Args:
            data (Data): Must have:
                - x: Node features (shape: [n, node_dim], default=torch.ones)
                - edge_index: Graph connectivity (shape: [2, num_edges])
                - edge_attr: Edge features (shape: [num_edges, edge_feat_dim])
        Returns:
            torch.Tensor: Output of shape (n, 4)
        """
        x, edge_index = data.x, data.edge_index
        
        # GNN forward pass
        # x = self.encoder(x, edge_index)

        for layer in self.encoder:
            x = layer(x, edge_index)
            # x = F.leaky_relu(x, negative_slope=0.01,)

        x = self.latent(x)
        x = self.decoder(x)
        
        return x
    