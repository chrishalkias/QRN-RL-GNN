"""
flexible_model.py - COMPLETE VERSION
=====================================

This file contains:
1. FlexibleGNN class
2. ActionSpace class  
3. combine_q_values function (THE MISSING PIECE!)

Replace your flexible_model.py with this entire file.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class FlexibleGNN(nn.Module):
    """
    GNN that outputs Q-values for variable-length action spaces.
    """
    
    def __init__(self,
                 node_dim=8,
                 edge_dim=3,
                 hidden_dim=64,
                 num_heads=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            GATv2Conv(node_dim, hidden_dim, heads=num_heads, edge_dim=edge_dim),
            nn.ELU(),
            GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, edge_dim=edge_dim),
            nn.ELU(),
        )
        
        # Edge Q-network: For entangle actions
        self.edge_q_net = nn.Sequential(
            nn.Linear(hidden_dim * num_heads * 2 + edge_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Node Q-network: For swap actions
        self.node_q_net = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        """
        Forward pass.
        
        Returns:
            dict with 'entangle_q' and 'swap_q' tensors
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 1. Encode nodes with GNN
        node_embeddings = x
        for layer in self.node_encoder:
            if isinstance(layer, GATv2Conv):
                node_embeddings = layer(node_embeddings, edge_index, edge_attr=edge_attr)
            else:
                node_embeddings = layer(node_embeddings)
        
        # 2. Compute Q-values for entangle actions
        if hasattr(data, 'physical_edges') and len(data.physical_edges) > 0:
            entangle_q = self._compute_edge_q_values(
                node_embeddings, 
                data.physical_edges,
                edge_index,
                edge_attr
            )
        else:
            entangle_q = torch.tensor([], device=x.device)
        
        # 3. Compute Q-values for swap actions
        swap_q = self.node_q_net(node_embeddings).squeeze(-1)
        
        return {
            'entangle_q': entangle_q,
            'swap_q': swap_q
        }
    
    def _compute_edge_q_values(self, node_embeddings, physical_edges, edge_index, edge_attr):
        """Compute Q-values for entangle actions on physical edges."""
        edge_features_list = []
        
        for i, j in physical_edges:
            edge_feat = self._get_edge_features(i.item(), j.item(), edge_index, edge_attr)
            
            edge_emb = torch.cat([
                node_embeddings[i],
                node_embeddings[j],
                edge_feat
            ])
            edge_features_list.append(edge_emb)
        
        edge_features = torch.stack(edge_features_list)
        q_values = self.edge_q_net(edge_features).squeeze(-1)
        
        return q_values
    
    def _get_edge_features(self, i, j, edge_index, edge_attr):
        """Find edge features for edge (i, j)."""
        mask = ((edge_index[0] == i) & (edge_index[1] == j)) | \
               ((edge_index[0] == j) & (edge_index[1] == i))
        
        indices = torch.where(mask)[0]
        
        if len(indices) > 0:
            return edge_attr[indices[0]]
        else:
            return torch.zeros(edge_attr.shape[1], device=edge_attr.device)


class ActionSpace:
    """
    Manages variable-length action spaces for arbitrary graph topologies.
    """
    
    def __init__(self, graph_data):
        """Build action space from graph state."""
        self.graph_data = graph_data
        self.n = graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else graph_data.x.shape[0]
        self.physical_edges = graph_data.physical_edges if hasattr(graph_data, 'physical_edges') else []
        
        self.actions = []
        
        # Entangle actions: one per physical edge
        for edge_idx in range(len(self.physical_edges)):
            self.actions.append(('entangle', edge_idx))
        
        # Swap actions: one per swap-eligible node
        for node_idx in range(self.n):
            if self._is_swap_eligible(node_idx):
                self.actions.append(('swap', node_idx))
        
        self.num_actions = len(self.actions)
    
    def _is_swap_eligible(self, node_idx):
        """Check if node can perform swap (has >= 2 physical neighbors)."""
        neighbor_count = 0
        for i, j in self.physical_edges:
            i_val = i.item() if hasattr(i, 'item') else i
            j_val = j.item() if hasattr(j, 'item') else j
            if node_idx == i_val or node_idx == j_val:
                neighbor_count += 1
        
        return neighbor_count >= 2
    
    def get_action(self, action_idx):
        """Decode action index to (action_type, action_target)."""
        return self.actions[action_idx]
    
    def get_action_mask(self, graph_data):
        """Get binary mask indicating which actions are currently valid."""
        mask = torch.zeros(self.num_actions, dtype=torch.bool, device=graph_data.x.device)
        
        for i, (action_type, target) in enumerate(self.actions):
            if action_type == 'entangle':
                mask[i] = self._can_entangle(target, graph_data)
            elif action_type == 'swap':
                mask[i] = self._can_swap(target, graph_data)
        
        return mask
    
    def _can_entangle(self, edge_idx, graph_data):
        """Check if entangle action is valid on this edge."""
        i, j = self.physical_edges[edge_idx]
        
        i_val = i.item() if hasattr(i, 'item') else i
        j_val = j.item() if hasattr(j, 'item') else j
        
        edge_index = graph_data.edge_index
        mask = ((edge_index[0] == i_val) & (edge_index[1] == j_val)) | \
               ((edge_index[0] == j_val) & (edge_index[1] == i_val))
        
        indices = torch.where(mask)[0]
        if len(indices) > 0:
            fidelity = graph_data.edge_attr[indices[0], 0]
            return fidelity == 0
        
        return True
    
    def _can_swap(self, node_idx, graph_data):
        """Check if swap action is valid at this node."""
        return graph_data.x[node_idx, 1] > 0


# =============================================================================
# THIS IS THE MISSING FUNCTION!
# =============================================================================

def combine_q_values(q_dict, action_space, device='cpu'):
    """
    Combine entangle and swap Q-values into single action vector.
    
    This is a utility function that takes the Q-value dictionary from 
    FlexibleGNN.forward() and combines them according to the action space ordering.
    
    Args:
        q_dict: Dictionary with keys:
            - 'entangle_q': Tensor [num_physical_edges] - Q for entangle actions
            - 'swap_q': Tensor [num_nodes] - Q for swap actions
        action_space: ActionSpace instance defining action ordering
        device: torch device ('cpu' or 'cuda')
    
    Returns:
        Tensor [num_actions] with Q-values in same order as action_space.actions
    
    Example:
        >>> q_dict = model(state)
        >>> action_space = ActionSpace(state)
        >>> q_values = combine_q_values(q_dict, action_space)
        >>> best_action_idx = q_values.argmax()
        >>> best_action = action_space.get_action(best_action_idx)
    """
    q_values = torch.zeros(action_space.num_actions, device=device)
    
    for i, (action_type, target) in enumerate(action_space.actions):
        if action_type == 'entangle':
            q_values[i] = q_dict['entangle_q'][target]
        elif action_type == 'swap':
            q_values[i] = q_dict['swap_q'][target]
    
    return q_values