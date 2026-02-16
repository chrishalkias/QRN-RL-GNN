"""
flexible_repeaters.py - Quantum repeater networks with arbitrary topologies

Supports: chains, rings, stars, grids, trees, and custom adjacency matrices
"""

import numpy as np
import torch
from torch_geometric.data import Data


class FlexibleRepeaterNetwork:
    """
    Quantum repeater network simulator with support for arbitrary topologies.
    """
    
    def __init__(self,
                 n: int = 4,
                 topology: str = 'chain',
                 adjacency: torch.Tensor = None,
                 cutoff=None,
                 tau: int = 1_000,
                 p_entangle: float = 1.0,
                 p_swap: float = 1.0):
        """
        Args:
            n: Number of nodes
            topology: 'chain', 'ring', 'star', 'grid', 'tree', 'custom'
            adjacency: Custom adjacency matrix (for topology='custom')
            cutoff: Fidelity cutoff time
            tau: Decoherence time constant
            p_entangle: Entanglement success probability
            p_swap: Swap success probability
        """
        self.n = n
        self.topology = topology
        self.tau, self.cutoff = tau, cutoff
        self.p_entangle, self.p_swap = p_entangle, p_swap
        
        # Build adjacency matrix based on topology
        self.adjacency = self._build_adjacency(topology, adjacency)
        
        # Get physical edges from adjacency
        self.physical_edges = self._get_physical_edges()
        
        # Fidelity matrix (stores entanglement fidelities)
        self.fidelities = torch.zeros((self.n, self.n), dtype=torch.float)
    
    def _build_adjacency(self, topology, custom_adj=None):
        """Build adjacency matrix based on topology type"""
        if topology == 'chain':
            return self._build_chain()
        elif topology == 'ring':
            return self._build_ring()
        elif topology == 'star':
            return self._build_star()
        elif topology == 'grid':
            return self._build_grid()
        elif topology == 'tree':
            return self._build_tree()
        elif topology == 'custom':
            if custom_adj is None:
                raise ValueError("Must provide adjacency matrix for custom topology")
            return custom_adj
        else:
            raise ValueError(f"Unknown topology: {topology}")
    
    def _build_chain(self):
        """Linear chain: 0-1-2-...-n"""
        adj = torch.zeros((self.n, self.n))
        for i in range(self.n - 1):
            adj[i, i+1] = 1
            adj[i+1, i] = 1
        return adj
    
    def _build_ring(self):
        """Ring: chain with wrap-around connection"""
        adj = self._build_chain()
        if self.n > 2:
            adj[0, self.n-1] = 1
            adj[self.n-1, 0] = 1
        return adj
    
    def _build_star(self):
        """Star: node 0 connected to all others"""
        adj = torch.zeros((self.n, self.n))
        for i in range(1, self.n):
            adj[0, i] = 1
            adj[i, 0] = 1
        return adj
    
    def _build_grid(self):
        """2D grid (requires perfect square n)"""
        import math
        side = int(math.sqrt(self.n))
        if side * side != self.n:
            raise ValueError(f"Grid topology requires perfect square n, got {self.n}")
        
        adj = torch.zeros((self.n, self.n))
        for i in range(self.n):
            row, col = i // side, i % side
            
            # Right neighbor
            if col < side - 1:
                j = i + 1
                adj[i, j] = 1
                adj[j, i] = 1
            
            # Bottom neighbor
            if row < side - 1:
                j = i + side
                adj[i, j] = 1
                adj[j, i] = 1
        
        return adj
    
    def _build_tree(self):
        """Binary tree topology"""
        adj = torch.zeros((self.n, self.n))
        for i in range(self.n):
            # Left child
            left = 2*i + 1
            if left < self.n:
                adj[i, left] = 1
                adj[left, i] = 1
            
            # Right child
            right = 2*i + 2
            if right < self.n:
                adj[i, right] = 1
                adj[right, i] = 1
        
        return adj
    
    def _get_physical_edges(self):
        """Extract list of physical edges from adjacency matrix"""
        edges = []
        for i in range(self.n):
            for j in range(i+1, self.n):  # Only upper triangle
                if self.adjacency[i, j] > 0:
                    edges.append((i, j))
        return torch.tensor(edges, dtype=torch.long)
    
    def reset(self):
        """Reset all entanglements to 0"""
        self.fidelities.zero_()
    
    def tick(self, T: int):
        """Time evolution: decay all fidelities"""
        self.fidelities *= np.exp(-T / self.tau)
        
        if self.cutoff is not None:
            cutoff_threshold = np.exp(-self.cutoff / self.tau)
            self.fidelities[self.fidelities < cutoff_threshold] = 0.0
    
    def entangle(self, edge: tuple):
        """
        Attempt to create entanglement on a physical edge.
        
        Args:
            edge: (i, j) tuple of node indices
        """
        i, j = edge
        
        # Check if edge is physical
        if self.adjacency[i, j] == 0:
            raise ValueError(f"Edge {edge} is not in the physical topology")
        
        # Probabilistic success
        if self.p_entangle < np.random.rand():
            return
        
        # Age existing links
        self.tick(1)
        
        # Create new entanglement
        self.fidelities[i, j] = 1.0
        self.fidelities[j, i] = 1.0
    
    def swapAT(self, node: int):
        """
        Perform entanglement swapping at a node.
        
        For arbitrary topologies, swaps ALL pairs of links through the node.
        """
        # Probabilistic success
        if self.p_swap < np.random.rand():
            return
        
        # Find all active links to this node
        left_neighbors = torch.where(self.fidelities[:node, node] > 0)[0]
        right_neighbors = torch.where(self.fidelities[node, node+1:] > 0)[0] + node + 1
        
        # Age all links
        self.tick(1)
        
        # Perform swaps for all pairs
        for i in left_neighbors.tolist():
            for j in right_neighbors.tolist():
                Eij = self.fidelities[i, node].item()
                Ejk = self.fidelities[node, j].item()
                new_fidelity = Eij * Ejk
                
                # Check if link already exists
                existing = self.fidelities[i, j].item()
                final_fidelity = max(new_fidelity, existing)
                
                # Update links
                self.fidelities[i, node] = 0.0
                self.fidelities[node, i] = 0.0
                self.fidelities[node, j] = 0.0
                self.fidelities[j, node] = 0.0
                self.fidelities[i, j] = final_fidelity
                self.fidelities[j, i] = final_fidelity
    
    def endToEndCheck(self, source=0, target=None, timeToWait=0):
        """
        Check for end-to-end entanglement.
        
        Args:
            source: Source node (default 0)
            target: Target node (default n-1)
            timeToWait: Time to wait before checking
        
        Returns:
            (success, fidelity) tuple
        """
        if target is None:
            target = self.n - 1
        
        self.tick(timeToWait)
        
        fidelity = self.fidelities[source, target].item()
        success = (fidelity > np.random.rand())
        
        if success:
            self.fidelities[source, target] = 0.0
            self.fidelities[target, source] = 0.0
        
        return success, fidelity
    
    def tensorState(self) -> Data:
        """
        Create PyG Data representation of current state.
        
        Returns Data with:
            - x: node features [n, 8]
            - edge_index: connectivity [2, num_edges]
            - edge_attr: edge features [num_edges, 3]
            - physical_edges: physical topology [num_physical_edges, 2]
            - num_nodes: n
        """
        # Build edge index from adjacency + entanglements
        has_entanglement = self.fidelities > 0
        is_neighbor = self.adjacency > 0
        
        valid_mask = is_neighbor | has_entanglement
        edge_index = torch.nonzero(valid_mask).T  # [2, num_edges]
        
        # Edge attributes: [fidelity, is_physical, is_entangled]
        fids = self.fidelities[edge_index[0], edge_index[1]].view(-1, 1)
        is_physical = self.adjacency[edge_index[0], edge_index[1]].view(-1, 1)
        is_entangled = (fids > 0).float()
        
        edge_attr = torch.cat([fids, is_physical, is_entangled], dim=-1)  # [E, 3]
        
        # Node attributes: [8 features]
        node_attr = torch.zeros((self.n, 8), dtype=torch.float)
        has_link = self.fidelities > 0
        
        # Feature 0: has any connection
        node_attr[:, 0] = has_link.any(dim=1).float()
        
        # Feature 1: has multiple connections (swap-eligible)
        node_attr[:, 1] = (has_link.sum(dim=1) > 1).float()
        
        # Feature 2: normalized physical degree
        physical_degree = self.adjacency.sum(dim=1)
        max_degree = physical_degree.max()
        if max_degree > 0:
            node_attr[:, 2] = physical_degree / max_degree
        
        # Feature 3: normalized entanglement degree
        entangle_degree = has_link.sum(dim=1)
        if max_degree > 0:
            node_attr[:, 3] = entangle_degree / max_degree
        
        # Feature 4: topology-specific position
        node_attr[:, 4] = self._get_position_feature()
        
        # Feature 5: max fidelity
        node_attr[:, 5] = self.fidelities.max(dim=1)[0]
        
        # Feature 6: average fidelity
        link_sums = self.fidelities.sum(dim=1)
        link_counts = (self.fidelities > 0).sum(dim=1).float().clamp(min=1)
        node_attr[:, 6] = link_sums / link_counts
        
        # Feature 7: distance to target (for chains, 0 otherwise)
        if self.topology == 'chain':
            distances = torch.arange(self.n - 1, -1, -1, dtype=torch.float)
            node_attr[:, 7] = distances / (self.n - 1) if self.n > 1 else distances
        
        # Create Data object
        data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
        data.physical_edges = self.physical_edges
        data.num_nodes = self.n
        
        return data
    
    def _get_position_feature(self):
        """Get topology-specific position feature"""
        if self.topology == 'chain':
            # Linear position [0, 1]
            return torch.arange(self.n, dtype=torch.float) / (self.n - 1) if self.n > 1 else torch.zeros(self.n)
        
        elif self.topology == 'star':
            # Binary: 1 for center, 0 for leaves
            physical_degree = self.adjacency.sum(dim=1)
            max_degree = physical_degree.max()
            return (physical_degree == max_degree).float()
        
        elif self.topology == 'ring':
            # Circular position
            return torch.arange(self.n, dtype=torch.float) / self.n
        
        elif self.topology == 'grid':
            # Manhattan distance from center
            import math
            side = int(math.sqrt(self.n))
            center = side // 2
            distances = []
            for i in range(self.n):
                row, col = i // side, i % side
                dist = abs(row - center) + abs(col - center)
                distances.append(dist)
            max_dist = max(distances)
            return torch.tensor(distances, dtype=torch.float) / max_dist if max_dist > 0 else torch.zeros(self.n)
        
        else:
            # Generic: use degree as proxy
            return self.adjacency.sum(dim=1) / (self.n - 1) if self.n > 1 else torch.zeros(self.n)


# Example usage:
if __name__ == "__main__":
    # Test different topologies
    topologies = {
        'chain_4': FlexibleRepeaterNetwork(n=4, topology='chain'),
        'ring_6': FlexibleRepeaterNetwork(n=6, topology='ring'),
        'star_5': FlexibleRepeaterNetwork(n=5, topology='star'),
        'grid_9': FlexibleRepeaterNetwork(n=9, topology='grid'),
    }
    
    for name, env in topologies.items():
        state = env.tensorState()
        print(f"\n{name}:")
        print(f"  Nodes: {state.num_nodes}")
        print(f"  Physical edges: {len(env.physical_edges)}")
        print(f"  Node features shape: {state.x.shape}")
        print(f"  Edge features shape: {state.edge_attr.shape}")