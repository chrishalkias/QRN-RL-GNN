import torch, math

class GraphBuilder:
    
    def __init__(self):
        pass

    def _build_adjacency(self, n:int, topology:str, custom_adj=None):
        """Build adjacency matrix based on topology type"""
        self.n = n
        topology_map = {
            'chain': self._build_chain,
            'ring': self._build_ring,
            'star': self._build_star,
            'grid': self._build_grid,
            'tree': self._build_tree,
            'custom': self._custom_adj,} 
        if topology == 'custom' and custom_adj is None:
            raise ValueError("Must provide adjacency matrix for custom topology")
        if topology not in topology_map.keys():
            raise ValueError("No such topology, please choose from the list or use custom")
        
        return topology_map[topology]()
    
    def _custom_adj(custom_adj):
        return custom_adj       
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