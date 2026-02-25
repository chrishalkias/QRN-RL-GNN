import numpy as np
import torch
from torch_geometric.data import Data

class RepeaterNetwork():
    def __init__(self,
                 n: int = 4,
                 cutoff = None,
                 tau: int = 1_000, # TODO replace tau with only cutoff
                 p_entangle: float = 1.0,
                 p_swap: float = 1.0
                ) -> None:
        """
        Fast Quantum network simulator
        TODO write a docstring (again)
        """
        # --- Initialize class attributes
        self.n = n
        self.tau, self.cutoff = tau, cutoff # TODO tau
        self.p_entangle, self.p_swap = p_entangle, p_swap
        
        # --- Tensor Representation 
        # Replaces the old self.matrix dictionary. Stores entanglements for all node pairs.
        self.fidelities = torch.zeros((self.n, self.n), dtype=torch.float)

    def checkEdgeLink(self, edge: tuple, linkType: int = 0):
        """Check whether in grid and correct linkType (for getLink and setLink)"""
        leftBoundary = edge[0] >= 0 and edge[1] >= 0
        rightBoundary = edge[0] <= self.n - 1 and edge[1] <= self.n - 1

        if not (leftBoundary and rightBoundary):
            raise IndexError(f'Edge {edge} out of bounds for chain of size {self.n}')
        if linkType not in (0, 1):
            raise ValueError(f'Invalid link type (expected 0 or 1 got {linkType})')

    def getLink(self, edge: tuple, linkType: int = 1) -> float:
        """Get the link (locality/entanglement) from the tensor"""
        self.checkEdgeLink(edge=edge, linkType=linkType)
        # Locality (linkType=0) is statically 1 for adjacent physical nodes 
        if linkType == 0:
            return 1.0 if abs(edge[0] - edge[1]) == 1 else 0.0
        
        return self.fidelities[edge[0], edge[1]].item()

    def setLink(self, edge: tuple, newValue: float, linkType: int = 1) -> None:
        """Set the link value (only this and tick() allowed to change matrix)"""
        self.checkEdgeLink(edge=edge, linkType=linkType)
        if linkType == 1:
            self.fidelities[edge[0], edge[1]] = newValue
            self.fidelities[edge[1], edge[0]] = newValue # Ensure undirected symmetry

    def reset(self) -> None:
        """resets all entanglements to 0"""
        self.fidelities.zero_()

    def saturated(self, node: int): #TODO fix this finally to enforce strictly 2 qubits per node
        """
        Enforces 2 qubits per repeater.
        Checks if node is already doubly entangled and therefore cannot be
        entangled with any more repeaters (used in self.entangle()).
        """
        return 42 # Placeholder logic 
        return int((self.fidelities[node, :] > 0).sum().item() +
               (self.fidelities[:, node] > 0).sum().item())

    def tick(self, T: int) -> None: # TODO implement the correct fidelity according to the pauli noise model
        """Implements the time evolution of the system using vectorized math"""
  
        self.fidelities *= np.exp(-T / self.tau) # TODO tau

        if self.cutoff is not None:
            self.fidelities[self.fidelities < np.exp(-self.cutoff / self.tau)] = 0.0 # TODO tau

    def entangle(self, edge: tuple) -> None:
        """
        Check if two nodes are adjacent and not saturated and
        entangle them with success probability p_entangle.
        """
        self.checkEdgeLink(edge=edge)
        getsEntangled = self.p_entangle > np.random.rand()
        (left_node, right_node) = edge

        if not getsEntangled:
            return
        
        # both repeaters are saturated
        if self.saturated(left_node) == 0 or self.saturated(right_node) == 0:
            return
        # qubits looking each other are occupied -> (x x---x x)
        if self.saturated(left_node) == +1 or self.saturated(left_node) == -1:
            return
        
        self.tick(1)
        areAdjacent = self.getLink(edge=edge, linkType=0)
        if areAdjacent:
            self.setLink(linkType=1, edge=edge, newValue=1.0)

    def swapAT(self, node: int) -> None:
        """Perform the swap operation on a node"""
        getsSwapped = self.p_swap > np.random.rand()

        if not getsSwapped:
            return 

        if node not in range(1, self.n - 1):
            raise ValueError(f'Node {node} not in system with n={self.n}')

        # Pytorch vectorized search for active links crossing the current node 
        # (This avoids the N^2 dictionary looping)
        left_links = torch.where(self.fidelities[:node, node] > 0)[0]
        right_links = torch.where(self.fidelities[node, node+1:] > 0)[0] + node + 1
        self.tick(1)
        for i in left_links.tolist():
            for j in right_links.tolist():
                Eij = self.fidelities[i, node].item()
                Ejk = self.fidelities[node, j].item()
                effectiveValue = (Eij * Ejk)
                
                self.setLink(linkType=1, edge=(i, node), newValue=0.0)
                self.setLink(linkType=1, edge=(node, j), newValue=0.0)
                self.setLink(linkType=1, edge=(i, j), newValue=effectiveValue)

    def endToEndCheck(self, timeToWait=0):
        """
        Check whether the graph is in an end-to-end entangled state.
        
        Returns:
            tuple: (success: bool, fidelity: float)
                - success: Whether end-to-end entanglement succeeded
                - fidelity: The fidelity value at time of check (before zeroing)
        """
        self.tick(timeToWait)
        
        # Read fidelity BEFORE probabilistic check and zeroing
        fidelity = self.fidelities[0, self.n-1].item()
        
        # Probabilistic success check
        endToEnd = (fidelity > np.random.rand()) # TODO have this match the Werner state fidelity
        self.global_state = endToEnd
        
        if endToEnd:
            # Zero out the link after recording fidelity
            self.setLink(edge=(0, self.n-1), linkType=1, newValue=0.0)
    
        return endToEnd, fidelity

    def tensorState(self) -> Data:
        """
        Enhanced tensorState with explicit, scalable position features.
        
        Node features:
        - [0] has_left: binary indicator of left connections
        - [1] has_right: binary indicator of right connections 
        
        Edge features (1 total):
        - [0] fidelity: link quality (no direction)
        """
        row, col = torch.meshgrid(torch.arange(self.n), torch.arange(self.n), indexing='ij')
        
        # --- Edges ---
        has_entanglement = self.fidelities > 0
        is_neighbor = torch.abs(row - col) == 1
        
        valid_mask = is_neighbor | has_entanglement
        edge_index = torch.nonzero(valid_mask).T 
        
        # Edge attributes: Fidelity only (no direction)
        edge_attr = self.fidelities[edge_index[0], edge_index[1]].view(-1, 1)
        
        # --- Node Features ---
        NODE_FEATURES = 2
        node_attr = torch.zeros((self.n, 2), dtype=torch.float)
        has_link = self.fidelities > 0
        
        # Features 0-1: Connection indicators
        node_attr[:, 0] = has_link.triu(1).any(dim=0).float()  # has_left
        node_attr[:, 1] = has_link.triu(1).any(dim=1).float()  # has_right
        
        return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    

    def entanglement_subroutine(self):
        """Global entanglement generation attempt"""
        for repeater in range(self.n-1):
            self.entangle(edge=(repeater, repeater+1))
