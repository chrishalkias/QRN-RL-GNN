import numpy as np
import torch
from torch_geometric.data import Data

class RepeaterNetwork():
  def __init__(self,
               n:int=4,
               cutoff = None,
               tau:int = 1_000,
               p_entangle:float = 1.0,
               p_swap:float = 1.0
              )-> None:
    """
        Fast Quantum network simulator
        [TODO] write a docstring (again)
    """
    #---Initialize class attributes
    self.n = n
    self.tau, self.cutoff = tau, cutoff
    self.p_entangle, self.p_swap = p_entangle, p_swap
    self.combinations = np.array([[a,b] for a in range(n) for b in range(n)])
    self.matrix = {tuple(self.combinations[i]): [0, 0] for i in range(n**2)}

    #---Make undirected graph
    uniquePairs = set()
    for pair in self.combinations:
        uniquePairs.add(tuple(sorted(pair))) if pair[0] != pair[1] else None
    undirected_matrix = np.array(list(uniquePairs))
    combinations = range(len(undirected_matrix))
    edge_matrix =  np.zeros((self.n**2,1))
    edges = [edge_matrix[i][0] for i in combinations]
    self.matrix = {tuple(undirected_matrix[i]): [edges[i], edges[i]]  for i in combinations}

    #---Set the linear network topology
    for key in self.matrix:
      node1, node2 = key
      areNeighbours = (node2==node1+1)
      if areNeighbours:
        self.setLink(edge=(key), linkType = 0, newValue=1)


  def checkEdgeLink(self, edge:tuple, linkType:bool =0):
    """Check whether in grid and correct linkType (for getLink and setLink)"""
    leftBoundary = edge[0] >= 0 and edge[1] >= 0
    rightBoundary = edge[0] <= self.n-1 and edge[1] <= self.n-1

    if not (leftBoundary and rightBoundary):
      raise IndexError(f'Edge {edge} out of bounds for chain of size {self.n}')
    if linkType not in (0,1):
      raise ValueError(f'Invalid link type (expected 0 or 1 got {linkType}')


  def getLink(self, edge:tuple, linkType:bool = 1) -> float:
    """Get the link (locality/entanglement) from self.matrix"""
    self.checkEdgeLink(edge=edge, linkType=linkType)
    return self.matrix[edge][linkType]


  def setLink(self, edge:tuple, newValue:float, linkType=1) -> None:
    """Set the link value (only this and tick() allowed to change matrix)"""
    self.checkEdgeLink(edge=edge, linkType=linkType)
    self.matrix[edge][linkType] = newValue


  def reset(self) -> None:
    """resets all entanglements to 0"""
    for edge in self.matrix.keys():
      self.setLink(edge=edge, linkType=1, newValue=0)


  def saturated(self, node:int): #chain only
    """
    Enforces 2 qubits per repeater.
    Checks if node is already doubly entangled and therefore cannot be
    entangled with any more repeaters (used in self.entangle()).
    ------------
    Returns:
    ------------
    0  : node is fully saturated
    -1 : left qubit is connected
    +1 : right qubit is connected
    ------------
    """
    return 42

    has_left_connection = (self.tensorState().x[node][0] > 0).item()
    has_right_connection = (self.tensorState().x[node][1] > 0).item()
    
    if has_left_connection and has_right_connection:
      return 0
    elif has_left_connection and (not has_right_connection):
      return -1
    elif has_right_connection and (not has_left_connection):
      return 1
    

  def tick(self, T:int) -> None:
    """Implements the time evolution of the system"""
    for key in self.matrix: #age all by 1 time unit
      self.matrix[key][1] *= np.exp(-1 / self.tau)

    if self.cutoff != None: #if age>cutoff kill
      for key, (adj, ent) in self.matrix.items():
        if ent < np.exp(-self.cutoff / self.tau):
          self.matrix[(key)][1] = 0

  def entangle(self:tuple, edge) -> None:
    """
    Check if two nodes are adjecent and not saturated and
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
    if self.saturated(left_node) == +1 or self.saturated(left_node) ==-1:
      return
    
    self.tick(1)
    areAdjecent = self.getLink(edge=edge, linkType=0)
    self.setLink(linkType = 1, edge=edge,newValue=1) if areAdjecent else None


  def swapAT(self, node:int) -> None:
    """Perform the swap operation on a node"""
    getsSwapped = self.p_swap > np.random.rand()

    if not getsSwapped:
      return 

    if node not in range(1, self.n-1):
      raise ValueError(f'Node {node} not in system withn={self.n}')

    for i,j in self.matrix.keys():
      isnt_looped = (i!= node and j!=node)
      is_ordered = (i<node) and (node<j) 

      if isnt_looped and is_ordered:
        has_links = (self.getLink((i,node))>0 and self.getLink((node,j))>0)

      if isnt_looped and is_ordered and has_links:
        link1, link2 = (i,node), (node,j)
        Eij, Ejk = self.getLink(link1), self.getLink(link2)
        effectiveValue = (Eij * Ejk)
        self.setLink(linkType=1, edge=link1, newValue=0.0)
        self.setLink(linkType=1, edge=link2, newValue=0.0)
        self.setLink(linkType=1, edge=(i,j), newValue=effectiveValue)


  def endToEndCheck(self, timeToWait=5):
    """Check wheather the graph is in an end-to-end entangled state"""
    linkToRead = (0,self.n-1)
    self.tick(timeToWait)
    endToEnd = (self.getLink(edge=linkToRead, linkType=1) > np.random.rand()) #[TODO] change this to the Wehner fidelity (3/4?)
    self.global_state = endToEnd
    self.setLink(edge=(0,self.n-1), linkType=1, newValue = 0) if endToEnd else None
    return endToEnd


  def tensorState(self) -> Data:
    """Returns the tensor graph state (to be used for GNN)"""
    sources = torch.arange(self.n - 1, dtype=torch.long)  # 0, 1, ..., n-2
    targets = sources + 1                            # 1, 2, ..., n-1

    edge_index = torch.stack([
          torch.cat([sources, targets]),
          torch.cat([targets, sources])])     # shape [2, n-1]
    
    edge_attr_list = []
    # Loop over the edges defined in edge_index
    for i in range(edge_index.shape[1]):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        # Retrieve the specific link from the matrix
        # Note: self.matrix keys are tuples, might need sorting if keys are undirected
        key = tuple(sorted((u, v))) 
        edge_attr_list.append(self.matrix[key][1])
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    edge_attr = edge_attr.view(-1,1) #GNN expects [num_edges, feature_dim]
    node_attr = torch.zeros((self.n, 2))

    for n1 in range(self.n):
        for n2 in range(self.n):
            if n2 < n1:
                if (self.getLink((n2, n1), 1) > 0):
                    node_attr[n2, 1] = 1
                    node_attr[n1, 0] = 1
            elif n1 < n2:
                if (self.getLink((n1, n2), 1) > 0):
                    node_attr[n1, 1] = 1
                    node_attr[n2, 0] = 1

    return Data(x=node_attr, edge_index=edge_index,edge_attr=edge_attr)