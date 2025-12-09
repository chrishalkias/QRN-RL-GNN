import numpy as np
import itertools
import torch
from torch_geometric.data import Data
from typing import List, Tuple

class RepeaterNetwork():
  def __init__(self,
               n=4,
               directed = False,
               geometry = 'chain',
               cutoff = None,
               tau = 1_000,
               p_entangle = 1.0,
               p_swap = 1.0
              ):
    """
    Implements the Graph description of the repeater network

    ------------------------------------------------------
    Methods:
    ------------------------------------------------------
    undirector()         > Remove ij-ji duplicates
    connect()            > Connects the Graph in specified geometry.
    checkEdgeLink()      > Check whether in grid and correct linkType
    getLink()            > Get a link value for (i,j)
    setLink()            > Update a link value for (i,j) to V
    reset()              > reset the system to the ground state
    saturated()          > Checks if a repeater has >2 connections [NOT IMPLEMENTED]
    tick()               > Propagates the system in time by dt = 1
    entangle()           > Entangles two repeaters (i,j) (E_ij->1)
    swap()               > Swaps E_(i,j)(j,k) -> E_(i,k) [LEGACY]
    swapAT()             > Performs a swap at node k
    endToEndCheck()      > Measures to check if end-to-end-entangled
    old_actions()        > Previous action accounting [LEGACY]
    actions()            > A list of all the possible actions
    new_actions()        > yet another actions function......
    actionCount()        > Counts the number of all actions
    tensorState()        > The state description used for the GNN

    --------------------------------------------------------
    Attributes:
    --------------------------------------------------------
    n_nodes     (int)    > Number of qubits
    directed    (bool)   > If the Graph is directed and looped
    global      (bool)   > 1 iff end-to-end entangled 0 therwise
    time        (int)    > Simulation time
    tau         (float)  > Link decay coefficient
    cutoff      (float)  > The cutoff time
    p_entangle  (float)  > Probability of entanglement success
    p_swap      (float)  > Probability of swap success
    geometry    (str)    > The geometry of the network
    matrix      (array)  > Complete matrix representaition
    """
    self.n = n
    self.directed, self.geometry = directed, geometry
    self.global_state = False # Objective: make this into True
    self.time , self.tau, self.cutoff = 0, tau, cutoff
    self.p_entangle, self.p_swap = p_entangle, p_swap

    self.combinations = np.array([[a,b] for a in range(n) for b in range(n)])
    self.matrix = {tuple(self.combinations[i]): [0, 0] for i in range(n**2)}

    self.undirector() if not directed else None
    self.connect(geometry = geometry)


  def undirector(self):
    """Makes the graph directed and un-looped by modifying the state self.Matrix"""
    uniquePairs = set()
    for pair in self.combinations:
        uniquePairs.add(tuple(sorted(pair))) if pair[0] != pair[1] else None
    undirected_matrix = np.array(list(uniquePairs))
    combinations = range(len(undirected_matrix))
    edge_matrix =  np.zeros((self.n**2,1))
    edges = [edge_matrix[i][0] for i in combinations]
    self.matrix = {tuple(undirected_matrix[i]): [edges[i], edges[i]]  for i in combinations}

  def connect(self, geometry='chain', p=0.5, distList=None):
    """
    Connects the graph by creating locality links. The directionality
    of the graph is adjusted and afterwards locality links are filled
    out accorging to the input.
    -------------
    Args:
    -------------
    geometry        >str ('chain' , 'ER')   > The type of connectivity to be used
    p (float)       >float                  > For ER only the probability of connection
    distList [TODO] >np.array               > A distance list for each node
    --------------
    Outputs:
    --------------
    Sets self.matrix to the correct adjeceny values and prints that the system has
    been initialized with the desired adjecency and directionality properties.
    """
    assert not self.directed, f'Directed Graph'
    for key in self.matrix:
      i, j = key
      if geometry == 'chain':
        areNeighbours = (j==i+1)
      elif geometry == 'ER':
        areNeighbours = (np.random.rand() < p)
      else:
        raise Exception(f'Geometry {geometry} not supported')
      if areNeighbours:
        self.setLink(edge=(key), linkType = 0, newValue=1)

  def checkEdgeLink(self, edge:tuple, linkType:bool =0):
    """Check whether in grid and correct linkType (for getLink and setLink)"""
    leftBoundary = edge[0] >= 0 and edge[1] >= 0
    rightBoundary = edge[0] <= self.n-1 and edge[1] <= self.n-1
    assert leftBoundary and rightBoundary, f'Edge {edge} out of bounds for chain of size {self.n}'
    assert (linkType == 0 or
            linkType == 1), f'Invalid link type (expected 0 or 1 got {linkType}'

  def getLink(self, edge:tuple, linkType:bool = 1) -> float:
    """Get the link (locality/entanglement) from self.matrix"""
    self.checkEdgeLink(edge=edge, linkType=linkType)
    return self.matrix[edge][linkType]


  def setLink(self, linkType, edge:tuple, newValue:float):
    """Set the link value (only this and tick() allowed to change matrix)"""
    self.checkEdgeLink(edge=edge, linkType=linkType)
    self.matrix[edge][linkType] = newValue


  def reset(self):
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
    

  def tick(self, T:int):
    """
    Implements the time evolution of the system:
    T timesteps ahead -> age all the links by T*dt (dt=1 by convention)
    Expires the links according to their age. If a link is older than cutoff/tau it goes to zero
    """
    self.time += int(T)
    for key in self.matrix:
      i,j = key # Needs an extra r_ij here
      self.matrix[key][1] *= np.exp(-int(T) / self.tau)

    if self.cutoff != None:
      for key, (adj, ent) in self.matrix.items():
        if ent < np.exp(-self.cutoff / self.tau):
          self.matrix[(key)][1] = 0

 #--------------------------------ACTIONS---------------------------------------



  def entangle(self, edge):
    """
    Check if two nodes are adjecent and not saturated and
    entangle them with success probability p_entangle.
    """
    self.checkEdgeLink(edge=edge)

    self.tick(1)
    getsEntangled = self.p_entangle > np.random.rand()
    (left_node, right_node) = edge

    if not getsEntangled:
      return None
    
    # both repeaters are saturated
    if self.saturated(left_node) == 0 or self.saturated(right_node) == 0:
      return None
    # qubits looking each other are occupied -> (x x---x x)
    if self.saturated(left_node) == +1 or self.saturated(left_node) ==-1:
      return None

    areAdjecent = self.getLink(edge=edge, linkType=0)
    self.setLink(linkType = 1, edge=edge,newValue=1) if areAdjecent else None


  def swap(self, edge1, edge2): #chain only
    """
    Perform the SWAP operation between the qubits of edge1=(i,j)
    and edge2=(j,k) with probability p_swap. Swap sets the entanglement
     between (i,j) and (j,k) to 0 and the entanglement (i,k) equal to the
    average value of the two previous entanglements.
    """
    swapEficciency = 1
    self.checkEdgeLink(edge=edge1)
    self.checkEdgeLink(edge=edge2)
    self.tick(1)
    getsSwapped = self.p_swap > np.random.rand()

    if not getsSwapped:
      return None

    (i, j), (k, l) = edge1, edge2
    Eij= self.getLink(edge=(i,j))
    Ejk = self.getLink(edge=(k,l))
    effectiveValue = 0.5*swapEficciency*(Eij + Ejk) if (Eij>0 and Ejk>0) else 0

    assert j==k, f'Edges need to share a repeater, got {edge1, edge2} instead'

    self.setLink(linkType=1, edge=(i, j), newValue=0.0)
    self.setLink(linkType=1, edge=(k, l), newValue=0.0)
    self.setLink(linkType=1, edge=(i,l), newValue=effectiveValue)


  def swapAT(self, node): #chain only
    """Perform the swap operation by specifying a certain node i. Let the system
    choose which links get updated depending on the nodes {j} with which i is
    entangled with.
    """
    getsSwapped = self.p_swap > np.random.rand()

    if not getsSwapped:
      return None

    swapEficciency = 1
    assert node <= self.n-1, f'Node {node} not in system withn={self.n}'

    for i,j in self.matrix.keys():
      isnt_looped = (i!= node and j!=node)
      is_ordered = (i<node) and (node<j) # for chain

      if isnt_looped and is_ordered:
        has_links = (self.getLink((i,node))>0 and self.getLink((node,j))>0)

      if isnt_looped and is_ordered and has_links:
        link1, link2 = (i,node), (node,j)
        Eij, Ejk = self.getLink(link1), self.getLink(link2)
        effectiveValue = 0.5*swapEficciency*(Eij + Ejk)
        self.setLink(linkType=1, edge=link1, newValue=0.0)
        self.setLink(linkType=1, edge=link2, newValue=0.0)
        self.setLink(linkType=1, edge=(i,j), newValue=effectiveValue)


  def endToEndCheck(self, timeToWait=5):
    """
    Check wheather the graph is in an end-to-end entangled state by waitting
    a specified amount of time then reading the link ((0,n) in the chain case),
    change the global state of the graph to 1 and set the link back to 0
    """
    linkToRead = (0,self.n-1)
    self.tick(timeToWait)
    endToEnd = (self.getLink(edge=linkToRead, linkType=1) > np.random.rand()) #[TODO] change this to the Wehner fidelity (3/4?)
    self.global_state = endToEnd
    self.setLink(edge=(0,self.n-1), linkType=1, newValue = 0) if endToEnd else None
    return endToEnd


  def old_actions(self, split = False):
    """Creates a dict() with all the possible actions"""
    entangles= {f'Entangle {key}': f'self.entangle(edge={key})' for key in self.matrix.keys() if self.matrix[key][0]==1}
    edgeList = list(itertools.combinations(self.matrix.keys(), 2))
    print(edgeList)
    def noLoopChain(x, y):
      return (x[1] == y[0]) and (x[0]!=x[1]) and (y[0]!=y[1])
    combList = [(first, second) for first, second in edgeList if noLoopChain(first, second)]
    print(combList)
    swaps = {f'Swap {comb}': f'self.swap(edge1={comb[0]}, edge2={comb[1]})' for comb in combList}
    return np.array([*(entangles| swaps).values()]) if not split else (entangles, swaps)

  def actions(self, split = False):
    entangles= [f'self.entangle(edge={key})' for key in self.matrix.keys() if self.matrix[key][0]==1]
    swaps = [f'self.swapAT({node})' for node in range(self.n) if node not in [0,self.n]]
    return entangles + swaps

  def new_actions(self):
    operations = []
    for node in range(self.n-1):
      operations.append(f'self.entangle(edge={node, node+1})')
      operations.append(f'self.swapAT({node})') # if node not in [0,self.n-1]]
    return operations

  def actionCount(self):
    """Returns the number of possible actions"""
    return len(self.actions())

  def tensorState(self) -> Data:
      """Returns the tensor graph state (to be used for GNN)"""
      sources = torch.arange(self.n - 1, dtype=torch.long)  # 0, 1, ..., n-2
      targets = sources + 1                            # 1, 2, ..., n-1
      edge_index = torch.stack([sources, targets])     # shape [2, n-1]
      edge_attr_list = [list(links)[1] for links in self.matrix.values()]
      edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
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
      data = Data(x=node_attr,
                  edge_index=edge_index,
                  edge_attr = edge_attr)
      return data


  def shortestPath(self):
    # When graphs become more complex we'll need to find the shortest path first
    # https://www.geeksforgeeks.org/generate-graph-using-dictionary-python/
    pass