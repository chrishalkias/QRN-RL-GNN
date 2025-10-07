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
               kappa = 1,
               tau = 1_000,
               p_entangle = 1,
               p_swap = 1
              ):
    """
    **This class implements the Graph description of the repeater network**

    All of the information about the network is encoded in the edges of it. i.e
    we consider the nodes to be all zero. The graph description takes the form
    of a dictionary which is composed by an adjacency list as keys and the edges
    as values so: adj_list ~ (i,j) : [r_ij, E_ij].

    An adjecency matrix is built for n qubits and the links are initialized
    to zero by a zero edge matrix. Local 1D connections can be made with
    (geometry='chain'). More exotic Graphs will be implemented soon.

    The two main methods are entangle(edge) and swap(edge1, edge2).

    ------------------------------------------------------
    Methods:
    ------------------------------------------------------

    checkEdgeLink()      > Check whether in grid and correct linkType
    getLink()            > Get a link value for (i,j)
    setLink()            > Update a link value for (i,j) to V
    director()           > Remove ij-ji duplicates
    connect()            > Connects the Graph in specified geometry.
    tick()               > Propagates the system in time by dt = 1
    entangle()           > Entangles two repeaters (i,j) (E_ij->1)
    swap()               > Swaps E_(i,j)(j,k) -> E_(i,k)
    endToEndCheck()      > Measures to check if end-to-end-entangled

    --------------------------------------------------------
    Attributes:
    --------------------------------------------------------
    n_nodes     (int)    > Number of qubits
    directed    (bool)   > If the Graph is directed and looped
    global      (bool)   > 1 iff end-to-end entangled 0 therwise
    time        (int)    > Simulation time
    tau         (float)  > Link decay coefficient
    kappa       (float)  > Link decay coefficient
    p_entangle  (float)  > Probability of entanglement success
    p_swap      (float)  > Probability of swap success
    geometry    (str)    > The geometry of the network
    matrix      (array)  > Complete matrix representaition

    -------------------------------------------------------------
    Example usage (perform swap-asap on a n=4 chain ad measure):
    -------------------------------------------------------------

    net=RepeaterNetwork()               # Init to default values
    initialMatrix = net.matrix          # to compare later
    net.endToEndCheck()                 # starts disentangled
    print(net.global_state)             # check if False
    net.entangle(edge=(0,1))            # entangle (0,1)
    net.entangle(edge=(1,2))            # entangle (1,2)
    net.entangle(edge=(2,3))            # entangle (2,3)
    print(net.matrix)                   # check entanglements
    net.swap(edge1=(0,1), edge2=(1,2))  # swap (0,1) and (1,2)
    net.swap(edge1=(0,2), edge2=(2,3))  # swap (0,2) and (2,3)
    print(net.matrix)                   # check swap
    net.endToEndCheck()                 # win if true
    print(net.global_state)             # check if True
    g.reset()                           # reset all entanglement

    --------------------------------------------------------------
    For Reinforcement Learning:
    --------------------------------------------------------------

    1) The agent's action space is entangle((i,j)) or swap((i,j), (j,k))
    2) entanglementCheck() acts as an environment reset
    3) global_state can act as the reward (with some slight modifications)

    """
    self.n = n
    self.directed, self.geometry = directed, geometry
    self.global_state = False # Objective: make this into True
    self.time , self.tau, self.kappa, self.c = 0, tau, kappa, 1
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
    assert leftBoundary and rightBoundary, f'Edge {edge} out of bounds'
    assert (linkType == 0 or
            linkType == 1), f'Invalid link type (expected 0 or 1 got {linkType}'

  def getLink(self, edge:tuple, linkType:bool = 1):
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


  def isSaturated(self, edge): #chain only
    """
    Checks if node is already doubly entangled and therefore cannot be
    entangled with any more repeaters (used in self.entangle()).
    ------------
    Outputs:
    ------------
    full_saturation > bool  > True iff the node is saturated
    pals            > tuple > The links involved in the saturation
    ------------
    """
    return False, 0
    assert self.geometry =='chain', f'Only chain geometry supported'
    totalEntanglements, pals = 0, []
    for others in self.matrix.keys():
      isSelf = (others[0]==edge[0] and others[1]==edge[1])
      connectivity = (others[1] == edge[0] or others[0] == edge[1]) #chain(ij)(jk)
      if (not isSelf) and connectivity and (self.getLink(others) > 0.0):
        totalEntanglements +=1
        pals.append(others)

    assert totalEntanglements <= 2, f'{edge} contains polysaturated node'
    return totalEntanglements >= 2, pals


  def tick(self, T:int):
    """
    Implements the time evolution of the system:
    T timesteps ahead -> age all the links by T*dt (dt=1 by convention)
    """
    self.time += int(T)
    for key in self.matrix:
      i,j = key # Needs an extra r_ij here
      self.matrix[key][1] *= np.exp(-self.kappa * int(T) / (self.tau * self.c))



 #--------------------------------ACTIONS---------------------------------------



  def entangle(self, edge):
    """
    Check if two nodes are adjecent and not saturated and
    entangle them with success probability p_entangle.
    """
    self.checkEdgeLink(edge=edge)

    self.tick(1)
    getsEntangled = self.p_entangle > np.random.rand()

    if not getsEntangled:
      return None

    if self.isSaturated(edge)[0]:
      linksInvolved = self.isSaturated(edge)[1]
      saturationDict = {link: self.getLink(link) for link in linksInvolved}
      oldestLink = sorted(saturationDict.items(), key=lambda item: item[1])[0][0]
      #print(f'Edge {edge} contains saturated node, dropping oldest link')
      self.setLink(edge=oldestLink, linkType=1, newValue=0)

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


  def endToEndCheck(self):
    """
    Check wheather the graph is in an end-to-end entangled state by waitting
    a specified amount of time then reading the link ((0,n) in the chain case),
    change the global state of the graph to 1 and set the link back to 0
    """
    timeToWait = 5
    linkToRead = (0,self.n-1)
    self.tick(timeToWait)
    endToEnd = (self.getLink(edge=linkToRead, linkType=1) > np.random.rand())
    self.global_state = endToEnd
    # self.setLink(edge=(0,self.n-1), linkType=1, newValue = 0)
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
      operations.append(f'self.swapAT({node})') #if node not in [0,self.n-1]]
    return operations

  def actionCount(self):
    """Returns the number of possible actions"""
    return len(self.actions())

  def tensorState(self) -> Data:
      """Returns the tensor graph state (to be used for GNN)"""
      sources = torch.arange(self.n - 1, dtype=torch.long)  # 0, 1, ..., n-2
      targets = sources + 1                            # 1, 2, ..., n-1
      edge_index = torch.stack([sources, targets])     # Shape [2, n-1]
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