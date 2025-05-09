# -*- coding: utf-8 -*-
# src/repeaters.py

'''
Created Thu 06 Mar 2025
The base simulation class, simulates the quantum network.
'''

import numpy as np
np.set_printoptions(legacy='1.25')
import itertools
import torch
from torch_geometric.data import Data

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

                         ██████  ██████  ███    ██ 
                        ██    ██ ██   ██ ████   ██ 
                        ██    ██ ██████  ██ ██  ██ 
                        ██ ▄▄ ██ ██   ██ ██  ██ ██ 
                         ██████  ██   ██ ██   ████ 
                            ▀▀                     
    Description:                       
      This class implements the Graph description of the repeater network.
      All of the information about the network is encoded in the edges of it. i.e
      we consider the nodes to be all zero. The graph description takes the form
      of a dictionary which is composed by an adjacency list as keys and the edges
      as values so: adj_list ~ (i,j) : [r_ij, E_ij].
      An adjecency matrix is built for n qubits and the links are initialized
      to zero by a zero edge matrix. Local 1D connections can be made with
      (geometry='chain'). More exotic Graphs will be implemented soon.
      The two main methods are entangle(edge) and swap(edge1, edge2).

    Methods:
      getLink()            > Get a link value for (i,j)
      setLink()            > Update a link value for (i,j) to V
      director()           > Remove ij-ji duplicates
      connect()            > Connects the Graph in specified geometry.
      tensorState()        > Returns a Graph state of the system
      tick()               > Propagates the system in time by dt = 1
      entangle()           > Entangles two repeaters (i,j) (E_ij->1)
      swap()               > Swaps E_(i,j)(j,k) -> E_(i,k)
      swapAT()             > Swaps at the specific node
      endToEndCheck()      > Measures to check if end-to-end-entangled

    Attributes:
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
    """
    self.n = n
    self.global_state = False
    (self.directed, self.geometry) = (directed, geometry)
    (self.time , self.tau, self.kappa, self.c) = (0, tau, kappa, 1)
    (self.p_entangle, self.p_swap) = (p_entangle, p_swap)
    self.combinations = np.array([[a,b] for a in range(n) for b in range(n)])
    self.matrix = {tuple(self.combinations[i]): [0, 0] for i in range(n**2)}
    self.undirector() if not directed else None
    self.connect(geometry = geometry)

  def tensorState(self):
    """Returns the tensor graph state (to be used for GNN)"""
    sources = torch.arange(self.n - 1, dtype=torch.long)  # 0, 1, ..., n-2
    targets = sources + 1                            # 1, 2, ..., n-1
    edge_index = torch.stack([sources, targets])     # Shape [2, n-1]
    edge_attr_list = [list(links)[1] for links in self.matrix.values()]
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    data = Data(x=torch.ones(self.n, 1), 
                edge_index=edge_index, 
                edge_attr = edge_attr)
    return data


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
    Args:
      geometry (str)        : The type of connectivity to be used
      p        (float)      : For ER only the probability of connection
      distList (array)      : A distance list for each node
    Outputs:
      Sets self.matrix to the correct adjeceny values and prints that the system
      has been initialized with the desired adjecency and directionality properties.
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
    """
    Check whether in grid and correct linkType (for getLink and setLink)
    Args:
      edge      (tuple) : The edge to be checked
      linkType  (bool)  : The linkType to be checked
    Returns:
      assert statements
    """
    leftBoundary  = edge[0] >= 0 and edge[1] >= 0
    rightBoundary = edge[0] <= self.n-1 and edge[1] <= self.n-1
    assert leftBoundary and rightBoundary, f'Edge {edge} out of bounds'
    assert (linkType == 0 or
            linkType == 1), f'Invalid link type (expected 0 or 1 got {linkType}'


  def getLink(self, edge:tuple, linkType:bool = 0):
    """Get the link (locality/entanglement) from self.matrix"""
    self.checkEdgeLink(edge=edge, linkType=linkType)
    return self.matrix[edge][linkType]


  def setLink(self, linkType:bool, edge:tuple, newValue:float):
    """Set the link value (only this and tick() allowed to change matrix)"""
    self.checkEdgeLink(edge=edge, linkType=linkType)
    self.matrix[edge][linkType] = newValue


  def resetState(self):
    """resets all entanglements to 0"""
    for edge in self.matrix.keys():
      self.setLink(edge=edge, linkType=1, newValue=0)


  def isSaturated(self, edge) -> bool:
    """
    Checks if node is already doubly entangled and therefore cannot be
    entangled with any more repeaters (used in self.entangle()).
    Args:
      edge (tuple) : The edge to be checked
    Outputs:
      full_saturation (bool)  : True iff the node is saturated
      pals            (tuple) : The links involved in the saturation
    """
    return False, [] #infinite saturation limit
    assert self.geometry =='chain', f'Only chain geometry supported'
    totalEntanglements, pals = 0, []

    for others in self.matrix.keys():
      isSelf = (others[0]==edge[0] and others[1]==edge[1])
      connectivity = (others[1] == edge[0] or others[0] == edge[1]) #chain
      if (not isSelf) and connectivity and (self.getLink(others,1) > 0.0):
        totalEntanglements +=1
        pals.append(others)

    assert totalEntanglements <= 2, f'{edge} contains polysaturated node'
    return totalEntanglements >= 2, pals


  def tick(self, T:int):
    """
    Implements the time evolution of the system:
    T timesteps ahead -> age all the links by T*dt (dt=1 by convention)
    Args:
      T (int)         : The number of timesteps to be evolved
    Returns:
      self.time (int) : The current time of the system
      updates the self.matrix with the new values
    """
    self.time += int(T)
    for key in self.matrix:
      i,j = key # Needs an extra r_ij here
      self.matrix[key][1] *= np.exp(-self.kappa * int(T) / (self.tau * self.c))


  def nodes(self) -> dict:
    """
    Maybe not usefull, another representation of the system instead of self.matrix.
    Returns:
      nodes (array) : Node description of the form [node, [partners], [ages]]
    """
    nodes = {a : [np.zeros((1,self.n)) for _ in range(2)] for a in range(self.n)}

    for keys, values in self.matrix.items():
      repeater_a, repeater_b = keys[0], keys[1]
      nodes_are_entangled = (values[1] > 0)

      if nodes_are_entangled:
        nodes[repeater_a][0][0][repeater_b] = 1
        nodes[repeater_b][0][0][repeater_a] = 1
        nodes[repeater_a][1][0][repeater_b] = values[1]
        nodes[repeater_b][1][0][repeater_a] = values[1]
    return nodes

 #-----------------------------ACTIONS---------------------------------------

  def entangle(self, edge):
    """
    Check if two nodes are adjecent and not saturated and
    entangle them with success probability p_entangle.
    Args:
      edge (tuple) : The edge to be entangled
    Returns:
      self.setLink
    """
    self.checkEdgeLink(edge=edge)

    self.tick(1)
    getsEntangled = self.p_entangle > np.random.rand()

    if not getsEntangled:
      return None

    if self.isSaturated(edge)[0]:
      linksInvolved  = self.isSaturated(edge)[1]
      saturationDict = {link: self.getLink(link, 1) for link in linksInvolved}
      oldestLink     = sorted(saturationDict.items(), key=lambda item: item[1])[0][0]
      #print(f'Edge {edge} contains saturated node, dropping oldest link')
      self.setLink(edge=oldestLink, linkType=1, newValue=0)

    areAdjecent = self.getLink(edge=edge, linkType=0)
    self.setLink(linkType = 1, edge=edge,newValue=1) if areAdjecent else None


  def swap(self, edge1, edge2):
    """
    Perform the SWAP operation between the qubits of edge1=(i,j)
    and edge2=(j,k) with probability p_swap. Swap sets the entanglement
     between (i,j) and (j,k) to 0 and the entanglement (i,k) equal to the
    average value of the two previous entanglements.
    Args:
      edge1 (tuple) : The first edge to be swapped
      edge2 (tuple) : The second edge to be swapped
    Returns:
      self.setLink
    """

    swapEficciency = 1
    self.checkEdgeLink(edge=edge1)
    self.checkEdgeLink(edge=edge2)
    self.tick(1)
    getsSwapped = self.p_swap > np.random.rand()

    if not getsSwapped:
      return None

    (i, j), (k, l) = edge1, edge2
    Eij            = self.getLink(edge=(i,j),linkType=1)
    Ejk            = self.getLink(edge=(k,l),linkType=1)
    effectiveValue = 0.5*swapEficciency*(Eij + Ejk) if (Eij>0 and Ejk>0) else 0

    assert j==k, f'Edges need to share a repeater, got {edge1, edge2} instead'

    self.setLink(linkType=1, edge=(i, j), newValue=0.0)
    self.setLink(linkType=1, edge=(k, l), newValue=0.0)
    self.setLink(linkType=1, edge=(i,l), newValue=effectiveValue)


  def swapAT(self, node): #chain only
    """
    Perform the swap operation by specifying a certain node i. Let the system
    choose which links get updated depending on the nodes j with which i is
    entangled with.
    Args:
      node    (int) : The node to swap
    Returns:
      set.Link
    """
    swapEficciency = 1
    assert node <= self.n-1, f'Node {node} not in system withn={self.n}'

    for i,j in self.matrix.keys():
      is_looped  = (i== node and j==node)
      is_ordered = (i < node < j) # for chain

      if (not is_looped) and is_ordered:
        link1, link2 = (i,node), (node,j)
        Eij = self.getLink(edge=link1, linkType=1)
        Ejk = self.getLink(edge=link2, linkType=1)
        effectiveValue = 0.5 * (Eij + Ejk) * swapEficciency
        if (Eij >0.0 and Ejk>0.0):
          self.setLink(linkType=1, edge=(i,j), newValue=effectiveValue)
          self.setLink(linkType=1, edge=link1, newValue=0.0)
          self.setLink(linkType=1, edge=link2, newValue=0.0)
        else:
          self.setLink(linkType=1, edge=(i,j), newValue=0.0)


  def actions(self, split = False) -> list:
    """
    Creates a dict() with all the possible actions
    Args:
      split   (bool) : choice to split the actions into entanglements and swaps
    Returns:
      actions (dict) : The dict of actions
    """
    entangles = {f'Entangle {key}':
                 f'self.entangle(edge={key})' for key in self.matrix.keys() if key[0]+1 == key[1]}
    edgeList  = list(itertools.combinations(self.matrix.keys(), 2))
    def find_edge_pairs():
      valid_pairs = []
      for (i, j) in self.matrix.keys():
          for (k, l) in self.matrix.keys():
              if i < j and k < l and j == k:
                  valid_pairs.append(((i, j), (k, l)))
      return valid_pairs
    edge_pairs = find_edge_pairs()
    #swaps = {f'swap{pair}': f'self.swap(edge1={pair[0]}, edge2={pair[1]})' for pair in edge_pairs}
    swapATs = {f'swapAT{node}': f'self.swapAT(node={node})' for node in range(self.n)}
    return np.array([*(entangles|swapATs).values()]) if not split else (entangles, swapATs)


  def globalActions(self) -> list:
    """
    Creates a list with all the possible global actions and then returns the
    global action (one operation per repeater) as dictated by the transformer.
    Args:
      transformer_output (list) : The output of the transformer [1, n]
    Returns:
      global_action (list) : The list of global actions [1,n]
    """
    actions         = [_ for _ in range(self.n)]
    neutral_element = '1' #neutral element can be changed to memory decay
    entangle_left   = lambda repeater: f'self.entangle({(repeater-1, repeater)})'
    entangle_right  = lambda repeater: f'self.entangle({(repeater, repeater+1)})'
    swap            = lambda repeater: f'self.swapAT({repeater})'

    for repeater in range(self.n):
      actions[repeater] = [neutral_element,   #no action
                           entangle_left(repeater),
                           entangle_right(repeater),
                           swap(repeater)
                           ]
      if repeater==0:          #left edge
        actions[repeater][1] = actions[repeater][2]
        actions[repeater][3] = neutral_element

      elif repeater==self.n-1: #right edge
        actions[repeater][2] = actions[repeater][1]
        actions[repeater][3] = neutral_element

    return np.array(actions)


  def actionCount(self) -> int:
    """Returns the number of possible actions"""
    return len(self.actions())


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
    return self.global_state
    # self.setLink(edge=(0,self.n-1), linkType=1, newValue = 0)