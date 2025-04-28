# Qonnect
    This is *Qonnect* a simple tabular game that is played according to the rules of quantum repeater networks.

## Game description
    Every edge (i,j) is represented as a square in a grid of size $N^2$ and every entanglement is represented with a dot on the associated square. The goal of the game is to have a dot on the square corresponding to the target end-to-end link $(i_A, i_B)$. The two kinds of operations that can be performed (Entanglement and Swaping) are associated with the creation and merging of dots in the game. Importantly since the system is represented by an undirected graph the game has a discrete reflection symmetry on the diagonal which turns out to be crucial for playing the game.

## Game rules
    The game has two permissible kinds of operations corresponding to entanglement generation and entanglement swaping
### Entangling:
    You can entangle two repeaters if they are adjecent to each other, this would make their corresponding square grey. by clicking on the square one dot is placed there. For a quantum repeater chain the first off diagonal corresponds to the permissible entanglement generation squares.
## Swaping:
    You can choose to perform entanglement swaping by merging two dots into one. To do so select the nodes to be merged and the resulting merged dot will be placed to the corresponding square according to the rules of the swap. These are the following: if two nodes share a repeater then the resulting merged dot will be placed to the square corresponding to the non-shared repeaters. There are 4 possible combinations of merges, these are given below:
    (i,j)(j,k) -> (i,k) <- (i,j)(k,j)
    (j,i)(j,k) -> (i,k) <- (j,i)(k,j)


## Known bugs
    There are a number of known bugs in the game and they are all related to the swaping operation.
<!--
add the known bugs
-->

## Instalation

```
git clone https://github.com/chrishalkias/QRN-RL-GNN/game
```