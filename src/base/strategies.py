import random

class Heuristics():
    def __init__(self, network):
        """
        This class provides a bundle of the used heuristics.
        All of the strategies have the following Markovian I/O:

        state -> [strategy] -> action

        2 * node -> entangle right at node
        2*node + 1 -> swap at node

        Methods:
            stochastic_action()
            swap_asap()
        """
        self.network = network
        pass


    
    def stochastic_action(self) -> list:
        """
        Perform a random action at each node
        """
        entangles = [f'self.entangle({node, node+1})' for node in range(self.network.n-1)]
        swaps = [f'self.swapAT({node})' for node in range(1, self.network.n-1)] # dont swap ad end nodes
        action = random.choice(entangles + swaps)
        return action



    def swap_asap(self):
        """
        Runs the random the swap-asap algorithm to determine the next action
        based on the giv.
        """
        swaps = []
        entangles = []

        for node in range(self.network.n):
            leftlink = self.network.tensorState().x[node][0]
            rightlink = self.network.tensorState().x[node][1]

            if leftlink and rightlink:
                swaps.append(f'self.swapAT({node})')
            else:
                if not leftlink and node!=0:
                    entangles.append(f'self.entangle({node-1, node})')
                if not rightlink and node!=self.network.n-1:
                    entangles.append(f'self.entangle({node, node+1})')
        if swaps:
            action = random.choice(swaps)
        elif entangles:
            action = random.choice(entangles)
        return action


  