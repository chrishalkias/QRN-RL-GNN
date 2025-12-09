import random

class Heuristics():
    def __init__(self, network):
        """
        This class provides a bundle of the used heuristics.

        Methods:
            has_left_link()
            has_right_link()
            random_action()
            alternating_action()
            swap_asap()
        """
        self.network = network
        pass

    def has_left_link(self, node):
        return self.network.getLink(edge = (node-1,node), linkType=1) if node != 0 else -1
    def has_right_link(self, node):
        return self.network.getLink(edge = (node,node+1), linkType=1) if node != self.network.n-1 else -1



    def random_action(self) -> list:
        """Perform a random action at each node"""
        waits = ['' for _ in range(self.network.n)]
        entangles = [f'self.entangle({(i,i+1)})' for i in range(self.network.n-1)]
        swaps = [f'self.swapAT({i})' if (i != 0) and (i !=self.network.n-1) else '' for i in range(self.network.n)] # dont swap ad end nodes
        return [random.choice([e, s, w]) for e, s, w in zip(entangles, swaps, waits) if random.choice([e, s, w]) is not None]

    def alternating_action(self, step) -> list:
        """At even timestep entangle all and at odd swap all"""
        if (step % 2) == 0:
            return [f'self.entangle({(i,i+1)})' for i in range(self.network.n-1)]
        elif (step % 2) == 1:
            return [f'self.swapAT({i})' if (i != 0) and (i !=self.network.n-1) else '' for i in range(self.network.n)]

    def swap_asap(self) -> list:
        """Performs the swap asap"""
        actions = []

        for i in range(self.network.n):
            rightlink = self.has_right_link(i)
            leftlink = self.has_left_link(i)

            if leftlink > 0 and rightlink > 0:
                actions.append(f'self.swapAT({i})')
            elif leftlink == 0:
                actions.append(f'self.entangle(edge={(i-1,i)})')
            elif rightlink == 0:
                actions.append(f'self.entangle(edge={(i,i+1)})')
        return actions
  