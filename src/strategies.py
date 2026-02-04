import random

class Heuristics():
    def __init__(self, network):
        """
        This class provides a bundle of the used heuristics.
        All of the strategies have the following Markovian I/O:

        state -> [strategy] -> action

        Methods:
            stochastic_action()
            alternating_action()
            swap_asap()
        """
        self.network = network
        pass


    
    def stochastic_action(self) -> list:
        """
        Perform a random action at each node
        """
        waits = ['' for _ in range(self.network.n)]
        entangles = [f'self.entangle({(i,i+1)})' for i in range(self.network.n-1)]
        swaps = [f'self.swapAT({i})' if (i != 0) and (i !=self.network.n-1) else '' for i in range(self.network.n)] # dont swap ad end nodes
        return random.choice([random.choice([e, s, w]) for e, s, w in zip(entangles, swaps, waits) if random.choice([e, s, w]) is not None])

    def alternating_action(self, step) -> list:
        """
        At even timestep entangle all and at odd swap all
        """
        if (step % 2) == 0:
            return [f'self.entangle({(i,i+1)})' for i in range(self.network.n-1)]
        elif (step % 2) == 1:
            return [f'self.swapAT({i})' if (i != 0) and (i !=self.network.n-1) else '' for i in range(self.network.n)]


    def swap_asap(self, variant='random'):
        """
        Performs (a variant of) the swap-asap strategy.
        """
        if variant not in ['random', 'FN', 'SN']:
            raise ValueError('Variant not supported')
        swaps = []
        entangles = []

        for node in range(self.network.n):
            leftlink = self.network.tensorState().x[node][0]
            rightlink = self.network.tensorState().x[node][1]
            # check swaps
            if leftlink and rightlink:
                swaps.append(f'self.swapAT({node})')
            # check entanglements
            else:
                if not leftlink and node!=0:
                    entangles.append(f'self.entangle(edge={(node-1,node)})')
                if not rightlink and node!=self.network.n-1:
                    entangles.append(f'self.entangle(edge={(node,node+1)})')
        if swaps:
            if variant == 'random':
                action = random.choice(swaps)
            else:
                raise ValueError('Only supported variant is random swap asap')
        elif entangles:
            action = random.choice(entangles)
        else:
            raise ValueError('No available actions')
        return action


  