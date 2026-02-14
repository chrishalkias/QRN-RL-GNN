import random
from base.repeaters import RepeaterNetwork
import numpy as np

class Strategies():
    def __init__(self, network: RepeaterNetwork):
        """
        ## Strategies for entanglement distribution in quantum repeater chains.
        Different strategies are utilized, all with the following prioritization rule:

        For each repeater on the network: 
            if it has 2 active connections  -> append it to a `swaps` list
            elif it has 1 active connection -> append it to a `priority`
            else if it has 0 connection     -> append it to a `entanglement` list

        Then an action is chosen -based on another criterion- from the lists based on the prioritization
        rule swaps > priority > entanglement

        ### Usage:
            ```
            environment = RepeaterNetwork()
            strategy = Strategies()
            action_str = strategy.FN_swap()
            exec(action_str.replace("self.", "environment."))
            ```
        """

        self.network = network

    def stochastic_action(self) -> str:
        """
        Description:
            Creates list of all the possible actions that can be applied to the network
            and chooses one at random.

        Returns:
            action (str) - A random action
        """
        entangles = [f'self.entangle(({node}, {node+1}))' for node in range(self.network.n-1)]
        swaps = [f'self.swapAT({node})' for node in range(1, self.network.n-1)] 
        action = random.choice(entangles + swaps)
        return action

    def swap_asap(self) -> str:
        """
        Performs the random swap asap strategy with a random criteron for swaps.
        """
        swaps = []
        priority_entangles = []
        entangles = []

        state_x = self.network.tensorState().x

        for node in range(self.network.n):
            leftlink = state_x[node][0] > 0
            rightlink = state_x[node][1] > 0


            if leftlink and rightlink:
                swaps.append(f'self.swapAT({node})')


            elif bool(leftlink) ^ bool(rightlink):
                if leftlink and node != self.network.n - 1:
                    priority_entangles.append(f'self.entangle(({node}, {node+1}))')
                elif rightlink and node != 0:
                    priority_entangles.append(f'self.entangle(({node-1}, {node}))')


            elif not leftlink and not rightlink:
                if node != 0:
                    entangles.append(f'self.entangle(({node-1}, {node}))')
                if node != self.network.n - 1:
                    entangles.append(f'self.entangle(({node}, {node+1}))')

        if swaps:
            return random.choice(swaps)
        elif priority_entangles:
            return random.choice(priority_entangles)
        elif entangles:
            return random.choice(entangles)
        return RuntimeError('No actions')

    def FN_swap(self) -> str:
        """
        ## Farthest Neighbor Swap 
        
        Performs the FN Swap Asap based on the distance creiterion, i.e:

        if two swaps are available -> choose the one that would result in the farthest link

        ### Returns:
            action (str) -> The strategies next action
        """
        swaps = []
        priority_entangles = []
        entangles = []
        
        state_x = self.network.tensorState().x

        for node in range(self.network.n):
            leftlink = state_x[node][0] > 0
            rightlink = state_x[node][1] > 0

            if leftlink and rightlink:
                # We know a swap is possible, but we search neighbors to calculate DISTANCE
                n_left = None
                n_right = None

                # Search Left
                for i in range(node):
                    if self.network.getLink((i, node), 1) > 0:
                        n_left = i
                        break
                
                # Search Right (Backwards to find Farthest)
                for j in range(self.network.n - 1, node, -1):
                    if self.network.getLink((node, j), 1) > 0:
                        n_right = j
                        break
                
                if n_left is not None and n_right is not None:
                    dist = abs(n_right - n_left)
                    swaps.append((dist, f'self.swapAT({node})'))

            elif bool(leftlink) ^ bool(rightlink):
                if leftlink and node != self.network.n - 1:
                    priority_entangles.append(f'self.entangle(({node}, {node+1}))')
                elif rightlink and node != 0:
                    priority_entangles.append(f'self.entangle(({node-1}, {node}))')
            
            elif not leftlink and not rightlink:
                if node != 0:
                    entangles.append(f'self.entangle(({node-1}, {node}))')
                if node != self.network.n - 1:
                    entangles.append(f'self.entangle(({node}, {node+1}))')

        if swaps:
            swaps.sort(key=lambda x: x[0], reverse=True)
            return swaps[0][1]
        elif priority_entangles:
            return random.choice(priority_entangles)
        elif entangles:
            return random.choice(entangles)
        else:
            return RuntimeError('No actions')

    def SN_swap(self) -> list:
        """
        ## Strongest Neighbor Swap 
        
        Performs the SN Swap Asap based on the fidelity creiterion, i.e:

        if two swaps are available -> choose the one that would result in the strongest link
        The resulting link strength upon swapping F1 and F2 is defined as F_res = F1 * F2

        ### Returns:
            action (str) -> The next action
        """
        swaps = []
        priority_entangles = []
        entangles = []

        state_x = self.network.tensorState().x

        for node in range(self.network.n):
            leftlink = state_x[node][0] > 0
            rightlink = state_x[node][1] > 0

            if leftlink and rightlink:
                # We know swap is possible, search neighbors to calculate FIDELITY
                max_fid_left = 0.0
                max_fid_right = 0.0

                for i in range(node):
                    f = self.network.getLink((i, node), 1)
                    if f > max_fid_left:
                        max_fid_left = f

                for j in range(node + 1, self.network.n):
                    f = self.network.getLink((node, j), 1)
                    if f > max_fid_right:
                        max_fid_right = f
                
                if max_fid_left > 0 and max_fid_right > 0:
                    predicted_fidelity = (max_fid_left * max_fid_right)
                    swaps.append((predicted_fidelity, f'self.swapAT({node})'))

            elif bool(leftlink) ^ bool(rightlink):
                if leftlink and node != self.network.n - 1:
                    priority_entangles.append(f'self.entangle(({node}, {node+1}))')
                elif rightlink and node != 0:
                    priority_entangles.append(f'self.entangle(({node-1}, {node}))')
            
            elif not leftlink and not rightlink:
                if node != 0:
                    entangles.append(f'self.entangle(({node-1}, {node}))')
                if node != self.network.n - 1:
                    entangles.append(f'self.entangle(({node}, {node+1}))')

        if swaps:
            swaps.sort(key=lambda x: x[0], reverse=True)
            return swaps[0][1]
        elif priority_entangles:
            return random.choice(priority_entangles)
        elif entangles:
            return random.choice(entangles)
        return None

    def doubling_swap(self) -> list:
        """
        ## Doubling Swap 
        
        Performs the SN Swap Asap based on the doubling creiterion, i.e:

        Only swap two links if they are of equal length

        ### Returns:
            action (str) -> The strategies next action
        """
        swaps = []
        priority_entangles = []
        entangles = []
        
        state_x = self.network.tensorState().x

        for node in range(self.network.n):
            leftlink = (state_x[node][0] > 0).item()
            rightlink = (state_x[node][1] > 0).item()

            if leftlink and rightlink:
                len_left = 0
                len_right = 0

                for i in range(node - 1, -1, -1):
                    if self.network.getLink((i, node), 1) > 0:
                        len_left = node - i
                        break 

                for j in range(node + 1, self.network.n):
                    if self.network.getLink((node, j), 1) > 0:
                        len_right = j - node
                        break 

                if len_left > 0 and len_right > 0 and len_left == len_right:
                    swaps.append(f'self.swapAT({node})')

            elif leftlink ^ rightlink:
                if leftlink and node != self.network.n - 1:
                    priority_entangles.append(f'self.entangle(({node}, {node+1}))')
                elif rightlink and node != 0:
                    priority_entangles.append(f'self.entangle(({node-1}, {node}))')

            elif (not leftlink) and not rightlink:
                if node != 0:
                    entangles.append(f'self.entangle(({node-1}, {node}))')
                if node != self.network.n - 1:
                    entangles.append(f'self.entangle(({node}, {node+1}))')

        if swaps:
            return random.choice(swaps)
        elif priority_entangles:
            return random.choice(priority_entangles)
        elif entangles:
            return random.choice(entangles)
        else:
            raise RuntimeError('No available actions. SOmething broke')
        
    def frontier(self, cutoff: bool = False) -> str:
            """
            ## Frontier method

            One of the learned strategies of early agents following the recipe:
                1. Identifies the farthest node currently connected to node 0 (the frontier).
                2. Tries to entangle the next segment (frontier -> frontier+1).
                3. If that segment exists, it swaps at the frontier to extend the link.

            ### Returns:
                action (str) -> The next action

            TODO: Implement a cutoff so that if the resulting link would expire, start anew
            """
            
            # 1. Find the current frontier (farthest node connected to 0)
            frontier = 0
            # Check backwards from N-1 down to 1
            for k in range(self.network.n - 1, 0, -1):
                if self.network.getLink((0, k), 1) > 0:
                    frontier = k
                    break
            if frontier == self.network.n-1:
                return
            # The next node we need to connect to
            target = frontier + 1

            # 2. Check if the next small link segment exists
            # e.g., if we have (0,2), we check if (2,3) exists
            segment_exists = self.network.getLink((frontier, target), 1) > 0

            if not segment_exists:
                # Case A: The next segment is missing -> Create it
                return f'self.entangle(({frontier}, {target}))'
            else:
                # Case B: The next segment exists.
                # Since 'frontier' is connected to 0, and 'target' is connected to 'frontier',
                # we swap at 'frontier' to join them into (0, target).
                return f'self.swapAT({frontier})'