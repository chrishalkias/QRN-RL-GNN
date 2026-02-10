import random

class Strategies():
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
    


    def FN_swap(self):
        """
        Farthest Neighbor Swap:
        Prioritizes swaps that create the longest active link (max distance |left - right|).
        """
        swaps = []
        entangles = []
        

        for node in range(self.network.n):
            leftlink = self.network.tensorState().x[node][0]
            rightlink = self.network.tensorState().x[node][1]

            if leftlink and rightlink:
                n_left = None
                n_right = None
                
                # Search for left link (i < node)
                for i in range(node):
                    if self.network.getLink((i, node), 1) > 0:
                        n_left = i
                        break
                
                # Search for right link (j > node)
                for j in range(node + 1, self.network.n):
                    if self.network.getLink((node, j), 1) > 0:
                        n_right = j
                        break
                
                if n_left is not None and n_right is not None:
                    dist = abs(n_right - n_left)
                    action = f'self.swapAT({node})'
                    swaps.append((dist, action))

            else:
                if not leftlink and node != 0:
                    entangles.append(f'self.entangle({node-1, node})')
                if not rightlink and node != self.network.n - 1:
                    entangles.append(f'self.entangle({node, node+1})')


        if swaps:
            swaps.sort(key=lambda x: x[0], reverse=True)
            return swaps[0][1] # Return the action string
        
        elif entangles:
            return random.choice(entangles)
        
        return None

    def SN_swap(self):
        """
        Strongest Neighbor Swap:
        Prioritizes swaps that result in the highest fidelity link.
        It scans ALL connections to finding the strongest candidates on both sides.
        """
        swaps = []
        entangles = []

        for node in range(self.network.n):
            leftlink = self.network.tensorState().x[node][0]
            rightlink = self.network.tensorState().x[node][1]

            if leftlink and rightlink:
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
                    # Calculate expected fidelity: 0.5 * (F_left + F_right)
                    predicted_fidelity = 0.5 * (max_fid_left + max_fid_right)
                    action = f'self.swapAT({node})'
                    swaps.append((predicted_fidelity, action))

            else:
                if not leftlink and node != 0:
                    entangles.append(f'self.entangle({node-1, node})')
                if not rightlink and node != self.network.n - 1:
                    entangles.append(f'self.entangle({node, node+1})')

        if swaps:
            # Sort by fidelity descending (greedy for fidelity)
            swaps.sort(key=lambda x: x[0], reverse=True)
            return swaps[0][1]
        
        elif entangles:
            return random.choice(entangles)
            
        return None


    def doubling_swap(self):
            """
            Doubling Strategy:
            Only performs a swap at a node if the link to the left and the link to the right
            are of the exact same length.
            """
            swaps = []
            entangles = []

            for node in range(self.network.n):
                left_connected = self.network.tensorState().x[node][0] > 0
                right_connected = self.network.tensorState().x[node][1] > 0

                if left_connected and right_connected:
                    len_left = 0
                    len_right = 0

                    # iterate backwards from node-1 to 0
                    for i in range(node - 1, -1, -1):
                        if self.network.getLink((i, node), 1) > 0:
                            len_left = node - i
                            break 

                    # iterate forwards
                    for j in range(node + 1, self.network.n):
                        if self.network.getLink((node, j), 1) > 0:
                            len_right = j - node
                            break 

                    #The Doubling Condition
                    if len_left > 0 and len_right > 0 and len_left == len_right:
                        swaps.append(f'self.swapAT({node})')

                else:
                    if not left_connected and node != 0:
                        entangles.append(f'self.entangle({node-1, node})')
                    if not right_connected and node != self.network.n - 1:
                        entangles.append(f'self.entangle({node, node+1})')


            if swaps:
                return random.choice(swaps)
            elif entangles:
                return random.choice(entangles)
            
            return

  