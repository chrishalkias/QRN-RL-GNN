import numpy as np

class Repeater:

    def __init__(self,
                n_channels: int = 4,
                cutoff=100,
                distillation = None,
                pe: float = 1.0,
                ps: float = 1.0,
                verbose:bool = False):
        """
        Tracks: entanglements, ages
        """
        
        self.tag = None
        self.n_channels = n_channels
        self.cutoff = cutoff
        self.distillation = distillation
        self.pe = pe
        self.ps = ps
        self.verbose = verbose
        self.fidelities = np.zeros(shape=self.n_channels, dtype='float16')
        self.ages = np.zeros_like(self.fidelities)
        if verbose: 
            print(f'Initializing repeater with:\n n: {n_channels} qubits \n pe: {pe} \n ps: {ps} \n cutoff: {cutoff} \n distill: {'yes' if distillation else 'no'}')

    def _set_tag(self, tag):
        """Set repeaters unique id tag"""
        self.tag = tag
        
    def preselect(self, qubit:int):
        self.ages[qubit] = 0
        self.fidelities[qubit] = 0.0
        if self.verbose: print(f'Set qubit {qubit} to 0')
        
    def _register(self, qubit:int, value:float):
        if self.fidelities[qubit] > 0:
            raise RuntimeError(f'Qubit at channel: {qubit} is occupied, run `preselect()` first')
        self.fidelities[qubit] = value
        if self.verbose: print(f'Registered qubit: {qubit} to value: {value}')

    def generate_link(self) -> int:
        """
        Generate entanglement by registering a link
        Searches for available qubits.
        Probabilisticly sets their state
        Returns:
            `-1` if EG failed
            `-2` if all qubits are occupied
            `qubit (int)` if EG was successful
        """
        if self.verbose: print('Attempting entanglement generation...')
        available_exist_ok = False

        for qubit in range(self.n_channels):

            if self.fidelities[qubit] == 0:
                available_exist_ok = True

                if self.pe < np.random.rand():
                    return -1
                
                self._register(qubit=qubit, value=1)
                break

        if available_exist_ok:
            return qubit
        
        else:
            return -2
    
    def perform_BSM(self, qubit:int) ->int:
        """
        Perform a bell state measurement for entanglement swapping.
        Preselects the selected qubit.

        Returns:
            0 if BSM failed
            1 if BSM succeded
        """
        self.preselect(qubit)
        if self.verbose: print(f'Performed BSM on qubit: {qubit}')
        return 0 if self.ps < np.random.rand() else 1
        
