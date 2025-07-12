import qutip as qt
import numpy as np


def quantum_eraser_sim(trials=1000, basis=0, phase_shift=0.0, coupling_strength=0.1, gravity_deco=0.0):
    """Fixed: Probabilistic sampling for measurements."""
    if trials <= 0:
        raise ValueError("Trials must be positive")
    
    bell = qt.bell_state('00')  # 2 qubits; no extra
    p_h = np.zeros(trials)
    for i in range(trials):
        state = bell.copy()
        if gravity_deco > 0:
            # Use rand_dm with correct dimensions for 2-qubit system
            noise = qt.rand_dm([2, 2]) * gravity_deco  # Match Bell state dimensions
            dm = state * state.dag() + noise
            dm = dm.unit()  # Normalize the density matrix
            # Convert back to state vector (approximate)
            eigenvals, eigenvecs = dm.eigenstates()
            # Get the dominant eigenstate
            max_idx = np.argmax(eigenvals)
            state = eigenvecs[max_idx]
        if basis == 0:  # Which-path: Z on first, I on second
            op = qt.tensor(qt.sigmaz(), qt.qeye(2))
        else:  # Eraser: X + phase Y on first, I on second
            op = qt.tensor(qt.sigmax() + phase_shift * coupling_strength * qt.sigmay(), qt.qeye(2))
        
        # Sample from the measurement distribution
        # Calculate probabilities for the two outcomes
        probs = [(1 + qt.expect(op, state)) / 2, (1 - qt.expect(op, state)) / 2]
        probs = [max(0, min(1, p)) for p in probs]  # Clamp to [0,1]
        # Normalize probabilities
        prob_sum = sum(probs)
        if prob_sum > 0:
            probs = [p / prob_sum for p in probs]
        else:
            probs = [0.5, 0.5]  # Default to equal probabilities
        
        # Sample from the distribution
        outcome = np.random.choice([1, 0], p=probs)
        p_h[i] = outcome
    return np.mean(p_h), np.std(p_h)
