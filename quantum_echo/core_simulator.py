import qutip as qt
import numpy as np
from qutip.measurement import measurement_statistics


def quantum_eraser_sim(trials=1000, basis=0, phase_shift=0.0, coupling_strength=0.1, gravity_deco=0.0):
    """Fixed: Probabilistic sampling for measurements."""
    # Input validation
    if trials <= 0:
        raise ValueError("Number of trials must be positive")
    if basis not in [0, 1, 2]:
        raise ValueError("Basis must be 0, 1, or 2")
    if coupling_strength <= 0:
        raise ValueError("Coupling strength must be positive")
    if gravity_deco < 0:
        raise ValueError("Gravity decoherence must be non-negative")
    
    bell = qt.bell_state('00')  # 2 qubits; no extra
    p_h = np.zeros(trials, dtype=float)  # Explicitly set dtype to float
    
    for i in range(trials):
        state = bell.copy()
        if gravity_deco > 0:
            # Use random_dm instead of rand_dm_ginibre (deprecated)
            noise = qt.random_dm(4) * gravity_deco  # dm size 4 (2q)
            dm = state * state.dag() + noise
            state = dm.unit().sqrtm()  # Back to ket approx
        
        if basis == 0:  # Which-path: Z on first, I on second
            op = qt.tensor(qt.sigmaz(), qt.qeye(2))
        elif basis == 1:  # Eraser: X on first, I on second
            op = qt.tensor(qt.sigmax(), qt.qeye(2))
        else:  # Eraser with phase: X + phase Y on first, I on second
            op = qt.tensor(qt.sigmax() + phase_shift * coupling_strength * qt.sigmay(), qt.qeye(2))
        
        # Sample measurement
        outcomes, probs, _ = measurement_statistics(state, op)
        # Ensure probs is a valid probability distribution
        probs = np.array(probs, dtype=float)
        probs = np.abs(probs)  # Take absolute values
        probs = probs / np.sum(probs)  # Normalize
        
        outcome = np.random.choice(outcomes, p=probs)
        p_h[i] = float((outcome + 1) / 2)  # P(+1), ensure float
    
    return np.mean(p_h), np.std(p_h)
