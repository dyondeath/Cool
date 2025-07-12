import qutip as qt
import numpy as np
from qutip.measurement import measurement_statistics


def quantum_eraser_sim(trials=1000, basis=0, phase_shift=0.0, coupling_strength=0.1, gravity_deco=0.0):
    """Fixed: Probabilistic sampling for measurements."""
    bell = qt.bell_state('00')  # 2 qubits; no extra
    p_h = np.zeros(trials)
    for i in range(trials):
        state = bell.copy()
        if gravity_deco > 0:
            noise = qt.rand_dm_ginibre(4) * gravity_deco  # dm size 4 (2q)
            dm = state * state.dag() + noise
            state = dm.unit().sqrtm()  # Back to ket approx
        if basis == 0:  # Which-path: Z on first, I on second
            op = qt.tensor(qt.sigmaz(), qt.qeye(2))
        else:  # Eraser: X + phase Y on first, I on second
            op = qt.tensor(qt.sigmax() + phase_shift * coupling_strength * qt.sigmay(), qt.qeye(2))
        # Sample measurement
        outcomes, probs, _ = measurement_statistics(state, op)
        outcome = np.random.choice(outcomes, p=probs)
        p_h[i] = (outcome + 1) / 2  # P(+1)
    return np.mean(p_h), np.std(p_h)
