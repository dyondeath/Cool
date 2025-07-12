import qutip as qt
import numpy as np


def quantum_eraser_sim(
    trials=1000,
    basis=0,
    phase_shift=0.0,
    coupling_strength=0.1,
    gravity_deco=0.0,
):
    """Core eraser simulation with optional gravity decoherence."""
    bell = qt.bell_state("00")
    p_h = np.zeros(trials)

    for i in range(trials):
        state = bell.copy()

        if gravity_deco > 0:
            noise = qt.rand_dm_ginibre(4) * gravity_deco
            dm = state * state.dag() + noise
            state = dm.unit().sqrtm()

        if basis == 0:
            op = qt.tensor(qt.sigmaz(), qt.qeye(2))
        else:
            op = qt.tensor(
                qt.sigmax() + phase_shift * coupling_strength * qt.sigmay(),
                qt.qeye(2),
            )

        meas = qt.expect(op, state)
        p_plus = (meas + 1) / 2
        p_h[i] = 1.0 if np.random.rand() < p_plus else 0.0

    return np.mean(p_h), np.std(p_h)
