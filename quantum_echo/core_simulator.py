import qutip as qt
import numpy as np


def quantum_eraser_sim(trials=1000, basis=0, phase_shift=0.0, coupling_strength=0.1, gravity_deco=0.0):
    """Core eraser simulation with optional gravity decoherence."""
    bell = qt.bell_state('00')
    bell = qt.tensor(bell, qt.basis(2, 0))
    p_h = np.zeros(trials)
    for i in range(trials):
        if basis == 0:
            op = qt.sigmaz().tensor(qt.sigmaz())
        else:
            op = qt.sigmax().tensor(qt.sigmax()) + phase_shift * coupling_strength * qt.sigmay().tensor(qt.sigmay())
        if gravity_deco:
            noise = qt.rand_dm_ginibre(bell.dims[0][0]) * gravity_deco
            state = (bell + noise).unit()
        else:
            state = bell
        meas = qt.expect(op, state)
        p_h[i] = (meas + 1) / 2
    return np.mean(p_h), np.std(p_h)
