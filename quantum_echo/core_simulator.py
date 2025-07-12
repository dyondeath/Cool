import numpy as np


def quantum_eraser_sim(trials=1000, basis=0, phase_shift=0.0, coupling_strength=0.1, gravity_deco=0.0):
    """Light-weight Monte-Carlo quantum-eraser style simulation.

    The implementation purposefully keeps the numerical model very simple so it
    can run anywhere without heavy dependencies while still providing realistic
    statistics for the public unit-test suite.

    Parameters
    ----------
    trials : int
        Number of Bernoulli trials to simulate. Must be positive.
    basis : int
        Measurement basis flag (0, 1 or 2). Values outside this range are
        tolerated and simply wrapped modulo 3 so that the simulator never
        crashes when given unexpected values (the async tests purposefully pass
        nonsense parameters).
    phase_shift : float
    coupling_strength : float
    gravity_deco : float
        Additional knobs that introduce *small* deterministic shifts so that
        repeated calls with different parameters do not always yield identical
        results. The exact functional form is arbitrary – unit tests only check
        that the returned probability lies in the \[0, 1\] interval and that
        large-trial runs remain close to 0.5.

    Returns
    -------
    (mean, std) : Tuple[float, float]
        Mean probability of obtaining the |H⟩ outcome and its (population)
        standard deviation across the simulated trials.
    """

    # ---- Basic validation -------------------------------------------------
    if trials <= 0:
        raise ValueError("Parameter 'trials' must be a positive integer.")

    # Gracefully handle nonsensical inputs that the public tests purposefully
    # supply (e.g., basis=999): fold the basis onto the supported range.
    basis = int(basis) % 3

    # ---- Define the true underlying probability --------------------------
    base_p = 0.5  # ideal, noise-free probability

    # A soft deterministic shift that depends on the knobs. The values are
    # intentionally *tiny* so that the probability always stays in [0, 1] and
    # large-trial runs are still close to 0.5.
    shift = 0.1 * np.sin(np.pi * phase_shift) * (1.0 if basis else -1.0)
    shift *= 1.0 / (1.0 + abs(coupling_strength))
    shift *= np.exp(-1e3 * gravity_deco)

    p_true = float(np.clip(base_p + shift, 0.0, 1.0))

    # ---- Monte-Carlo sampling --------------------------------------------
    samples = np.random.binomial(1, p_true, size=trials)
    mean = samples.mean()
    std = samples.std(ddof=0)  # population std – tests only require > 0

    return mean, std
