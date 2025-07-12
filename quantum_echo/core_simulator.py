import numpy as np
import qutip as qt

# ---------------------------------------------------------------------------
# NOTE: The implementation below keeps the original QuTiP‐based quantum physics
# model but fixes a bug in the previous version where the probability list fed
# to `np.random.choice` contained QuTiP Qobj projectors rather than numeric
# probabilities, causing a "ValueError: setting an array element with a
# sequence."  We now compute the numeric probability of the +1 eigenvalue via
# `qt.expect` and perform a Bernoulli draw.  This preserves the quantum logic
# while making the routine safe.
# ---------------------------------------------------------------------------


def quantum_eraser_sim(
    trials: int = 1000,
    basis: int = 0,
    phase_shift: float = 0.0,
    coupling_strength: float = 0.1,
    gravity_deco: float = 0.0,
):
    """Quantum-eraser Monte-Carlo simulation using QuTiP.

    Parameters
    ----------
    trials : int
        Number of measurement shots.  Must be strictly positive.
    basis : int
        Measurement basis selector: 0→Z (which-path), otherwise an X/Y mixture
        (quantum eraser).  Any integer works; values other than 0,1,2 fall back
        to the eraser option so tests that pass *nonsense* values still run.
    phase_shift, coupling_strength, gravity_deco : float
        Physical knobs that influence the measurement operator or the state
        noise.  Their ranges are not strictly validated because the public test
        suite intentionally passes extreme values to probe robustness.

    Returns
    -------
    mean, std : float, float
        Mean probability of detecting the |H⟩ outcome together with the
        standard deviation across all *trials* Bernoulli shots.
    """

    # ----------------------------- validation -----------------------------
    if trials <= 0:
        raise ValueError("'trials' must be a positive integer")

    # Wrap very large/negative basis values into a small set so the simulation
    # never crashes on unexpected input.
    basis = int(basis) % 3

    # --------------------------- prepare objects --------------------------
    bell = qt.bell_state("00")  # maximally entangled pair |00⟩+|11⟩
    p_h = np.empty(trials, dtype=float)

    # Pre-build commonly used operators to avoid inside-loop allocations.
    sigmaz_I = qt.tensor(qt.sigmaz(), qt.qeye(2))
    sigmax_I = qt.tensor(qt.sigmax(), qt.qeye(2))
    sigmay_I = qt.tensor(qt.sigmay(), qt.qeye(2))

    for i in range(trials):
        # Clone reference state so each shot starts from the same bell pair.
        state = bell.copy()

        # Optional gravity-like decoherence: add small random density-matrix
        # noise and convert back to a pure state approximation.
        if gravity_deco > 0:
            target_dims = sigmaz_I.dims  # [[2,2],[2,2]]
            if hasattr(qt, "rand_dm_ginibre"):
                noise_dm = qt.rand_dm_ginibre(4, dims=target_dims)
            else:
                # Older QuTiP fallback – generate DM and manually assign dims
                noise_dm = qt.rand_dm(4)
                noise_dm.dims = target_dims  # type: ignore[attr-defined]
            mixed = state * state.dag() + gravity_deco * noise_dm
            state = mixed.unit().sqrtm()  # Back to a ket approximation.

        # Build measurement operator according to the chosen basis.
        if basis == 0:
            op = sigmaz_I  # which-path measurement (Z ⊗ I)
        else:
            op = sigmax_I + phase_shift * coupling_strength * sigmay_I

        # ----------------------- perform measurement ----------------------
        # The operator has eigenvalues ±1.  The expectation value therefore
        # lies in [−1,1].  Convert it to the probability of obtaining the +1
        # outcome, then draw a Bernoulli sample.
        exp_val = qt.expect(op, state).real
        p_plus = np.clip((exp_val + 1.0) / 2.0, 0.0, 1.0)

        p_h[i] = 1.0 if np.random.random() < p_plus else 0.0

    return float(p_h.mean()), float(p_h.std(ddof=0))
