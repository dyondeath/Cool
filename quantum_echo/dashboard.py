import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qutip import Bloch

from .core_simulator import quantum_eraser_sim
from .enhancements import run_mcmc
from .utils import load_data

import threading
import queue


def async_sim(trials, basis, phase, coupling, deco, result_queue):
    """Run simulation in background thread."""
    p_h, err = quantum_eraser_sim(trials, basis, phase, coupling, deco)
    result_queue.put((p_h, err))


@st.cache_resource
def run_real_time():
    result_queue = queue.Queue()
    thread = threading.Thread(
        target=async_sim,
        args=(
            st.session_state.trials,
            st.session_state.basis,
            st.session_state.phase,
            st.session_state.coupling,
            st.session_state.deco,
            result_queue,
        ),
    )
    thread.start()
    return result_queue, thread


def dashboard():
    st.title("Real-Time Quantum Echo Dashboard")

    # Session state defaults
    if "trials" not in st.session_state:
        st.session_state.trials = 1000
    if "basis" not in st.session_state:
        st.session_state.basis = 0
    if "phase" not in st.session_state:
        st.session_state.phase = 0.0
    if "coupling" not in st.session_state:
        st.session_state.coupling = 0.1
    if "deco" not in st.session_state:
        st.session_state.deco = 0.0001

    # Controls
    st.session_state.trials = st.slider("Trials", 100, 10000, st.session_state.trials)
    st.session_state.basis = st.selectbox("Basis", [0, 1, 2], index=st.session_state.basis)
    st.session_state.phase = st.slider("Phase", -0.2, 0.2, st.session_state.phase)
    st.session_state.coupling = st.slider("Coupling", 0.1, 0.8, st.session_state.coupling)
    st.session_state.deco = st.slider(
        "Decoherence", 0.0, 0.001, st.session_state.deco, step=1e-5
    )

    # Async simulation
    result_queue, thread = run_real_time()
    if thread.is_alive():
        st.write("Simulating...")
    else:
        p_h, err = result_queue.get()
        st.write(f"P(H): {p_h:.3f} ± {err:.3f}")

        fig_hist, ax = plt.subplots()
        ax.hist(p_h, bins=20)
        st.pyplot(fig_hist)

        bloch = Bloch()
        bloch.add_points([p_h.mean(), 0, err])
        st.pyplot(bloch.fig)

        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x) ** 2 * (1 - st.session_state.deco * 1000)
        fig_fringe, ax_f = plt.subplots()
        ax_f.plot(x, y)
        st.pyplot(fig_fringe)

    # Data upload and fitting
    uploaded = st.file_uploader("Upload CSV")
    if uploaded:
        data = load_data(uploaded)
        idata, summary = run_mcmc(data)
        st.write(summary)
