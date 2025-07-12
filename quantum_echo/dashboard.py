import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qutip import Bloch

from .core_simulator import quantum_eraser_sim
from .enhancements import run_mcmc
from .utils import load_data, fit_mcmc_proxy


@st.cache_data
def run_and_plot(trials, basis, phase, coupling, deco):
    p_h, err = quantum_eraser_sim(trials, basis, phase, coupling, deco)
    fig_hist, ax = plt.subplots()
    ax.hist(p_h, bins=20)
    ax.set_title(f'P(H): {np.mean(p_h):.3f} ± {err:.3f}')
    st.pyplot(fig_hist)

    b = Bloch()
    b.add_points([p_h.mean(), 0, err])
    b.render()
    st.pyplot(b.fig)

    x = np.linspace(0, np.pi*2, 100)
    y = np.sin(x)**2 * (1 - deco*10)
    fig_fringe, ax_f = plt.subplots()
    ax_f.plot(x, y)
    ax_f.set_title(f'Visibility: ~{1 - deco*10:.1%}')
    st.pyplot(fig_fringe)

    return p_h, err


def dashboard():
    st.title('Real-Time Quantum Echo Dashboard')
    trials = st.slider('Trials', 100, 10000, 1000)
    basis = st.selectbox('Basis (0: Which-Path, 1/2: Eraser)', [0,1,2])
    phase = st.slider('Phase Shift', -0.2, 0.2, 0.0)
    coupling = st.slider('Coupling Strength', 0.1, 0.8, 0.1)
    deco = st.slider('Gravity Decoherence', 0.0, 0.001, 0.0001, step=1e-5)

    p_h, err = run_and_plot(trials, basis, phase, coupling, deco)

    uploaded_file = st.file_uploader('Upload CSV Data')
    if uploaded_file:
        data = load_data(uploaded_file)
        st.write(fit_mcmc_proxy(data))
        idata, summary = run_mcmc(data[:,1])
        st.write(summary)
