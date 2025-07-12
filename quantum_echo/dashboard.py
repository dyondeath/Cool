import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from .core_simulator import quantum_eraser_sim  # Async wrapper below
from .enhancements import run_mcmc
from .utils import load_data
from qutip import Bloch
import threading
import queue
import time
from rich.console import Console
from rich.logging import RichHandler
import logging

# Setup Rich logging
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger(__name__)

def async_sim(trials, basis, phase, coupling, deco, result_queue):
    """Async wrapper for quantum simulation."""
    logger.info(f"Starting async simulation with {trials} trials")
    p_h, err = quantum_eraser_sim(trials, basis, phase, coupling, deco)
    result_queue.put((p_h, err))
    logger.info(f"Simulation complete: P(H)={p_h:.3f}±{err:.3f}")

@st.cache_resource
def get_sim_queue():
    """Get a fresh queue for simulation results."""
    return queue.Queue()

def run_real_time_sim(trials, basis, phase, coupling, deco):
    """Run simulation in background thread."""
    result_queue = get_sim_queue()
    thread = threading.Thread(target=async_sim, args=(trials, basis, phase, coupling, deco, result_queue))
    thread.daemon = True
    thread.start()
    return result_queue, thread

def dashboard():
    st.title('Real-Time Quantum Echo Dashboard')
    
    # Initialize session state
    if 'trials' not in st.session_state: 
        st.session_state.trials = 1000
    if 'basis' not in st.session_state: 
        st.session_state.basis = 0
    if 'phase' not in st.session_state: 
        st.session_state.phase = 0.0
    if 'coupling' not in st.session_state: 
        st.session_state.coupling = 0.1
    if 'deco' not in st.session_state: 
        st.session_state.deco = 0.0001
    if 'sim_results' not in st.session_state:
        st.session_state.sim_results = None
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()

    # Parameter controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.trials = st.slider('Trials', 100, 10000, st.session_state.trials)
        st.session_state.basis = st.selectbox('Basis (0: Which-Path, 1/2: Eraser)', [0,1,2], index=st.session_state.basis)
        st.session_state.phase = st.slider('Phase Shift', -0.2, 0.2, st.session_state.phase)
    
    with col2:
        st.session_state.coupling = st.slider('Coupling Strength', 0.1, 0.8, st.session_state.coupling)
        st.session_state.deco = st.slider('Decoherence', 0.0, 0.001, st.session_state.deco, step=1e-5)

    # Auto-refresh simulation every 0.5 seconds
    if time.time() - st.session_state.last_update > 0.5:
        st.session_state.last_update = time.time()
        st.rerun()

    # Real-time simulation
    if st.button("Start Real-Time Simulation") or st.session_state.sim_results is None:
        result_queue, thread = run_real_time_sim(
            st.session_state.trials, 
            st.session_state.basis, 
            st.session_state.phase, 
            st.session_state.coupling, 
            st.session_state.deco
        )
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Wait for results with timeout
        timeout = 30  # 30 second timeout
        start_time = time.time()
        
        while thread.is_alive() and (time.time() - start_time) < timeout:
            elapsed = time.time() - start_time
            progress = min(elapsed / 10.0, 1.0)  # Assume 10s max simulation time
            progress_bar.progress(progress)
            status_text.text(f"Simulating... {elapsed:.1f}s")
            time.sleep(0.1)
        
        if not thread.is_alive():
            try:
                p_h, err = result_queue.get_nowait()
                st.session_state.sim_results = (p_h, err)
                progress_bar.progress(1.0)
                status_text.text("Simulation complete!")
            except queue.Empty:
                st.error("Simulation failed to return results")
        else:
            st.error("Simulation timed out")

    # Display results
    if st.session_state.sim_results is not None:
        p_h, err = st.session_state.sim_results
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("P(H)", f"{p_h:.3f}", f"±{err:.3f}")
        with col2:
            visibility = 1 - st.session_state.deco * 1000
            st.metric("Visibility", f"{visibility:.1%}")
        with col3:
            st.metric("Trials", st.session_state.trials)

        # Plots
        fig_container = st.container()
        
        with fig_container:
            col1, col2 = st.columns(2)
            
            with col1:
                # P(H) histogram (simulate streaming data)
                fig_hist, ax = plt.subplots(figsize=(6, 4))
                # Generate sample data for histogram
                hist_data = np.random.normal(p_h, err, 100)
                ax.hist(hist_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(p_h, color='red', linestyle='--', label=f'Mean: {p_h:.3f}')
                ax.set_xlabel('P(H)')
                ax.set_ylabel('Frequency')
                ax.set_title('P(H) Distribution')
                ax.legend()
                st.pyplot(fig_hist)
                plt.close()
                
            with col2:
                # Bloch sphere visualization
                fig_bloch, ax_bloch = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection='3d'))
                b = Bloch(fig=fig_bloch, axes=ax_bloch)
                # Add point based on measurement result
                theta = np.arccos(2 * p_h - 1)  # Convert P(H) to Bloch angle
                phi = st.session_state.phase * np.pi
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                b.add_points([x, y, z])
                b.render()
                st.pyplot(fig_bloch)
                plt.close()

        # Fringe pattern
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x + st.session_state.phase)**2 * (1 - st.session_state.deco*1000)
        fig_fringe, ax_f = plt.subplots(figsize=(10, 4))
        ax_f.plot(x, y, 'b-', linewidth=2)
        ax_f.set_xlabel('Phase')
        ax_f.set_ylabel('Intensity')
        ax_f.set_title(f'Interference Fringe (Visibility: {1 - st.session_state.deco*1000:.1%})')
        ax_f.grid(True, alpha=0.3)
        st.pyplot(fig_fringe)
        plt.close()

    # Data upload and MCMC fitting
    st.subheader("Data Analysis")
    uploaded_file = st.file_uploader("Upload CSV Data for MCMC Fitting", type=['csv'])
    
    if uploaded_file is not None:
        try:
            data = load_data(uploaded_file)
            st.write("Data loaded successfully!")
            st.write(f"Shape: {data.shape}")
            
            if st.button("Run MCMC Analysis"):
                with st.spinner("Running MCMC..."):
                    idata, summary = run_mcmc(data.flatten() if data.ndim > 1 else data)
                    st.subheader("MCMC Results")
                    st.text(summary.to_string())
                    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    # Rich logging output
    st.subheader("System Log")
    if st.button("Show Recent Logs"):
        st.text("Check terminal for Rich-formatted logs")
