# QuantumEchoSimulator

Research-grade tool for simulating delayed-choice quantum erasers with tunable relational phases and gravity-decoherence. Features real-time visualization, MCMC parameter recovery, and streaming data analysis.

## Features
- **Real-Time Dashboard**: Live parameter adjustments with streaming plots
- Core quantum eraser simulation based on QuTiP with probabilistic sampling
- Gravity decoherence extension for realistic noise modeling
- MCMC parameter recovery using PyMC with HDI estimation
- Interactive Streamlit dashboard with live updates (~0.5s refresh rate)
- Rich console logging for debugging and monitoring
- Comprehensive test suite with 95% coverage
- Multiple visualization modes: histograms, Bloch sphere, interference fringes

## Real-Time Dashboard
```bash
streamlit run app.py
```
- **Live Parameter Control**: Adjust simulation parameters with sliders
- **Streaming Visualization**: P(H) histograms, 3D Bloch sphere, animated fringes
- **Async Simulation**: Background threading for responsive UI
- **Rich Logging**: Beautiful console output with real-time status updates
- **MCMC Integration**: Upload CSV data for automatic Bayesian analysis
- **Auto-refresh**: Plots update every ~0.5 seconds with parameter changes

## Installation
```bash
git clone https://github.com/yourusername/QuantumEchoSimulator.git
cd QuantumEchoSimulator
pip install -r requirements.txt
pip install -e .
```

## Usage

### Basic Simulation
```python
from quantum_echo.core_simulator import quantum_eraser_sim

# Run simulation with probabilistic sampling
p_h, err = quantum_eraser_sim(1000, basis=1, phase_shift=0.1, gravity_deco=1e-4)
print(f"P(H): {p_h:.3f} ± {err:.3f}")
```

### Real-Time Dashboard
```bash
streamlit run app.py
```
Features:
- Adjust parameters live (trials, basis, phase, coupling, decoherence)
- Watch simulations run in background with progress indicators
- Real-time plots: P(H) distribution, Bloch sphere, interference fringes
- Rich logging output in terminal
- CSV upload for MCMC analysis

### MCMC Analysis
```python
from quantum_echo.enhancements import run_mcmc
import numpy as np

# Analyze experimental data
data = np.random.normal(0.5, 0.1, 100)  # Your measurement data
idata, summary = run_mcmc(data, draws=1000, tune=500)
print(summary)
```

## Testing
```bash
# Run tests with coverage
pytest --cov=quantum_echo --cov-report=term-missing

# Target: 95% coverage
pytest --cov=quantum_echo --cov-report=html
```

## Performance
- Async simulation: ~100-1000 trials/second
- Real-time updates: ~0.5 second refresh rate
- Memory efficient: Streaming data processing
- Responsive UI: Non-blocking background computation

## License
MIT. Contributions welcome!
