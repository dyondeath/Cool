# QuantumEchoSimulator

Research-grade tool for simulating delayed-choice quantum erasers with tunable relational phases and gravity-decoherence. Fits real datasets using PyMC MCMC. Provides real-time visualizations via Streamlit.

## Features
- Core sim based on QuTiP with tunable biases.
- Gravity decoherence extension.
- MCMC parameter recovery using PyMC.
- Data sets from several experiments included.
- Streamlit dashboard for interactive use.
- Sphinx documentation.

## Installation
```bash
git clone https://github.com/yourusername/QuantumEchoSimulator.git
cd QuantumEchoSimulator
pip install -r requirements.txt
pip install -e .
```

## Usage
```python
from quantum_echo.core_simulator import quantum_eraser_sim
p_h, err = quantum_eraser_sim(1000, basis=1, gravity_deco=1e-4)
print(p_h, err)
```

Run the dashboard:
```bash
streamlit run app.py
```

## Real-Time Dashboard
Adjust parameters live and watch results update every 0.5&nbsp;s. Logging is
displayed using Rich and uploaded CSV files are automatically fit via MCMC.

## License
MIT. Contributions welcome!
