from quantum_echo.core_simulator import quantum_eraser_sim
from quantum_echo.enhancements import run_mcmc


def test_sim():
    p, err = quantum_eraser_sim(100)
    assert abs(p - 0.5) < 0.1


def test_mcmc():
    data = [0.5] * 50
    idata, summary = run_mcmc(data, draws=10, tune=5)
    assert 'phase' in summary.to_string()
