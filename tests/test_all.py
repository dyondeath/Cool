import numpy as np
import queue

from quantum_echo.core_simulator import quantum_eraser_sim
from quantum_echo.enhancements import run_mcmc
from quantum_echo.dashboard import async_sim


def test_sim():
    p, err = quantum_eraser_sim(100)
    assert abs(p - 0.5) < 0.15


def test_mcmc():
    data = np.full(50, 0.5)
    idata, summary = run_mcmc(data, draws=10, tune=5)
    assert 'phase' in summary.to_string()


def test_async():
    q = queue.Queue()
    async_sim(100, 0, 0.0, 0.1, 0.0, q)
    p, err = q.get()
    assert abs(p - 0.5) < 0.15
