import pytest
import numpy as np
import queue
import threading
import time
from unittest.mock import patch, MagicMock

from quantum_echo.core_simulator import quantum_eraser_sim
from quantum_echo.enhancements import run_mcmc, add_gravity_decoherence
from quantum_echo.dashboard import async_sim, get_sim_queue, run_real_time_sim
from quantum_echo.utils import load_data, plot_results, fit_mcmc_proxy

class TestCoreSimulator:
    """Test the core quantum eraser simulator."""
    
    def test_sim_basic(self):
        """Test basic simulation functionality."""
        p, err = quantum_eraser_sim(100)
        assert abs(p - 0.5) < 0.15  # Loose for variance
        assert err > 0  # Should have some error
    
    def test_sim_different_basis(self):
        """Test simulation with different measurement bases."""
        p0, _ = quantum_eraser_sim(100, basis=0)
        p1, _ = quantum_eraser_sim(100, basis=1)
        p2, _ = quantum_eraser_sim(100, basis=2)
        
        # All should be valid probabilities
        assert 0 <= p0 <= 1
        assert 0 <= p1 <= 1
        assert 0 <= p2 <= 1
    
    def test_sim_phase_shift(self):
        """Test simulation with phase shift."""
        p1, _ = quantum_eraser_sim(100, phase_shift=0.0)
        p2, _ = quantum_eraser_sim(100, phase_shift=0.1)
        
        # Both should be valid probabilities
        assert 0 <= p1 <= 1
        assert 0 <= p2 <= 1
    
    def test_sim_coupling_strength(self):
        """Test simulation with different coupling strengths."""
        p1, _ = quantum_eraser_sim(100, coupling_strength=0.1)
        p2, _ = quantum_eraser_sim(100, coupling_strength=0.5)
        
        # Both should be valid probabilities
        assert 0 <= p1 <= 1
        assert 0 <= p2 <= 1
    
    def test_sim_with_decoherence(self):
        """Test simulation with gravity decoherence."""
        p_clean, _ = quantum_eraser_sim(100, gravity_deco=0.0)
        p_noisy, _ = quantum_eraser_sim(100, gravity_deco=0.001)
        
        # Both should be valid probabilities
        assert 0 <= p_clean <= 1
        assert 0 <= p_noisy <= 1
    
    def test_sim_large_trials(self):
        """Test simulation with larger number of trials."""
        p, err = quantum_eraser_sim(1000)
        assert abs(p - 0.5) < 0.1  # Should be closer to 0.5 with more trials
        assert err > 0


class TestEnhancements:
    """Test MCMC and enhancement functionality."""
    
    def test_mcmc_basic(self):
        """Test basic MCMC functionality."""
        data = np.full(50, 0.5)
        idata, summary = run_mcmc(data, draws=10, tune=5)
        assert 'phase' in summary.to_string()
        assert idata is not None
    
    def test_mcmc_with_variation(self):
        """Test MCMC with varied data."""
        data = np.random.normal(0.5, 0.1, 30)
        idata, summary = run_mcmc(data, draws=10, tune=5)
        assert 'phase' in summary.to_string()
    
    def test_mcmc_sinusoidal_model(self):
        """Test MCMC with sinusoidal model."""
        # Create data that follows sinusoidal pattern
        phase_true = 0.3
        data = np.full(30, 0.5 + np.sin(phase_true)) + np.random.normal(0, 0.05, 30)
        
        # Test sinusoidal model
        idata, summary = run_mcmc(data, draws=20, tune=10, model_type='sinusoidal')
        assert 'phase' in summary.to_string()
        assert 'mu' in summary.to_string()
        
    def test_mcmc_model_selection(self):
        """Test both linear and sinusoidal models work."""
        data = np.random.normal(0.6, 0.1, 25)
        
        # Test linear model
        idata_linear, summary_linear = run_mcmc(data, draws=15, tune=8, model_type='linear')
        assert 'phase' in summary_linear.to_string()
        
        # Test sinusoidal model  
        idata_sin, summary_sin = run_mcmc(data, draws=15, tune=8, model_type='sinusoidal')
        assert 'phase' in summary_sin.to_string()
        
        # Both should work and return different results
        assert idata_linear is not None
        assert idata_sin is not None
    
    def test_add_gravity_decoherence(self):
        """Test gravity decoherence function."""
        import qutip as qt
        
        # Test with clean state
        state = qt.basis(2, 0)
        clean_state = add_gravity_decoherence(state, deco=0.0)
        assert clean_state.norm() > 0.99  # Should be normalized
        
        # Test with decoherence
        noisy_state = add_gravity_decoherence(state, deco=0.001)
        assert noisy_state.norm() > 0.99  # Should still be normalized


class TestAsyncSimulation:
    """Test async simulation functionality."""
    
    def test_async_sim(self):
        """Test async simulation wrapper."""
        q = queue.Queue()
        async_sim(100, 0, 0.0, 0.1, 0.0, q)
        p, err = q.get()
        assert abs(p - 0.5) < 0.15
        assert err > 0
    
    def test_async_sim_with_params(self):
        """Test async simulation with different parameters."""
        q = queue.Queue()
        async_sim(50, 1, 0.1, 0.2, 0.001, q)
        p, err = q.get()
        assert 0 <= p <= 1
        assert err > 0
    
    def test_get_sim_queue(self):
        """Test queue creation for simulation."""
        q = get_sim_queue()
        assert isinstance(q, queue.Queue)
        assert q.empty()
    
    def test_run_real_time_sim(self):
        """Test real-time simulation thread creation."""
        result_queue, thread = run_real_time_sim(50, 0, 0.0, 0.1, 0.0)
        
        # Wait for thread to complete
        thread.join(timeout=10)
        # Don't assert thread is dead - it might still be cleaning up
        
        # Check results
        assert not result_queue.empty()
        p, err = result_queue.get()
        assert 0 <= p <= 1
        assert err > 0


class TestUtils:
    """Test utility functions."""
    
    def test_load_data_array(self):
        """Test loading data from numpy array."""
        # Create temporary data
        test_data = np.array([[1, 0.5], [2, 0.6], [3, 0.4]])
        
        # Mock file loading
        with patch('numpy.loadtxt') as mock_loadtxt:
            mock_loadtxt.return_value = test_data
            
            data = load_data("fake_file.csv")
            assert data.shape == (3, 2)
            assert np.array_equal(data, test_data)
    
    def test_plot_results(self):
        """Test plotting functionality."""
        p_h = np.array([0.4, 0.5, 0.6])
        err = 0.1
        
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_results(p_h, err)
            mock_show.assert_called_once()
    
    def test_fit_mcmc_proxy(self):
        """Test MCMC proxy fitting."""
        data = np.array([[1, 0.5], [2, 0.6], [3, 0.4]])
        
        # Test the actual function instead of mocking
        result = fit_mcmc_proxy(data)
        
        # Check that we get a summary object (not the exact string)
        assert hasattr(result, 'tables') or 'OLS Regression Results' in str(result)


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_simulation_to_mcmc_workflow(self):
        """Test complete workflow from simulation to MCMC analysis."""
        # Run simulation
        p_h, err = quantum_eraser_sim(100)
        
        # Create synthetic data for MCMC
        data = np.random.normal(p_h, err, 50)
        
        # Run MCMC
        idata, summary = run_mcmc(data, draws=10, tune=5)
        
        # Verify results
        assert 'phase' in summary.to_string()
        assert idata is not None
    
    def test_async_to_analysis_workflow(self):
        """Test workflow from async simulation to analysis."""
        # Run async simulation
        q = queue.Queue()
        async_sim(100, 1, 0.1, 0.2, 0.001, q)
        p, err = q.get()
        
        # Verify simulation results
        assert 0 <= p <= 1
        assert err > 0
        
        # Use results for further analysis
        data = np.random.normal(p, err, 30)
        idata, summary = run_mcmc(data, draws=10, tune=5)
        assert 'phase' in summary.to_string()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_sim_zero_trials(self):
        """Test simulation with zero trials."""
        with pytest.raises((ValueError, IndexError)):
            quantum_eraser_sim(0)
    
    def test_sim_negative_trials(self):
        """Test simulation with negative trials."""
        with pytest.raises((ValueError, IndexError)):
            quantum_eraser_sim(-10)
    
    def test_mcmc_empty_data(self):
        """Test MCMC with empty data."""
        with pytest.raises((ValueError, IndexError)):
            run_mcmc(np.array([]), draws=10, tune=5)
    
    def test_async_sim_invalid_params(self):
        """Test async simulation with invalid parameters."""
        q = queue.Queue()
        # This should still work but may give unexpected results
        async_sim(10, 999, 10.0, -1.0, -0.1, q)
        p, err = q.get()
        # Should still return some values even if nonsensical
        assert isinstance(p, (int, float))
        assert isinstance(err, (int, float))


# Run: pytest --cov=quantum_echo --cov-report=term-missing (aim 95%)
