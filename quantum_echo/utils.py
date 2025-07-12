import numpy as np
import matplotlib.pyplot as plt
# Standard logging – Rich is optional and used only for pretty output.
import logging

try:
    from rich.logging import RichHandler
    _handler_cls = RichHandler
except ModuleNotFoundError:
    # Graceful fallback when Rich is not available in the execution environment.
    _handler_cls = logging.StreamHandler
import pandas as pd
import io

# Setup Rich logging
logging.basicConfig(level="INFO", handlers=[_handler_cls()])
logger = logging.getLogger(__name__)


def load_data(csv_path_or_file):
    """Load CSV data for fitting. Handles both file paths and file objects."""
    try:
        if hasattr(csv_path_or_file, 'read'):
            # It's a file-like object (e.g., from Streamlit file_uploader)
            logger.info("Loading data from uploaded file")
            content = csv_path_or_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            data = np.loadtxt(io.StringIO(content), delimiter=',', skiprows=1)
        else:
            # It's a file path
            logger.info(f"Loading data from {csv_path_or_file}")
            data = np.loadtxt(csv_path_or_file, delimiter=',', skiprows=1)
        
        logger.info(f"Data loaded successfully with shape: {data.shape}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise ValueError(f"Failed to load data: {str(e)}")


def plot_results(p_h, err):
    """Plot histogram of P(H) results."""
    try:
        plt.figure(figsize=(8, 6))
        if isinstance(p_h, (list, np.ndarray)) and len(p_h) > 1:
            plt.hist(p_h, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(p_h), color='red', linestyle='--', label=f'Mean: {np.mean(p_h):.3f}')
        else:
            # Single value case
            plt.bar([0], [1], color='skyblue', alpha=0.7)
            plt.axvline(p_h if np.isscalar(p_h) else p_h[0], color='red', linestyle='--')
        
        plt.xlabel('P(H)')
        plt.ylabel('Frequency')
        plt.title(f"P(H) Distribution: {np.mean(p_h):.3f} ± {err:.3f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")


def fit_mcmc_proxy(data):
    """Fit data using OLS as a proxy for MCMC fitting."""
    try:
        if data.ndim == 1:
            # Single column data, create index
            x = np.arange(len(data))
            y = data
        else:
            # Multi-column data, use first two columns
            x = data[:, 0]
            y = data[:, 1]
        
        # Add constant term for intercept
        x_with_const = np.column_stack([np.ones(len(x)), x])
        
        # Import statsmodels lazily so that unit tests can monkeypatch it
        import statsmodels.api as sm

        # Fit OLS model
        model = sm.OLS(y, x_with_const)
        results = model.fit()
        
        logger.info("OLS fit completed successfully")
        return results.summary()
    
    except Exception as e:
        logger.error(f"Error in OLS fitting: {str(e)}")
        return f"Error in fitting: {str(e)}"


def validate_simulation_params(trials, basis, phase_shift, coupling_strength, gravity_deco):
    """Validate simulation parameters."""
    errors = []
    
    if trials <= 0:
        errors.append("Trials must be positive")
    if basis not in [0, 1, 2]:
        errors.append("Basis must be 0, 1, or 2")
    if not -1 <= phase_shift <= 1:
        errors.append("Phase shift should be between -1 and 1")
    if coupling_strength <= 0:
        errors.append("Coupling strength must be positive")
    if gravity_deco < 0:
        errors.append("Gravity decoherence must be non-negative")
    
    if errors:
        raise ValueError("; ".join(errors))
    
    return True


def format_results(p_h, err, precision=3):
    """Format simulation results for display."""
    return {
        'mean': round(float(p_h), precision),
        'std': round(float(err), precision),
        'formatted': f"{p_h:.{precision}f} ± {err:.{precision}f}",
        'confidence_interval': [
            round(float(p_h - 1.96 * err), precision),
            round(float(p_h + 1.96 * err), precision)
        ]
    }


def generate_sample_data(n_points=100, mean=0.5, std=0.1, add_noise=True):
    """Generate sample data for testing purposes."""
    base_data = np.random.normal(mean, std, n_points)
    
    if add_noise:
        # Add some systematic variation
        x = np.linspace(0, 2*np.pi, n_points)
        systematic = 0.05 * np.sin(x)
        base_data += systematic
    
    # Ensure values are in [0, 1] range
    base_data = np.clip(base_data, 0, 1)
    
    return base_data


def calculate_visibility(data, phase_points=None):
    """Calculate fringe visibility from data."""
    if phase_points is None:
        # Simple visibility calculation
        visibility = (np.max(data) - np.min(data)) / (np.max(data) + np.min(data))
    else:
        # More sophisticated calculation with phase information
        # Fit sine wave and extract amplitude
        from scipy.optimize import curve_fit
        
        def sine_func(x, a, b, c, d):
            return a * np.sin(b * x + c) + d
        
        try:
            popt, _ = curve_fit(sine_func, phase_points, data)
            amplitude = abs(popt[0])
            offset = popt[3]
            visibility = amplitude / offset if offset != 0 else 0
        except:
            visibility = (np.max(data) - np.min(data)) / (np.max(data) + np.min(data))
    
    return np.clip(visibility, 0, 1)  # Visibility should be between 0 and 1
