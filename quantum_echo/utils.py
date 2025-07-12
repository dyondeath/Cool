import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS
import logging
from rich.logging import RichHandler

logging.basicConfig(level="INFO", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def load_data(csv_path):
    """Load CSV data for fitting."""
    logger.info(f"Loading data from {csv_path}")
    return np.loadtxt(csv_path, delimiter=',', skiprows=1)


def plot_results(p_h, err):
    plt.hist(p_h, bins=20)
    plt.title(f"P(H) Dist: {np.mean(p_h):.3f} ± {err:.3f}")
    plt.show()


def fit_mcmc_proxy(data):
    model = OLS(data[:,1], data[:,0])
    return model.fit().summary()
