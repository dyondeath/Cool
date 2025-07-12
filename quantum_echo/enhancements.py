import qutip as qt
import pymc as pm
import arviz as az
import pytensor.tensor as pt


def add_gravity_decoherence(state, deco=1e-4):
    """Apply gravity-like phase noise."""
    if deco > 0:
        noise = qt.rand_dm_ginibre(state.dims[0][0]) * deco
        state = (state + noise).unit()
    return state


def run_mcmc(data, draws=1000, tune=500):
    with pm.Model():
        phase = pm.Normal('phase', mu=0, sigma=0.2)
        mu = pm.Deterministic('mu', 0.5 + pt.sin(phase))
        pm.Normal('like', mu=mu, sigma=0.1, observed=data)
        idata = pm.sample(draws, tune=tune, chains=2, progressbar=False)
    return idata, az.summary(idata)
