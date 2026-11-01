"""Bayesian uncertainty with NumPyro / NUTS and exact (implicit) gradients.

Three likelihood treatments of the linear responses R are available:

``explicit``  R are sampled together with the constants (legacy
              HGtitrManifree treatment: joint posterior of constants and
              responses).  Prior R[p, q] ~ Normal(0, tau_q),
              tau_q = prior_scale * max_p |R_hat[p, q]|.
``marginal``  the same model with R integrated out analytically (exact
              Gaussian integral, Woodbury form):
                  y_q ~ Normal(0, s_q^2 I + tau_q^2 A A^T)
              The posterior of the constants is identical to ``explicit`` but
              the sampler only sees the nonlinear dimensions.  (A flat response
              prior must NOT be used: its marginal diverges as complexes vanish,
              because responses can then grow without bound.)
``eiv``       ``marginal`` plus log10 scale factors on stock concentrations with
              Normal(0, sd) priors (errors in variables) for every Experiment
              that sets ``titrant_factor_sd`` / ``host_factor_sd``.

Channel noise sigma_q has a log-uniform prior around the ML estimate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from .fit import Experiment, FitResult
from .jax_speciation import species_conc
from .observables import design_matrix

jax.config.update("jax_enable_x64", True)


@dataclass
class Posterior:
    samples: dict
    names: list
    seconds: float
    mcmc: MCMC
    summary: dict

    def draws(self, name):
        return np.asarray(self.samples[name])

    def interval(self, name, level=0.95):
        x = self.draws(name)
        a = (1 - level) / 2
        return np.quantile(x, [a, 1 - a], axis=0)


def _build(fitres: FitResult, treatment: str, prior_scale: float, bounds, sigma_span):
    prob = fitres.problem
    model = prob.model
    S = jnp.asarray(model.stoich)
    exps = prob.exps
    masks = [np.where(m)[0] for m in prob.masks]
    lo, hi = bounds
    p = model.n_complex
    sig_hat = [np.asarray(s) for s in fitres.sigmas]
    R_hat = [np.asarray(R) for R in fitres.responses]
    data = [(jnp.asarray(e.HT), jnp.asarray(e.GT), jnp.asarray(e.Y)) for e in exps]

    def numpyro_model():
        lb = numpyro.sample("logbeta", dist.Uniform(jnp.full(p, lo), jnp.full(p, hi)))
        for j, (e, (HT, GT, Y)) in enumerate(zip(exps, data)):
            if treatment == "eiv" and e.titrant_factor_sd is not None:
                GT = GT * 10.0 ** numpyro.sample(f"logf_G_{j}", dist.Normal(0.0, e.titrant_factor_sd))
            if treatment == "eiv" and e.host_factor_sd is not None:
                HT = HT * 10.0 ** numpyro.sample(f"logf_H_{j}", dist.Normal(0.0, e.host_factor_sd))
            C = species_conc(lb, HT, GT, S)
            A = design_matrix(C, np.asarray(model.stoich), HT, GT, e.kind, masks[j], e.baseline)
            N, Q = Y.shape
            P = A.shape[1]
            if e.sigma is not None:
                sig = jnp.asarray(e.sigma)
            else:
                ls = numpyro.sample(f"log_sigma_{j}", dist.Uniform(jnp.log(sig_hat[j] / sigma_span),
                                                                   jnp.log(sig_hat[j] * sigma_span)))
                sig = jnp.exp(ls)
            # proper Gaussian response prior, shared by the explicit and marginal treatments:
            # R[p, q] ~ Normal(0, tau_q),  tau_q = prior_scale * max_p |R_hat[p, q]|
            tau = prior_scale * (np.abs(R_hat[j]).max(axis=0) + 1e-12)          # (Q,)
            if treatment == "explicit":
                R = numpyro.sample(f"R_{j}", dist.Normal(jnp.zeros((P, Q)), jnp.broadcast_to(jnp.asarray(tau), (P, Q))))
                numpyro.sample(f"y_{j}", dist.Normal(A @ R, sig[None, :]), obs=Y)
            else:
                # exact Gaussian marginal over R (Woodbury, per channel):
                # y_q ~ N(0, s_q^2 I + tau_q^2 A A^T)
                Bq = A[None, :, :] * jnp.asarray(tau)[:, None, None]            # (Q, N, P)
                Mq = jnp.eye(P)[None] + jnp.einsum("qnp,qnr->qpr", Bq, Bq) / sig[:, None, None] ** 2
                Lq = jnp.linalg.cholesky(Mq)
                bty = jnp.einsum("qnp,nq->qp", Bq, Y) / sig[:, None] ** 2          # (Q, P)
                z = jax.scipy.linalg.solve_triangular(Lq, bty[..., None], lower=True)[..., 0]
                quad = jnp.sum(Y * Y, 0) / sig ** 2 - jnp.sum(z * z, 1)
                logdet = 2 * N * jnp.log(sig) + 2 * jnp.sum(jnp.log(jnp.diagonal(Lq, axis1=1, axis2=2)), 1)
                ll = jnp.sum(-0.5 * (N * jnp.log(2 * jnp.pi) + logdet + quad))
                numpyro.factor(f"marg_{j}", ll)

    init = {"logbeta": jnp.asarray(fitres.logbeta)}
    for j, e in enumerate(exps):
        if e.sigma is None:
            init[f"log_sigma_{j}"] = jnp.log(jnp.asarray(sig_hat[j]))
        if treatment == "explicit":
            init[f"R_{j}"] = jnp.asarray(R_hat[j])
    return numpyro_model, init


def sample_posterior(fitres: FitResult, treatment: str = "marginal", num_warmup: int = 1000,
                     num_samples: int = 2000, num_chains: int = 4, seed: int = 0, prior_scale: float = 10.0,
                     bounds=(-2.0, 18.0), sigma_span: float = 100.0, progress: bool = False,
                     target_accept: float = 0.9) -> Posterior:
    """Run NUTS from the deterministic optimum."""
    model_fn, init = _build(fitres, treatment, prior_scale, bounds, sigma_span)
    kernel = NUTS(model_fn, init_strategy=init_to_value(values=init), target_accept_prob=target_accept,
                  dense_mass=True)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="sequential", progress_bar=progress)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed))
    secs = time.perf_counter() - t0
    samples = mcmc.get_samples()
    summ = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
    return Posterior({k: np.asarray(v) for k, v in samples.items()}, list(samples), secs, mcmc, summ)


def diagnostics(post: Posterior, name: str = "logbeta"):
    s = post.summary[name]
    return dict(r_hat=np.asarray(s["r_hat"]), n_eff=np.asarray(s["n_eff"]),
                ess_per_s=np.asarray(s["n_eff"]) / post.seconds)
