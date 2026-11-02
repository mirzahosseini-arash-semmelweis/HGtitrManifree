"""End-to-end example: simulate -> fit candidate models -> select -> uncertainty.

python examples/quickstart.py            (about 1 minute; NUTS step needs jax + numpyro)

It is recommended to install `hgtitr` in editable mode:
cd C:\path\to\HGtitrmanifree
py -3.1x -m pip install -e .
"""
import numpy as np

from hgtitr import Experiment, Model, compare_models, fit, profile_ci, solve_speciation, titration
from hgtitr.forward import free_manifold, sensitivity_map

# 1. Design: 1 uM host (3 mL) titrated with 0.1 M guest, 30 additions
HT, GT = titration(c_cell_H=1e-6, c_titrant_G=1e-1, V0=3.0, V_add=np.linspace(0, 0.01, 30))

# 2. Simulate a sequential 1:2 system (exact speciation) with noise
truth = Model(["H1G1", "H1G2"])
C = solve_speciation(HT, GT, truth, [5.0, 8.0]).C          # columns: H, G, H1G1, H1G2
y = C[:, 2:] @ np.array([2e5, 3e5]) + np.random.default_rng(1).normal(0, 1e-3, HT.size)
exp = Experiment(HT, GT, y)

# 3. Fit candidate models (multistart, VARPRO) and compare by AICc
cands = [fit(Model(["H1G1"]), exp, start=[5], n_starts=4),
         fit(Model(["H1G1", "H1G2"]), exp, start=[5, 8], n_starts=8),
         fit(Model(["H1G1", "H1G2", "H1G3"]), exp, start=[5, 8, 10], n_starts=8)]
for row in compare_models(cands):
    print(f"{row['model']:22s} k={row['k']:2d} dAICc={row['delta']:7.2f} w={row['weight']:.3f} {row['logbeta']}")
best = cands[1]
print(best.summary())
print("profile 95% CI:", [tuple(np.round(profile_ci(best, j)[:2], 3)) for j in range(2)])

# 4. Forward manifold: exact sensitivities d(signal)/d(log beta) over the (H_T, G_T) plane
man = free_manifold(truth, best.logbeta, (HT.min(), HT.max()), (GT.min(), GT.max()), n=(20, 200))
_, _, sig, dS = sensitivity_map(truth, best.logbeta, [0, 0, *best.responses[0][:, 0]], man.h, man.g)
i = np.argmax(np.abs(dS[:, 1]))
print(f"most informative total composition for log beta12: H_T={man.HT[i]:.2e} M, G_T={man.GT[i]:.2e} M")

# 5. Bayesian posterior (NUTS, responses integrated out analytically)
try:
    from hgtitr.bayes import diagnostics, sample_posterior

    post = sample_posterior(best, "marginal", num_warmup=500, num_samples=1000, num_chains=2)
    lb = post.draws("logbeta")
    print("posterior mean", lb.mean(0).round(4), "sd", lb.std(0).round(4), "corr", np.corrcoef(lb.T)[0, 1].round(3))
    print("R-hat", diagnostics(post)["r_hat"].round(4))
except ImportError:
    print("install jax + numpyro for the Bayesian step: pip install 'hgtitrmanifree[bayes]'")
