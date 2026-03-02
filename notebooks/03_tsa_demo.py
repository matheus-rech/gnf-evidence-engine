# %% [markdown]
# # 03 - Trial Sequential Analysis (TSA) Demo
#
# Demonstrates:
# 1. Required Information Size (RIS) calculation
# 2. TSA with O'Brien-Fleming spending boundaries
# 3. TSA plot generation
# 4. Comparison of spending functions
# 5. Interpreting conclusions

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.tsa.information_size import RequiredInformationSize
from src.tsa.trial_sequential import TrialSequentialAnalysis
from src.tsa.tsa_plot import TSAPlot
from src.tsa.spending_functions import OBrienFleming, LanDeMets, Pocock

print("Imports OK")

# %% [markdown]
# ## 1. Required Information Size

# %%
ris_calc = RequiredInformationSize(alpha=0.05, beta=0.20)

ris_cont = ris_calc.continuous(delta=0.30, sigma=1.0, i2=0.40)
print(ris_cont.summary())

ris_bin = ris_calc.binary(p_control=0.30, relative_risk_reduction=0.25, i2=0.20)
print("\n" + ris_bin.summary())

# %% [markdown]
# ## 2. Simulate Cumulative Studies

# %%
np.random.seed(123)

TRUE_EFFECT = -0.45
K_TOTAL = 15

study_effects = TRUE_EFFECT + np.random.normal(0, 0.20, K_TOTAL)
study_ses = 0.18 + np.random.uniform(0, 0.12, K_TOTAL)
study_variances = study_ses ** 2
study_ns = np.random.randint(40, 160, K_TOTAL)
study_labels = [f"Study {i+1} ({2010+i})" for i in range(K_TOTAL)]

print("Simulated studies:")
for i, (eff, se, n) in enumerate(zip(study_effects, study_ses, study_ns)):
    print(f"  {study_labels[i]:<20} SMD={eff:.3f}  SE={se:.3f}  N={n}")

# %% [markdown]
# ## 3. Run TSA with O'Brien-Fleming Boundaries

# %%
tsa_obf = TrialSequentialAnalysis(
    alpha=0.05, beta=0.20,
    spending_function="obrien_fleming",
    futility_spending=True,
)

result_obf = tsa_obf.run(
    effects=list(study_effects), variances=list(study_variances),
    sample_sizes=list(study_ns), study_labels=study_labels,
    delta=0.30, sigma=1.0, i2=0.30,
)

print(result_obf.summary())

print("\n{:<5} {:<22} {:>10} {:>12} {:>12} {:>12} {:>12}".format(
    "Step", "Study", "Cum. N", "Z-stat", "t (info)", "Upper B", "Crossed?"
))
print("-" * 88)
for step in result_obf.steps:
    print("{:<5} {:<22} {:>10} {:>12.4f} {:>12.3f} {:>12.4f} {:>12}".format(
        step.step_index + 1, step.study_label[:22], step.n_cumulative,
        step.z_stat, step.information_fraction, step.boundary_upper,
        "YES" if step.crossed_upper else "no",
    ))

# %% [markdown]
# ## 4. TSA Plot

# %%
tsa_plot = TSAPlot(result_obf, show_ris_line=True, show_futility=True, show_study_labels=True)
fig = tsa_plot.render(title="TSA: Ketamine Depression Meta-Analysis (O'Brien-Fleming)")
fig.savefig("/tmp/tsa_plot_demo.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("TSA plot saved to /tmp/tsa_plot_demo.png")

# %% [markdown]
# ## 5. Compare Spending Functions

# %%
print("\nSpending function comparison at key information fractions:")
print(f"{'t':<8} {'OBF':>12} {'Lan-DeMets':>12} {'Pocock':>12}")
print("-" * 46)

obf = OBrienFleming(alpha=0.05)
ld = LanDeMets(alpha=0.05)
poc = Pocock(alpha=0.05)

for t in [0.10, 0.25, 0.50, 0.75, 1.00]:
    print(f"{t:<8.2f} {obf.spent_alpha(t):>12.5f} {ld.spent_alpha(t):>12.5f} {poc.spent_alpha(t):>12.5f}")

# %% [markdown]
# ## 6. Run TSA with Different Spending Functions

# %%
print("\nCumulative Z at final step, by spending function:")
for fn_name, fn_label in [
    ("obrien_fleming", "O'Brien-Fleming"),
    ("lan_demets", "Lan-DeMets"),
    ("pocock", "Pocock"),
]:
    tsa_fn = TrialSequentialAnalysis(
        alpha=0.05, beta=0.20, spending_function=fn_name, futility_spending=False,
    )
    r = tsa_fn.run(
        effects=list(study_effects), variances=list(study_variances),
        sample_sizes=list(study_ns), study_labels=study_labels,
        delta=0.30, sigma=1.0, i2=0.30,
    )
    final = r.steps[-1]
    print(f"  {fn_label:<20} Z={r.final_z:.4f}  boundary={final.boundary_upper:.4f}  conclusion={r.conclusion}")

# %% [markdown]
# ## 7. TSA-Adjusted Confidence Interval

# %%
print(f"\nFinal pooled effect: {result_obf.steps[-1].pooled_effect:.4f}")
final_step = result_obf.steps[-1]
conv_lo = final_step.pooled_effect - 1.96 * final_step.pooled_se
conv_hi = final_step.pooled_effect + 1.96 * final_step.pooled_se
print(f"  Conventional CI: ({conv_lo:.3f}, {conv_hi:.3f})")
print(f"  TSA-adjusted CI: ({result_obf.tsa_adjusted_ci_lower:.3f}, {result_obf.tsa_adjusted_ci_upper:.3f})")
print(f"  Information fraction: {result_obf.final_information_fraction:.3f}")
