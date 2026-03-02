# %% [markdown]
# # 02 - Meta-Analysis Demo
#
# Demonstrates:
# 1. Fixed-effects model (inverse-variance weighting)
# 2. Random-effects model (DerSimonian-Laird)
# 3. Heterogeneity analysis (Q, I2, H2, tau2)
# 4. Forest plot generation
# 5. Funnel plot with Egger's test and trim-and-fill

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.meta_analysis.fixed_effects import FixedEffectsModel
from src.meta_analysis.random_effects import RandomEffectsModel
from src.meta_analysis.heterogeneity import HeterogeneityAnalysis
from src.meta_analysis.forest_plot import ForestPlot
from src.meta_analysis.funnel_plot import FunnelPlot

print("Imports OK")

# %%
np.random.seed(42)

labels = [
    "Murrough 2013", "Murrough 2015", "Ionescu 2018",
    "Singh 2016a", "Singh 2016b", "Canuso 2018",
    "Grunebaum 2017", "Su 2017", "Wilkinson 2018",
    "Bartova 2021", "McInnes 2022", "Daly 2019",
]

effects = np.array([-0.82, -0.65, -0.71, -0.93, -0.88, -0.59, -0.77, -0.84, -0.61, -0.70, -0.68, -0.76])
ns = np.array([47, 53, 68, 80, 83, 68, 40, 71, 90, 55, 62, 120])
ses = np.sqrt(4 / ns)
variances = ses ** 2

# %% [markdown]
# ## 1. Fixed-Effects Model

# %%
fe_model = FixedEffectsModel()
fe_result = fe_model.fit_from_arrays(
    effects=list(effects), variances=list(variances),
    study_labels=labels, effect_type="SMD",
)
print(fe_result.summary())

# %% [markdown]
# ## 2. Random-Effects Model (DL)

# %%
re_model = RandomEffectsModel(estimator="DL")
re_result = re_model.fit_from_arrays(
    effects=list(effects), variances=list(variances),
    study_labels=labels, effect_type="SMD",
)
print(re_result.summary())

# %% [markdown]
# ## 3. Heterogeneity Analysis

# %%
het = HeterogeneityAnalysis(compute_reml=False)
het_stats = het.analyse(effects, variances)
print(het_stats.summary())

# %% [markdown]
# ## 4. Forest Plot

# %%
fp = ForestPlot(re_result)
fig = fp.render(title="Ketamine vs. Placebo/Control - Depression Outcomes (SMD)")
fig.savefig("/tmp/forest_plot_demo.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Forest plot saved to /tmp/forest_plot_demo.png")

# %% [markdown]
# ## 5. Funnel Plot

# %%
funnel = FunnelPlot(
    effects=list(effects), ses=list(ses),
    study_labels=labels, effect_type="SMD",
)

fig_funnel = funnel.render(title="Funnel Plot - Ketamine Meta-Analysis")
fig_funnel.savefig("/tmp/funnel_plot_demo.png", dpi=150, bbox_inches="tight")
plt.close(fig_funnel)

egger = funnel.eggers_test()
print(f"\nEgger's test:")
print(f"  Intercept: {egger.intercept:.3f} (SE={egger.intercept_se:.3f}, p={egger.intercept_p:.3f})")
print(f"  Conclusion: {egger.conclusion}")

tf = funnel.trim_and_fill()
print(f"\nTrim-and-fill:")
print(f"  Studies trimmed: {tf.n_trimmed}")
print(f"  Adjusted pooled: {tf.adjusted_effect:.3f} (95% CI: {tf.adjusted_ci_lower:.3f}, {tf.adjusted_ci_upper:.3f})")

# %% [markdown]
# ## 6. Compare Fixed vs Random Effects

# %%
print("\nSummary Comparison:")
print(f"{'Model':<25} {'Pooled SMD':>12} {'95% CI':>22} {'I2':>8} {'p-val':>8}")
print("-" * 75)
print(f"{'Fixed Effects':<25} {fe_result.pooled_effect:>12.3f} ({fe_result.ci_lower:.3f}, {fe_result.ci_upper:.3f})  {fe_result.i2:>6.1f}%  {fe_result.p_value:>8.4f}")
print(f"{'Random Effects (DL)':<25} {re_result.pooled_effect:>12.3f} ({re_result.ci_lower:.3f}, {re_result.ci_upper:.3f})  {re_result.i2:>6.1f}%  {re_result.p_value:>8.4f}")

print(f"\ntau2 (DL) = {re_result.tau2:.4f}, tau = {re_result.tau:.4f}")
if re_result.prediction_interval_lower is not None:
    print(f"Prediction interval: ({re_result.prediction_interval_lower:.3f}, {re_result.prediction_interval_upper:.3f})")
