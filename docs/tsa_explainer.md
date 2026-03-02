# Trial Sequential Analysis (TSA) Explainer

## What is TSA?

Trial Sequential Analysis applies group sequential methods from clinical trials
to **cumulative meta-analysis**. It addresses a critical problem: as studies
accumulate over time, each new analysis creates an additional opportunity to
cross a significance threshold by chance -- inflating the type-I error rate.

Without TSA, a meta-analysis that "just reached significance" with the 8th
study may reflect random variation rather than a true effect, especially if
the accrued sample is small relative to what a well-powered trial would require.

---

## Core Concepts

### Information Fraction (t)

The information fraction tracks how much evidence has accumulated relative
to the total needed:

```
t = N_accrued / N_required (RIS)
```

where RIS = Required Information Size (the sample size needed for a definitive
single trial).

As t -> 1.0, the analysis approaches full information.

### Cumulative Z-Statistic

At each update step (when a new study is added), the cumulative z-statistic
is computed from the current pooled meta-analysis:

```
z_cum = theta_bar_cumulative / SE(theta_bar_cumulative)
```

### Alpha-Spending Boundaries

Alpha-spending functions allocate the total type-I error budget (alpha = 0.05)
across sequential looks at the data. Instead of comparing z to a fixed 1.96
at every update, the boundary moves:

- Early in the trial (small t): boundary is **very high** (conservative)
- At full information (t = 1): boundary approaches z_alpha/2 = 1.96

---

## Spending Functions

### O'Brien-Fleming (OBF) -- Default

The most conservative function. Spends almost no alpha early:

```
alpha*(t) = 2 * [1 - Phi(z_alpha/2 / sqrt(t))]
```

At t = 0.10: boundary z ~= 5.1 (near-impossible to cross early)
At t = 1.00: boundary z ~= 1.96 (conventional threshold)

**Best for:** Neuroscience meta-analyses where false positives carry
significant clinical consequences.

### Lan-DeMets (alpha*t)

Linear spending -- proportional to information accrued:

```
alpha*(t) = alpha * t
```

### Pocock

Spends more alpha early than OBF. Produces approximately constant z-boundaries
across all looks. Not recommended for final analysis (lower power).

### Haybittle-Peto

Fixed interim boundary z = 3.0, spending near-zero alpha until the last look.

---

## Required Information Size (RIS)

The RIS is the total sample size needed to detect a clinically meaningful
effect with specified power.

### Continuous Outcomes (SMD)

```
RIS = 4 * (z_alpha/2 + z_beta)^2 * sigma^2 / Delta^2
```

Parameters:
- alpha = type-I error (e.g., 0.05)
- beta = type-II error (e.g., 0.20 for 80% power)
- sigma = pooled SD of the outcome
- Delta = minimal clinically important difference (MCID)

For SMD with sigma = 1: `RIS = 4 * (1.96 + 0.84)^2 / MCID^2`

### Binary Outcomes

```
RIS = (z_alpha/2 + z_beta)^2 * [p1*(1-p1) + p2*(1-p2)] / (p1 - p2)^2
```

### Heterogeneity Adjustment (Wetterslev 2008)

```
RIS_adjusted = RIS_unadjusted / (1 - I^2)
```

For I^2 = 40%: RIS is multiplied by 1/(1-0.40) = 1.67 (67% more patients needed).

---

## Futility Boundaries

The futility (inner) boundary marks where continuing is unlikely to change
the conclusion.

```
z_futility = z_alpha * sqrt(t) - z_beta * sqrt(1-t)
```

**Note:** Futility boundaries are **non-binding** in meta-analysis.

---

## TSA Conclusions

| Conclusion | Meaning |
|------------|---------|
| `FIRM_EVIDENCE` | z_cum crossed the efficacy boundary. True effect, not random fluctuation. |
| `INSUFFICIENT` | Accrued information below RIS; evidence not yet definitive. |
| `FUTILE` | Evidence unlikely to reach efficacy boundary even with more studies. |
| `HARM_SIGNAL` | z_cum crossed the lower boundary (negative direction). |

---

## TSA-Adjusted Confidence Interval

The conventional 95% CI uses z = 1.96 regardless of information fraction.
The TSA-adjusted CI uses the current alpha-spending boundary z:

```
CI_TSA = theta_bar +/- z_boundary(t) * SE(theta_bar)
```

When t < 1 and information is incomplete, z_boundary > 1.96, so the
TSA-adjusted CI is **wider** -- appropriately conveying uncertainty.

---

## References

1. Wetterslev J, et al. Trial sequential analysis may establish when firm evidence is reached. J Clin Epidemiol. 2008;61:64-75.
2. Thorlund K, et al. Improved power and sample size calculations for TSA. Contemp Clin Trials. 2011;32:906-917.
3. O'Brien PC, Fleming TR. A multiple testing procedure for clinical trials. Biometrics. 1979;35:549-556.
4. Lan KKG, DeMets DL. Discrete sequential boundaries for clinical trials. Biometrika. 1983;70:659-663.
5. Higgins JPT, Thomas J, et al. Cochrane Handbook for Systematic Reviews v6.3. Chapter 23: TSA. 2022.
