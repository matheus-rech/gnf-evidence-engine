# Meta-Analysis Methods Guide

## Overview

This document describes the statistical methods implemented in `src/meta_analysis/`.
It covers model assumptions, weight calculations, heterogeneity statistics, and
publication bias assessment.

---

## 1. Effect Measures

### Continuous Outcomes

**Standardized Mean Difference (SMD)**

Used when studies measure the same outcome on different scales.

```
SMD = (mu_treatment - mu_control) / sigma_pooled
```

Cohen's d and Hedges' g are both forms of SMD. The GNF engine maps both to
the "SMD" effect type internally.

**Mean Difference (MD)**

Used when studies share the same measurement scale (e.g., HDRS score change).

```
MD = mu_treatment - mu_control
```

### Binary Outcomes

**Odds Ratio (OR)**

```
OR = (a/b) / (c/d)
```

where a = events in treatment, b = non-events in treatment, etc.

**Risk Ratio (RR)**

```
RR = [a/(a+b)] / [c/(c+d)]
```

**Risk Difference (RD)**

```
RD = a/(a+b) - c/(c+d)
```

### Time-to-Event

**Hazard Ratio (HR)**

Extracted from reported HR and 95% CI. Pooled on the log scale.

---

## 2. Fixed-Effect Model

**Module:** `src/meta_analysis/fixed_effects.py`

The fixed-effect (also called common-effect) model assumes all studies
estimate the **same** true population effect theta.

### Weights

Each study receives a weight proportional to its precision:

```
w_i = 1 / v_i
```

where v_i is the within-study variance, derived from:

```
v_i = SE_i^2 = ((CI_upper - CI_lower) / (2 * z_alpha/2))^2
```

### Pooled Estimate

```
theta_bar = sum(w_i * theta_i) / sum(w_i)
```

### Variance of Pooled Estimate

```
Var(theta_bar) = 1 / sum(w_i)
```

### 95% Confidence Interval

```
theta_bar +/- z_0.025 * SE(theta_bar)
```

### Z-Test

```
z = theta_bar / SE(theta_bar),    p = 2 * Phi(-|z|)
```

**Assumption:** Between-study variance tau^2 = 0 (fixed true effect).

When heterogeneity is present, the fixed-effect model underestimates uncertainty.
Use the random-effects model in that case.

---

## 3. Random-Effects Model

**Module:** `src/meta_analysis/random_effects.py`

The random-effects model treats each study's true effect as a draw from a
normal distribution with mean theta and variance tau^2 (between-study variance).

```
theta_i | theta, tau^2 ~ Normal(theta, tau^2)
theta_i_hat | theta_i, v_i ~ Normal(theta_i, v_i)
```

### Step 1: Estimate tau^2 (DerSimonian-Laird)

The DL moment estimator:

```
c = sum(w_i) - sum(w_i^2) / sum(w_i)

tau^2 = max(0, (Q - (k-1)) / c)
```

where Q is Cochran's Q statistic (see Section 4), k is the number of studies.

### REML Estimator (via rpy2)

When R/metafor is available, tau^2 can be estimated by restricted maximum
likelihood (`rma(method="REML")`), which generally has lower bias than DL
for small k.

### Step 2: Random-Effects Weights

```
w_i* = 1 / (v_i + tau^2)
```

### Step 3: Pooled Estimate

Same formula as FE, substituting w_i* for w_i.

### Step 4: Prediction Interval

The prediction interval quantifies where 95% of true effects would lie
in future studies:

```
theta_bar +/- t_{k-2, 0.025} * sqrt(tau^2 + Var(theta_bar))
```

(Higgins et al. 2009; uses t-distribution with k-2 df)

### Knapp-Hartung Correction

Optional (`knha=True`): replaces z-distribution with t-distribution,
recommended for small k (< 5 studies).

---

## 4. Heterogeneity

**Module:** `src/meta_analysis/heterogeneity.py`

### Cochran's Q Test

```
Q = sum(w_i * (theta_i - theta_bar)^2)
```

Under H0 (homogeneity), Q ~ chi^2_{k-1}.

**Caution:** Q has low power for small k and high power for large k,
regardless of true heterogeneity magnitude.

### I^2 Statistic

I^2 estimates the proportion of total variance due to between-study variance:

```
I^2 = max(0, (Q - (k-1)) / Q) * 100%
```

Confidence intervals for I^2 use the Higgins-Thompson method (chi-squared
quantiles for Q, then transformed).

**Interpretation (Higgins et al. BMJ 2003):**

| I^2 | Interpretation |
|----|----------------|
| 0-25% | Low heterogeneity |
| 25-50% | Moderate |
| 50-75% | High |
| 75-100% | Very high |

### H^2 Statistic

```
H^2 = Q / (k-1)
```

H^2 = 1 when I^2 = 0; H^2 increases with heterogeneity.

### tau^2 Estimators

Three tau^2 estimators are implemented:

1. **DerSimonian-Laird (DL)**: Moment-based, computationally simple.
2. **REML**: Maximum likelihood via rpy2/metafor. Lower bias than DL.
3. **Paule-Mandel (PM)**: Iterative generalized least squares.

---

## 5. Forest Plot

**Module:** `src/meta_analysis/forest_plot.py`

### Visual Elements

- **Horizontal lines**: Per-study 95% CI bars
- **Squares**: Point estimates, area proportional to study weight
- **Diamond**: Pooled estimate (width = 95% CI)
- **Dashed diamond**: Prediction interval (random effects only)
- **Vertical dashed line**: Null value (0 for SMD/MD, 1 for OR/RR/HR)
- **Annotation panel**: Heterogeneity statistics

### Log Scale

For OR, RR, HR, the x-axis is displayed on a log scale.

---

## 6. Publication Bias Assessment

**Module:** `src/meta_analysis/funnel_plot.py`

### Funnel Plot

A scatter plot of effect size (x) vs. standard error (y, inverted).
In the absence of publication bias, points scatter symmetrically around
the pooled estimate in a funnel shape.

### Egger's Regression Test

Regresses the standardized effect (theta_i / SE_i) on precision (1/SE_i):

```
theta_i / SE_i = a + b * (1/SE_i) + epsilon
```

The intercept `a` reflects asymmetry (bias). A significant intercept
(p < 0.05) suggests publication bias.

### Trim-and-Fill

The Duval-Tweedie procedure:
1. Identify and remove the most extreme studies on one side
2. Estimate how many are missing
3. Impute symmetric counterparts
4. Re-compute the pooled estimate

---

## References

1. Borenstein M, et al. Introduction to Meta-Analysis. Wiley, 2009.
2. DerSimonian R, Laird N. Meta-analysis in clinical trials. Control Clin Trials. 1986;7:177-188.
3. Higgins JPT, Thompson SG. Quantifying heterogeneity. Stat Med. 2002;21:1539-1558.
4. Higgins JPT, et al. Measuring inconsistency in meta-analyses. BMJ. 2003;327:557-560.
5. Egger M, et al. Bias in meta-analysis detected by a simple test. BMJ. 1997;315:629-634.
6. Duval S, Tweedie R. Trim and fill. Biometrics. 2000;56:455-463.
7. Viechtbauer W. Conducting meta-analyses in R with the metafor package. J Stat Softw. 2010;36:1-48.
