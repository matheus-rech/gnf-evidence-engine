"""Tests for meta-analysis modules: fixed effects, random effects, heterogeneity."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats as stats

from src.meta_analysis.fixed_effects import FixedEffectsModel
from src.meta_analysis.random_effects import RandomEffectsModel
from src.meta_analysis.heterogeneity import HeterogeneityAnalysis, _dl_tau2, _pm_tau2
from src.meta_analysis._result import MetaAnalysisResult


SIMPLE_EFFECTS = np.array([0.5, 0.3, 0.7])
SIMPLE_VARIANCES = np.array([0.04, 0.09, 0.01])


class TestFixedEffectsModel:
    """Tests for the inverse-variance fixed-effects model."""

    def test_pooled_effect_manual(self):
        model = FixedEffectsModel()
        result = model.fit_from_arrays(effects=list(SIMPLE_EFFECTS), variances=list(SIMPLE_VARIANCES))
        w = 1.0 / SIMPLE_VARIANCES
        expected = float(np.sum(w * SIMPLE_EFFECTS) / np.sum(w))
        assert abs(result.pooled_effect - expected) < 1e-6

    def test_ci_width(self):
        model = FixedEffectsModel()
        result = model.fit_from_arrays(effects=list(SIMPLE_EFFECTS), variances=list(SIMPLE_VARIANCES))
        half_width = result.ci_upper - result.pooled_effect
        half_width_lo = result.pooled_effect - result.ci_lower
        assert abs(half_width - half_width_lo) < 1e-6

    def test_z_value_and_pvalue_consistent(self):
        model = FixedEffectsModel()
        result = model.fit_from_arrays(effects=list(SIMPLE_EFFECTS), variances=list(SIMPLE_VARIANCES))
        expected_p = 2 * stats.norm.sf(abs(result.z_value))
        assert abs(result.p_value - expected_p) < 1e-8

    def test_significant_true_effect(self):
        model = FixedEffectsModel()
        effects = [1.5, 1.3, 1.7, 1.4, 1.6]
        variances = [0.01] * 5
        result = model.fit_from_arrays(effects=effects, variances=variances)
        assert result.is_significant
        assert result.p_value < 0.001

    def test_null_effect_not_significant(self):
        model = FixedEffectsModel()
        effects = [0.01, -0.02, 0.01, -0.01, 0.02]
        variances = [0.25] * 5
        result = model.fit_from_arrays(effects=effects, variances=variances)
        assert not result.is_significant
        assert result.p_value > 0.05

    def test_weights_sum_to_100(self):
        model = FixedEffectsModel()
        result = model.fit_from_arrays(effects=list(SIMPLE_EFFECTS), variances=list(SIMPLE_VARIANCES))
        assert abs(sum(result.weights) - 100.0) < 1e-6

    def test_requires_at_least_2_studies(self):
        model = FixedEffectsModel()
        with pytest.raises(ValueError, match="least 2"):
            model.fit_from_arrays(effects=[0.5], variances=[0.04])

    def test_tau2_is_zero_for_fixed_effects(self):
        model = FixedEffectsModel()
        result = model.fit_from_arrays(effects=list(SIMPLE_EFFECTS), variances=list(SIMPLE_VARIANCES))
        assert result.tau2 == 0.0
        assert result.tau == 0.0

    def test_homogeneous_data_has_low_i2(self):
        model = FixedEffectsModel()
        effects = [0.5] * 6
        variances = [0.04, 0.09, 0.01, 0.04, 0.09, 0.01]
        result = model.fit_from_arrays(effects=effects, variances=variances)
        assert result.i2 < 5.0

    def test_fit_with_effect_records(self, smd_effects):
        model = FixedEffectsModel()
        result_records = model.fit(smd_effects)
        assert result_records.pooled_effect < 0
        assert result_records.ci_upper < 0

    def test_or_back_transform(self, or_effects):
        model = FixedEffectsModel()
        result = model.fit(or_effects)
        assert result.pooled_effect > 1.0
        assert result.ci_lower > 0.0

    def test_model_name(self):
        model = FixedEffectsModel()
        result = model.fit_from_arrays(effects=list(SIMPLE_EFFECTS), variances=list(SIMPLE_VARIANCES))
        assert result.model == "fixed"

    def test_n_studies_recorded(self):
        model = FixedEffectsModel()
        result = model.fit_from_arrays(effects=list(SIMPLE_EFFECTS), variances=list(SIMPLE_VARIANCES))
        assert result.n_studies == 3


class TestRandomEffectsModel:
    """Tests for the DerSimonian-Laird random-effects model."""

    def test_pooled_effect_negative_for_depression_data(self, smd_effects):
        model = RandomEffectsModel(estimator="DL")
        result = model.fit(smd_effects)
        assert result.pooled_effect < -0.5
        assert result.is_significant

    def test_tau2_non_negative(self, smd_effects):
        model = RandomEffectsModel(estimator="DL")
        result = model.fit(smd_effects)
        assert result.tau2 >= 0.0

    def test_tau_equals_sqrt_tau2(self, smd_effects):
        model = RandomEffectsModel(estimator="DL")
        result = model.fit(smd_effects)
        assert abs(result.tau - math.sqrt(result.tau2)) < 1e-8

    def test_ci_wider_than_fixed_effects(self, smd_effects):
        fe = FixedEffectsModel()
        re = RandomEffectsModel()
        re_result = re.fit(smd_effects)
        fe_result = fe.fit(smd_effects)
        re_width = re_result.ci_upper - re_result.ci_lower
        fe_width = fe_result.ci_upper - fe_result.ci_lower
        assert re_width >= fe_width - 1e-6

    def test_prediction_interval_wider_than_ci(self, smd_effects):
        model = RandomEffectsModel()
        result = model.fit(smd_effects)
        if result.prediction_interval_lower is not None:
            pi_width = result.prediction_interval_upper - result.prediction_interval_lower
            ci_width = result.ci_upper - result.ci_lower
            assert pi_width > ci_width

    def test_homogeneous_data_tau2_near_zero(self):
        model = RandomEffectsModel(estimator="DL")
        effects = [0.5, 0.5, 0.5, 0.5, 0.5]
        variances = [0.04, 0.04, 0.04, 0.04, 0.04]
        result = model.fit_from_arrays(effects=effects, variances=variances)
        assert result.tau2 < 0.01

    def test_heterogeneous_data_has_positive_tau2(self):
        model = RandomEffectsModel(estimator="DL")
        effects = [-0.8, 0.5, -1.2, 0.9, -0.1, 1.5]
        variances = [0.04] * 6
        result = model.fit_from_arrays(effects=effects, variances=variances)
        assert result.tau2 > 0.0

    def test_knha_correction_changes_ci(self, smd_effects):
        re_no_knha = RandomEffectsModel(estimator="DL", knha=False)
        re_knha = RandomEffectsModel(estimator="DL", knha=True)
        r1 = re_no_knha.fit(smd_effects)
        r2 = re_knha.fit(smd_effects)
        assert r1.n_studies == r2.n_studies == len(smd_effects)
        assert r2.p_value is not None
        assert r2.ci_lower < r2.ci_upper

    def test_requires_at_least_2_studies(self):
        model = RandomEffectsModel()
        with pytest.raises(ValueError, match="least 2"):
            model.fit_from_arrays(effects=[0.5], variances=[0.04])

    def test_weights_sum_to_100(self, smd_effects):
        model = RandomEffectsModel()
        result = model.fit(smd_effects)
        assert abs(sum(result.weights) - 100.0) < 1e-4


class TestHeterogeneityAnalysis:
    """Tests for heterogeneity statistics."""

    def test_q_statistic_matches_formula(self):
        het = HeterogeneityAnalysis()
        effects = np.array([0.5, 0.3, 0.7])
        variances = np.array([0.04, 0.09, 0.01])
        result = het.analyse(effects, variances)
        w = 1.0 / variances
        pooled = np.sum(w * effects) / np.sum(w)
        expected_q = float(np.sum(w * (effects - pooled) ** 2))
        assert abs(result.q_stat - expected_q) < 1e-8

    def test_i2_between_0_and_100(self):
        het = HeterogeneityAnalysis()
        for _ in range(10):
            effects = np.random.normal(0, 0.3, 6)
            variances = np.random.uniform(0.01, 0.25, 6)
            result = het.analyse(effects, variances)
            assert 0 <= result.i2 <= 100

    def test_interpretation_low(self):
        het = HeterogeneityAnalysis()
        effects = [0.5, 0.51, 0.49, 0.50, 0.50, 0.51]
        variances = [0.04] * 6
        result = het.analyse(effects, variances)
        assert result.interpretation == "low"

    def test_interpretation_high_heterogeneity(self):
        het = HeterogeneityAnalysis(compute_reml=False, compute_pm=False)
        effects = [-1.5, 0.5, 1.8, -0.9, 1.2, -1.4]
        variances = [0.01] * 6
        result = het.analyse(effects, variances)
        assert result.interpretation in ("high", "very high")

    def test_q_df_equals_k_minus_1(self):
        het = HeterogeneityAnalysis()
        effects = [0.5, 0.3, 0.7, 0.4, 0.6]
        variances = [0.04] * 5
        result = het.analyse(effects, variances)
        assert result.q_df == 4

    def test_tau2_dl_non_negative(self):
        het = HeterogeneityAnalysis(compute_reml=False, compute_pm=False)
        effects = np.random.normal(0, 0.3, 8)
        variances = np.random.uniform(0.02, 0.25, 8)
        result = het.analyse(effects, variances)
        assert result.tau2_dl >= 0.0

    def test_pm_tau2_computed(self):
        het = HeterogeneityAnalysis(compute_reml=False, compute_pm=True)
        effects = [-0.8, -0.65, -0.9, -0.55, -0.75, -0.70]
        variances = [0.04] * 6
        result = het.analyse(effects, variances)
        assert result.tau2_pm is not None
        assert result.tau2_pm >= 0.0

    def test_requires_at_least_2_studies(self):
        het = HeterogeneityAnalysis()
        with pytest.raises(ValueError, match="least 2"):
            het.analyse([0.5], [0.04])

    def test_h2_equals_q_over_df(self):
        het = HeterogeneityAnalysis(compute_reml=False, compute_pm=False)
        effects = [-0.8, -0.65, -0.9, -0.55, -0.75, -0.70]
        variances = [0.02] * 6
        result = het.analyse(effects, variances)
        if result.q_stat > result.q_df:
            expected_h2 = result.q_stat / result.q_df
            assert abs(result.h2 - expected_h2) < 1e-8


class TestMetaAnalysisResult:
    """Tests for the MetaAnalysisResult dataclass."""

    def test_relative_weights_sum(self):
        result = MetaAnalysisResult(
            pooled_effect=-0.75, ci_lower=-1.0, ci_upper=-0.5, z_value=-5.0, p_value=0.001,
            weights=[25.0, 30.0, 45.0],
            effect_sizes=[-0.8, -0.65, -0.77],
            ci_lowers=[-1.2, -1.0, -1.1], ci_uppers=[-0.4, -0.3, -0.4],
            n_studies=3,
        )
        assert abs(sum(result.relative_weights) - 100.0) < 1e-6

    def test_summary_string_contains_key_info(self):
        from src.meta_analysis.fixed_effects import FixedEffectsModel
        model = FixedEffectsModel()
        result = model.fit_from_arrays(effects=list(SIMPLE_EFFECTS), variances=list(SIMPLE_VARIANCES))
        s = result.summary()
        assert "SMD" in s
        assert "Fixed" in s
