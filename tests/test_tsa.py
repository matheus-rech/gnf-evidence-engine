"""Tests for TSA modules: trial sequential analysis, information size, spending functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats as stats

from src.tsa.spending_functions import (
    OBrienFleming, LanDeMets, Pocock, HaybittlePeto, get_spending_function,
)
from src.tsa.information_size import RequiredInformationSize
from src.tsa.trial_sequential import TrialSequentialAnalysis


class TestOBrienFleming:
    def test_spent_alpha_at_t0_is_zero(self):
        fn = OBrienFleming(alpha=0.05)
        assert fn.spent_alpha(0.0) == 0.0

    def test_spent_alpha_at_t1_equals_alpha(self):
        fn = OBrienFleming(alpha=0.05)
        assert abs(fn.spent_alpha(1.0) - 0.05) < 1e-8

    def test_spent_alpha_monotone(self):
        fn = OBrienFleming(alpha=0.05)
        ts = np.linspace(0, 1, 50)
        alphas = [fn.spent_alpha(t) for t in ts]
        for a1, a2 in zip(alphas, alphas[1:]):
            assert a2 >= a1 - 1e-10

    def test_conservative_early(self):
        fn = OBrienFleming(alpha=0.05)
        early_spend = fn.spent_alpha(0.10)
        assert early_spend < 0.001

    def test_boundary_z_large_early(self):
        fn = OBrienFleming(alpha=0.05)
        z_early = fn.boundary_z(0.0, 0.10, 0.0)
        assert z_early > 4.0

    def test_name(self):
        fn = OBrienFleming(alpha=0.05)
        assert "O'Brien" in fn.name()


class TestLanDeMets:
    def test_linearity(self):
        fn = LanDeMets(alpha=0.05)
        for t in [0.1, 0.25, 0.5, 0.75, 1.0]:
            expected = 0.05 * t
            assert abs(fn.spent_alpha(t) - expected) < 1e-10

    def test_incremental_alpha_positive(self):
        fn = LanDeMets(alpha=0.05)
        da = fn.incremental_alpha(0.4, 0.6)
        assert da > 0

    def test_name(self):
        fn = LanDeMets()
        assert "Lan" in fn.name()


class TestPocock:
    def test_spent_alpha_at_t0_is_zero(self):
        fn = Pocock(alpha=0.05)
        assert fn.spent_alpha(0.0) == 0.0

    def test_spent_alpha_at_t1_equals_alpha(self):
        fn = Pocock(alpha=0.05)
        assert abs(fn.spent_alpha(1.0) - 0.05) < 1e-8

    def test_spends_more_early_than_obf(self):
        poc = Pocock(alpha=0.05)
        obf = OBrienFleming(alpha=0.05)
        t = 0.25
        assert poc.spent_alpha(t) > obf.spent_alpha(t)

    def test_monotone(self):
        fn = Pocock(alpha=0.05)
        ts = np.linspace(0, 1, 50)
        alphas = [fn.spent_alpha(t) for t in ts]
        for a1, a2 in zip(alphas, alphas[1:]):
            assert a2 >= a1 - 1e-10


class TestHaybittlePeto:
    def test_very_little_spent_interim(self):
        fn = HaybittlePeto(alpha=0.05, interim_z=3.0)
        assert fn.spent_alpha(0.5) < 0.01

    def test_final_boundary_near_alpha(self):
        fn = HaybittlePeto(alpha=0.05, interim_z=3.0)
        z_final = fn.final_boundary_z()
        assert 1.8 < z_final < 2.1


class TestGetSpendingFunction:
    def test_valid_names(self):
        for name in ["obrien_fleming", "lan_demets", "pocock", "haybittle_peto"]:
            fn = get_spending_function(name)
            assert fn is not None

    def test_case_insensitive(self):
        fn = get_spending_function("OBrien_Fleming", alpha=0.05)
        assert isinstance(fn, OBrienFleming)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown spending function"):
            get_spending_function("unknown_function")


class TestRequiredInformationSize:
    def test_continuous_basic(self):
        calc = RequiredInformationSize(alpha=0.05, beta=0.20)
        ris_large = calc.continuous(delta=0.50, sigma=1.0)
        ris_small = calc.continuous(delta=0.20, sigma=1.0)
        assert ris_small.ris_unadjusted > ris_large.ris_unadjusted

    def test_continuous_formula(self):
        alpha, beta = 0.05, 0.20
        delta, sigma = 0.30, 1.0
        z_a = stats.norm.ppf(1 - alpha / 2)
        z_b = stats.norm.ppf(1 - beta)
        expected = 4 * (z_a + z_b) ** 2 * sigma ** 2 / delta ** 2
        calc = RequiredInformationSize(alpha=alpha, beta=beta)
        result = calc.continuous(delta=delta, sigma=sigma)
        assert abs(result.ris_unadjusted - expected) < 0.01

    def test_i2_inflation(self):
        calc = RequiredInformationSize()
        r_no_het = calc.continuous(delta=0.30, sigma=1.0, i2=0.0)
        r_with_het = calc.continuous(delta=0.30, sigma=1.0, i2=0.50)
        assert r_with_het.ris_adjusted > r_no_het.ris_adjusted

    def test_i2_exactly_zero_no_inflation(self):
        calc = RequiredInformationSize()
        r = calc.continuous(delta=0.30, sigma=1.0, i2=0.0)
        assert abs(r.ris_adjusted - r.ris_unadjusted) < 1e-8

    def test_binary_event_rate_based(self):
        calc = RequiredInformationSize()
        r_large_rrr = calc.binary(p_control=0.30, relative_risk_reduction=0.30)
        r_small_rrr = calc.binary(p_control=0.30, relative_risk_reduction=0.10)
        assert r_small_rrr.ris_unadjusted > r_large_rrr.ris_unadjusted

    def test_invalid_delta_raises(self):
        calc = RequiredInformationSize()
        with pytest.raises(ValueError):
            calc.continuous(delta=0.0, sigma=1.0)

    def test_invalid_sigma_raises(self):
        calc = RequiredInformationSize()
        with pytest.raises(ValueError):
            calc.continuous(delta=0.30, sigma=-1.0)

    def test_from_smd_uses_sigma_1(self):
        calc = RequiredInformationSize()
        r1 = calc.from_smd(delta_smd=0.30)
        r2 = calc.continuous(delta=0.30, sigma=1.0)
        assert abs(r1.ris_unadjusted - r2.ris_unadjusted) < 1e-8

    def test_higher_power_requires_more_information(self):
        c80 = RequiredInformationSize(beta=0.20)
        c90 = RequiredInformationSize(beta=0.10)
        r80 = c80.continuous(delta=0.30, sigma=1.0)
        r90 = c90.continuous(delta=0.30, sigma=1.0)
        assert r90.ris_unadjusted > r80.ris_unadjusted


class TestTrialSequentialAnalysis:
    """Integration tests for the TrialSequentialAnalysis engine."""

    LARGE_EFFECTS = [-0.80] * 12
    SMALL_VARS = [0.02] * 12
    SAMPLE_SIZES = [100] * 12
    SMALL_EFFECTS = [-0.10] * 8
    LARGE_VARS = [0.25] * 8
    SMALL_NS = [30] * 8

    def test_firm_evidence_large_consistent_effects(self):
        tsa = TrialSequentialAnalysis(alpha=0.05, beta=0.20, spending_function="obrien_fleming")
        result = tsa.run(
            effects=self.LARGE_EFFECTS, variances=self.SMALL_VARS,
            sample_sizes=self.SAMPLE_SIZES, delta=0.30, sigma=1.0, i2=0.0,
        )
        assert result.conclusion in ("FIRM_EVIDENCE", "HARM_SIGNAL")

    def test_insufficient_small_effects(self):
        tsa = TrialSequentialAnalysis(alpha=0.05, beta=0.20, spending_function="obrien_fleming")
        result = tsa.run(
            effects=self.SMALL_EFFECTS, variances=self.LARGE_VARS,
            sample_sizes=self.SMALL_NS, delta=0.30, sigma=1.0, i2=0.0,
        )
        assert result.conclusion != "FIRM_EVIDENCE"
        assert result.conclusion in ("INSUFFICIENT", "FUTILE")

    def test_n_steps_equals_n_studies(self):
        tsa = TrialSequentialAnalysis()
        result = tsa.run(effects=[-0.5] * 6, variances=[0.05] * 6, sample_sizes=[80] * 6, delta=0.30, sigma=1.0)
        assert len(result.steps) == 6

    def test_monotone_cumulative_n(self):
        tsa = TrialSequentialAnalysis()
        result = tsa.run(effects=[-0.5] * 6, variances=[0.05] * 6, sample_sizes=[80] * 6, delta=0.30, sigma=1.0)
        ns = [step.n_cumulative for step in result.steps]
        for n1, n2 in zip(ns, ns[1:]):
            assert n2 > n1

    def test_information_fraction_in_0_1(self):
        tsa = TrialSequentialAnalysis()
        result = tsa.run(effects=[-0.5] * 6, variances=[0.05] * 6, sample_sizes=[80] * 6, delta=0.30, sigma=1.0)
        for step in result.steps:
            assert 0.0 <= step.information_fraction <= 1.0

    def test_boundary_positive(self):
        tsa = TrialSequentialAnalysis()
        result = tsa.run(effects=[-0.5] * 6, variances=[0.05] * 6, sample_sizes=[80] * 6, delta=0.30, sigma=1.0)
        for step in result.steps:
            if not math.isinf(step.boundary_upper):
                assert step.boundary_upper > 0

    def test_tsa_adjusted_ci_wider_when_insufficient(self):
        tsa = TrialSequentialAnalysis()
        result = tsa.run(effects=[-0.3] * 5, variances=[0.09] * 5, sample_sizes=[50] * 5, delta=0.30, sigma=1.0)
        final = result.steps[-1]
        conv_width = 2 * 1.96 * final.pooled_se
        tsa_width = result.tsa_adjusted_ci_upper - result.tsa_adjusted_ci_lower
        if result.conclusion == "INSUFFICIENT":
            assert tsa_width >= conv_width - 1e-4

    def test_requires_at_least_2_studies(self):
        tsa = TrialSequentialAnalysis()
        with pytest.raises(ValueError, match="least 2"):
            tsa.run(effects=[-0.5], variances=[0.04], sample_sizes=[80])

    def test_lan_demets_spends_faster(self):
        data = {"effects": [-0.7] * 10, "variances": [0.03] * 10, "sample_sizes": [120] * 10}
        obf_tsa = TrialSequentialAnalysis(spending_function="obrien_fleming")
        ld_tsa = TrialSequentialAnalysis(spending_function="lan_demets")
        obf_result = obf_tsa.run(**data, delta=0.30, sigma=1.0)
        ld_result = ld_tsa.run(**data, delta=0.30, sigma=1.0)
        if obf_result.conclusion_at_step and ld_result.conclusion_at_step:
            assert ld_result.conclusion_at_step <= obf_result.conclusion_at_step

    def test_ris_adjusted_for_heterogeneity(self):
        tsa = TrialSequentialAnalysis()
        result = tsa.run(effects=[-0.5] * 6, variances=[0.05] * 6, sample_sizes=[80] * 6, delta=0.30, sigma=1.0, i2=0.40)
        assert result.ris_result.ris_adjusted > result.ris_result.ris_unadjusted
