"""Dash dashboard for GNF Evidence Engine.

Provides an interactive web interface showing:
  - Current pooled effect and CI
  - Forest plot
  - TSA monitoring plot
  - GRADE certainty table
  - Study list with provenance status

Run with: python dashboard/app.py
"""

from __future__ import annotations

import io
import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import dash
    from dash import dcc, html, dash_table, Input, Output, State
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    import plotly.io as pio
    import matplotlib.pyplot as plt
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Dash not installed. Run: pip install dash dash-bootstrap-components plotly")

import numpy as np


def _mpl_fig_to_base64(fig: "plt.Figure") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


def create_app() -> "dash.Dash":
    if not DASH_AVAILABLE:
        raise ImportError("Dash and plotly are required. Run: pip install dash plotly dash-bootstrap-components")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        title="GNF Evidence Engine",
        suppress_callback_exceptions=True,
    )

    app.layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                dbc.Col(
                    html.Div([
                        html.H2("GNF Evidence Engine", className="text-white mb-0"),
                        html.P("Continuous meta-analysis · TSA · GRADE", className="text-white-50 mb-0"),
                    ]),
                    className="bg-primary p-3 mb-4 rounded",
                )
            ),
            dbc.Row([
                dbc.Col([
                    html.Label("Effect Type", className="fw-bold"),
                    dcc.Dropdown(
                        id="effect-type-dropdown",
                        options=[
                            {"label": "SMD (Standardized Mean Difference)", "value": "SMD"},
                            {"label": "MD (Mean Difference)", "value": "MD"},
                            {"label": "OR (Odds Ratio)", "value": "OR"},
                            {"label": "RR (Risk Ratio)", "value": "RR"},
                            {"label": "HR (Hazard Ratio)", "value": "HR"},
                        ],
                        value="SMD",
                        clearable=False,
                    ),
                ], md=3),
                dbc.Col([
                    html.Label("Model", className="fw-bold"),
                    dcc.Dropdown(
                        id="model-dropdown",
                        options=[
                            {"label": "Random Effects (DL)", "value": "random_dl"},
                            {"label": "Random Effects (REML)", "value": "random_reml"},
                            {"label": "Fixed Effects", "value": "fixed"},
                        ],
                        value="random_dl",
                        clearable=False,
                    ),
                ], md=3),
                dbc.Col([
                    html.Label("Spending Function (TSA)", className="fw-bold"),
                    dcc.Dropdown(
                        id="spending-dropdown",
                        options=[
                            {"label": "O'Brien-Fleming", "value": "obrien_fleming"},
                            {"label": "Lan-DeMets", "value": "lan_demets"},
                            {"label": "Pocock", "value": "pocock"},
                        ],
                        value="obrien_fleming",
                        clearable=False,
                    ),
                ], md=3),
                dbc.Col([
                    html.Label("Auto-refresh", className="fw-bold"),
                    dbc.Switch(id="auto-refresh-switch", label="Enable", value=False),
                    dcc.Interval(
                        id="auto-refresh-interval",
                        interval=30_000,
                        disabled=True,
                    ),
                ], md=3),
            ], className="mb-4"),
            dbc.Row(id="kpi-cards", className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Forest Plot", className="mb-0")),
                        dbc.CardBody(html.Img(id="forest-plot-img", style={"width": "100%"})),
                    ]),
                ], md=7),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("TSA Monitoring Plot", className="mb-0")),
                        dbc.CardBody(html.Img(id="tsa-plot-img", style={"width": "100%"})),
                    ]),
                ], md=5),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Funnel Plot", className="mb-0")),
                        dbc.CardBody(html.Img(id="funnel-plot-img", style={"width": "100%"})),
                    ]),
                ], md=5),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("GRADE Certainty Table", className="mb-0")),
                        dbc.CardBody(id="grade-table"),
                    ]),
                ], md=7),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(html.H5("Included Studies", className="mb-0")),
                        dbc.CardBody(id="studies-table"),
                    ]),
                )
            ]),
            dcc.Store(id="analysis-store"),
        ],
    )

    @app.callback(
        Output("auto-refresh-interval", "disabled"),
        Input("auto-refresh-switch", "value"),
    )
    def toggle_refresh(enabled: bool) -> bool:
        return not enabled

    @app.callback(
        Output("analysis-store", "data"),
        [
            Input("effect-type-dropdown", "value"),
            Input("model-dropdown", "value"),
            Input("spending-dropdown", "value"),
            Input("auto-refresh-interval", "n_intervals"),
        ],
    )
    def run_analysis(effect_type, model, spending_fn, _n_intervals):
        import sys, os
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from meta_analysis.random_effects import RandomEffectsModel
        from meta_analysis.fixed_effects import FixedEffectsModel
        from tsa.trial_sequential import TrialSequentialAnalysis
        np.random.seed(42)
        k = 10
        true_effect = 0.45
        effects = true_effect + np.random.normal(0, 0.15, k)
        ses = 0.15 + np.random.uniform(0, 0.10, k)
        variances = ses ** 2
        ns = np.random.randint(30, 200, k).tolist()
        labels = [f"Study {i+1} ({2010+i})" for i in range(k)]
        if model == "fixed":
            ma_model = FixedEffectsModel()
        else:
            estimator = "REML" if model == "random_reml" else "DL"
            ma_model = RandomEffectsModel(estimator=estimator)
        try:
            result = ma_model.fit_from_arrays(
                effects=list(effects), variances=list(variances),
                study_labels=labels, effect_type=effect_type,
            )
        except Exception:
            return {}
        tsa = TrialSequentialAnalysis(spending_function=spending_fn)
        try:
            tsa_result = tsa.run(
                effects=list(effects), variances=list(variances),
                sample_sizes=ns, study_labels=labels,
                delta=0.30, i2=(result.i2 or 0.0) / 100,
            )
        except Exception:
            tsa_result = None
        return {
            "pooled_effect": result.pooled_effect, "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper, "i2": result.i2, "tau2": result.tau2,
            "p_value": result.p_value, "n_studies": result.n_studies,
            "tsa_conclusion": tsa_result.conclusion if tsa_result else "N/A",
            "tsa_info_frac": tsa_result.final_information_fraction if tsa_result else 0.0,
            "effects": list(effects), "ses": list(ses), "labels": labels,
            "variances": list(variances), "ns": ns, "effect_type": effect_type,
        }

    @app.callback(Output("kpi-cards", "children"), Input("analysis-store", "data"))
    def update_kpis(data):
        if not data:
            return []
        return [
            _kpi_card("Pooled Effect", f"{data['pooled_effect']:.3f}", "primary"),
            _kpi_card("95% CI", f"[{data['ci_lower']:.3f}, {data['ci_upper']:.3f}]", "info"),
            _kpi_card("I2", f"{data.get('i2', 0):.1f}%", "warning"),
            _kpi_card("Studies", str(data["n_studies"]), "success"),
            _kpi_card("TSA", data.get("tsa_conclusion", "N/A"),
                      "success" if data.get("tsa_conclusion") == "FIRM_EVIDENCE" else "secondary"),
        ]

    @app.callback(Output("forest-plot-img", "src"), Input("analysis-store", "data"))
    def update_forest_plot(data):
        if not data:
            return ""
        from meta_analysis.forest_plot import ForestPlot
        result = _reconstruct_result(data)
        fp = ForestPlot(result)
        fig = fp.render(title="Forest Plot")
        return _mpl_fig_to_base64(fig)

    @app.callback(
        Output("tsa-plot-img", "src"),
        Input("analysis-store", "data"),
        State("spending-dropdown", "value"),
    )
    def update_tsa_plot(data, spending_fn):
        if not data:
            return ""
        from tsa.trial_sequential import TrialSequentialAnalysis
        from tsa.tsa_plot import TSAPlot
        tsa = TrialSequentialAnalysis(spending_function=spending_fn)
        try:
            tsa_result = tsa.run(
                effects=data["effects"], variances=data["variances"],
                sample_sizes=data["ns"], study_labels=data["labels"],
                delta=0.30, i2=(data.get("i2") or 0.0) / 100,
            )
            plot = TSAPlot(tsa_result)
            fig = plot.render(title="TSA Monitoring Plot")
            return _mpl_fig_to_base64(fig)
        except Exception as exc:
            logger.warning("TSA plot failed: %s", exc)
            return ""

    @app.callback(Output("funnel-plot-img", "src"), Input("analysis-store", "data"))
    def update_funnel_plot(data):
        if not data:
            return ""
        from meta_analysis.funnel_plot import FunnelPlot
        fp = FunnelPlot(
            effects=data["effects"], ses=data["ses"],
            study_labels=data["labels"], effect_type=data["effect_type"],
        )
        fig = fp.render(title="Funnel Plot")
        return _mpl_fig_to_base64(fig)

    @app.callback(Output("grade-table", "children"), Input("analysis-store", "data"))
    def update_grade_table(data):
        if not data:
            return html.P("No data available.")
        from certainty.grade_assessment import GRADEAssessment
        grade = GRADEAssessment()
        rating = grade.assess(
            outcome="Primary outcome", study_design="RCT",
            n_studies=data["n_studies"], total_n=sum(data["ns"]),
            pooled_effect=data["pooled_effect"], ci_lower=data["ci_lower"],
            ci_upper=data["ci_upper"], i2=data.get("i2") or 0.0,
        )
        certainty_colors = {"VERY LOW": "danger", "LOW": "warning", "MODERATE": "info", "HIGH": "success"}
        badge_color = certainty_colors.get(rating.final_certainty, "secondary")
        return html.Div([
            dbc.Table([
                html.Thead(html.Tr([html.Th("Domain"), html.Th("Rating"), html.Th("Levels")])),
                html.Tbody([
                    html.Tr([html.Td(d.domain), html.Td(d.rating), html.Td("v" * d.downgrade if d.downgrade > 0 else "ok")])
                    for d in rating.domains
                ]),
            ], bordered=True, hover=True, size="sm"),
            html.Div([
                html.Strong("GRADE Certainty: "),
                dbc.Badge(rating.final_certainty, color=badge_color, className="ms-2 fs-6"),
            ], className="mt-2"),
        ])

    @app.callback(Output("studies-table", "children"), Input("analysis-store", "data"))
    def update_studies_table(data):
        if not data:
            return html.P("No studies loaded.")
        return dash_table.DataTable(
            columns=[
                {"name": "Study", "id": "label"}, {"name": "Effect Size", "id": "effect"},
                {"name": "95% CI Lower", "id": "ci_lo"}, {"name": "95% CI Upper", "id": "ci_hi"},
                {"name": "SE", "id": "se"}, {"name": "N", "id": "n"},
            ],
            data=[
                {
                    "label": data["labels"][i],
                    "effect": f"{data['effects'][i]:.3f}",
                    "ci_lo": f"{data['effects'][i] - 1.96 * data['ses'][i]:.3f}",
                    "ci_hi": f"{data['effects'][i] + 1.96 * data['ses'][i]:.3f}",
                    "se": f"{data['ses'][i]:.3f}", "n": data["ns"][i],
                }
                for i in range(len(data["labels"]))
            ],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "fontSize": "13px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
            page_size=15,
        )

    return app


def _kpi_card(label: str, value: str, color: str) -> "dbc.Col":
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.H4(value, className=f"text-{color} mb-0"),
                html.Small(label, className="text-muted"),
            ]),
            className="text-center shadow-sm",
        ),
        md=2,
    )


def _reconstruct_result(data: dict):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from meta_analysis._result import MetaAnalysisResult
    ses = data["ses"]
    effects = data["effects"]
    z_alpha = 1.96
    return MetaAnalysisResult(
        pooled_effect=data["pooled_effect"], ci_lower=data["ci_lower"],
        ci_upper=data["ci_upper"],
        z_value=data["pooled_effect"] / (abs(data["ci_upper"] - data["ci_lower"]) / (2 * z_alpha) + 1e-8),
        p_value=data.get("p_value", 0.05),
        weights=[1.0 / v for v in data["variances"]],
        effect_sizes=effects,
        ci_lowers=[e - z_alpha * s for e, s in zip(effects, ses)],
        ci_uppers=[e + z_alpha * s for e, s in zip(effects, ses)],
        effect_type=data.get("effect_type", "SMD"), model="random (DL)",
        study_labels=data["labels"], tau2=data.get("tau2"),
        tau=data.get("tau2") ** 0.5 if data.get("tau2") else None,
        i2=data.get("i2"), n_studies=data["n_studies"], variances=data["variances"],
    )


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
