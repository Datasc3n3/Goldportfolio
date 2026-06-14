"""
Goud Portfolio Dashboard
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils import (
    load_portfolio, calculate_pure_gold_weight,
    filter_bars, calculate_portfolio_value,
    goal_progress, next_purchase_date,
    format_eur, format_gram, KARAT_PURITY
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed"

# Data laden
try:
    gold = pd.read_csv(PROC_DIR / "gold_with_macro.csv", index_col=0, parse_dates=True)
    gold.index = pd.to_datetime(gold.index)
    portfolio_raw = load_portfolio()
    portfolio = pd.read_csv(PROC_DIR / "portfolio_cleaned.csv")
    portfolio["datum_aankoop"] = pd.to_datetime(portfolio["datum_aankoop"])
    bars = pd.read_csv(PROC_DIR / "bars_with_spot.csv")
    bars["datum_aankoop"] = pd.to_datetime(bars["datum_aankoop"])
    DATA_LOADED = True
except Exception as e:
    print(f"Waarschuwing data laden: {e}")
    DATA_LOADED = False

TROY_OZ_TO_GRAM = 31.1035
GOAL = 100_000
PRICE_COL = "gold_eur_gram"

# Portfolio statistieken
if DATA_LOADED:
    portfolio_calc = calculate_pure_gold_weight(portfolio)
    stats = calculate_portfolio_value(portfolio, gold)
    goal_stats = goal_progress(stats["total_portfolio_value_eur"], GOAL)
    laatste_aankoop = pd.Timestamp(bars["datum_aankoop"].max())
    volgende = next_purchase_date(laatste_aankoop, interval_months=2)
    current_spot = gold[PRICE_COL].iloc[-1]

    # Historische rendementen (voor scenario-definitie)
    yearly_prices = gold[PRICE_COL].resample("YE").last().dropna()
    yearly_returns = yearly_prices.pct_change().dropna()
    p25_hist = yearly_returns.quantile(0.25)
    p50_hist = yearly_returns.median()
    # 6-jaar CAGR
    cagr_6yr = (gold[PRICE_COL].iloc[-1] / gold[PRICE_COL].iloc[0]) ** (1 / 6) - 1
    bars_roi = ((stats["current_value_bars_eur"] - stats["total_invested_eur"]) / stats["total_invested_eur"] * 100) if stats.get("total_invested_eur", 0) > 0 else 0
else:
    stats, goal_stats = {}, {}
    volgende = datetime.now()
    current_spot = 0
    p25_hist, p50_hist, cagr_6yr = 0.05, 0.075, 0.18
    bars_roi = 0.0

# Scenario definities
# Scenario set A: Historisch conservatief (2020–2026 data)
SCENARIOS_HIST = {
    "Bearish":  {"rate": p25_hist,   "label": f"+{p25_hist:.0%}/jr",  "color": "#C0392B", "beschrijving": "Terugval naar pre-2024 tempo"},
    "Neutraal": {"rate": p50_hist,   "label": f"+{p50_hist:.0%}/jr",  "color": "#B8860B", "beschrijving": "Mediaan 2020–2026"},
    "Bullish":  {"rate": cagr_6yr,   "label": f"+{cagr_6yr:.0%}/jr", "color": "#1A7A4A", "beschrijving": "6-jaar historisch CAGR"},
}

# Scenario set B: Klimaat & Geopolitiek (huidig klimaat)
SCENARIOS_KLIMAAT = {
    "Bearish":  {"rate": 0.10, "label": "+10%/jr", "color": "#C0392B", "beschrijving": "Goud vertraagt - Goldman Sachs base case"},
    "Neutraal": {"rate": 0.18, "label": "+18%/jr", "color": "#B8860B", "beschrijving": "6-jaar CAGR = huidige realiteit (GS bullish)"},
    "Bullish":  {"rate": 0.35, "label": "+35%/jr", "color": "#1A7A4A", "beschrijving": "2024–2025 tempo houdt aan (geopolitieke escalatie)"},
}

GEM_PREMIUM = 0.0371   # Gemiddelde dealer premium op basis van historische aankopen
BAR_GRAM    = 10       # Gram per nieuwe aankoop
INTERVAL    = 2        # Maanden tussen aankopen
JAREN       = 3


def bereken_projectie(scenarios: dict, huidig: float, spot: float) -> dict:
    """Bereken 3-jaar portfolio projectie inclusief nieuwe aankopen elke 2 maanden."""
    maanden = JAREN * 12
    dates = pd.date_range(start=datetime.now(), periods=maanden + 1, freq="ME")
    result = {}
    for naam, s in scenarios.items():
        monthly_rate = (1 + s["rate"]) ** (1 / 12) - 1
        waarde = huidig
        sp = spot
        values = [waarde]
        extra_inv = 0
        maand_hit = None
        for m in range(1, maanden + 1):
            sp = sp * (1 + monthly_rate)
            if m % INTERVAL == 0:
                waarde += BAR_GRAM * sp
                extra_inv += BAR_GRAM * sp * (1 + GEM_PREMIUM)
            waarde = waarde * (1 + monthly_rate)
            values.append(waarde)
            if waarde >= GOAL and maand_hit is None:
                maand_hit = m
        result[naam] = {
            **s,
            "values": values,
            "dates": dates,
            "extra_inv": extra_inv,
            "eind": values[-1],
            "bereikt": values[-1] >= GOAL,
            "maand_hit": maand_hit,
        }
    return result


if DATA_LOADED:
    proj_hist    = bereken_projectie(SCENARIOS_HIST,    stats["total_portfolio_value_eur"], current_spot)
    proj_klimaat = bereken_projectie(SCENARIOS_KLIMAAT, stats["total_portfolio_value_eur"], current_spot)


# Kleur constanten
C = {
    "bg":       "#F7F8FA",
    "white":    "#FFFFFF",
    "border":   "#E5E7EB",
    "text":     "#1A1A2E",
    "muted":    "#6B7280",
    "gold":     "#B8860B",
    "blue":     "#2C5F8A",
    "green":    "#1A7A4A",
    "red":      "#C0392B",
    "light_bg": "#FAFAFA",
}

def base_layout(ytitle="", xtitle="", rangemode="tozero", margin=None, extra_legend=None):
    """Geeft een volledige layout dict terug - nooit conflicten met update_layout."""
    leg = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"], size=9),
               orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0,
               tracegroupgap=10)
    if extra_legend:
        leg.update(extra_legend)
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="Inter, sans-serif", size=11),
        margin=margin or dict(l=10, r=10, t=50, b=10),
        xaxis=dict(title=xtitle, gridcolor="#F3F4F6", zeroline=False,
                   showline=True, linecolor=C["border"]),
        yaxis=dict(title=ytitle, gridcolor="#F3F4F6", zeroline=True,
                   zerolinecolor=C["border"], showline=True, linecolor=C["border"],
                   rangemode=rangemode),
        legend=leg,
    )


# Hulpfuncties layout
def fmt_k(val: float) -> str:
    """Format getal als K of M voor at-a-glance leesbaarheid."""
    if abs(val) >= 1_000_000:
        return f"€{val/1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"€{val/1_000:.1f}K"
    return f"€{val:.0f}"


def kpi_card(label: str, value: str, sub: str = "", trend: float = None, accent: str = "gold"):
    trend_el = None
    if trend is not None:
        cls = "kpi-trend-pos" if trend >= 0 else "kpi-trend-neg"
        arrow = "▲" if trend >= 0 else "▼"
        trend_el = html.Div(f"{arrow} {abs(trend):.1f}%", className=cls)

    return dbc.Col(
        html.Div(className=f"kpi-card {'blue' if accent == 'blue' else 'green' if accent == 'green' else 'neutral' if accent == 'neutral' else ''}",
            children=[
                html.Div(label, className="kpi-label"),
                html.Div(value, className="kpi-value"),
                html.Div(sub,   className="kpi-sub") if sub else None,
                trend_el,
            ]
        ),
        xs=6, sm=6, md=3, lg=3, className="mb-3",
    )


def chart_card(title: str, subtitle: str, graph_id: str, height: int = 340):
    return html.Div(className="chart-card", children=[
        html.Div(title,    className="chart-card-title"),
        html.Div(subtitle, className="chart-card-subtitle"),
        dcc.Graph(id=graph_id, config={"displayModeBar": False},
                  style={"height": f"{height}px"}),
    ])


def scenario_resultaten(proj: dict):
    rows = []
    for naam, d in proj.items():
        bereikt_txt  = "✓ Doel bereikt" if d["bereikt"] else "✗ Doel niet bereikt"
        bereikt_color = C["green"] if d["bereikt"] else C["red"]
        wanneer = ""
        if d["bereikt"] and d["maand_hit"]:
            dt = d["dates"][d["maand_hit"]]
            wanneer = f" ({dt.strftime('%b %Y')})"

        rows.append(html.Div(className="scenario-result-row", children=[
            html.Div(style={"display": "flex", "alignItems": "center", "flex": "1"}, children=[
                html.Div(style={"width": "10px", "height": "10px", "borderRadius": "50%",
                                "backgroundColor": d["color"], "marginRight": "10px"}),
                html.Div([
                    html.Span(naam, style={"fontWeight": "600", "fontSize": "0.85rem"}),
                    html.Span(f" {d['label']}", style={"color": C["muted"], "fontSize": "0.8rem"}),
                ])
            ]),
            html.Div([
                html.Div(fmt_k(d["eind"]),
                         style={"fontWeight": "700", "fontSize": "0.95rem", "textAlign": "right"}),
                html.Div(bereikt_txt + wanneer,
                         style={"color": bereikt_color, "fontSize": "0.72rem", "textAlign": "right"}),
            ])
        ]))
    return html.Div(rows)


FILL_COLORS = {
    "#C0392B": "rgba(192,57,43,0.07)",
    "#B8860B": "rgba(184,134,11,0.07)",
    "#1A7A4A": "rgba(26,122,74,0.07)",
}


def scenario_chart_figure(proj: dict, titel: str) -> go.Figure:
    fig = go.Figure()

    fig.add_hline(
        y=GOAL, line_dash="dot", line_color=C["blue"], line_width=1.5,
        annotation=dict(text=f"Doel € {GOAL/1000:.0f}K", font=dict(color=C["blue"], size=10),
                        xanchor="left"),
    )

    for naam, d in proj.items():
        fig.add_trace(go.Scatter(
            x=d["dates"], y=d["values"],
            name=f"{naam} ({d['label']})",
            line=dict(color=d["color"], width=2.5),
            fill="tozeroy",
            fillcolor=FILL_COLORS.get(d["color"], "rgba(0,0,0,0.05)"),
            hovertemplate=f"{naam}<br>%{{x|%b %Y}}<br>€%{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="Inter, sans-serif", size=11),
        margin=dict(l=10, r=10, t=36, b=50),
        title=dict(text=titel, font=dict(size=12, color=C["muted"]), x=0),
        xaxis=dict(gridcolor="#F3F4F6", zeroline=False, showline=True, linecolor=C["border"]),
        yaxis=dict(gridcolor="#F3F4F6", zeroline=True, zerolinecolor=C["border"],
                   showline=True, linecolor=C["border"],
                   title="Waarde (EUR)", tickformat=",.0f", rangemode="tozero"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"], size=9),
                    orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        showlegend=True,
    )
    return fig


# App initialisatie
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Goud Portfolio",
)
server = app.server


# Tab layouts

# TAB 1: Overzicht
overzicht_layout = html.Div(className="page-container", children=[

    # KPI rij
    html.Div("Portefeuille op een blik", className="section-title"),
    dbc.Row([
        kpi_card("Geïnvesteerd (staven)",
                 fmt_k(stats.get("total_invested_eur", 0)),
                 f"{stats.get('total_bars', 0)} goudstaven, 110g 24k"),
        kpi_card("Staven rendement",
                 f"{bars_roi:+.1f}%",
                 f"Staven op spot: {fmt_k(stats.get('current_value_bars_eur', 0))}",
                 trend=bars_roi,
                 accent="neutral"),
        kpi_card("Totale goudwaarde",
                 fmt_k(stats.get("total_portfolio_value_eur", 0)),
                 f"Staven + sieraden (erfgoed) | Spot: {fmt_k(stats.get('current_spot_eur_gram', 0))}/g",
                 accent="blue"),
        kpi_card("Doel voortgang",
                 f"{goal_stats.get('progress_percent', 0):.1f}%",
                 f"Nog {fmt_k(goal_stats.get('remaining_eur', 0))} te gaan naar EUR 100.000",
                 accent="blue"),
    ]),

    # Voortgangsbalk naar doel
    html.Div([
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "fontSize": "0.75rem", "color": C["muted"], "marginBottom": "4px"},
                 children=[
                     html.Span("€ 0"),
                     html.Span(f"Doel: € {GOAL:,.0f}"),
                 ]),
        html.Div(className="progress-wrap", children=[
            html.Div(className="progress-fill",
                     style={"width": f"{min(goal_stats.get('progress_percent', 0), 100):.1f}%"}),
        ]),
        html.Div(f"{goal_stats.get('progress_percent', 0):.1f}% bereikt",
                 style={"fontSize": "0.72rem", "color": C["muted"], "marginTop": "4px"}),
    ], className="mb-4"),

    html.Hr(className="divider"),

    # Grafieken rij
    dbc.Row([
        dbc.Col(chart_card(
            "Goudprijs EUR/gram (2020 – heden)",
            "Met jouw aankoopmomenten - diamonds = jouw aankopen",
            "gold-price-chart", height=340
        ), xs=12, lg=8),
        dbc.Col(chart_card(
            "Portefeuille samenstelling",
            "Verdeling goudstaven vs sieraden (gram)",
            "portfolio-pie", height=340
        ), xs=12, lg=4),
    ]),

    dbc.Row([
        dbc.Col(chart_card(
            "Aankoopprijs vs spotprijs",
            "Jouw betaalde prijs per gram vs marktprijs op aankoopdatum",
            "purchase-vs-spot", height=300
        ), xs=12, lg=7),
        dbc.Col(
            html.Div(className="chart-card", style={"height": "100%"}, children=[
                html.Div("Portfolio details", className="chart-card-title"),
                html.Div("Pure goudgewicht & waarde per categorie",
                         className="chart-card-subtitle"),
                html.Div(id="portfolio-details"),
            ]),
            xs=12, lg=5,
        ),
    ]),
])


def scenario_tab(tab_id_prefix: str, titel: str, subtitel: str,
                 bron_tekst: str, proj: dict) -> html.Div:
    """Genereer een scenario tab layout."""
    huidig = stats.get("total_portfolio_value_eur", 0) if DATA_LOADED else 0
    benodigd = (GOAL / max(huidig, 1)) ** (1 / JAREN) - 1

    return html.Div(className="page-container", children=[
        html.Div(titel, className="section-title"),
        html.Div(subtitel, style={"fontSize": "0.82rem", "color": C["muted"], "marginBottom": "16px"}),

        # Scenario KPI's
        dbc.Row([
            kpi_card("Bearish",
                     fmt_k(proj["Bearish"]["eind"]),
                     proj["Bearish"]["label"],
                     trend=proj["Bearish"]["rate"] * 100,
                     accent="neutral"),
            kpi_card("Neutraal",
                     fmt_k(proj["Neutraal"]["eind"]),
                     proj["Neutraal"]["label"],
                     trend=proj["Neutraal"]["rate"] * 100,
                     accent="gold"),
            kpi_card("Bullish",
                     fmt_k(proj["Bullish"]["eind"]),
                     proj["Bullish"]["label"],
                     trend=proj["Bullish"]["rate"] * 100,
                     accent="green"),
            kpi_card("Benodigd rendement",
                     f"{benodigd:.0%}/jr",
                     f"Om € {GOAL/1000:.0f}K te bereiken in {JAREN} jaar",
                     accent="blue"),
        ]),

        html.Hr(className="divider"),

        # Projectie grafiek + resultaten
        dbc.Row([
            dbc.Col(
                html.Div(className="chart-card", children=[
                    html.Div("3-jaar projectie (incl. 10g aankoop elke 2 maanden)",
                             className="chart-card-title"),
                    html.Div(f"Startwaarde: {fmt_k(huidig)} | Doel: € {GOAL/1000:.0f}K | {bron_tekst}",
                             className="chart-card-subtitle"),
                    dcc.Graph(
                        id=f"{tab_id_prefix}-scenario-chart",
                        config={"displayModeBar": False},
                        style={"height": "380px"},
                    ),
                    html.Div(f"Bronnen: {bron_tekst}", className="source-note"),
                ]),
                xs=12, lg=8,
            ),
            dbc.Col(
                html.Div(className="chart-card", style={"height": "100%"}, children=[
                    html.Div("Scenario resultaten", className="chart-card-title"),
                    html.Div("Eindwaarde na 3 jaar | 10g elke 2 maanden",
                             className="chart-card-subtitle"),
                    scenario_resultaten(proj),
                    html.Hr(className="divider"),
                    # Gevoeligheids mini grafiek
                    dcc.Graph(
                        id=f"{tab_id_prefix}-sensitivity-chart",
                        config={"displayModeBar": False},
                        style={"height": "180px"},
                    ),
                    html.Div("Gevoeligheidsanalyse: benodigd jaarrendement", className="source-note"),
                ]),
                xs=12, lg=4,
            ),
        ]),
    ])


if DATA_LOADED:
    hist_tab_layout = scenario_tab(
        tab_id_prefix="hist",
        titel="Scenario Analyse - Historisch (2020-2026)",
        subtitel="Gebaseerd op 6 jaar historische goudrendementen in EUR. Conservatief en data-gedreven.",
        bron_tekst="World Gold Council, Yahoo Finance GC=F 2020–2026",
        proj=proj_hist,
    )
    klimaat_tab_layout = scenario_tab(
        tab_id_prefix="klimaat",
        titel="Scenario Analyse - Klimaat & Geopolitiek",
        subtitel=(
            "Gebaseerd op huidige macro-context: Goldman Sachs target $3.700–$4.500/oz, "
            "centrale bank recordaankopen, dollardaling, geopolitieke onzekerheid (Gaza, Oekraïne, Taiwan)."
        ),
        bron_tekst="Goldman Sachs (apr 2025), World Gold Council 2025 Outlook, CME Group",
        proj=proj_klimaat,
    )
else:
    hist_tab_layout    = html.Div("Data niet geladen.", className="page-container")
    klimaat_tab_layout = html.Div("Data niet geladen.", className="page-container")


# App layout
app.layout = html.Div(style={"backgroundColor": C["bg"], "minHeight": "100vh"}, children=[

    # Header
    html.Div(className="dashboard-header", children=[
        html.Div(className="header-left", children=[
            html.Span("⚜", className="header-logo"),
            html.Div([
                html.Div("GOUD PORTFOLIO", className="header-title"),
                html.Div("Eindproef Data Scientist",
                         className="header-subtitle"),
            ]),
        ]),
        html.Div(className="header-right", children=[
            html.Div(
                f"Spot: {fmt_k(current_spot)}/gram" if DATA_LOADED else "Laden...",
                className="header-spot",
            ),
            html.Div(f"Bijgewerkt: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                     className="header-time"),
        ]),
    ]),

    # Tabs
    dbc.Tabs(
        id="main-tabs",
        active_tab="overzicht",
        className="nav-tabs",
        children=[
            dbc.Tab(overzicht_layout,    label="Overzicht",                   tab_id="overzicht"),
            dbc.Tab(hist_tab_layout,     label="Historisch Scenario",          tab_id="historisch"),
            dbc.Tab(klimaat_tab_layout,  label="Klimaat & Geopolitiek",        tab_id="klimaat"),
        ],
    ),
])


# Callbacks

@callback(Output("gold-price-chart", "figure"), Input("main-tabs", "active_tab"))
def cb_gold_price(_):
    if not DATA_LOADED:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gold.index, y=gold[PRICE_COL],
        line=dict(color=C["gold"], width=2),
        fill="tozeroy",
        fillcolor="rgba(184,134,11,0.08)",
        name="Goudprijs (EUR/gram)",
        hovertemplate="%{x|%d/%m/%Y}<br>€ %{y:.2f}/gram<extra></extra>",
    ))

    if "spot_prijs_aankoop" in bars.columns:
        fig.add_trace(go.Scatter(
            x=bars["datum_aankoop"], y=bars["spot_prijs_aankoop"],
            mode="markers",
            name="Jouw aankopen",
            marker=dict(color=C["blue"], size=9, symbol="diamond",
                        line=dict(width=1, color=C["white"])),
            hovertemplate="%{x|%d/%m/%Y}<br>Spot: € %{y:.2f}/gram<extra></extra>",
        ))

    fig.update_layout(**base_layout(ytitle="EUR/gram"))
    return fig


@callback(Output("portfolio-pie", "figure"), Input("main-tabs", "active_tab"))
def cb_pie(_):
    if not DATA_LOADED:
        return go.Figure()

    grp = portfolio_calc.groupby("type")["gram"].sum()
    labels = {"staaf": "Goudstaven", "juweel": "Sieraden"}
    fig = go.Figure(go.Pie(
        labels=[labels.get(t, t) for t in grp.index],
        values=grp.values,
        hole=0.55,
        marker=dict(colors=[C["gold"], C["blue"]], line=dict(color=C["white"], width=2)),
        textfont=dict(size=11),
        hovertemplate="%{label}: %{value:.1f}g (%{percent})<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        margin=dict(l=10, r=10, t=10, b=40),
        extra_legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
    ))
    return fig


@callback(Output("purchase-vs-spot", "figure"), Input("main-tabs", "active_tab"))
def cb_purchase_vs_spot(_):
    if not DATA_LOADED:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gold.index, y=gold[PRICE_COL],
        name="Spotprijs",
        line=dict(color="#D1D5DB", width=1.5),
        hovertemplate="%{x|%d/%m/%Y}<br>Spot: € %{y:.2f}/gram<extra></extra>",
    ))

    if "prijs_per_gram" in bars.columns:
        fig.add_trace(go.Scatter(
            x=bars["datum_aankoop"], y=bars["prijs_per_gram"],
            mode="markers+lines",
            name="Jouw aankoopprijs",
            marker=dict(color=C["blue"], size=9, line=dict(width=1, color=C["white"])),
            line=dict(color=C["blue"], width=1.5, dash="dot"),
            hovertemplate="%{x|%d/%m/%Y}<br>Betaald: € %{y:.2f}/gram<extra></extra>",
        ))

    fig.update_layout(**base_layout(ytitle="EUR/gram"))
    return fig


@callback(Output("portfolio-details", "children"), Input("main-tabs", "active_tab"))
def cb_details(_):
    if not DATA_LOADED:
        return html.P("Data niet beschikbaar.")

    items = [
        ("Totaal gewicht",      format_gram(stats.get("total_weight_gram", 0))),
        ("Zuiver goud (staven)", format_gram(stats.get("total_pure_gold_gram", 0))),
        ("Sieraden gewicht",    format_gram(stats.get("jewelry_weight_gram", 0))),
        ("Staven waarde",       format_eur(stats.get("current_value_bars_eur", 0))),
        ("Sieraden waarde",     format_eur(stats.get("jewelry_value_eur", 0))),
        ("ROI (staven)",        f"{stats.get('roi_percent', 0):.1f}%"
                                if not np.isnan(stats.get("roi_percent", np.nan)) else "N/A"),
        ("Gem. premie betaald", f"{((bars['prijs_per_gram'] / bars['spot_prijs_aankoop']) - 1).mean() * 100:.1f}%"
                                if DATA_LOADED and "prijs_per_gram" in bars.columns else "N/A"),
    ]
    return html.Div([
        html.Div(className="detail-row", children=[
            html.Span(lbl, className="detail-label"),
            html.Span(val, className="detail-value"),
        ]) for lbl, val in items
    ])


def make_sensitivity_figure(huidig: float) -> go.Figure:
    rendementen = np.linspace(0.0, 0.45, 200)
    eindwaarden = [huidig * (1 + r) ** JAREN for r in rendementen]
    benodigd = (GOAL / max(huidig, 1)) ** (1 / JAREN) - 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rendementen * 100, y=eindwaarden,
        line=dict(color=C["blue"], width=2),
        fill="tozeroy",
        fillcolor="rgba(44,95,138,0.06)",
        name="Eindwaarde",
        hovertemplate="%{x:.0f}%/jr → € %{y:,.0f}<extra></extra>",
    ))
    fig.add_vline(x=benodigd * 100, line_dash="dot", line_color=C["gold"], line_width=1.5,
                  annotation=dict(text=f"{benodigd:.0%}", font=dict(color=C["gold"], size=9)))
    fig.add_hline(y=GOAL, line_dash="dot", line_color=C["red"], line_width=1,
                  annotation=dict(text=f"€{GOAL/1000:.0f}K", font=dict(color=C["red"], size=9)))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="Inter, sans-serif", size=9),
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False,
        xaxis=dict(title="%/jr", gridcolor="#F3F4F6", zeroline=False),
        yaxis=dict(title="", gridcolor="#F3F4F6", zeroline=False, tickformat=",.0f"),
    )
    return fig


@callback(Output("hist-scenario-chart", "figure"), Input("main-tabs", "active_tab"))
def cb_hist_scenario(_):
    if not DATA_LOADED:
        return go.Figure()
    return scenario_chart_figure(proj_hist, "Historisch conservatief scenario")


@callback(Output("hist-sensitivity-chart", "figure"), Input("main-tabs", "active_tab"))
def cb_hist_sensitivity(_):
    if not DATA_LOADED:
        return go.Figure()
    return make_sensitivity_figure(stats["total_portfolio_value_eur"])


@callback(Output("klimaat-scenario-chart", "figure"), Input("main-tabs", "active_tab"))
def cb_klimaat_scenario(_):
    if not DATA_LOADED:
        return go.Figure()
    return scenario_chart_figure(proj_klimaat, "Klimaat & geopolitiek scenario")


@callback(Output("klimaat-sensitivity-chart", "figure"), Input("main-tabs", "active_tab"))
def cb_klimaat_sensitivity(_):
    if not DATA_LOADED:
        return go.Figure()
    return make_sensitivity_figure(stats["total_portfolio_value_eur"])


# Run app
if __name__ == "__main__":
    print("Dashboard starten...")
    print("Open http://127.0.0.1:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)
