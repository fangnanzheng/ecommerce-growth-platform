from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.forecasting import forecast_series, prepare_series
from src.utils.io import read_table


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "data" / "output"

ACCENT = "#38bdf8"
ACCENT_2 = "#22c55e"
WARNING = "#f59e0b"
TEXT_MUTED = "#94a3b8"
PLOT_TEMPLATE = "plotly_dark"


st.set_page_config(
    page_title="E-Commerce Growth Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --accent: #38bdf8;
            --accent-2: #22c55e;
            --panel: rgba(15, 23, 42, 0.72);
            --panel-border: rgba(148, 163, 184, 0.18);
            --muted: #94a3b8;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 0%, rgba(56, 189, 248, 0.12), transparent 28%),
                linear-gradient(180deg, #0b1120 0%, #0f172a 48%, #111827 100%);
        }

        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }

        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 1500px;
            padding-top: 3.2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        div[data-testid="stTabs"] button {
            color: #cbd5e1;
            font-weight: 700;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #f8fafc;
        }

        .hero {
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.86));
            border-radius: 8px;
            padding: 28px 30px;
            margin-bottom: 22px;
        }

        .hero-label {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .hero-title {
            color: #f8fafc;
            font-size: clamp(2.0rem, 4vw, 3.5rem);
            line-height: 1.04;
            font-weight: 850;
            margin: 0 0 12px 0;
        }

        .hero-copy {
            color: #cbd5e1;
            max-width: 880px;
            font-size: 1rem;
            line-height: 1.65;
            margin: 0;
        }

        .metric-card {
            border: 1px solid var(--panel-border);
            background: var(--panel);
            border-radius: 8px;
            padding: 18px 18px 16px 18px;
            min-height: 118px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 10px;
        }

        .metric-value {
            color: #f8fafc;
            font-size: clamp(1.7rem, 3vw, 2.35rem);
            font-weight: 800;
            line-height: 1.1;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.84rem;
            margin-top: 10px;
        }

        .insight-card {
            border-left: 3px solid var(--accent);
            background: rgba(15, 23, 42, 0.64);
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 92px;
        }

        .insight-title {
            color: #f8fafc;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .insight-body {
            color: #cbd5e1;
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .method-card {
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(15, 23, 42, 0.58);
            border-radius: 8px;
            padding: 16px 18px;
            min-height: 128px;
        }

        .method-title {
            color: #f8fafc;
            font-size: 0.95rem;
            font-weight: 850;
            margin-bottom: 8px;
        }

        .method-body {
            color: #cbd5e1;
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .section-title {
            color: #f8fafc;
            font-size: 1.05rem;
            font-weight: 850;
            margin: 18px 0 4px 0;
        }

        .section-caption {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 12px;
        }

        .stDataFrame {
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_processed_table(table_name: str) -> pd.DataFrame | None:
    try:
        return read_table(PROCESSED_DIR, table_name)
    except FileNotFoundError:
        return None


def read_optional_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def chart_style(fig, height: int = 430):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=24, r=24, t=58, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#dbeafe", family="Arial"),
        title=dict(font=dict(size=18, color="#f8fafc"), x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.16)", zerolinecolor="rgba(148,163,184,0.16)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.16)", zerolinecolor="rgba(148,163,184,0.16)")
    return fig


def format_category(category: object) -> str:
    if category is None or pd.isna(category):
        return "Unknown"
    return str(category).replace("_", " ").capitalize()


def format_method_name(method: str) -> str:
    return {
        "sarimax": "SARIMAX",
        "ets": "ETS",
        "moving_average": "Moving average",
    }.get(method, method)


def format_column_name(column: str) -> str:
    special_names = {
        "customer_unique_id": "Customer ID",
        "roc_auc": "ROC AUC",
        "avg_order_value": "Avg order value",
        "avg_repeat_probability": "Avg repeat probability",
        "abs_coefficient": "Absolute coefficient",
        "cvr": "CVR",
    }
    if column in special_names:
        return special_names[column]
    return column.replace("_", " ").capitalize()


def format_display_table(df: pd.DataFrame, category_columns: list[str] | None = None) -> pd.DataFrame:
    display_df = df.copy()
    for column in category_columns or []:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(format_category)
    display_df = display_df.rename(columns={column: format_column_name(column) for column in display_df.columns})
    return display_df


def format_feature_name(feature: object) -> str:
    if feature is None or pd.isna(feature):
        return "Unknown"
    text = str(feature)
    if "=" in text:
        prefix, value = text.split("=", 1)
        prefix = prefix.replace("_", " ").capitalize()
        value = format_category(value)
        return f"{prefix}: {value}"
    return text.replace("_", " ").capitalize()


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def method_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="method-card">
            <div class="method-title">{title}</div>
            <div class="method-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def render_missing_state() -> None:
    apply_theme()
    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">Setup required</div>
            <div class="hero-title">Processed data was not found</div>
            <p class="hero-copy">Run the commands below to generate sample data and analysis outputs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        "\n".join(
            [
                "python -m src.data_pipeline.etl_pyspark --sample-data",
                "python -m src.models.segmentation",
                "python -m src.models.repeat_purchase_predict",
                "python -m src.models.forecasting",
                "python -m src.experiments.ab_testing",
                "python -m src.utils.excel_generator",
            ]
        ),
        language="bash",
    )


def sidebar_filters(fact_orders: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Controls")
    st.sidebar.caption("Filter the executive view without regenerating the pipeline.")

    min_date = fact_orders["order_purchase_ts"].min().date()
    max_date = fact_orders["order_purchase_ts"].max().date()
    date_range = st.sidebar.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    states = sorted(fact_orders["customer_state"].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect("Customer states", states, default=states[:])

    categories = sorted(fact_orders["product_category"].dropna().unique().tolist())
    default_categories = (
        fact_orders.groupby("product_category")["payment_value"].sum().sort_values(ascending=False).head(12).index.tolist()
    )
    selected_categories = st.sidebar.multiselect(
        "Categories",
        categories,
        default=default_categories,
        format_func=format_category,
    )

    filtered = fact_orders.copy()
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["order_purchase_ts"].dt.date >= start_date)
            & (filtered["order_purchase_ts"].dt.date <= end_date)
        ]

    if selected_states:
        filtered = filtered[filtered["customer_state"].isin(selected_states)]

    if selected_categories:
        filtered = filtered[filtered["product_category"].isin(selected_categories)]

    report_path = OUTPUT_DIR / "ecommerce_growth_report.xlsx"
    st.sidebar.divider()
    st.sidebar.subheader("Artifacts")
    if report_path.exists():
        st.sidebar.success(f"Excel report ready: {report_path.name}")
    else:
        st.sidebar.info("Excel report not generated yet.")

    st.sidebar.caption("Use Ctrl+C in the terminal to stop Streamlit.")
    return filtered


@st.cache_data(show_spinner=False)
def cached_forecast(
    _daily_sales: pd.DataFrame,
    state: str,
    category: str,
    periods: int,
    alpha: float,
    method: str,
) -> tuple[pd.Series, pd.DataFrame]:
    state_filter = None if state == "ALL" else state
    _, _, series = prepare_series(_daily_sales, category=category, state=state_filter)
    forecast = forecast_series(series, periods=periods, alpha=alpha, method=method)
    return series, forecast


apply_theme()

fact_orders = load_processed_table("fact_orders")
customer_features = load_processed_table("customer_features")
category_daily_sales = load_processed_table("category_daily_sales")
state_category_daily_sales = load_processed_table("state_category_daily_sales")

if fact_orders is None or customer_features is None:
    render_missing_state()
    st.stop()

segmented_customers = load_optional_csv(OUTPUT_DIR / "segmented_customers.csv")
segment_summary = load_optional_csv(OUTPUT_DIR / "segment_summary.csv")
repeat_predictions = load_optional_csv(OUTPUT_DIR / "repeat_purchase_predictions.csv")
repeat_state_summary = load_optional_csv(OUTPUT_DIR / "repeat_state_summary.csv")
repeat_category_summary = load_optional_csv(OUTPUT_DIR / "repeat_category_summary.csv")
repeat_feature_importance = load_optional_csv(OUTPUT_DIR / "repeat_feature_importance.csv")
repeat_topk_summary = load_optional_csv(OUTPUT_DIR / "repeat_topk_summary.csv")
data_quality_summary = load_optional_csv(OUTPUT_DIR / "data_quality_summary.csv")
sales_forecast = load_optional_csv(OUTPUT_DIR / "sales_forecast.csv")
repeat_metrics = read_optional_json(OUTPUT_DIR / "repeat_purchase_metrics.json")
ab_results = read_optional_json(OUTPUT_DIR / "ab_test_results.json")

fact_orders["order_purchase_ts"] = pd.to_datetime(fact_orders["order_purchase_ts"])
filtered_orders = sidebar_filters(fact_orders)

total_revenue = float(filtered_orders["payment_value"].sum())
total_orders = int(filtered_orders["order_id"].nunique())
total_customers = int(filtered_orders["customer_unique_id"].nunique())
avg_order_value = total_revenue / total_orders if total_orders else 0
late_delivery_rate = float(filtered_orders["is_late_delivery"].mean()) if len(filtered_orders) else 0
repeat_customer_rate = float((customer_features["order_count"] > 1).mean()) if len(customer_features) else 0
top_category = (
    filtered_orders.groupby("product_category")["payment_value"].sum().sort_values(ascending=False).index[0]
    if len(filtered_orders)
    else "n/a"
)
top_category_display = format_category(top_category) if top_category != "n/a" else "n/a"

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">Olist Brazilian E-Commerce Dataset</div>
        <div class="hero-title">E-Commerce Growth & Experimentation Platform</div>
        <p class="hero-copy">
            An end-to-end analytics product for diagnosing marketplace growth, prioritizing
            repeat-purchase campaigns, forecasting demand, and validating experiments.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    metric_card("Revenue", f"R$ {total_revenue:,.0f}", "Filtered gross payment value")
with kpi_cols[1]:
    metric_card("Orders", f"{total_orders:,}", "Unique orders in scope")
with kpi_cols[2]:
    metric_card("Customers", f"{total_customers:,}", "Unique marketplace buyers")
with kpi_cols[3]:
    metric_card("AOV", f"R$ {avg_order_value:,.2f}", "Average order value")

st.write("")
insight_cols = st.columns(3)
with insight_cols[0]:
    insight_card("Category leader", f"{top_category_display} is the top filtered revenue category.")
with insight_cols[1]:
    insight_card("Repeat behavior", f"Only {repeat_customer_rate:.1%} of customers bought more than once, so retention must be framed as repeat propensity.")
with insight_cols[2]:
    insight_card("Delivery signal", f"{late_delivery_rate:.1%} of filtered orders arrived after the estimated date.")

tab_growth, tab_quality, tab_repeat, tab_forecast, tab_experiments = st.tabs(
    ["Growth Story", "Data Quality", "Repeat Purchase", "Forecast", "Experiments"]
)

with tab_growth:
    section_header(
        "Growth Diagnosis",
        "Use this page in interviews as the executive opening: revenue scale, category concentration, regional mix, and operational friction.",
    )
    growth_min_date = filtered_orders["order_purchase_ts"].min().date()
    growth_max_date = filtered_orders["order_purchase_ts"].max().date()
    growth_window = st.slider(
        "Growth story date window",
        min_value=growth_min_date,
        max_value=growth_max_date,
        value=(growth_min_date, growth_max_date),
        format="YYYY-MM-DD",
    )
    growth_orders = filtered_orders[
        (filtered_orders["order_purchase_ts"].dt.date >= growth_window[0])
        & (filtered_orders["order_purchase_ts"].dt.date <= growth_window[1])
    ].copy()

    if growth_orders.empty:
        st.warning("No orders in the selected Growth Story date window.")
        st.stop()

    daily_revenue = (
        growth_orders.assign(order_date=growth_orders["order_purchase_ts"].dt.date)
        .groupby("order_date", as_index=False)
        .agg(revenue=("payment_value", "sum"), orders=("order_id", "nunique"))
    )
    revenue_fig = px.line(
        daily_revenue,
        x="order_date",
        y="revenue",
        title="Daily Revenue",
        labels={"order_date": "Order date", "revenue": "Revenue"},
        color_discrete_sequence=[ACCENT],
    )
    revenue_fig.update_traces(line=dict(width=2.4), hovertemplate="%{x}<br>Revenue: R$ %{y:,.0f}<extra></extra>")
    st.plotly_chart(chart_style(revenue_fig, 460), width="stretch")

    category_col, state_col = st.columns([1, 1])
    category_revenue = (
        growth_orders.groupby("product_category", as_index=False)["payment_value"]
        .sum()
        .sort_values("payment_value", ascending=False)
        .head(10)
        .sort_values("payment_value")
    )
    category_revenue["product_category_display"] = category_revenue["product_category"].map(format_category)
    category_fig = px.bar(
        category_revenue,
        x="payment_value",
        y="product_category_display",
        orientation="h",
        title="Top Categories",
        labels={"payment_value": "Revenue", "product_category_display": "Category"},
        color_discrete_sequence=[ACCENT],
    )
    category_fig.update_traces(hovertemplate="%{y}<br>Revenue: R$ %{x:,.0f}<extra></extra>")
    category_col.plotly_chart(chart_style(category_fig, 430), width="stretch")

    state_revenue = (
        growth_orders.groupby("customer_state", as_index=False)["payment_value"]
        .sum()
        .sort_values("payment_value", ascending=False)
        .head(12)
    )
    state_fig = px.bar(
        state_revenue,
        x="customer_state",
        y="payment_value",
        title="Revenue by State",
        labels={"customer_state": "State", "payment_value": "Revenue"},
        color_discrete_sequence=[ACCENT_2],
    )
    state_col.plotly_chart(chart_style(state_fig, 430), width="stretch")

    selected_period_revenue = float(growth_orders["payment_value"].sum())
    category_state = (
        growth_orders.groupby(["customer_state", "product_category"], as_index=False)["payment_value"]
        .sum()
        .sort_values("payment_value", ascending=False)
        .head(60)
    )
    category_state["product_category_display"] = category_state["product_category"].map(format_category)
    category_state["revenue_share"] = category_state["payment_value"] / selected_period_revenue
    category_state["share_label"] = category_state["revenue_share"].map(lambda value: f"{value:.1%}")
    mix_fig = px.treemap(
        category_state,
        path=["customer_state", "product_category_display"],
        values="payment_value",
        title="State and Category Revenue Mix",
        color="payment_value",
        labels={"payment_value": "Payment value"},
        custom_data=["revenue_share", "payment_value"],
        color_continuous_scale=["#164e63", "#38bdf8", "#22c55e"],
    )
    mix_fig.update_traces(
        texttemplate="%{label}<br>%{customdata[0]:.1%}",
        hovertemplate="<b>%{label}</b><br>Payment value: R$ %{customdata[1]:,.0f}<br>Share of selected revenue: %{customdata[0]:.1%}<extra></extra>",
    )
    mix_fig.update_layout(coloraxis_colorbar=dict(title="Payment value"))
    st.plotly_chart(chart_style(mix_fig, 560), width="stretch")

with tab_quality:
    section_header(
        "Data Quality Checks",
        "A lightweight audit layer for raw files, referential integrity, processed table freshness, and model-ready targets.",
    )
    if data_quality_summary is None:
        st.warning("Data quality output is missing. Run python -m src.data_pipeline.data_quality")
    else:
        status_counts = data_quality_summary["status"].value_counts().to_dict()
        quality_cols = st.columns(4)
        with quality_cols[0]:
            metric_card("Passed", f"{status_counts.get('pass', 0):,}", "Checks with expected values")
        with quality_cols[1]:
            metric_card("Warnings", f"{status_counts.get('warn', 0):,}", "Non-blocking data caveats")
        with quality_cols[2]:
            metric_card("Failed", f"{status_counts.get('fail', 0):,}", "Blocking data quality issues")
        with quality_cols[3]:
            metric_card("Total checks", f"{len(data_quality_summary):,}", "Raw + processed validation")

        check_type = st.multiselect(
            "Filter check status",
            options=["pass", "warn", "fail"],
            default=["pass", "warn", "fail"],
            format_func=lambda value: value.capitalize(),
        )
        quality_view = data_quality_summary[data_quality_summary["status"].isin(check_type)].copy()
        quality_view = format_display_table(quality_view)
        st.dataframe(quality_view, width="stretch", hide_index=True)

with tab_repeat:
    if repeat_predictions is None:
        st.warning("Repeat purchase output is missing. Run python -m src.models.repeat_purchase_predict")
    else:
        section_header(
            "Repeat Purchase Propensity",
            "Olist is dominated by one-time buyers, so the model ranks first-order customers by likelihood to buy again.",
        )
        with st.expander("Method and banding rule", expanded=True):
            st.markdown("**Target.** For customer `i`, define:")
            st.latex(
                r"""
                y_i =
                \begin{cases}
                1, & \text{if customer } i \text{ has more than one valid order} \\
                0, & \text{otherwise}
                \end{cases}
                """
            )
            st.markdown("**Model.** We use first-order features only, then estimate repeat propensity with balanced logistic regression:")
            st.latex(
                r"""
                \hat p_i = P(y_i = 1 \mid x_i) =
                \frac{1}{1 + e^{-(\beta_0 + \beta^\top x_i)}}
                """
            )
            st.markdown(
                "Numeric features are median-imputed and standardized. State, first category, and payment type are one-hot encoded."
            )
            st.markdown(
                "**Bands.** The 10/30/60 split is not a statistical law. It is an operating policy for campaign prioritization:"
            )
            st.latex(
                r"""
                High = top\ 10\%, \quad Medium = next\ 30\%, \quad Low = remaining\ 60\%
                """
            )
            st.markdown(
                "The model's statistical quality is evaluated separately with ROC AUC, average precision, precision, and recall."
            )

        roc_auc = repeat_metrics.get("roc_auc")
        avg_precision = repeat_metrics.get("average_precision")
        metric_cols = st.columns(5)
        with metric_cols[0]:
            metric_card("Repeat Rate", f"{repeat_metrics.get('repeat_rate', 0):.2%}", "Observed customer repeat")
        with metric_cols[1]:
            metric_card("Precision", f"{repeat_metrics.get('precision', 0):.2%}", "High-propensity hit rate")
        with metric_cols[2]:
            metric_card("Recall", f"{repeat_metrics.get('recall', 0):.2%}", "Captured repeat buyers")
        with metric_cols[3]:
            metric_card("ROC AUC", f"{roc_auc:.3f}" if roc_auc is not None else "n/a", "Signal strength")
        with metric_cols[4]:
            metric_card(
                "Avg Precision",
                f"{avg_precision:.3f}" if avg_precision is not None else "n/a",
                "Rare-event ranking quality",
            )

        section_header(
            "Repeat Propensity Distribution",
            "The model scores first-time buyers by their likelihood to place another order. This is more defensible than classic churn for Olist.",
        )
        repeat_cols = st.columns([1, 1])
        propensity_counts = repeat_predictions["propensity_band"].value_counts().reset_index()
        propensity_counts.columns = ["propensity_band", "customers"]
        propensity_counts["propensity_band"] = pd.Categorical(
            propensity_counts["propensity_band"],
            categories=["Low", "Medium", "High"],
            ordered=True,
        )
        propensity_counts = propensity_counts.sort_values("propensity_band")
        band_colors = {"Low": ACCENT_2, "Medium": WARNING, "High": "#ef4444"}
        propensity_fig = go.Figure(
            go.Bar(
                x=propensity_counts["propensity_band"].astype(str),
                y=propensity_counts["customers"],
                marker_color=[band_colors[band] for band in propensity_counts["propensity_band"].astype(str)],
                width=0.42,
                hovertemplate="Band: %{x}<br>Customers: %{y:,.0f}<extra></extra>",
            )
        )
        propensity_fig.update_layout(
            title="Repeat Propensity Bands",
            xaxis_title="Propensity band",
            yaxis_title="Customers",
            bargap=0.45,
            showlegend=False,
        )
        propensity_fig.update_xaxes(
            categoryorder="array",
            categoryarray=["Low", "Medium", "High"],
        )
        repeat_cols[0].plotly_chart(chart_style(propensity_fig, 390), width="stretch")

        if repeat_state_summary is not None:
            state_repeat = repeat_state_summary.sort_values("avg_repeat_probability", ascending=False).head(12)
            state_repeat_fig = px.bar(
                state_repeat,
                x="first_customer_state",
                y="avg_repeat_probability",
                title="Repeat Propensity by State",
                labels={"first_customer_state": "State", "avg_repeat_probability": "Avg repeat probability"},
                color_discrete_sequence=[ACCENT],
            )
            repeat_cols[1].plotly_chart(chart_style(state_repeat_fig, 390), width="stretch")

        if repeat_category_summary is not None:
            category_repeat = repeat_category_summary[
                repeat_category_summary["customers"] >= 100
            ].sort_values("avg_repeat_probability", ascending=False).head(15)
            category_repeat["first_product_category_display"] = category_repeat["first_product_category"].map(format_category)
            category_fig = px.bar(
                category_repeat.sort_values("avg_repeat_probability"),
                x="avg_repeat_probability",
                y="first_product_category_display",
                orientation="h",
                title="Repeat Propensity by First Category",
                labels={
                    "avg_repeat_probability": "Avg repeat probability",
                    "first_product_category_display": "First category",
                },
                color_discrete_sequence=[ACCENT_2],
            )
            st.plotly_chart(chart_style(category_fig, 430), width="stretch")

        explain_cols = st.columns([1, 1])
        if repeat_topk_summary is not None:
            topk_fig = px.bar(
                repeat_topk_summary,
                x="population_slice",
                y="lift_vs_baseline",
                title="Repeat Lift by Targeting Depth",
                labels={"population_slice": "Targeting slice", "lift_vs_baseline": "Lift vs baseline"},
                color_discrete_sequence=[WARNING],
            )
            explain_cols[0].plotly_chart(chart_style(topk_fig, 380), width="stretch")

        if repeat_feature_importance is not None:
            feature_view = repeat_feature_importance.head(15).copy()
            feature_view["feature_display"] = feature_view["feature"].map(format_feature_name)
            feature_fig = px.bar(
                feature_view.sort_values("coefficient"),
                x="coefficient",
                y="feature_display",
                orientation="h",
                title="Top Repeat Propensity Drivers",
                labels={"coefficient": "Logistic coefficient", "feature_display": "Feature"},
                color="coefficient",
                color_continuous_scale=["#ef4444", "#94a3b8", "#22c55e"],
            )
            explain_cols[1].plotly_chart(chart_style(feature_fig, 380), width="stretch")

        repeat_table = format_display_table(
            repeat_predictions.head(100),
            category_columns=["first_product_category"],
        )
        st.dataframe(repeat_table, width="stretch", hide_index=True)

with tab_forecast:
    forecast_source = state_category_daily_sales if state_category_daily_sales is not None else category_daily_sales
    if forecast_source is None:
        st.warning("Forecast source data is missing. Run python -m src.data_pipeline.etl_pyspark --engine pandas")
    else:
        section_header(
            "State and Category Forecast",
            "Choose a state/category pair and forecast method to compare expected revenue with an 80% interval.",
        )
        with st.expander("Forecast methods", expanded=False):
            st.markdown("**SARIMAX.** Default model:")
            st.latex(r"SARIMAX(1,1,1)\times(1,0,1)_7")
            st.markdown("The 80% interval is taken from the model forecast distribution.")
            st.markdown("**ETS.** Additive trend and weekly seasonality:")
            st.latex(r"\hat y_{t+h} = \ell_t + h b_t + s_{t+h-7}")
            st.markdown("Intervals are approximated with residual volatility.")
            st.markdown("**Moving average.** Transparent baseline:")
            st.latex(r"\hat y_{T+h} = \frac{1}{w}\sum_{j=0}^{w-1} y_{T-j}")
            st.markdown("Intervals widen with horizon using historical residual volatility.")

        forecast_source["order_date"] = pd.to_datetime(forecast_source["order_date"])
        state_options = ["ALL"]
        if "customer_state" in forecast_source.columns:
            state_options += sorted(forecast_source["customer_state"].dropna().unique().tolist())

        top_scope = (
            forecast_source.groupby(["customer_state", "product_category"])["revenue"].sum()
            if "customer_state" in forecast_source.columns
            else forecast_source.groupby(["product_category"])["revenue"].sum()
        ).sort_values(ascending=False)
        default_state = top_scope.index[0][0] if "customer_state" in forecast_source.columns else "ALL"
        default_category = top_scope.index[0][1] if "customer_state" in forecast_source.columns else top_scope.index[0]

        control_cols = st.columns([1, 1, 1, 1])
        selected_state = control_cols[0].selectbox(
            "Forecast state",
            options=state_options,
            index=state_options.index(default_state) if default_state in state_options else 0,
        )
        category_frame = forecast_source
        if selected_state != "ALL" and "customer_state" in category_frame.columns:
            category_frame = category_frame[category_frame["customer_state"] == selected_state]
        category_options = (
            category_frame.groupby("product_category")["revenue"].sum().sort_values(ascending=False).index.tolist()
        )
        selected_category = control_cols[1].selectbox(
            "Forecast category",
            options=category_options,
            index=category_options.index(default_category) if default_category in category_options else 0,
            format_func=format_category,
        )
        method_labels = {
            "SARIMAX": "sarimax",
            "ETS": "ets",
            "Moving Average": "moving_average",
        }
        selected_method_label = control_cols[2].selectbox(
            "Forecast method",
            options=list(method_labels.keys()),
            index=0,
        )
        selected_method = method_labels[selected_method_label]
        forecast_periods = control_cols[3].slider("Forecast days", min_value=14, max_value=90, value=30, step=7)

        try:
            history, forecast = cached_forecast(
                forecast_source,
                selected_state,
                selected_category,
                forecast_periods,
                0.2,
                selected_method,
            )
            history_df = history.reset_index()
            history_df.columns = ["order_date", "revenue"]

            forecast_fig = go.Figure()
            forecast_fig.add_trace(
                go.Scatter(
                    x=history_df["order_date"],
                    y=history_df["revenue"],
                    mode="lines",
                    name="Actual revenue",
                    line=dict(color=ACCENT, width=2),
                )
            )
            forecast_fig.add_trace(
                go.Scatter(
                    x=forecast["order_date"],
                    y=forecast["upper_revenue"],
                    mode="lines",
                    name="Upper interval",
                    line=dict(color="rgba(56, 189, 248, 0.18)", width=0),
                    showlegend=False,
                )
            )
            forecast_fig.add_trace(
                go.Scatter(
                    x=forecast["order_date"],
                    y=forecast["lower_revenue"],
                    mode="lines",
                    name="80% interval",
                    fill="tonexty",
                    fillcolor="rgba(56, 189, 248, 0.18)",
                    line=dict(color="rgba(56, 189, 248, 0.18)", width=0),
                )
            )
            forecast_fig.add_trace(
                go.Scatter(
                    x=forecast["order_date"],
                    y=forecast["forecast_revenue"],
                    mode="lines",
                    name="Forecast revenue",
                    line=dict(color=ACCENT_2, width=3),
                )
            )
            forecast_fig.update_layout(
                title=f"Revenue Forecast: {selected_state} / {format_category(selected_category)} / {selected_method_label}"
            )
            st.plotly_chart(chart_style(forecast_fig, 470), width="stretch")
            forecast_table = forecast.copy()
            if "product_category" in forecast_table.columns:
                forecast_table["product_category"] = forecast_table["product_category"].map(format_category)
            forecast_table = format_display_table(forecast_table)
            st.dataframe(forecast_table, width="stretch", hide_index=True)
        except ValueError as exc:
            st.warning(str(exc))

with tab_experiments:
    conversion = ab_results.get("conversion_test", {})
    order_value = ab_results.get("order_value_test", {})
    if not conversion:
        st.warning("A/B testing output is missing. Run python -m src.experiments.ab_testing")
    else:
        section_header(
            "Experiment Readout",
            "This module demonstrates how a growth team would validate an experiment before making a rollout decision.",
        )
        with st.expander("Experiment design and test statistics", expanded=True):
            st.markdown("**Demo assignment.** The portfolio dataset is historical, so we simulate a deterministic split:")
            st.latex(
                r"""
                variant_i =
                \begin{cases}
                control, & hash(order\_id_i) \bmod 2 = 0 \\
                treatment, & hash(order\_id_i) \bmod 2 = 1
                \end{cases}
                """
            )
            st.markdown("In production, this should be user-level random assignment with exposure logging.")
            st.markdown("**Conversion Z-test.** Let `x_c`, `x_t` be converted orders and `n_c`, `n_t` be sample sizes:")
            st.latex(
                r"""
                \hat p_c=\frac{x_c}{n_c}, \quad
                \hat p_t=\frac{x_t}{n_t}, \quad
                \hat p=\frac{x_c+x_t}{n_c+n_t}
                """
            )
            st.latex(
                r"""
                z =
                \frac{\hat p_t-\hat p_c}
                {\sqrt{\hat p(1-\hat p)\left(\frac{1}{n_c}+\frac{1}{n_t}\right)}}
                """
            )
            st.markdown(
                "**Order-value Welch T-test.** Let the two groups have means `x_bar_c`, `x_bar_t` and sample variances `s_c^2`, `s_t^2`:"
            )
            st.latex(
                r"""
                t =
                \frac{\bar x_t-\bar x_c}
                {\sqrt{\frac{s_t^2}{n_t}+\frac{s_c^2}{n_c}}}
                """
            )
            st.markdown("Both tests are two-sided. We use `alpha = 0.05` for the significance flag.")

        exp_cols = st.columns(5)
        with exp_cols[0]:
            metric_card("Control CVR", f"{conversion.get('control_conversion_rate', 0):.2%}", "Baseline variant")
        with exp_cols[1]:
            metric_card("Treatment CVR", f"{conversion.get('treatment_conversion_rate', 0):.2%}", "New variant")
        with exp_cols[2]:
            metric_card("Absolute Lift", f"{conversion.get('absolute_lift', 0):.2%}", "Treatment minus control")
        with exp_cols[3]:
            metric_card("Z-test p-value", f"{conversion.get('p_value', 0):.4f}", "Conversion evidence")
        with exp_cols[4]:
            metric_card("T-test p-value", f"{order_value.get('p_value', 0):.4f}", "Order value evidence")

        sample_size = ab_results.get("estimated_sample_size_per_group")
        if sample_size:
            insight_card(
                "Power planning",
                f"Given the observed baseline and target lift assumption, the estimated sample size is {sample_size:,} observations per group.",
            )

        result_cols = st.columns([1, 1])
        conversion_df = pd.DataFrame(
            {
                "Metric": [
                    "Control conversion rate",
                    "Treatment conversion rate",
                    "Absolute lift",
                    "Relative lift",
                    "Z statistic",
                    "Z-test p-value",
                    "Significant at 5%",
                ],
                "Value": [
                    conversion.get("control_conversion_rate"),
                    conversion.get("treatment_conversion_rate"),
                    conversion.get("absolute_lift"),
                    conversion.get("relative_lift"),
                    conversion.get("z_statistic"),
                    conversion.get("p_value"),
                    conversion.get("is_significant_05"),
                ],
            }
        )
        order_value_df = pd.DataFrame(
            {
                "Metric": [
                    "Control mean order value",
                    "Treatment mean order value",
                    "Mean difference",
                    "T statistic",
                    "T-test p-value",
                    "Significant at 5%",
                ],
                "Value": [
                    order_value.get("control_mean_order_value"),
                    order_value.get("treatment_mean_order_value"),
                    order_value.get("mean_difference"),
                    order_value.get("t_statistic"),
                    order_value.get("p_value"),
                    order_value.get("is_significant_05"),
                ],
            }
        )
        conversion_df["Value"] = conversion_df["Value"].map(
            lambda value: "Yes" if value is True else "No" if value is False else value
        )
        order_value_df["Value"] = order_value_df["Value"].map(
            lambda value: "Yes" if value is True else "No" if value is False else value
        )
        conversion_df["Value"] = conversion_df["Value"].map(
            lambda value: f"{value:.4f}" if isinstance(value, float) else str(value)
        )
        order_value_df["Value"] = order_value_df["Value"].map(
            lambda value: f"{value:.4f}" if isinstance(value, float) else str(value)
        )
        with result_cols[0]:
            section_header("Conversion Z-Test", "Validates whether treatment conversion differs from control.")
            st.dataframe(conversion_df, width="stretch", hide_index=True)
        with result_cols[1]:
            section_header("Order Value T-Test", "Validates whether treatment order value differs from control.")
            st.dataframe(order_value_df, width="stretch", hide_index=True)
