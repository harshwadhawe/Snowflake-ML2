import streamlit as st
import pandas as pd
import altair as alt
from snowflake.snowpark.context import get_active_session

session = get_active_session()

DB = "HACKATHON"
SCHEMA = "PUBLIC"

st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")

st.title("Customer Churn Intelligence Dashboard")
st.caption("CFPB Complaint-Based Churn Prediction  |  XGBoost + SHAP Explainability  |  Snowflake ML")

# --- KPI Metrics ---
st.divider()

metrics_data = {"auc": 0.783, "recall": 0.704, "f1": 0.460}
churn_stats = session.sql(f"""
    SELECT
        COUNT(*) AS total,
        SUM(CHURN_LABEL) AS churned,
        ROUND(SUM(CHURN_LABEL) * 100.0 / COUNT(*), 1) AS churn_pct
    FROM {DB}.{SCHEMA}.CHURN_GOLD
""").to_pandas().iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Records", f"{int(churn_stats['TOTAL']):,}")
col2.metric("Churned Customers", f"{int(churn_stats['CHURNED']):,}")
col3.metric("Churn Rate", f"{churn_stats['CHURN_PCT']}%")
col4.metric("Model AUC", f"{metrics_data['auc']:.3f}", delta="Target: 0.75", delta_color="normal")
col5.metric("Recall / F1", f"{metrics_data['recall']:.2f} / {metrics_data['f1']:.2f}")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "SHAP Explainability",
    "Segment Analysis",
    "Model Details"
])

# --- Tab 1: SHAP ---
with tab1:
    st.subheader("SHAP Feature Importance")
    st.markdown("Mean absolute SHAP values reveal the true drivers of churn predictions — beyond simple feature importance.")

    shap_df = session.sql(f"""
        SELECT FEATURE, MEAN_ABS_SHAP
        FROM {DB}.{SCHEMA}.SHAP_FEATURE_IMPORTANCE
        ORDER BY MEAN_ABS_SHAP DESC
    """).to_pandas()

    driver_df = session.sql(f"""
        SELECT REPLACE(FEATURE, '_E', '') AS FEATURE, SCORE
        FROM {DB}.{SCHEMA}.MODEL_DRIVERS
        ORDER BY SCORE DESC
    """).to_pandas()

    left, right = st.columns(2)

    with left:
        st.markdown("**SHAP Values** (prediction impact)")
        shap_chart = alt.Chart(shap_df).mark_bar(
            cornerRadiusTopRight=4, cornerRadiusBottomRight=4
        ).encode(
            x=alt.X("MEAN_ABS_SHAP:Q", title="Mean |SHAP|"),
            y=alt.Y("FEATURE:N", sort="-x", title=None),
            color=alt.Color("MEAN_ABS_SHAP:Q", scale=alt.Scale(scheme="reds"), legend=None),
            tooltip=["FEATURE", alt.Tooltip("MEAN_ABS_SHAP:Q", format=".4f")]
        ).properties(height=400)
        st.altair_chart(shap_chart, use_container_width=True)

    with right:
        st.markdown("**XGBoost Feature Importance** (split-based)")
        driver_chart = alt.Chart(driver_df).mark_bar(
            cornerRadiusTopRight=4, cornerRadiusBottomRight=4
        ).encode(
            x=alt.X("SCORE:Q", title="Importance Score"),
            y=alt.Y("FEATURE:N", sort="-x", title=None),
            color=alt.Color("SCORE:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["FEATURE", alt.Tooltip("SCORE:Q", format=".4f")]
        ).properties(height=400)
        st.altair_chart(driver_chart, use_container_width=True)

    st.info(
        "**Key Insight:** SHAP reveals **Company** and **Sub-Issue** as the strongest churn drivers — "
        "meaning specific companies and complaint sub-issues are highly predictive of customer friction, "
        "even more than the complaint issue category itself."
    )

# --- Tab 2: Segment Analysis ---
with tab2:
    st.subheader("Churn by Segment")

    segment_type = st.selectbox(
        "Select Segment Dimension",
        ["SUB_PRODUCT", "STATE", "SUBMITTED_VIA"],
        format_func=lambda x: x.replace("_", " ").title()
    )

    seg_df = session.sql(f"""
        SELECT SEGMENT_VALUE, TOTAL_COUNT, CHURN_COUNT, CHURN_RATE, AVG_CHURN_PROBA
        FROM {DB}.{SCHEMA}.CHURN_SEGMENT_SCORES
        WHERE SEGMENT_TYPE = '{segment_type}'
        ORDER BY CHURN_RATE DESC
    """).to_pandas()

    top_n = st.slider("Show top N segments", 5, min(30, len(seg_df)), min(15, len(seg_df)))
    seg_display = seg_df.head(top_n)

    bar_chart = alt.Chart(seg_display).mark_bar(
        cornerRadiusTopRight=4, cornerRadiusBottomRight=4
    ).encode(
        x=alt.X("CHURN_RATE:Q", title="Churn Rate", axis=alt.Axis(format=".0%")),
        y=alt.Y("SEGMENT_VALUE:N", sort="-x", title=None),
        color=alt.Color("CHURN_RATE:Q", scale=alt.Scale(scheme="orangered"), legend=None),
        tooltip=[
            "SEGMENT_VALUE",
            alt.Tooltip("CHURN_RATE:Q", format=".1%", title="Churn Rate"),
            alt.Tooltip("TOTAL_COUNT:Q", format=",", title="Total"),
            alt.Tooltip("CHURN_COUNT:Q", format=",", title="Churned"),
            alt.Tooltip("AVG_CHURN_PROBA:Q", format=".3f", title="Avg Predicted Prob")
        ]
    ).properties(height=max(top_n * 28, 200))
    st.altair_chart(bar_chart, use_container_width=True)

    st.markdown("#### Segment Detail Table")
    st.dataframe(
        seg_display.style.format({
            "CHURN_RATE": "{:.1%}",
            "AVG_CHURN_PROBA": "{:.3f}",
            "TOTAL_COUNT": "{:,}",
            "CHURN_COUNT": "{:,}"
        }),
        use_container_width=True
    )

# --- Tab 3: Model Details ---
with tab3:
    st.subheader("Model Registry & Pipeline")

    left3, right3 = st.columns(2)

    with left3:
        st.markdown("#### Model Card")
        st.markdown(f"""
| Property | Value |
|----------|-------|
| **Model Name** | `CFPB_CHURN_PREDICTOR` |
| **Algorithm** | XGBoost Classifier |
| **AUC** | {metrics_data['auc']:.3f} |
| **Recall** | {metrics_data['recall']:.3f} |
| **F1 Score** | {metrics_data['f1']:.3f} |
| **Registry** | `{DB}.{SCHEMA}` |
| **Feature Store** | Entity: `CUSTOMER`, View: `CHURN_FEATURES_VIEW` |
| **Training Data** | CFPB Complaints (Finance & Economics) |
| **Records** | {int(churn_stats['TOTAL']):,} |
""")

    with right3:
        st.markdown("#### Hyperparameters")
        hp_df = pd.DataFrame({
            "Parameter": [
                "n_estimators", "max_depth", "learning_rate",
                "scale_pos_weight", "subsample", "colsample_bytree",
                "min_child_weight", "gamma"
            ],
            "Value": ["500", "12", "0.05", "4.0", "0.8", "0.8", "3", "0.1"]
        })
        st.dataframe(hp_df, use_container_width=True)

        st.markdown("#### Feature Pipeline")
        st.markdown("""
**Categorical** (OrdinalEncoder): `ISSUE`, `SUB_ISSUE`, `SUB_PRODUCT`, `STATE`, `SUBMITTED_VIA`, `COMPANY`

**Engineered**: `IS_FEE_RELATED`, `IS_DISPUTE_RELATED`, `RESPONSE_DELAY_DAYS`, `IS_TIMELY`, `IS_DISPUTED`
""")

    st.markdown("#### Churn Label Definition")
    st.code("""
CHURN_LABEL = 1 WHEN COMPANY_RESPONSE_TO_CONSUMER IN (
    'Closed with monetary relief',
    'Closed without relief',
    'Untimely response'
)""", language="sql")

    st.markdown("#### End-to-End Architecture")
    st.code("""
Marketplace Data (CFPB Complaints)
    -> Snowpark Feature Engineering
    -> Feature Store (Entity: CUSTOMER, View: CHURN_FEATURES_VIEW)
    -> Versioned Dataset Generation
    -> XGBoost Training (Snowflake ML)
    -> SHAP Explainability
    -> Model Registry (CFPB_CHURN_PREDICTOR)
    -> Streamlit Dashboard""", language="text")
