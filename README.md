# Customer Churn Intelligence

**TAMU CSEGSA x Snowflake Hackathon 2026 | Track A — ML Classification**

Predict customer churn from CFPB financial complaint data using XGBoost + SHAP explainability — entirely inside Snowflake. No external APIs, no external compute.

---

## Overview

Every financial institution loses customers silently. A complaint gets filed, a resolution feels inadequate, and the customer never comes back. This project answers: **can we predict which customers are about to churn using only the complaints they've already filed?**

| Metric | Value |
|--------|-------|
| Model | XGBoost Classifier |
| AUC | 0.783 |
| Recall | 0.704 |
| F1 Score | 0.460 |
| Training Records | 572,202 |
| Churned | 103,870 (18.2%) |

---

## Architecture

```
Snowflake Marketplace (FINANCIAL_CFPB_COMPLAINT)
  → Snowpark Feature Engineering
  → CHURN_GOLD (materialized gold table, change tracking enabled)
  → Feature Store (Entity: CUSTOMER, View: CHURN_FEATURES_VIEW V1)
  → Versioned Dataset (CHURN_TRAIN_SET) → 80/20 split
  → Snowflake ML Pipeline (OrdinalEncoder + XGBClassifier)
  → SHAP TreeExplainer (2,000-row sample)
  → Model Registry (CFPB_CHURN_PREDICTOR)
  → Support Tables: MODEL_DRIVERS, SHAP_FEATURE_IMPORTANCE, CHURN_SEGMENT_SCORES
  → Streamlit Dashboard (reads live from all support tables)
```

---

## Setup

**Requirements:** Snowflake account with access to the free Marketplace listing `SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.FINANCIAL_CFPB_COMPLAINT` and warehouse `HACKATHON_WH`.

1. Create database and schema:
```sql
CREATE DATABASE IF NOT EXISTS HACKATHON;
CREATE SCHEMA IF NOT EXISTS HACKATHON.PUBLIC;
```

2. Open `ML_customerchurn_v2.ipynb` in Snowflake Notebooks and run all cells top to bottom. This creates all four tables the dashboard depends on.

3. Create a new Streamlit app (Projects > Streamlit > + Streamlit App), set database to `HACKATHON`, schema to `PUBLIC`, and paste in `churn_app.py`.

---

## Snowflake Notebook — Training Pipeline

### Churn Label Definition

```sql
CHURN_LABEL = 1 WHEN COMPANY_RESPONSE_TO_CONSUMER IN (
    'Closed with monetary relief',
    'Closed without relief',
    'Untimely response'
)
```

Responses that signal the customer's problem was not resolved — the strongest proxy for intent to leave.

### Feature Engineering

| Feature | Type | Description |
|---------|------|-------------|
| `ISSUE` | Categorical | Complaint issue category |
| `SUB_ISSUE` | Categorical | Specific sub-issue |
| `SUB_PRODUCT` | Categorical | Financial product sub-type |
| `STATE` | Categorical | Customer state |
| `SUBMITTED_VIA` | Categorical | Submission channel (web, phone, etc.) |
| `COMPANY` | Categorical | Financial institution name |
| `IS_FEE_RELATED` | Binary | Issue mentions fees or interest |
| `IS_DISPUTE_RELATED` | Binary | Issue involves a dispute |
| `RESPONSE_DELAY_DAYS` | Numeric | Days between receipt and company response |
| `IS_TIMELY` | Binary | Whether response was timely |
| `IS_DISPUTED` | Binary | Whether consumer disputed the resolution |

### Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 500 |
| `max_depth` | 12 |
| `learning_rate` | 0.05 |
| `scale_pos_weight` | 4.0 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 3 |
| `gamma` | 0.1 |

`scale_pos_weight=4.0` compensates for the 18.2% churn rate, boosting recall at the cost of some precision — the right trade-off for a churn use case where missing a churner is costlier than a false alarm.

---

## Streamlit Dashboard

### Tab 1 — SHAP Explainability

![SHAP Explainability Dashboard](images/dashboard.png)

Two charts side by side:

**Left — SHAP Values (prediction impact)**
Mean absolute SHAP values across the test set. Shows how much each feature *actually moved* predictions — the honest measure of influence.

**Right — XGBoost Feature Importance (split-based)**
The internal split-gain importance the model reports natively.

Both methods agree: `Company` and `Sub_Issue` dominate. It's not *what kind* of complaint — it's *who* the complaint is against and *what specifically* went wrong.

### Tab 2 — Segment Analysis

![Churn by Segment](images/chrun_by_segment.png)

Filterable bar chart showing churn rate per segment across Sub-Product, State, or Submission Channel. The slider trims to the top N. The detail table adds raw counts and average predicted probability, separating high-rate/low-volume segments from those that represent real scale.

### Tab 3 — Model Details

![Model Registry & Pipeline](images/model_details.png)

Full model card: algorithm, metrics, registry location, Feature Store reference, training data source, record count, hyperparameter table, feature pipeline, churn label SQL, and end-to-end architecture diagram.

---

## SHAP Deep Dive

SHAP (SHapley Additive exPlanations) is a game-theoretic framework that explains individual predictions by assigning each feature a contribution value — positive pushes toward churn, negative pushes away.

**Why SHAP over standard feature importance?**

Standard XGBoost importance counts split frequency, which can overweight high-cardinality features and reveals no direction. SHAP is:
- **Consistent** — features that contribute more always receive a higher SHAP value
- **Local** — explains each prediction individually, not just the model globally
- **Directional** — shows whether a feature pushed a prediction toward or away from churn

**Top drivers:**

| Rank | Feature | Why it matters |
|------|---------|----------------|
| 1 | `COMPANY` | Certain institutions have structurally higher churn risk |
| 2 | `SUB_ISSUE` | The specific nature of the problem is highly predictive |
| 3 | `ISSUE` | Broad complaint category carries secondary signal |
| 4 | `IS_DISPUTE_RELATED` | Disputes signal active adversarial relationship |
| 5 | `SUB_PRODUCT` | Product type (checking vs. credit card) shapes risk profile |

**Implementation note:** Snowflake's bundled SHAP has a JSON parsing incompatibility with the Snowflake ML XGBoost wrapper. The notebook patches `shap.explainers._tree.XGBTreeModelLoader.__init__` to fix this. The patch has zero effect on computed SHAP values.

---

## Files

| File | Purpose |
|------|---------|
| `ML_customerchurn_v2.ipynb` | Training pipeline — run in Snowflake Notebooks |
| `churn_app.py` | Streamlit dashboard — paste into Streamlit in Snowflake |
| `images/` | Dashboard screenshots |
