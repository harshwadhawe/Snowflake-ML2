# Customer Churn Classification with SHAP Explainability

## 1. Project Overview

**Challenge:** Train a binary churn classifier using Snowflake ML on financial data. Explain predictions with SHAP values and surface results through an interactive Streamlit interface.

**Time Estimate:** 6–8 hours | **Points:** 100

### Scoring Breakdown

| Category | Points |
|----------|--------|
| Technical depth (Snowflake platform usage) | 30 |
| Model quality (AUC, F1, SHAP rigor) | 25 |
| Real-world impact & problem framing | 20 |
| Presentation & demo quality | 15 |
| Innovation & creative platform use | 10 |

### Deliverables

- Trained classifier in Model Registry with AUC >= 0.75
- SHAP feature importance report
- Streamlit app: churn scores per segment + explainability panel
- Snowflake Notebook with end-to-end pipeline

---

## 2. Infrastructure

| Resource | Value |
|----------|-------|
| **Account** | `lab52074` |
| **Username** | `LEARNER` |
| **Role** | `TRAINING_ROLE` |
| **Warehouse** | `HACKATHON_WH` |
| **Database** | `HACKATHON` |
| **Schema** | `PUBLIC` |
| **Marketplace Source DB** | `SNOWFLAKE_PUBLIC_DATA_FREE` |
| **Marketplace Source Schema** | `PUBLIC_DATA_FREE` |

---

## 3. Data Sources

### Source: Snowflake Marketplace — Finance & Economics (FREE)

**Table:** `SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.FINANCIAL_CFPB_COMPLAINT`

Consumer Financial Protection Bureau (CFPB) complaint records covering credit cards, bank accounts, and other financial products.

| # | Column | Data Type | Description |
|---|--------|-----------|-------------|
| 1 | ID_ZIP | TEXT | Zip code identifier |
| 2 | ID_STATE | TEXT | State identifier |
| 3 | PRODUCT | TEXT | Financial product type (e.g., "Credit card", "Bank account") |
| 4 | SUB_PRODUCT | TEXT | Product subcategory (e.g., "General-purpose credit card") |
| 5 | ISSUE | TEXT | Primary complaint issue |
| 6 | SUB_ISSUE | TEXT | Detailed complaint sub-issue |
| 7 | COMPANY | TEXT | Company the complaint is against |
| 8 | CONSUMER_COMPLAINT_NARRATIVE | TEXT | Free-text complaint narrative |
| 9 | COMPANY_PUBLIC_RESPONSE | TEXT | Company's public response |
| 10 | COMPANY_RESPONSE_TO_CONSUMER | TEXT | Company's direct response to consumer (used for churn label) |
| 11 | STATE | TEXT | Consumer's state |
| 12 | ZIP_CODE | TEXT | Consumer's zip code |
| 13 | TAGS | TEXT | Consumer tags (e.g., "Older American") |
| 14 | CONSUMER_CONSENT_PROVIDED | TEXT | Whether consumer consented to share narrative |
| 15 | SUBMITTED_VIA | TEXT | Submission channel (Web, Phone, Referral, etc.) |
| 16 | CONSUMER_DISPUTED | TEXT | Whether consumer disputed the response ("Yes"/"No") |
| 17 | DATE_RECEIVED | DATE | Date complaint was received |
| 18 | DATE_SENT_TO_COMPANY | DATE | Date complaint was sent to company |
| 19 | TIMELY_RESPONSE | BOOLEAN | Whether company responded in time |
| 20 | COMPLAINT_ID | NUMBER | Unique complaint identifier |

### Data Filtering

Records are filtered to credit card and bank account products:
```sql
WHERE LOWER(PRODUCT) LIKE '%credit card%'
   OR LOWER(PRODUCT) LIKE '%bank account%'
```

**Total filtered records:** 572,202

---

## 4. Data Dictionary — Output Tables

### 4.1 `HACKATHON.PUBLIC.CHURN_GOLD`

Primary gold-layer feature table with engineered features. **572,202 rows.** Change tracking enabled.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| ISSUE | VARCHAR | Yes | Primary complaint issue category |
| SUB_ISSUE | VARCHAR | Yes | Detailed complaint sub-issue |
| SUB_PRODUCT | VARCHAR | Yes | Product subcategory |
| STATE | VARCHAR | Yes | Consumer's US state |
| SUBMITTED_VIA | VARCHAR | Yes | Submission channel |
| COMPANY | VARCHAR | Yes | Company name |
| CHURN_LABEL | NUMBER(38,0) | No | Binary churn label (1 = churn, 0 = no churn) |
| IS_FEE_RELATED | NUMBER(38,0) | No | 1 if issue contains "fee" or "interest" |
| IS_DISPUTE_RELATED | NUMBER(38,0) | No | 1 if issue contains "dispute" or "purchase shown on your statement" |
| RESPONSE_DELAY_DAYS | NUMBER(38,0) | Yes | Days between DATE_RECEIVED and DATE_SENT_TO_COMPANY |
| IS_TIMELY | NUMBER(38,0) | No | 1 if TIMELY_RESPONSE = "Yes" |
| IS_DISPUTED | NUMBER(38,0) | No | 1 if CONSUMER_DISPUTED = "Yes" |
| ROW_ID | NUMBER(38,0) | No | Unique row identifier (SEQ8) for Feature Store entity join key |

### 4.2 `HACKATHON.PUBLIC.CHURN_FEATURES_GOLD`

Earlier version of gold table (v1 notebook). **571,841 rows.** Change tracking enabled.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| CHURN_LABEL | NUMBER(38,0) | No | Binary churn label |
| ISSUE | VARCHAR | Yes | Primary complaint issue |
| SUB_PRODUCT | VARCHAR | Yes | Product subcategory |
| STATE | VARCHAR | Yes | Consumer's state |
| SUBMITTED_VIA | VARCHAR | Yes | Submission channel |
| IS_FEE_RELATED | NUMBER(38,0) | No | Fee-related flag |
| IS_DISPUTE_RELATED | NUMBER(38,0) | No | Dispute-related flag |

### 4.3 `HACKATHON.PUBLIC.CHURN_FEATURES_GOLD_KEYED`

Keyed version of v1 gold table with ROW_ID added. **571,841 rows.** Change tracking enabled.

| Column | Type | Description |
|--------|------|-------------|
| ROW_ID | NUMBER(38,0) | Unique row identifier |
| *(same as CHURN_FEATURES_GOLD)* | | |

### 4.4 `HACKATHON.PUBLIC.CHURN_SEGMENT_SCORES`

Pre-computed churn rates by segment for Streamlit dashboard. **84 rows.**

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| SEGMENT_TYPE | VARCHAR | Yes | Dimension name: `STATE`, `SUBMITTED_VIA`, or `SUB_PRODUCT` |
| SEGMENT_VALUE | VARCHAR | Yes | Dimension value (e.g., "CA", "Web", "Checking account") |
| TOTAL_COUNT | NUMBER(38,0) | Yes | Total records in segment |
| CHURN_COUNT | NUMBER(38,0) | Yes | Churned records in segment |
| CHURN_RATE | FLOAT | Yes | Churn rate (CHURN_COUNT / TOTAL_COUNT) |
| AVG_CHURN_PROBA | FLOAT | Yes | Average predicted churn probability from model |

### 4.5 `HACKATHON.PUBLIC.MODEL_DRIVERS`

XGBoost split-based feature importance. **11 rows.**

| Column | Type | Description |
|--------|------|-------------|
| FEATURE | VARCHAR | Feature name (encoded cols have `_E` suffix) |
| SCORE | FLOAT | XGBoost feature importance score (gain-based) |

### 4.6 `HACKATHON.PUBLIC.SHAP_FEATURE_IMPORTANCE`

SHAP-based feature importance (mean absolute SHAP values). **11 rows.**

| Column | Type | Description |
|--------|------|-------------|
| FEATURE | VARCHAR | Feature name (clean, no `_E` suffix) |
| MEAN_ABS_SHAP | FLOAT | Mean absolute SHAP value across 2,000 samples |

### 4.7 `HACKATHON.PUBLIC.MODEL_FEATURE_IMPORTANCE`

Feature importance from v1 notebook model. **6 rows.**

| Column | Type | Description |
|--------|------|-------------|
| FEATURE | VARCHAR | Feature name |
| IMPORTANCE_SCORE | FLOAT | XGBoost importance score |

---

## 5. Feature Engineering

### Churn Label Definition

```sql
CHURN_LABEL = 1 WHEN COMPANY_RESPONSE_TO_CONSUMER IN (
    'Closed with monetary relief',
    'Closed without relief',
    'Untimely response'
)
-- Otherwise CHURN_LABEL = 0
```

**Class distribution:**
| Label | Count | Percentage |
|-------|-------|-----------|
| 0 (No Churn) | 468,332 | 81.85% |
| 1 (Churn) | 103,870 | 18.15% |

### Categorical Features (OrdinalEncoder)

| Feature | Source Column | Description |
|---------|-------------|-------------|
| ISSUE | ISSUE | Primary complaint category |
| SUB_ISSUE | SUB_ISSUE | Detailed sub-category |
| SUB_PRODUCT | SUB_PRODUCT | Financial product type |
| STATE | STATE | US state |
| SUBMITTED_VIA | SUBMITTED_VIA | Channel (Web, Phone, Referral, etc.) |
| COMPANY | COMPANY | Financial institution name |

### Engineered Numeric Features

| Feature | Logic | Description |
|---------|-------|-------------|
| IS_FEE_RELATED | `LOWER(ISSUE) LIKE '%fee%' OR LIKE '%interest%'` | Complaint about fees/interest |
| IS_DISPUTE_RELATED | `LOWER(ISSUE) LIKE '%dispute%' OR LIKE '%purchase shown on your statement%'` | Transaction dispute |
| RESPONSE_DELAY_DAYS | `DATEDIFF(day, DATE_RECEIVED, DATE_SENT_TO_COMPANY)` | Company response latency |
| IS_TIMELY | `TIMELY_RESPONSE = 'Yes'` | Timely response flag |
| IS_DISPUTED | `CONSUMER_DISPUTED = 'Yes'` | Consumer disputed flag |

---

## 6. Code Changes & Bug Fixes

### Fix 1: FeatureStore Missing `default_warehouse`

**Error:** `TypeError: FeatureStore.__init__() missing 1 required positional argument: 'default_warehouse'`

**Before:**
```python
fs = FeatureStore(session, "HACKATHON", "PUBLIC", creation_mode=CreationMode.CREATE_IF_NOT_EXIST)
```

**After:**
```python
fs = FeatureStore(session, "HACKATHON", "PUBLIC",
    default_warehouse="HACKATHON_WH",
    creation_mode=CreationMode.CREATE_IF_NOT_EXIST)
```

### Fix 2: Column `like_any` Not Available

**Error:** `AttributeError: 'Column' object has no attribute 'like_any'`

**Before:**
```python
F.lower(F.col("PRODUCT")).like_any(["%credit card%", "%bank account%"])
```

**After:**
```python
(F.lower(F.col("PRODUCT")).like("%credit card%")) |
(F.lower(F.col("PRODUCT")).like("%bank account%"))
```

### Fix 3: FeatureStore `CreationMode` Required

**Error:** `ValueError: Feature store internal tag SNOWML_FEATURE_STORE_OBJECT does not exist`

**Fix:** Added `creation_mode=CreationMode.CREATE_IF_NOT_EXIST` and imported `CreationMode`.

### Fix 4: FeatureView API Corrections

**Error:** `FeatureView` required `entities` and `feature_df` parameters.

**Before:**
```python
FeatureView(name="CHURN_VIEW_FINAL", query=df_gold_full)
```

**After:**
```python
FeatureView(name="CHURN_VIEW_FINAL", entities=[entity],
    feature_df=df_features_only, desc="...")
```

Also required creating an `Entity` with `ROW_ID` as join key.

### Fix 5: `generate_dataset` API Corrections

**Before:**
```python
fs.generate_dataset(name=..., features=[fv], label_col="CHURN_LABEL", output_type="table")
```

**After:**
```python
fs.generate_dataset(name=..., spine_df=spine_df, features=[fv],
    spine_label_cols=["CHURN_LABEL"])
```

### Fix 6: Dataset Read API

**Before:** `my_dataset.read().random_split(...)`

**After:** `my_dataset.read.to_snowpark_dataframe().random_split(...)`

### Fix 7: Temp Stage Permission Error

**Error:** `Insufficient privileges to operate on schema 'USER$LEARNER.PUBLIC'`

**Fix:** Added explicit database/schema context and pre-created a stage:
```python
session.sql("USE DATABASE HACKATHON").collect()
session.sql("USE SCHEMA PUBLIC").collect()
session.sql("CREATE STAGE IF NOT EXISTS HACKATHON.PUBLIC.SNOWPARK_TEMP_STAGE").collect()
```

### Fix 8: Dataset Version Conflicts

**Error:** `Dataset CHURN_TRAIN_SET version V7 already exists`

**Fix:** Used timestamp-based versioning:
```python
ds_version = datetime.datetime.now().strftime("%Y%m%d_%H%M")
```

### Fix 9: SHAP/XGBoost Compatibility

**Error:** `'XGBTreeModelLoader' object has no attribute 'num_trees'` and `'cat_feature_indices'`

**Fix:** Monkey-patched `XGBTreeModelLoader.__init__` to add missing attributes:
```python
self.num_trees = num_trees
self.n_trees_per_iter = max(int(num_trees / n_targets), 1)
self.cat_feature_indices = []
```

### Fix 10: Streamlit `hide_index` Not Supported

**Error:** `DataFrameSelectorMixin.dataframe() got an unexpected keyword argument 'hide_index'`

**Fix:** Removed `hide_index=True` from `st.dataframe()` calls (older Streamlit version in SiS).

---

## 7. Model Training & Results

### Algorithm

**XGBoost Classifier** via `snowflake.ml.modeling.xgboost.XGBClassifier` wrapped in a Snowpark ML `Pipeline` with `OrdinalEncoder`.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_estimators | 500 |
| max_depth | 12 |
| learning_rate | 0.05 |
| scale_pos_weight | 4.0 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 3 |
| gamma | 0.1 |

### AUC Progression

| Iteration | Changes | AUC | Recall | F1 |
|-----------|---------|-----|--------|-----|
| Baseline | Weak label (`LIKE '%relief%'`), basic features, default hyperparams | 0.532 | 0.114 | 0.188 |
| v1 | Broader label (3 response types), tuned hyperparams | 0.675 | 0.723 | 0.426 |
| v2 | Added SUB_ISSUE, COMPANY, RESPONSE_DELAY_DAYS, IS_TIMELY, IS_DISPUTED | 0.712 | 0.750 | 0.466 |
| v3 (Final) | Probability-based AUC (predict_proba), increased capacity | **0.782** | **0.708** | **0.466** |

### Final Model Metrics

| Metric | Value | Target |
|--------|-------|--------|
| **AUC** | **0.782** | >= 0.75 |
| **Recall** | 0.708 | — |
| **F1 Score** | 0.466 | — |

### Key Improvements That Drove AUC

1. **Better churn label** — Changed from `LIKE '%relief%'` to specific `.in_()` matching 3 high-friction response types
2. **Richer features** — Added 5 new features: `SUB_ISSUE`, `COMPANY`, `RESPONSE_DELAY_DAYS`, `IS_TIMELY`, `IS_DISPUTED`
3. **Proper AUC computation** — Used `predict_proba` from native XGBoost for probability-based AUC instead of hard 0/1 predictions
4. **Tuned hyperparameters** — Higher capacity (500 trees, depth 12), regularization (subsample, colsample_bytree, gamma)

---

## 8. SHAP Explainability

### SHAP Feature Importance (Mean |SHAP| Values)

| Rank | Feature | Mean |SHAP| | Interpretation |
|------|---------|-------------|----------------|
| 1 | COMPANY | 0.7328 | Which company the complaint is against is the strongest predictor |
| 2 | SUB_ISSUE | 0.7027 | Specific sub-issues strongly indicate churn risk |
| 3 | ISSUE | 0.3394 | Broad issue category matters but less than specifics |
| 4 | IS_DISPUTE_RELATED | 0.2012 | Transaction disputes are a strong churn signal |
| 5 | SUB_PRODUCT | 0.1707 | Product type (checking, credit card, etc.) |
| 6 | IS_FEE_RELATED | 0.1489 | Fee/interest complaints drive churn |
| 7 | STATE | 0.1305 | Geographic variation in churn |
| 8 | RESPONSE_DELAY_DAYS | 0.0797 | Slower responses correlate with churn |
| 9 | SUBMITTED_VIA | 0.0701 | Submission channel has moderate impact |
| 10 | IS_DISPUTED | 0.0655 | Consumer dispute flag |
| 11 | IS_TIMELY | 0.0131 | Timeliness has the least individual impact |

### XGBoost Feature Importance (Split-Based)

| Rank | Feature | Score |
|------|---------|-------|
| 1 | IS_FEE_RELATED | 0.2508 |
| 2 | IS_DISPUTE_RELATED | 0.2028 |
| 3 | SUB_ISSUE | 0.1780 |
| 4 | ISSUE | 0.0767 |
| 5 | COMPANY | 0.0619 |
| 6 | IS_DISPUTED | 0.0539 |
| 7 | SUB_PRODUCT | 0.0501 |
| 8 | IS_TIMELY | 0.0453 |
| 9 | SUBMITTED_VIA | 0.0286 |
| 10 | STATE | 0.0263 |
| 11 | RESPONSE_DELAY_DAYS | 0.0256 |

### Key Insight

SHAP and XGBoost importance tell different stories: SHAP reveals **COMPANY** and **SUB_ISSUE** as the strongest prediction drivers (meaning specific companies and complaint types are highly predictive), while XGBoost split-based importance highlights **IS_FEE_RELATED** and **IS_DISPUTE_RELATED** (binary engineered features that are easy to split on). Both views are valuable — SHAP captures the true prediction impact, while split-based importance shows what the model uses most frequently.

---

## 9. Model Registry & Feature Store

### Model Registry

| Property | Value |
|----------|-------|
| **Model Name** | `HACKATHON.PUBLIC.CFPB_CHURN_PREDICTOR` |
| **Model Type** | USER_MODEL |
| **Owner** | TRAINING_ROLE |
| **Default Version** | V1 |
| **Latest Version** | V_20260410_1655 |
| **Total Versions** | 15 |

### Feature Store

| Object | Name | Details |
|--------|------|---------|
| **Entity** | `CUSTOMER` | Join key: `ROW_ID` |
| **Entity** (v1) | `CHURN_ENTITY` | Join key: `ROW_ID` |
| **Feature View** | `CHURN_FEATURES_VIEW` | All features except CHURN_LABEL |
| **Dataset** | `CHURN_TRAIN_SET` | Versioned with timestamp (e.g., `20260410_1655`) |
| **Internal Tag** | `SNOWML_FEATURE_STORE_OBJECT` | Auto-created by Feature Store |
| **Metadata Tag** | `SNOWML_FEATURE_VIEW_METADATA` | Auto-created by Feature Store |

---

## 10. Streamlit Dashboard

### Apps

| App Name | Object Name | URL ID | Warehouse |
|----------|------------|--------|-----------|
| Customer Churn Intelligence | `CHURN_INTELLIGENCE_APP` | `blvwkmmckunyiywyidsi` | HACKATHON_WH |
| Friction Analysis | `AB42W2L5NKKPV61X` | `nyfgla3ybnlh43rc7r6e` | HACKATHON_WH |

### Dashboard Tabs

**Tab 1 — SHAP Explainability**
- Side-by-side bar charts: SHAP values (left) vs XGBoost importance (right)
- Color-coded by magnitude (reds for SHAP, blues for XGBoost)
- Key insight callout highlighting top churn drivers

**Tab 2 — Segment Analysis**
- Interactive dropdown: SUB_PRODUCT, STATE, or SUBMITTED_VIA
- Adjustable top-N slider
- Horizontal bar chart with churn rate by segment
- Detailed data table with churn rate, count, and average predicted probability

**Tab 3 — Model Details**
- Model card with AUC/Recall/F1 metrics
- Hyperparameter table
- Feature pipeline description
- Churn label SQL definition
- End-to-end architecture diagram

### Top-Level KPI Bar
- Total Records: 572,202
- Churned Customers: 103,870
- Churn Rate: 18.2%
- Model AUC: 0.782 (with target indicator)
- Recall / F1: 0.71 / 0.47

### Source File

`/churn_app.py` — deployed to `@HACKATHON.PUBLIC.STREAMLIT_STAGE`

---

## 11. Workspace Files

| File | Description |
|------|-------------|
| `/ML_customerchurn.ipynb` | v1 notebook — initial pipeline |
| `/ML_customerchurn_v2.ipynb` | v2 notebook — final pipeline with enhanced features and SHAP |
| `/churn_app.py` | Streamlit dashboard source code |
| `/Analytics_p1.sql` | SQL analytics queries |
| `/Initial.sql` | Initial setup SQL |
| `/demo.py` | Demo script |
| `/snowflake.py` | Snowflake utilities |
| `/engagement_logs.csv` | Engagement log data |
| `/Sample Data/` | Directory with sample CSVs (access_logs, hr_attrition, etc.) |

---

## 12. End-to-End Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SNOWFLAKE MARKETPLACE                        │
│  SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE                    │
│  └── FINANCIAL_CFPB_COMPLAINT (572K+ records)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SNOWPARK FEATURE ENGINEERING                    │
│  • Filter: credit card + bank account products                  │
│  • Label: 3 high-friction company responses → CHURN_LABEL       │
│  • Engineer: IS_FEE_RELATED, IS_DISPUTE_RELATED,                │
│    RESPONSE_DELAY_DAYS, IS_TIMELY, IS_DISPUTED                  │
│  • Output: HACKATHON.PUBLIC.CHURN_GOLD                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FEATURE STORE                                │
│  Entity: CUSTOMER (join_key: ROW_ID)                            │
│  Feature View: CHURN_FEATURES_VIEW                              │
│  Dataset: CHURN_TRAIN_SET (versioned)                           │
│  Train/Test Split: 80/20                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   XGBOOST TRAINING                               │
│  Pipeline: OrdinalEncoder → XGBClassifier                       │
│  500 trees, depth 12, lr 0.05, scale_pos_weight 4.0             │
│  AUC: 0.782 | Recall: 0.708 | F1: 0.466                        │
└────────────┬───────────────────────────┬────────────────────────┘
             │                           │
             ▼                           ▼
┌────────────────────────┐  ┌────────────────────────────────────┐
│     MODEL REGISTRY     │  │        SHAP EXPLAINABILITY          │
│  CFPB_CHURN_PREDICTOR  │  │  TreeExplainer on 2K samples        │
│  15 versions           │  │  Top: COMPANY (0.73), SUB_ISSUE     │
│  Metrics logged        │  │  (0.70), ISSUE (0.34)               │
└────────────────────────┘  └────────────────────────────────────┘
             │                           │
             └─────────┬─────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD                            │
│  Tab 1: SHAP Explainability (dual bar charts)                   │
│  Tab 2: Segment Analysis (State/Product/Channel)                │
│  Tab 3: Model Details (card, hyperparams, architecture)         │
│  KPI Bar: 572K records, 18.2% churn, AUC 0.782                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Generated: April 10, 2026 | Platform: Snowflake | Notebook: ML_customerchurn_v2.ipynb*
