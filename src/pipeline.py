"""
EDU-026-003: Predictive Analytics for K-12 Achievement Gap in Louisiana
CSC 580 - Advanced Data Mining, Fusion and Applications
Louisiana Tech University | Spring 2026
Author: Akshitha Merugu
Instructor: Dr. Pradeep Chowriappa

DESCRIPTION:
    Complete ML pipeline from data loading through SHAP explainability.
    Trains three classifiers (Logistic Regression, Random Forest, XGBoost)
    on Louisiana LEAP assessment data and generates risk predictions for
    school-subgroup combinations.

HOW TO RUN:
    1. Upload all 11 data files to your working directory
    2. Set DATA_DIR to the folder containing your files
    3. Run: python pipeline.py

REQUIRED FILES:
    LEAP subgroup summaries (5 files):
        2019-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx
        2022-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx
        2023-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx
        2024-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx
        2025-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx

    School Performance Scores (5 files):
        2019-school-performance-scores.xlsx
        2022-school-performance-scores.xlsx
        2023-school-performance-scores.xlsx
        2024-school-performance-scores.xlsx
        2025-school-performance-scores.xlsx

    Census poverty data (1 file):
        ACSST5Y2023_S1701-2026-05-07T193340.csv

INSTALL DEPENDENCIES:
    pip install pandas scikit-learn shap openpyxl

FOR GOOGLE COLAB:
    !pip install shap
    Change DATA_DIR = '/content/'
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
import shap
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION: Change this path to match your setup
# Google Colab: DATA_DIR = "/content/"
# Local:        DATA_DIR = "./data/raw/"
# ============================================================
DATA_DIR = "./data/raw/"
OUTPUT_DIR = "./results/"

print("=" * 70)
print("EDU-026-003: K-12 Achievement Gap Prediction Pipeline")
print("Akshitha Merugu | CSC 580 | Louisiana Tech University")
print("=" * 70)


# ============================================================
# STEP 1: LOAD LEAP SUBGROUP DATA
# ============================================================
def load_leap_file(filepath, year):
    """
    Load one LEAP subgroup XLSX file.

    The Louisiana DOE files have headers spread across rows 3-4,
    with actual data starting at row 5. This function handles
    that non-standard format.

    Args:
        filepath (str): Full path to the XLSX file
        year (int): Year of the data (2019, 2022-2025)

    Returns:
        pd.DataFrame: Raw dataframe with year column added
    """
    df = pd.read_excel(filepath, header=None)

    # Rows 3-4 are the merged header rows
    header_main = df.iloc[3].fillna("").astype(str).tolist()
    header_sub = df.iloc[4].fillna("").astype(str).tolist()

    # Build column names by combining subject + achievement level
    columns = []
    current_subject = ""
    for i, (main, sub) in enumerate(zip(header_main, header_sub)):
        main = main.strip()
        sub = sub.strip()
        if main and main not in ["", "nan"]:
            if main in ["English Language Arts", "Mathematics",
                        "Science", "Social Studies", "All Subjects"]:
                current_subject = main
                columns.append(f"{current_subject}_{sub}" if sub else current_subject)
            else:
                columns.append(main)
        elif sub:
            columns.append(f"{current_subject}_{sub}" if current_subject else sub)
        else:
            columns.append(f"col_{i}")

    # Data starts at row 5
    df = df.iloc[5:].copy()
    df.columns = columns[:len(df.columns)]
    df["year"] = year

    # Keep only school-level records (not state or district totals)
    df = df[df.iloc[:, 0].astype(str).str.strip() == "School"].copy()
    return df


def clean_percentage(series):
    """
    Clean percentage columns from LEAP files.
    Handles suppressed values: '< 5' -> 3, 'NR' -> NaN, '*' -> NaN
    """
    return pd.to_numeric(
        series.astype(str).str.strip()
        .str.replace("< 5", "3", regex=False)
        .str.replace("<=5", "3", regex=False)
        .str.replace("NR", "", regex=False)
        .str.replace("*", "", regex=False),
        errors="coerce"
    )


def standardize_leap(df, year):
    """
    Extract a consistent schema from a LEAP file regardless of year.

    Args:
        df (pd.DataFrame): Raw LEAP dataframe
        year (int): Year of the data

    Returns:
        pd.DataFrame: Standardized dataframe with consistent columns
    """
    result = pd.DataFrame()
    result["school_system"] = df["School System Name"].astype(str).str.strip()
    result["school_code"] = df["School Code"].astype(str).str.strip()
    result["school_name"] = df["School Name"].astype(str).str.strip()
    result["grade"] = df["Grade"].astype(str).str.strip()
    result["subgroup"] = df["Subgroup"].astype(str).str.strip()
    result["year"] = year

    # Extract ELA and Math achievement levels
    for subject_key, subject_short in [
        ("English Language Arts", "ela"),
        ("Mathematics", "math")
    ]:
        for level in ["Unsatisfactory", "Approaching Basic",
                      "Basic", "Mastery", "Advanced"]:
            matching = [c for c in df.columns
                        if level in c and subject_key in c]
            if matching:
                col_name = f"{subject_short}_{level.lower().replace(' ', '_')}"
                result[col_name] = clean_percentage(df[matching[0]])

    # Proficiency = Mastery + Advanced
    if "ela_mastery" in result.columns and "ela_advanced" in result.columns:
        result["ela_proficiency"] = result["ela_mastery"] + result["ela_advanced"]
    if "math_mastery" in result.columns and "math_advanced" in result.columns:
        result["math_proficiency"] = result["math_mastery"] + result["math_advanced"]

    return result


print("\n[STEP 1/10] Loading LEAP subgroup data (5 years)...")
leap_files = {
    2019: "2019-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx",
    2022: "2022-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx",
    2023: "2023-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx",
    2024: "2024-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx",
    2025: "2025-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx",
}

all_leap = []
for year, filename in leap_files.items():
    raw = load_leap_file(f"{DATA_DIR}{filename}", year)
    standardized = standardize_leap(raw, year)
    all_leap.append(standardized)
    print(f"   {year}: {len(standardized):,} records")

leap = pd.concat(all_leap, ignore_index=True)
print(f"   TOTAL: {len(leap):,} raw records")
print(f"   Subgroups: {leap['subgroup'].nunique()} unique groups")


# ============================================================
# STEP 2: AGGREGATE TO SCHOOL-SUBGROUP LEVEL
# ============================================================
print("\n[STEP 2/10] Aggregating across grades 3-8...")

# KDD Step: Data Reduction
# Average achievement percentages across grades for each school-subgroup
numeric_cols = [c for c in leap.columns if c not in
                ["school_system", "school_code", "school_name",
                 "grade", "subgroup", "year"]]

school_subgroup = leap.groupby(
    ["school_system", "school_code", "school_name", "subgroup", "year"]
)[numeric_cols].mean().reset_index()

print(f"   Aggregated to {len(school_subgroup):,} school-subgroup records")


# ============================================================
# STEP 3: LOAD SCHOOL PERFORMANCE SCORES
# ============================================================
print("\n[STEP 3/10] Loading School Performance Scores (5 years)...")

sps_all = []
for year in [2019, 2022, 2023, 2024, 2025]:
    df = pd.read_excel(
        f"{DATA_DIR}{year}-school-performance-scores.xlsx",
        header=None
    )

    # Find header row (contains "Site Code")
    header_row = 0
    for i in range(10):
        if any("Site Code" in str(v) for v in df.iloc[i].tolist()):
            header_row = i
            break

    df.columns = df.iloc[header_row].astype(str).tolist()
    df = df.iloc[header_row + 1:]

    # Find SPS column for this year
    sps_col = [c for c in df.columns
               if "SPS" in str(c) and str(year) in str(c)]
    type_col = [c for c in df.columns if "School Type" in str(c)]

    sps_df = pd.DataFrame()
    sps_df["school_code"] = df["Site Code"].astype(str).str.strip()
    if sps_col:
        sps_df["sps"] = pd.to_numeric(df[sps_col[0]], errors="coerce")
    if type_col:
        sps_df["school_type"] = df[type_col[0]].astype(str).str.strip()
    sps_df["year"] = year
    sps_all.append(sps_df)
    print(f"   {year}: {len(sps_df):,} schools")

sps = pd.concat(sps_all, ignore_index=True)


# ============================================================
# STEP 4: LOAD CENSUS POVERTY DATA
# ============================================================
print("\n[STEP 4/10] Loading Census ACS poverty data...")

census = pd.read_csv(
    f"{DATA_DIR}ACSST5Y2023_S1701-2026-05-07T193340.csv"
)

parish_poverty = {}
for col in census.columns:
    if ("Percent below poverty level" in col
            and "Estimate" in col
            and "Margin" not in col):
        parish_name = (col.split("!!")[0]
                       .replace(" Parish, Louisiana", "")
                       .strip())
        try:
            parish_poverty[parish_name] = float(
                str(census.iloc[0][col]).replace(",", "")
            )
        except (ValueError, TypeError):
            pass

print(f"   {len(parish_poverty)} parishes with poverty data")


# ============================================================
# STEP 5: MERGE ALL DATA SOURCES
# ============================================================
print("\n[STEP 5/10] Merging LEAP + SPS + Census (data integration)...")

school_subgroup["school_code"] = school_subgroup["school_code"].str.strip()
sps["school_code"] = sps["school_code"].str.strip()

# Merge LEAP with SPS on school_code + year
merged = school_subgroup.merge(sps, on=["school_code", "year"], how="left")

# Merge with Census on parish name
merged["parish"] = (merged["school_system"]
                    .str.replace(" Parish", "")
                    .str.replace(" School District", "")
                    .str.strip())
merged["poverty_rate"] = merged["parish"].map(parish_poverty)

print(f"   Merged dataset: {len(merged):,} records")
print(f"   Records with SPS: {merged['sps'].notna().sum():,}")
print(f"   Records with poverty: {merged['poverty_rate'].notna().sum():,}")


# ============================================================
# STEP 6: FEATURE ENGINEERING
# ============================================================
print("\n[STEP 6/10] Engineering features (data transformation)...")

# Target variable: at-risk if ELA Unsatisfactory > 20% OR Math > 25%
# Threshold calibrated to produce ~30% positive class rate
merged["at_risk"] = (
    (merged["ela_unsatisfactory"] > 20) |
    (merged["math_unsatisfactory"] > 25)
).astype(int)

# Encode categorical variables
le_subgroup = LabelEncoder()
merged["subgroup_enc"] = le_subgroup.fit_transform(
    merged["subgroup"].fillna("Unknown")
)

le_type = LabelEncoder()
merged["type_enc"] = le_type.fit_transform(
    merged["school_type"].fillna("Unknown")
)

# Binary flags for demographic groups
merged["is_minority"] = merged["subgroup"].isin([
    "Black or African American", "Hispanic/Latino", "English Learner"
]).astype(int)

merged["is_disadv"] = merged["subgroup"].isin([
    "Economically Disadvantaged",
    "Students with Disabilities",
    "English Learner"
]).astype(int)

# Interaction term: poverty x minority (captures compounding effect)
merged["pov_x_min"] = (
    merged["poverty_rate"].fillna(0) * merged["is_minority"]
)

# Final feature list (11 features)
features = [
    "sps",              # School Performance Score
    "poverty_rate",     # Parish-level poverty (Census ACS)
    "subgroup_enc",     # Encoded subgroup identity
    "type_enc",         # Encoded school type
    "is_minority",      # Binary: minority subgroup flag
    "is_disadv",        # Binary: disadvantaged subgroup flag
    "pov_x_min",        # Interaction: poverty x minority
    "ela_mastery",      # ELA mastery rate (key predictor)
    "math_mastery",     # Math mastery rate
    "ela_proficiency",  # ELA proficiency (mastery + advanced)
    "math_proficiency", # Math proficiency (mastery + advanced)
]

# Drop records missing target variable, impute remaining features
model_df = merged.dropna(subset=["ela_unsatisfactory", "at_risk"]).copy()
for f in features:
    if model_df[f].notna().any():
        model_df[f] = model_df[f].fillna(model_df[f].median())
    else:
        model_df[f] = model_df[f].fillna(0)

print(f"   Model-ready records: {len(model_df):,}")
print(f"   At-risk rate: {model_df['at_risk'].mean() * 100:.1f}%")
print(f"   Features: {features}")


# ============================================================
# STEP 7: TEMPORAL TRAIN / VALIDATE / TEST SPLIT
# ============================================================
print("\n[STEP 7/10] Temporal split (Lecture 2: split BEFORE scaling)...")

# NOTE: 2020 excluded (COVID cancelled testing)
#       2021 excluded (unreliable post-COVID recovery data)
train = model_df[model_df["year"].isin([2019, 2022, 2023])]
val   = model_df[model_df["year"] == 2024]
test  = model_df[model_df["year"] == 2025]

X_train, y_train = train[features], train["at_risk"]
X_val,   y_val   = val[features],   val["at_risk"]
X_test,  y_test  = test[features],  test["at_risk"]

print(f"   Training set:   {len(X_train):,} records (2019+2022+2023)")
print(f"   Validation set: {len(X_val):,} records (2024)")
print(f"   Test set:       {len(X_test):,} records (2025)")


# ============================================================
# STEP 8: TRAIN THREE MODELS
# ============================================================
print("\n[STEP 8/10] Training three models...")

# Scale features for Logistic Regression (tree models don't need this)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# Model 1: Logistic Regression (baseline - simplest possible model)
print("   Training Logistic Regression (baseline)...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

# Model 2: Random Forest (comparison - ensemble of trees)
print("   Training Random Forest (comparison)...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42
)
rf.fit(X_train, y_train)

# Model 3: Gradient Boosting / XGBoost (primary - iterative correction)
print("   Training Gradient Boosting / XGBoost (primary)...")
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=5,
    learning_rate=0.1, subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)

# Evaluate all three models
print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)
print(f"{'Model':<30} {'Acc':>7} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}")
print("-" * 70)

for name, model, Xv, Xt in [
    ("Logistic Regression", lr, X_val_s, X_test_s),
    ("Random Forest",       rf, X_val,   X_test),
    ("Gradient Boosting",   gb, X_val,   X_test),
]:
    vp = model.predict(Xv)
    vpr = model.predict_proba(Xv)[:, 1]
    tp = model.predict(Xt)
    tpr = model.predict_proba(Xt)[:, 1]
    print(f"\n  {name} (validation):")
    print(f"    Acc={accuracy_score(y_val, vp)*100:.1f}%  "
          f"AUC={roc_auc_score(y_val, vpr):.4f}  "
          f"Rec={recall_score(y_val, vp)*100:.1f}%  "
          f"Prec={precision_score(y_val, vp)*100:.1f}%")
    print(f"  {name} (test):")
    print(f"    Acc={accuracy_score(y_test, tp)*100:.1f}%  "
          f"AUC={roc_auc_score(y_test, tpr):.4f}  "
          f"Rec={recall_score(y_test, tp)*100:.1f}%  "
          f"Prec={precision_score(y_test, tp)*100:.1f}%")


# ============================================================
# STEP 9: DETAILED ANALYSIS OF PRIMARY MODEL (XGBoost)
# ============================================================
print("\n" + "=" * 70)
print("GRADIENT BOOSTING (PRIMARY MODEL) DETAILED ANALYSIS")
print("=" * 70)

gb_pred = gb.predict(X_test)
gb_prob = gb.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, gb_pred)

print(f"\n  Confusion Matrix:")
print(f"    True Negatives  (correctly safe): {cm[0][0]:,}")
print(f"    False Positives (false alarms):   {cm[0][1]:,}")
print(f"    False Negatives (missed at-risk): {cm[1][0]:,}")
print(f"    True Positives  (correctly caught): {cm[1][1]:,}")

at_risk_total  = cm[1][0] + cm[1][1]
at_risk_caught = cm[1][1]
print(f"\n  OPERATIONAL METRIC:")
print(f"    At-risk subgroups in 2025:  {at_risk_total:,}")
print(f"    Correctly flagged by model: {at_risk_caught:,}")
print(f"    Percentage caught:          {at_risk_caught/at_risk_total*100:.1f}%")
print(f"    Prediction lead time:       1 full academic year")


# ============================================================
# STEP 10: SHAP EXPLAINABILITY
# ============================================================
print("\n[STEP 9/10] Running SHAP analysis (TreeExplainer)...")

explainer   = shap.TreeExplainer(gb)
shap_values = explainer.shap_values(X_test)
shap_importance = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": shap_importance
}).sort_values("mean_abs_shap", ascending=False)

print("\n  SHAP Feature Importance (mean |SHAP value|):")
for _, row in shap_df.iterrows():
    bar = "█" * int(row["mean_abs_shap"] * 10)
    print(f"    {row['feature']:25s}: {row['mean_abs_shap']:.4f} {bar}")


# ============================================================
# ACHIEVEMENT GAP ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("REAL ACHIEVEMENT GAPS (2025 LEAP DATA)")
print("=" * 70)

test_full = test.copy()
test_full["risk_score"] = gb_prob
test_full["risk_flag"]  = gb_pred

subgroups_to_show = [
    "Total Population", "Black or African American", "White",
    "Economically Disadvantaged", "English Learner",
    "Students with Disabilities", "Hispanic/Latino", "Homeless"
]

print(f"\n  {'Subgroup':<40s} {'ELA Unsat':>9} {'At-Risk %':>9} "
      f"{'Avg Score':>9} {'N':>6}")
print("  " + "-" * 78)

for sg in subgroups_to_show:
    mask = test_full["subgroup"] == sg
    if mask.sum() > 0:
        ela_u = test_full.loc[mask, "ela_unsatisfactory"].mean()
        ar    = test_full.loc[mask, "at_risk"].mean() * 100
        rs    = test_full.loc[mask, "risk_score"].mean()
        n     = mask.sum()
        print(f"  {sg:<40s} {ela_u:>8.1f}% {ar:>8.1f}% "
              f"{rs:>9.2f} {n:>6}")

black_risk = test_full[test_full["subgroup"]=="Black or African American"]["at_risk"].mean()
white_risk = test_full[test_full["subgroup"]=="White"]["at_risk"].mean()
print(f"\n  Black/White risk ratio: {black_risk/white_risk:.1f}x")


# ============================================================
# SAVE OUTPUTS
# ============================================================
print("\n[STEP 10/10] Saving outputs...")
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save predictions
output_path = f"{OUTPUT_DIR}predictions_2025.csv"
test_full[[
    "school_name", "parish", "subgroup", "year",
    "ela_unsatisfactory", "math_unsatisfactory",
    "sps", "risk_score", "risk_flag", "at_risk"
]].to_csv(output_path, index=False)
print(f"   Predictions saved to {output_path}")

# Save SHAP importance
shap_path = f"{OUTPUT_DIR}shap_importance.csv"
shap_df.to_csv(shap_path, index=False)
print(f"   SHAP importance saved to {shap_path}")

print(f"\n{'=' * 70}")
print(f"PIPELINE COMPLETE")
print(f"{'=' * 70}")
print(f"  Total records processed:   {len(leap):,}")
print(f"  Model-ready records:       {len(model_df):,}")
print(f"  Primary model:             Gradient Boosting (XGBoost)")
print(f"  Test Accuracy:             {accuracy_score(y_test, gb_pred)*100:.1f}%")
print(f"  Test AUC:                  {roc_auc_score(y_test, gb_prob):.4f}")
print(f"  Test Recall:               {recall_score(y_test, gb_pred)*100:.1f}%")
print(f"  Operational recall:        {at_risk_caught}/{at_risk_total} = {at_risk_caught/at_risk_total*100:.1f}%")
print(f"  Top SHAP feature:          {shap_df.iloc[0]['feature']} ({shap_df.iloc[0]['mean_abs_shap']:.4f})")
print(f"{'=' * 70}")
