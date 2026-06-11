# EDU-026-003: Predictive Analytics for K-12 Achievement Gap in Louisiana

**Course:** CSC 580 - Advanced Data Mining, Fusion and Applications  
**Institution:** Louisiana Tech University | Spring 2026  
**Author:** Akshitha Merugu  
**Instructor:** Dr. Pradeep Chowriappa  
**RFP:** EDU-026-003

---

## Project Overview

Louisiana consistently ranks among the lowest-performing states in K-12 education. Black fourth-graders score approximately 27 points below White peers in reading, a gap unchanged since 1998. English Learners, students with disabilities, and economically disadvantaged students face compounding barriers that push them further behind.

The core problem is timing. Teachers only see LEAP assessment results after the school year ends. By the time a teacher knows which students struggled, those students have already moved on.

**This project builds a machine learning early warning system that predicts which school-subgroups are at risk of underperforming on LEAP assessments at least one full year in advance**, giving educators enough time to intervene with tutoring, mentoring, or additional support.

---

## Results

| Metric | Target | Achieved |
|---|---|---|
| Test Accuracy | - | 87.9% |
| ROC-AUC | > 0.90 | 0.946 |
| Recall | > 80% | 83.3% |
| Precision | > 75% | 77.1% |
| At-risk groups caught | > 80% | 2,652 / 3,185 (83.3%) |
| Prediction lead time | >= 1 semester | 1 full academic year |

### Achievement Gaps Confirmed (2025 Data)

| Subgroup | At-Risk Rate |
|---|---|
| English Learner | 91.7% |
| Students with Disabilities | 72.9% |
| Homeless | 66.7% |
| Hispanic/Latino | 43.0% |
| Black/African American | 32.0% |
| Economically Disadvantaged | 27.1% |
| White | 8.6% |

**Black students are 3.7x more likely to be at-risk than White students.**

---

## Repository Structure

```
EDU-026-003/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── src/
│   └── pipeline.py              # Complete ML pipeline (main code)
├── dashboard/
│   └── dashboard.html           # Interactive educator dashboard
├── data/
│   ├── raw/                     # Raw XLSX files (not included, see below)
│   └── processed/               # Generated after running pipeline
└── results/
    ├── predictions_2025.csv     # Model predictions on 2025 data
    └── shap_importance.csv      # SHAP feature importance values
```

---

## Data Sources

All data is publicly available. Download from the links below and place in `data/raw/`.

| File | Source | Link |
|---|---|---|
| LEAP Subgroup Summaries (2019, 2022-2025) | Louisiana DOE | https://doe.louisiana.gov/data-and-reports |
| School Performance Scores (2019, 2022-2025) | Louisiana DOE | https://doe.louisiana.gov/data-and-reports |
| Census ACS S1701 (2023 5-Year) | U.S. Census Bureau | https://data.census.gov |

**Note:** 2020 excluded (COVID cancelled testing). 2021 excluded (unreliable recovery-year data).

**Required filenames:**
```
2019-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx
2022-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx
2023-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx
2024-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx
2025-state-lea-school-leap-grade-3-8-achievement-level-subgroup-summary.xlsx
2019-school-performance-scores.xlsx
2022-school-performance-scores.xlsx
2023-school-performance-scores.xlsx
2024-school-performance-scores.xlsx
2025-school-performance-scores.xlsx
ACSST5Y2023_S1701-2026-05-07T193340.csv
```

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload all 11 data files using the Files panel on the left
3. Create a new notebook and run:

```python
!pip install shap openpyxl
```

4. Upload `src/pipeline.py` or paste its contents into a cell
5. Change line 57 in the code:
```python
DATA_DIR = "/content/"   # Colab upload folder
OUTPUT_DIR = "/content/" # Save results here
```
6. Run the cell

### Option 2: Local Machine

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/EDU-026-003.git
cd EDU-026-003

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your data files in data/raw/

# 4. Run the pipeline
python src/pipeline.py
```

---

## Interactive Dashboard

The dashboard is a standalone HTML file that requires no installation.

**To open:** Download `dashboard/dashboard.html` and double-click it. Opens in any browser.

### Dashboard Tabs

| Tab | Audience | Shows |
|---|---|---|
| Overview | Everyone | Key stats, prediction outcomes, key findings |
| Achievement Gaps | Administrators | Bar chart + table of all subgroup disparities |
| Model Performance | Technical staff | 3-model comparison, confusion matrix |
| School Drill-Down | Teachers / Counselors | Top risk schools with plain-language explanations |
| SHAP Explainability | Everyone | Feature importance + teacher-facing example |

**Screenshot:**

> The dashboard shows a teacher exactly what to act on:  
> *"Students with Disabilities at IDEA Bridge Elementary are HIGH RISK.  
> Factors: ELA mastery very low (52% unsatisfactory), SPS 38 vs state avg 75.  
> Recommended: prioritize for immediate reading intervention."*

---

## Methodology

This project follows the **CRISP-DM / KDD pipeline** (Fayyad et al., 1996):

| Phase | What was done |
|---|---|
| 1. Business understanding | Defined K-12 achievement gap problem |
| 2. Data understanding | Inspected 330,279 LEAP records across 5 years |
| 3. Data preparation | Cleaned, merged, reduced, transformed into 87,106 records |
| 4. Modeling | Trained Logistic Regression, Random Forest, XGBoost |
| 5. Evaluation | Temporal validation on held-out 2025 data |
| 6. Deployment | Interactive dashboard for educators |

### Temporal Split (no data leakage)

```
Train:    2019 + 2022 + 2023  →  27,747 records
Validate: 2024               →  10,890 records
Test:     2025               →  10,908 records
```

The model NEVER sees future data during training. This mirrors real-world deployment.

### Three Models Compared

| Model | Role | Why used |
|---|---|---|
| Logistic Regression | Baseline | Proves need for nonlinear models |
| Random Forest | Comparison | Ensemble bagging with 200 trees |
| Gradient Boosting (XGBoost) | Primary | Iterative boosting, best validation AUC, aligns with required papers |

### 11 Features

| Feature | Source |
|---|---|
| ELA Mastery rate | LEAP (strongest predictor, SHAP=1.21) |
| ELA Proficiency | LEAP |
| Math Mastery rate | LEAP |
| Math Proficiency | LEAP |
| School Performance Score | DOE SPS |
| Subgroup identity (encoded) | LEAP |
| School type (encoded) | SPS |
| Minority status flag | Derived |
| Disadvantaged status flag | Derived |
| Poverty rate | Census ACS |
| Poverty x Minority interaction | Derived (captures compounding effect) |

---

## References

**Required (from RFP):**
1. Pham & Nguyen (2025). Predicting student academic performance using hybrid ML. *Computers & Education*, 215.
2. Albreiki et al. (2024). Academic achievement prediction through interpretable modeling. *Scientific Reports*, 14.
3. Mills & Wolf (2017). Vouchers in the bayou: Louisiana Scholarship Program effects. *Educational Evaluation and Policy Analysis*, 39(3).

**Additional:**
4. Baker (2019). Challenges for educational data mining. *J. Educational Data Mining*, 11(1).
5. Romero & Ventura (2020). Educational data mining and learning analytics. *WIREs DMKD*, 10(3).
6. Chen & Guestrin (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD*.
7. Lundberg & Lee (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.
8. Kemper et al. (2020). Predicting student dropout: A ML approach. *European J. Higher Education*, 10(1).
9. Han, Kamber & Pei (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.
10. Fayyad et al. (1996). From data mining to knowledge discovery in databases. *AI Magazine*, 17(3).

---

## License

This project is for academic purposes only as part of CSC 580 at Louisiana Tech University. All data is publicly available from Louisiana DOE and U.S. Census Bureau.
