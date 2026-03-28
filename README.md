# 🫀 Cardiovascular Disease Prediction

A machine learning project to predict the presence of cardiovascular disease using clinical patient data. The pipeline includes exploratory data analysis (EDA), preprocessing, SMOTE-based oversampling, model training with Logistic Regression and XGBoost, and explainability via SHAP values.

---

## 📁 Project Structure

```
├── Cardiovascular_Disease_Dataset.csv   # Raw dataset (1000 patients, 14 features)
├── preprocessed_data.csv                # Cleaned & preprocessed dataset
├── EDA.ipynb                            # Exploratory Data Analysis notebook
└── ML_pipeline.ipynb                    # Model training & evaluation notebook
```

---

## 📊 Dataset

The dataset contains **1,000 patient records** with the following features:

| Feature | Description |
|---|---|
| `patientid` | Unique patient identifier |
| `age` | Age of the patient |
| `gender` | Gender (1 = Male, 0 = Female) |
| `chestpain` | Chest pain type (0–3) |
| `restingBP` | Resting blood pressure (mm Hg) |
| `serumcholestrol` | Serum cholesterol (mg/dl) |
| `fastingbloodsugar` | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| `restingrelectro` | Resting electrocardiogram results (0–2) |
| `maxheartrate` | Maximum heart rate achieved |
| `exerciseangia` | Exercise-induced angina (1 = Yes, 0 = No) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `noofmajorvessels` | Number of major vessels colored by fluoroscopy (0–3) |
| `target` | **Target variable**: 1 = Disease present, 0 = No disease |

---

## 🔬 Exploratory Data Analysis (`EDA.ipynb`)

Key preprocessing steps performed:

- **Renamed columns** for readability
- **Dropped** `patientid` (non-informative identifier)
- **Imputed missing values**: Zeroes in `Serum_cholestrol` replaced with `NaN` and filled with the column median
- **Outlier detection & capping** using the IQR method for: `Resting_BP`, `Serum_cholestrol`, `Max_Heartrate`, `OldPeak`
- Exported cleaned data as `preprocessed_data.csv`

---

## 🤖 ML Pipeline (`ML_pipeline.ipynb`)

### Models Trained

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 89% | 0.9458 |
| **XGBoost** | **95%** | **0.9930** |

### Pipeline Steps

1. **Feature Selection** — Dropped `Slope` (derived from diagnosis; not available in early prediction scenarios)
2. **Stratified Train/Test Split** — 80/20 split preserving class distribution
3. **Feature Scaling** — `StandardScaler` applied to continuous features for Logistic Regression
4. **SMOTE** — Synthetic Minority Oversampling applied to training data to handle class imbalance
5. **Model Training** — Logistic Regression and XGBoost classifiers
6. **Evaluation** — Classification report, ROC-AUC score, confusion matrix
7. **Cross-Validation** — 5-fold CV on XGBoost: mean ROC-AUC ≈ **0.9904**

### XGBoost — Classification Report

```
              precision    recall  f1-score   support

           0       0.95      0.93      0.94        84
           1       0.95      0.97      0.96       116

    accuracy                           0.95       200
```

### Logistic Regression — Classification Report

```
              precision    recall  f1-score   support

           0       0.86      0.88      0.87        84
           1       0.91      0.90      0.90       116

    accuracy                           0.89       200
```

---

## 🧠 Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to interpret both models:

- **Logistic Regression**: `shap.LinearExplainer`
- **XGBoost**: `shap.TreeExplainer`

Top features driving predictions (from correlation and SHAP analysis):

1. `Slope`
2. `Chestpain`
3. `Number_of_Major_Vessels`
4. `Resting_BP`
5. `Resting_Electrocardiogram`

---

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run EDA
jupyter notebook EDA.ipynb

# 4. Run ML pipeline
jupyter notebook ML_pipeline.ipynb
```

---

## 📌 Notes

- `Slope` was excluded from the XGBoost model as it is strongly correlated with the diagnosis process and may not be available for early prediction.
- SMOTE was applied **only on training data** to prevent data leakage.
- StandardScaler was fit on training data and applied to test data only.

---
