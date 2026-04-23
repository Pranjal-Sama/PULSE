# ============================================================
#  PULSE — Diabetes ML Pipeline (Production Grade)
#  Dataset : diabetes.csv (Pima Indians)
#  Author  : PULSE Team | Bharati Vidyapeeth
# ============================================================

# ============================================================
# 🟢 STEP 1 — SETUP & IMPORTS
# ============================================================
import pandas as pd
import numpy as np

# --- NON-INTERACTIVE BACKEND TO PREVENT THREAD CRASHES ---
import matplotlib
matplotlib.use('Agg') 
# ---------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Run: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  imbalanced-learn not installed. Run: pip install imbalanced-learn")

# ── Output directories ──────────────────────────────────────
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 65)
print("   PULSE — Diabetes Prediction Pipeline")
print("=" * 65)


# ============================================================
# 🟢 STEP 2 — LOAD DATASET
# ============================================================
df = pd.read_csv('diabetes.csv')

# Standardize the target column name to match the PULSE architecture
df.rename(columns={'Outcome': 'target'}, inplace=True)

print(f"\n📂 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n── First 5 rows ──")
print(df.head())
print("\n── Statistical summary ──")
print(df.describe().round(2))


# ============================================================
# 🟢 STEP 3 — DATA CLEANING
# ============================================================
print("\n" + "=" * 65)
print("   STEP 3 — Data Cleaning")
print("=" * 65)

# In the Pima dataset, 0 means missing for these specific clinical features
zero_is_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("\nReplacing biologically impossible '0' values with NaN...")
for col in zero_is_missing_cols:
    zero_count = (df[col] == 0).sum()
    if zero_count > 0:
        print(f"  Found {zero_count} missing (0) values in '{col}'")
        df[col].replace(0, np.nan, inplace=True)

# Smart imputation: median for skewed, mean for symmetric
for col in df.columns:
    if df[col].isnull().sum() > 0:
        skew = df[col].skew()
        if abs(skew) > 1:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  ✅ Imputed '{col}' with MEDIAN (skew={skew:.2f})")
        else:
            df[col].fillna(df[col].mean(), inplace=True)
            print(f"  ✅ Imputed '{col}' with MEAN (skew={skew:.2f})")

print(f"\nClean dataset shape: {df.shape}")


# ============================================================
# 🟢 STEP 4 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 65)
print("   STEP 4 — EDA")
print("=" * 65)

# ── 4a. Target distribution ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("PULSE — Diabetes Target Distribution", fontsize=14, fontweight='bold')

target_counts = df['target'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].pie(target_counts, labels=['No Diabetes (0)', 'Diabetes (1)'],
            autopct='%1.1f%%', colors=colors, startangle=90,
            explode=(0.05, 0.05), shadow=True)
axes[0].set_title("Class Distribution (Pie)")

sns.countplot(x='target', data=df, palette={'0': '#2ecc71', '1': '#e74c3c'}, ax=axes[1])
axes[1].set_title("Class Count")
plt.tight_layout()
plt.savefig('plots/diabetes_01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/diabetes_01_target_distribution.png")

imbalance_ratio = target_counts.min() / target_counts.max()
print(f"Class balance ratio: {imbalance_ratio:.2f} "
      f"({'Balanced ✅' if imbalance_ratio > 0.75 else 'Imbalanced ⚠️'})")

# ── 4b. Correlation heatmap ──────────────────────────────────
plt.figure(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, square=True)
plt.title("PULSE — Diabetes Correlation Heatmap", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/diabetes_02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/diabetes_02_correlation_heatmap.png")


# ============================================================
# 🟢 STEP 5 — FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 65)
print("   STEP 5 — Feature Engineering")
print("=" * 65)

# 1. Age group buckets
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)

# 2. BMI Class (0=Under/Normal, 1=Overweight, 2=Obese, 3=Extreme Obese)
df['BMI_Class'] = pd.cut(df['BMI'], bins=[0, 25, 30, 35, 100], labels=[0, 1, 2, 3]).astype(int)

# 3. High Glucose Flag (> 140 is prediabetic/diabetic post-meal)
df['High_Glucose'] = (df['Glucose'] > 140).astype(int)

# 4. High Blood Pressure Flag (> 80 Diastolic)
df['High_BP'] = (df['BloodPressure'] > 80).astype(int)

print("✅ Engineered 4 new features:")
print("   Age_Group | BMI_Class | High_Glucose | High_BP")
print(f"   New dataset shape: {df.shape}")


# ============================================================
# 🟢 STEP 6 — FEATURE SELECTION
# ============================================================
print("\n" + "=" * 65)
print("   STEP 6 — Feature Selection")
print("=" * 65)

FEATURE_COLS = [c for c in df.columns if c != 'target']
X_raw = df[FEATURE_COLS]
y = df['target']

rf_selector = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_selector.fit(X_raw, y)
importances = pd.Series(rf_selector.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

threshold = 0.01
selected_features = importances[importances >= threshold].index.tolist()
print(f"\n✅ Selected {len(selected_features)} features (importance ≥ {threshold}):")
print(selected_features)

X = df[selected_features]


# ============================================================
# 🟢 STEP 7, 8, 9 — SPLIT, BALANCE, SCALE
# ============================================================
print("\n" + "=" * 65)
print("   STEP 7,8,9 — Split, Balance, Scale")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if imbalance_ratio < 0.75 and SMOTE_AVAILABLE:
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("✅ SMOTE applied for class imbalance.")
else:
    X_train_res, y_train_res = X_train.copy(), y_train.copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)
print("✅ Standard Scaling applied.")


# ============================================================
# 🟢 STEP 10 — MODEL TRAINING & TUNING
# ============================================================
print("\n" + "=" * 65)
print("   STEP 10 — Model Training + Hyperparameter Tuning")
print("=" * 65)

cw = None if SMOTE_AVAILABLE else 'balanced'
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_configs = [
    ("Logistic Regression", LogisticRegression(class_weight=cw, random_state=42), 
     {'C': [0.01, 0.1, 1, 10]}),
    ("Random Forest", RandomForestClassifier(class_weight=cw, random_state=42, n_jobs=-1),
     {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=42),
     {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4]})
]

if XGBOOST_AVAILABLE:
    scale_pos = (y_train_res == 0).sum() / (y_train_res == 1).sum()
    model_configs.append((
        "XGBoost", XGBClassifier(eval_metric='logloss', scale_pos_weight=1 if SMOTE_AVAILABLE else scale_pos, random_state=42, verbosity=0),
        {'n_estimators': [100, 200], 'max_depth': [3, 4, 5], 'learning_rate': [0.05, 0.1]}
    ))

tuned_models = {}
for name, base_model, param_grid in model_configs:
    print(f"  🔍 Tuning: {name}...", end='', flush=True)
    gs = GridSearchCV(base_model, param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_train_scaled, y_train_res)
    tuned_models[name] = gs.best_estimator_
    print(f" done. Best F1: {gs.best_score_:.4f}")


# ============================================================
# 🟢 STEP 11 & 12 — EVALUATION
# ============================================================
print("\n" + "=" * 65)
print("   STEP 11 & 12 — Model Evaluation")
print("=" * 65)

results = {}
for name, model in tuned_models.items():
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }

best_name = max(results, key=lambda n: (results[n]['f1'], results[n]['recall']))
best_result = results[best_name]
best_model  = best_result['model']

print(f"\n🏆 Selected Best Model: {best_name}")
print(f"  Accuracy : {best_result['accuracy']:.4f}")
print(f"  Recall   : {best_result['recall']:.4f}")
print(f"  F1 Score : {best_result['f1']:.4f}")


# ============================================================
# 🟢 STEP 13 — SAVE ASSETS
# ============================================================
pickle.dump(best_model,          open('models/diabetes_model.pkl', 'wb'))
pickle.dump(scaler,              open('models/diabetes_scaler.pkl', 'wb'))
pickle.dump(selected_features,   open('models/diabetes_features.pkl', 'wb'))
pickle.dump(best_name,           open('models/diabetes_model_name.pkl', 'wb'))
print("\n✅ Saved to models/: diabetes_model.pkl | diabetes_scaler.pkl | diabetes_features.pkl")


# ============================================================
# 🟢 STEP 14 — PREDICTION FUNCTION (WITH CLINICAL ADVICE)
# ============================================================
def pulse_predict_diabetes(input_dict: dict) -> dict:
    """Predict diabetes risk and return structured advice."""
    input_array = np.array([[input_dict.get(f, 0) for f in selected_features]])
    input_scaled = scaler.transform(input_array)
    
    prob = best_model.predict_proba(input_scaled)[0][1]
    risk_pct = prob * 100

    return _build_diabetes_recommendation(risk_pct, input_dict)

def _build_diabetes_recommendation(risk_pct: float, inputs: dict) -> dict:
    if risk_pct <= 25:
        label = "LOW RISK 🟢"
        summary = "Your glycemic and metabolic indicators are stable. Maintain your healthy habits."
        detail = {
            "Diet & Nutrition": [
                "Maintain a balanced diet rich in complex carbohydrates and fiber.",
                "Limit added sugars and high-glycemic-index processed foods.",
                "Drink water primarily; avoid sugary sodas and juices."
            ],
            "Lifestyle": [
                "Aim for 150 minutes of moderate exercise per week.",
                "Keep your BMI in the healthy range (18.5 - 24.9).",
                "Ensure 7-8 hours of sleep to maintain healthy metabolic hormones."
            ]
        }
    elif risk_pct <= 50:
        label = "MEDIUM RISK 🟡 (Possible Prediabetes)"
        summary = "You show signs of metabolic stress or insulin resistance. Action taken now can prevent progression to Type 2 Diabetes."
        detail = {
            "Medical Checks": [
                "Schedule a Fasting Blood Glucose and HbA1c test with your doctor.",
                "Check blood pressure and cholesterol (often linked to insulin resistance)."
            ],
            "Dietary Overhaul": [
                "Switch completely to complex carbs (brown rice, quinoa, whole wheat).",
                "Increase fiber intake significantly (beans, legumes, vegetables) to slow glucose absorption.",
                "Incorporate healthy fats and lean proteins to stabilize blood sugar spikes."
            ],
            "Exercise": [
                "Incorporate strength training 2-3 times a week (muscle heavily absorbs glucose).",
                "Take a 10-15 minute walk immediately after large meals to lower post-meal blood sugar."
            ]
        }
    else:
        label = "HIGH RISK 🔴"
        summary = "⚠️ URGENT: Your indicators strongly suggest the presence of Diabetes. Please consult an endocrinologist or physician immediately."
        detail = {
            "IMMEDIATE ACTIONS": [
                "Book an appointment with a doctor for a definitive HbA1c and Oral Glucose Tolerance Test (OGTT).",
                "Do not attempt to self-medicate with insulin or other drugs without a prescription."
            ],
            "Diet (Strict Control)": [
                "Eliminate all simple sugars, sweets, and refined carbohydrates immediately.",
                "Adopt a strict low-glycemic or clinically managed low-carb diet.",
                "Eat meals at consistent times every day to prevent erratic blood sugar swings."
            ],
            "Complication Prevention": [
                "Check your feet daily for cuts or sores (diabetic neuropathy risk).",
                "Schedule a comprehensive eye exam (diabetic retinopathy risk).",
                "Monitor your blood pressure closely."
            ]
        }
        
    return {
        'risk_pct': round(risk_pct, 2),
        'risk_label': label,
        'advice_summary': summary,
        'advice_detail': detail,
    }


# ============================================================
# 🟢 STEP 15 — TEST THE FUNCTION
# ============================================================
print("\n" + "=" * 65)
print("   STEP 15 — Diabetes Prediction Demo")
print("=" * 65)

def make_diabetes_input(pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age):
    age_group = int(pd.cut([age], bins=[0,30,45,60,100], labels=[0,1,2,3])[0])
    bmi_class = int(pd.cut([bmi], bins=[0,25,30,35,100], labels=[0,1,2,3])[0])
    high_gluc = int(glucose > 140)
    high_bp   = int(bp > 80)
    
    return {
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': bp,
        'SkinThickness': skin, 'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': pedigree, 'Age': age,
        'Age_Group': age_group, 'BMI_Class': bmi_class,
        'High_Glucose': high_gluc, 'High_BP': high_bp
    }

# High Risk (Classic diabetic indicators)
test_high = make_diabetes_input(pregnancies=4, glucose=185, bp=85, skin=35, insulin=200, bmi=38.5, pedigree=0.85, age=55)

# Low Risk (Healthy indicators)
test_low = make_diabetes_input(pregnancies=0, glucose=90, bp=70, skin=20, insulin=50, bmi=22.0, pedigree=0.2, age=25)

for label, data in [("High Risk Patient", test_high), ("Low Risk Patient", test_low)]:
    res = pulse_predict_diabetes(data)
    print(f"\n{label} -> {res['risk_pct']}% | {res['risk_label']}")
    print(f"Summary: {res['advice_summary']}")

print("\n✅ Diabetes Pipeline Execution Complete.")