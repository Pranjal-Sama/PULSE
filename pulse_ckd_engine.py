# ============================================================
#  PULSE — Chronic Kidney Disease (CKD) ML Pipeline
#  Dataset : Chronic_Kidney_Dsease_data.csv
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
print("   PULSE — Chronic Kidney Disease Prediction Pipeline")
print("=" * 65)


# ============================================================
# 🟢 STEP 2 — LOAD DATASET
# ============================================================
df = pd.read_csv('Chronic_Kidney_Dsease_data.csv')

# Drop administrative/non-clinical columns
cols_to_drop = ['PatientID', 'DoctorInCharge']
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# Standardize the target column name to match the PULSE architecture
if 'Diagnosis' in df.columns:
    df.rename(columns={'Diagnosis': 'target'}, inplace=True)

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

# Convert all columns to numeric, coercing any accidental string entries
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Report missing values
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "✅ No missing values found.")

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
fig.suptitle("PULSE — CKD Target Distribution", fontsize=14, fontweight='bold')

target_counts = df['target'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].pie(target_counts, labels=['No CKD (0)', 'CKD (1)'],
            autopct='%1.1f%%', colors=colors, startangle=90,
            explode=(0.05, 0.05), shadow=True)
axes[0].set_title("Class Distribution (Pie)")

sns.countplot(x='target', data=df, palette={'0': '#2ecc71', '1': '#e74c3c'}, ax=axes[1])
axes[1].set_title("Class Count")
plt.tight_layout()
plt.savefig('plots/ckd_01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/ckd_01_target_distribution.png")

imbalance_ratio = target_counts.min() / target_counts.max()
print(f"Class balance ratio: {imbalance_ratio:.2f} "
      f"({'Balanced ✅' if imbalance_ratio > 0.75 else 'Imbalanced ⚠️'})")

# ── 4b. Correlation heatmap ──────────────────────────────────
# Due to many columns, we will plot correlation only for the top 15 correlated features
plt.figure(figsize=(12, 10))
top_corr_cols = df.corr()['target'].abs().sort_values(ascending=False).head(15).index
corr = df[top_corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, square=True)
plt.title("PULSE — Top 15 Feature Correlation Heatmap (CKD)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/ckd_02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/ckd_02_correlation_heatmap.png")


# ============================================================
# 🟢 STEP 5 — FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 65)
print("   STEP 5 — Feature Engineering")
print("=" * 65)

# 1. GFR Stages (Based on standard nephrology guidelines)
# Stage 1: >=90, Stage 2: 60-89, Stage 3: 30-59, Stage 4: 15-29, Stage 5: <15
if 'GFR' in df.columns:
    df['GFR_Stage'] = pd.cut(df['GFR'], bins=[-1, 15, 30, 60, 90, 200], labels=[5, 4, 3, 2, 1]).astype(int)

# 2. Elevated Serum Creatinine (> 1.2 is generally a flag for renal distress)
if 'SerumCreatinine' in df.columns:
    df['High_Creatinine'] = (df['SerumCreatinine'] > 1.2).astype(int)

# 3. Elevated BUN (Blood Urea Nitrogen > 20 is a flag)
if 'BUNLevels' in df.columns:
    df['High_BUN'] = (df['BUNLevels'] > 20).astype(int)

# 4. High Systolic BP
if 'SystolicBP' in df.columns:
    df['High_BP_Sys'] = (df['SystolicBP'] > 130).astype(int)

print("✅ Engineered 4 new renal-specific features:")
print("   GFR_Stage | High_Creatinine | High_BUN | High_BP_Sys")
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

# Quick RF to rank feature importance
rf_selector = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_selector.fit(X_raw, y)
importances = pd.Series(rf_selector.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

# For CKD dataset (which has ~50 cols), we take the top 20 most important
selected_features = importances.head(20).index.tolist()
print(f"\n✅ Selected Top {len(selected_features)} features:")
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
    print("✅ SMOTE skipped (classes balanced or module unavailable).")

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
pickle.dump(best_model,          open('models/ckd_model.pkl', 'wb'))
pickle.dump(scaler,              open('models/ckd_scaler.pkl', 'wb'))
pickle.dump(selected_features,   open('models/ckd_features.pkl', 'wb'))
pickle.dump(best_name,           open('models/ckd_model_name.pkl', 'wb'))
print("\n✅ Saved to models/: ckd_model.pkl | ckd_scaler.pkl | ckd_features.pkl")


# ============================================================
# 🟢 STEP 14 — PREDICTION FUNCTION (WITH CLINICAL ADVICE)
# ============================================================
def pulse_predict_ckd(input_dict: dict) -> dict:
    """Predict CKD risk and return structured advice."""
    input_array = np.array([[input_dict.get(f, 0) for f in selected_features]])
    input_scaled = scaler.transform(input_array)
    
    prob = best_model.predict_proba(input_scaled)[0][1]
    risk_pct = prob * 100

    return _build_ckd_recommendation(risk_pct, input_dict)

def _build_ckd_recommendation(risk_pct: float, inputs: dict) -> dict:
    if risk_pct <= 20:
        label = "LOW RISK 🟢"
        summary = "Your kidney function indicators look normal. Maintain your current healthy lifestyle."
        detail = {
            "Hydration": [
                "Drink adequate water daily (approx 2-3 liters) to help kidneys clear sodium and toxins."
            ],
            "Diet & Lifestyle": [
                "Maintain a balanced diet to keep blood pressure and blood sugar in check.",
                "Limit the use of over-the-counter painkillers like Ibuprofen/NSAIDs, which can stress the kidneys over time.",
                "Exercise regularly to maintain a healthy BMI."
            ],
            "Monitoring": [
                "Continue with annual routine checkups including basic metabolic panels."
            ]
        }
    elif risk_pct <= 50:
        label = "MEDIUM RISK 🟡 (Renal Stress)"
        summary = "You have markers indicating possible early-stage renal stress. Early intervention is key."
        detail = {
            "Medical Actions": [
                "Schedule a follow-up test for Serum Creatinine, BUN, and a urinalysis for protein (proteinuria).",
                "Ensure strict control of your blood pressure and blood sugar (these are the top two causes of kidney damage)."
            ],
            "Medication Check": [
                "Review all your current medications with a doctor. STOP using NSAIDs (like Ibuprofen, Naproxen) immediately."
            ],
            "Dietary Changes": [
                "Adopt a moderately low-sodium diet (< 2,300 mg/day) to reduce fluid retention and BP.",
                "Avoid crash diets or extremely high-protein diets (like Keto) as they force the kidneys to work harder."
            ]
        }
    else:
        label = "HIGH RISK 🔴"
        summary = "⚠️ URGENT: Your physiological indicators strongly suggest compromised kidney function. Consult a Nephrologist immediately."
        detail = {
            "IMMEDIATE ACTIONS": [
                "Book an appointment with a Nephrologist within the next 7 days.",
                "Request a comprehensive Renal Panel, GFR assessment, and Renal Ultrasound."
            ],
            "Strict Renal Diet": [
                "Low Sodium: Avoid all processed foods, canned soups, and fast food.",
                "Low Potassium: Limit bananas, oranges, potatoes, and tomatoes (if advised by doctor).",
                "Low Phosphorus: Limit dairy products, nuts, and dark colas.",
                "Controlled Protein: Work with a renal dietitian to consume the exact right amount of protein."
            ],
            "Fluid Management": [
                "You may need to restrict your fluid intake depending on your doctor's advice (to prevent edema/swelling)."
            ],
            "Symptom Monitoring": [
                "Watch for swelling in the ankles/legs (edema), extreme fatigue, metallic taste in the mouth, or changes in urination frequency."
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
print("   STEP 15 — CKD Prediction Demo")
print("=" * 65)

# We will use average values for features we don't explicitly pass to prevent errors
def make_ckd_input(age, sys_bp, gfr, creatinine, bun, protein, hba1c):
    # Simulate the feature engineering
    gfr_stage = 1
    if gfr < 15: gfr_stage = 5
    elif gfr < 30: gfr_stage = 4
    elif gfr < 60: gfr_stage = 3
    elif gfr < 90: gfr_stage = 2
    
    high_creat = 1 if creatinine > 1.2 else 0
    high_bun = 1 if bun > 20 else 0
    high_bp_sys = 1 if sys_bp > 130 else 0
    
    return {
        'Age': age, 'SystolicBP': sys_bp, 'GFR': gfr, 'SerumCreatinine': creatinine,
        'BUNLevels': bun, 'ProteinInUrine': protein, 'HbA1c': hba1c,
        'GFR_Stage': gfr_stage, 'High_Creatinine': high_creat, 'High_BUN': high_bun, 'High_BP_Sys': high_bp_sys,
        # Mocking other potentially selected features with neutral values
        'DiastolicBP': 80, 'BMI': 24, 'HemoglobinLevels': 14.0, 'SerumElectrolytesSodium': 140, 'FatigueLevels': 0
    }

# High Risk (Low GFR, High Creatinine, Protein in urine)
test_high = make_ckd_input(age=65, sys_bp=160, gfr=25, creatinine=3.5, bun=45, protein=3, hba1c=8.5)

# Low Risk (Healthy indicators)
test_low = make_ckd_input(age=35, sys_bp=115, gfr=110, creatinine=0.8, bun=12, protein=0, hba1c=5.2)

for label, data in [("High Risk Patient", test_high), ("Low Risk Patient", test_low)]:
    res = pulse_predict_ckd(data)
    print(f"\n{label} -> {res['risk_pct']}% | {res['risk_label']}")
    print(f"Summary: {res['advice_summary']}")

print("\n✅ CKD Pipeline Execution Complete.")