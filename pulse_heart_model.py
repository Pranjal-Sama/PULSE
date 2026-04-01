# ============================================================
#  PULSE — Heart Disease ML Pipeline (Production Grade)
#  Dataset : Heart_disease_cleveland_new.csv
#  Author  : PULSE Team | Bharati Vidyapeeth
# ============================================================

# ============================================================
# 🟢 STEP 1 — SETUP & IMPORTS
# ============================================================
import pandas as pd
import numpy as np
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.inspection import permutation_importance

# Optional heavy-hitters — graceful fallback if not installed
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
print("   PULSE — Heart Disease Prediction Pipeline")
print("=" * 65)


# ============================================================
# 🟢 STEP 2 — LOAD DATASET
# ============================================================
df = pd.read_csv('Heart_disease_cleveland_new.csv')

print(f"\n📂 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n── First 5 rows ──")
print(df.head())
print("\n── Column info ──")
print(df.info())
print("\n── Statistical summary ──")
print(df.describe().round(2))


# ============================================================
# 🟢 STEP 3 — DATA CLEANING
# ============================================================
print("\n" + "=" * 65)
print("   STEP 3 — Data Cleaning")
print("=" * 65)

# Replace any stray '?' strings with NaN
df.replace('?', np.nan, inplace=True)

# Convert all columns to numeric (some may have been read as object due to '?')
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Report missing values
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "✅ No missing values found.")

# Smart imputation: median for skewed, mean for symmetric
# Check skewness for each numeric column with nulls
for col in df.columns:
    if df[col].isnull().sum() > 0:
        skew = df[col].skew()
        if abs(skew) > 1:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  Imputed '{col}' with MEDIAN (skew={skew:.2f})")
        else:
            df[col].fillna(df[col].mean(), inplace=True)
            print(f"  Imputed '{col}' with MEAN (skew={skew:.2f})")

# Enforce correct dtypes
# Categorical integer columns (should not be float)
int_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target', 'age']
for col in int_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Sanity check: known valid ranges
range_checks = {
    'age': (1, 120),
    'trestbps': (50, 250),
    'chol': (50, 700),
    'thalach': (40, 250),
    'oldpeak': (0, 10),
}
print("\n── Range outlier check ──")
for col, (lo, hi) in range_checks.items():
    if col in df.columns:
        out = df[(df[col] < lo) | (df[col] > hi)]
        if len(out) > 0:
            print(f"  ⚠️  '{col}': {len(out)} values outside [{lo}, {hi}]")
        else:
            print(f"  ✅ '{col}': all values in valid range")

print(f"\nClean dataset shape: {df.shape}")


# ============================================================
# 🟢 STEP 4 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 65)
print("   STEP 4 — EDA")
print("=" * 65)

# ── 4a. Target distribution ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("PULSE — Target Distribution Analysis", fontsize=14, fontweight='bold')

target_counts = df['target'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].pie(target_counts, labels=['No Disease (0)', 'Disease (1)'],
            autopct='%1.1f%%', colors=colors, startangle=90,
            explode=(0.05, 0.05), shadow=True)
axes[0].set_title("Class Distribution (Pie)")

sns.countplot(x='target', data=df, palette={'0': '#2ecc71', '1': '#e74c3c'}, ax=axes[1])
axes[1].set_title("Class Count")
axes[1].set_xlabel("Target (0=Healthy, 1=Disease)")
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 9),
                     textcoords='offset points', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/01_target_distribution.png")

imbalance_ratio = target_counts.min() / target_counts.max()
print(f"\nClass balance ratio: {imbalance_ratio:.2f} "
      f"({'Balanced ✅' if imbalance_ratio > 0.75 else 'Imbalanced ⚠️'})")

# ── 4b. Correlation heatmap ──────────────────────────────────
plt.figure(figsize=(14, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, square=True,
            annot_kws={'size': 9})
plt.title("PULSE — Feature Correlation Heatmap (Lower Triangle)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/02_correlation_heatmap.png")

# Print top correlations with target
target_corr = corr['target'].drop('target').abs().sort_values(ascending=False)
print("\n── Feature correlation with target (absolute) ──")
print(target_corr.round(3).to_string())

# ── 4c. Feature distributions by target ─────────────────────
cont_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(1, len(cont_features), figsize=(18, 5))
fig.suptitle("PULSE — Continuous Features by Target Class", fontsize=13, fontweight='bold')
for i, feat in enumerate(cont_features):
    for label, color in [(0, '#2ecc71'), (1, '#e74c3c')]:
        axes[i].hist(df[df['target'] == label][feat], bins=20, alpha=0.6,
                     color=color, label=f'{"No Disease" if label == 0 else "Disease"}',
                     edgecolor='white')
    axes[i].set_title(feat.upper())
    axes[i].legend(fontsize=7)
    axes[i].set_xlabel(feat)
plt.tight_layout()
plt.savefig('plots/03_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/03_feature_distributions.png")

# ── 4d. Categorical feature analysis ────────────────────────
cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle("PULSE — Categorical Features vs Target", fontsize=13, fontweight='bold')
axes = axes.flatten()
for i, feat in enumerate(cat_features):
    ct = pd.crosstab(df[feat], df['target'])
    ct.plot(kind='bar', ax=axes[i], color=['#2ecc71', '#e74c3c'],
            edgecolor='white', legend=True)
    axes[i].set_title(feat.upper())
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=0)
    axes[i].legend(['No Disease', 'Disease'], fontsize=7)
plt.tight_layout()
plt.savefig('plots/04_categorical_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/04_categorical_features.png")

# ── 4e. Box plots for outlier detection ─────────────────────
fig, axes = plt.subplots(1, len(cont_features), figsize=(18, 5))
fig.suptitle("PULSE — Box Plots (Outlier Detection)", fontsize=13, fontweight='bold')
for i, feat in enumerate(cont_features):
    df.boxplot(column=feat, by='target', ax=axes[i], patch_artist=True)
    axes[i].set_title(feat.upper())
    axes[i].set_xlabel("Target")
plt.suptitle('')
plt.tight_layout()
plt.savefig('plots/05_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/05_boxplots.png")


# ============================================================
# 🟢 STEP 5 — FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 65)
print("   STEP 5 — Feature Engineering")
print("=" * 65)

# The Cleveland dataset is pre-encoded (no raw categoricals to one-hot).
# However we can engineer meaningful derived features:

# 1. Age group buckets — nonlinear age effect
df['age_group'] = pd.cut(df['age'],
                          bins=[0, 40, 50, 60, 70, 100],
                          labels=[0, 1, 2, 3, 4]).astype(int)

# 2. High-risk BP flag
df['high_bp'] = (df['trestbps'] > 130).astype(int)

# 3. High cholesterol flag
df['high_chol'] = (df['chol'] > 240).astype(int)

# 4. Low max heart rate (poor cardiovascular fitness)
df['low_thalach'] = (df['thalach'] < 120).astype(int)

# 5. ST depression severity bucket
df['oldpeak_cat'] = pd.cut(df['oldpeak'],
                            bins=[-0.1, 0, 1, 2, 10],
                            labels=[0, 1, 2, 3]).astype(int)

print("✅ Engineered 5 new features:")
print("   age_group | high_bp | high_chol | low_thalach | oldpeak_cat")
print(f"   New dataset shape: {df.shape}")


# ============================================================
# 🟢 STEP 6 — FEATURE SELECTION
# ============================================================
print("\n" + "=" * 65)
print("   STEP 6 — Feature Selection")
print("=" * 65)

# Define features and target
FEATURE_COLS = [c for c in df.columns if c != 'target']
X_raw = df[FEATURE_COLS]
y = df['target']

# ── Quick RF-based feature importance to rank all features ──
rf_selector = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_selector.fit(X_raw, y)
importances = pd.Series(rf_selector.feature_importances_, index=FEATURE_COLS)
importances_sorted = importances.sort_values(ascending=False)

print("\n── Random Forest Feature Importances ──")
print(importances_sorted.round(4).to_string())

# Keep features above a low importance threshold OR in top 13
threshold = 0.01
selected_features = importances_sorted[importances_sorted >= threshold].index.tolist()
print(f"\n✅ Selected {len(selected_features)} features (importance ≥ {threshold}):")
print(selected_features)

# Plot feature importances
plt.figure(figsize=(10, 6))
colors_imp = ['#e74c3c' if i < 5 else '#3498db' for i in range(len(importances_sorted))]
importances_sorted.plot(kind='barh', color=colors_imp[::-1])
plt.xlabel("Importance Score")
plt.title("PULSE — Feature Importances (Random Forest)", fontsize=13, fontweight='bold')
plt.axvline(threshold, color='orange', linestyle='--', label=f'Threshold = {threshold}')
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/06_feature_importance.png")

X = df[selected_features]


# ============================================================
# 🟢 STEP 7 — TRAIN / TEST SPLIT
# ============================================================
print("\n" + "=" * 65)
print("   STEP 7 — Train / Test Split")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set : {X_train.shape[0]} samples")
print(f"Test set     : {X_test.shape[0]} samples")
print(f"Train class balance: {dict(y_train.value_counts().sort_index())}")
print(f"Test class balance : {dict(y_test.value_counts().sort_index())}")


# ============================================================
# 🟢 STEP 8 — HANDLE CLASS IMBALANCE
# ============================================================
print("\n" + "=" * 65)
print("   STEP 8 — Handle Class Imbalance")
print("=" * 65)

if imbalance_ratio < 0.75 and SMOTE_AVAILABLE:
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"✅ SMOTE applied. New training size: {X_train_res.shape[0]}")
    print(f"   Balanced classes: {dict(pd.Series(y_train_res).value_counts().sort_index())}")
else:
    X_train_res, y_train_res = X_train.copy(), y_train.copy()
    if imbalance_ratio >= 0.75:
        print("✅ Classes are sufficiently balanced — SMOTE not applied.")
    else:
        print("⚠️  SMOTE skipped (imbalanced-learn not installed).")
        print("   Using class_weight='balanced' in models instead.")


# ============================================================
# 🟢 STEP 9 — FEATURE SCALING
# ============================================================
print("\n" + "=" * 65)
print("   STEP 9 — Feature Scaling (StandardScaler)")
print("=" * 65)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

print("✅ Scaling applied: zero mean, unit variance.")
print(f"   Scaler fitted on {X_train_scaled.shape[0]} training samples.")


# ============================================================
# 🟢 STEP 10 — MODEL TRAINING WITH HYPERPARAMETER TUNING
# ============================================================
print("\n" + "=" * 65)
print("   STEP 10 — Model Training + Hyperparameter Tuning")
print("=" * 65)

# Use class_weight='balanced' as fallback if SMOTE wasn't applied
cw = None if SMOTE_AVAILABLE else 'balanced'

# ── Hyperparameter grids ────────────────────────────────────
lr_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'max_iter': [1000]
}

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_configs = [
    ("Logistic Regression",
     LogisticRegression(class_weight=cw, random_state=42),
     lr_params),

    ("Random Forest",
     RandomForestClassifier(class_weight=cw, random_state=42, n_jobs=-1),
     rf_params),

    ("Gradient Boosting",
     GradientBoostingClassifier(random_state=42),
     gb_params),
]

if XGBOOST_AVAILABLE:
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.5],
    }
    scale_pos = (y_train_res == 0).sum() / (y_train_res == 1).sum()
    model_configs.append((
        "XGBoost",
        XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=1 if SMOTE_AVAILABLE else scale_pos,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        xgb_params
    ))

tuned_models = {}
print(f"\nRunning GridSearchCV (5-fold CV) — optimizing for F1 Score\n")

for name, base_model, param_grid in model_configs:
    print(f"  🔍 Tuning: {name}...", end='', flush=True)
    gs = GridSearchCV(
        base_model, param_grid,
        scoring='f1',          # F1 balances precision & recall
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    gs.fit(X_train_scaled, y_train_res)
    tuned_models[name] = gs.best_estimator_
    print(f"  done. Best CV F1: {gs.best_score_:.4f} | Params: {gs.best_params_}")


# ============================================================
# 🟢 STEP 11 — MODEL EVALUATION
# ============================================================
print("\n" + "=" * 65)
print("   STEP 11 — Full Model Evaluation on Test Set")
print("=" * 65)

results = {}
for name, model in tuned_models.items():
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_proba),
    }

# Print classification reports
for name, res in results.items():
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {res['accuracy']:.4f}")
    print(f"  Precision : {res['precision']:.4f}")
    print(f"  Recall    : {res['recall']:.4f}  ← KEY (minimize missed diseases)")
    print(f"  F1 Score  : {res['f1']:.4f}")
    print(f"  ROC-AUC   : {res['roc_auc']:.4f}")
    print(f"\n  Full Classification Report:")
    print(classification_report(y_test, res['y_pred'],
                                 target_names=['No Disease', 'Disease']))


# ============================================================
# 🟢 STEP 12 — MODEL COMPARISON + VISUALIZATIONS
# ============================================================
print("\n" + "=" * 65)
print("   STEP 12 — Model Comparison")
print("=" * 65)

# ── Summary table ───────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: {
        'Accuracy':  r['accuracy'],
        'Precision': r['precision'],
        'Recall':    r['recall'],
        'F1 Score':  r['f1'],
        'ROC-AUC':   r['roc_auc'],
    }
    for name, r in results.items()
}).T.round(4)

print("\n── Model Comparison Table ──")
print(metrics_df.to_string())

# ── Bar chart comparison ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']].plot(
    kind='bar', ax=ax, width=0.7, edgecolor='white',
    colormap='Set2'
)
ax.set_title("PULSE — Model Performance Comparison", fontsize=13, fontweight='bold')
ax.set_ylabel("Score")
ax.set_ylim(0.5, 1.05)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
ax.legend(loc='lower right', fontsize=9)
ax.axhline(0.85, color='red', linestyle='--', alpha=0.4, label='85% benchmark')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/07_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/07_model_comparison.png")

# ── Confusion matrices (all models) ─────────────────────────
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
if n_models == 1:
    axes = [axes]
fig.suptitle("PULSE — Confusion Matrices", fontsize=13, fontweight='bold')
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                cbar=False, linewidths=1)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    # Annotate TN/FP/FN/TP
    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, -0.2, f"TN={tn} | FP={fp} | FN={fn} | TP={tp}",
            transform=ax.transAxes, ha='center', fontsize=9, color='darkslategray')
plt.tight_layout()
plt.savefig('plots/08_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/08_confusion_matrices.png")

# ── ROC curves (all models) ──────────────────────────────────
plt.figure(figsize=(9, 7))
colors_roc = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    plt.plot(fpr, tpr, color=colors_roc[i], lw=2,
             label=f"{name} (AUC = {res['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
plt.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate (Recall)", fontsize=12)
plt.title("PULSE — ROC Curves", fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/09_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/09_roc_curves.png")

# ── Cross-validation scores (5-fold) ────────────────────────
print("\n── 5-Fold Cross-Validation Scores (F1) on full training data ──")
cv_results = {}
for name, model in tuned_models.items():
    scores = cross_val_score(model, X_train_scaled, y_train_res,
                              cv=cv, scoring='f1', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot(
    [cv_results[n] for n in cv_results],
    labels=list(cv_results.keys()),
    patch_artist=True,
    boxprops=dict(facecolor='#3498db', alpha=0.6),
    medianprops=dict(color='red', linewidth=2),
)
ax.set_title("PULSE — 5-Fold CV F1 Score Distribution", fontsize=13, fontweight='bold')
ax.set_ylabel("F1 Score")
ax.set_ylim(0.5, 1.05)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/10_cv_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/10_cv_scores.png")


# ============================================================
# 🟢 STEP 13 — SELECT BEST MODEL & SAVE
# ============================================================
print("\n" + "=" * 65)
print("   STEP 13 — Select Best Model & Save")
print("=" * 65)

# Selection criterion:
# We pick the model with the HIGHEST F1 Score (which inherently
# balances Recall and Precision). If two models tie, we prefer
# the one with higher Recall (fewer missed sick patients).
best_name = max(results, key=lambda n: (results[n]['f1'], results[n]['recall']))
best_result = results[best_name]
best_model  = best_result['model']

print(f"\n🏆 Selected best model: {best_name}")
print(f"   Accuracy  : {best_result['accuracy']:.4f}")
print(f"   Precision : {best_result['precision']:.4f}")
print(f"   Recall    : {best_result['recall']:.4f}")
print(f"   F1 Score  : {best_result['f1']:.4f}")
print(f"   ROC-AUC   : {best_result['roc_auc']:.4f}")

# Save model, scaler, feature list, and model name
pickle.dump(best_model,          open('models/heart_model.pkl',       'wb'))
pickle.dump(scaler,              open('models/heart_scaler.pkl',      'wb'))
pickle.dump(selected_features,   open('models/heart_features.pkl',    'wb'))
pickle.dump(best_name,           open('models/heart_model_name.pkl',  'wb'))

print("\n✅ Saved to models/:")
print("   heart_model.pkl | heart_scaler.pkl | heart_features.pkl | heart_model_name.pkl")


# ============================================================
# 🟢 STEP 14 — PREDICTION FUNCTION
# ============================================================

def pulse_predict_heart(input_dict: dict) -> dict:
    """
    Predict heart disease risk from a dictionary of feature values.

    Parameters
    ----------
    input_dict : dict
        Keys must match the feature names in 'selected_features'.
        Example:
          {
            'age': 54, 'sex': 1, 'cp': 2, 'trestbps': 130,
            'chol': 250, 'fbs': 0, 'restecg': 2, 'thalach': 145,
            'exang': 0, 'oldpeak': 1.5, 'slope': 2, 'ca': 0,
            'thal': 2, 'age_group': 2, 'high_bp': 0,
            'high_chol': 1, 'low_thalach': 0, 'oldpeak_cat': 1
          }

    Returns
    -------
    dict with keys: risk_pct, risk_label, advice_summary, advice_detail
    """
    # Build input array in the correct feature order
    input_array = np.array([[input_dict.get(f, 0) for f in selected_features]])
    input_scaled = scaler.transform(input_array)

    prob     = best_model.predict_proba(input_scaled)[0][1]
    risk_pct = prob * 100

    return _build_recommendation(risk_pct)


def _build_recommendation(risk_pct: float) -> dict:
    """Build a rich, detailed recommendation based on risk percentage."""

    # ── LOW RISK ≤ 15% ──────────────────────────────────────
    if risk_pct <= 15:
        label = "LOW RISK 🟢"
        summary = (
            "Your heart health indicators look good. "
            "Focus on maintaining your current healthy lifestyle."
        )
        detail = {
            "Diet": [
                "Follow a heart-healthy diet: plenty of fruits, vegetables, and whole grains.",
                "Include omega-3 rich foods: salmon, walnuts, flaxseeds (2–3 times/week).",
                "Limit processed foods, trans fats, and excess sodium (< 2,300 mg/day).",
                "Aim for 8–10 glasses of water daily.",
            ],
            "Exercise": [
                "Maintain 150 minutes of moderate aerobic activity per week.",
                "Surya Namaskar (Sun Salutation) — 12 rounds daily, morning.",
                "Pranpranayama (deep breathing): Anulom-Vilom 10 min/day — improves heart rate variability.",
                "Include strength training 2×/week to maintain muscle mass.",
            ],
            "Lifestyle": [
                "Keep BMI within 18.5–24.9.",
                "Maintain blood pressure below 120/80 mmHg.",
                "Avoid smoking; limit alcohol to ≤ 1 drink/day.",
                "Sleep 7–9 hours nightly; poor sleep raises cardiovascular risk by 20%.",
                "Annual health check-up including lipid profile and ECG.",
            ],
            "Natural Supplements (consult doctor first)": [
                "Arjuna bark (Terminalia arjuna) — traditional cardiac tonic.",
                "Garlic extract (600 mg/day) — helps maintain healthy cholesterol.",
                "Coenzyme Q10 (100 mg/day) — supports mitochondrial energy in heart cells.",
            ],
        }

    # ── MEDIUM RISK 15–40% ───────────────────────────────────
    elif risk_pct <= 40:
        label = "MEDIUM RISK 🟡"
        summary = (
            "Your risk is moderate. Lifestyle modifications NOW can significantly "
            "reduce your chances of developing heart disease."
        )
        detail = {
            "Diet (Strict)": [
                "DASH Diet or Mediterranean Diet — clinically proven to reduce CVD risk.",
                "Reduce sodium to < 1,500 mg/day if blood pressure is elevated.",
                "Eliminate trans fats completely; replace saturated fat with olive oil.",
                "Increase soluble fiber: oats, beans, lentils (aim for 25–30 g/day).",
                "Reduce refined sugar and simple carbohydrates — they elevate triglycerides.",
                "Add turmeric (curcumin) and ginger to daily cooking — anti-inflammatory.",
            ],
            "Exercise (Structured)": [
                "Cardiac rehab-style routine: 30 min brisk walking × 5 days/week.",
                "Yoga sequence: Bhujangasana, Pawanmuktasana, Setu Bandhasana — 20 min/day.",
                "Avoid sudden intense exercise; build up gradually.",
                "Monitor heart rate during exercise — stay at 50–70% of max HR.",
                "Swimming and cycling — excellent low-impact cardio options.",
            ],
            "Monitoring": [
                "Check blood pressure daily (home BP monitor recommended).",
                "Fasting lipid profile every 6 months.",
                "Blood glucose every 3–6 months (diabetes strongly linked to CVD).",
                "Visit a cardiologist for a baseline stress test / ECG.",
            ],
            "Stress Management": [
                "Chronic stress raises cortisol → raises BP and inflammation.",
                "Meditation: 10–20 min daily (apps: Headspace, Calm).",
                "Limit work hours; take regular breaks.",
            ],
            "Natural Supplements (consult doctor)": [
                "Ashwagandha (300–600 mg/day) — reduces cortisol and blood pressure.",
                "Omega-3 fish oil (2–4 g/day EPA+DHA) — lowers triglycerides significantly.",
                "Magnesium glycinate (300–400 mg) — supports cardiac muscle function.",
                "Hawthorn extract — traditional cardiac tonic, improves coronary blood flow.",
                "Red yeast rice — natural statin-like effect (consult physician first).",
            ],
        }

    # ── HIGH RISK > 40% ──────────────────────────────────────
    else:
        label = "HIGH RISK 🔴"
        summary = (
            "⚠️ URGENT: Your physiological indicators suggest HIGH risk of heart disease. "
            "Please consult a cardiologist IMMEDIATELY. Do NOT delay."
        )
        detail = {
            "IMMEDIATE ACTIONS": [
                "Book an appointment with a cardiologist within the next 7 days.",
                "Request: ECG, 2D Echocardiogram, Stress Test, Full Lipid Panel, hsCRP.",
                "If you experience chest pain, breathlessness, or palpitations — call emergency services.",
                "Do NOT start new strenuous exercise without medical clearance.",
            ],
            "Medical Management": [
                "Discuss with your doctor: statins (if high LDL), antihypertensives (if high BP),",
                "  beta-blockers or ACE inhibitors depending on your specific profile.",
                "Ask about aspirin therapy — only under physician guidance.",
                "Monitor: BP twice daily, weight daily, symptoms diary.",
            ],
            "Diet (Therapeutic)": [
                "Strict low-sodium diet: < 1,200 mg/day.",
                "Eliminate: fried foods, processed meats, full-fat dairy, alcohol.",
                "Eat: oily fish, legumes, leafy greens, berries, nuts in moderation.",
                "Small, frequent meals — avoid large meals that stress the heart.",
                "Maintain a food diary and share with your dietitian.",
            ],
            "Gentle Movement Only": [
                "Only after medical clearance: gentle 15–20 min flat walks daily.",
                "NO heavy lifting, NO high-intensity workouts until cleared.",
                "Restorative Yoga only: Shavasana, gentle breathing exercises.",
            ],
            "Emotional & Mental Health": [
                "High-risk diagnosis is stressful — consider counselling.",
                "Involve family/support network in your care plan.",
                "Depression and anxiety worsen cardiac outcomes — address them proactively.",
            ],
            "Natural Supplements (only ALONGSIDE medical treatment)": [
                "Omega-3 (under doctor supervision — may interact with blood thinners).",
                "Coenzyme Q10 (especially if on statin — statins deplete CoQ10).",
                "NEVER stop prescribed medication in favor of supplements.",
            ],
        }

    return {
        'risk_pct':      round(risk_pct, 2),
        'risk_label':    label,
        'advice_summary': summary,
        'advice_detail':  detail,
    }


# ============================================================
# 🟢 STEP 15 — TEST THE PREDICTION FUNCTION
# ============================================================
print("\n" + "=" * 65)
print("   STEP 15 — Prediction Function Demo")
print("=" * 65)

# Build feature dict for a test row
# (Derived features auto-calculated from raw values)
def make_input(age, sex, cp, trestbps, chol, fbs, restecg,
               thalach, exang, oldpeak, slope, ca, thal):
    """Helper to auto-compute engineered features."""
    age_group   = int(pd.cut([age], bins=[0,40,50,60,70,100], labels=[0,1,2,3,4])[0])
    high_bp     = int(trestbps > 130)
    high_chol   = int(chol > 240)
    low_thalach = int(thalach < 120)
    oldpeak_cat = int(pd.cut([oldpeak], bins=[-0.1,0,1,2,10], labels=[0,1,2,3])[0])

    return {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca,
        'thal': thal, 'age_group': age_group, 'high_bp': high_bp,
        'high_chol': high_chol, 'low_thalach': low_thalach,
        'oldpeak_cat': oldpeak_cat,
    }

# Test Case 1: Low-risk profile (young, healthy parameters)
test_low = make_input(
    age=42, sex=0, cp=2, trestbps=118, chol=200, fbs=0,
    restecg=0, thalach=165, exang=0, oldpeak=0.2, slope=2, ca=0, thal=2
)

# Test Case 2: Medium-risk profile
test_med = make_input(
    age=55, sex=1, cp=1, trestbps=135, chol=260, fbs=1,
    restecg=2, thalach=140, exang=0, oldpeak=1.2, slope=1, ca=1, thal=2
)

# Test Case 3: High-risk profile
test_high = make_input(
    age=67, sex=1, cp=3, trestbps=160, chol=310, fbs=1,
    restecg=2, thalach=108, exang=1, oldpeak=2.6, slope=1, ca=3, thal=3
)

for label, test_case in [("Low-risk", test_low),
                           ("Medium-risk", test_med),
                           ("High-risk", test_high)]:
    result = pulse_predict_heart(test_case)
    print(f"\n{'─'*55}")
    print(f"  Test: {label}")
    print(f"  Risk: {result['risk_pct']:.1f}%  →  {result['risk_label']}")
    print(f"  Summary: {result['advice_summary'][:80]}...")
    print(f"  Advice categories: {list(result['advice_detail'].keys())}")


# ── Also test on first actual test row from dataset ──────────
sample_dict = dict(zip(selected_features, X_test.iloc[0].tolist()))
sample_result = pulse_predict_heart(sample_dict)
print(f"\n{'─'*55}")
print(f"  Test: Real sample from test set (actual target = {y_test.iloc[0]})")
print(f"  Predicted Risk: {sample_result['risk_pct']:.1f}%  →  {sample_result['risk_label']}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("   ✅ PULSE Pipeline Complete!")
print("=" * 65)
print(f"\n  Best Model    : {best_name}")
print(f"  Test Accuracy : {best_result['accuracy']:.4f}")
print(f"  Test Recall   : {best_result['recall']:.4f}")
print(f"  Test F1 Score : {best_result['f1']:.4f}")
print(f"  ROC-AUC       : {best_result['roc_auc']:.4f}")
print(f"\n  Saved files:")
print(f"    models/heart_model.pkl")
print(f"    models/heart_scaler.pkl")
print(f"    models/heart_features.pkl")
print(f"    models/heart_model_name.pkl")
print(f"\n  Plots saved in: plots/")
print(f"    01_target_distribution.png")
print(f"    02_correlation_heatmap.png")
print(f"    03_feature_distributions.png")
print(f"    04_categorical_features.png")
print(f"    05_boxplots.png")
print(f"    06_feature_importance.png")
print(f"    07_model_comparison.png")
print(f"    08_confusion_matrices.png")
print(f"    09_roc_curves.png")
print(f"    10_cv_scores.png")
print("\n" + "=" * 65)
print("  ⚠️  DISCLAIMER: PULSE is a screening tool only.")
print("       It does NOT replace professional medical diagnosis.")
print("=" * 65)
