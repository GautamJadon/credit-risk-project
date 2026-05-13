"""
Full ML training pipeline:
  - Preprocessing (encode, scale, split)
  - Train 4 models (LR, DT, RF, XGBoost)
  - Hyperparameter tuning
  - Save best model + artifacts
"""
import numpy as np
import pandas as pd
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve, f1_score)
from xgboost import XGBClassifier

SAVE_DIR = '/home/claude/credit-risk/backend/ml/saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('/home/claude/credit-risk/backend/data/processed', exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────
df = pd.read_csv('/home/claude/credit-risk/backend/data/raw/loan_data.csv')
print(f"Loaded: {df.shape}  |  Default rate: {df['TARGET'].mean():.3f}")

# ── 2. Encode Categoricals ────────────────────────────────────────
CAT_COLS = ['gender', 'education_type', 'income_type',
            'family_status', 'housing_type']

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, f'{SAVE_DIR}/label_encoders.pkl')
print("Label encoders saved.")

# ── 3. Features & Target ──────────────────────────────────────────
FEATURE_COLS = [
    'age', 'gender', 'education_type', 'income_type', 'family_status',
    'housing_type', 'family_members', 'annual_income', 'credit_amount',
    'annuity_amount', 'goods_price', 'ext_source_1', 'ext_source_2',
    'ext_source_3', 'employment_years', 'dti_ratio', 'lti_ratio',
    'ext_source_mean'
]

X = df[FEATURE_COLS].values
y = df['TARGET'].values
joblib.dump(FEATURE_COLS, f'{SAVE_DIR}/feature_names.pkl')

# ── 4. Train/Test Split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── 5. Scale ──────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, f'{SAVE_DIR}/scaler.pkl')
print("Scaler saved.")

# ── 6. Class weight ───────────────────────────────────────────────
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pw  = neg / pos
print(f"Class ratio neg/pos: {scale_pw:.2f}")

# ── 7. Define Models ──────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, C=0.5, random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        class_weight='balanced', max_depth=8, min_samples_leaf=20, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        class_weight='balanced', n_estimators=150, max_depth=12,
        min_samples_leaf=10, n_jobs=-1, random_state=42
    ),
    'XGBoost': XGBClassifier(
        scale_pos_weight=scale_pw, n_estimators=200, max_depth=5,
        learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, eval_metric='auc',
        random_state=42, n_jobs=-1, verbosity=0
    )
}

# ── 8. Train & Evaluate ───────────────────────────────────────────
results = {}
print("\n" + "="*55)
for name, model in models.items():
    cv = cross_val_score(model, X_train_s, y_train, cv=5,
                         scoring='roc_auc', n_jobs=-1)
    model.fit(X_train_s, y_train)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)
    auc    = roc_auc_score(y_test, y_prob)
    f1     = f1_score(y_test, y_pred, pos_label=1)
    rep    = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        'model': model, 'auc': auc, 'f1': f1,
        'cv_mean': cv.mean(), 'cv_std': cv.std(),
        'y_prob': y_prob, 'y_pred': y_pred, 'report': rep
    }
    print(f"{name:<25}  CV AUC: {cv.mean():.4f}±{cv.std():.4f}  Test AUC: {auc:.4f}  F1: {f1:.4f}")

# ── 9. Select best model ──────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['auc'])
best      = results[best_name]
print(f"\n★ Best model: {best_name}  (AUC {best['auc']:.4f})")

# ── 10. Save best model ───────────────────────────────────────────
joblib.dump(best['model'], f'{SAVE_DIR}/best_model.pkl')
joblib.dump({'name': best_name, 'auc': best['auc'], 'f1': best['f1']},
            f'{SAVE_DIR}/model_meta.pkl')
print("Best model saved.")

# ── 11. Save comparison results ───────────────────────────────────
comparison = {
    name: {
        'cv_auc': round(r['cv_mean'], 4),
        'test_auc': round(r['auc'], 4),
        'f1_default': round(r['f1'], 4),
        'precision': round(r['report']['1']['precision'], 4),
        'recall': round(r['report']['1']['recall'], 4)
    }
    for name, r in results.items()
}
with open(f'{SAVE_DIR}/model_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print("\n📊 Final Comparison:")
for name, m in comparison.items():
    star = " ★" if name == best_name else ""
    print(f"  {name:<25}{star}  AUC={m['test_auc']}  F1={m['f1_default']}")

print("\n✅ All artifacts saved to:", SAVE_DIR)
print("   - best_model.pkl")
print("   - scaler.pkl")
print("   - label_encoders.pkl")
print("   - feature_names.pkl")
print("   - model_meta.pkl")
print("   - model_comparison.json")
