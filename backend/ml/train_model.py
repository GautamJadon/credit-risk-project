"""
Credit Risk ML Training Script
Generates synthetic loan data, trains 4 models, saves best model + artifacts.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, json, warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
N = 15000

print("🔧 Generating synthetic loan dataset...")

age             = np.random.randint(21, 70, N)
income          = np.random.lognormal(11.2, 0.6, N).clip(50000, 5000000)
loan_amount     = np.random.lognormal(12.5, 0.7, N).clip(50000, 5000000)
annuity         = loan_amount / np.random.uniform(18, 60, N)
employment_yrs  = np.random.exponential(5, N).clip(0, 40)
ext1            = np.random.beta(4, 2, N)
ext2            = np.random.beta(3, 2, N)
ext3            = np.random.beta(4, 3, N)
family_members  = np.random.choice([1,2,3,4,5,6], N, p=[0.15,0.35,0.3,0.12,0.06,0.02])

gender         = np.random.choice(['M','F'], N, p=[0.57,0.43])
education      = np.random.choice([
    'Higher education','Secondary','Incomplete higher','Lower secondary'],
    N, p=[0.35,0.45,0.15,0.05])
income_type    = np.random.choice([
    'Working','Commercial associate','Pensioner','State servant'],
    N, p=[0.50,0.23,0.18,0.09])
family_status  = np.random.choice([
    'Married','Single','Civil marriage','Separated','Widow'],
    N, p=[0.64,0.18,0.10,0.05,0.03])
housing_type   = np.random.choice([
    'House / apartment','Rented apartment','With parents','Municipal apartment','Co-op apartment'],
    N, p=[0.72,0.11,0.09,0.05,0.03])

# Realistic default probability
dti        = annuity / (income / 12)
lti        = loan_amount / income
ext_mean   = (ext1 + ext2 + ext3) / 3

# log_odds = (
#     -3.5
#     + 0.025  * (age - 40)
#     - 0.003  * (income / 10000)
#     + 0.001  * (loan_amount / 10000)
#     + 1.8    * dti.clip(0, 3)
#     + 0.9    * lti.clip(0, 5)
#     - 3.5    * ext_mean
#     - 0.08   * employment_yrs
#     + 0.12   * (family_members > 4).astype(float)
#     + np.where(education == 'Lower secondary', 0.5, 0)
#     + np.where(income_type == 'Pensioner', 0.3, 0)
#     + np.random.normal(0, 0.4, N)
# )

log_odds = (
    -7.5
    + 0.015  * (age - 40)
    - 0.005  * (income / 10000)
    + 0.0005 * (loan_amount / 10000)
    + 1.8    * dti.clip(0, 3)
    + 0.9    * lti.clip(0, 5)
    - 4.0    * ext_mean
    - 0.06   * employment_yrs
    + 0.10   * (family_members > 4).astype(float)
    + np.where(education == 'Lower secondary', 0.3, 0)
    + np.where(income_type == 'Pensioner', 0.2, 0)
    + np.random.normal(0, 0.3, N)
)
prob_default = 1 / (1 + np.exp(-log_odds))
target = (np.random.uniform(0, 1, N) < prob_default).astype(int)

print(f"   Default rate: {target.mean()*100:.1f}%  (target: ~8–12%)")

df = pd.DataFrame({
    'age': age,
    'income': income,
    'loan_amount': loan_amount,
    'annuity': annuity,
    'employment_yrs': employment_yrs,
    'ext_source_1': ext1,
    'ext_source_2': ext2,
    'ext_source_3': ext3,
    'family_members': family_members.astype(float),
    'dti_ratio': dti,
    'lti_ratio': lti,
    'ext_source_mean': ext_mean,
    'gender': gender,
    'education_type': education,
    'income_type': income_type,
    'family_status': family_status,
    'housing_type': housing_type,
    'TARGET': target
})

# Encode categoricals
CATEGORICAL = ['gender','education_type','income_type','family_status','housing_type']
encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    
FEATURES = [c for c in df.columns if c != 'TARGET']

X = df[FEATURES].values
y = df['TARGET'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

neg, pos = (y_train==0).sum(), (y_train==1).sum()
scale_pos = neg / pos
print(f"   Class ratio: {scale_pos:.1f}:1  |  Train: {len(X_train)}  Test: {len(X_test)}")

print("\n📊 Training models...")
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED),
    'Decision Tree':       DecisionTreeClassifier(class_weight='balanced', max_depth=8, random_state=SEED),
    'Random Forest':       RandomForestClassifier(class_weight='balanced', n_estimators=200, max_depth=10, n_jobs=-1, random_state=SEED),
    'XGBoost':             XGBClassifier(scale_pos_weight=scale_pos, n_estimators=300, max_depth=6,
                               learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                               eval_metric='auc', random_state=SEED, n_jobs=-1, verbosity=0),
}

results = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    model.fit(X_train_s, y_train)
    y_prob = model.predict_proba(X_test_s)[:,1]
    y_pred = model.predict(X_test_s)
    auc    = roc_auc_score(y_test, y_prob)
    rep    = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {'model':model,'auc':auc,'cv_mean':cv.mean(),'cv_std':cv.std(),'y_prob':y_prob,'y_pred':y_pred,'report':rep}
    print(f"   {name:25s}  CV AUC={cv.mean():.4f}  Test AUC={auc:.4f}")

best_name = max(results, key=lambda n: results[n]['auc'])
best      = results[best_name]
print(f"\n✅ Best model: {best_name}  (AUC={best['auc']:.4f})")

os.makedirs('saved_models', exist_ok=True)
joblib.dump(best['model'],  'saved_models/best_model.pkl')
joblib.dump(scaler,          'saved_models/scaler.pkl')
joblib.dump(encoders,        'saved_models/label_encoders.pkl')
joblib.dump(FEATURES,        'saved_models/feature_names.pkl')

# Save model metadata
meta = {
    'model_type':    type(best['model']).__name__,
    'best_model_name': best_name,
    'auc':           round(best['auc'],4),
    'cv_auc':        round(best['cv_mean'],4),
    'feature_count': len(FEATURES),
    'features':      FEATURES,
    'comparison': {
        n: {'auc': round(r['auc'],4), 'cv_auc': round(r['cv_mean'],4)}
        for n, r in results.items()
    }
}
with open('saved_models/model_meta.json','w') as f:
    json.dump(meta, f, indent=2)
print("   Artifacts saved: best_model.pkl, scaler.pkl, label_encoders.pkl, feature_names.pkl")

# ── Plot ROC curves ────────────────────────────────────────────
os.makedirs('static', exist_ok=True)
plt.figure(figsize=(8,6))
colors = ['steelblue','darkorange','forestgreen','crimson']
for (name, r), c in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    plt.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={r['auc']:.4f})")
plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curves — All Models')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/roc_curves.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ROC curve saved → static/roc_curves.png")

# ── Plot Feature Importance ────────────────────────────────────
if hasattr(best['model'], 'feature_importances_'):
    imp = best['model'].feature_importances_
    idx = np.argsort(imp)[-15:]
    plt.figure(figsize=(8,6))
    plt.barh([FEATURES[i] for i in idx], imp[idx], color='steelblue')
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Feature Importances — {best_name}')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("   Feature importance saved → static/feature_importance.png")

# Code for confusion matrix and classification report
cm = confusion_matrix(y_test, best['y_pred'])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.close()
print("   Confusion matrix saved → static/confusion_matrix.png")

print("\n🎉 Training complete!")
