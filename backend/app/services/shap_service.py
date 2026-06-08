"""
SHAP Service: generates per-prediction feature explanations.
"""
import shap
import numpy as np
from backend.app.services.ml_service import FEATURE_LABELS


class SHAPService:
    def __init__(self, model, feature_names: list):
        self.model        = model
        self.feature_names = feature_names
        # TreeExplainer for XGBoost / RF / DT; LinearExplainer for LR
        model_type = type(model).__name__
        if model_type in ('LogisticRegression',):
            self.explainer = shap.LinearExplainer(model, np.zeros((1, len(feature_names))))
            self._tree = False
        else:
            self.explainer = shap.TreeExplainer(model)
            self._tree = True

    def explain(self, X_scaled: np.ndarray) -> list:
        """Return sorted list of feature contributions for one sample."""
        if self._tree:
            sv = self.explainer.shap_values(X_scaled)
            # XGBoost returns 2D; RF may return list[2D]
            if isinstance(sv, list):
                shap_vals = sv[1][0] if len(sv) == 2 else sv[0][0]
            else:
                shap_vals = sv[0]
        else:
            sv = self.explainer.shap_values(X_scaled)
            shap_vals = sv[0] if hasattr(sv,'__len__') and len(sv.shape)>1 else sv

        contributions = []
        for fname, fval, sval in zip(self.feature_names, X_scaled[0], shap_vals):
            contributions.append({
                'feature':   fname,
                'label':     FEATURE_LABELS.get(fname, fname.replace('_',' ').title()),
                'value':     round(float(fval), 4),
                'shap_value': round(float(sval), 4),
                'direction': 'increases_risk' if sval > 0 else 'decreases_risk',
            })

        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        return contributions[:12]
# """
# SHAP Service: generates per-prediction feature explanations.

# Compatible with:
# - XGBoost
# - Random Forest
# - Decision Tree
# - Logistic Regression
# - SHAP >= 0.50
# """

# import shap
# import numpy as np

# from backend.app.services.ml_service import FEATURE_LABELS


# class SHAPService:
#     def __init__(self, model, feature_names: list):
#         self.model = model
#         self.feature_names = feature_names

#         model_type = type(model).__name__

#         try:
#             if model_type == "LogisticRegression":
#                 self.explainer = shap.LinearExplainer(
#                     model,
#                     np.zeros((1, len(feature_names)))
#                 )
#                 self._tree = False

#             else:
#                 self.explainer = shap.TreeExplainer(model)
#                 self._tree = True

#         except Exception as e:
#             print(f"SHAP initialization failed: {e}")
#             self.explainer = None

#     def explain(self, X_scaled: np.ndarray) -> list:
#         """
#         Generate SHAP explanation for a single prediction.
#         Returns top feature contributions.
#         """

#         if self.explainer is None:
#             return []

#         try:

#             # ------------------------------------------------------
#             # Generate SHAP values
#             # ------------------------------------------------------

#             explanation = self.explainer(X_scaled)

#             if hasattr(explanation, "values"):
#                 sv = explanation.values
#             else:
#                 sv = explanation

#             sv = np.asarray(sv)

#             # ------------------------------------------------------
#             # Handle different SHAP output formats
#             # ------------------------------------------------------

#             if sv.ndim == 3:
#                 # Binary classification
#                 # Shape: (samples, features, classes)

#                 if sv.shape[2] == 2:
#                     shap_vals = sv[0, :, 1]
#                 else:
#                     shap_vals = sv[0, :, 0]

#             elif sv.ndim == 2:
#                 # Shape: (samples, features)
#                 shap_vals = sv[0]

#             elif sv.ndim == 1:
#                 shap_vals = sv

#             else:
#                 raise ValueError(
#                     f"Unexpected SHAP output shape: {sv.shape}"
#                 )

#             # ------------------------------------------------------
#             # Build response
#             # ------------------------------------------------------

#             contributions = []

#             for fname, fval, sval in zip(
#                 self.feature_names,
#                 X_scaled[0],
#                 shap_vals
#             ):

#                 contributions.append({
#                     "feature": fname,
#                     "label": FEATURE_LABELS.get(
#                         fname,
#                         fname.replace("_", " ").title()
#                     ),
#                     "value": round(float(fval), 4),
#                     "shap_value": round(float(sval), 4),
#                     "direction": (
#                         "increases_risk"
#                         if float(sval) > 0
#                         else "decreases_risk"
#                     )
#                 })

#             contributions.sort(
#                 key=lambda x: abs(x["shap_value"]),
#                 reverse=True
#             )

#             return contributions[:12]

#         except Exception as e:

#             print(
#                 f"SHAP explanation failed: {repr(e)}"
#             )

#             # ------------------------------------------------------
#             # Fallback
#             # ------------------------------------------------------

#             fallback = []

#             for fname, fval in zip(
#                 self.feature_names,
#                 X_scaled[0]
#             ):
#                 fallback.append({
#                     "feature": fname,
#                     "label": FEATURE_LABELS.get(
#                         fname,
#                         fname.replace("_", " ").title()
#                     ),
#                     "value": round(float(fval), 4),
#                     "shap_value": 0.0,
#                     "direction": "decreases_risk"
#                 })

#             return fallback[:12]