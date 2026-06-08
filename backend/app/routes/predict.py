# from fastapi import APIRouter, Request, HTTPException
# from backend.app.models.schemas import LoanApplicationInput, PredictionResponse, RiskFactor
# from backend.app.services.ml_service import build_feature_vector, categorize_risk
# from backend.app.services.shap_service import SHAPService

# # from app.models.schemas import LoanApplicationInput, PredictionResponse, RiskFactor
# # from app.services.ml_service import build_feature_vector, categorize_risk
# # from app.services.shap_service import SHAPService

# import json

# router = APIRouter()

# @router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
# async def predict(data: LoanApplicationInput, request: Request):
#     """
#     Predict loan default probability with SHAP explanation.
#     Returns risk score (0–1), category, recommendation, and top risk factors.
#     """
#     try:
#         state = request.app.state
#         X     = build_feature_vector(data, state.feature_names)
#         X_s   = state.scaler.transform(X)

#         prob  = float(state.model.predict_proba(X_s)[0][1])
#         risk  = categorize_risk(prob)

#         shap_svc    = SHAPService(state.model, state.feature_names)
#         factors_raw = shap_svc.explain(X_s)
#         factors     = [RiskFactor(**f) for f in factors_raw]

#         return PredictionResponse(
#             risk_probability = round(prob, 4),
#             risk_percentage  = round(prob * 100, 1),
#             risk_category    = risk['category'],
#             risk_color       = risk['color'],
#             recommendation   = risk['recommendation'],
#             description      = risk['description'],
#             top_factors      = factors,
#             model_name       = state.meta.get('best_model_name','XGBoost'),
#             model_auc        = state.meta.get('auc', 0.94),
#             version          = "1.0.0",
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# @router.get("/model-info", tags=["System"])
# async def model_info(request: Request):
#     """Return metadata about the loaded ML model."""
#     from backend.app.models.schemas import ModelInfoResponse
#     m = request.app.state.meta
#     return ModelInfoResponse(
#         model_type    = m.get('model_type',''),
#         model_name    = m.get('best_model_name',''),
#         auc           = m.get('auc', 0),
#         cv_auc        = m.get('cv_auc', 0),
#         feature_count = m.get('feature_count', 0),
#         features      = m.get('features', []),
#         comparison    = m.get('comparison', {}),
#         version       = "1.0.0",
#     )


from fastapi import APIRouter, Request, HTTPException

from backend.app.models.schemas import (
    LoanApplicationInput,
    PredictionResponse,
    RiskFactor,
)

from backend.app.services.ml_service import (
    build_feature_vector,
    categorize_risk,
)

from backend.app.services.shap_service import SHAPService


router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
async def predict(
    data: LoanApplicationInput,
    request: Request
):
    """
    Predict loan default probability
    and generate SHAP explanations.
    """

    try:

        state = request.app.state

        # ------------------------------
        # Build raw feature vector
        # ------------------------------

        X_raw = build_feature_vector(
            data,
            state.feature_names
        )

        # ------------------------------
        # Scale features
        # ------------------------------

        X_scaled = state.scaler.transform(
            X_raw
        )

        # ------------------------------
        # Predict probability
        # ------------------------------

        prob = float(
            state.model
            .predict_proba(X_scaled)[0][1]
        )

        risk = categorize_risk(prob)

        # ------------------------------
        # SHAP explanation
        # ------------------------------

        shap_service = SHAPService(
            state.model,
            state.feature_names
        )

        factors_raw = shap_service.explain(
            X_scaled=X_scaled,
            X_raw=X_raw
        )

        factors = [
            RiskFactor(**factor)
            for factor in factors_raw
        ]

        # ------------------------------
        # Response
        # ------------------------------

        return PredictionResponse(
            risk_probability=round(prob, 4),
            risk_percentage=round(prob * 100, 1),
            risk_category=risk["category"],
            risk_color=risk["color"],
            recommendation=risk["recommendation"],
            description=risk["description"],
            top_factors=factors,
            model_name=state.meta.get(
                "best_model_name",
                "Random Forest"
            ),
            model_auc=state.meta.get(
                "auc",
                0.94
            ),
            version="1.0.0",
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@router.get(
    "/model-info",
    tags=["System"]
)
async def model_info(
    request: Request
):
    """
    Return model metadata.
    """

    from backend.app.models.schemas import (
        ModelInfoResponse
    )

    meta = request.app.state.meta

    return ModelInfoResponse(
        model_type=meta.get(
            "model_type",
            ""
        ),
        model_name=meta.get(
            "best_model_name",
            ""
        ),
        auc=meta.get(
            "auc",
            0
        ),
        cv_auc=meta.get(
            "cv_auc",
            0
        ),
        feature_count=meta.get(
            "feature_count",
            0
        ),
        features=meta.get(
            "features",
            []
        ),
        comparison=meta.get(
            "comparison",
            {}
        ),
        version="1.0.0",
    )