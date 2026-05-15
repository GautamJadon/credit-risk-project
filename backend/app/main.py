"""
Credit Risk Assessment API
FastAPI application with ML model serving and SHAP explainability.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path
import joblib, json, os

# from app.config import settings
# from app.routes import predict, health

# from backend.app.config import settings
# from backend.app.routes import predict, health
# from backend.app.models.schemas import LoanApplicationInput

from app.config import settings
from app.routes import predict, health
from app.models.schemas import LoanApplicationInput

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup, release at shutdown."""
    base = Path(__file__).parent.parent
    app.state.model         = joblib.load(base / settings.MODEL_PATH)
    app.state.scaler        = joblib.load(base / settings.SCALER_PATH)
    app.state.feature_names = joblib.load(base / settings.FEATURE_NAMES_PATH)
    app.state.encoders      = joblib.load(base / settings.ENCODERS_PATH)
    with open(base / settings.META_PATH) as f:
        app.state.meta = json.load(f)
    print(f"✅ Loaded: {app.state.meta['best_model_name']}  AUC={app.state.meta['auc']}")
    yield
    print("👋 Shutdown")

app = FastAPI(
    title=settings.APP_NAME,
    description="ML-powered personal loan default prediction with SHAP explainability. "
                "Built with XGBoost, FastAPI, and deployed on AWS.",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# # Serve static assets (ROC curves, plots)
# static_dir = Path(__file__).parent.parent / "ml" / "static"
# static_dir.mkdir(exist_ok=True)
# app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(health.router,  prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")

@app.get("/", tags=["System"])
def root():
    return {
        "message": "Credit Risk Assessment API",
        "docs":    "/docs",
        "health":  "/api/v1/health",
        "predict": "/api/v1/predict  [POST]",
    }
