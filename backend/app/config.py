from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    MODEL_PATH: str = "saved_models/best_model.pkl"
    SCALER_PATH: str = "saved_models/scaler.pkl"
    FEATURE_NAMES_PATH: str = "saved_models/feature_names.pkl"
    ENCODERS_PATH: str = "saved_models/label_encoders.pkl"
    META_PATH: str = "saved_models/model_meta.json"
    FRONTEND_URL: str = "*"
    APP_NAME: str = "Credit Risk Assessment API"
    VERSION: str = "1.0.0"

    class Config:
        env_file = ".env"

settings = Settings()
