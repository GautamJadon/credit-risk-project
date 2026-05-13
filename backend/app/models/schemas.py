from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

class LoanApplicationInput(BaseModel):
    age: int = Field(..., ge=18, le=75, description="Age in years")
    gender: str = Field(..., description="M or F")
    education_type: str = Field(..., description="Education level")
    income_type: str = Field(..., description="Employment type")
    family_status: str = Field(..., description="Marital status")
    housing_type: str = Field(..., description="Housing situation")
    income: float = Field(..., gt=0, description="Annual income")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    annuity: float = Field(..., gt=0, description="Annual repayment amount")
    employment_yrs: Optional[float] = Field(0.0, ge=0, description="Years employed")
    ext_source_1: Optional[float] = Field(0.5, ge=0, le=1)
    ext_source_2: Optional[float] = Field(0.5, ge=0, le=1)
    ext_source_3: Optional[float] = Field(0.5, ge=0, le=1)
    family_members: Optional[int] = Field(2, ge=1, le=20)

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in ('M', 'F'):
            raise ValueError("gender must be M or F")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35, "gender": "M",
                "education_type": "Higher education",
                "income_type": "Working",
                "family_status": "Married",
                "housing_type": "House / apartment",
                "income": 600000, "loan_amount": 1200000, "annuity": 55000,
                "employment_yrs": 6, "ext_source_1": 0.72,
                "ext_source_2": 0.68, "ext_source_3": 0.75,
                "family_members": 3
            }
        }


class RiskFactor(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str
    label: str


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_percentage: float
    risk_category: str
    risk_color: str
    recommendation: str
    description: str
    top_factors: List[RiskFactor]
    model_name: str
    model_auc: float
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    model_name: str
    auc: float
    cv_auc: float
    feature_count: int
    features: List[str]
    comparison: Dict[str, Any]
    version: str
