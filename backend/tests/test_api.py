"""API tests — use context manager so lifespan/model-loading runs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from app.main import app

VALID_PAYLOAD = {
    "age": 35, "gender": "M",
    "education_type": "Higher education",
    "income_type": "Working",
    "family_status": "Married",
    "housing_type": "House / apartment",
    "income": 600000, "loan_amount": 1200000, "annuity": 55000,
    "employment_yrs": 6,
    "ext_source_1": 0.72, "ext_source_2": 0.68, "ext_source_3": 0.75,
    "family_members": 3,
}

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_valid(client):
    r = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    d = r.json()
    assert 0 <= d["risk_probability"] <= 1
    assert d["risk_category"] in ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]
    assert len(d["top_factors"]) > 0
    assert d["model_auc"] > 0.7

def test_predict_invalid_age(client):
    r = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "age": 200})
    assert r.status_code == 422

def test_predict_invalid_gender(client):
    r = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "gender": "X"})
    assert r.status_code == 422

def test_model_info(client):
    r = client.get("/api/v1/model-info")
    assert r.status_code == 200
    d = r.json()
    assert d["feature_count"] == 17
    assert d["auc"] > 0.7

def test_low_risk_profile(client):
    """High credit scores + good income → low risk."""
    payload = {**VALID_PAYLOAD, "ext_source_1":0.95,"ext_source_2":0.92,
               "ext_source_3":0.88,"income":2000000,"loan_amount":500000,"annuity":30000}
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 200
    d = r.json()
    assert d["risk_probability"] < 0.5, f"Expected low risk, got {d['risk_probability']}"

def test_high_risk_profile(client):
    """Low credit scores + high DTI → higher risk."""
    payload = {**VALID_PAYLOAD, "ext_source_1":0.1,"ext_source_2":0.15,"ext_source_3":0.2,
               "income":150000,"loan_amount":2000000,"annuity":120000,"employment_yrs":0}
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 200
    d = r.json()
    assert d["risk_probability"] > 0.2, f"Expected higher risk, got {d['risk_probability']}"
