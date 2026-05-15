# 🛡️ CreditGuard — AI Loan Risk Assessment System

> ML-powered personal loan default prediction with SHAP explainability, FastAPI backend, Bootstrap frontend, and AWS deployment-ready architecture.

[![Tests](https://img.shields.io/badge/tests-7%2F7%20passing-success)](./backend/tests)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 📊 Model Performance

| Model | CV AUC | Test AUC |
|-------|--------|----------|
| Logistic Regression | 0.938 | **0.940** ⭐ |
| Random Forest | 0.936 | 0.938 |
| XGBoost | 0.934 | 0.939 |
| Decision Tree | 0.893 | 0.909 |

---

## 🏗️ Architecture

```
User Browser → CloudFront CDN → S3 (Frontend)
                              ↓
                         EC2 + Nginx
                              ↓
                    FastAPI + Uvicorn (port 8000)
                     ├── XGBoost Model (.pkl)
                     ├── SHAP Explainer
                     └── PostgreSQL (optional)
```

---

## ⚡ Quick Start (Local)

### 1. Clone & setup
```bash
git clone https://github.com/yourusername/creditguard
cd creditguard/backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the model
```bash
cd ml
python train_model.py
```

### 3. Start the API
```bash
cd ..
uvicorn backend.app.main:app --reload --port 8000
# → API running at http://localhost:8000
# → Swagger docs at http://localhost:8000/docs
```

### 4. Open the frontend
Open `frontend/index.html` in your browser — or serve it:
```bash
cd ../frontend && python -m http.server 3000
# → http://localhost:3000
```

### 5. Run tests
```bash
cd backend && pytest tests/ -v
```

---

## 🐳 Docker (Full Stack)
```bash
cd infrastructure
docker compose up --build
# → Frontend: http://localhost
# → API:      http://localhost/api/v1
# → Docs:     http://localhost/docs
```

---

## 📁 Project Structure

```
creditguard/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Settings
│   │   ├── routes/
│   │   │   ├── predict.py       # POST /api/v1/predict
│   │   │   └── health.py        # GET  /api/v1/health
│   │   ├── models/schemas.py    # Pydantic I/O schemas
│   │   └── services/
│   │       ├── ml_service.py    # Feature engineering
│   │       └── shap_service.py  # SHAP explanations
│   ├── ml/
│   │   ├── train_model.py       # Training script
│   │   └── saved_models/        # Trained artifacts
│   ├── tests/test_api.py        # pytest suite (7 tests)
│   └── requirements.txt
├── frontend/
│   ├── index.html               # Main app
│   ├── css/style.css            # Dark theme UI
│   └── js/app.js                # API integration
└── infrastructure/
    ├── docker-compose.yml
    └── nginx/nginx.conf
```

---

## 🌐 API Reference

### `POST /api/v1/predict`
```json
{
  "age": 35, "gender": "M",
  "education_type": "Higher education",
  "income_type": "Working",
  "family_status": "Married",
  "housing_type": "House / apartment",
  "income": 600000,
  "loan_amount": 1200000,
  "annuity": 55000,
  "employment_yrs": 6,
  "ext_source_1": 0.72,
  "ext_source_2": 0.68,
  "ext_source_3": 0.75,
  "family_members": 3
}
```
**Response:**
```json
{
  "risk_probability": 0.0167,
  "risk_percentage": 1.7,
  "risk_category": "LOW RISK",
  "risk_color": "#16a34a",
  "recommendation": "Approve with standard interest rate.",
  "top_factors": [...],
  "model_name": "Logistic Regression",
  "model_auc": 0.9396
}
```

### `GET /api/v1/health` — Service health check
### `GET /api/v1/model-info` — Model metadata + comparison

---

## ☁️ AWS Deployment

See `docs/aws_deployment.md` for full step-by-step guide covering:
- EC2 t2.micro setup + SSH
- Nginx reverse proxy + SSL (Let's Encrypt)
- S3 static website hosting for frontend
- CloudFront CDN distribution
- GitHub Actions CI/CD

**Estimated cost:** ~$0/month on AWS Free Tier (first 12 months)

---

## 🎓 Academic Context

**Project Title:** Credit Risk Assessment and Default Prediction for Personal Loans Using a Python-Based Web Application: An Empirical Study in Retail Credit Markets

**Key Contributions:**
1. Empirical comparison of 4 ML algorithms on credit default prediction
2. SHAP-based per-prediction explainability for regulatory compliance
3. End-to-end production deployment on AWS cloud infrastructure
4. Real-time risk categorisation (Low / Medium / High) with lending recommendations

---

## 👤 Author
Gautam Jadon — Summer Internship Project, Delhi School of Business, 2025-27

**Stack:** Python 3.12 · FastAPI · scikit-learn · XGBoost · SHAP · Bootstrap 5 · Docker · Nginx · AWS EC2 · S3 · CloudFront
