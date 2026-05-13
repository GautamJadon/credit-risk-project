"""
ML Service: loads artifacts, builds feature vectors, runs predictions.
"""
import numpy as np
import joblib
import json
from backend.app.models.schemas import LoanApplicationInput

LABEL_MAP = {
    'gender':         {'M': 0, 'F': 1},
    'education_type': {'Higher education':0,'Secondary':1,'Incomplete higher':2,'Lower secondary':3},
    'income_type':    {'Working':0,'Commercial associate':1,'Pensioner':2,'State servant':3},
    'family_status':  {'Married':0,'Single':1,'Civil marriage':2,'Separated':3,'Widow':4},
    'housing_type':   {'House / apartment':0,'Rented apartment':1,'With parents':2,
                       'Municipal apartment':3,'Co-op apartment':4},
}

def build_feature_vector(data: LoanApplicationInput, feature_names: list) -> np.ndarray:
    """Convert Pydantic input into a numpy array matching training features."""
    income       = float(data.income)
    loan_amount  = float(data.loan_amount)
    annuity      = float(data.annuity)
    ext1         = float(data.ext_source_1 or 0.5)
    ext2         = float(data.ext_source_2 or 0.5)
    ext3         = float(data.ext_source_3 or 0.5)
    dti_ratio    = annuity / (income + 1e-9)
    lti_ratio    = loan_amount / (income + 1e-9)
    ext_mean     = (ext1 + ext2 + ext3) / 3

    raw = {
        'age':             float(data.age),
        'income':          income,
        'loan_amount':     loan_amount,
        'annuity':         annuity,
        'employment_yrs':  float(data.employment_yrs or 0),
        'ext_source_1':    ext1,
        'ext_source_2':    ext2,
        'ext_source_3':    ext3,
        'family_members':  float(data.family_members or 2),
        'dti_ratio':       dti_ratio,
        'lti_ratio':       lti_ratio,
        'ext_source_mean': ext_mean,
        'gender':          float(LABEL_MAP['gender'].get(data.gender, 0)),
        'education_type':  float(LABEL_MAP['education_type'].get(data.education_type, 0)),
        'income_type':     float(LABEL_MAP['income_type'].get(data.income_type, 0)),
        'family_status':   float(LABEL_MAP['family_status'].get(data.family_status, 0)),
        'housing_type':    float(LABEL_MAP['housing_type'].get(data.housing_type, 0)),
    }
    return np.array([[raw[f] for f in feature_names]])


def categorize_risk(probability: float) -> dict:
    if probability < 0.30:
        return {
            'category': 'LOW RISK',
            'color': '#16a34a',
            'recommendation': 'Approve with standard interest rate.',
            'description': 'Strong repayment profile. High creditworthiness indicated by external scores and income metrics.',
        }
    elif probability < 0.60:
        return {
            'category': 'MEDIUM RISK',
            'color': '#d97706',
            'recommendation': 'Approve with enhanced monitoring or slightly higher interest rate.',
            'description': 'Moderate default risk. Recommend verifying income documents and employment status.',
        }
    else:
        return {
            'category': 'HIGH RISK',
            'color': '#dc2626',
            'recommendation': 'Decline loan or require collateral / guarantor.',
            'description': 'High probability of default based on DTI ratio, external credit scores, and income profile.',
        }


FEATURE_LABELS = {
    'ext_source_mean':  'Average Credit Score',
    'ext_source_1':     'Credit Bureau Score 1',
    'ext_source_2':     'Credit Bureau Score 2',
    'ext_source_3':     'Credit Bureau Score 3',
    'dti_ratio':        'Debt-to-Income Ratio',
    'lti_ratio':        'Loan-to-Income Ratio',
    'income':           'Annual Income',
    'loan_amount':      'Loan Amount',
    'annuity':          'Annual Repayment',
    'age':              'Applicant Age',
    'employment_yrs':   'Years Employed',
    'family_members':   'Family Members',
    'gender':           'Gender',
    'education_type':   'Education Level',
    'income_type':      'Income Type',
    'family_status':    'Family Status',
    'housing_type':     'Housing Type',
}
