"""
Generate realistic synthetic loan dataset for training.
Mimics Home Credit / LendingClub distributions.
"""
import numpy as np
import pandas as pd
np.random.seed(42)

N = 15000

age = np.random.randint(21, 68, N)
gender = np.random.choice(['M', 'F'], N, p=[0.55, 0.45])
education = np.random.choice(
    ['Higher education', 'Secondary / secondary special',
     'Incomplete higher', 'Lower secondary'],
    N, p=[0.38, 0.42, 0.14, 0.06]
)
income_type = np.random.choice(
    ['Working', 'Commercial associate', 'Pensioner', 'State servant'],
    N, p=[0.52, 0.23, 0.17, 0.08]
)
family_status = np.random.choice(
    ['Married', 'Single / not married', 'Civil marriage', 'Separated'],
    N, p=[0.55, 0.28, 0.10, 0.07]
)
housing_type = np.random.choice(
    ['House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment'],
    N, p=[0.70, 0.12, 0.11, 0.07]
)
family_members = np.random.choice([1, 2, 3, 4, 5, 6], N, p=[0.15, 0.30, 0.28, 0.17, 0.07, 0.03])

annual_income = np.random.lognormal(mean=12.0, sigma=0.6, size=N).clip(30000, 3000000)
credit_amount = np.random.lognormal(mean=13.0, sigma=0.65, size=N).clip(50000, 5000000)
annuity       = credit_amount * np.random.uniform(0.03, 0.09, N)
goods_price   = credit_amount * np.random.uniform(0.80, 1.15, N)

ext1 = np.clip(np.random.beta(5, 2, N), 0.05, 0.99)
ext2 = np.clip(np.random.beta(6, 2, N), 0.05, 0.99)
ext3 = np.clip(np.random.beta(5, 2.5, N), 0.05, 0.99)

employment_years = np.clip(np.random.exponential(5, N), 0, 35)

dti   = annuity / (annual_income + 1)
lti   = credit_amount / (annual_income + 1)
ext_mean = (ext1 + ext2 + ext3) / 3

logit = (
    -3.5
    + 0.02  * (40 - age) / 10
    + 4.0   * dti
    + 1.5   * lti
    - 5.0   * ext_mean
    - 0.05  * employment_years
    + 0.3   * (income_type == 'Pensioner').astype(float)
    + 0.2   * (housing_type == 'Rented apartment').astype(float)
    - 0.3   * (education == 'Higher education').astype(float)
    + np.random.normal(0, 0.5, N)
)
prob_default = 1 / (1 + np.exp(-logit))
target = (np.random.uniform(0, 1, N) < prob_default).astype(int)

print(f"Default rate: {target.mean():.3f} ({target.sum()} defaults out of {N})")

df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'education_type': education,
    'income_type': income_type,
    'family_status': family_status,
    'housing_type': housing_type,
    'family_members': family_members,
    'annual_income': annual_income.round(2),
    'credit_amount': credit_amount.round(2),
    'annuity_amount': annuity.round(2),
    'goods_price': goods_price.round(2),
    'ext_source_1': ext1.round(4),
    'ext_source_2': ext2.round(4),
    'ext_source_3': ext3.round(4),
    'employment_years': employment_years.round(2),
    'dti_ratio': dti.round(4),
    'lti_ratio': lti.round(4),
    'ext_source_mean': ext_mean.round(4),
    'TARGET': target
})

df.to_csv('/home/claude/credit-risk/backend/data/raw/loan_data.csv', index=False)
print(f"Dataset saved: {df.shape}")
