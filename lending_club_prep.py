#!/usr/bin/env python
"""Feature engineering for Lending Club defaults dataset."""
import pandas as pd
import numpy as np

print("Chargement lending_club_defaults.csv...")
df = pd.read_csv('lending_club_defaults.csv', low_memory=False)
print(f"Prets en defaut : {len(df):,}")

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

# 1. Convert term to numeric months (' 36 months' -> 36)
df['term_months'] = df['term'].str.strip().str.extract(r'(\d+)').astype(float)

# 2. Employment length to numeric
emp_map = {
    '< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3,
    '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
    '8 years': 8, '9 years': 9, '10+ years': 10
}
df['emp_length_years'] = df['emp_length'].map(emp_map)

# 3. Issue date to year and month
df['issue_date'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
df['issue_year'] = df['issue_date'].dt.year

# 4. Grade to numeric (A=7 best, G=1 worst)
grade_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
df['grade_num'] = df['grade'].map(grade_map)

# 5. Log transformations
df['log_annual_inc'] = np.log1p(df['annual_inc'])
df['log_funded_amnt'] = np.log1p(df['funded_amnt'])
df['log_revol_bal'] = np.log1p(df['revol_bal'])

# 6. Imputation NaN par médiane / mode
for col in ['emp_length_years', 'revol_util', 'dti', 'inq_last_6mths',
            'open_acc', 'total_acc', 'delinq_2yrs', 'pub_rec']:
    df[col] = df[col].fillna(df[col].median())

# 7. Flag high-risk purposes
high_risk_purposes = ['small_business', 'educational', 'wedding']
df['high_risk_purpose'] = df['purpose'].isin(high_risk_purposes).astype(int)

# 8. Home ownership binary
df['owns_home'] = df['home_ownership'].isin(['OWN', 'MORTGAGE']).astype(int)

# 9. Income verification
df['income_verified'] = df['verification_status'].isin(['Verified', 'Source Verified']).astype(int)

# ═══════════════════════════════════════════════════════════════════════════
# SELECT FEATURES
# ═══════════════════════════════════════════════════════════════════════════
features_loan = [
    'loan_amnt', 'int_rate', 'installment', 'term_months',
    'grade_num', 'log_funded_amnt'
]
features_borrower = [
    'log_annual_inc', 'dti', 'emp_length_years', 'owns_home',
    'income_verified', 'revol_util', 'log_revol_bal'
]
features_credit = [
    'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'total_acc'
]
features_other = ['high_risk_purpose', 'issue_year']

all_features = features_loan + features_borrower + features_credit + features_other

# ═══════════════════════════════════════════════════════════════════════════
# FINAL DATASET
# ═══════════════════════════════════════════════════════════════════════════
df_final = df[['LGD'] + all_features].copy()
df_final = df_final.dropna()

print(f'\nDataset final : {df_final.shape}')
print(f'LGD stats :')
print(df_final['LGD'].describe().round(4))

# Sample down to 50k pour performance (sinon trop lent pour Tobit/Beta)
SAMPLE_SIZE = 50000
df_sample = df_final.sample(n=SAMPLE_SIZE, random_state=42)
print(f'\nSample de {SAMPLE_SIZE:,} observations pour modelisation')
df_sample.to_csv('lending_club_lgd_ready.csv', index=False)
print('Sauvegarde : lending_club_lgd_ready.csv')
