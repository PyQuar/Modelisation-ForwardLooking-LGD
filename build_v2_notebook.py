#!/usr/bin/env python
"""Build the v2 improved LGD Forward-Looking notebook."""
import json, uuid

def cell(cell_type, source):
    """Create a notebook cell."""
    c = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.split('\n') if isinstance(source, str) else source,
    }
    # Fix: source lines need newlines except last
    lines = source.split('\n') if isinstance(source, str) else source
    c["source"] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c

cells = []

# ═══════════════════════════════════════════════════════════════════════════
# CELL 0 — Title
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("markdown", """# Modélisation Forward-Looking LGD — IFRS 9 / Bâle III
## Contexte : Région SADC

Ce notebook implémente une démarche complète de modélisation statistique de la **Loss Given Default (LGD)** selon une approche *forward-looking* conforme à IFRS 9.
La variable cible `LGD` est bornée dans `[0, 1]` — elle représente la proportion de l'exposition perdue en cas de défaut.

**Données** : Dataset synthétique SADC (500 observations) + données macroéconomiques réelles World Bank (10 pays SADC, 2005–2024).

**Améliorations v2** : Split train/test 80/20, validation croisée 5-fold pour tous les modèles, modèle satellite enrichi, tableau des coefficients Tobit, tests statistiques formels, scénarios corrigés."""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 1 — Section 0 header
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("markdown", """---
## Section 0 — Setup & Imports"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — Imports
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Core ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings('ignore')

# ── Scipy ───────────────────────────────────────────────────────────────────
from scipy import stats
from scipy.special import logit, expit
from scipy.optimize import minimize, approx_fprime
from scipy.stats import norm

# ── Statsmodels ─────────────────────────────────────────────────────────────
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.othermod.betareg import BetaModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.nonparametric.smoothers_lowess import lowess

# ── Scikit-learn ─────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── XGBoost + SHAP ───────────────────────────────────────────────────────────
import xgboost as xgb
import shap

# ── Visualisation ────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Config ───────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('muted')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print('Environnement prêt ✓')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 3 — Section 1 header
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("markdown", """---
## Section 1 — Collecte Données Macro SADC (World Bank API)

On récupère 3 indicateurs macroéconomiques pour **10 pays SADC** sur la période 2005–2024 via l'API World Bank.
Ces données serviront à :
1. Calibrer les scénarios forward-looking sur des percentiles historiques réels
2. Contextualiser le dataset synthétique dans un cadre économique vérifié"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 4 — WB API fetch
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """import time

SADC_COUNTRIES = ['ZAF','ZWE','ZMB','MOZ','TZA','BWA','NAM','MWI','LSO','SWZ']
WB_INDICATORS  = {
    'GDP_Growth':    'NY.GDP.MKTP.KD.ZG',
    'Inflation_CPI': 'FP.CPI.TOTL.ZG',
    'Lending_Rate':  'FR.INR.LEND',
}

def fetch_wb_indicator(indicator_code, countries, start=2005, end=2024, retries=3):
    country_str = ';'.join(countries)
    url = (f"https://api.worldbank.org/v2/country/{country_str}"
           f"/indicator/{indicator_code}?format=json&per_page=1000"
           f"&date={start}:{end}")
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            if len(data) < 2 or data[1] is None:
                print(f"  ⚠ Pas de données pour {indicator_code} (tentative {attempt+1})")
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return pd.DataFrame(columns=['country','iso3','year','value'])
            rows = [{'country': d['country']['value'],
                     'iso3':    d['countryiso3code'],
                     'year':    int(d['date']),
                     'value':   d['value']}
                    for d in data[1] if d['value'] is not None]
            if rows:
                return pd.DataFrame(rows)
            if attempt < retries - 1:
                time.sleep(2)
                continue
        except Exception as e:
            print(f"  ⚠ Erreur API (tentative {attempt+1}) : {e}")
            if attempt < retries - 1:
                time.sleep(2)
                continue
    return pd.DataFrame(columns=['country','iso3','year','value'])

wb = {}
for name, code in WB_INDICATORS.items():
    print(f'Récupération {name} ({code})...', end=' ')
    wb[name] = fetch_wb_indicator(code, SADC_COUNTRIES)
    print(f'{len(wb[name])} observations')

print(f'\\nTotal : {sum(len(v) for v in wb.values())} observations récupérées')

# Vérification : au moins des données pour GDP
assert len(wb['GDP_Growth']) > 0, "Aucune donnée GDP récupérée — vérifier la connexion Internet" """))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 5 — WB percentiles
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Calcul des percentiles historiques SADC pour calibrer les scénarios ──────
# On exclut les valeurs extrêmes (hyperinflation Zimbabwe, etc.)
for name, df_wb in wb.items():
    if len(df_wb) == 0 or 'value' not in df_wb.columns:
        print(f'{name:20s} : ⚠ Pas de données disponibles')
        continue
    q = df_wb['value'].dropna().quantile([.10, .25, .50, .75, .90])
    print(f'{name:20s} : P10={q[.10]:8.2f}  P25={q[.25]:8.2f}  P50={q[.50]:8.2f}'
          f'  P75={q[.75]:8.2f}  P90={q[.90]:8.2f}')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 6 — WB visualization
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Visualisation des distributions macro SADC ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
fig.suptitle('Distributions historiques SADC (World Bank 2005–2024)', fontsize=12, fontweight='bold')

for ax, (name, df_wb) in zip(axes, wb.items()):
    if len(df_wb) == 0 or 'value' not in df_wb.columns:
        ax.set_title(f'{name} — Pas de données')
        continue
    vals = df_wb['value'].dropna()
    if len(vals) < 5:
        ax.set_title(f'{name} — Données insuffisantes')
        continue
    vals_clipped = vals.clip(lower=vals.quantile(0.02), upper=vals.quantile(0.98))
    ax.hist(vals_clipped, bins=30, color='steelblue', alpha=0.7, edgecolor='white', density=True)
    kde = stats.gaussian_kde(vals_clipped)
    x = np.linspace(vals_clipped.min(), vals_clipped.max(), 200)
    ax.plot(x, kde(x), 'r-', lw=2)
    ax.axvline(vals.median(), color='navy', linestyle='--', lw=1.5, label=f'Médiane={vals.median():.1f}')
    ax.set_title(name.replace('_', ' '))
    ax.set_xlabel('%'); ax.legend(fontsize=8)

plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 7 — Section 2 header
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("markdown", """---
## Section 2 — Chargement & Audit du Dataset LGD"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 8 — Load data
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """df = pd.read_csv('synthetic_sadc_lgd_dataset (1).csv')

print(f'Shape : {df.shape}')
print(f'\\nTypes :\\n{df.dtypes}')
print(f'\\nValeurs manquantes :\\n{df.isnull().sum()}')
print(f'\\nDoublons : {df.duplicated().sum()}')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 9 — Descriptive stats
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """print('=== Statistiques descriptives ===')
df.describe(percentiles=[.10,.25,.50,.75,.90]).T.round(3)"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 10 — LGD audit
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Audit critique : valeurs limites de la LGD ───────────────────────────────
n = len(df)
exact_zero = (df['LGD'] == 0).sum()
exact_one  = (df['LGD'] == 1).sum()
interior   = n - exact_zero - exact_one

print(f'LGD = 0  : {exact_zero:>4d}  ({exact_zero/n:.1%})')
print(f'LGD = 1  : {exact_one:>4d}  ({exact_one/n:.1%})')
print(f'0 < LGD < 1 : {interior:>4d}  ({interior/n:.1%})')
print(f'\\n→ Présence de masses ponctuelles aux bornes → Tobit et FRM adaptés')
print(f'→ Beta Regression nécessite la transformation Smithson-Verkuilen')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 11 — VIF
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Analyse VIF (multicolinéarité) ───────────────────────────────────────────
num_cols = ['GDP_Growth_Percent','Inflation_Rate_Percent','Policy_Rate_Percent',
            'Lending_Rate_Percent','Exposure_at_Default','Asset_Coverage_Value',
            'Loan_Duration_Months','Risk_Score','Applicant_Age','Household_Income']

X_vif = sm.add_constant(df[num_cols])
vif_data = pd.DataFrame({
    'Variable': num_cols,
    'VIF': [variance_inflation_factor(X_vif.values, i+1) for i in range(len(num_cols))]
}).sort_values('VIF', ascending=False)

print('=== Variance Inflation Factors ===')
print(vif_data.to_string(index=False))
print(f'\\nVIF max = {vif_data["VIF"].max():.2f}')
print('→ VIF < 5 : pas de multicolinéarité sévère' if vif_data['VIF'].max() < 5
      else '→ ⚠ VIF > 5 détecté — envisager de retirer des variables corrélées')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 12 — Section 3 header
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("markdown", """---
## Section 3 — Analyse Exploratoire (EDA)"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 13 — EDA distribution (+ Shapiro-Wilk, KS test)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── 3.1 Distribution de la LGD ───────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Distribution de la LGD — Analyse exploratoire', fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig)

# Panel 1 : Histogramme + KDE
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['LGD'], bins=40, density=True, alpha=0.65, color='steelblue',
         edgecolor='white', label='Empirique')
kde = stats.gaussian_kde(df['LGD'])
x   = np.linspace(0, 1, 300)
ax1.plot(x, kde(x), 'r-', lw=2, label='KDE')
ax1.set_xlabel('LGD'); ax1.set_ylabel('Densité')
ax1.set_title('Distribution LGD + KDE')
ax1.legend()

# Panel 2 : QQ-plot vs distribution Bêta ajustée
ax2 = fig.add_subplot(gs[0, 1])
lgd_int = df['LGD'][(df['LGD'] > 0) & (df['LGD'] < 1)]
a_fit, b_fit, _, _ = stats.beta.fit(lgd_int, floc=0, fscale=1)
stats.probplot(lgd_int, dist=stats.beta(a_fit, b_fit), plot=ax2)
ax2.set_title(f'QQ-plot vs Beta({a_fit:.2f}, {b_fit:.2f})')

# Panel 3 : CDF empirique
ax3 = fig.add_subplot(gs[0, 2])
sorted_lgd = np.sort(df['LGD'])
ecdf = np.arange(1, n+1) / n
ax3.plot(sorted_lgd, ecdf, 'navy', lw=2)
ax3.set_xlabel('LGD'); ax3.set_ylabel('Probabilité cumulée')
ax3.set_title('CDF empirique de la LGD')

# Panel 4 : Boxplot par Loan_Category
ax4 = fig.add_subplot(gs[1, 0])
df.boxplot(column='LGD', by='Loan_Category', ax=ax4)
plt.sca(ax4); plt.title('LGD par catégorie de prêt'); plt.suptitle('')

# Panel 5 : Boxplot par Employment_Status
ax5 = fig.add_subplot(gs[1, 1])
df.boxplot(column='LGD', by='Employment_Status', ax=ax5)
plt.sca(ax5); plt.title('LGD par statut emploi'); plt.suptitle('')
plt.xticks(rotation=15)

# Panel 6 : Scatter LGD vs Risk_Score
ax6 = fig.add_subplot(gs[1, 2])
sc = ax6.scatter(df['Risk_Score'], df['LGD'], c=df['LGD'],
                  cmap='RdYlGn_r', alpha=0.5, s=18)
plt.colorbar(sc, ax=ax6, label='LGD')
ax6.set_xlabel('Risk Score'); ax6.set_ylabel('LGD')
ax6.set_title('LGD vs Risk Score')

plt.tight_layout()
plt.show()

# ── Tests statistiques formels ────────────────────────────────────────────────
print(f'Skewness LGD : {df["LGD"].skew():.3f}')
print(f'Kurtosis LGD : {df["LGD"].kurt():.3f}')

# Shapiro-Wilk (normalité) — sur un sous-échantillon si n > 500
sw_stat, sw_pval = stats.shapiro(df['LGD'].sample(min(n, 500), random_state=42))
print(f'\\nTest de Shapiro-Wilk (normalité) : W = {sw_stat:.4f}, p = {sw_pval:.4e}')
print(f'  → {"Rejet de la normalité (p < 0.05)" if sw_pval < 0.05 else "Non-rejet de la normalité"}')

# Kolmogorov-Smirnov vs distribution Bêta ajustée
ks_stat, ks_pval = stats.kstest(lgd_int, 'beta', args=(a_fit, b_fit))
print(f'\\nTest KS vs Beta({a_fit:.2f}, {b_fit:.2f}) : D = {ks_stat:.4f}, p = {ks_pval:.4e}')
print(f'  → {"Rejet : LGD ne suit pas une loi Bêta pure" if ks_pval < 0.05 else "Non-rejet : LGD compatible avec une loi Bêta"}')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 13b — EDA interpretation markdown
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("markdown", """### Interprétation EDA

- **Distribution LGD** : Les tests de Shapiro-Wilk et KS confirment que la LGD ne suit ni une loi normale ni une loi Bêta pure → la **régression Bêta** nécessite la transformation Smithson-Verkuilen, et le **Tobit** est justifié par les masses aux bornes 0 et 1.
- **LGD par catégorie** : Les boxplots révèlent si certaines catégories de prêt ou statuts d'emploi ont un profil de pertes systématiquement différent.
- **Corrélations macro-LGD** : Les corrélations faibles (< 0.05) entre variables macro et LGD dans ce dataset synthétique limitent le pouvoir du modèle satellite macro-only. Un **satellite enrichi** (macro + caractéristiques prêt) sera développé en Section 6."""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 14 — LOWESS
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── 3.2 Variables macro vs LGD avec LOWESS ─────────────────────────────────
macro_cols = ['GDP_Growth_Percent','Inflation_Rate_Percent','Policy_Rate_Percent','Lending_Rate_Percent']

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle('Variables macro vs LGD — LOWESS smoother', fontsize=12, fontweight='bold')

for ax, col in zip(axes, macro_cols):
    ax.scatter(df[col], df['LGD'], alpha=0.3, s=12, color='steelblue')
    lw_result = lowess(df['LGD'], df[col], frac=0.4)
    ax.plot(lw_result[:, 0], lw_result[:, 1], 'r-', lw=2.5, label='LOWESS')
    r_corr = df[col].corr(df['LGD'])
    ax.set_title(f'{col}\\nr = {r_corr:.3f}', fontsize=9)
    ax.set_xlabel(col); ax.set_ylabel('LGD')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 15 — Correlation matrix
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── 3.3 Matrice de corrélation ──────────────────────────────────────────────
corr = df[num_cols + ['LGD']].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={'fontsize': 8})
ax.set_title('Matrice de corrélation', fontsize=12)
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 16 — Section 4 header
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("markdown", """---
## Section 4 — Feature Engineering"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 17 — LGD transformations
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── 4.1 Transformations de la LGD ────────────────────────────────────────────
# Smithson-Verkuilen : mappe [0,1] → (0,1) strictement, sans modifier la forme
df['LGD_SV']    = (df['LGD'] * (n - 1) + 0.5) / n
# Logit : mappe (0,1) → (-∞, +∞) pour le modèle satellite OLS
df['LGD_logit'] = np.log(df['LGD_SV'] / (1 - df['LGD_SV']))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, title, color in zip(
        axes,
        ['LGD', 'LGD_SV', 'LGD_logit'],
        ['LGD original', 'LGD transformé SV — (0,1)', 'logit(LGD_SV)'],
        ['steelblue', 'seagreen', 'tomato']):
    ax.hist(df[col], bins=40, color=color, alpha=0.75, edgecolor='white')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(col)
    ax.set_ylabel('Fréquence')

plt.suptitle('Transformations de la variable cible LGD', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 18 — Feature engineering
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── 4.2 Nouvelles variables économiques ──────────────────────────────────────
# Taux réel : Lending - Inflation (peut être négatif en SADC → répression financière)
df['Real_Lending_Rate']         = df['Lending_Rate_Percent'] - df['Inflation_Rate_Percent']
# Spread banque : marge entre taux de prêt et taux directeur
df['Rate_Spread']               = df['Lending_Rate_Percent'] - df['Policy_Rate_Percent']
# Ratio de couverture par le collatéral (prédicteur théorique central — Merton 1974)
df['Collateral_Coverage_Ratio'] = (df['Asset_Coverage_Value'] / df['Exposure_at_Default']).clip(upper=10.0)
df['Undercollateralized']       = (df['Collateral_Coverage_Ratio'] < 1.0).astype(int)
# Indicateur de récession
df['Recession_Indicator']       = (df['GDP_Growth_Percent'] < 0).astype(int)
# Log-transformations des variables financières (réduire asymétrie)
df['Log_Exposure']              = np.log1p(df['Exposure_at_Default'])
df['Log_Asset_Coverage']        = np.log1p(df['Asset_Coverage_Value'])
df['Log_Income']                = np.log1p(df['Household_Income'])
df['Loan_Duration_Years']       = df['Loan_Duration_Months'] / 12

print('Variables créées :')
new_vars = ['Real_Lending_Rate','Rate_Spread','Collateral_Coverage_Ratio',
            'Undercollateralized','Recession_Indicator']
print(df[new_vars].describe().T.round(3))"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 19 — Encoding + Feature sets + TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── 4.3 Encodage one-hot des catégoriels ─────────────────────────────────────
df_enc = pd.get_dummies(df, columns=['Loan_Category','Employment_Status'],
                         drop_first=True, dtype=float)

print('Catégorie de référence — Loan_Category : Agricultural')
print('Catégorie de référence — Employment_Status : Employed')
print('\\nColonnes ajoutées :', [c for c in df_enc.columns
                                if 'Loan_Category' in c or 'Employment' in c])

# ── Sets de variables ─────────────────────────────────────────────────────────
MACRO_FEATURES = ['GDP_Growth_Percent','Real_Lending_Rate',
                   'Rate_Spread','Recession_Indicator']
LOAN_FEATURES  = ['Log_Exposure','Collateral_Coverage_Ratio',
                   'Loan_Duration_Years','Log_Income','Risk_Score',
                   'Applicant_Age','Undercollateralized']
CAT_FEATURES   = [c for c in df_enc.columns
                  if 'Loan_Category' in c or 'Employment' in c]
ALL_FEATURES   = MACRO_FEATURES + LOAN_FEATURES + CAT_FEATURES

X    = df_enc[ALL_FEATURES]
y    = df_enc['LGD']
y_sv = df_enc['LGD_SV']
y_logit = df_enc['LGD_logit']

# ── Nettoyage : remplacer inf et NaN ─────────────────────────────────────────
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print(f'\\nMatrice X : {X.shape}  |  y : {y.shape}')
print(f"NaN restants dans X : {X.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# ── SPLIT TRAIN / TEST 80/20 ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE)

# Correspondre les versions SV et logit
y_sv_train = y_sv.loc[X_train.index]
y_sv_test  = y_sv.loc[X_test.index]
y_logit_train = y_logit.loc[X_train.index]
y_logit_test  = y_logit.loc[X_test.index]

# Matrices pour statsmodels (avec constante)
X_sm_train = sm.add_constant(X_train)
X_sm_test  = sm.add_constant(X_test)
X_sm       = sm.add_constant(X)   # full pour le satellite

# KFold pour la validation croisée (défini une seule fois)
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print(f'\\nTrain : {X_train.shape[0]} obs  |  Test : {X_test.shape[0]} obs')
print(f'LGD train mean : {y_train.mean():.4f}  |  LGD test mean : {y_test.mean():.4f}')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 20 — Section 5 header
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("markdown", """---
## Section 5 — Développement des Modèles

Cinq méthodes spécifiques à la modélisation LGD sont comparées :
1. **Tobit** — régression censurée deux côtés
2. **Régression Bêta** — distribution naturelle pour les proportions dans (0,1)
3. **FRM** — Fractional Response Model (Papke & Wooldridge, 1996)
4. **Random Forest** — méthode ensembliste
5. **XGBoost + SHAP** — boosting avec explicabilité

**Tous les modèles sont entraînés sur le set d'entraînement (80%) et évalués hors-échantillon sur le set de test (20%).**"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 21 — TOBIT (with Hessian coefficient table)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ══════════════════════════════════════════════════════════════════════════════
# 5.1 — RÉGRESSION TOBIT (censurée deux côtés [0, 1])
# ══════════════════════════════════════════════════════════════════════════════

def tobit_nll(params, X_np, y_np, lo=0.0, hi=1.0):
    beta  = params[:-1]
    sigma = np.exp(params[-1])
    Xb    = X_np @ beta
    mask_lo = (y_np == lo)
    mask_hi = (y_np == hi)
    mask_in = ~mask_lo & ~mask_hi
    ll = 0.0
    if mask_lo.any():
        ll += np.sum(np.log(norm.cdf((lo - Xb[mask_lo]) / sigma) + 1e-12))
    if mask_hi.any():
        ll += np.sum(np.log(1 - norm.cdf((hi - Xb[mask_hi]) / sigma) + 1e-12))
    if mask_in.any():
        ll += np.sum(norm.logpdf((y_np[mask_in] - Xb[mask_in]) / sigma) - np.log(sigma))
    return -ll

def tobit_predict(X_np, beta, sigma, lo=0.0, hi=1.0):
    Xb       = X_np @ beta
    al       = (lo - Xb) / sigma
    ah       = (hi - Xb) / sigma
    phi_l, phi_h = norm.pdf(al), norm.pdf(ah)
    Phi_l, Phi_h = norm.cdf(al), norm.cdf(ah)
    denom    = Phi_h - Phi_l + 1e-12
    E_in     = Xb + sigma * (phi_l - phi_h) / denom
    return np.clip(E_in * denom + lo * Phi_l + hi * (1 - Phi_h), lo, hi)

# Entraîner sur TRAIN uniquement
X_np_train = np.nan_to_num(sm.add_constant(X_train.values), nan=0.0, posinf=0.0, neginf=0.0)
y_np_train = y_train.values

try:
    ols_init = np.linalg.lstsq(X_np_train, y_np_train, rcond=None)[0]
except np.linalg.LinAlgError:
    ols_init = np.zeros(X_np_train.shape[1])

sig_init  = np.log(np.std(y_np_train - X_np_train @ ols_init) + 1e-6)
p0        = np.append(ols_init, sig_init)

res_tobit = minimize(tobit_nll, p0, args=(X_np_train, y_np_train),
                      method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-10})

beta_tobit  = res_tobit.x[:-1]
sigma_tobit = np.exp(res_tobit.x[-1])

# Prédictions train et test
X_np_full  = np.nan_to_num(sm.add_constant(X.values), nan=0.0, posinf=0.0, neginf=0.0)
X_np_test  = np.nan_to_num(sm.add_constant(X_test.values), nan=0.0, posinf=0.0, neginf=0.0)
y_pred_tobit_train = tobit_predict(X_np_train, beta_tobit, sigma_tobit)
y_pred_tobit_test  = tobit_predict(X_np_test, beta_tobit, sigma_tobit)
y_pred_tobit       = tobit_predict(X_np_full, beta_tobit, sigma_tobit)

# ── Tableau des coefficients via Hessienne numérique ──────────────────────────
eps = np.sqrt(np.finfo(float).eps)
hess = np.zeros((len(res_tobit.x), len(res_tobit.x)))
for i in range(len(res_tobit.x)):
    def fi(params): return approx_fprime(params, tobit_nll, eps, X_np_train, y_np_train)[i]
    hess[i, :] = approx_fprime(res_tobit.x, fi, eps)

try:
    cov_mat   = np.linalg.inv(hess)
    std_errs  = np.sqrt(np.abs(np.diag(cov_mat)))
    z_values  = res_tobit.x / std_errs
    p_values  = 2 * (1 - norm.cdf(np.abs(z_values)))

    param_names = ['const'] + list(X_train.columns) + ['log(sigma)']
    coef_df = pd.DataFrame({
        'Variable': param_names,
        'Coef': res_tobit.x,
        'Std.Err': std_errs,
        'z': z_values,
        'p-value': p_values,
        'Signif.': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in p_values]
    })
    print('=== Tobit — Tableau des coefficients ===')
    print(coef_df.to_string(index=False, float_format='%.4f'))
except np.linalg.LinAlgError:
    print('⚠ Hessienne singulière — coefficients sans erreurs standard')

print(f'\\nTobit convergé : {res_tobit.success}')
print(f'σ estimé       : {sigma_tobit:.4f}')
print(f'Log-vraisemblance : {-res_tobit.fun:.2f}')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 22 — BETA REGRESSION (train set, improved)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ══════════════════════════════════════════════════════════════════════════════
# 5.2 — RÉGRESSION BÊTA (Ferrari & Cribari-Neto, 2004)
# ══════════════════════════════════════════════════════════════════════════════
# y ~ Beta(μφ, (1-μ)φ)  |  logit(μ) = Xβ
# Entraîné sur y_sv_train (transformation Smithson-Verkuilen)

beta_model  = BetaModel(y_sv_train, X_sm_train)
beta_result = beta_model.fit(disp=False, maxiter=500)

# Prédictions
y_pred_beta_train = beta_result.predict(X_sm_train)
y_pred_beta_test  = beta_result.predict(X_sm_test)
y_pred_beta       = beta_result.predict(X_sm)

print('=== Régression Bêta (entraîné sur train set) ===')
print(beta_result.summary())"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 23 — FRM (train set)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ══════════════════════════════════════════════════════════════════════════════
# 5.3 — FRACTIONAL RESPONSE MODEL — FRM (Papke & Wooldridge, 1996)
# ══════════════════════════════════════════════════════════════════════════════
# GLM Bernoulli avec lien logit + erreurs robustes HC3

frm_model  = sm.GLM(y_train, X_sm_train,
                     family=sm.families.Binomial(link=sm.families.links.Logit()))
frm_result = frm_model.fit(cov_type='HC3')

# Prédictions
y_pred_frm_train = frm_result.predict(X_sm_train)
y_pred_frm_test  = frm_result.predict(X_sm_test)
y_pred_frm       = frm_result.predict(X_sm)

# Test de surdispersion (quasi-binomial)
frm_quasi  = sm.GLM(y_train, X_sm_train,
                     family=sm.families.Binomial(link=sm.families.links.Logit()))
frm_quasi_result = frm_quasi.fit(scale='X2')

print('=== Fractional Response Model (FRM) — entraîné sur train set ===')
print(frm_result.summary())
print(f'\\nParamètre de dispersion (quasi-binomial) : {frm_quasi_result.scale:.4f}')
print('(>> 1 → surdispersion présente, quasi-binomial approprié)')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 24 — RANDOM FOREST (train set)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ══════════════════════════════════════════════════════════════════════════════
# 5.4 — RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
rf_model = RandomForestRegressor(
    n_estimators=500, max_depth=8, min_samples_leaf=10,
    max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1
)

# CV sur train set
rf_cv = cross_val_score(rf_model, X_train, y_train, cv=kf,
                         scoring='neg_root_mean_squared_error')
print(f'RF — CV RMSE (train) : {-rf_cv.mean():.4f} ± {rf_cv.std():.4f}')

rf_model.fit(X_train, y_train)
y_pred_rf_train = np.clip(rf_model.predict(X_train), 0, 1)
y_pred_rf_test  = np.clip(rf_model.predict(X_test), 0, 1)
y_pred_rf       = np.clip(rf_model.predict(X), 0, 1)

# Importance des variables
fi_rf = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
fi_rf.head(12).plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)
ax.invert_yaxis()
ax.set_title('Random Forest — Top 12 importances des variables')
ax.set_xlabel('Mean Decrease Impurity')
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 25 — XGBOOST + SHAP (train set, SHAP on test)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ══════════════════════════════════════════════════════════════════════════════
# 5.5 — XGBOOST + SHAP
# ══════════════════════════════════════════════════════════════════════════════
xgb_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    objective='reg:squarederror',
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
)

# CV sur train set
xgb_cv = cross_val_score(xgb_model, X_train, y_train, cv=kf,
                          scoring='neg_root_mean_squared_error')
print(f'XGB — CV RMSE (train) : {-xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}')

xgb_model.fit(X_train, y_train)
y_pred_xgb_train = np.clip(xgb_model.predict(X_train), 0, 1)
y_pred_xgb_test  = np.clip(xgb_model.predict(X_test), 0, 1)
y_pred_xgb       = np.clip(xgb_model.predict(X), 0, 1)

# ── SHAP sur le set de TEST (hors-échantillon) ──────────────────────────────
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('XGBoost — Explicabilité SHAP (set de test)', fontsize=13, fontweight='bold')

plt.sca(axes[0])
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
axes[0].set_title('Importance globale (|SHAP| moyen)')

plt.sca(axes[1])
shap.summary_plot(shap_values, X_test, show=False)
axes[1].set_title('Beeswarm — Direction et magnitude')

plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 25b — CV 5-fold for ALL models (including Tobit, Beta, FRM)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ══════════════════════════════════════════════════════════════════════════════
# 5.6 — VALIDATION CROISÉE 5-FOLD POUR TOUS LES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
cv_results = {}

# RF et XGB déjà faits via cross_val_score
cv_results['Random Forest'] = -rf_cv.mean()
cv_results['XGBoost']       = -xgb_cv.mean()

# CV manuelle pour Tobit, Beta, FRM
for model_name in ['Tobit', 'Beta Reg.', 'FRM']:
    fold_rmses = []
    for train_idx, val_idx in kf.split(X_train):
        Xf_tr = X_train.iloc[train_idx]
        Xf_va = X_train.iloc[val_idx]
        yf_tr = y_train.iloc[train_idx]
        yf_va = y_train.iloc[val_idx]

        if model_name == 'Tobit':
            Xn_tr = np.nan_to_num(sm.add_constant(Xf_tr.values), nan=0.0, posinf=0.0, neginf=0.0)
            yn_tr = yf_tr.values
            try:
                ols_i = np.linalg.lstsq(Xn_tr, yn_tr, rcond=None)[0]
            except:
                ols_i = np.zeros(Xn_tr.shape[1])
            sig_i = np.log(np.std(yn_tr - Xn_tr @ ols_i) + 1e-6)
            p0_i  = np.append(ols_i, sig_i)
            res_i = minimize(tobit_nll, p0_i, args=(Xn_tr, yn_tr),
                             method='L-BFGS-B', options={'maxiter': 1000})
            beta_i  = res_i.x[:-1]
            sigma_i = np.exp(res_i.x[-1])
            Xn_va = np.nan_to_num(sm.add_constant(Xf_va.values), nan=0.0, posinf=0.0, neginf=0.0)
            yf_pred = tobit_predict(Xn_va, beta_i, sigma_i)

        elif model_name == 'Beta Reg.':
            Xf_tr_sm = sm.add_constant(Xf_tr)
            Xf_va_sm = sm.add_constant(Xf_va)
            ysv_tr   = y_sv.loc[Xf_tr.index]
            try:
                bm_i = BetaModel(ysv_tr, Xf_tr_sm).fit(disp=False, maxiter=300)
                yf_pred = bm_i.predict(Xf_va_sm)
            except:
                yf_pred = np.full(len(yf_va), yf_tr.mean())

        elif model_name == 'FRM':
            Xf_tr_sm = sm.add_constant(Xf_tr)
            Xf_va_sm = sm.add_constant(Xf_va)
            try:
                fm_i = sm.GLM(yf_tr, Xf_tr_sm,
                              family=sm.families.Binomial(link=sm.families.links.Logit()))
                fm_r = fm_i.fit(cov_type='HC3')
                yf_pred = fm_r.predict(Xf_va_sm)
            except:
                yf_pred = np.full(len(yf_va), yf_tr.mean())

        fold_rmses.append(np.sqrt(mean_squared_error(yf_va, yf_pred)))

    cv_results[model_name] = np.mean(fold_rmses)
    print(f'{model_name:<15} — CV 5-fold RMSE : {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}')

print(f"{'Random Forest':<15} — CV 5-fold RMSE : {cv_results['Random Forest']:.4f}")
print(f"{'XGBoost':<15} — CV 5-fold RMSE : {cv_results['XGBoost']:.4f}")

print('\\n=== Classement CV RMSE (meilleur → pire) ===')
for name, rmse in sorted(cv_results.items(), key=lambda x: x[1]):
    print(f'  {name:<15} : {rmse:.4f}')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 26 — Section 6 header (enriched satellite)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("markdown", """---
## Section 6 — Modèle Satellite Macroéconomique

Le **modèle satellite** relie les variables macro à la LGD pour permettre la projection sous scénarios.

**Approche deux étages** :
1. **Satellite macro-only** : uniquement variables macro → faible R² attendu (corrélations faibles dans ce dataset) mais nécessaire pour isoler le risque systémique
2. **Satellite enrichi** : macro + caractéristiques prêt/emprunteur → meilleur R², utilisé pour la projection forward-looking via l'**approche mean-adjustment** (EBA) : on fixe les variables macro aux valeurs du scénario et les variables prêt/emprunteur à leur moyenne portefeuille"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 27 — Satellite macro-only
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ══════════════════════════════════════════════════════════════════════════════
# 6.1 — SATELLITE MACRO-ONLY
# ══════════════════════════════════════════════════════════════════════════════
X_sat_macro = sm.add_constant(df_enc[MACRO_FEATURES])

# OLS sur logit(LGD)
sat_macro_ols = OLS(y_logit, X_sat_macro).fit(cov_type='HC3')
print('=== Satellite Macro-Only — OLS sur logit(LGD) ===')
print(sat_macro_ols.summary())
print(f'\\n→ R² macro-only = {sat_macro_ols.rsquared:.4f}')
print('  (Faible R² attendu : les corrélations LGD-macro sont < 0.05 dans ce dataset)')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 28 — Satellite enrichi + FRM satellite
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ══════════════════════════════════════════════════════════════════════════════
# 6.2 — SATELLITE ENRICHI (macro + prêt + emprunteur)
# ══════════════════════════════════════════════════════════════════════════════
# Approche EBA (European Banking Authority) : modèle complet, projection
# via mean-adjustment (fixer macro aux scénarios, loan/borrower aux moyennes)

X_sat_full = sm.add_constant(X)  # ALL_FEATURES

# OLS enrichi sur logit(LGD)
sat_enriched_ols = OLS(y_logit, X_sat_full).fit(cov_type='HC3')
print('=== Satellite Enrichi — OLS sur logit(LGD) ===')
print(f'R² enrichi = {sat_enriched_ols.rsquared:.4f}  (vs macro-only R² = {sat_macro_ols.rsquared:.4f})')
print(sat_enriched_ols.summary())

# FRM satellite enrichi
sat_frm = sm.GLM(y, X_sat_full,
                  family=sm.families.Binomial(link=sm.families.links.Logit()))
sat_frm_r = sat_frm.fit(cov_type='HC3')
print('\\n=== Satellite Enrichi — FRM ===')
print(f'Pseudo R² FRM = {1 - sat_frm_r.deviance / sat_frm_r.null_deviance:.4f}')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 29 — Satellite diagnostics
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Diagnostics du modèle satellite enrichi ──────────────────────────────────
sat_resid  = sat_enriched_ols.resid
sat_fitted = sat_enriched_ols.fittedvalues

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('Diagnostics Modèle Satellite Enrichi (OLS logit-LGD)', fontsize=12)

# Résidus vs fitted
axes[0].scatter(sat_fitted, sat_resid, alpha=0.4, s=14)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Valeurs ajustées (logit)'); axes[0].set_ylabel('Résidus')
axes[0].set_title('Résidus vs Fitted')

# QQ-plot
stats.probplot(sat_resid, plot=axes[1])
axes[1].set_title('QQ-plot des résidus')

# Résidus vs GDP_Growth
axes[2].scatter(df_enc['GDP_Growth_Percent'], sat_resid, alpha=0.4, s=14)
axes[2].axhline(0, color='red', linestyle='--')
axes[2].set_xlabel('GDP Growth (%)'); axes[2].set_ylabel('Résidus')
axes[2].set_title('Résidus vs GDP Growth')

plt.tight_layout()
plt.show()

# Test de Breusch-Pagan
bp_stat, bp_pval, _, _ = het_breuschpagan(sat_resid, X_sat_full)
print(f'Test Breusch-Pagan : stat = {bp_stat:.3f}, p-value = {bp_pval:.4f}')
print('→ p < 0.05 : hétéroscédasticité → erreurs HC3 justifiées' if bp_pval < 0.05
      else "→ p ≥ 0.05 : pas d'hétéroscédasticité significative")"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 30 — Section 7 header
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("markdown", """---
## Section 7 — Scénarios Forward-Looking (IFRS 9)

Les **3 scénarios** sont calibrés sur les percentiles historiques réels SADC (Section 1) :
- **Baseline (50%)** : conditions proches de la médiane historique
- **Adverse (30%)** : ralentissement économique, pressions sur les taux
- **Severely Adverse (20%)** : récession profonde, taux réels fortement négatifs

**Approche mean-adjustment (EBA)** : le satellite enrichi est utilisé en fixant les variables macro aux valeurs du scénario et les variables prêt/emprunteur à leur moyenne portefeuille."""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 31 — WB calibration
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Calcul des références World Bank pour chaque variable macro du modèle ────
wb_merged = (
    wb['GDP_Growth'].rename(columns={'value': 'gdp'})
    .merge(wb['Inflation_CPI'].rename(columns={'value': 'cpi'}),
           on=['country','iso3','year'])
    .merge(wb['Lending_Rate'].rename(columns={'value': 'lend'}),
           on=['country','iso3','year'])
)
wb_merged['real_rate'] = wb_merged['lend'] - wb_merged['cpi']

# Plafonner les extrêmes
wb_merged = wb_merged[
    (wb_merged['cpi'] <= 150) &
    (wb_merged['lend'] <= 80)
]

gdp_q  = wb_merged['gdp'].quantile
rr_q   = wb_merged['real_rate'].quantile
rs_q   = df_enc['Rate_Spread'].quantile

print('Calibration des scénarios (percentiles SADC historiques) :')
print(f"  GDP Growth  : P10={gdp_q(.10):.2f}  P25={gdp_q(.25):.2f}  P50={gdp_q(.50):.2f}  P75={gdp_q(.75):.2f}  P90={gdp_q(.90):.2f}")
print(f"  Real Rate   : P10={rr_q(.10):.2f}  P25={rr_q(.25):.2f}  P50={rr_q(.50):.2f}  P75={rr_q(.75):.2f}  P90={rr_q(.90):.2f}")
print(f"  Rate Spread : P10={rs_q(.10):.2f}  P25={rs_q(.25):.2f}  P50={rs_q(.50):.2f}  P75={rs_q(.75):.2f}  P90={rs_q(.90):.2f}")"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 32 — Scenario definition (FIXED Severely Adverse GDP)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Définition des 3 scénarios (GDP Severely Adverse forcé < 0) ───────────────
# Le P10 du GDP peut être positif dans les données historiques SADC.
# Pour le scénario Severely Adverse avec Recession_Indicator=1,
# on force un GDP négatif : min(P10, -0.5%)
sa_gdp = min(gdp_q(.10), -0.5)

scenarios = {
    'Baseline': {
        'GDP_Growth_Percent': gdp_q(.50),
        'Real_Lending_Rate':  rr_q(.50),
        'Rate_Spread':        rs_q(.50),
        'Recession_Indicator': 0,
        'Probability': 0.50,
        'Narrative': 'Conditions proches de la médiane historique SADC'
    },
    'Adverse': {
        'GDP_Growth_Percent': gdp_q(.25),
        'Real_Lending_Rate':  rr_q(.75),
        'Rate_Spread':        rs_q(.75),
        'Recession_Indicator': 0,
        'Probability': 0.30,
        'Narrative': 'Ralentissement économique, pressions inflationnistes (quartile bas PIB)'
    },
    'Severely Adverse': {
        'GDP_Growth_Percent': sa_gdp,
        'Real_Lending_Rate':  rr_q(.90),
        'Rate_Spread':        rs_q(.90),
        'Recession_Indicator': 1,
        'Probability': 0.20,
        'Narrative': f'Récession profonde (GDP={sa_gdp:.2f}%), taux réels élevés'
    }
}

assert abs(sum(s['Probability'] for s in scenarios.values()) - 1.0) < 1e-9

print('=== Définition des scénarios forward-looking ===')
for name, s in scenarios.items():
    print(f"\\n{name} (poids {s['Probability']:.0%}) — {s['Narrative']}")
    print(f"  GDP Growth      : {s['GDP_Growth_Percent']:.2f}%")
    print(f"  Real Lend. Rate : {s['Real_Lending_Rate']:.2f}%")
    print(f"  Rate Spread     : {s['Rate_Spread']:.2f}%")
    print(f"  Récession       : {'Oui' if s['Recession_Indicator'] else 'Non'}")"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 33 — LGD calculation via ENRICHED satellite + mean-adjustment
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Calcul LGD par scénario via le satellite ENRICHI (mean-adjustment) ────────
# Approche : fixer les variables macro aux valeurs du scénario,
#            les variables prêt/emprunteur à leur moyenne portefeuille

# Moyennes portefeuille pour les variables non-macro
portfolio_means = X[LOAN_FEATURES + CAT_FEATURES].mean()

for name, s in scenarios.items():
    # Construire le vecteur de features complet
    x_scenario = {}
    for f in MACRO_FEATURES:
        x_scenario[f] = s[f]
    for f in LOAN_FEATURES + CAT_FEATURES:
        x_scenario[f] = portfolio_means[f]

    X_s = pd.DataFrame([x_scenario])[ALL_FEATURES]
    X_s = sm.add_constant(X_s, has_constant='add')

    # Prédiction via OLS enrichi
    logit_lgd = sat_enriched_ols.predict(X_s).values[0]
    s['LGD_logit'] = logit_lgd
    s['LGD']       = float(expit(logit_lgd))

    # Prédiction via FRM enrichi
    s['LGD_FRM']   = float(sat_frm_r.predict(X_s).values[0])

# LGD pondérée probabilisée
FL_LGD      = sum(s['LGD'] * s['Probability'] for s in scenarios.values())
FL_LGD_FRM  = sum(s['LGD_FRM'] * s['Probability'] for s in scenarios.values())
HIST_LGD    = float(y.mean())
FL_ADJ      = FL_LGD - HIST_LGD

print('=== LGD par scénario (Satellite Enrichi — Mean-Adjustment) ===')
print(f"{'Scénario':<20} {'LGD (OLS)':>10} {'LGD (FRM)':>10} {'logit':>10}")
for name, s in scenarios.items():
    print(f"  {name:<20} : {s['LGD']:.4f}     {s['LGD_FRM']:.4f}     {s['LGD_logit']:.4f}")
print(f"\\n  LGD FL pondérée (OLS)  : {FL_LGD:.4f}")
print(f"  LGD FL pondérée (FRM)  : {FL_LGD_FRM:.4f}")
print(f"  LGD historique (PiT)   : {HIST_LGD:.4f}")
print(f"  Ajustement FL          : {FL_ADJ:+.4f}")"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 34 — Scenario visualization
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Visualisation des scénarios ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Analyse Forward-Looking LGD — IFRS 9', fontsize=13, fontweight='bold')

# Panel 1 : LGD par scénario
ax = axes[0]
names  = list(scenarios.keys())
lgds   = [scenarios[n]['LGD'] for n in names]
probs  = [scenarios[n]['Probability'] for n in names]
colors = ['#2ecc71', '#e67e22', '#e74c3c']

bars = ax.bar(names, lgds, color=colors, alpha=0.85, edgecolor='black', width=0.5)
ax.axhline(FL_LGD,   color='navy',  linestyle='--', lw=2,
           label=f'FL-LGD pondérée = {FL_LGD:.4f}')
ax.axhline(HIST_LGD, color='gray',  linestyle=':',  lw=2,
           label=f'LGD historique  = {HIST_LGD:.4f}')

for bar, lgd, p in zip(bars, lgds, probs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'LGD={lgd:.4f}\\n(w={p:.0%})', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('LGD Forward-Looking')
ax.set_title('LGD par Scénario Macroéconomique')
ax.set_ylim(0, max(lgds) * 1.3)
ax.legend()

# Panel 2 : Tornado — sensibilité de la LGD à chaque variable macro (+1 std)
ax = axes[1]
base_lgd = scenarios['Baseline']['LGD']
sensitivity = {}

for feat in MACRO_FEATURES:
    if feat == 'Recession_Indicator':
        continue
    # Mean-adjustment : fixer les autres variables à la moyenne portefeuille
    base_vals = {f: scenarios['Baseline'][f] for f in MACRO_FEATURES}
    perturbed = base_vals.copy()
    perturbed[feat] = base_vals[feat] + df_enc[feat].std()

    x_pert = {}
    for f in MACRO_FEATURES:
        x_pert[f] = perturbed[f]
    for f in LOAN_FEATURES + CAT_FEATURES:
        x_pert[f] = portfolio_means[f]

    X_p = pd.DataFrame([x_pert])[ALL_FEATURES]
    X_p = sm.add_constant(X_p, has_constant='add')
    lgd_p = float(expit(sat_enriched_ols.predict(X_p).values[0]))
    sensitivity[feat] = lgd_p - base_lgd

sens = pd.Series(sensitivity).sort_values()
c_bar = ['#e74c3c' if v > 0 else '#2ecc71' for v in sens.values]
sens.plot(kind='barh', ax=ax, color=c_bar, alpha=0.85, edgecolor='black')
ax.axvline(0, color='black', lw=1)
ax.set_xlabel('ΔLGD pour +1 écart-type de la variable')
ax.set_title('Tornado — Sensibilité LGD aux variables macro\\n(choc de +1 std depuis Baseline)')

plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 35 — Section 8 header
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("markdown", """---
## Section 8 — Validation & Comparaison des Modèles

Les métriques sont calculées sur le **set de test (20%)** pour évaluer la performance hors-échantillon.
Le **ratio de surapprentissage** (RMSE train / RMSE test) permet de détecter l'overfitting."""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 36 — Metrics (TRAIN + TEST + overfitting ratio)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Métriques de validation (train + test) ────────────────────────────────────
def compute_metrics(y_true, y_pred, model_name):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))

    # Coefficient de Gini
    idx        = np.argsort(y_pred)
    lgd_sorted = y_true[idx]
    cum_lgd    = np.cumsum(lgd_sorted) / (lgd_sorted.sum() + 1e-12)
    n_         = len(y_true)
    gini       = abs(2 * cum_lgd.sum() / n_ - 1 - 1/n_)

    return {'Modèle': model_name, 'RMSE': rmse, 'MAE': mae,
            'R²': r2, 'Gini': gini, 'Biais': bias}

# Prédictions train et test
predictions_train = {
    'Tobit':         y_pred_tobit_train,
    'Beta Reg.':     y_pred_beta_train,
    'FRM':           y_pred_frm_train,
    'Random Forest': y_pred_rf_train,
    'XGBoost':       y_pred_xgb_train,
}
predictions_test = {
    'Tobit':         y_pred_tobit_test,
    'Beta Reg.':     y_pred_beta_test,
    'FRM':           y_pred_frm_test,
    'Random Forest': y_pred_rf_test,
    'XGBoost':       y_pred_xgb_test,
}

# Métriques TRAIN
metrics_train = pd.DataFrame(
    [compute_metrics(y_train, pred, name) for name, pred in predictions_train.items()]
).set_index('Modèle')

# Métriques TEST
metrics_test = pd.DataFrame(
    [compute_metrics(y_test, pred, name) for name, pred in predictions_test.items()]
).set_index('Modèle')

# Ratio de surapprentissage
overfit_ratio = (metrics_test['RMSE'] / metrics_train['RMSE']).round(3)

print('=== Métriques sur le SET DE TEST (hors-échantillon) ===')
print(metrics_test.round(5).to_string())

print('\\n=== Métriques sur le SET D\\'ENTRAÎNEMENT ===')
print(metrics_train.round(5).to_string())

print('\\n=== Ratio de surapprentissage (RMSE test / RMSE train) ===')
for name, ratio in overfit_ratio.items():
    flag = ' ⚠ OVERFITTING' if ratio > 1.3 else ''
    print(f'  {name:<15} : {ratio:.3f}{flag}')

metrics_test.to_csv('model_comparison.csv')
print('\\n→ Exporté dans model_comparison.csv')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 37 — Lorenz curves (TEST set)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Courbes de Lorenz (pouvoir discriminant — set de test) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Validation des modèles LGD (set de test)', fontsize=13, fontweight='bold')

model_colors = ['#e74c3c','#3498db','#2ecc71','#9b59b6','#f39c12']

ax = axes[0]
for (name, pred), color in zip(predictions_test.items(), model_colors):
    idx        = np.argsort(np.array(pred))
    lgd_sorted = np.array(y_test)[idx]
    cum_lgd    = np.cumsum(lgd_sorted) / lgd_sorted.sum()
    x_vals     = np.linspace(0, 1, len(cum_lgd))
    gini_val   = metrics_test.loc[name, 'Gini']
    ax.plot(x_vals, cum_lgd, color=color, lw=2,
            label=f'{name} (Gini={gini_val:.4f})')

ax.plot([0,1],[0,1],'k--', lw=1, label='Modèle aléatoire')
ax.set_xlabel('Part cumulée des observations (triées par LGD prédit)')
ax.set_ylabel('Part cumulée de la LGD réelle')
ax.set_title('Courbes de Lorenz — Pouvoir discriminant')
ax.legend(loc='upper left', fontsize=8)

# ── Comparaison RMSE / MAE / R² ──────────────────────────────────────────────
ax = axes[1]
metrics_test[['RMSE','MAE']].plot(kind='bar', ax=ax, color=['steelblue','tomato'],
                                   alpha=0.8, edgecolor='black')
ax.set_title('RMSE & MAE par modèle (test set)')
ax.set_ylabel('Erreur')
ax.set_xticklabels(metrics_test.index, rotation=20, ha='right')
ax.legend()

plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 38 — Residuals (TEST set)
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Analyse des résidus — set de test (grille 2 × 5) ──────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
fig.suptitle('Analyse des résidus par modèle (set de test)', fontsize=12, fontweight='bold')

model_colors = ['#e74c3c','#3498db','#2ecc71','#9b59b6','#f39c12']

for i, (name, pred) in enumerate(predictions_test.items()):
    resid = np.array(y_test) - np.array(pred)

    # Résidus vs fitted
    axes[0, i].scatter(pred, resid, alpha=0.3, s=10, color=model_colors[i])
    axes[0, i].axhline(0, color='red', linestyle='--', lw=1)
    axes[0, i].set_xlabel('LGD prédit'); axes[0, i].set_ylabel('Résidu')
    axes[0, i].set_title(name, fontsize=9)

    # QQ-plot
    stats.probplot(resid, plot=axes[1, i])
    axes[1, i].set_title(f'{name} — QQ', fontsize=9)

plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 39 — Section 9 header
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("markdown", """---
## Section 9 — Synthèse Réglementaire IFRS 9"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 40 — IFRS 9 final table
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Tableau final IFRS 9 ─────────────────────────────────────────────────────
rows = []
for name, s in scenarios.items():
    rows.append({
        'Scénario':      name,
        'Poids':         f"{s['Probability']:.0%}",
        'LGD scénario':  round(s['LGD'], 4),
        'LGD pondérée':  round(s['LGD'] * s['Probability'], 4),
        'Narrative':     s['Narrative']
    })

ifrs9_df = pd.DataFrame(rows)
print('=' * 75)
print('   TABLEAU IFRS 9 — LGD FORWARD-LOOKING PONDÉRÉE')
print('=' * 75)
print(ifrs9_df[['Scénario','Poids','LGD scénario','LGD pondérée']].to_string(index=False))
print('-' * 75)
print(f"{'FL-LGD Pondérée (Σ)':<35} {'':>8} {'':>13} {FL_LGD:.4f}")
print(f"{'LGD Historique PiT (Through-the-Cycle)':<35} {'':>8} {'':>13} {HIST_LGD:.4f}")
print(f"{'Ajustement Forward-Looking':<35} {'':>8} {'':>13} {FL_ADJ:+.4f}")
print('=' * 75)"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 41 — Regulatory grid
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Recommandation de modèle (grille réglementaire) ──────────────────────────
print('=== GRILLE DE SÉLECTION DE MODÈLE (Bâle III / BCBS) ===')
reco = pd.DataFrame([
    {'Modèle': 'Tobit',         'Variable bornée': 'Oui',   'Interprétable': 'Oui',     'IC disponibles': 'Non*', 'Limite principale': 'Normalité de y* requise'},
    {'Modèle': 'Régression Bêta','Variable bornée':'Oui*', 'Interprétable': 'Oui',     'IC disponibles': 'Oui',  'Limite principale': 'Échoue sans transfo SV'},
    {'Modèle': 'FRM',            'Variable bornée':'Oui',   'Interprétable': 'Oui',     'IC disponibles': 'Oui',  'Limite principale': 'Variance non modélisée'},
    {'Modèle': 'Random Forest',  'Variable bornée':'Oui**','Interprétable': 'Partiel', 'IC disponibles': 'Non',  'Limite principale': "Boîte noire, pas d'IC"},
    {'Modèle': 'XGBoost',        'Variable bornée':'Oui**','Interprétable': 'SHAP',    'IC disponibles': 'Non',  'Limite principale': 'Surapprentissage n=500'},
])
print(reco.to_string(index=False))
print('\\n* Avec transformation Smithson-Verkuilen')
print('** Après clipping sur [0,1]')
print('*  Tobit IC via Hessienne numérique (Section 5.1)')
print()
print('RECOMMANDATION MODÈLE PRINCIPAL : Régression Bêta ou FRM')
print('  → Fondement théorique solide pour y ∈ (0,1)')
print('  → Coefficients interprétables + intervalles de confiance')
print('  → Exigés par le BCBS (transparence du modèle)')
print()
print('RECOMMANDATION BENCHMARK     : XGBoost + SHAP')
print('  → Comparer les prédictions; documenter toute divergence significative')"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 42 — Final synthesis figure
# ═══════════════════════════════════════════════════════════════════════════
cells.append(cell("code", """# ── Figure de synthèse finale ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Synthèse Réglementaire — Modélisation Forward-Looking LGD (IFRS 9 / Bâle III)',
             fontsize=12, fontweight='bold')

# Panel 1 : LGD FL pondérée vs historique
ax = axes[0]
vals  = [HIST_LGD, FL_LGD]
lbls  = ['LGD\\nHistorique\\n(PiT)', 'LGD\\nForward-Looking\\n(IFRS 9)']
cols  = ['#95a5a6', '#2980b9']
bars  = ax.bar(lbls, vals, color=cols, alpha=0.85, edgecolor='black', width=0.4)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f'{v:.4f}',
            ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('LGD'); ax.set_ylim(0, max(vals)*1.25)
ax.set_title('Ajustement Forward-Looking\\n' + f'Δ = {FL_ADJ:+.4f}')

# Panel 2 : Poids et LGD par scénario
ax = axes[1]
sc_names = list(scenarios.keys())
sc_lgds  = [scenarios[n]['LGD'] for n in sc_names]
sc_probs = [scenarios[n]['Probability'] for n in sc_names]
sc_cols  = ['#2ecc71','#e67e22','#e74c3c']
bars2    = ax.bar(sc_names, sc_lgds, color=sc_cols, alpha=0.85, edgecolor='black', width=0.5)
ax.axhline(FL_LGD, color='navy', linestyle='--', lw=2, label=f'FL-LGD = {FL_LGD:.4f}')
for bar, lgd, p in zip(bars2, sc_lgds, sc_probs):
    ax.text(bar.get_x() + bar.get_width()/2, lgd + 0.002,
            f'{lgd:.4f}\\n({p:.0%})', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('LGD scénario'); ax.set_ylim(0, max(sc_lgds)*1.3)
ax.set_title('LGD par Scénario (poids IFRS 9)'); ax.legend()

# Panel 3 : Comparaison des performances
ax = axes[2]
model_names = metrics_test.index.tolist()
r2_vals     = metrics_test['R²'].clip(lower=0).values
gini_vals   = metrics_test['Gini'].values
rmse_inv    = (1 - metrics_test['RMSE'] / metrics_test['RMSE'].max()).values

x = np.arange(len(model_names))
w = 0.25
ax.bar(x - w, r2_vals,    width=w, label='R²',       color='steelblue',  alpha=0.8)
ax.bar(x,     gini_vals,  width=w, label='Gini',     color='seagreen',   alpha=0.8)
ax.bar(x + w, rmse_inv,   width=w, label='1-RMSE/max', color='tomato',   alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=8)
ax.set_ylabel('Score (plus haut = meilleur)')
ax.set_title('Comparaison des performances (test set)\\n(métriques normalisées)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

print('\\n✓ Notebook v2 complet — Forward-Looking LGD modélisé avec succès.')
print('  Améliorations v2 : train/test split, CV 5-fold tous modèles,')
print('  satellite enrichi, Tobit coefficients, tests formels, scénarios corrigés.')"""))

# ═══════════════════════════════════════════════════════════════════════════
# Assemble notebook
# ═══════════════════════════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('lgd_forward_looking.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook v2 écrit avec succès : {len(cells)} cellules")
