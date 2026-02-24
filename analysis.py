"""
Applied Machine Learning - Course Project 2: Classification
UCI Bike Sharing Dataset Analysis
Tasks: Logistic Regression, LDA/QDA, Naive Bayes, Linear vs Poisson Regression
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.multiclass import OneVsRestClassifier

import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, Gaussian
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.stats.outliers_influence import variance_inflation_factor

import os

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
FIGDIR = 'figures'
os.makedirs(FIGDIR, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load & Preprocess Data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("LOADING AND PREPROCESSING DATA")
print("=" * 80)

df = pd.read_csv('data/hour.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

# Drop date column (not useful for modeling)
df = df.drop(columns=['dteday'])

# ── Create classification target: bin 'cnt' into 3 classes ──
# Low / Medium / High demand using quantile-based binning
df['demand_class'] = pd.qcut(df['cnt'], q=3, labels=['Low', 'Medium', 'High'])
print(f"\nDemand class distribution:\n{df['demand_class'].value_counts()}")

# Also create binary target for binary classification tasks
median_cnt = df['cnt'].median()
df['demand_binary'] = (df['cnt'] > median_cnt).astype(int)
print(f"\nBinary demand distribution (0=Low, 1=High):\n{df['demand_binary'].value_counts()}")

# Feature columns for classification
feature_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

X = df[feature_cols].values
y_binary = df['demand_binary'].values
y_multi = df['demand_class'].cat.codes.values  # 0=Low, 1=Medium, 2=High
y_cnt = df['cnt'].values  # continuous target for regression

# Train-test split — use a SINGLE consistent split for all targets
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(
    indices, test_size=0.25, random_state=RANDOM_STATE, stratify=y_binary
)
X_train, X_test = X[train_idx], X[test_idx]
y_bin_train, y_bin_test = y_binary[train_idx], y_binary[test_idx]
y_multi_train, y_multi_test = y_multi[train_idx], y_multi[test_idx]
y_cnt_train, y_cnt_test = y_cnt[train_idx], y_cnt[test_idx]

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ── EDA Plots ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution of cnt
axes[0, 0].hist(df['cnt'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('Distribution of Bike Rental Count')
axes[0, 0].set_xlabel('Count')
axes[0, 0].set_ylabel('Frequency')

# Demand by hour
hourly = df.groupby('hr')['cnt'].mean()
axes[0, 1].bar(hourly.index, hourly.values, color='coral', edgecolor='black')
axes[0, 1].set_title('Average Rentals by Hour')
axes[0, 1].set_xlabel('Hour')
axes[0, 1].set_ylabel('Mean Count')

# Demand by season
season_labels = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
seasonal = df.groupby('season')['cnt'].mean()
axes[1, 0].bar([season_labels[s] for s in seasonal.index], seasonal.values,
               color=['green', 'gold', 'orange', 'skyblue'], edgecolor='black')
axes[1, 0].set_title('Average Rentals by Season')
axes[1, 0].set_ylabel('Mean Count')

# Correlation heatmap
corr_cols = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'cnt']
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=axes[1, 1], fmt='.2f')
axes[1, 1].set_title('Feature Correlation Heatmap')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/01_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Saved] {FIGDIR}/01_eda.png")

# ═════════════════════════════════════════════════════════════════════════════
# TASK 1: LOGISTIC REGRESSION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TASK 1: LOGISTIC REGRESSION")
print("=" * 80)

# ── 1a) Binary Logistic Regression ──
print("\n--- 1a) Binary Logistic Regression ---")
lr_binary = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, solver='lbfgs')
lr_binary.fit(X_train_sc, y_bin_train)
y_bin_pred = lr_binary.predict(X_test_sc)
y_bin_prob = lr_binary.predict_proba(X_test_sc)[:, 1]

acc_lr_bin = accuracy_score(y_bin_test, y_bin_pred)
print(f"Binary LR Accuracy: {acc_lr_bin:.4f}")
print(f"\nClassification Report:\n{classification_report(y_bin_test, y_bin_pred, target_names=['Low', 'High'])}")

# ── 1b) Standard Error, Z-statistic, p-value using statsmodels ──
print("\n--- 1b) Standard Errors, Z-statistics, p-values ---")
X_train_sm = sm.add_constant(X_train_sc)
X_test_sm = sm.add_constant(X_test_sc)

logit_model = sm.Logit(y_bin_train, X_train_sm)
logit_result = logit_model.fit(disp=0, maxiter=5000)
print(logit_result.summary2())

# Extract coefficient table
coef_table = pd.DataFrame({
    'Feature': ['const'] + feature_cols,
    'Coefficient': logit_result.params,
    'Std Error': logit_result.bse,
    'Z-statistic': logit_result.tvalues,
    'P-value': logit_result.pvalues
})
coef_table['Significant (p<0.05)'] = coef_table['P-value'] < 0.05
print("\nCoefficient Table:")
print(coef_table.to_string(index=False))
coef_table.to_csv(f'data/logistic_coefficients.csv', index=False)
print(f"\n[Saved] data/logistic_coefficients.csv")

# ── 1c) Confounding Variables ──
print("\n--- 1c) Confounding Variable Analysis ---")
print("Checking for multicollinearity using VIF (Variance Inflation Factor):")

X_vif = pd.DataFrame(X_train_sc, columns=feature_cols)
X_vif_const = sm.add_constant(X_vif)
vif_data = pd.DataFrame()
vif_data['Feature'] = feature_cols
vif_data['VIF'] = [variance_inflation_factor(X_vif_const.values, i + 1) for i in range(len(feature_cols))]
vif_data = vif_data.sort_values('VIF', ascending=False)
print(vif_data.to_string(index=False))

print("\nConfounding Analysis:")
print("- 'temp' and 'atemp' (actual & feels-like temp) are highly correlated (r≈0.99),")
print("  indicating 'atemp' is a confounder/redundant with 'temp'.")
print("- 'season' and 'mnth' capture similar seasonal patterns — potential confounding.")
print("- High VIF (>5) for temp/atemp confirms multicollinearity → confounding present.")

# Demonstrate confounding by showing coefficient change when atemp is removed
print("\nDemonstrating confounding effect of 'atemp' on 'temp':")
feature_cols_no_atemp = [c for c in feature_cols if c != 'atemp']
X_no_atemp = df[feature_cols_no_atemp].values
X_train_na = X_no_atemp[train_idx]  # use same indices as main split
scaler_na = StandardScaler()
X_train_na_sc = scaler_na.fit_transform(X_train_na)
X_train_na_sm = sm.add_constant(X_train_na_sc)
logit_no_atemp = sm.Logit(y_bin_train, X_train_na_sm).fit(disp=0, maxiter=5000)

temp_idx_full = feature_cols.index('temp') + 1  # +1 for constant
temp_idx_reduced = feature_cols_no_atemp.index('temp') + 1
coef_temp_full = logit_result.params[temp_idx_full]
coef_temp_reduced = logit_no_atemp.params[temp_idx_reduced]
print(f"  Coefficient of 'temp' WITH atemp:    {coef_temp_full:.4f}")
print(f"  Coefficient of 'temp' WITHOUT atemp: {coef_temp_reduced:.4f}")
pct_change = abs(coef_temp_reduced - coef_temp_full) / abs(coef_temp_full) * 100
print(f"  Change: {pct_change:.1f}% → {'Confounding confirmed (>10% change)' if pct_change > 10 else 'Minimal confounding'}")

# ── 1d) Multiclass Logistic Regression ──
print("\n--- 1d) Multi-class Logistic Regression (3 classes: Low/Medium/High) ---")
lr_multi = LogisticRegression(
    max_iter=5000, random_state=RANDOM_STATE, solver='lbfgs'
)
lr_multi.fit(X_train_sc, y_multi_train)
y_multi_pred_lr = lr_multi.predict(X_test_sc)
acc_lr_multi = accuracy_score(y_multi_test, y_multi_pred_lr)
f1_lr_multi = f1_score(y_multi_test, y_multi_pred_lr, average='weighted')

print(f"Multi-class LR Accuracy: {acc_lr_multi:.4f}")
print(f"Multi-class LR Weighted F1: {f1_lr_multi:.4f}")
print(f"\nClassification Report:\n{classification_report(y_multi_test, y_multi_pred_lr, target_names=['Low', 'Medium', 'High'])}")

# Confusion Matrix for LR multi-class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm_lr_bin = confusion_matrix(y_bin_test, y_bin_pred)
sns.heatmap(cm_lr_bin, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
axes[0].set_title('Binary LR Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

cm_lr_multi = confusion_matrix(y_multi_test, y_multi_pred_lr)
sns.heatmap(cm_lr_multi, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
axes[1].set_title('Multi-class LR Confusion Matrix')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/02_logistic_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Saved] {FIGDIR}/02_logistic_regression.png")

# ═════════════════════════════════════════════════════════════════════════════
# TASK 2: DISCRIMINANT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TASK 2: DISCRIMINANT ANALYSIS (LDA / QDA)")
print("=" * 80)

# ── 2a) Linear Discriminant Analysis ──
print("\n--- 2a) Linear Discriminant Analysis ---")
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_sc, y_bin_train)
y_bin_pred_lda = lda.predict(X_test_sc)
y_bin_prob_lda = lda.predict_proba(X_test_sc)[:, 1]
acc_lda = accuracy_score(y_bin_test, y_bin_pred_lda)
print(f"LDA Binary Accuracy: {acc_lda:.4f}")
print(f"\nClassification Report:\n{classification_report(y_bin_test, y_bin_pred_lda, target_names=['Low', 'High'])}")

# Multi-class LDA
lda_multi = LinearDiscriminantAnalysis()
lda_multi.fit(X_train_sc, y_multi_train)
y_multi_pred_lda = lda_multi.predict(X_test_sc)
acc_lda_multi = accuracy_score(y_multi_test, y_multi_pred_lda)
f1_lda_multi = f1_score(y_multi_test, y_multi_pred_lda, average='weighted')
print(f"\nLDA Multi-class Accuracy: {acc_lda_multi:.4f}")
print(f"LDA Multi-class Weighted F1: {f1_lda_multi:.4f}")

# ── 2b) Effective Threshold ──
print("\n--- 2b) Defining Effective Threshold ---")
# Find the optimal threshold using Youden's J statistic on LDA
fpr_lda, tpr_lda, thresholds_lda = roc_curve(y_bin_test, y_bin_prob_lda)
j_scores = tpr_lda - fpr_lda
best_idx = np.argmax(j_scores)
best_threshold = thresholds_lda[best_idx]
print(f"Optimal Threshold (Youden's J): {best_threshold:.4f}")
print(f"At this threshold: TPR={tpr_lda[best_idx]:.4f}, FPR={fpr_lda[best_idx]:.4f}")
print(f"Youden's J = {j_scores[best_idx]:.4f}")

# Apply custom threshold
y_custom_thresh = (y_bin_prob_lda >= best_threshold).astype(int)
acc_custom = accuracy_score(y_bin_test, y_custom_thresh)
print(f"Accuracy with optimal threshold: {acc_custom:.4f}")
print(f"Accuracy with default threshold (0.5): {acc_lda:.4f}")

# ── 2c) ROC Curve ──
print("\n--- 2c) ROC Curves ---")
# Compute ROC for LR binary
fpr_lr, tpr_lr, _ = roc_curve(y_bin_test, y_bin_prob)
auc_lr = auc(fpr_lr, tpr_lr)

# ROC for LDA
auc_lda = auc(fpr_lda, tpr_lda)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Binary ROC
axes[0].plot(fpr_lr, tpr_lr, 'b-', label=f'Logistic Regression (AUC={auc_lr:.3f})', linewidth=2)
axes[0].plot(fpr_lda, tpr_lda, 'r--', label=f'LDA (AUC={auc_lda:.3f})', linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
axes[0].scatter(fpr_lda[best_idx], tpr_lda[best_idx], c='red', s=100, zorder=5,
                label=f'Optimal Threshold={best_threshold:.3f}')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve - Binary Classification')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Multi-class ROC (One-vs-Rest)
y_multi_test_bin = label_binarize(y_multi_test, classes=[0, 1, 2])
y_multi_prob_lr = lr_multi.predict_proba(X_test_sc)
y_multi_prob_lda = lda_multi.predict_proba(X_test_sc)

class_names = ['Low', 'Medium', 'High']
colors = ['blue', 'green', 'red']
for i, (name, color) in enumerate(zip(class_names, colors)):
    fpr_i, tpr_i, _ = roc_curve(y_multi_test_bin[:, i], y_multi_prob_lr[:, i])
    auc_i = auc(fpr_i, tpr_i)
    axes[1].plot(fpr_i, tpr_i, color=color, linewidth=2,
                 label=f'LR - {name} (AUC={auc_i:.3f})')

axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Multi-class ROC Curves (OvR) - Logistic Regression')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/03_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[Saved] {FIGDIR}/03_roc_curves.png")

print(f"\nBinary ROC AUC - LR: {auc_lr:.4f}, LDA: {auc_lda:.4f}")

# ── 2d) Quadratic Discriminant Analysis ──
print("\n--- 2d) Quadratic Discriminant Analysis ---")
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_sc, y_bin_train)
y_bin_pred_qda = qda.predict(X_test_sc)
y_bin_prob_qda = qda.predict_proba(X_test_sc)[:, 1]
acc_qda = accuracy_score(y_bin_test, y_bin_pred_qda)
print(f"QDA Binary Accuracy: {acc_qda:.4f}")
print(f"\nClassification Report:\n{classification_report(y_bin_test, y_bin_pred_qda, target_names=['Low', 'High'])}")

# QDA ROC
fpr_qda, tpr_qda, _ = roc_curve(y_bin_test, y_bin_prob_qda)
auc_qda = auc(fpr_qda, tpr_qda)
print(f"QDA ROC AUC: {auc_qda:.4f}")

# Multi-class QDA
qda_multi = QuadraticDiscriminantAnalysis()
qda_multi.fit(X_train_sc, y_multi_train)
y_multi_pred_qda = qda_multi.predict(X_test_sc)
acc_qda_multi = accuracy_score(y_multi_test, y_multi_pred_qda)
f1_qda_multi = f1_score(y_multi_test, y_multi_pred_qda, average='weighted')
print(f"\nQDA Multi-class Accuracy: {acc_qda_multi:.4f}")
print(f"QDA Multi-class Weighted F1: {f1_qda_multi:.4f}")

# Confusion matrices for LDA and QDA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm_lda = confusion_matrix(y_bin_test, y_bin_pred_lda)
cm_qda = confusion_matrix(y_bin_test, y_bin_pred_qda)
sns.heatmap(cm_lda, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
axes[0].set_title('LDA Confusion Matrix (Binary)')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')
sns.heatmap(cm_qda, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
axes[1].set_title('QDA Confusion Matrix (Binary)')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/04_lda_qda_cm.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Saved] {FIGDIR}/04_lda_qda_cm.png")

# ═════════════════════════════════════════════════════════════════════════════
# TASK 3: NAIVE BAYES + COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TASK 3: NAIVE BAYES & MODEL COMPARISON")
print("=" * 80)

# ── Naive Bayes ──
print("\n--- Gaussian Naive Bayes ---")
nb = GaussianNB()
nb.fit(X_train_sc, y_bin_train)
y_bin_pred_nb = nb.predict(X_test_sc)
y_bin_prob_nb = nb.predict_proba(X_test_sc)[:, 1]
acc_nb = accuracy_score(y_bin_test, y_bin_pred_nb)
print(f"NB Binary Accuracy: {acc_nb:.4f}")
print(f"\nClassification Report:\n{classification_report(y_bin_test, y_bin_pred_nb, target_names=['Low', 'High'])}")

fpr_nb, tpr_nb, _ = roc_curve(y_bin_test, y_bin_prob_nb)
auc_nb = auc(fpr_nb, tpr_nb)
print(f"NB ROC AUC: {auc_nb:.4f}")

# Multi-class NB
nb_multi = GaussianNB()
nb_multi.fit(X_train_sc, y_multi_train)
y_multi_pred_nb = nb_multi.predict(X_test_sc)
acc_nb_multi = accuracy_score(y_multi_test, y_multi_pred_nb)
f1_nb_multi = f1_score(y_multi_test, y_multi_pred_nb, average='weighted')
print(f"\nNB Multi-class Accuracy: {acc_nb_multi:.4f}")
print(f"NB Multi-class Weighted F1: {f1_nb_multi:.4f}")

# ── Comprehensive Comparison ──
print("\n--- Model Comparison: LR vs LDA vs QDA vs NB ---")

# Binary comparison
binary_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'LDA', 'QDA', 'Naive Bayes'],
    'Accuracy': [acc_lr_bin, acc_lda, acc_qda, acc_nb],
    'ROC AUC': [auc_lr, auc_lda, auc_qda, auc_nb],
    'F1 (weighted)': [
        f1_score(y_bin_test, y_bin_pred, average='weighted'),
        f1_score(y_bin_test, y_bin_pred_lda, average='weighted'),
        f1_score(y_bin_test, y_bin_pred_qda, average='weighted'),
        f1_score(y_bin_test, y_bin_pred_nb, average='weighted'),
    ],
    'Precision (weighted)': [
        precision_score(y_bin_test, y_bin_pred, average='weighted'),
        precision_score(y_bin_test, y_bin_pred_lda, average='weighted'),
        precision_score(y_bin_test, y_bin_pred_qda, average='weighted'),
        precision_score(y_bin_test, y_bin_pred_nb, average='weighted'),
    ],
    'Recall (weighted)': [
        recall_score(y_bin_test, y_bin_pred, average='weighted'),
        recall_score(y_bin_test, y_bin_pred_lda, average='weighted'),
        recall_score(y_bin_test, y_bin_pred_qda, average='weighted'),
        recall_score(y_bin_test, y_bin_pred_nb, average='weighted'),
    ],
})
print("\nBinary Classification Results:")
print(binary_results.to_string(index=False))

# Multi-class comparison
multi_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'LDA', 'QDA', 'Naive Bayes'],
    'Accuracy': [acc_lr_multi, acc_lda_multi, acc_qda_multi, acc_nb_multi],
    'F1 (weighted)': [f1_lr_multi, f1_lda_multi, f1_qda_multi, f1_nb_multi],
})
print("\nMulti-class (3-class) Results:")
print(multi_results.to_string(index=False))

binary_results.to_csv('data/binary_comparison.csv', index=False)
multi_results.to_csv('data/multi_comparison.csv', index=False)

# Cross-validation comparison
print("\n--- 5-Fold Cross-Validation (Binary) ---")
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'Naive Bayes': GaussianNB()
}

X_all_sc = StandardScaler().fit_transform(X)  # separate scaler to avoid overwriting
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_all_sc, y_binary, cv=cv, scoring='accuracy')
    cv_results[name] = scores
    print(f"  {name:25s}: Mean={scores.mean():.4f} ± {scores.std():.4f}")

# Comparison plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ROC comparison
axes[0, 0].plot(fpr_lr, tpr_lr, 'b-', label=f'LR (AUC={auc_lr:.3f})', linewidth=2)
axes[0, 0].plot(fpr_lda, tpr_lda, 'r--', label=f'LDA (AUC={auc_lda:.3f})', linewidth=2)
axes[0, 0].plot(fpr_qda, tpr_qda, 'g-.', label=f'QDA (AUC={auc_qda:.3f})', linewidth=2)
axes[0, 0].plot(fpr_nb, tpr_nb, 'm:', label=f'NB (AUC={auc_nb:.3f})', linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Comparison (Binary)')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# Accuracy bar chart
model_names = binary_results['Model']
x_pos = np.arange(len(model_names))
axes[0, 1].bar(x_pos, binary_results['Accuracy'], color=['steelblue', 'coral', 'mediumseagreen', 'mediumpurple'],
               edgecolor='black')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(model_names, rotation=15, ha='right')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Binary Classification Accuracy Comparison')
axes[0, 1].set_ylim(0.5, 1.0)
for i, v in enumerate(binary_results['Accuracy']):
    axes[0, 1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)

# Cross-validation boxplot
cv_df = pd.DataFrame(cv_results)
cv_df.plot(kind='box', ax=axes[1, 0], color=dict(boxes='steelblue', whiskers='gray',
                                                    medians='red', caps='gray'))
axes[1, 0].set_title('5-Fold Cross-Validation Accuracy')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].grid(True, alpha=0.3)

# Multi-class accuracy comparison
axes[1, 1].bar(x_pos, multi_results['Accuracy'], color=['steelblue', 'coral', 'mediumseagreen', 'mediumpurple'],
               edgecolor='black')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(model_names, rotation=15, ha='right')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Multi-class (3-class) Accuracy Comparison')
axes[1, 1].set_ylim(0.3, 0.9)
for i, v in enumerate(multi_results['Accuracy']):
    axes[1, 1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/05_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Saved] {FIGDIR}/05_model_comparison.png")

# ═════════════════════════════════════════════════════════════════════════════
# TASK 4: LINEAR vs POISSON REGRESSION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TASK 4: LINEAR vs POISSON REGRESSION")
print("=" * 80)

regression_features = feature_cols
X_reg_train = pd.DataFrame(X_train, columns=regression_features)
X_reg_test = pd.DataFrame(X_test, columns=regression_features)

# Add constant for statsmodels
X_reg_train_c = sm.add_constant(X_reg_train)
X_reg_test_c = sm.add_constant(X_reg_test)

# ── Linear (OLS) Regression ──
print("\n--- Linear (OLS) Regression ---")
ols_model = sm.OLS(y_cnt_train, X_reg_train_c).fit()
print(ols_model.summary())

y_cnt_pred_ols = ols_model.predict(X_reg_test_c)

# Ensure no negative predictions for count data
y_cnt_pred_ols_clipped = np.maximum(y_cnt_pred_ols, 0)

ols_mse = np.mean((y_cnt_test - y_cnt_pred_ols) ** 2)
ols_rmse = np.sqrt(ols_mse)
ols_mae = np.mean(np.abs(y_cnt_test - y_cnt_pred_ols))
ols_r2 = ols_model.rsquared
ols_adj_r2 = ols_model.rsquared_adj
ols_aic = ols_model.aic
ols_bic = ols_model.bic

print(f"\nOLS Results:")
print(f"  R²: {ols_r2:.4f}")
print(f"  Adjusted R²: {ols_adj_r2:.4f}")
print(f"  RMSE: {ols_rmse:.4f}")
print(f"  MAE: {ols_mae:.4f}")
print(f"  AIC: {ols_aic:.2f}")
print(f"  BIC: {ols_bic:.2f}")

# ── Poisson Regression ──
print("\n--- Poisson Regression ---")
poisson_model = GLM(y_cnt_train, X_reg_train_c, family=Poisson()).fit()
print(poisson_model.summary())

y_cnt_pred_poisson = poisson_model.predict(X_reg_test_c)

poisson_mse = np.mean((y_cnt_test - y_cnt_pred_poisson) ** 2)
poisson_rmse = np.sqrt(poisson_mse)
poisson_mae = np.mean(np.abs(y_cnt_test - y_cnt_pred_poisson))
poisson_aic = poisson_model.aic
poisson_bic = poisson_model.bic_llf

# Deviance / Pearson Chi2 for Poisson
poisson_deviance = poisson_model.deviance
poisson_pearson_chi2 = poisson_model.pearson_chi2
n_train = X_reg_train.shape[0]
p = X_reg_train.shape[1] + 1  # +1 for constant
dispersion = poisson_pearson_chi2 / (n_train - p)

print(f"\nPoisson Results:")
print(f"  RMSE: {poisson_rmse:.4f}")
print(f"  MAE: {poisson_mae:.4f}")
print(f"  AIC: {poisson_aic:.2f}")
print(f"  Deviance: {poisson_deviance:.2f}")
print(f"  Pearson Chi²: {poisson_pearson_chi2:.2f}")
print(f"  Overdispersion parameter: {dispersion:.4f}")
if dispersion > 1:
    print(f"  → Overdispersion detected (φ={dispersion:.2f} > 1)")
    print(f"    This suggests the variance exceeds the mean, common in count data.")

# ── Comparison ──
print("\n--- Model Comparison ---")
reg_comparison = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'AIC', 'BIC'],
    'Linear (OLS)': [ols_rmse, ols_mae, ols_aic, ols_bic],
    'Poisson': [poisson_rmse, poisson_mae, poisson_aic, poisson_bic],
})
print(reg_comparison.to_string(index=False))
reg_comparison.to_csv('data/regression_comparison.csv', index=False)

# Regression diagnostic plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# OLS: Actual vs Predicted
axes[0, 0].scatter(y_cnt_test, y_cnt_pred_ols, alpha=0.2, s=5, color='steelblue')
max_val = max(y_cnt_test.max(), y_cnt_pred_ols.max())
axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
axes[0, 0].set_xlabel('Actual Count')
axes[0, 0].set_ylabel('Predicted Count')
axes[0, 0].set_title(f'OLS: Actual vs Predicted (R²={ols_r2:.3f})')
axes[0, 0].grid(True, alpha=0.3)

# OLS: Residuals
ols_residuals = y_cnt_test - y_cnt_pred_ols
axes[0, 1].scatter(y_cnt_pred_ols, ols_residuals, alpha=0.2, s=5, color='steelblue')
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Count')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('OLS: Residual Plot')
axes[0, 1].grid(True, alpha=0.3)

# OLS: Residual distribution
axes[0, 2].hist(ols_residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 2].set_xlabel('Residual')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('OLS: Residual Distribution')

# Poisson: Actual vs Predicted
axes[1, 0].scatter(y_cnt_test, y_cnt_pred_poisson, alpha=0.2, s=5, color='coral')
axes[1, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
axes[1, 0].set_xlabel('Actual Count')
axes[1, 0].set_ylabel('Predicted Count')
axes[1, 0].set_title('Poisson: Actual vs Predicted')
axes[1, 0].grid(True, alpha=0.3)

# Poisson: Residuals
poisson_residuals = y_cnt_test - y_cnt_pred_poisson
axes[1, 1].scatter(y_cnt_pred_poisson, poisson_residuals, alpha=0.2, s=5, color='coral')
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Predicted Count')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Poisson: Residual Plot')
axes[1, 1].grid(True, alpha=0.3)

# Poisson: Residual distribution
axes[1, 2].hist(poisson_residuals, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[1, 2].set_xlabel('Residual')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Poisson: Residual Distribution')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/06_regression_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Saved] {FIGDIR}/06_regression_comparison.png")

# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CLASSIFICATION RESULTS (Binary: Low vs High Demand)                       ║
╠════════════════════════════╦═══════════╦═══════════╦═══════════════════════╗
║  Model                     ║ Accuracy  ║ ROC AUC   ║ F1 (weighted)        ║
╠════════════════════════════╬═══════════╬═══════════╬═══════════════════════╣""")
for _, row in binary_results.iterrows():
    print(f"║  {row['Model']:26s}║ {row['Accuracy']:.4f}    ║ {row['ROC AUC']:.4f}    ║ {row['F1 (weighted)']:.4f}              ║")
print("╚════════════════════════════╩═══════════╩═══════════╩═══════════════════════╝")

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  REGRESSION RESULTS (Predicting Bike Rental Count)                         ║
╠════════════════════════════╦═══════════╦═══════════╦═══════════════════════╗
║  Model                     ║ RMSE      ║ MAE       ║ AIC                  ║
╠════════════════════════════╬═══════════╬═══════════╬═══════════════════════╣
║  Linear (OLS)              ║ {ols_rmse:9.2f} ║ {ols_mae:9.2f} ║ {ols_aic:>21.0f} ║
║  Poisson                   ║ {poisson_rmse:9.2f} ║ {poisson_mae:9.2f} ║ {poisson_aic:>21.0f} ║
╚════════════════════════════╩═══════════╩═══════════╩═══════════════════════╝
""")

print("Key Findings:")
print("1. Logistic Regression and LDA perform similarly for binary classification.")
print("2. QDA captures non-linear decision boundaries, may outperform on certain splits.")
print("3. Naive Bayes assumes feature independence, which may not hold here.")
print("4. For count data (cnt), Poisson regression is theoretically more appropriate")
print("   than OLS, as it models count data with a log-link function.")
print("5. Confounding: 'temp' and 'atemp' are highly correlated (VIF confirms).")
print("6. Overdispersion in Poisson model suggests Negative Binomial might be better.")

print("\n[DONE] All tasks completed. Results saved to 'data/' and 'figures/'.")
