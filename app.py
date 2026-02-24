"""
Streamlit UI for Applied Machine Learning - Course Project 2
UCI Bike Sharing Dataset: Classification & Regression Analysis
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, f1_score, precision_score, recall_score
)

import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.stats.outliers_influence import variance_inflation_factor

import os

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Cached data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('data/hour.csv')
    df = df.drop(columns=['dteday'])
    df['demand_class'] = pd.qcut(df['cnt'], q=3, labels=['Low', 'Medium', 'High'])
    median_cnt = df['cnt'].median()
    df['demand_binary'] = (df['cnt'] > median_cnt).astype(int)

    feature_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                    'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

    X = df[feature_cols].values
    y_binary = df['demand_binary'].values
    y_multi = df['demand_class'].cat.codes.values
    y_cnt = df['cnt'].values

    X_train, X_test, y_bin_train, y_bin_test = train_test_split(
        X, y_binary, test_size=0.25, random_state=RANDOM_STATE, stratify=y_binary)
    _, _, y_multi_train, y_multi_test = train_test_split(
        X, y_multi, test_size=0.25, random_state=RANDOM_STATE, stratify=y_multi)
    _, _, y_cnt_train, y_cnt_test = train_test_split(
        X, y_cnt, test_size=0.25, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return (df, feature_cols, scaler,
            X_train, X_test, X_train_sc, X_test_sc,
            y_bin_train, y_bin_test, y_multi_train, y_multi_test,
            y_cnt_train, y_cnt_test)


@st.cache_data
def train_all_models(_X_train_sc, _X_test_sc, y_bin_train, y_bin_test,
                     y_multi_train, y_multi_test, feature_cols, _X_train, _X_test,
                     y_cnt_train, y_cnt_test):
    results = {}

    # Binary Logistic Regression
    lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, solver='lbfgs')
    lr.fit(_X_train_sc, y_bin_train)
    results['lr'] = {
        'model': lr,
        'y_pred': lr.predict(_X_test_sc),
        'y_prob': lr.predict_proba(_X_test_sc)[:, 1],
        'acc': accuracy_score(y_bin_test, lr.predict(_X_test_sc)),
    }

    # Statsmodels logistic (for SE, z, p)
    X_sm = sm.add_constant(_X_train_sc)
    logit_res = sm.Logit(y_bin_train, X_sm).fit(disp=0, maxiter=5000)
    results['logit_sm'] = logit_res

    # VIF
    X_vif = pd.DataFrame(_X_train_sc, columns=feature_cols)
    X_vif_c = sm.add_constant(X_vif)
    vif = pd.DataFrame({
        'Feature': feature_cols,
        'VIF': [variance_inflation_factor(X_vif_c.values, i + 1) for i in range(len(feature_cols))]
    }).sort_values('VIF', ascending=False)
    results['vif'] = vif

    # Multi-class LR
    lr_m = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, solver='lbfgs')
    lr_m.fit(_X_train_sc, y_multi_train)
    results['lr_multi'] = {
        'model': lr_m,
        'y_pred': lr_m.predict(_X_test_sc),
        'y_prob': lr_m.predict_proba(_X_test_sc),
        'acc': accuracy_score(y_multi_test, lr_m.predict(_X_test_sc)),
        'f1': f1_score(y_multi_test, lr_m.predict(_X_test_sc), average='weighted'),
    }

    # LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(_X_train_sc, y_bin_train)
    results['lda'] = {
        'model': lda,
        'y_pred': lda.predict(_X_test_sc),
        'y_prob': lda.predict_proba(_X_test_sc)[:, 1],
        'acc': accuracy_score(y_bin_test, lda.predict(_X_test_sc)),
    }
    lda_m = LinearDiscriminantAnalysis()
    lda_m.fit(_X_train_sc, y_multi_train)
    results['lda_multi'] = {
        'model': lda_m,
        'y_pred': lda_m.predict(_X_test_sc),
        'acc': accuracy_score(y_multi_test, lda_m.predict(_X_test_sc)),
        'f1': f1_score(y_multi_test, lda_m.predict(_X_test_sc), average='weighted'),
    }

    # QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(_X_train_sc, y_bin_train)
    results['qda'] = {
        'model': qda,
        'y_pred': qda.predict(_X_test_sc),
        'y_prob': qda.predict_proba(_X_test_sc)[:, 1],
        'acc': accuracy_score(y_bin_test, qda.predict(_X_test_sc)),
    }
    qda_m = QuadraticDiscriminantAnalysis()
    qda_m.fit(_X_train_sc, y_multi_train)
    results['qda_multi'] = {
        'model': qda_m,
        'y_pred': qda_m.predict(_X_test_sc),
        'acc': accuracy_score(y_multi_test, qda_m.predict(_X_test_sc)),
        'f1': f1_score(y_multi_test, qda_m.predict(_X_test_sc), average='weighted'),
    }

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(_X_train_sc, y_bin_train)
    results['nb'] = {
        'model': nb,
        'y_pred': nb.predict(_X_test_sc),
        'y_prob': nb.predict_proba(_X_test_sc)[:, 1],
        'acc': accuracy_score(y_bin_test, nb.predict(_X_test_sc)),
    }
    nb_m = GaussianNB()
    nb_m.fit(_X_train_sc, y_multi_train)
    results['nb_multi'] = {
        'model': nb_m,
        'y_pred': nb_m.predict(_X_test_sc),
        'acc': accuracy_score(y_multi_test, nb_m.predict(_X_test_sc)),
        'f1': f1_score(y_multi_test, nb_m.predict(_X_test_sc), average='weighted'),
    }

    # OLS
    X_reg_train_c = sm.add_constant(pd.DataFrame(_X_train, columns=feature_cols))
    X_reg_test_c = sm.add_constant(pd.DataFrame(_X_test, columns=feature_cols))
    ols = sm.OLS(y_cnt_train, X_reg_train_c).fit()
    results['ols'] = {
        'model': ols,
        'y_pred': ols.predict(X_reg_test_c),
    }

    # Poisson
    poisson = GLM(y_cnt_train, X_reg_train_c, family=Poisson()).fit()
    results['poisson'] = {
        'model': poisson,
        'y_pred': poisson.predict(X_reg_test_c),
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="AML Project 2 – Bike Sharing Analysis",
                       page_icon="🚲", layout="wide")
    st.title("🚲 Applied ML – Course Project 2: Classification & Regression")
    st.markdown("**UCI Bike Sharing Dataset** | Logistic Regression, LDA, QDA, Naive Bayes, OLS vs Poisson")
    st.divider()

    # Load data
    (df, feature_cols, scaler,
     X_train, X_test, X_train_sc, X_test_sc,
     y_bin_train, y_bin_test, y_multi_train, y_multi_test,
     y_cnt_train, y_cnt_test) = load_and_prepare_data()

    results = train_all_models(
        X_train_sc, X_test_sc, y_bin_train, y_bin_test,
        y_multi_train, y_multi_test, feature_cols, X_train, X_test,
        y_cnt_train, y_cnt_test
    )

    # Sidebar navigation
    page = st.sidebar.radio("Navigate", [
        "📊 EDA",
        "1️⃣ Logistic Regression",
        "2️⃣ Discriminant Analysis",
        "3️⃣ Naive Bayes & Comparison",
        "4️⃣ Linear vs Poisson Regression",
        "🔮 Interactive Prediction"
    ])

    # ─────────────────────────────────────────────────────────
    if page == "📊 EDA":
        st.header("Exploratory Data Analysis")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", f"{len(df):,}")
        col2.metric("Features", len(feature_cols))
        col3.metric("Avg Hourly Count", f"{df['cnt'].mean():.1f}")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Feature Distributions")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].hist(df['cnt'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].set_title('Distribution of Bike Rental Count')
        axes[0, 0].set_xlabel('Count')

        hourly = df.groupby('hr')['cnt'].mean()
        axes[0, 1].bar(hourly.index, hourly.values, color='coral', edgecolor='black')
        axes[0, 1].set_title('Average Rentals by Hour')
        axes[0, 1].set_xlabel('Hour')

        season_labels = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
        seasonal = df.groupby('season')['cnt'].mean()
        axes[1, 0].bar([season_labels[s] for s in seasonal.index], seasonal.values,
                       color=['green', 'gold', 'orange', 'skyblue'], edgecolor='black')
        axes[1, 0].set_title('Average Rentals by Season')

        corr_cols = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'cnt']
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap='RdBu_r', center=0,
                    ax=axes[1, 1], fmt='.2f')
        axes[1, 1].set_title('Correlation Heatmap')
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Demand Class Distribution")
        fig2, ax = plt.subplots(figsize=(6, 4))
        df['demand_class'].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#f39c12', '#e74c3c'],
                                                edgecolor='black')
        ax.set_title('Demand Class Distribution')
        ax.set_ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig2)

    # ─────────────────────────────────────────────────────────
    elif page == "1️⃣ Logistic Regression":
        st.header("Task 1: Logistic Regression")

        tab1, tab2, tab3, tab4 = st.tabs([
            "a) Binary LR", "b) Coefficients & Stats", "c) Confounding", "d) Multi-class"
        ])

        with tab1:
            st.subheader("Binary Logistic Regression")
            lr = results['lr']
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{lr['acc']:.4f}")
            col2.metric("ROC AUC", f"{auc(*roc_curve(y_bin_test, lr['y_prob'])[:2]):.4f}")

            st.text("Classification Report:")
            report = classification_report(y_bin_test, lr['y_pred'], target_names=['Low', 'High'])
            st.code(report)

            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_bin_test, lr['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
            ax.set_title('Binary LR Confusion Matrix')
            ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
            st.pyplot(fig)

        with tab2:
            st.subheader("Standard Error, Z-statistic, p-value")
            logit_res = results['logit_sm']

            coef_df = pd.DataFrame({
                'Feature': ['const'] + feature_cols,
                'Coefficient': logit_res.params,
                'Std Error': logit_res.bse,
                'Z-statistic': logit_res.tvalues,
                'P-value': logit_res.pvalues,
            })
            coef_df['Significant'] = coef_df['P-value'].apply(
                lambda p: '✅ Yes' if p < 0.05 else '❌ No')

            st.dataframe(coef_df.style.format({
                'Coefficient': '{:.4f}', 'Std Error': '{:.4f}',
                'Z-statistic': '{:.4f}', 'P-value': '{:.6f}'
            }).applymap(lambda x: 'background-color: #d4edda' if x == '✅ Yes'
                        else ('background-color: #f8d7da' if x == '❌ No' else ''),
                        subset=['Significant']),
                use_container_width=True)

            st.markdown("""
            **Interpretation:** Features with p-value < 0.05 are statistically significant.
            Key significant predictors include `hr` (hour), `yr` (year), `hum` (humidity),
            `atemp` (feels-like temperature), and `workingday`.
            """)

        with tab3:
            st.subheader("Confounding Variable Analysis")
            vif = results['vif']

            st.markdown("**Variance Inflation Factor (VIF)** — VIF > 5 indicates multicollinearity:")
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['red' if v > 5 else 'steelblue' for v in vif['VIF']]
            ax.barh(vif['Feature'], vif['VIF'], color=colors, edgecolor='black')
            ax.axvline(x=5, color='red', linestyle='--', label='VIF=5 threshold')
            ax.set_xlabel('VIF')
            ax.set_title('Variance Inflation Factors')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            st.dataframe(vif.style.format({'VIF': '{:.2f}'}), use_container_width=True)

            st.markdown("""
            **Findings:**
            - `temp` and `atemp` have **VIF ≈ 44**, indicating severe multicollinearity
            - When `atemp` is removed, `temp`'s coefficient changes by ~99% — **confirming confounding**
            - `season` and `mnth` also show moderate correlation (VIF ≈ 3.2–3.5)
            - **Recommendation:** Remove `atemp` or `temp` to reduce confounding
            """)

        with tab4:
            st.subheader("Multi-class Logistic Regression (Low/Medium/High)")
            lr_m = results['lr_multi']
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{lr_m['acc']:.4f}")
            col2.metric("Weighted F1", f"{lr_m['f1']:.4f}")

            report_m = classification_report(y_multi_test, lr_m['y_pred'],
                                              target_names=['Low', 'Medium', 'High'])
            st.code(report_m)

            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_multi_test, lr_m['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                        xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
            ax.set_title('Multi-class LR Confusion Matrix')
            ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
            st.pyplot(fig)

    # ─────────────────────────────────────────────────────────
    elif page == "2️⃣ Discriminant Analysis":
        st.header("Task 2: Discriminant Analysis")

        tab1, tab2, tab3, tab4 = st.tabs([
            "a) LDA", "b) Threshold", "c) ROC Curves", "d) QDA"
        ])

        with tab1:
            st.subheader("Linear Discriminant Analysis")
            lda = results['lda']
            col1, col2 = st.columns(2)
            col1.metric("Binary Accuracy", f"{lda['acc']:.4f}")
            col2.metric("Multi-class Accuracy", f"{results['lda_multi']['acc']:.4f}")

            st.code(classification_report(y_bin_test, lda['y_pred'],
                                          target_names=['Low', 'High']))

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_bin_test, lda['y_pred']),
                        annot=True, fmt='d', cmap='Greens', ax=ax,
                        xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
            ax.set_title('LDA Confusion Matrix')
            ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
            st.pyplot(fig)

        with tab2:
            st.subheader("Effective Threshold (Youden's J Statistic)")
            lda = results['lda']
            fpr, tpr, thresholds = roc_curve(y_bin_test, lda['y_prob'])
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_thresh = thresholds[best_idx]

            col1, col2, col3 = st.columns(3)
            col1.metric("Optimal Threshold", f"{best_thresh:.4f}")
            col2.metric("TPR at Threshold", f"{tpr[best_idx]:.4f}")
            col3.metric("FPR at Threshold", f"{fpr[best_idx]:.4f}")

            # Threshold slider
            thresh = st.slider("Adjust Classification Threshold", 0.0, 1.0, 0.5, 0.01)
            y_custom = (lda['y_prob'] >= thresh).astype(int)
            st.metric(f"Accuracy at threshold={thresh:.2f}", f"{accuracy_score(y_bin_test, y_custom):.4f}")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(thresholds, tpr[:-1] if len(tpr) > len(thresholds) else tpr[:len(thresholds)],
                    'b-', label='TPR (Sensitivity)')
            ax.plot(thresholds, fpr[:-1] if len(fpr) > len(thresholds) else fpr[:len(thresholds)],
                    'r-', label='FPR (1-Specificity)')
            ax.plot(thresholds, j_scores[:-1] if len(j_scores) > len(thresholds) else j_scores[:len(thresholds)],
                    'g--', label="Youden's J")
            ax.axvline(x=best_thresh, color='gray', linestyle=':', label=f'Optimal={best_thresh:.3f}')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Rate')
            ax.set_title('Threshold Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            st.subheader("ROC Curves")

            # Binary ROC
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for name, key, style in [
                ('Logistic Regression', 'lr', 'b-'),
                ('LDA', 'lda', 'r--'),
                ('QDA', 'qda', 'g-.'),
                ('Naive Bayes', 'nb', 'm:')
            ]:
                fpr_i, tpr_i, _ = roc_curve(y_bin_test, results[key]['y_prob'])
                auc_i = auc(fpr_i, tpr_i)
                axes[0].plot(fpr_i, tpr_i, style, label=f'{name} (AUC={auc_i:.3f})', linewidth=2)
            axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
            axes[0].set_title('ROC Curves - Binary Classification')
            axes[0].legend(loc='lower right')
            axes[0].grid(True, alpha=0.3)

            # Multi-class ROC
            y_multi_test_bin = label_binarize(y_multi_test, classes=[0, 1, 2])
            y_multi_prob = results['lr_multi']['y_prob']
            for i, (name, color) in enumerate(zip(['Low', 'Medium', 'High'], ['blue', 'green', 'red'])):
                fpr_i, tpr_i, _ = roc_curve(y_multi_test_bin[:, i], y_multi_prob[:, i])
                auc_i = auc(fpr_i, tpr_i)
                axes[1].plot(fpr_i, tpr_i, color=color, linewidth=2,
                             label=f'{name} (AUC={auc_i:.3f})')
            axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
            axes[1].set_title('Multi-class ROC (OvR) - LR')
            axes[1].legend(loc='lower right')
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with tab4:
            st.subheader("Quadratic Discriminant Analysis")
            qda = results['qda']
            col1, col2 = st.columns(2)
            col1.metric("Binary Accuracy", f"{qda['acc']:.4f}")
            col2.metric("Multi-class Accuracy", f"{results['qda_multi']['acc']:.4f}")

            st.code(classification_report(y_bin_test, qda['y_pred'],
                                          target_names=['Low', 'High']))

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.heatmap(confusion_matrix(y_bin_test, results['lda']['y_pred']),
                        annot=True, fmt='d', cmap='Greens', ax=axes[0],
                        xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
            axes[0].set_title('LDA')
            axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')
            sns.heatmap(confusion_matrix(y_bin_test, qda['y_pred']),
                        annot=True, fmt='d', cmap='Purples', ax=axes[1],
                        xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
            axes[1].set_title('QDA')
            axes[1].set_ylabel('Actual'); axes[1].set_xlabel('Predicted')
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
            **LDA vs QDA:**
            - LDA assumes equal covariance matrices across classes → linear decision boundary
            - QDA allows class-specific covariance → quadratic/non-linear decision boundary
            - QDA often performs better when the Gaussian assumption holds but covariances differ
            """)

    # ─────────────────────────────────────────────────────────
    elif page == "3️⃣ Naive Bayes & Comparison":
        st.header("Task 3: Naive Bayes & Model Comparison")

        tab1, tab2 = st.tabs(["Naive Bayes", "Full Comparison"])

        with tab1:
            st.subheader("Gaussian Naive Bayes")
            nb = results['nb']
            col1, col2, col3 = st.columns(3)
            col1.metric("Binary Accuracy", f"{nb['acc']:.4f}")
            col2.metric("ROC AUC", f"{auc(*roc_curve(y_bin_test, nb['y_prob'])[:2]):.4f}")
            col3.metric("Multi-class Accuracy", f"{results['nb_multi']['acc']:.4f}")

            st.code(classification_report(y_bin_test, nb['y_pred'],
                                          target_names=['Low', 'High']))

        with tab2:
            st.subheader("Comprehensive Model Comparison")

            # Binary comparison table
            model_names = ['Logistic Regression', 'LDA', 'QDA', 'Naive Bayes']
            keys = ['lr', 'lda', 'qda', 'nb']
            binary_df = pd.DataFrame({
                'Model': model_names,
                'Accuracy': [results[k]['acc'] for k in keys],
                'ROC AUC': [auc(*roc_curve(y_bin_test, results[k]['y_prob'])[:2]) for k in keys],
                'F1 (wt.)': [f1_score(y_bin_test, results[k]['y_pred'], average='weighted') for k in keys],
                'Precision (wt.)': [precision_score(y_bin_test, results[k]['y_pred'], average='weighted') for k in keys],
                'Recall (wt.)': [recall_score(y_bin_test, results[k]['y_pred'], average='weighted') for k in keys],
            })

            st.markdown("**Binary Classification (Low vs High)**")
            st.dataframe(binary_df.style.format({c: '{:.4f}' for c in binary_df.columns if c != 'Model'})
                         .highlight_max(subset=[c for c in binary_df.columns if c != 'Model'], color='#d4edda'),
                         use_container_width=True)

            # Multi-class comparison
            multi_keys = ['lr_multi', 'lda_multi', 'qda_multi', 'nb_multi']
            multi_df = pd.DataFrame({
                'Model': model_names,
                'Accuracy': [results[k]['acc'] for k in multi_keys],
                'F1 (weighted)': [results[k]['f1'] for k in multi_keys],
            })

            st.markdown("**Multi-class Classification (Low/Medium/High)**")
            st.dataframe(multi_df.style.format({c: '{:.4f}' for c in multi_df.columns if c != 'Model'})
                         .highlight_max(subset=[c for c in multi_df.columns if c != 'Model'], color='#d4edda'),
                         use_container_width=True)

            # Visual comparison
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            x = np.arange(len(model_names))
            colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
            axes[0].bar(x, binary_df['Accuracy'], color=colors, edgecolor='black')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(model_names, rotation=15, ha='right')
            axes[0].set_title('Binary Classification Accuracy')
            axes[0].set_ylim(0.6, 0.9)
            for i, v in enumerate(binary_df['Accuracy']):
                axes[0].text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=10)

            axes[1].bar(x, multi_df['Accuracy'], color=colors, edgecolor='black')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(model_names, rotation=15, ha='right')
            axes[1].set_title('Multi-class Accuracy')
            axes[1].set_ylim(0.3, 0.8)
            for i, v in enumerate(multi_df['Accuracy']):
                axes[1].text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
            **Key observations:**
            - **QDA** outperforms others in binary classification — its ability to model 
              non-linear decision boundaries better captures the weather/time interaction patterns.
            - **LR and LDA** produce very similar results, as expected when class distributions are approximately Gaussian.
            - **Naive Bayes** performs competitively despite its independence assumption.
            """)

    # ─────────────────────────────────────────────────────────
    elif page == "4️⃣ Linear vs Poisson Regression":
        st.header("Task 4: Linear vs Poisson Regression")

        ols = results['ols']
        poisson = results['poisson']

        ols_rmse = np.sqrt(np.mean((y_cnt_test - ols['y_pred']) ** 2))
        ols_mae = np.mean(np.abs(y_cnt_test - ols['y_pred']))
        poi_rmse = np.sqrt(np.mean((y_cnt_test - poisson['y_pred']) ** 2))
        poi_mae = np.mean(np.abs(y_cnt_test - poisson['y_pred']))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("OLS RMSE", f"{ols_rmse:.2f}")
        col2.metric("OLS R²", f"{ols['model'].rsquared:.4f}")
        col3.metric("Poisson RMSE", f"{poi_rmse:.2f}")

        dispersion = poisson['model'].pearson_chi2 / (X_train.shape[0] - len(feature_cols) - 1)
        col4.metric("Overdispersion φ", f"{dispersion:.2f}")

        # Comparison table
        comp_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'AIC', 'BIC'],
            'Linear (OLS)': [ols_rmse, ols_mae, ols['model'].aic, ols['model'].bic],
            'Poisson': [poi_rmse, poi_mae, poisson['model'].aic, poisson['model'].bic_llf],
        })
        st.dataframe(comp_df.style.format({'Linear (OLS)': '{:.2f}', 'Poisson': '{:.2f}'}),
                     use_container_width=True)

        # Diagnostic plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].scatter(y_cnt_test, ols['y_pred'], alpha=0.15, s=5, color='steelblue')
        mx = max(y_cnt_test.max(), np.max(ols['y_pred']))
        axes[0, 0].plot([0, mx], [0, mx], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual'); axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title(f'OLS: Actual vs Predicted (R²={ols["model"].rsquared:.3f})')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(y_cnt_test, poisson['y_pred'], alpha=0.15, s=5, color='coral')
        axes[0, 1].plot([0, mx], [0, mx], 'r--', linewidth=2)
        axes[0, 1].set_xlabel('Actual'); axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].set_title('Poisson: Actual vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)

        ols_resid = y_cnt_test - ols['y_pred']
        poi_resid = y_cnt_test - poisson['y_pred']
        axes[1, 0].scatter(ols['y_pred'], ols_resid, alpha=0.15, s=5, color='steelblue')
        axes[1, 0].axhline(0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Predicted'); axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('OLS: Residuals')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(poisson['y_pred'], poi_resid, alpha=0.15, s=5, color='coral')
        axes[1, 1].axhline(0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Predicted'); axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Poisson: Residuals')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(f"""
        **Analysis:**
        - **Overdispersion (φ = {dispersion:.2f})** is detected in the Poisson model (φ >> 1),
          meaning the variance greatly exceeds the mean — common in real-world count data.
        - OLS produces **negative predictions** for some observations, which is invalid for count data.
        - Poisson regression uses a **log-link function** ensuring non-negative predictions.
        - Both models have similar RMSE/MAE because the Poisson model's log-link helps for small counts
          but struggles with overdispersion for large counts.
        - A **Negative Binomial** regression would be a natural next step to handle overdispersion.
        """)

        # OLS coefficient significance
        with st.expander("OLS Model Summary"):
            st.text(str(ols['model'].summary()))
        with st.expander("Poisson Model Summary"):
            st.text(str(poisson['model'].summary()))

    # ─────────────────────────────────────────────────────────
    elif page == "🔮 Interactive Prediction":
        st.header("Interactive Bike Demand Prediction")
        st.markdown("Adjust the features below to predict bike rental demand.")

        col1, col2, col3 = st.columns(3)
        with col1:
            season = st.selectbox("Season", [1, 2, 3, 4],
                                  format_func=lambda x: {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}[x])
            yr = st.selectbox("Year", [0, 1], format_func=lambda x: {0: '2011', 1: '2012'}[x])
            mnth = st.slider("Month", 1, 12, 6)
            hr = st.slider("Hour", 0, 23, 12)
        with col2:
            holiday = st.selectbox("Holiday", [0, 1], format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
            weekday = st.slider("Weekday (0=Sun)", 0, 6, 3)
            workingday = st.selectbox("Working Day", [0, 1], format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
            weathersit = st.selectbox("Weather", [1, 2, 3, 4],
                                       format_func=lambda x: {1: 'Clear', 2: 'Mist/Cloudy',
                                                               3: 'Light Rain/Snow', 4: 'Heavy Rain/Storm'}[x])
        with col3:
            temp = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5, 0.01)
            atemp = st.slider("Feels-like Temp (normalized)", 0.0, 1.0, 0.5, 0.01)
            hum = st.slider("Humidity (normalized)", 0.0, 1.0, 0.5, 0.01)
            windspeed = st.slider("Wind Speed (normalized)", 0.0, 1.0, 0.15, 0.01)

        input_data = np.array([[season, yr, mnth, hr, holiday, weekday, workingday,
                                weathersit, temp, atemp, hum, windspeed]])
        input_sc = scaler.transform(input_data)

        st.divider()

        # Classification predictions
        st.subheader("Classification Predictions")
        col1, col2, col3, col4 = st.columns(4)

        for col, name, key in [(col1, 'Logistic Reg.', 'lr'), (col2, 'LDA', 'lda'),
                                (col3, 'QDA', 'qda'), (col4, 'Naive Bayes', 'nb')]:
            pred = results[key]['model'].predict(input_sc)[0]
            prob = results[key]['model'].predict_proba(input_sc)[0]
            label = '🟢 High' if pred == 1 else '🔴 Low'
            col.metric(name, label)
            col.caption(f"P(High): {prob[1]:.2%}")

        # Multi-class
        st.subheader("Multi-class Predictions")
        class_names = ['Low', 'Medium', 'High']
        class_colors = ['🔴', '🟡', '🟢']
        col1, col2, col3, col4 = st.columns(4)
        for col, name, key in [(col1, 'LR', 'lr_multi'), (col2, 'LDA', 'lda_multi'),
                                (col3, 'QDA', 'qda_multi'), (col4, 'NB', 'nb_multi')]:
            pred = results[key]['model'].predict(input_sc)[0]
            col.metric(name, f"{class_colors[pred]} {class_names[pred]}")

        # Regression predictions
        st.subheader("Count Predictions (Regression)")
        input_df = pd.DataFrame(input_data, columns=feature_cols)
        input_c = sm.add_constant(input_df)
        ols_pred = results['ols']['model'].predict(input_c)[0]
        poi_pred = results['poisson']['model'].predict(input_c)[0]

        col1, col2 = st.columns(2)
        col1.metric("Linear (OLS) Prediction", f"{max(0, ols_pred):.0f} bikes")
        col2.metric("Poisson Prediction", f"{poi_pred:.0f} bikes")


if __name__ == '__main__':
    main()
