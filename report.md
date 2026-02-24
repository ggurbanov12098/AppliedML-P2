# Applied Machine Learning – Course Project 2: Classification & Regression
## UCI Bike Sharing Dataset Analysis

**Course:** CSCI-6767 Applied Machine Learning & Data Analytics (Spring 2026)

---

## 1. Motivation

Urban bike-sharing systems are crucial to sustainable transportation. Accurately predicting rental demand helps operators optimize fleet deployment and station management. We apply classification and regression models to the **UCI Bike Sharing Dataset** (17,379 hourly records from Washington D.C.'s Capital Bikeshare, 2011–2012) to predict demand levels. The dataset contains 12 raw features including weather conditions (temperature, humidity, wind speed), temporal features (hour, season, weekday), and contextual flags (holiday, working day). No missing values exist, making it ideal for comparative model evaluation.

**Target Engineering:** We bin the continuous count (`cnt`) into **Low/Medium/High** demand classes using quantile-based binning for classification, and use the raw count for regression tasks.

## 2. Method

### 2.1 Feature Engineering

The raw 12-feature dataset exhibits several structural deficiencies that limit model performance when features are used as-is. We applied the following transformations (37 engineered features total):

| Transformation | Raw Feature(s) | Engineered Features | Rationale |
|---|---|---|---|
| **One-hot encoding** | `hr` (0–23) | `hr_1` … `hr_23` (23 dummies) | Demand is bimodal (peaks at 8am & 5pm), not linear in hour |
| **One-hot encoding** | `season` (1–4) | `season_2`, `season_3`, `season_4` | Demand is non-monotonic: Fall > Summer > Winter > Spring |
| **One-hot encoding** | `weathersit` (1–4) | `weather_2`, `weather_3` | Nominal categories (clear/mist/rain), not ordinal; cat 4 merged into 3 |
| **Cyclical encoding** | `mnth` (1–12) | `mnth_sin`, `mnth_cos` | Captures Dec→Jan continuity via sin/cos |
| **Dropped** | `atemp` | — | VIF ≈ 44 with `temp`; confirmed confounder |
| **Kept as-is** | `yr`, `holiday`, `weekday`, `workingday`, `temp`, `hum`, `windspeed` | 7 features | Already numeric/binary, appropriate encoding |

### 2.2 Logistic Regression
We apply binary and multinomial logistic regression on the 37 engineered features. Using `statsmodels`, we extract standard errors, Z-statistics, and p-values for each coefficient. Confounding is assessed via Variance Inflation Factors (VIF) computed on the **original raw features** to demonstrate the issue, while models use the engineered (corrected) features.

### 2.3 Discriminant Analysis
Linear Discriminant Analysis (LDA) assumes equal covariance matrices across classes, yielding a linear decision boundary. Quadratic Discriminant Analysis (QDA) relaxes this assumption, allowing class-specific covariance matrices. QDA uses `reg_param=0.1` for numerical stability in the 37-dimensional feature space. We determine the optimal classification threshold using Youden's J statistic and plot ROC curves for all models.

### 2.4 Naive Bayes
Gaussian Naive Bayes assumes conditional independence of features given the class. We compare it against LR, LDA, and QDA using accuracy, F1-score, precision, recall, and ROC AUC. Note: the one-hot encoded features structurally violate the independence assumption (e.g., `hr_1=1` implies `hr_2=0`), which limits NB performance.

### 2.5 Linear vs. Poisson Regression
OLS regression serves as the baseline for predicting the continuous count. Poisson regression, designed for count data with a log-link function, is compared using RMSE, MAE, AIC, and residual diagnostics. We assess overdispersion to evaluate Poisson model adequacy.

## 3. Experiments & Results

### 3.1 Logistic Regression Results

| Metric | Binary LR | Multi-class LR |
|--------|-----------|----------------|
| Accuracy | 0.8879 | 0.7731 |
| Weighted F1 | 0.8879 | 0.7737 |

**Significant predictors** (p < 0.05): `yr` (Z=28.3), `hum` (Z=−24.2), `workingday` (Z=8.7), `weekday` (Z=7.4), plus nearly all hour-of-day dummies (capturing the bimodal commute pattern). The hour dummies are overwhelmingly significant, confirming time-of-day as the strongest driver.

**Confounding (raw features):** `temp` and `atemp` exhibit VIF ≈ 44, confirming severe multicollinearity. `season` and `mnth` show moderate correlation (VIF ≈ 3.2–3.5). Resolution: `atemp` was dropped from the engineered feature set.

### 3.2 Discriminant Analysis Results

| Model | Binary Acc. | ROC AUC |
|-------|------------|---------|
| LDA | 0.8757 | 0.9538 |
| QDA | 0.8226 | 0.9103 |

**Optimal Threshold (LDA):** Youden's J identifies threshold = 0.5728, yielding TPR = 0.9142 and FPR = 0.1516 (J = 0.763). With engineered features, LDA now outperforms QDA thanks to the high-dimensional one-hot space where linear separation is highly effective.

### 3.3 Model Comparison (LR vs. LDA vs. QDA vs. NB)

| Model | Accuracy | ROC AUC | F1 (weighted) |
|-------|----------|---------|---------------|
| **Logistic Regression** | **0.8879** | **0.9567** | **0.8879** |
| **LDA** | 0.8757 | 0.9538 | 0.8754 |
| **QDA** | 0.8226 | 0.9103 | 0.8225 |
| **Naive Bayes** | 0.7146 | 0.7920 | 0.7091 |

Logistic Regression achieves the best performance across all metrics. The one-hot encoded features create a high-dimensional space where **linear models excel** — LR and LDA benefit most. QDA is solid but lower; it needs regularization (`reg_param=0.1`) with 37 features. Naive Bayes drops significantly because the one-hot features structurally violate its independence assumption.

### 3.4 Linear vs. Poisson Regression

| Metric | Linear (OLS) | Poisson |
|--------|-------------|---------|
| RMSE | 102.68 | 92.00 |
| MAE | 75.85 | 61.49 |
| R² | 0.6821 | — |

**Overdispersion:** φ = 32.92 >> 1, still present but **reduced by 69%** from the raw-feature baseline (~108). The proper hour-of-day encoding captures the bimodal demand pattern, greatly reducing unexplained variance. Poisson achieves **lower RMSE and MAE** than OLS. OLS can predict negative counts (invalid for count data), while Poisson's log-link ensures non-negativity. A **Negative Binomial** model would further address remaining overdispersion.

**5-Fold Cross-Validation (Binary):** LR: 0.888 ± 0.002, LDA: 0.878 ± 0.005, QDA: 0.824 ± 0.004, NB: 0.719 ± 0.009.

### 3.5 Feature Engineering Impact

The table below summarizes improvements from feature engineering (same models, no hyperparameter tuning changes):

| Metric | Before (raw) | After (engineered) | Improvement |
|--------|-------------|-------------------|-------------|
| Binary LR Accuracy | 0.788 | 0.888 | **+10.0 pts** |
| Binary LR AUC | 0.857 | 0.957 | **+10.0 pts** |
| Multi-class LR Accuracy | 0.640 | 0.773 | **+13.3 pts** |
| OLS R² | 0.393 | 0.682 | **+74% relative** |
| OLS RMSE | 143.8 | 102.7 | **−29%** |
| Poisson RMSE | 145.6 | 92.0 | **−37%** |
| Overdispersion (φ) | 107.7 | 32.9 | **−69%** |

### 3.6 Error Analysis
- **Classification errors** concentrate on "Medium" demand samples, which lie near decision boundaries between Low and High classes.
- **Regression residuals** show heteroscedasticity: errors grow with predicted count, suggesting variance depends on the mean — a further argument for Poisson/NB regression over OLS.
- Hour of day (`hr`) is the strongest predictor across all models, reflected in the highly significant hour dummies.
- Naive Bayes performance degradation is expected: one-hot features create perfect negative correlations that violate the independence assumption.

## 4. Conclusion

With proper feature engineering, **Logistic Regression becomes the best classifier** for this dataset, outperforming QDA which previously led on raw features. The key insight: when categorical/cyclical features are correctly encoded, the resulting high-dimensional space makes linear decision boundaries highly effective. For count prediction, Poisson regression with engineered features achieves RMSE = 92 (vs. 146 on raw features), explaining ~68% of variance. The confounding between `temp` and `atemp` (VIF ≈ 44) was resolved by dropping `atemp`. Remaining overdispersion (φ = 33) suggests a Negative Binomial model as a natural next step.

## 5. Running the Code

```bash
# Install dependencies
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn streamlit scipy ucimlrepo

# Run analysis (generates figures and CSV outputs)
python analysis.py

# Launch interactive UI
streamlit run app.py
```

## References

1. UCI Machine Learning Repository – Bike Sharing Dataset. https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
2. Fanaee-T, H. & Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. *Progress in AI*, 2(2-3), 113-127.
3. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
