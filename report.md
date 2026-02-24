# Applied Machine Learning – Course Project 2: Classification & Regression
## UCI Bike Sharing Dataset Analysis

**Course:** CSCI-6767 Applied Machine Learning & Data Analytics (Spring 2026)

---

## 1. Motivation

Urban bike-sharing systems are crucial to sustainable transportation. Accurately predicting rental demand helps operators optimize fleet deployment and station management. We apply classification and regression models to the **UCI Bike Sharing Dataset** (17,379 hourly records from Washington D.C.'s Capital Bikeshare, 2011–2012) to predict demand levels. The dataset contains 12 features including weather conditions (temperature, humidity, wind speed), temporal features (hour, season, weekday), and contextual flags (holiday, working day). No missing values exist, making it ideal for comparative model evaluation.

**Target Engineering:** We bin the continuous count (`cnt`) into **Low/Medium/High** demand classes using quantile-based binning for classification, and use the raw count for regression tasks.

## 2. Method

### 2.1 Logistic Regression
We apply binary and multinomial logistic regression. Using `statsmodels`, we extract standard errors, Z-statistics, and p-values for each coefficient. Confounding is assessed via Variance Inflation Factors (VIF) and coefficient stability analysis.

### 2.2 Discriminant Analysis
Linear Discriminant Analysis (LDA) assumes equal covariance matrices across classes, yielding a linear decision boundary. Quadratic Discriminant Analysis (QDA) relaxes this assumption, allowing class-specific covariance matrices. We determine the optimal classification threshold using Youden's J statistic and plot ROC curves for all models.

### 2.3 Naive Bayes
Gaussian Naive Bayes assumes conditional independence of features given the class. We compare it against LR, LDA, and QDA using accuracy, F1-score, precision, recall, and ROC AUC.

### 2.4 Linear vs. Poisson Regression
OLS regression serves as the baseline for predicting the continuous count. Poisson regression, designed for count data with a log-link function, is compared using RMSE, MAE, AIC, and residual diagnostics. We assess overdispersion to evaluate Poisson model adequacy.

## 3. Experiments & Results

### 3.1 Logistic Regression Results

| Metric | Binary LR | Multi-class LR |
|--------|-----------|----------------|
| Accuracy | 0.7883 | 0.6201 |
| Weighted F1 | 0.7882 | 0.6184 |

**Significant predictors** (p < 0.05): `hr` (Z=38.2), `yr` (Z=21.0), `hum` (Z=−22.7), `atemp` (Z=5.4), `workingday` (Z=5.6), `season` (Z=8.3).

**Confounding:** `temp` and `atemp` exhibit VIF ≈ 44, confirming severe multicollinearity. Removing `atemp` changes `temp`'s coefficient by **99.1%**, conclusively demonstrating confounding. `season` and `mnth` show moderate correlation (VIF ≈ 3.2–3.5).

### 3.2 Discriminant Analysis Results

| Model | Binary Acc. | ROC AUC |
|-------|------------|---------|
| LDA | 0.7883 | 0.8557 |
| QDA | 0.8235 | 0.8817 |

**Optimal Threshold (LDA):** Youden's J identifies threshold = 0.4974, yielding TPR = 0.7823 and FPR = 0.2057. QDA outperforms LDA by capturing non-linear weather–time interaction patterns through class-specific covariance estimation.

### 3.3 Model Comparison (LR vs. LDA vs. QDA vs. NB)

| Model | Accuracy | ROC AUC | F1 (weighted) |
|-------|----------|---------|---------------|
| **Logistic Regression** | 0.7883 | 0.8571 | 0.7882 |
| **LDA** | 0.7883 | 0.8557 | 0.7882 |
| **QDA** | **0.8235** | **0.8817** | **0.8235** |
| **Naive Bayes** | 0.7899 | 0.8619 | 0.7897 |

QDA achieves the best performance across all metrics. LR and LDA produce nearly identical results, consistent with theory (LDA is equivalent to LR under Gaussian assumptions). Naive Bayes is competitive despite its independence assumption, performing slightly better than LR/LDA on ROC AUC.

### 3.4 Linear vs. Poisson Regression

| Metric | Linear (OLS) | Poisson |
|--------|-------------|---------|
| RMSE | 179.50 | 179.50 |
| MAE | 140.88 | 140.88 |
| AIC | 172,669 | 2,263,475 |
| R² | 0.3375 | — |

**Overdispersion:** φ = 174.08 >> 1, indicating the Poisson model's equal mean-variance assumption is violated. OLS can predict negative counts (invalid for count data), while Poisson's log-link ensures non-negativity. The high AIC for Poisson reflects overdispersion misspecification. A **Negative Binomial** model would better handle this overdispersion.

### 3.5 Error Analysis
- **Classification errors** concentrate on "Medium" demand samples, which lie near decision boundaries between Low and High classes.
- **Regression residuals** show heteroscedasticity: errors grow with predicted count, suggesting variance depends on the mean — a further argument for Poisson/NB regression over OLS.
- Hour of day (`hr`) is the strongest predictor across all models, reflecting the bimodal rush-hour pattern in bike demand.

## 4. Conclusion

QDA is the best classifier for this dataset, capturing non-linear feature interactions. For count prediction, while OLS provides a simple baseline, Poisson regression is theoretically more appropriate for count data despite overdispersion issues. The confounding between `temp` and `atemp` highlights the importance of multicollinearity diagnostics in applied modeling.

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
