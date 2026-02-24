Spring 2026 Applied Machine Learning & Data Analytics (CSCI-6767 - 20542)  
Assignment Instructions  
Course Project – 2. 

Classification  
Apply Logistic Regression and Linear Discriminative Analysis to a real-world application. You can create your own data or obtain it from internet resources. Implement the following operations:  

1.	Logistic Regression (20%)  
a)	Apply Logistic Regression.  
b)	Calculate Standard error, Z-statistic and p-value for each coefficient.  
c)	Is there confounding variable?  
d)	Apply Logistic regression with more than two classes (if any)  

2.	Discriminative Analysis (20%)  
a)	Apply Linear Discriminative Analysis.  
b)	Define effective threshold for your model.  
c)	Draw ROC curve.  
d)	Apply Quadratic Discriminative Analysis (QDA).  

3.	Apply Naïve Bayes algorithms. Compare LR, LDA, QDA and NB (20%)  

4.	Compare Linear and Poisson Regression Model (20%)

5.	Write a report (20%)

6.	Extra task: Design a clean, user-friendly UI and demonstrate the program output with example results. (20%)

Presentations  
Teams are randomly formed by 2 students. Each team should prepare a presentation and be prepared to give a short explanation. For each presentation a time slot of 10 minutes is scheduled (5 minutes for presentation + 5 minutes for Q&A).

Instruction for the report  
Final project write-ups can be at most 5 pages long (including appendices and figures). We will allow for extra pages containing only references. Please include a section that describes what each team member worked on and contributed to the project. Each team member will be addressed the questions related to the project’s theoretical model and program code accordingly.  
Your project report should include the following information:  
•	Motivation: What problem are you tackling, and what's the setting you're considering?  
•	Method: What machine learning techniques have you tried and why?  
•	Experiments: Describe the experiments that you've run, the outcomes, and any error analysis that you've done. You should have tried at least one baseline. Note: negative results that indicate something did not work are welcome.  

Submission  
The final version of the project report accompanied by the all-source code, slides, relevant data and experimental results should be submitted to the Blackboard System after presentation.  
•	If you submit your solution after the deadline your score will be deducted by 20% per day.  
•	If any group member does not participate in the presentation, he/she will be penalized with a 50% deduction from their grade.





Optional Dataset choice: UCI Bike Sharing Dataset

The UCI Bike Sharing Dataset is a solid choice for your multipurpose project, as it supports all required tasks including classification (via derived targets), LDA/QDA, Naïve Bayes comparisons, and especially regression analysis.[20]

## Dataset Overview
It includes hourly (17,379 samples) and daily (731 samples) rental counts ("cnt") from 2011-2012 in Capital Bikeshare, with 12-16 features like weather (temp, humidity), time (hour, season, weekday), and user types (casual/registered). No missing values; numeric/categorical mix ideal for preprocessing.[20]

## Fit for Project Tasks
- **Classification (LR, LDA, QDA, NB)**: Bin "cnt" into low/medium/high demand classes (e.g., quantiles/seasonal thresholds) for binary/multiclass prediction. Compute SE/Z/p-values, confounders (e.g., weather vs. temp), ROC/thresholds. Multi-class via demand levels or user type.[19]
- **Naïve Bayes Comparison**: Works well for categorical-derived features; compare accuracy/F1 across models on binned targets.[2]
- **Regression Comparison**: Native strength—model "cnt" with Linear (continuous) vs. Poisson (count data, overdispersion common); assess residuals/AIC.[8][10]

| Task | Why It Fits | Potential Challenges |
|------|-------------|----------------------|
| Logistic Regression | Binned demand; stats on coeffs (e.g., hour significant). | Derive target; check multicollinearity (temp/hum). |
| LDA/QDA | Gaussian assumptions on numeric features; ROC easy. | Scale features; QDA for non-linear weather effects. |
| NB Comparison | Independence assumption tests casual/registered split. | Feature engineering for categoricals. |
| Linear vs. Poisson | Hourly counts perfect for Poisson; linear baseline. | Overdispersion favors Poisson. [2] |

## Pros and Cons
- **Pros**: Real-world, time-series insights (e.g., peaks by hour/weather); enables confounders (e.g., season confounding temp); UI demo via Streamlit plots. Larger than Iris, smaller than massive sets.[19]
- **Cons**: Primarily regression-focused originally, so classification needs target engineering (simple via pandas cut/qcut). No natural multi-class beyond seasons.[20]

Download hour.csv for granularity; start with EDA on weather-hour interactions. Excellent for motivation (urban sustainability).[20]

Sources
[1] Data-driven insights into (E-)bike-sharing: mining a large-scale dataset on usage and urban characteristics: descriptive analysis and performance modeling https://link.springer.com/10.1007/s11116-025-10661-2
[2] Demand Prediction on Bike Sharing Data Using Regression Analysis Approach http://jicet.org/index.php/JICET/article/view/52
[3] Spatiotemporal Data-Driven Hourly Bike-Sharing Demand Prediction Using ApexBoost Regression https://link.springer.com/10.1007/s41060-025-00820-0
[4] Environmental Benefits Evaluation of a Bike-Sharing System in the Boston Area: A Longitudinal Study https://www.mdpi.com/2413-8851/9/5/159
[5] Examining the Impact of Electric Bike-Sharing on For-Hire Vehicles in Medium-Sized Cities: An Empirical Study in Yancheng, China https://www.mdpi.com/2071-1050/17/2/754
[6] Centralized and Federated Heart Disease Classification Models Using UCI Dataset and their Shapley-value Based Interpretability https://arxiv.org/abs/2408.06183
[7] Discrete wavelet transform application for bike sharing system check-in/out demand prediction https://www.tandfonline.com/doi/full/10.1080/19427867.2023.2219045
[8] Research on the prediction of bike sharing system’s demand based on linear regression model https://www.shs-conferences.org/10.1051/shsconf/202418101006
[9] Optimized Demand Forecasting for Bike-Sharing Stations Through Multi-Method Fusion and Gated Graph Convolutional Neural Networks https://ieeexplore.ieee.org/document/10756689/
[10] Causal inference of Seoul bike sharing demand https://www.mathematicsgroup.com/articles/CMA-2-105.php
[11] A Demand-Centric Repositioning Strategy for Bike-Sharing Systems https://www.mdpi.com/1424-8220/22/15/5580/pdf?version=1658902567
[12] BIKED: A Dataset for Computational Bicycle Design with Machine Learning
  Benchmarks http://arxiv.org/pdf/2103.05844v2.pdf
[13] CNN-GRU-AM for Shared Bicycles Demand Forecasting https://downloads.hindawi.com/journals/cin/2021/5486328.pdf
[14] Predicting Station-level Hourly Demands in a Large-scale Bike-sharing
  Network: A Graph Convolutional Neural Network Approach https://arxiv.org/ftp/arxiv/papers/1712/1712.04997.pdf
[15] Data-Driven Analysis of Bicycle Sharing Systems as Public Transport Systems Based on a Trip Index Classification https://www.mdpi.com/1424-8220/20/15/4315/pdf
[16] FF-STGCN: A usage pattern similarity based dual-network for bike-sharing demand prediction https://dx.plos.org/10.1371/journal.pone.0298684
[17] Categorizing Bicycling Environment Quality Based on Mobile Sensor Data and Bicycle Flow Data https://www.mdpi.com/2071-1050/13/8/4085/pdf
[18] Analysis on the Riding Characteristics of Mobike Sharing Bicycle in Beijing City, China https://www.abstr-int-cartogr-assoc.net/1/37/2019/ica-abs-1-37-2019.pdf
[19] GitHub - Pirkn/Bike-Sharing-Data-Analysis: An exploratory data analysis of the Bike Sharing Dataset from the UCI Machine Learning Repository. https://github.com/Pirkn/Bike-Sharing-Data-Analysis
[20] Bike Sharing - UCI Machine Learning Repository https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
