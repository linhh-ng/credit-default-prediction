### credit-default-prediction
Predicting credit default on 100k Lending Club loans. This project benchmarks a linear model (Logistic Regression) against non-linear algorithms (Random Forest, XGBoost) to determine which model better captures risk, using SHAP values to explain the results.

### Credit Default Prediction: Do non-linear models actually outperform linear ones?

A machine learning pipeline comparing Logistic Regression, Random Forest, and XGBoost on 100,000 Lending Club loan records. The main goal was to test whether tree-based, non-linear models actually outperform traditional linear methods when predicting whether a borrower will default.

### Why I built this

I built this project to test a limitation I identified during my econometrics dissertation. While modelling ESG factors and stock market returns, my residual patterns showed clear non-linear dynamics. This raised a practical question: when financial data exhibits non-linear patterns, do tree-based machine learning methods actually provide a measurable advantage, or is standard logistic regression robust enough to hold its own? Consumer credit default provided a clean dataset to find out.

### Data

- Source: Lending Club Loan Data (2007-2018), via Kaggle
- Size: 100,000 loans
- Target: Binary — Charged Off (1) vs Fully Paid (0)
- Variables: 16 variables total (13 numeric risk variables, 3 categorical: home ownership, loan purpose, employment length)

### Methods 

1. Data cleaning and Outliers: Capped extreme values to keep the models grounded (capped Debt-to-Income at 60 and annual income at 300k).
2. Encoding: Labeled encoding of categorical features for tree compatibility. 
3. Imbalance Handling: Handled the 20% default rate using class_weight='balanced' for Logistic Regression and Random Forest, and scale_pos_weight for XGBoost.
4. Validation: Stratified 80/20 train/test split (70,282 train / 17,571 test).
5. Evaluation: ROC-AUC, Classification Report, SHAP (bar charts, beeswarm, and dependence plots) 

### Results

| Model               | ROC-AUC | Type          |
|---------------------|---------|---------------|
| Logistic Regression | 0.7413  | Linear        |
| Random Forest       | 0.7390  | Non-Linear    |
| XGBoost             | 0.7420  | Non-Linear    |
  
The results were mixed. While XGBoost achieved the highest ROC-AUC (0.7420), the 0.0007 margin over Logistic Regression (0.7413) is too narrow to prove that non-linear models are significantly better for this dataset. 

Additionally, Random Forest underperformed the linear baseline. This suggests that the chosen credit features are largely linearly separable. The slight performance gain from XGBoost likely stems from its boosting algorithm rather than the discovery of complex non-linear relationships.

### Results from SHAP 

Even though the predictive scores were similar, plotting the SHAP values revealed exactly how the model was making decisions.

1. Interest rate dominates all other features by a large margin.
Interest rate ranked as the dominant predictor of default by a significant margin. This indicates that the lender's risk assessment, priced into the interest rate at origination, already captures much of the signal available in borrower characteristics.

2. DTI and FICO operate as independent risk factors.
The initial hypothesis was that a high Debt-to-Income (DTI) ratio would amplify default risk for low-FICO borrowers. However, the SHAP analysis contradicted this. The dependence plot revealed that FICO scores were uniformly scattered along the DTI axis, with no specific threshold. Therefore, DTI and FICO function as independent risk factors, adding to overall risk without complex non-linear interactions.

3. Two variables behave counterintuitively.
Two variables showed unexpected directional effects. Higher monthly instalments slightly decreased default risk, while longer employment histories marginally increased it. These patterns point to confounding factors rather than direct causality. For example, higher instalments are likely correlated with larger loans that are only approved for higher-income, low-risk borrowers.

4. Home ownership shows a predictive signal.
Even though home_ownership is a simple categorical variable, it ranked sixth in overall feature importance, above continuous variables such as annual income and open accounts. This suggests that a borrower's housing status acts as a reliable proxy for baseline wealth and stability.

### Limitation

Overall, this model predicts correlation rather than causation. By omitting unobserved borrower characteristics and broader macroeconomic conditions, it presents the exact same identification problem I addressed in my econometrics dissertation. While machine learning is exceptional at recognising complex predictive patterns, this project reinforced that isolating true causal drivers is a methodological challenge. It implies the need for the fixed-effects and instrumental variable methods I utilised in my prior research.

### Tools

Python | Pandas | Scikit-learn | XGBoost | SHAP | Matplotlib
