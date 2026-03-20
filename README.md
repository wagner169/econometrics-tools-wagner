# econometrics_tools_wagner

Personal Python package for econometrics and machine learning utilities.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/wagner169/econometrics-tools-wagner.git

Available functions
Econometrics

vif(X) → Calculates Variance Inflation Factor for multicollinearity analysis

ols(y, X) → Fits an OLS regression model using statsmodels

boxcox_transform(y, return_transformed=True, verbose=True) → Applies Box-Cox transformation and suggests interpretation

Machine Learning Evaluation

plot_roc(model, X_test, y_test, scaler=None) → Plots ROC curve and returns AUC

plot_multiple_roc(models, X_test, y_test, scaler=None) → Compares ROC curves across multiple models

plot_precision_recall(model, X_test, y_test, scaler=None) → Plots Precision-Recall curve and returns PR AUC
