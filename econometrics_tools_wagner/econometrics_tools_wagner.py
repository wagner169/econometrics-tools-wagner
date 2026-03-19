import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, auc, precision_recall_curve


## =========================================================
## VIF: Variance Inflation Factor
## Sirve para detectar multicolinealidad entre variables X
## =========================================================
def vif(X):
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    return vif_data


## =========================================================
## OLS: Ordinary Least Squares
## Ajusta un modelo de regresión lineal con statsmodels
## =========================================================
def ols(y, X):
    X = sm.add_constant(X)
    modelo = sm.OLS(y, X).fit()
    return modelo


## =========================================================
## Box-Cox Transform
## Evalúa y transforma una variable y para mejorar normalidad
## y homocedasticidad en modelos lineales
## =========================================================
def boxcox_transform(y, return_transformed=True, verbose=True):
    
    ## Validación: Box-Cox solo funciona con valores positivos
    if np.any(y <= 0):
        raise ValueError("Box-Cox requiere que todos los valores de y sean positivos.")
    
    ## Transformación
    y_transformed, lambda_opt = stats.boxcox(y)
    
    ## Mensajes de interpretación
    if verbose:
        print(f"Lambda óptimo: {lambda_opt:.4f}")
        
        if abs(lambda_opt - 1) < 0.15:
            print("Recomendación: no transformar la variable.")
        elif abs(lambda_opt) < 0.15:
            print("Recomendación: usar log(y).")
        elif abs(lambda_opt - 0.5) < 0.15:
            print("Recomendación: usar sqrt(y).")
        elif lambda_opt < 0:
            print("Recomendación: transformación fuerte (inversa o Box-Cox).")
        else:
            print(f"Recomendación: usar Box-Cox con lambda = {lambda_opt:.4f}.")
    
    ## Retorno
    if return_transformed:
        return y_transformed, lambda_opt
    else:
        return lambda_opt


## =========================================================
## ROC Curve
## Grafica la curva ROC de un modelo de clasificación binaria
## y devuelve el valor AUC
## =========================================================
def plot_roc(model, X_test, y_test, scaler=None):
    
    ## Escalar si se proporciona scaler
    if scaler is not None:
        X_test = scaler.transform(X_test)
    
    ## Validación del modelo
    if not hasattr(model, "predict_proba"):
        raise ValueError("El modelo no tiene el método predict_proba().")
    
    ## Probabilidades de la clase positiva
    y_prob = model.predict_proba(X_test)[:, 1]
    
    ## Cálculo ROC y AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    ## Gráfico
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], '--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
    
    return roc_auc


## =========================================================
## Multiple ROC Curves
## Compara varios modelos de clasificación binaria en una sola
## gráfica ROC y devuelve un diccionario con los AUC
## =========================================================
def plot_multiple_roc(models, X_test, y_test, scaler=None):
    
    plt.figure()
    auc_scores = {}
    
    ## Escalar una sola vez si aplica
    X = scaler.transform(X_test) if scaler else X_test
    
    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"El modelo '{name}' no tiene el método predict_proba().")
        
        y_prob = model.predict_proba(X)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        auc_scores[name] = roc_auc
        
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], '--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
    
    return auc_scores


## =========================================================
## Precision-Recall Curve
## Grafica la curva Precision-Recall de un modelo de
## clasificación binaria y devuelve el PR AUC
## =========================================================
def plot_precision_recall(model, X_test, y_test, scaler=None):
    
    ## Escalar si aplica
    X = scaler.transform(X_test) if scaler else X_test
    
    ## Validación del modelo
    if not hasattr(model, "predict_proba"):
        raise ValueError("El modelo no tiene el método predict_proba().")
    
    ## Probabilidades de la clase positiva
    y_prob = model.predict_proba(X)[:, 1]
    
    ## Cálculo de precision, recall y PR AUC
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    ## Gráfico
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
    
    return pr_auc