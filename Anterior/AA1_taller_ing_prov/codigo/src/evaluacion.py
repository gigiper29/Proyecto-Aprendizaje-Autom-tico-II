"""Evaluación de modelos: regresión (métricas CV + hold-out, inferencia OLS, supuestos) y clasificación (CV + hold-out)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, make_scorer, mean_absolute_error,
                             precision_score, r2_score, recall_score, root_mean_squared_error)
from sklearn.model_selection import cross_validate
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

from .datos import CV_CLF, CV_GRUPOS

# Paleta de referencia (categórica, orden fijo) y tinta neutra para referencias
COLORES = {"serie1": "#2a78d6", "serie2": "#eb6834", "serie3": "#1baf7a", "tinta": "#52514e"}

METRICAS_CV = {"R2": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"}


def metricas(y, y_pred) -> dict:
    return {"R2": r2_score(y, y_pred), "RMSE": root_mean_squared_error(y, y_pred),
            "MAE": mean_absolute_error(y, y_pred)}


def evaluar(modelo, X_tr, y_tr, g_tr, X_te, y_te, cv=CV_GRUPOS) -> dict:
    """CV por grupos sobre train (media y desvío de los folds) + ajuste en todo train + métricas en test.

    `brecha_*` = CV − test: si el test es mucho peor que la CV, hay sobreajuste o el test difiere de train.
    """
    res = cross_validate(modelo, X_tr, y_tr, groups=g_tr, cv=cv, scoring=list(METRICAS_CV.values()))
    fila = {}
    for nombre, clave in METRICAS_CV.items():
        valores = res[f"test_{clave}"] * (-1 if clave.startswith("neg_") else 1)
        fila[f"{nombre}_cv"], fila[f"{nombre}_cv_sd"] = valores.mean(), valores.std()
    ajustado = clone(modelo).fit(X_tr, y_tr)
    y_pred = ajustado.predict(X_te)
    for nombre, valor in metricas(y_te, y_pred).items():
        fila[f"{nombre}_test"] = valor
        fila[f"brecha_{nombre}"] = fila[f"{nombre}_cv"] - valor
    return {"metricas": fila, "modelo": ajustado, "y_pred_test": y_pred}


def ajustar_ols(pre, X_tr, y_tr, g_tr):
    """OLS de statsmodels sobre la matriz que produce `pre` (ajustado aquí sobre train), para inferencia.

    Errores estándar cluster-robustos por suerte: los registros de una misma suerte (≈3.6) no son
    independientes, y además son robustos a heterocedasticidad. Devuelve (resultado OLS, Z de train).
    """
    Z = pre.fit_transform(X_tr).astype(float)
    exog = sm.add_constant(Z, has_constant="add")
    ols = sm.OLS(y_tr.to_numpy(), exog).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g_tr)[0]})
    return ols, Z


def tabla_coeficientes(ols, alfa=0.05) -> pd.DataFrame:
    ic = ols.conf_int(alfa)
    return pd.DataFrame({"coef": ols.params, "ee_cluster": ols.bse, "p_value": ols.pvalues,
                         "ic_inf": ic[0], "ic_sup": ic[1]}).drop(index="const")


def pruebas_supuestos(ols, n_shapiro=5000, semilla=42) -> pd.Series:
    """Pruebas formales de los supuestos. Con n de miles rechazan ante desviaciones mínimas: complementan
    la lectura visual, no la reemplazan."""
    resid = ols.resid
    exog = ols.model.exog
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, exog)
    jb, jb_p, asimetria, curtosis = jarque_bera(resid)
    rng = np.random.default_rng(semilla)
    sub = rng.choice(resid, size=min(n_shapiro, len(resid)), replace=False)
    sw, sw_p = stats.shapiro(sub)
    # RESET sobre un OLS clásico (sin cluster) con los mismos datos: agrega ŷ² y ŷ³
    reset = linear_reset(sm.OLS(ols.model.endog, exog).fit(), power=3, use_f=True)
    return pd.Series({
        "Breusch-Pagan LM": bp_lm, "Breusch-Pagan p": bp_p,
        "Jarque-Bera": jb, "Jarque-Bera p": jb_p, "asimetría": asimetria, "curtosis": curtosis,
        f"Shapiro W (n={len(sub)})": sw, "Shapiro p": sw_p,
        "RESET F": float(reset.fvalue), "RESET p": float(reset.pvalue),
    })


def vif(Z: pd.DataFrame) -> pd.Series:
    """VIF de cada columna de la matriz transformada (con constante, que no se reporta)."""
    A = sm.add_constant(Z.astype(float), has_constant="add").to_numpy()
    return pd.Series([variance_inflation_factor(A, i) for i in range(1, A.shape[1])],
                     index=Z.columns, name="VIF").sort_values(ascending=False)


def grafico_diagnostico(ols, y_te, y_pred_te, titulo, n_puntos=4000, semilla=42):
    """2×2: (a) residuos vs. ajustados + LOWESS [linealidad], (b) scale-location [homocedasticidad],
    (c) QQ-plot [normalidad], (d) real vs. predicho en test. Los paneles de dispersión muestran una
    submuestra de puntos para que se lean; LOWESS y QQ usan todos los residuos."""
    ajust, resid = ols.fittedvalues, ols.resid
    est = resid / resid.std()
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(resid), size=min(n_puntos, len(resid)), replace=False)
    nube = dict(s=6, alpha=0.25, color=COLORES["serie1"], edgecolors="none")
    curva = dict(color=COLORES["serie2"], lw=2)
    ref = dict(color=COLORES["tinta"], lw=1, ls="--")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    a = ax[0, 0]
    a.scatter(ajust[idx], resid[idx], **nube)
    a.plot(*lowess(resid, ajust, frac=0.3, it=0).T, **curva)
    a.axhline(0, **ref)
    a.set(title="(a) Residuos vs. ajustados (train)", xlabel="Ajustado", ylabel="Residuo")

    a = ax[0, 1]
    raiz = np.sqrt(np.abs(est))
    a.scatter(ajust[idx], raiz[idx], **nube)
    a.plot(*lowess(raiz, ajust, frac=0.3, it=0).T, **curva)
    a.set(title="(b) Scale-location (train)", xlabel="Ajustado", ylabel="√|residuo estandarizado|")

    a = ax[1, 0]
    (osm, osr), _ = stats.probplot(est, dist="norm")
    a.scatter(osm, osr, **nube)
    lim = [osm.min(), osm.max()]
    a.plot(lim, lim, **ref)
    a.set(title="(c) QQ-plot de residuos estandarizados", xlabel="Cuantil teórico N(0,1)", ylabel="Cuantil observado")

    a = ax[1, 1]
    a.scatter(y_te, y_pred_te, s=6, alpha=0.3, color=COLORES["serie3"], edgecolors="none")
    lim = [min(np.min(y_te), np.min(y_pred_te)), max(np.max(y_te), np.max(y_pred_te))]
    a.plot(lim, lim, **ref)
    a.set(title=f"(d) Real vs. predicho (test, R² = {r2_score(y_te, y_pred_te):.3f})", xlabel="Real", ylabel="Predicho")

    for a in ax.flat:
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    return fig


# --- Clasificación ------------------------------------------------------------------------------------------
# Kappa es la métrica principal (corrige el acierto por azar); recall de "Bajo" por el costo de no detectar un lote de bajo desempeño
METRICAS_CLF = {
    "Accuracy": "accuracy",
    "Precision": "precision_macro",
    "Recall": "recall_macro",
    "F1": "f1_macro",
    "Kappa": make_scorer(cohen_kappa_score),
    "Recall_Bajo": make_scorer(recall_score, labels=["Bajo"], average="macro"),
}


def metricas_clf(y, y_pred) -> dict:
    return {"Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y, y_pred, average="macro"),
            "F1": f1_score(y, y_pred, average="macro"),
            "Kappa": cohen_kappa_score(y, y_pred),
            "Recall_Bajo": recall_score(y, y_pred, labels=["Bajo"], average="macro")}


def evaluar_clf(modelo, X_tr, y_tr, g_tr, X_te, y_te, cv=CV_CLF) -> dict:
    """CV por lote sobre train (media y desvío de los folds) + ajuste en todo train + métricas en test.

    `brecha_*` = CV − test. Con hiperparámetros elegidos en la misma CV, la media de CV es algo optimista.
    """
    res = cross_validate(modelo, X_tr, y_tr, groups=g_tr, cv=cv, scoring=METRICAS_CLF)
    fila = {}
    for nombre in METRICAS_CLF:
        valores = res[f"test_{nombre}"]
        fila[f"{nombre}_cv"], fila[f"{nombre}_cv_sd"] = valores.mean(), valores.std()
    ajustado = clone(modelo).fit(X_tr, y_tr)
    y_pred = ajustado.predict(X_te)
    for nombre, valor in metricas_clf(y_te, y_pred).items():
        fila[f"{nombre}_test"] = valor
        fila[f"brecha_{nombre}"] = fila[f"{nombre}_cv"] - valor
    return {"metricas": fila, "modelo": ajustado, "y_pred_test": y_pred, "folds": res}
