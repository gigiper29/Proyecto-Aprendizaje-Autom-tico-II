"""Preprocesamiento de clasificación (BD_IPSA_1940), según las decisiones de los notebooks 04–07.

Mismo esquema que regresión (`preprocesamiento.py`), dentro de un `Pipeline`:
1. `ReglasNegocioClasificacion`: reglas fijas por fila (no aprende nada en `fit`). Deriva `mes_cosecha` y `anio`,
   recodifica "sin dato" como NaN y recorta las lluvias con el tope físico.
2. `ColumnTransformer`: todo lo que depende de los datos (medianas, escalado, categorías). Su `fit` solo ve
   `X_train`, o el train de cada fold en la CV.

`tipo="logistica"` usa todas las predictoras y el spline de `semsmad`. `tipo="knn"` usa el conjunto reducido del
target, con las numéricas estandarizadas y las dummies sin escalar. Los dos aplican `log1p(lluvias)`
(KNN no tiene selección de variables propia).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

from .preprocesamiento import NODOS_SEMANAS, TOPE_MM_MES, ImputadorMedianaGrupo

TOPE_DOSIS = 3  # producto único; dosis de 0.2–2.0 y ningún valor entre 2.0 y 9.0 (05 §4)
INICIO_VENTANA_LLUVIA = 10  # ventana INFERIDA de `lluvias`: del mes 10 al corte (05 §6; pregunta abierta)
# Spline de `edad` solo para la sensibilidad de sacarosa: nodos interiores [12, 13, 15] (06 §7)
NODOS_EDAD = np.array([10, 12, 13, 15, 22], dtype=float).reshape(-1, 1)

NUMERICAS = ["edad", "cortes", "anio"]
# Única fuente de verdad del indicador de lluvia: vale 1 cuando FALTA el dato (lluvias = 0 en el crudo)
FLAG_LLUVIA = "falta_dato_lluvia"
CATEGORICAS = ["grupo_tenencia", "mes_cosecha"]
# Conjunto reducido de KNN: sin las candidatas a descarte del EDA de relaciones/VIF con el split por hacienda
# (06 §3, con el split por hacienda; respaldado por el orden de salida de L1 en el notebook 08).
EXCLUIDAS_KNN = {"TCH": {"semsmad", "mes_cosecha", "dosismad"},
                 "sacarosa": {"cortes", "grupo_tenencia", "falta_dato_lluvia"}}


class ReglasNegocioClasificacion(TransformerMixin, BaseEstimator):
    """Reglas fijas aprobadas en los notebooks 04–05 (imputación y outliers de clasificación). Sin estado.

    Con `tasa_lluvia=True` (sensibilidad de sacarosa), `lluvias` pasa a ser mm/mes de la ventana inferida.
    """

    def __init__(self, tasa_lluvia=False):
        self.tasa_lluvia = tasa_lluvia

    def fit(self, X, y=None):
        self.columnas_salida_ = self._aplicar(X.head(1)).columns.to_numpy(dtype=object)
        return self

    def transform(self, X):
        return self._aplicar(X)

    def get_feature_names_out(self, input_features=None):
        return self.columnas_salida_

    def _aplicar(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Derivadas: mes_cosecha = mes (categórica, sin recodificar), anio = periodo // 100
        X["anio"] = X["periodo"] // 100
        X["mes_cosecha"] = X["mes"]
        X = X.drop(columns=["mes", "periodo"])

        # dosis 0 en un registro madurado = error de captura; > 3 = error de captura (hueco 2.0–9.0) → NaN
        dosis = X["dosismad"].astype(float)
        X["dosismad"] = dosis.mask((dosis == 0) | (dosis > TOPE_DOSIS))
        # Semanas negativas: el madurante figura aplicado después del corte (error de fecha) → NaN
        semanas = X["semsmad"].astype(float)
        X["semsmad"] = semanas.mask(semanas < 0)

        # Lluvias: 0 = sin dato → NaN; recorte a 300 mm/mes × (edad − 10), solo en ventanas de ≥ 1 mes
        ventana = X["edad"] - INICIO_VENTANA_LLUVIA
        lluvias = X["lluvias"].astype(float).replace(0, np.nan)
        lluvias = lluvias.clip(upper=(TOPE_MM_MES * ventana).where(ventana >= 1))
        if self.tasa_lluvia:
            # Sensibilidad: tasa mm/mes; con ventana < 1 mes la tasa no es fiable → NaN (se imputa)
            lluvias = lluvias / ventana.where(ventana >= 1)
        X["lluvias"] = lluvias
        return X


def _falta_dato(X):
    """1 cuando falta el dato de lluvia (funciones de módulo, para que el Pipeline se pueda serializar)."""
    return X.isna().astype(int).set_axis([FLAG_LLUVIA], axis=1)


def _nombre_flag_lluvia(transformer, input_features):
    return np.array([FLAG_LLUVIA], dtype=object)


def construir_preprocesador_clf(tipo="logistica", target=None, edad_spline=False, tasa_lluvia=False) -> Pipeline:
    """Pipeline de preprocesamiento de clasificación.

    - `tipo="logistica"`: todas las predictoras; `semsmad` con spline de nodos fijos; OneHot con `drop="first"`
      (mismo diseño que el VIF del notebook 06). Las sensibilidades `edad_spline` y `tasa_lluvia` son solo para sacarosa.
    - `tipo="knn"`: conjunto reducido de `target` (EXCLUIDAS_KNN); numéricas con StandardScaler ajustado en train;
      dummies OneHot completas y sin escalar (escalarlas no mejora la CV, notebook 07 §3.1).
    En los dos, `log1p(lluvias)` y el indicador `falta_dato_lluvia` (1 = falta el dato).
    """
    logistica = tipo == "logistica"
    if not logistica and target not in EXCLUIDAS_KNN:
        raise ValueError("KNN necesita target='TCH' o 'sacarosa' para elegir su conjunto de predictoras")
    excluidas = set() if logistica else EXCLUIDAS_KNN[target]
    usar = lambda c: c not in excluidas

    # log1p en los dos modelos: la fila de outliers aprobada ("Conservar + log1p") no lo limita a la logística
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    # 1 = falta el dato de lluvia. Es una regla por fila (no aprende nada), con nombre explícito
    falta_lluvia = FunctionTransformer(_falta_dato, feature_names_out=_nombre_flag_lluvia)
    # Mediana de train por grupo de tenencia × mes; respaldo: mes; último respaldo: mediana global de train
    imp_lluvia = ImputadorMedianaGrupo(["lluvias"], niveles=(("grupo_tenencia", "mes_cosecha"), ("mes_cosecha",)))
    if logistica:
        # Sin `include_bias`: las bases sumarían 1 y repetirían el intercepto
        semanas = make_pipeline(SimpleImputer(strategy="median"),
                                SplineTransformer(knots=NODOS_SEMANAS, extrapolation="constant", include_bias=False))
    else:
        semanas = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    if logistica and edad_spline:
        edad = SplineTransformer(knots=NODOS_EDAD, extrapolation="constant", include_bias=False)
    else:
        edad = StandardScaler()

    bloques = [
        ("lluvias", make_pipeline(imp_lluvia, log1p, StandardScaler()), ["lluvias", "grupo_tenencia", "mes_cosecha"]),
        ("edad", edad, ["edad"]),
        ("numericas", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         [c for c in NUMERICAS if c != "edad" and usar(c)]),
        ("categoricas", OneHotEncoder(handle_unknown="ignore", drop="first" if logistica else None,
                                      sparse_output=False), [c for c in CATEGORICAS if usar(c)]),
    ]
    if usar(FLAG_LLUVIA):
        bloques.append((FLAG_LLUVIA, falta_lluvia, ["lluvias"]))
    if usar("dosismad"):
        bloques.append(("dosis", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["dosismad"]))
    if usar("semsmad"):
        bloques.append(("semanas", semanas, ["semsmad"]))

    return Pipeline([
        ("reglas", ReglasNegocioClasificacion(tasa_lluvia=tasa_lluvia)),
        ("columnas", ColumnTransformer(bloques, remainder="drop", verbose_feature_names_out=False)),
    ]).set_output(transform="pandas")
