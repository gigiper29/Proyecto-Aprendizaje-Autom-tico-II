"""Preprocesamiento de regresión (HISTORICO_SUERTES), según las decisiones del EDA (notebook 01).

Dos pasos dentro de un mismo `Pipeline`:
1. `ReglasNegocio`: reglas fijas por fila (no aprende nada en `fit`). Corrige errores de captura,
   recodifica "sin dato" como NaN, recorta lluvias con el tope físico y deriva flags.
2. `ColumnTransformer`: todo lo que depende de los datos (medianas, categorías poco frecuentes,
   escalado). Su `fit` solo ve `X_train`, o el train de cada fold en la CV.
"""
import unicodedata

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

from .datos import LLUVIAS_VENTANA, TOPE_PRECOSECHA

LLUVIA_2M = "Lluvias (2 Meses Ant.)"
LLUVIA_FC = "Luvias 9 -FC"
LLUVIAS_TRIMESTRALES = ["Lluvias 0 -3", "Lluvias tres a seis", "Lluvias seis a nueve"]
TOPE_MM_MES = 300  # límite físico: 900 mm/trimestre (regla de negocio de referencia, a escala trimestral, §7 del EDA)
SEMANAS_POR_MES = 30.44 / 7
# Tope ESTADÍSTICO (no agronómico: ningún documento del proyecto fija una duración máxima del ciclo) para
# `Edad Ult Cos`: por encima de la cola de cosechas postergadas (máx. 33.2 meses tras la fila de 78.2).
TOPE_EDAD_MESES = 36

NUMERICAS = ["Edad Ult Cos", "Cod.Estado #", "Dist Km", "anio", "M3 Riego"]
CATEGORICAS = ["Variedad", "Suelo", "Zona", "Producto", "T.Corte", "Tenencia", "Cultivo", "mes_cosecha"]
FLAGS = ["madurada", "tiene_registro_riego", "tiene_dato_climatico"]
PRECOSECHA = "Sac.Caña Precosecha"

# Nodos fijos para la U invertida de `Semanas mad.` (óptimo esperado en 8–16 semanas según la literatura de maduración). No dependen de los datos.
# Último nodo en 40 (no 100): entre 40 y 100 semanas hay muy pocos datos. Más allá de 40, `extrapolation="constant"`
# mantiene el efecto en el valor de la semana 40.
NODOS_SEMANAS = np.array([0, 6, 12, 18, 30, 40], dtype=float).reshape(-1, 1)


def _normalizar_texto(s: pd.Series) -> pd.Series:
    """Quita acentos, pasa a mayúsculas y colapsa espacios (BONUS … FISIOLÓGICO = FISIOLOGICO)."""
    def limpiar(v):
        v = unicodedata.normalize("NFKD", v).encode("ascii", "ignore").decode()
        return " ".join(v.upper().split())
    return s.map(limpiar, na_action="ignore")


class ReglasNegocio(TransformerMixin, BaseEstimator):
    """Reglas fijas aprobadas en el EDA (imputación "no aplicado" y outliers). Sin estado."""

    def fit(self, X, y=None):
        self.columnas_salida_ = self._aplicar(X.head(1)).columns.to_numpy(dtype=object)
        return self

    def transform(self, X):
        return self._aplicar(X)

    def get_feature_names_out(self, input_features=None):
        return self.columnas_salida_

    @staticmethod
    def _aplicar(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        edad = X["Edad Ult Cos"]

        # Derivadas de fecha y del vacío climático Cenicaña
        X["anio"] = X["Período"] // 100
        X["mes_cosecha"] = X["Período"] % 100
        X["tiene_dato_climatico"] = X["Temp. Media Ciclo"].notna().astype(int)
        X = X.drop(columns=["Período", "Temp. Media Ciclo"])

        X["Zona"] = X["Zona"].str.upper()
        X["Suelo"] = X["Suelo"].str.strip().fillna("Desconocido")
        X["Tenencia"] = X["Tenencia"].map(lambda v: str(int(v)), na_action="ignore")

        # Bloque madurante: vacío = "no aplicado"; con dosis pero sin nombre = "DESCONOCIDO"
        producto = _normalizar_texto(X["Producto"])
        dosis = X["Dosis Madurante"]
        producto = producto.mask(producto.isna() & (dosis > 0), "DESCONOCIDO").fillna("NO APLICADO")
        madurada = producto != "NO APLICADO"
        X["Producto"] = producto
        X["madurada"] = madurada.astype(int)
        # Producto informado con dosis 0 → error de captura → NaN (mediana del producto en train)
        X["Dosis Madurante"] = dosis.mask(madurada & (dosis == 0)).where(madurada, 0.0)
        # Semanas negativas o más largas que el ciclo → error de fecha → NaN (mediana de madurados en train)
        semanas = X["Semanas mad."]
        fecha_imposible = (semanas < 0) | (semanas > edad * SEMANAS_POR_MES)
        X["Semanas mad."] = semanas.mask(fecha_imposible).where(madurada, 0.0)

        # Lluvias: 0 = sin dato → NaN; recorte al tope físico de 300 mm/mes
        lluvias = X[LLUVIAS_VENTANA].astype(float).replace(0, np.nan)
        lluvias[LLUVIAS_TRIMESTRALES] = lluvias[LLUVIAS_TRIMESTRALES].clip(upper=3 * TOPE_MM_MES)
        lluvias[LLUVIA_2M] = lluvias[LLUVIA_2M].clip(upper=2 * TOPE_MM_MES)
        meses_fc = edad - 9
        # La tasa mm/mes solo se evalúa con ventanas de ≥ 1 mes (Edad redondeada a décimas)
        tope_fc = (TOPE_MM_MES * meses_fc).where(meses_fc >= 1)
        lluvias[LLUVIA_FC] = lluvias[LLUVIA_FC].clip(upper=tope_fc)
        # Edad ≤ 9: la ventana 9-FC no existe → 0 real, no "sin dato" (incluye las 23 incoherentes)
        lluvias.loc[meses_fc <= 0, LLUVIA_FC] = 0.0
        X[LLUVIAS_VENTANA] = lluvias

        # Riego: el 0 de M3 se conserva (no se distingue "no regado" de "sin registro") + flag
        X["tiene_registro_riego"] = (X["M3 Riego"] > 0).astype(int)
        # DDUlt = 0 sin riego registrado es incoherente → NaN. Con riego, el 0 es real (9 filas regadas el día del corte)
        X["DDUlt.Riego"] = X["DDUlt.Riego"].astype(float).mask(X["M3 Riego"] == 0)

        # Al final: las reglas de coherencia de arriba usan la edad real (duración del ciclo según las fechas)
        X["Edad Ult Cos"] = edad.clip(upper=TOPE_EDAD_MESES)

        if PRECOSECHA in X:
            # Sin dato: 0 = "no muestreado" y > TOPE_PRECOSECHA = error de captura. Una sola máscara para las dos colas
            pre = X[PRECOSECHA].astype(float)
            X[PRECOSECHA] = pre.mask((pre == 0) | (pre > TOPE_PRECOSECHA))
        return X


class ImputadorMedianaGrupo(TransformerMixin, BaseEstimator):
    """Imputa `columnas` con la mediana de train por grupo, con respaldo jerárquico.

    `niveles` va del grupo más fino al más grueso; el último respaldo es la mediana global de
    train. Con `filtro`, las medianas se calculan solo sobre las filas donde esa columna vale 1
    (p. ej. `madurada`), para que los ceros de "no aplicado" no las arrastren.
    """

    def __init__(self, columnas, niveles=(), filtro=None):
        self.columnas = columnas
        self.niveles = niveles
        self.filtro = filtro

    def fit(self, X, y=None):
        cols = list(self.columnas)
        base = X[X[self.filtro] == 1] if self.filtro else X
        self.medianas_ = [base.groupby(list(nivel))[cols].median() for nivel in self.niveles]
        self.mediana_global_ = base[cols].median()
        return self

    def transform(self, X):
        out = X[list(self.columnas)].copy()
        for nivel, medianas in zip(self.niveles, self.medianas_):
            nivel = list(nivel)
            clave = pd.MultiIndex.from_frame(X[nivel]) if len(nivel) > 1 else pd.Index(X[nivel[0]])
            out = out.fillna(medianas.reindex(clave).set_axis(X.index))
        return out.fillna(self.mediana_global_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columnas, dtype=object)


def construir_preprocesador(tipo="lineal", con_precosecha=False) -> Pipeline:
    """Pipeline de preprocesamiento. `tipo="lineal"` agrega log1p, spline y escalado;
    `tipo="arbol"` deja las numéricas en su escala. `con_precosecha` es solo para %Sac.Caña."""
    lineal = tipo == "lineal"
    escalar = [StandardScaler()] if lineal else []

    def con(*pasos):
        return make_pipeline(*pasos, *escalar)

    log1p = [FunctionTransformer(np.log1p, feature_names_out="one-to-one")] if lineal else []
    imp_lluvia = ImputadorMedianaGrupo(LLUVIAS_VENTANA, niveles=(("Zona", "mes_cosecha"), ("Zona",)))
    imp_dosis = ImputadorMedianaGrupo(["Dosis Madurante"], niveles=(("Producto",),), filtro="madurada")
    imp_semanas = ImputadorMedianaGrupo(["Semanas mad."], filtro="madurada")
    # Sin `include_bias`: las bases del spline sumarían 1 y repetirían el intercepto (colinealidad exacta)
    semanas = (make_pipeline(imp_semanas, SplineTransformer(knots=NODOS_SEMANAS, extrapolation="constant",
                                                            include_bias=False))
               if lineal else imp_semanas)
    # En el lineal, `madurada` es función exacta de `Producto` (0 ⇔ "NO APLICADO"): ya la cubren sus dummies
    flags = [f for f in FLAGS if f != "madurada"] if lineal else FLAGS

    bloques = [
        ("lluvias", con(imp_lluvia, *log1p), LLUVIAS_VENTANA + ["Zona", "mes_cosecha"]),
        ("sin_dato_lluvia", MissingIndicator(features="missing-only"), LLUVIAS_VENTANA),
        ("dosis", con(imp_dosis), ["Dosis Madurante", "Producto", "madurada"]),
        ("semanas", semanas, ["Semanas mad.", "madurada"]),
        ("dd_riego", con(SimpleImputer(strategy="median")), ["DDUlt.Riego"]),
        # Dist Km (2), Tenencia (1) y Cultivo (2) tienen nulos aislados: mediana / moda de train
        ("numericas", con(SimpleImputer(strategy="median")), NUMERICAS),
        ("categoricas", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(min_frequency=0.01, handle_unknown="infrequent_if_exist",
                          drop="first" if lineal else None, sparse_output=False)), CATEGORICAS),
        ("flags", "passthrough", flags),
    ]
    if con_precosecha:
        bloques.append(("precosecha", StandardScaler() if lineal else "passthrough", [PRECOSECHA]))

    pasos = [
        ("reglas", ReglasNegocio()),
        ("columnas", ColumnTransformer(bloques, remainder="drop", verbose_feature_names_out=False)),
    ]
    if lineal:
        # Quita columnas constantes en train (p. ej. `tiene_dato_climatico` = 0 en todo el subconjunto 2017–2021)
        pasos.append(("sin_constantes", VarianceThreshold()))
    return Pipeline(pasos).set_output(transform="pandas")
