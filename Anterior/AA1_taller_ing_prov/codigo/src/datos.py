"""Carga de datos y listas de variables compartidas entre notebooks."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
RUTA_HISTORICO = RAIZ / "data" / "raw" / "HISTORICO_SUERTES.csv"

# Lluvias por ventana del ciclo (el efecto depende del momento del ciclo). `Lluvias Ciclo` es su suma: solo para EDA/chequeos.
LLUVIAS_VENTANA = [
    "Lluvias (2 Meses Ant.)",
    "Lluvias 0 -3",
    "Lluvias tres a seis",
    "Lluvias seis a nueve",
    "Luvias 9 -FC",  # sic: así viene en el CSV
]

# Predictoras crudas seleccionadas para regresión (solo variables conocidas antes de la cosecha). Las derivadas
# (`madurada`, `mes_cosecha`, `anio`, `tiene_dato_climatico`) se construyen aparte.
PREDICTORAS_REGRESION = [
    "Edad Ult Cos",
    "Cod.Estado #",
    "Variedad",
    "Suelo",
    "Zona",
    "Dist Km",
    "Producto",
    "Dosis Madurante",
    "Semanas mad.",
    "T.Corte",
    "Tenencia",
    "Cultivo",
    *LLUVIAS_VENTANA,
    "M3 Riego",
    "DDUlt.Riego",
    "Período",
]

# Solo para el target %Sac.Caña (y solo en el subconjunto con dato real: el 0 significa "no muestreado").
PREDICTORAS_SOLO_SAC = ["Sac.Caña Precosecha"]
# Rango con dato real de la precosecha. 0 = "no muestreado"; > 25 = error de captura (10 filas de 33.7–44.8, sin
# valores entre 22 y 33.7; una sacarosa así es imposible). Las dos colas son "sin dato" y no se imputan (imputar un proxy del target metería un valor sintético).
TOPE_PRECOSECHA = 25

# Columnas que solo usa `ReglasNegocio` para derivar flags; no llegan al modelo.
COLUMNAS_AUXILIARES = ["Temp. Media Ciclo"]  # → tiene_dato_climatico

TARGETS_REGRESION = ["TCH", "%Sac.Caña"]

# CV sobre train: los registros de una misma suerte nunca se reparten entre folds.
CV_GRUPOS = GroupKFold(n_splits=5)


def cargar_historico(ruta=RUTA_HISTORICO) -> pd.DataFrame:
    """Lee HISTORICO_SUERTES y excluye las cosechas de semilla (Destino 1=Semilla == 1)."""
    df = pd.read_csv(ruta, encoding="utf-8-sig", low_memory=False)
    return df[df["Destino 1=Semilla"] != 1].reset_index(drop=True)


def filtrar_filas(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las filas con TCH < 50 o > 250 en suertes de ≤ 1 ha (error del denominador: la tasa de extremos sube en los lotes pequeños).

    Es una regla fija por fila (no usa estadísticos), por eso se aplica antes del split y
    afecta a train y a test por igual.
    """
    error_area = ((df["TCH"] < 50) | (df["TCH"] > 250)) & (df["Area Neta"] <= 1)
    return df[~error_area].reset_index(drop=True)


def grupos(df: pd.DataFrame) -> pd.Series:
    """Identificador de lote. `Suerte` sola se repite entre haciendas, así que la clave es Hacienda + Suerte."""
    return (df["Hacienda"].astype(str) + "_" + df["Suerte"].astype(str).str.strip()).rename("id_suerte")


def split_regresion(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Hold-out 80/20 por grupos: ninguna suerte queda a la vez en train y en test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_train, idx_test = next(gss.split(df, groups=grupos(df)))
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_test].reset_index(drop=True)


def particiones_regresion():
    """Carga → filtros fijos → split 80/20 por suerte (mismo flujo que verifica el notebook 02)."""
    return split_regresion(filtrar_filas(cargar_historico()))


def preparar_xy(df: pd.DataFrame, target: str, con_precosecha=False):
    """Devuelve X (columnas crudas que espera el preprocesador), y y los grupos para GroupKFold.

    Elimina las filas sin target (el target no se imputa). Con `con_precosecha`, se restringe
    al subconjunto con dato real de `Sac.Caña Precosecha` (2017–2021, 0 < valor ≤ TOPE_PRECOSECHA).
    """
    columnas = PREDICTORAS_REGRESION + COLUMNAS_AUXILIARES
    mascara = df[target].notna()
    if con_precosecha:
        columnas = columnas + PREDICTORAS_SOLO_SAC
        mascara &= (df["Período"] // 100 <= 2021) & df["Sac.Caña Precosecha"].between(0, TOPE_PRECOSECHA, inclusive="right")
    d = df[mascara].reset_index(drop=True)
    return d[columnas], d[target], grupos(d)


# --- Clasificación (BD_IPSA_1940) -------------------------------------------------------------------
# Alcance: solo CC01-1940 madurada con BONUS 250 EC y cosechada a máquina (las 4 columnas son constantes).
RUTA_IPSA = RAIZ / "data" / "raw" / "BD_IPSA_1940.csv"

# Predictoras crudas seleccionadas para clasificación (solo variables conocidas antes de la cosecha). Son las columnas DE ENTRADA, no la matriz X
# final: `mes` y `periodo` solo sirven para derivar `mes_cosecha` (= `mes`, categórica) y `anio` (= `periodo // 100`)
# en el transformador de reglas de negocio del Pipeline, como `Período` en regresión. No usar esta lista directamente
# como X. Excluidas: `me` (= %ME Tot, leakage), `vejez` (poscorte), `pct_diatrea` (integridad del dato: ~75 % de
# relleno no identificable: solo ~25 % tiene medición real), las 4 constantes e identificadores.
PREDICTORAS_CLASIFICACION = [
    "edad",
    "cortes",
    "semsmad",
    "dosismad",
    "lluvias",
    "grupo_tenencia",
    "mes",
    "periodo",
]
COLUMNAS_CONSTANTES_IPSA = ["variedad", "madurada", "producto", "tipocorte"]
TARGETS_CLASIFICACION = ["TCH", "sacarosa"]

# Umbrales de negocio de referencia del Ingenio, fijos: no dependen del train. Bordes de `pd.cut` con right=False
# → [−inf, b1) Bajo, [b1, b2) Medio, [b2, inf) Alto. TCH viene en enteros: Medio = 125–150, Alto > 150.
# Sacarosa viene con resolución 0.1: Medio = 12.2–12.9, Alto ≥ 13.0 (el hueco 12.9–13 de la referencia no tiene filas).
UMBRALES_CLASIFICACION = {"TCH": (125, 151), "sacarosa": (12.2, 13.0)}
CLASES = ["Bajo", "Medio", "Alto"]


def cargar_ipsa(ruta=RUTA_IPSA) -> pd.DataFrame:
    """Lee BD_IPSA_1940 (la primera columna es un índice sin nombre). `TAL` queda como texto sin espacios."""
    df = pd.read_csv(ruta, encoding="utf-8-sig", index_col=0)
    df["TAL"] = df["TAL"].astype(str).str.strip()
    return df.reset_index(drop=True)


def grupos_ipsa(df: pd.DataFrame) -> pd.Series:
    """Identificador de lote en BD_IPSA: hacienda (FAZ) + suerte (TAL). Solo descriptivo (el split agrupa por hacienda)."""
    return (df["FAZ"].astype(str) + "_" + df["TAL"]).rename("id_suerte")


def grupos_hacienda_ipsa(df: pd.DataFrame) -> pd.Series:
    """Grupo del split y de la CV de clasificación: la hacienda (FAZ, 285 grupos).

    Agrupar por lote dejaba "hermanos" (misma hacienda y período, predictoras casi idénticas) a ambos lados del corte
    y KNN los memorizaba (el resultado de un lote vecino cosechado el mismo mes no se conoce antes de cosechar). Cada hacienda queda entera en train o en test.
    """
    return df["FAZ"].astype(str).rename("id_hacienda")


def clase_combinada(df: pd.DataFrame) -> pd.Series:
    """Clase TCH × sacarosa (9 estratos), para estratificar el split y la CV de clasificación."""
    return categorizar(df["TCH"], "TCH").astype(str) + "_" + categorizar(df["sacarosa"], "sacarosa").astype(str)


def split_clasificacion(df: pd.DataFrame, random_state=42):
    """Hold-out 80/20 por hacienda (FAZ), estratificado por la clase combinada TCH × sacarosa.

    `StratifiedGroupKFold(5)`; el test es el fold 0, el primero del método y sin comparar alternativas. Reemplaza,
    solo en clasificación, el GroupShuffleSplit de regresión, que dejaba la sacarosa desbalanceada en test (Medio 31 % frente a 38 % en train, χ² p = 0.014).
    Agrupa por hacienda y no por lote (FAZ + TAL). Es el split definitivo del ciclo de
    clasificación; si cambia el random_state o el criterio de grupo, hay que volver a correr los notebooks 06 y 07.
    """
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    idx_train, idx_test = next(sgk.split(df, clase_combinada(df), groups=grupos_hacienda_ipsa(df)))
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_test].reset_index(drop=True)


# CV sobre train de clasificación: por hacienda (grupos de `preparar_xy_clf`) y estratificada por la clase del target
CV_CLF = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


def preparar_xy_clf(df: pd.DataFrame, target: str):
    """Devuelve X (columnas de entrada que espera el preprocesador), la clase Bajo/Medio/Alto y la hacienda (grupo)."""
    return df[PREDICTORAS_CLASIFICACION], categorizar(df[target], target).astype(str), grupos_hacienda_ipsa(df)


def categorizar(y: pd.Series, target: str) -> pd.Series:
    """Convierte TCH o sacarosa en Bajo/Medio/Alto con los umbrales fijos de `UMBRALES_CLASIFICACION`."""
    b1, b2 = UMBRALES_CLASIFICACION[target]
    # Redondeo a la resolución del dato para evitar que 12.2 se lea como 12.19999 en el borde.
    return pd.cut(y.round(1), [-float("inf"), b1, b2, float("inf")], right=False, labels=CLASES)
