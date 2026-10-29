# Predicción y clasificación del rendimiento de la caña de azúcar — Ingenio Providencia

Código fuente del taller (Jupyter Notebooks + módulos compartidos).

## Contenido

| Ruta | Contenido |
|---|---|
| `notebooks/01_eda_regresion.ipynb` | EDA de HISTORICO_SUERTES: nulos, ceros de lluvia como "sin dato", escala de lluvias, outliers. |
| `notebooks/02_split_pipeline_regresion.ipynb` | Split 80/20 agrupado por suerte y Pipeline de preprocesamiento de regresión. |
| `notebooks/03_benchmark_lineal_regresion.ipynb` | Regresión lineal múltiple (OLS con errores cluster-robustos, supuestos, VIF), Ridge y Lasso. |
| `notebooks/04_eda_clasificacion.ipynb` | BD_IPSA_1940: alcance, selección de predictoras (fuga de información), umbrales Bajo/Medio/Alto y nulos. |
| `notebooks/05_outliers_clasificacion.ipynb` | Outliers de clasificación con reglas fijas; integridad de `pct_diatrea`. |
| `notebooks/06_relaciones_vif_clasificacion.ipynb` | Relevancia (χ² y MI con permutación) y VIF, solo sobre train (split por hacienda). |
| `notebooks/07_pipeline_clasificacion.ipynb` | Pipeline de clasificación (reglas, imputación, escalado), anti-leakage y escalado de KNN. |
| `notebooks/08_modelado_clasificacion.ipynb` | Logística L1/L2 y KNN con `k` optimizado; CV frente a hold-out; verificaciones finales. |
| `notebooks/09_figuras_informe.ipynb` | Figuras del informe. |
| `src/` | Funciones compartidas: carga y split (`datos.py`), preprocesamiento (`preprocesamiento*.py`), modelos y evaluación. |
| `requirements.txt` | Versiones exactas del entorno (Python 3.14). |

Las decisiones de preprocesamiento y modelado, con su justificación, están en el informe y comentadas en cada
notebook. Los notebooks se entregan **ejecutados**, con sus salidas.

## Cómo reproducir

1. Crear un entorno con Python 3.14 e instalar las dependencias: `pip install -r requirements.txt`.
2. **Los datos no se incluyen**. Copiar los archivos originales en:
   - `data/raw/HISTORICO_SUERTES*`
   - `data/raw/BD_IPSA_1940*`
3. Ejecutar los notebooks desde la carpeta `notebooks/`. Cada uno corre de principio a fin por sí solo, porque lo
   compartido está en `src/`. El orden sugerido es 01 → 09: los splits son deterministas (`random_state=42`).
