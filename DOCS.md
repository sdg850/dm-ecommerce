# E-Commerce Analytics — Documentación Técnica

Sistema de Machine Learning para análisis de clientes en e-commerce. Integra tres modelos de ML (supervisados y no supervisados) exportados como archivos `.pkl` y expuestos a través de un pipeline Python y un dashboard Streamlit.

---

## Tabla de Contenidos

1. [Visión General](#1-visión-general)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Estructura de Archivos](#3-estructura-de-archivos)
4. [Dataset](#4-dataset)
5. [Modelos ML](#5-modelos-ml)
   - [Modelo 1 — Segmentación RFM](#modelo-1--segmentación-rfm)
   - [Modelo 2 — Prevención de Churn](#modelo-2--prevención-de-churn)
   - [Modelo 3 — Predicción de LTV](#modelo-3--predicción-de-ltv)
6. [Pipeline de Predicción](#6-pipeline-de-predicción-apppy)
7. [Dashboard Streamlit](#7-dashboard-streamlit-streamlit_apppy)
8. [Stack Tecnológico](#8-stack-tecnológico)
9. [Instalación y Configuración](#9-instalación-y-configuración)
10. [Criterios de Éxito por Modelo](#10-criterios-de-éxito-por-modelo)

---

## 1. Visión General

El sistema responde tres preguntas de negocio centrales:

| # | Pregunta | Tipo de Modelo | Técnica |
|---|----------|---------------|---------|
| 1 | ¿Quiénes son mis clientes y cómo se agrupan? | Descriptivo / No supervisado | Clustering K-Means (RFM) |
| 2 | ¿Qué clientes están en riesgo de abandonar? | Predictivo / Supervisado | Clasificación (Churn) |
| 3 | ¿Cuánto vale cada cliente a largo plazo? | Predictivo / Supervisado | Regresión (LTV) |

Los modelos se entrenan en Jupyter Notebooks, se exportan como archivos `.pkl` y se consumen por un pipeline Python que alimenta un dashboard Streamlit interactivo.

---

## 2. Arquitectura del Sistema

```
CSV Dataset
    │
    ▼
colabsFiles/DataCleaningProcess.ipynb   ← Limpieza y exploración del dataset
colabsFiles/data_cleaning.py            ← DataCleaner (clase reutilizable)
    │
    ├── colabsFiles/model_rfm.ipynb     → models/rfm_model.pkl
    ├── colabsFiles/model_churn.ipynb   → models/churn_model.pkl
    └── colabsFiles/model_ltv.ipynb     → models/ltv_model.pkl
                                                │
                                                ▼
                                           app.py
                                      EcommercePipeline
                                   (carga los 3 pkl y predice)
                                                │
                                                ▼
                                      streamlit_app.py
                                    Dashboard interactivo
```

**Flujo de datos:**
1. CSV raw → `DataCleaner` → dataset limpio
2. Notebooks de entrenamiento → modelos `.pkl` en `models/`
3. `EcommercePipeline.predict_all(customer_data)` → RFM + Churn + LTV
4. Streamlit dashboard → visualización interactiva de predicciones
           │                  │                  │

---

## 3. Estructura de Archivos

```
ux-project/
├── README.md                        ← Guía de inicio rápido
├── DOCS.md                          ← Esta documentación técnica
├── requirements.txt                 ← Dependencias pip
├── environment.yml                  ← Entorno Conda (ecenv, Python 3.11)
├── app.py                           ← EcommercePipeline (carga pkl, predice)
├── streamlit_app.py                 ← Dashboard Streamlit
│
├── colabsFiles/
│   ├── data_cleaning.py             ← DataCleaner (clase reutilizable)
│   ├── DataCleaningProcess.ipynb    ← Exploración y limpieza del dataset
│   ├── model_rfm.ipynb              ← Entrenamiento Modelo 1 (RFM)
│   ├── model_churn.ipynb            ← Entrenamiento Modelo 2 (Churn)
│   └── model_ltv.ipynb              ← Entrenamiento Modelo 3 (LTV)
│
├── models/
│   ├── rfm_model.pkl                ← Modelo RFM exportado
│   ├── churn_model.pkl              ← Modelo Churn exportado
│   └── ltv_model.pkl                ← Modelo LTV exportado
│
└── data-base/
    ├── ecommerce_customer_churn_dataset.csv       ← Dataset principal (raw)
    ├── clean_ecommerce_customer_churn_dataset.csv ← Dataset limpio
    └── ecommerce_enriched_predictions.csv         ← Dataset con predicciones
```

---

## 4. Dataset

**Archivo:** `data-base/ecommerce_customer_churn_dataset.csv`

| Columna | Tipo | Uso en modelos |
|---------|------|---------------|
| `Days_Since_Last_Purchase` | Numérico | RFM (Recencia), Churn |
| `Total_Purchases` | Numérico | RFM (Frecuencia), LTV |
| `Login_Frequency` | Numérico | RFM (Frecuencia), Churn, LTV |
| `Average_Order_Value` | Numérico | RFM (Monetario), Churn, LTV |
| `Lifetime_Value` | Numérico | RFM (Monetario), **target LTV** |
| `Cart_Abandonment_Rate` | Numérico (%) | Churn |
| `Returns_Rate` | Numérico | Churn, LTV |
| `Customer_Service_Calls` | Entero | Churn |
| `Discount_Usage_Rate` | Numérico (%) | Churn, LTV |
| `Session_Duration_Avg` | Numérico | Churn, LTV |
| `Pages_Per_Session` | Numérico | Churn, LTV |
| `Email_Open_Rate` | Numérico (%) | Churn, LTV |
| `Social_Media_Engagement_Score` | Numérico | Churn |
| `Membership_Years` | Numérico | Churn, LTV |
| `Churned` | Binario (0/1) | **target Churn** |
| `Age`, `Gender`, `Country`, `City` | Categórico | No usados en modelos actuales |

**Preprocesamiento:** Imputación de medianas en todas las columnas numéricas con valores nulos. No se eliminan filas.

**Dataset enriquecido de salida:** `data-base/ecommerce_enriched_predictions.csv`  
Contiene todas las columnas originales más: `R_Score`, `F_Score`, `M_Score`, `RFM_Cluster`, `RFM_Segment`, `Churn_Prob`, `Churn_Risk`, `Predicted_LTV`, `LTV_Tier`.

---

## 5. Modelos ML

### Modelo 1 — Segmentación RFM (`model_rfm.py`)

**Clase:** `RFMSegmentation(BaseMLModel)`  
**Tipo:** No supervisado — Clustering

#### Ingeniería de Variables

```python
R_Score = max(Days_Since_Last_Purchase) - Days_Since_Last_Purchase  # recencia invertida
F_Score = Total_Purchases × 0.65 + Login_Frequency × 0.35          # frecuencia ponderada
M_Score = Average_Order_Value × 0.40 + Lifetime_Value × 0.60       # monetario ponderado
```

Las tres variables se normalizan con `MinMaxScaler` al rango [0, 1] antes del clustering.

#### Selección Óptima de K

1. Se evalúan K = 2, 3, …, 10 con `KMeans(random_state=42, n_init=10)`
2. Se calcula **WCSS** (Within-Cluster Sum of Squares) y **Silhouette Score** para cada K
3. El K óptimo del codo se detecta via la **máxima segunda derivada** de WCSS
4. Si el Silhouette en el codo < 0.35, se usa el K con máximo Silhouette Score global

#### Nomenclatura de Segmentos

Los clusters se ordenan por score compuesto (`R×0.30 + F×0.40 + M×0.30`) y se asignan etiquetas:

| Rango | Segmento | Estrategia |
|-------|----------|-----------|
| 1° | Champions | Programa VIP, embajadores de marca |
| 2° | Clientes Leales | Newsletter personalizado, descuentos por volumen |
| 3° | Potenciales | Nurturing activo, categorías nuevas |
| 4° | En Riesgo | Campaña de win-back urgente |
| 5° | Perdidos/Inactivos | Recuperación con beneficio exclusivo |

#### Columnas añadidas al dataset enriquecido

| Columna | Descripción |
|---------|-------------|
| `R_Score` | Score de recencia |
| `F_Score` | Score de frecuencia |
| `M_Score` | Score monetario |
| `RFM_Cluster` | ID numérico del cluster (0, 1, …, K-1) |
| `RFM_Segment` | Nombre del segmento |

#### Criterios de Éxito

| Métrica | Umbral |
|---------|--------|
| Silhouette Score | > 0.45 |
| Codo en WCSS | Claro (segunda derivada > 0) |
| Cobertura de segmentos | Todos > 5% de la base |

---

### Modelo 2 — Prevención de Churn (`model_churn.py`)

**Clase:** `ChurnPrevention(BaseMLModel)`  
**Tipo:** Supervisado — Clasificación binaria  
**Target:** `Churned` (0 = activo, 1 = fugado)

#### Features Utilizadas (12)

```
Login_Frequency, Cart_Abandonment_Rate, Returns_Rate,
Customer_Service_Calls, Days_Since_Last_Purchase, Session_Duration_Avg,
Pages_Per_Session, Discount_Usage_Rate, Email_Open_Rate,
Social_Media_Engagement_Score, Membership_Years, Average_Order_Value
```

#### Manejo del Desbalance de Clases

- **Random Forest / Logistic Regression:** `class_weight='balanced'`
- **XGBoost:** `scale_pos_weight = n_negativos / n_positivos`
- **Gradient Boosting:** sin ajuste nativo; compensado por la métrica de selección

#### Modelos Evaluados

| Algoritmo | Escalado necesario | Parámetros clave |
|-----------|-------------------|-----------------|
| Random Forest | No | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=5` |
| Gradient Boosting | No | `n_estimators=200`, `learning_rate=0.05`, `subsample=0.8` |
| Logistic Regression | Sí (StandardScaler) | `C=1.0`, `class_weight='balanced'` |
| XGBoost (opcional) | No | `n_estimators=200`, `scale_pos_weight=auto` |

#### Criterio de Selección

```
Puntuación = Recall × 0.60 + F1 × 0.40
```

Se prioriza el Recall para minimizar falsos negativos (clientes que se van sin ser detectados).

#### Protocolo de Intervención por Nivel de Riesgo

| Churn_Prob | Churn_Risk | Acción |
|------------|-----------|--------|
| > 66% | Alto | Contacto proactivo del retention team + oferta exclusiva |
| 33% – 66% | Medio | Campaña de re-engagement segmentada + descuento |
| < 33% | Bajo | Monitoreo y programa de fidelización continuo |

#### Columnas añadidas al dataset enriquecido

| Columna | Descripción |
|---------|-------------|
| `Churn_Prob` | Probabilidad de fuga [0.0 – 1.0] |
| `Churn_Risk` | Nivel de riesgo: "Bajo", "Medio", "Alto" |

#### Criterios de Éxito

| Métrica | Umbral |
|---------|--------|
| Recall (clase Churn) | > 75% |
| F1-Score | > 0.70 |

---

### Modelo 3 — Predicción de LTV (`model_ltv.py`)

**Clase:** `LTVPrediction(BaseMLModel)`  
**Tipo:** Supervisado — Regresión  
**Target:** `Lifetime_Value` (valor numérico continuo, en dólares)

#### Features Utilizadas (9)

```
Total_Purchases, Membership_Years, Discount_Usage_Rate,
Average_Order_Value, Login_Frequency, Session_Duration_Avg,
Pages_Per_Session, Returns_Rate, Email_Open_Rate
```

#### Modelos Evaluados

| Algoritmo | Escalado necesario | Parámetros clave |
|-----------|-------------------|-----------------|
| Linear Regression | Sí (StandardScaler) | Baseline |
| Random Forest | No | `n_estimators=200`, `max_depth=12`, `min_samples_leaf=5` |
| Gradient Boosting | No | `n_estimators=200`, `learning_rate=0.05`, `subsample=0.8` |

El modelo con mayor **R²** en test se selecciona automáticamente.

#### Estratificación del LTV Predicho

Los percentiles 33 y 66 del LTV predicho definen los tres tiers:

| LTV_Tier | Rango | Estrategia |
|----------|-------|-----------|
| Alto | > p66 | Retención VIP, prevención proactiva de churn |
| Medio | p33 – p66 | Upselling, cross-selling, aumento de ticket |
| Bajo | < p33 | Desarrollo, campañas de frecuencia |

#### Columnas añadidas al dataset enriquecido

| Columna | Descripción |
|---------|-------------|
| `Predicted_LTV` | Valor de vida predicho en dólares |
| `LTV_Tier` | Categoría: "Bajo", "Medio", "Alto" |

#### Criterios de Éxito

| Métrica | Umbral |
|---------|--------|
| R² | ≥ 0.70 |
| MAE | Minimizar |

---

## 6. Pipeline de Predicción (`app.py`)

**Clase:** `EcommercePipeline`

Carga los tres modelos `.pkl` exportados y expone predicciones unificadas para un cliente dado.

```python
from app import EcommercePipeline

pipe = EcommercePipeline()
result = pipe.predict_all(customer_data)
# result contiene: RFM_Segment, Churn_Prob, Churn_Risk, Predicted_LTV, LTV_Tier
```

Los datos de entrada se limpian automáticamente con `DataCleaner` antes de predecir.

---

## 7. Dashboard Streamlit (`streamlit_app.py`)

Interfaz web interactiva que permite:
- Ingresar datos de un cliente mediante sliders y selectores
- Obtener predicciones de RFM, Churn y LTV en tiempo real
- Visualizar distribuciones del dataset real con gráficos Plotly

**Para ejecutarlo ver la sección de ejecución del README.**

---

## 8. Stack Tecnológico

| Categoría | Librería | Versión | Rol |
|-----------|----------|---------|-----|
| Dashboard | `streamlit` | 1.56.0 | UI interactiva |
| ML | `scikit-learn` | 1.7.2 | K-Means, clasificadores, regresores |
| Datos | `pandas` | 3.0.1 | Manipulación de DataFrames |
| Datos | `numpy` | 2.4.3 | Operaciones numéricas |
| Visualización | `plotly` | 6.6.0 | Gráficos interactivos |
| Visualización | `matplotlib` | 3.10.8 | Gráficos estáticos |
| Visualización | `seaborn` | 0.13.2 | Heatmaps y distribuciones |

---

## 9. Instalación y Configuración

### Opción A — Conda (recomendado, Python 3.11)

```bash
# Crear entorno dentro del repo
conda env create -f environment.yml --prefix ./ecenv

# Activar
conda activate ./ecenv
```

### Opción B — pip + venv

```bash
python3.11 -m venv ./ecenv
source ./ecenv/bin/activate
pip install -r requirements.txt
```

---

## 10. Criterios de Éxito por Modelo

| Modelo | Métrica | Umbral |
|--------|---------|--------|
| RFM Segmentation | Silhouette Score | > 0.45 |
| RFM Segmentation | Cobertura de segmentos | Todos > 5% |
| Churn Prevention | Recall (clase=1) | > 75% |
| Churn Prevention | F1-Score | > 0.70 |
| LTV Prediction | R² | ≥ 0.70 |
| LTV Prediction | MAE | Minimizar |

| Sección | Contenido |
|---------|-----------|
| Visión General | Los 3 modelos de negocio y sus preguntas |
| Arquitectura | Diagrama ASCII del flujo completo de datos |
| Estructura de archivos | Árbol completo del proyecto |
| Dataset | Tabla de columnas, usos por modelo, CSV de salida enriquecido |
| Modelo 1 RFM | Ingeniería de variables, selección de K, nomenclatura de segmentos, criterios de éxito |
| Modelo 2 Churn | Features, manejo del desbalance, clasificadores, protocolo de intervención |
| Modelo 3 LTV | Features, regresores, estratificación por percentiles, criterios de éxito |
| RAG Engine | Componentes, 3 métodos documentados con firma y ejemplos |
| Pipeline | Secuencia interna, `run_full_pipeline()` con ejemplo de retorno |
| Schemas Pydantic | `CustomerInsightSchema` y `UIDesignSchema` con tipos |
| BaseMLModel | Contrato abstracto con docstrings explicativos |
| API pública | Tabla de exports del `__init__.py` |
| main.py | Constantes configurables, cómo ejecutar |
| Stack tecnológico | Tabla completa de dependencias con roles |
| Instalación | Conda y pip, configuración de `.env` |
| Ejecución | Pipeline completo, uso programático, uso desacoplado |
| Extensibilidad | Guía paso a paso para añadir un Modelo 4 |
| Variables de entorno | Tabla con descripción y si son requeridas |
| Criterios de éxito | Tabla consolidada de métricas y umbrales |