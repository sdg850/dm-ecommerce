# E-Commerce Analytics Dashboard

Sistema de Machine Learning para segmentación de clientes, predicción de churn y estimación de Lifetime Value (LTV), con dashboard interactivo en Streamlit.

---

## Requisitos

- Python 3.11
- Conda o pip + venv

### Verificar Python

```bash
python3.11 --version
```

### Instalar Conda (si no lo tienes)

```bash
# macOS — via Homebrew
brew install --cask miniconda

# O descarga directa
# https://docs.conda.io/en/latest/miniconda.html
```

```powershell
# Windows — descarga el instalador desde:
# https://docs.conda.io/en/latest/miniconda.html
# Luego verifica en PowerShell:
conda --version
```

### Instalar pip (si usas venv)

```bash
# macOS / Linux
python3.11 -m ensurepip --upgrade
```

```powershell
# Windows (PowerShell)
python -m ensurepip --upgrade
```

---

## Instalación

### Opción A — Conda (recomendado)

```bash
# macOS / Linux
conda env create -f environment.yml --prefix ./ecenv
conda activate ./ecenv
```

```powershell
# Windows (PowerShell)
conda env create -f environment.yml --prefix .\ecenv
conda activate .\ecenv
```

### Opción B — pip + venv

```bash
# macOS / Linux
python3.11 -m venv ./ecenv
source ./ecenv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
python -m venv .\ecenv
.\ecenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Generar los modelos (primera vez)

Los modelos `.pkl` deben entrenarse antes de ejecutar el dashboard. Ejecutar los notebooks en este orden:

```
1. colabsFiles/DataCleaningProcess.ipynb   ← limpieza del dataset
2. colabsFiles/model_rfm.ipynb             ← genera models/rfm_model.pkl
3. colabsFiles/model_churn.ipynb           ← genera models/churn_model.pkl
4. colabsFiles/model_ltv.ipynb             ← genera models/ltv_model.pkl
```

> Si los archivos `.pkl` ya existen en `models/`, puedes saltar este paso.

---

## Ejecutar el dashboard

```bash
streamlit run streamlit_app.py
```

Abre el navegador en `http://localhost:8501`.

---

## Usar el pipeline por código

```python
from app import EcommercePipeline

pipe = EcommercePipeline()

customer = {
    "Age": 35,
    "Membership_Years": 3,
    "Login_Frequency": 5,
    "Session_Duration_Avg": 10.0,
    "Pages_Per_Session": 5.0,
    "Cart_Abandonment_Rate": 40.0,
    "Total_Purchases": 20,
    "Average_Order_Value": 120.0,
    "Days_Since_Last_Purchase": 30,
    "Discount_Usage_Rate": 30.0,
    "Returns_Rate": 2,
    "Email_Open_Rate": 25.0,
    "Customer_Service_Calls": 2,
    "Social_Media_Engagement_Score": 4.0,
    "Lifetime_Value": 800.0,
}

result = pipe.predict_all(customer)
print(result)
# {
#   "RFM_Segment": "Clientes Leales",
#   "Churn_Prob": 0.23,
#   "Churn_Risk": "Bajo",
#   "Predicted_LTV": 950.4,
#   "LTV_Tier": "Alto"
# }
```

---

## Estructura del proyecto

```
ux-project/
├── README.md                  ← Este archivo
├── DOCS.md                    ← Documentación técnica detallada
├── requirements.txt           ← Dependencias pip
├── environment.yml            ← Entorno Conda (Python 3.11)
├── app.py                     ← EcommercePipeline (carga pkl y predice)
├── streamlit_app.py           ← Dashboard Streamlit
├── colabsFiles/
│   ├── data_cleaning.py       ← DataCleaner reutilizable
│   ├── DataCleaningProcess.ipynb
│   ├── model_rfm.ipynb
│   ├── model_churn.ipynb
│   └── model_ltv.ipynb
├── models/
│   ├── rfm_model.pkl
│   ├── churn_model.pkl
│   └── ltv_model.pkl
└── data-base/
    ├── ecommerce_customer_churn_dataset.csv
    ├── clean_ecommerce_customer_churn_dataset.csv
    └── ecommerce_enriched_predictions.csv
```

---

## Documentación técnica

Ver [DOCS.md](DOCS.md) para detalles sobre modelos ML, features, métricas y arquitectura del sistema.
