"""
app.py — Pipeline de predicción para e-commerce analytics.

Carga los modelos exportados (pickle) y expone una clase EcommercePipeline
que limpia los datos de entrada con DataCleaner y genera predicciones de:
  - Segmentación RFM  (K-Means)
  - Riesgo de Churn   (clasificación)
  - Lifetime Value    (regresión)

Uso rápido:
    from app import EcommercePipeline
    pipe = EcommercePipeline()
    result = pipe.predict_all(customer_data)

Generación de modelos:
    Ejecuta primero los notebooks de colabsFiles/ en orden:
        1. colabsFiles/model_rfm.ipynb
        2. colabsFiles/model_churn.ipynb
        3. colabsFiles/model_ltv.ipynb
    Esto guardará los archivos .pkl en la carpeta models/.
"""

import os
import pickle
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Import DataCleaner directly (bypasses __init__.py which pulls langchain)
_ROOT = Path(__file__).parent
_dc_spec = importlib.util.spec_from_file_location(
    "data_cleaning",
    _ROOT / "colabsFiles" / "data_cleaning.py",
)
_dc_mod = importlib.util.module_from_spec(_dc_spec)
_dc_spec.loader.exec_module(_dc_mod)
DataCleaner = _dc_mod.DataCleaner

# ── Configuración ─────────────────────────────────────────────────────────────

MODELS_DIR   = Path(__file__).parent / "models"
RAW_DATA_PATH = Path(__file__).parent / "data-base" / "ecommerce_customer_churn_dataset.csv"


# ── Pipeline principal ────────────────────────────────────────────────────────

class EcommercePipeline:
    """
    Pipeline de predicción que integra los tres modelos exportados.

    Parámetros
    ----------
    models_dir : directorio donde están los archivos .pkl
                 (por defecto: models/ en la raíz del proyecto)

    Métodos públicos
    ----------------
    predict_all(data)    → DataFrame con RFM + Churn + LTV
    segment_rfm(data)    → DataFrame con RFM_Cluster / RFM_Segment
    predict_churn(data)  → DataFrame con Churn_Prob / Churn_Risk
    predict_ltv(data)    → DataFrame con Predicted_LTV / LTV_Tier
    load_and_clean_csv() → DataFrame limpio del CSV original
    """

    def __init__(self, models_dir: str | Path = MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self._rfm_bundle:   dict = {}
        self._churn_bundle: dict = {}
        self._ltv_bundle:   dict = {}
        self._imputer = SimpleImputer(strategy="mean")
        self._load_models()

    # ── Carga de modelos ──────────────────────────────────────────────

    def _load_models(self) -> None:
        """Carga los tres bundles de pickle desde models/."""
        required = {
            "rfm":   "rfm_model.pkl",
            "churn": "churn_model.pkl",
            "ltv":   "ltv_model.pkl",
        }
        missing = [
            name for name, fname in required.items()
            if not (self.models_dir / fname).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Modelos no encontrados: {missing}\n"
                f"Ejecuta primero los notebooks en colabsFiles/ para generar los .pkl en {self.models_dir}"
            )

        self._rfm_bundle   = self._load_pkl("rfm_model.pkl")
        self._churn_bundle = self._load_pkl("churn_model.pkl")
        self._ltv_bundle   = self._load_pkl("ltv_model.pkl")

        print("[APP] Modelos cargados:")
        print(f"  RFM   → K={self._rfm_bundle['optimal_k']}  Silhouette={self._rfm_bundle['silhouette']:.4f}")
        print(f"  Churn → {self._churn_bundle['model_name']}  Recall={self._churn_bundle['recall']:.4f}  F1={self._churn_bundle['f1']:.4f}")
        print(f"  LTV   → {self._ltv_bundle['model_name']}    R²={self._ltv_bundle['r2']:.4f}  MAE=${self._ltv_bundle['mae']:.2f}")

    def _load_pkl(self, filename: str) -> dict:
        path = self.models_dir / filename
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Limpieza de entrada ───────────────────────────────────────────

    def _prepare_input(self, data: dict | pd.DataFrame) -> pd.DataFrame:
        """
        Convierte dict/DataFrame de entrada a un DataFrame listo para predecir.
        Imputa nulos numéricos con la media (estrategia ligera para registros individuales).
        """
        df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if df[numeric_cols].isnull().any().any():
            df[numeric_cols] = self._imputer.fit_transform(df[numeric_cols])
        return df

    def load_and_clean_csv(
        self,
        input_path: str | Path = RAW_DATA_PATH,
        output_path: str | Path | None = None,
        **cleaner_kwargs,
    ) -> pd.DataFrame:
        """
        Carga y limpia el CSV completo usando DataCleaner.

        Útil para limpiar el dataset entero antes de hacer predicciones en batch.
        Los kwargs se pasan directamente a DataCleaner (max_age, min_purchases, etc.)
        """
        cleaner = DataCleaner(
            input_path=str(input_path),
            output_path=str(output_path) if output_path else None,
            **cleaner_kwargs,
        )
        return cleaner.run()

    # ── Predicciones individuales ─────────────────────────────────────

    def segment_rfm(self, data: dict | pd.DataFrame) -> pd.DataFrame:
        """
        Asigna segmento RFM a uno o varios clientes.

        Requiere columnas: Days_Since_Last_Purchase, Total_Purchases,
                           Login_Frequency, Average_Order_Value, Lifetime_Value
        """
        df = self._prepare_input(data)
        bundle = self._rfm_bundle

        max_recency = bundle["max_recency"]
        df["R_Score"] = max_recency - df["Days_Since_Last_Purchase"]
        df["F_Score"] = df["Total_Purchases"] * 0.65 + df["Login_Frequency"] * 0.35
        df["M_Score"] = df["Average_Order_Value"] * 0.40 + df["Lifetime_Value"] * 0.60

        rfm_cols = bundle["rfm_cols"]
        scaled = bundle["scaler"].transform(df[rfm_cols])
        df["RFM_Cluster"]  = bundle["model"].predict(scaled)
        df["RFM_Segment"]  = df["RFM_Cluster"].map(bundle["segment_names"])
        return df

    def predict_churn(self, data: dict | pd.DataFrame) -> pd.DataFrame:
        """
        Predice probabilidad y nivel de riesgo de churn.

        Requiere columnas: Login_Frequency, Cart_Abandonment_Rate, Returns_Rate,
                           Customer_Service_Calls, Days_Since_Last_Purchase,
                           Session_Duration_Avg, Pages_Per_Session,
                           Discount_Usage_Rate, Email_Open_Rate,
                           Social_Media_Engagement_Score, Membership_Years,
                           Average_Order_Value
        """
        df = self._prepare_input(data)
        bundle   = self._churn_bundle
        features = bundle["features"]
        x = df[features]

        if bundle["needs_scale"]:
            x = bundle["scaler"].transform(x)

        df["Churn_Prob"] = bundle["model"].predict_proba(x)[:, 1]
        df["Churn_Risk"] = pd.cut(
            df["Churn_Prob"],
            bins=[-0.001, 0.33, 0.66, 1.001],
            labels=["Bajo", "Medio", "Alto"],
        )
        return df

    def predict_ltv(self, data: dict | pd.DataFrame) -> pd.DataFrame:
        """
        Predice el Lifetime Value y asigna tier (Bajo / Medio / Alto).

        Requiere columnas: Total_Purchases, Membership_Years, Discount_Usage_Rate,
                           Average_Order_Value, Login_Frequency, Session_Duration_Avg,
                           Pages_Per_Session, Returns_Rate, Email_Open_Rate
        """
        df = self._prepare_input(data)
        bundle   = self._ltv_bundle
        features = bundle["features"]
        x = df[features]

        if bundle["needs_scale"]:
            x = bundle["scaler"].transform(x)

        df["Predicted_LTV"] = bundle["model"].predict(x)
        df["LTV_Tier"] = pd.cut(
            df["Predicted_LTV"],
            bins=[-np.inf, bundle["p33"], bundle["p66"], np.inf],
            labels=["Bajo", "Medio", "Alto"],
        )
        return df

    def predict_all(self, data: dict | pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta los tres modelos en secuencia y retorna un DataFrame enriquecido
        con columnas: RFM_Cluster, RFM_Segment, Churn_Prob, Churn_Risk,
                      Predicted_LTV, LTV_Tier.
        """
        df = self._prepare_input(data)
        df = self.segment_rfm(df)
        df = self.predict_churn(df)
        df = self.predict_ltv(df)
        return df

    # ── Utilidades ────────────────────────────────────────────────────

    def model_info(self) -> dict:
        """Retorna un resumen de los modelos cargados."""
        return {
            "rfm": {
                "optimal_k":  self._rfm_bundle["optimal_k"],
                "silhouette": self._rfm_bundle["silhouette"],
                "segments":   list(self._rfm_bundle["segment_names"].values()),
            },
            "churn": {
                "model_name": self._churn_bundle["model_name"],
                "recall":     self._churn_bundle["recall"],
                "f1":         self._churn_bundle["f1"],
                "roc_auc":    self._churn_bundle["roc_auc"],
            },
            "ltv": {
                "model_name": self._ltv_bundle["model_name"],
                "r2":         self._ltv_bundle["r2"],
                "mae":        self._ltv_bundle["mae"],
                "p33":        self._ltv_bundle["p33"],
                "p66":        self._ltv_bundle["p66"],
            },
        }


# ── Demo / Ejecución directa ──────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Perfil de cliente de ejemplo
    SAMPLE_CUSTOMER = {
        "Age": 35,
        "Login_Frequency": 3,
        "Cart_Abandonment_Rate": 78.5,
        "Returns_Rate": 4,
        "Customer_Service_Calls": 7,
        "Days_Since_Last_Purchase": 95,
        "Session_Duration_Avg": 8.2,
        "Pages_Per_Session": 3.1,
        "Discount_Usage_Rate": 82.0,
        "Email_Open_Rate": 5.0,
        "Social_Media_Engagement_Score": 2.0,
        "Membership_Years": 1.2,
        "Average_Order_Value": 42.5,
        "Total_Purchases": 4,
        "Lifetime_Value": 285.0,
    }

    print("=" * 60)
    print("  ECOMMERCE ANALYTICS PIPELINE — Demo")
    print("=" * 60)

    pipe = EcommercePipeline()

    print("\n── Información de modelos cargados ──")
    print(json.dumps(pipe.model_info(), indent=2))

    print("\n── Predicción para cliente de ejemplo ──")
    result = pipe.predict_all(SAMPLE_CUSTOMER)

    cols = ["RFM_Segment", "Churn_Prob", "Churn_Risk", "Predicted_LTV", "LTV_Tier"]
    available = [c for c in cols if c in result.columns]
    print(result[available].to_string(index=False))

    print("\n── Limpieza del dataset completo (batch) ──")
    df_clean = pipe.load_and_clean_csv()
    print(f"Dataset limpio: {df_clean.shape[0]:,} registros")

    print("\n── Predicciones en batch (primeros 5) ──")
    batch_result = pipe.predict_all(df_clean.head(5))
    print(batch_result[available].to_string(index=False))
