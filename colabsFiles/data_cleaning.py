import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


class DataCleaner:
    """
    Limpia y prepara el dataset de e-commerce para modelos de ML.

    Parámetros configurables
    ------------------------
    input_path       : ruta al CSV original
    output_path      : ruta donde se guarda el CSV limpio (None = no guardar)
    max_age          : edad máxima válida (default 100)
    min_purchases    : compras mínimas requeridas (default 1)
    max_cart_rate    : tasa de abandono de carrito máxima (default 100)
    iqr_multiplier   : multiplicador IQR para AOV outliers (default 1.5)
    impute_strategy  : estrategia de imputación numérica (default 'mean')

    Uso típico
    ----------
    cleaner = DataCleaner("data-base/raw.csv", "data-base/clean.csv")
    df_clean = cleaner.run()
    """

    def __init__(
        self,
        input_path: str,
        output_path: str | None = None,
        max_age: int = 100,
        min_purchases: int = 1,
        max_cart_rate: float = 100.0,
        iqr_multiplier: float = 1.5,
        impute_strategy: str = "mean",
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.max_age = max_age
        self.min_purchases = min_purchases
        self.max_cart_rate = max_cart_rate
        self.iqr_multiplier = iqr_multiplier
        self.impute_strategy = impute_strategy

        self.df: pd.DataFrame | None = None

    # ── Pasos individuales ────────────────────────────────────────────

    def load(self) -> "DataCleaner":
        """Carga el CSV original."""
        self.df = pd.read_csv(self.input_path)
        print(f"[LOAD] {len(self.df):,} registros cargados desde '{self.input_path}'")
        return self

    def remove_low_quality(self) -> "DataCleaner":
        """Elimina registros con edad fuera de rango y duplicados."""
        df = self.df.copy()

        before = len(df)
        df = df[df["Age"] <= self.max_age]
        print(f"[QUALITY] Edad > {self.max_age} eliminados → quedan {len(df):,} (removed {before - len(df):,})")

        before = len(df)
        df.drop_duplicates(inplace=True)
        print(f"[QUALITY] Duplicados eliminados → quedan {len(df):,} (removed {before - len(df):,})")

        self.df = df
        return self

    def remove_outliers(self, plot_aov: bool = False) -> "DataCleaner":
        """Elimina outliers de compras, abandono de carrito y AOV (IQR)."""
        df = self.df

        # Clientes sin compras
        before = len(df)
        df = df[df["Total_Purchases"] >= self.min_purchases]
        print(f"[OUTLIERS] Sin compras eliminados → quedan {len(df):,} (removed {before - len(df):,})")

        # Tasa de abandono fuera de rango
        before = len(df)
        df = df[df["Cart_Abandonment_Rate"] <= self.max_cart_rate]
        print(f"[OUTLIERS] Cart abandonment > {self.max_cart_rate} eliminados → quedan {len(df):,} (removed {before - len(df):,})")

        # AOV por IQR
        before = len(df)
        df = self._filter_iqr(df, "Average_Order_Value")
        print(f"[OUTLIERS] AOV outliers (IQR ×{self.iqr_multiplier}) eliminados → quedan {len(df):,} (removed {before - len(df):,})")

        if plot_aov:
            sns.boxplot(y=df["Average_Order_Value"], color="skyblue")
            plt.title("Distribución de Average Order Value")
            plt.ylabel("Valor Promedio de Orden ($)")
            plt.show()

        self.df = df 
        return self

    def impute_nulls(self) -> "DataCleaner":
        """Imputa nulos en columnas numéricas con la estrategia configurada."""
        df = self.df
        imputer = SimpleImputer(strategy=self.impute_strategy)
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        df = df.reset_index(drop=True)

        null_pct = round((df.isnull().sum() / len(df)) * 100, 2)
        remaining = null_pct[null_pct > 0]
        if remaining.empty:
            print("[IMPUTE] Sin nulos restantes ✓")
        else:
            print(f"[IMPUTE] Nulos restantes:\n{remaining}")

        self.df = df
        return self

    def save(self) -> "DataCleaner":
        """Exporta el dataset limpio al output_path si está definido."""
        if self.output_path and self.df is not None:
            self.df.to_csv(self.output_path, index=False)
            print(f"[SAVE] Dataset limpio guardado en '{self.output_path}' ({len(self.df):,} registros)")
        return self

    # ── Pipeline completo ─────────────────────────────────────────────

    def run(self, plot_aov: bool = False) -> pd.DataFrame:
        """Ejecuta el pipeline completo y retorna el DataFrame limpio."""
        return (
            self.load()
                .remove_low_quality()
                .remove_outliers(plot_aov=plot_aov)
                .impute_nulls()
                .save()
                .df
        )

    # ── Helpers privados ──────────────────────────────────────────────

    def _filter_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr
        return df[(df[column] >= lower) & (df[column] <= upper)]


if __name__ == "__main__":
    cleaner = DataCleaner(
        input_path="../../data-base/ecommerce_customer_churn_dataset.csv",
        output_path="../../data-base/clean_ecommerce_customer_churn_dataset.csv",
    )
    df_clean = cleaner.run()
    print(df_clean.info())

