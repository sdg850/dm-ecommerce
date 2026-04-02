"""
streamlit_app.py — Dashboard de Predicciones de E-Commerce Analytics.

Ejecutar:
    streamlit run streamlit_app.py

Requiere que los modelos estén generados en models/
(ver colabsFiles/model_rfm.ipynb, model_churn.ipynb, model_ltv.ipynb)
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Bootstrap: importar EcommercePipeline desde app.py ───────────────────────
_ROOT = Path(__file__).parent
_app_spec = importlib.util.spec_from_file_location("app", _ROOT / "app.py")
_app_mod  = importlib.util.module_from_spec(_app_spec)
_app_spec.loader.exec_module(_app_mod)
EcommercePipeline = _app_mod.EcommercePipeline
RAW_DATA_PATH     = _app_mod.RAW_DATA_PATH

# ── Constantes de UI ──────────────────────────────────────────────────────────
SEGMENT_COLORS = {
    "Champions":       "#6C63FF",
    "Clientes Leales": "#48CAE4",
    "Potenciales":     "#52B788",
    "En Riesgo":       "#F77F00",
    "Perdidos/Inactivos": "#E63946",
}
RISK_COLORS = {"Bajo": "#52B788", "Medio": "#F77F00", "Alto": "#E63946"}
TIER_COLORS = {"Bajo": "#E63946", "Medio": "#F77F00", "Alto": "#52B788"}

ALL_INPUT_FIELDS = {
    "Age":                          ("Edad",                          18,  100, 35,  1),
    "Membership_Years":             ("Años de membresía",             0,   30,  3,   1),
    "Login_Frequency":              ("Frecuencia de login",           0,   30,  5,   1),
    "Session_Duration_Avg":         ("Duración sesión promedio (min)",0.0, 60.0,10.0, 0.5),
    "Pages_Per_Session":            ("Páginas por sesión",            1.0, 30.0, 5.0, 0.5),
    "Cart_Abandonment_Rate":        ("Tasa abandon. carrito (%)",     0.0, 100.0,40.0,1.0),
    "Total_Purchases":              ("Compras totales",               1,   200,  20,  1),
    "Average_Order_Value":          ("Valor promedio de orden ($)",   0.0, 2000.0,120.0,5.0),
    "Days_Since_Last_Purchase":     ("Días desde última compra",      0,   365, 30,  1),
    "Discount_Usage_Rate":          ("Tasa uso descuentos (%)",       0.0, 100.0,30.0,1.0),
    "Returns_Rate":                 ("Tasa de devoluciones",          0,   20,  2,   1),
    "Email_Open_Rate":              ("Tasa apertura email (%)",       0.0, 100.0,25.0,1.0),
    "Customer_Service_Calls":       ("Llamadas a servicio al cliente",0,   20,  2,   1),
    "Social_Media_Engagement_Score":("Score engagement social",       0.0, 10.0, 4.0, 0.1),
    "Lifetime_Value":               ("Lifetime Value actual ($)",     0.0, 5000.0,800.0,10.0),
}


# ── Carga del pipeline (cacheado) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelos...")
def load_pipeline() -> EcommercePipeline:
    return EcommercePipeline()


@st.cache_data(show_spinner="Cargando y limpiando dataset...")
def load_dataset(_pipe: EcommercePipeline) -> pd.DataFrame:
    return _pipe.load_and_clean_csv()


# ── Helpers de visualización ────────────────────────────────────────────────
def badge(label: str, color: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="
            background:{color}22;border:1px solid {color};border-radius:8px;
            padding:12px 18px;text-align:center;margin-bottom:4px">
            <div style="font-size:0.75rem;color:{color};font-weight:600;
                        text-transform:uppercase;letter-spacing:1px">{label}</div>
            <div style="font-size:1.4rem;font-weight:700;color:#FAFAFA;margin-top:4px">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def risk_gauge(prob: float) -> go.Figure:
    color = "#52B788" if prob < 0.33 else ("#F77F00" if prob < 0.66 else "#E63946")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "color": "#FAFAFA"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555"},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#1A1F2E",
            "steps": [
                {"range": [0,  33], "color": "rgba(82,183,136,0.13)"},
                {"range": [33, 66], "color": "rgba(247,127,0,0.13)"},
                {"range": [66,100], "color": "rgba(230,57,70,0.13)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": prob * 100},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=200, margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor="#0E1117", font_color="#FAFAFA",
    )
    return fig


# ── Páginas ───────────────────────────────────────────────────────────────────

def page_single(pipe: EcommercePipeline) -> None:
    st.header("🔍 Predicción Individual de Cliente")
    st.caption("Los resultados se actualizan automáticamente al cambiar cualquier valor.")

    st.subheader("Datos del cliente")
    inputs: dict = {}
    cols = st.columns(3)
    for i, (key, (label, mn, mx, default, step)) in enumerate(ALL_INPUT_FIELDS.items()):
        with cols[i % 3]:
            sk = f"single_{key}"
            if sk not in st.session_state:
                st.session_state[sk] = float(default) if isinstance(step, float) else int(default)
            if isinstance(step, float):
                inputs[key] = st.number_input(label, min_value=float(mn), max_value=float(mx),
                                               value=float(st.session_state[sk]), step=step, key=sk)
            else:
                inputs[key] = st.number_input(label, min_value=int(mn), max_value=int(mx),
                                               value=int(st.session_state[sk]), step=step, key=sk)

    result = pipe.predict_all(inputs)
    if True:

        rfm_seg    = result["RFM_Segment"].iloc[0]
        churn_prob = float(result["Churn_Prob"].iloc[0])
        churn_risk = str(result["Churn_Risk"].iloc[0])
        ltv_pred   = float(result["Predicted_LTV"].iloc[0])
        ltv_tier   = str(result["LTV_Tier"].iloc[0])

        st.divider()
        st.subheader("📊 Resultados")

        c1, c2, c3 = st.columns(3)
        with c1:
            seg_color = SEGMENT_COLORS.get(rfm_seg, "#6C63FF")
            badge("Segmento RFM", seg_color, rfm_seg)
        with c2:
            risk_color = RISK_COLORS.get(churn_risk, "#F77F00")
            badge("Riesgo de Churn", risk_color, churn_risk)
        with c3:
            tier_color = TIER_COLORS.get(ltv_tier, "#6C63FF")
            badge("LTV Tier", tier_color, ltv_tier)

        st.divider()
        col_gauge, col_ltv = st.columns([1, 1])
        with col_gauge:
            st.markdown("**Probabilidad de Churn**")
            st.plotly_chart(risk_gauge(churn_prob), use_container_width=True)
        with col_ltv:
            st.markdown("**Lifetime Value Predicho**")
            st.metric(
                label="",
                value=f"${ltv_pred:,.0f}",
                delta=f"Tier {ltv_tier}",
                delta_color="normal" if ltv_tier == "Alto" else ("off" if ltv_tier == "Medio" else "inverse"),
            )
            st.progress(min(float(ltv_pred) / pipe._ltv_bundle["p66"] / 2, 1.0))

        # Estrategia recomendada
        strategies = {
            ("Champions",       "Bajo"):  "🏆 Cliente VIP — Acceso anticipado, programa de embajadores, experiencia premium.",
            ("Champions",       "Medio"): "🏆 Cliente Champion con señales de riesgo — Revisión proactiva + beneficio exclusivo.",
            ("Champions",       "Alto"):  "🚨 Champion en riesgo crítico — Intervención inmediata del retention team.",
            ("Clientes Leales", "Bajo"):  "💙 Fidelización — Newsletter personalizado, descuentos por volumen.",
            ("Clientes Leales", "Medio"): "⚠️ Leal en riesgo — Campaña de re-engagement por canal preferido.",
            ("Clientes Leales", "Alto"):  "🚨 Leal en riesgo alto — Oferta win-back urgente.",
            ("Potenciales",     "Bajo"):  "📈 Nurturing activo — Recomendaciones personalizadas, nuevas categorías.",
            ("Potenciales",     "Medio"): "📈 Potencial en riesgo — Descuento de activación + contenido de valor.",
            ("Potenciales",     "Alto"):  "🚨 Potencial perdiendo interés — Campaña de reactivación inmediata.",
            ("En Riesgo",       "Bajo"):  "⚠️ En riesgo — Checkout simplificado, garantías prominentes.",
            ("En Riesgo",       "Medio"): "🚨 En riesgo medio — Oferta personalizada + contacto directo.",
            ("En Riesgo",       "Alto"):  "🚨 En riesgo crítico — Protocolo de retención de emergencia.",
        }
        strategy = strategies.get(
            (rfm_seg, churn_risk),
            "📋 Monitorear señales de deterioro y mantener comunicación activa.",
        )
        st.info(f"**Estrategia recomendada:** {strategy}")


def page_batch(pipe: EcommercePipeline) -> None:
    st.header("📂 Predicción en Batch")
    st.caption("Carga un CSV o usa el dataset limpio del proyecto. Las predicciones se actualizan al cambiar la muestra.")

    source = st.radio(
        "Fuente de datos",
        ["Dataset del proyecto (data-base/)", "Subir CSV propio"],
        horizontal=True,
    )

    df_input: pd.DataFrame | None = None

    if source == "Dataset del proyecto (data-base/)":
        if st.button("Cargar y limpiar dataset", type="primary"):
            with st.spinner("Limpiando datos con DataCleaner..."):
                st.session_state["batch_df"] = load_dataset(pipe)
            st.session_state.pop("batch_result", None)
        if "batch_df" in st.session_state:
            df_input = st.session_state["batch_df"]
            st.success(f"Dataset cargado: {len(df_input):,} registros tras limpieza")
    else:
        uploaded = st.file_uploader("Sube un CSV", type="csv")
        if uploaded:
            file_key = uploaded.name + str(uploaded.size)
            if st.session_state.get("uploaded_file_key") != file_key:
                st.session_state["batch_df"] = pd.read_csv(uploaded)
                st.session_state["uploaded_file_key"] = file_key
                st.session_state.pop("batch_result", None)
            df_input = st.session_state.get("batch_df")
            if df_input is not None:
                st.success(f"Archivo cargado: {len(df_input):,} registros")

    if df_input is not None:
        sample_n = st.slider(
            "Registros a predecir (muestra)", 100,
            min(len(df_input), 10_000),
            min(1000, len(df_input)), 100,
            key="batch_sample_n",
        )
        df_sample = df_input.sample(n=sample_n, random_state=42) if sample_n < len(df_input) else df_input

        # Auto-run whenever sample size changes or data first loaded
        prev_n   = st.session_state.get("batch_prev_n")
        prev_src = st.session_state.get("batch_prev_src")
        needs_run = prev_n != sample_n or prev_src != source or "batch_result" not in st.session_state

        if needs_run:
            with st.spinner(f"Prediciendo {sample_n:,} registros..."):
                st.session_state["batch_result"] = pipe.predict_all(df_sample)
            st.session_state["batch_prev_n"]   = sample_n
            st.session_state["batch_prev_src"] = source

    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        _render_batch_results(result, pipe)


def _render_batch_results(result: pd.DataFrame, pipe: EcommercePipeline) -> None:
    st.divider()
    st.subheader(f"📊 Resultados — {len(result):,} clientes")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total clientes",   f"{len(result):,}")
    k2.metric("Churn riesgo Alto", f"{(result['Churn_Risk']=='Alto').sum():,}",
              delta=f"{(result['Churn_Risk']=='Alto').mean()*100:.1f}%", delta_color="inverse")
    k3.metric("LTV promedio",     f"${result['Predicted_LTV'].mean():,.0f}")
    k4.metric("Churn prob media", f"{result['Churn_Prob'].mean()*100:.1f}%", delta_color="inverse")

    tab1, tab2, tab3, tab4 = st.tabs(["🗂 RFM", "⚠️ Churn", "💰 LTV", "📋 Datos"])

    # ── Tab RFM ──────────────────────────────────────────────────────
    with tab1:
        seg_counts = result["RFM_Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segmento", "Clientes"]
        seg_counts["Color"] = seg_counts["Segmento"].map(
            lambda s: SEGMENT_COLORS.get(s, "#6C63FF")
        )

        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig = px.pie(
                seg_counts, names="Segmento", values="Clientes",
                color="Segmento",
                color_discrete_map=SEGMENT_COLORS,
                title="Distribución de Segmentos RFM",
                hole=0.4,
            )
            fig.update_layout(paper_bgcolor="#0E1117", font_color="#FAFAFA",
                              legend=dict(orientation="v"))
            st.plotly_chart(fig, use_container_width=True)

        with col_bar:
            seg_ltv = result.groupby("RFM_Segment")["Predicted_LTV"].mean().reset_index()
            seg_ltv.columns = ["Segmento", "LTV_Promedio"]
            fig2 = px.bar(
                seg_ltv.sort_values("LTV_Promedio", ascending=True),
                x="LTV_Promedio", y="Segmento", orientation="h",
                color="Segmento", color_discrete_map=SEGMENT_COLORS,
                title="LTV Promedio por Segmento",
                labels={"LTV_Promedio": "LTV Promedio ($)"},
            )
            fig2.update_layout(paper_bgcolor="#0E1117", font_color="#FAFAFA",
                               showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Tabla de perfiles
        profile = result.groupby("RFM_Segment").agg(
            Clientes=("RFM_Segment", "count"),
            Churn_Prob_Media=("Churn_Prob", "mean"),
            LTV_Predicho_Media=("Predicted_LTV", "mean"),
        ).round(2).reset_index()
        profile["Churn_Prob_Media"] = profile["Churn_Prob_Media"].map(lambda x: f"{x*100:.1f}%")
        profile["LTV_Predicho_Media"] = profile["LTV_Predicho_Media"].map(lambda x: f"${x:,.0f}")
        st.dataframe(profile, use_container_width=True, hide_index=True)

    # ── Tab Churn ─────────────────────────────────────────────────────
    with tab2:
        col_churn1, col_churn2 = st.columns(2)
        with col_churn1:
            risk_counts = result["Churn_Risk"].value_counts().reset_index()
            risk_counts.columns = ["Riesgo", "Clientes"]
            fig3 = px.pie(
                risk_counts, names="Riesgo", values="Clientes",
                color="Riesgo", color_discrete_map=RISK_COLORS,
                title="Distribución de Riesgo de Churn",
                hole=0.4,
            )
            fig3.update_layout(paper_bgcolor="#0E1117", font_color="#FAFAFA")
            st.plotly_chart(fig3, use_container_width=True)

        with col_churn2:
            fig4 = px.histogram(
                result, x="Churn_Prob", nbins=40, color="Churn_Risk",
                color_discrete_map=RISK_COLORS,
                title="Distribución de Probabilidad de Churn",
                labels={"Churn_Prob": "Probabilidad de Churn"},
                barmode="overlay",
            )
            fig4.update_traces(opacity=0.75)
            fig4.update_layout(paper_bgcolor="#0E1117", font_color="#FAFAFA")
            st.plotly_chart(fig4, use_container_width=True)

        # Churn por segmento
        churn_seg = result.groupby("RFM_Segment")["Churn_Prob"].mean().mul(100).round(1).reset_index()
        churn_seg.columns = ["Segmento", "Churn_Prob (%)"]
        fig5 = px.bar(
            churn_seg.sort_values("Churn_Prob (%)", ascending=False),
            x="Segmento", y="Churn_Prob (%)",
            color="Segmento", color_discrete_map=SEGMENT_COLORS,
            title="Probabilidad Media de Churn por Segmento",
        )
        fig5.update_layout(paper_bgcolor="#0E1117", font_color="#FAFAFA", showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    # ── Tab LTV ───────────────────────────────────────────────────────
    with tab3:
        col_ltv1, col_ltv2 = st.columns(2)
        with col_ltv1:
            tier_counts = result["LTV_Tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Clientes"]
            fig6 = px.pie(
                tier_counts, names="Tier", values="Clientes",
                color="Tier", color_discrete_map=TIER_COLORS,
                title="Distribución de LTV Tiers",
                hole=0.4,
            )
            fig6.update_layout(paper_bgcolor="#0E1117", font_color="#FAFAFA")
            st.plotly_chart(fig6, use_container_width=True)

        with col_ltv2:
            fig7 = px.histogram(
                result, x="Predicted_LTV", nbins=50, color="LTV_Tier",
                color_discrete_map=TIER_COLORS,
                title="Distribución de LTV Predicho",
                labels={"Predicted_LTV": "LTV Predicho ($)"},
                barmode="overlay",
            )
            fig7.update_traces(opacity=0.75)
            fig7.update_layout(paper_bgcolor="#0E1117", font_color="#FAFAFA")
            st.plotly_chart(fig7, use_container_width=True)

        # Scatter LTV vs Churn
        fig8 = px.scatter(
            result.sample(min(2000, len(result)), random_state=1),
            x="Churn_Prob", y="Predicted_LTV",
            color="RFM_Segment", color_discrete_map=SEGMENT_COLORS,
            size_max=8, opacity=0.6,
            title="LTV Predicho vs Probabilidad de Churn",
            labels={"Churn_Prob": "Prob. Churn", "Predicted_LTV": "LTV Predicho ($)"},
        )
        fig8.update_layout(paper_bgcolor="#0E1117", font_color="#FAFAFA")
        st.plotly_chart(fig8, use_container_width=True)

    # ── Tab Datos ─────────────────────────────────────────────────────
    with tab4:
        PRED_COLS = ["RFM_Segment", "Churn_Prob", "Churn_Risk", "Predicted_LTV", "LTV_Tier"]
        show_cols = [c for c in PRED_COLS if c in result.columns]
        display = result[show_cols].copy()
        display["Churn_Prob"] = display["Churn_Prob"].map(lambda x: f"{x*100:.1f}%")
        display["Predicted_LTV"] = display["Predicted_LTV"].map(lambda x: f"${x:,.0f}")

        st.dataframe(display, use_container_width=True)

        csv = result[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Descargar predicciones (CSV)",
            data=csv,
            file_name="predicciones_ecommerce.csv",
            mime="text/csv",
        )


def page_model_info(pipe: EcommercePipeline) -> None:
    st.header("⚙️ Información de Modelos")

    info = pipe.model_info()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🗂 RFM — K-Means")
        rfm = info["rfm"]
        st.metric("K óptimo",       rfm["optimal_k"])
        st.metric("Silhouette",      f"{rfm['silhouette']:.4f}")
        st.markdown("**Segmentos:**")
        for seg in rfm["segments"]:
            color = SEGMENT_COLORS.get(seg, "#6C63FF")
            st.markdown(
                f'<span style="background:{color}33;border:1px solid {color};'
                f'border-radius:6px;padding:3px 10px;color:{color};'
                f'font-weight:600">{seg}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

    with col2:
        st.subheader("⚠️ Churn — Clasificación")
        churn = info["churn"]
        st.metric("Modelo",   churn["model_name"])
        st.metric("Recall",   f"{churn['recall']:.4f}",
                  delta="✓ > 0.75" if churn["recall"] > 0.75 else "⚠ < 0.75",
                  delta_color="normal" if churn["recall"] > 0.75 else "inverse")
        st.metric("F1-Score", f"{churn['f1']:.4f}",
                  delta="✓ > 0.70" if churn["f1"] > 0.70 else "⚠ < 0.70",
                  delta_color="normal" if churn["f1"] > 0.70 else "inverse")
        st.metric("ROC-AUC",  f"{churn['roc_auc']:.4f}")

    with col3:
        st.subheader("💰 LTV — Regresión")
        ltv = info["ltv"]
        st.metric("Modelo", ltv["model_name"])
        st.metric("R²",     f"{ltv['r2']:.4f}",
                  delta="✓ ≥ 0.70" if ltv["r2"] >= 0.70 else "⚠ < 0.70",
                  delta_color="normal" if ltv["r2"] >= 0.70 else "inverse")
        st.metric("MAE",    f"${ltv['mae']:.2f}")
        st.metric("Tier Bajo/Medio",  f"${ltv['p33']:,.0f}")
        st.metric("Tier Medio/Alto",  f"${ltv['p66']:,.0f}")

    # Feature importance (churn)
    st.divider()
    if pipe._churn_bundle.get("feature_importance"):
        st.subheader("🔑 Feature Importance — Churn")
        fi = pipe._churn_bundle["feature_importance"]
        fi_df = pd.DataFrame(fi.items(), columns=["Feature", "Importancia"]).sort_values(
            "Importancia", ascending=True
        )
        fig = px.bar(fi_df, x="Importancia", y="Feature", orientation="h",
                     color="Importancia", color_continuous_scale="purples")
        fig.update_layout(paper_bgcolor="#0E1117", font_color="#FAFAFA",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    if pipe._ltv_bundle.get("feature_importance"):
        st.subheader("🔑 Feature Importance — LTV")
        fi = pipe._ltv_bundle["feature_importance"]
        fi_df = pd.DataFrame(fi.items(), columns=["Feature", "Importancia"]).sort_values(
            "Importancia", ascending=True
        )
        fig2 = px.bar(fi_df, x="Importancia", y="Feature", orientation="h",
                      color="Importancia", color_continuous_scale="teal")
        fig2.update_layout(paper_bgcolor="#0E1117", font_color="#FAFAFA",
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="E-Commerce Analytics",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar
    with st.sidebar:
        st.markdown(
            "<h2 style='text-align:center;color:#6C63FF'>🛒 E-Commerce<br>Analytics</h2>",
            unsafe_allow_html=True,
        )
        st.divider()
        page = st.radio(
            "Navegación",
            ["🔍 Predicción Individual", "📂 Predicción Batch", "⚙️ Info de Modelos"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Modelos: RFM · Churn · LTV")
        st.caption("Powered by Scikit-learn + Streamlit")

    # Cargar pipeline
    try:
        pipe = load_pipeline()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Routing
    if page == "🔍 Predicción Individual":
        page_single(pipe)
    elif page == "📂 Predicción Batch":
        page_batch(pipe)
    else:
        page_model_info(pipe)


if __name__ == "__main__":
    main()
