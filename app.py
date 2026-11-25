import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import warnings
import requests
import io
import re
from itertools import combinations, product
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🎯 Diseño Interactivo de Experimentos",
    page_icon="📊",
    layout="wide"
)

st.title("🎯 Diseño Interactivo de Experimentos")
st.markdown("---")

# ============================================================
# INICIALIZAR SESSION STATE
# ============================================================

if 'datos_cargados' not in st.session_state:
    st.session_state.datos_cargados = {}
if 'resultados_anova' not in st.session_state:
    st.session_state.resultados_anova = {}
if 'diseno_actual' not in st.session_state:
    st.session_state.diseno_actual = None
if 'factores_definidos' not in st.session_state:
    st.session_state.factores_definidos = {}
if 'respuestas_definidas' not in st.session_state:
    st.session_state.respuestas_definidas = {}
if 'pruebas_multiples' not in st.session_state:
    st.session_state.pruebas_multiples = {}

# Variables exclusivas para la pestaña Fraccionado
if 'fr_df' not in st.session_state: st.session_state.fr_df = None
if 'fr_analizado' not in st.session_state: st.session_state.fr_analizado = False
if 'fr_colapsado' not in st.session_state: st.session_state.fr_colapsado = None

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def selector_tipo_anova(key_suffix="", show_help=True):
    if show_help:
        st.markdown("""
        **📊 Tipos de ANOVA:**
        - **Tipo I (Secuencial):** El orden de entrada importa.
        - **Tipo II (Parcial):** Recomendado para diseños balanceados.
        - **Tipo III (Parcial con interacción):** Útil para diseños no balanceados.
        """)
    tipo_anova = st.radio(
        "Selecciona el tipo de ANOVA:",
        ["Tipo I (Secuencial)", "Tipo II (Parcial)", "Tipo III (Parcial con interacción)"],
        horizontal=True, key=f"tipo_anova_{key_suffix}"
    )
    tipo_map = {"Tipo I (Secuencial)": 1, "Tipo II (Parcial)": 2, "Tipo III (Parcial con interacción)": 3}
    return tipo_map[tipo_anova]

def analizar_anova_mejorada(df, formula, titulo="ANOVA", key_suffix="", tipo="dca", anova_type=2):
    try:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
        modelo = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(modelo, typ=anova_type)
        if 'mean_sq' not in anova_table.columns:
            anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']
        rename_map = {'sum_sq': 'Sum Sq', 'df': 'DF', 'mean_sq': 'Mean Sq', 'F': 'F-Ratio', 'PR(>F)': 'P-Value'}
        anova_table = anova_table.rename(columns=rename_map)
        columnas_orden = [c for c in anova_table.columns if c not in ['Mean Sq']]
        if 'DF' in columnas_orden:
            idx_df = columnas_orden.index('DF')
            columnas_orden.insert(idx_df + 1, 'Mean Sq')
            anova_table = anova_table[columnas_orden]

        # Guardamos el resultado
        st.session_state.resultados_anova[key_suffix] = {
            'modelo': modelo, 'anova': anova_table, 'df': df, 'tipo': tipo,
            'formula': formula, 'anova_type': anova_type, 'titulo': titulo
        }

        st.success(f"✅ {titulo} calculado (Tipo {anova_type})")
        st.dataframe(anova_table.style.format({
            'Sum Sq': '{:.4f}', 'Mean Sq': '{:.4f}', 'F-Ratio': '{:.4f}', 'P-Value': '{:.6f}'
        }, na_rep="None"))
        st.write(f"**R² = {modelo.rsquared:.6f}**")
        st.write(f"**R² ajustado = {modelo.rsquared_adj:.6f}**")
        return modelo, anova_table
    except Exception as e:
        st.error(f"❌ Error en ANOVA: {str(e)}")
        return None, None

def mostrar_tabla_anova(key_suffix):
    """Imprime de forma persistente la tabla ANOVA guardada en session_state."""
    if key_suffix in st.session_state.resultados_anova:
        res = st.session_state.resultados_anova[key_suffix]
        st.success(f"✅ {res.get('titulo', 'ANOVA')} calculado (Tipo {res.get('anova_type', 2)})")
        st.dataframe(res['anova'].style.format({
            'Sum Sq': '{:.4f}', 'Mean Sq': '{:.4f}', 'F-Ratio': '{:.4f}', 'P-Value': '{:.6f}'
        }, na_rep="None"))
        st.write(f"**R² = {res['modelo'].rsquared:.6f}** | **R² ajustado = {res['modelo'].rsquared_adj:.6f}**")

def graficos_interaccion_multifactorial(df, factores, respuesta, max_pares=6):
    if len(factores) < 2:
        return None
    pares = list(combinations(factores, 2))[:max_pares]
    n_filas = (len(pares) + 1) // 2
    fig = make_subplots(rows=n_filas, cols=2, subplot_titles=[f"{f1} × {f2}" for f1, f2 in pares])
    for idx, (f1, f2) in enumerate(pares):
        row = idx // 2 + 1
        col = idx % 2 + 1
        medias = df.groupby([f1, f2])[respuesta].mean().reset_index()
        for nivel in medias[f2].unique():
            datos_nivel = medias[medias[f2] == nivel]
            fig.add_trace(go.Scatter(x=datos_nivel[f1], y=datos_nivel[respuesta],
                                     mode='lines+markers', name=f"{f2}={nivel}",
                                     legendgroup=f"{f2}={nivel}", showlegend=(idx == 0)),
                          row=row, col=col)
        fig.update_xaxes(title_text=f1, row=row, col=col)
        fig.update_yaxes(title_text=respuesta, row=row, col=col)
    fig.update_layout(height=min(400, 200 * n_filas), showlegend=True,
                      title_text="Gráficos de Interacción")
    return fig

def graficos_interaccion_heatmap(df, factores, respuesta):
    if len(factores) < 2:
        return None
    try:
        f1, f2 = factores[0], factores[1]
        pivot = df.pivot_table(index=f1, columns=f2, values=respuesta, aggfunc='mean')
        fig = px.imshow(pivot, text_auto=True, aspect="auto",
                        color_continuous_scale='RdBu_r', title=f"Mapa de Calor: {f1} × {f2}")
        fig.update_layout(height=400)
        return fig
    except:
        return None

def graficar_boxplot_por_factor(df, factores, respuesta):
    if not factores:
        return None
    n_factores = len(factores)
    fig = make_subplots(rows=1, cols=n_factores, subplot_titles=[f"{f}" for f in factores])
    for i, factor in enumerate(factores):
        fig.add_trace(
            go.Box(y=df[respuesta], x=df[factor], name=factor,
                   marker_color='#3498db', boxpoints='all', jitter=0.3, pointpos=-1.8),
            row=1, col=i+1
        )
        fig.update_xaxes(title_text=factor, row=1, col=i+1)
        fig.update_yaxes(title_text=respuesta, row=1, col=i+1)
    fig.update_layout(height=400, showlegend=False, title_text="Boxplot por Factor")
    return fig

def graficar_sumas_cuadrados(anova_table):
    try:
        df_sc = anova_table.reset_index().rename(columns={'index': 'Fuente'})
        col_sc = 'Sum Sq' if 'Sum Sq' in df_sc.columns else 'sum_sq' if 'sum_sq' in df_sc.columns else None
        if not col_sc:
            return None
        if 'Total' in df_sc['Fuente'].values:
            df_sc = df_sc[df_sc['Fuente'] != 'Total']
        fig = px.bar(df_sc, x='Fuente', y=col_sc, title="Gráfico de Sumas de Cuadrados",
                     color='Fuente', text_auto='.2f')
        if df_sc[col_sc].max() / max(df_sc[col_sc].min(), 0.001) > 100:
            fig.update_yaxes(type="log")
        fig.update_layout(showlegend=False, height=400)
        fig.update_traces(textposition='outside')
        return fig
    except:
        return None

def graficar_verificacion_modelo_completa(modelo, df, factor, respuesta):
    if modelo is None:
        return None
    residuos = modelo.resid
    predichos = modelo.fittedvalues
    run_order = np.arange(1, len(residuos) + 1)
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Q-Q (Normalidad)", "Residuos vs Ajustados",
        "Residuos vs Orden", "Histograma de Residuos"
    ])
    from scipy.stats import probplot
    (osm, osr), (slope, intercept, r) = probplot(residuos, dist="norm", fit=True)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', marker=dict(color='blue', size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[min(osm), max(osm)],
                             y=[min(osm)*slope + intercept, max(osm)*slope + intercept],
                             mode='lines', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=predichos, y=residuos, mode='markers', marker=dict(color='green', size=6)), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_trace(go.Scatter(x=run_order, y=residuos, mode='lines+markers', marker=dict(color='orange', size=6)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_trace(go.Histogram(x=residuos, marker_color='purple', nbinsx=15), row=2, col=2)
    fig.update_layout(height=700, showlegend=False, title_text="Verificación de Supuestos")
    return fig

def graficar_interaccion_mejorada(df, f1, f2, respuesta):
    medias = df.groupby([f1, f2])[respuesta].mean().reset_index()
    fig = px.line(medias, x=f1, y=respuesta, color=f2, markers=True, title=f"Interacción {f1} × {f2}")
    for i, row in medias.iterrows():
        fig.add_annotation(x=row[f1], y=row[respuesta], text=f"{row[respuesta]:.2f}",
                           showarrow=False, yshift=10, font=dict(size=10))
    fig.update_layout(height=400)
    return fig

def graficar_efectos_e_interacciones(df, factores, resp):
    """Genera gráficos de líneas para Efectos Principales e Interacciones de 2 vías."""
    st.markdown("---")
    st.subheader("📈 Gráficos de Efectos Principales")
    cols = st.columns(min(len(factores), 3))
    for i, f in enumerate(factores):
        with cols[i % len(cols)]:
            medias = df.groupby(f)[resp].mean().reset_index()
            fig = px.line(medias, x=f, y=resp, markers=True, title=f"Efecto Principal: {f}")
            fig.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    if len(factores) >= 2:
        st.subheader("🔄 Gráficos de Interacciones (Pares)")
        pares = list(combinations(factores, 2))
        if len(pares) > 6:
            st.caption("Mostrando las primeras 6 interacciones...")
            pares = pares[:6]

        cols_int = st.columns(2)
        for i, (f1, f2) in enumerate(pares):
            with cols_int[i % 2]:
                medias = df.groupby([f1, f2])[resp].mean().reset_index()
                fig = px.line(medias, x=f1, y=resp, color=f2, markers=True, title=f"Interacción: {f1} × {f2}")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ⭐ NUEVO: SUMMARY STATISTICS (ESTILO STATGRAPHICS)
# ============================================================

def calcular_summary_statistics(modelo, df, factores, respuesta, alpha=0.05):
    from scipy.stats import t as t_dist
    try:
        anova_table = sm.stats.anova_lm(modelo, typ=2)
        if 'Residual' in anova_table.index:
            mse = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
            gl_error = anova_table.loc['Residual', 'df']
        else:
            mse = modelo.mse_resid
            gl_error = modelo.df_resid
    except Exception:
        mse = modelo.mse_resid
        gl_error = modelo.df_resid

    t_crit = t_dist.ppf(1 - alpha / 2, gl_error) if gl_error > 0 else np.nan

    filas = []
    grand_mean = df[respuesta].mean()
    n_total = len(df)
    se_grand = np.sqrt(mse / n_total) if pd.notna(mse) and mse > 0 else np.nan
    filas.append({
        'Level': 'GRAND MEAN', 'Count': n_total, 'Mean': grand_mean,
        'Stnd. Error': se_grand,
        'Lower Limit': grand_mean - t_crit * se_grand if pd.notna(t_crit) else np.nan,
        'Upper Limit': grand_mean + t_crit * se_grand if pd.notna(t_crit) else np.nan,
    })

    for factor in factores:
        filas.append({'Level': factor, 'Count': None, 'Mean': None,
                      'Stnd. Error': None, 'Lower Limit': None, 'Upper Limit': None})
        try:
            niveles = sorted(df[factor].dropna().unique(), key=lambda x: (str(type(x)), str(x)))
        except Exception:
            niveles = list(df[factor].dropna().unique())
        if len(niveles) > 30:
            filas.append({'Level': '   (factor continuo)', 'Count': None, 'Mean': None,
                          'Stnd. Error': None, 'Lower Limit': None, 'Upper Limit': None})
            continue
        for nivel in niveles:
            sub = df[df[factor] == nivel]
            n = len(sub)
            if n == 0:
                continue
            media = sub[respuesta].mean()
            se = np.sqrt(mse / n) if pd.notna(mse) and mse > 0 else np.nan
            filas.append({
                'Level': str(nivel), 'Count': n, 'Mean': media, 'Stnd. Error': se,
                'Lower Limit': media - t_crit * se if pd.notna(t_crit) else np.nan,
                'Upper Limit': media + t_crit * se if pd.notna(t_crit) else np.nan,
            })

    return pd.DataFrame(filas), mse, gl_error, t_crit

def mostrar_summary_statistics(modelo, df, factores, respuesta, key_suffix="", alpha=0.05, incluir_descriptivas=True):
    st.subheader("📋 Summary Statistics — Least Squares Means (95% CI)")
    try:
        tabla, mse, gl_error, t_crit = calcular_summary_statistics(modelo, df, factores, respuesta, alpha)
        tabla_fmt = tabla.copy()
        for col in ['Mean', 'Stnd. Error', 'Lower Limit', 'Upper Limit']:
            tabla_fmt[col] = tabla_fmt[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (int, float, np.floating)) else "")
        tabla_fmt['Count'] = tabla_fmt['Count'].apply(lambda x: str(int(x)) if pd.notna(x) and x is not None and x != "" else "")

        st.caption(
            f"**MSE = {mse:.6f}  |  GL Error = {int(gl_error)}  |  "
            f"Nivel de confianza = {(1-alpha)*100:.1f}%  |  t crítico = {t_crit:.4f}**"
        )
        st.dataframe(tabla_fmt, use_container_width=True, hide_index=True)

        csv = tabla.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar tabla LS Means (CSV)", data=csv, file_name=f"lsmeans_{key_suffix}.csv", mime="text/csv", key=f"dl_ss_{key_suffix}")

        if incluir_descriptivas:
            with st.expander("📈 Estadísticas descriptivas (media, mediana, moda, cuartiles, CV)", expanded=False):
                filas_desc = []
                for factor in factores:
                    try:
                        niveles = sorted(df[factor].dropna().unique(), key=lambda x: (str(type(x)), str(x)))
                    except Exception:
                        niveles = list(df[factor].dropna().unique())
                    if len(niveles) > 30: continue
                    for nivel in niveles:
                        s = df[df[factor] == nivel][respuesta]
                        moda_vals = s.mode()
                        filas_desc.append({
                            'Factor': factor, 'Nivel': str(nivel), 'N': len(s),
                            'Media': s.mean(), 'Mediana': s.median(),
                            'Moda': moda_vals.iloc[0] if len(moda_vals) > 0 else np.nan,
                            'Desv.Est.': s.std(), 'Varianza': s.var(),
                            'Mínimo': s.min(), 'Q1': s.quantile(0.25),
                            'Q3': s.quantile(0.75), 'Máximo': s.max(),
                            'CV (%)': (s.std() / s.mean()) * 100 if s.mean() != 0 else np.nan,
                        })
                s = df[respuesta]
                moda_vals = s.mode()
                filas_desc.insert(0, {
                    'Factor': 'GLOBAL', 'Nivel': '—', 'N': len(s),
                    'Media': s.mean(), 'Mediana': s.median(),
                    'Moda': moda_vals.iloc[0] if len(moda_vals) > 0 else np.nan,
                    'Desv.Est.': s.std(), 'Varianza': s.var(),
                    'Mínimo': s.min(), 'Q1': s.quantile(0.25),
                    'Q3': s.quantile(0.75), 'Máximo': s.max(),
                    'CV (%)': (s.std() / s.mean()) * 100 if s.mean() != 0 else np.nan,
                })
                df_desc = pd.DataFrame(filas_desc)
                st.dataframe(
                    df_desc.style.format({
                        'Media': '{:.4f}', 'Mediana': '{:.4f}', 'Moda': '{:.4f}',
                        'Desv.Est.': '{:.4f}', 'Varianza': '{:.4f}',
                        'Mínimo': '{:.4f}', 'Q1': '{:.4f}', 'Q3': '{:.4f}',
                        'Máximo': '{:.4f}', 'CV (%)': '{:.2f}'
                    }),
                    use_container_width=True, hide_index=True
                )
                st.download_button("📥 Descargar descriptivas (CSV)", data=df_desc.to_csv(index=False).encode('utf-8'), file_name=f"descriptivas_{key_suffix}.csv", mime="text/csv", key=f"dl_desc_{key_suffix}")
    except Exception as e:
        st.warning(f"⚠️ No se pudieron calcular las estadísticas: {e}")

# ============================================================
# PRUEBAS MÚLTIPLES Y RESIDUOS
# ============================================================

def realizar_tukey(df, factor, respuesta, alpha=0.05):
    try:
        df[factor] = df[factor].astype('category')
        return pairwise_tukeyhsd(df[respuesta], df[factor], alpha=alpha)
    except: return None

def interpretar_tukey(tukey):
    interpretacion = []
    if tukey is None: return ["❌ No se pudo realizar la prueba de Tukey"]
    try:
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        significativas = tukey_df[tukey_df['reject'] == True]
        if len(significativas) == 0:
            interpretacion.append("📊 **Tukey HSD:** No hay diferencias significativas")
            return interpretacion
        interpretacion.append(f"📊 **Tukey HSD:** {len(significativas)} diferencias significativas")
        for _, row in significativas.iterrows():
            interpretacion.append(f"  • {row['group1']} vs {row['group2']}: Diff = {row['meandiff']:.4f} (p = {row['p-adj']:.6f})")
    except Exception as e:
        interpretacion.append(f"❌ Error: {str(e)}")
    return interpretacion

def realizar_lsd(df, factor, respuesta, mse, gl_error, alpha=0.05):
    try:
        if mse is None or gl_error is None or gl_error == 0: return None
        medias = df.groupby(factor)[respuesta].mean()
        grupos = medias.index.tolist()
        t_crit = stats.t.ppf(1 - alpha/2, gl_error)
        resultados = []
        for i in range(len(grupos)):
            for j in range(i+1, len(grupos)):
                g1, g2 = grupos[i], grupos[j]
                diff = medias[g1] - medias[g2]
                n1 = len(df[df[factor] == g1])
                n2 = len(df[df[factor] == g2])
                se_diff = np.sqrt(mse * (1/n1 + 1/n2))
                t_val = diff / se_diff
                p_val = 2 * (1 - stats.t.cdf(abs(t_val), gl_error))
                lsd = t_crit * se_diff
                significativo = p_val < alpha
                resultados.append({
                    'group1': g1, 'group2': g2, 'meandiff': diff,
                    'p-adj': p_val, 'lower': diff - lsd, 'upper': diff + lsd,
                    'reject': significativo
                })
        return pd.DataFrame(resultados)
    except Exception as e: return None

def interpretar_lsd(lsd_df):
    interpretacion = []
    if lsd_df is None or len(lsd_df) == 0: return ["❌ No se pudo realizar la prueba de LSD"]
    try:
        significativas = lsd_df[lsd_df['reject'] == True]
        if len(significativas) == 0:
            interpretacion.append("📊 **LSD:** No hay diferencias significativas")
            return interpretacion
        interpretacion.append(f"📊 **LSD:** {len(significativas)} diferencias significativas")
        for _, row in significativas.iterrows():
            interpretacion.append(f"  • {row['group1']} vs {row['group2']}: Diff = {row['meandiff']:.4f} (p = {row['p-adj']:.6f})")
    except Exception as e: interpretacion.append(f"❌ Error: {str(e)}")
    return interpretacion

def extraer_mse_gl(anova_table):
    try:
        if 'Residual' in anova_table.index: return anova_table.loc['Residual', 'Mean Sq'], anova_table.loc['Residual', 'DF']
        else: return anova_table['Mean Sq'].iloc[-1], anova_table['DF'].iloc[-1]
    except: return None, None

def mostrar_graficos_residuos_directo(modelo, df, factor=None, respuesta=None, key_suffix="", titulo="Residuos"):
    if modelo is None: return
    st.subheader(f"📈 {titulo}")
    with st.spinner("Generando gráficos de verificación del modelo..."):
        fig = graficar_verificacion_modelo_completa(modelo, df, factor, respuesta)
        if fig: st.plotly_chart(fig, use_container_width=True)
        st.subheader("📊 Pruebas de Normalidad de Residuos")
        from scipy.stats import shapiro, anderson
        residuos = modelo.resid
        col1, col2 = st.columns(2)
        with col1:
            try:
                shapiro_stat, shapiro_p = shapiro(residuos)
                st.metric("Shapiro-Wilk", f"p = {shapiro_p:.6f}")
                if shapiro_p > 0.05: st.success("✅ Los residuos siguen distribución normal (p > 0.05)")
                else: st.warning("⚠️ Los residuos NO siguen distribución normal (p ≤ 0.05)")
            except: st.warning("⚠️ No se pudo realizar la prueba de Shapiro-Wilk")
        with col2:
            try:
                anderson_result = anderson(residuos)
                st.metric("Anderson-Darling", f"Stat = {anderson_result.statistic:.4f}")
                if anderson_result.statistic < anderson_result.critical_values[2]: st.success("✅ Los residuos siguen distribución normal (al 5%)")
                else: st.warning("⚠️ Los residuos NO siguen distribución normal (al 5%)")
            except: st.warning("⚠️ No se pudo realizar la prueba de Anderson-Darling")

# ============================================================
# ⭐ NUEVO: FUNCIONES GLOBALES PARA PARETO, DANIEL Y EXCLUIR
# ============================================================

def mostrar_pareto_y_daniel(modelo, anova, resp, n_corridas):
    st.markdown("---")
    st.subheader("📊 Gráficos para Selección de Efectos (Pareto y Daniel)")
    try:
        def clean_name(name):
            return name.replace('C(', '').replace(')', '').replace('Factor_', '').replace(':', '')
        coefs = modelo.params.drop('Intercept', errors='ignore')
        names = [clean_name(x) for x in coefs.index]
        effects = 2 * coefs.values
        gl_error = anova.loc['Residual', 'DF'] if 'Residual' in anova.index else 0

        if gl_error > 0:
            mse = anova.loc['Residual', 'Mean Sq']
            se = np.sqrt(mse * (4 / n_corridas))
            limite = stats.t.ppf(0.975, gl_error) * se
        else:
            s0 = 1.5 * np.median(np.abs(effects))
            reduced = effects[np.abs(effects) < 2.5 * s0]
            pse = 1.5 * np.median(np.abs(reduced)) if len(reduced) > 0 else 1.5 * np.median(np.abs(effects))
            limite = 2.0 * pse

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Diagrama de Pareto**")
            df_pareto = pd.DataFrame({'Efecto': names, 'Magnitud': np.abs(effects)}).sort_values(by='Magnitud', ascending=True)
            colors = ['#2b8cbe' if m > limite else '#a6bddb' for m in df_pareto['Magnitud']]
            fig_pareto = go.Figure(go.Bar(x=df_pareto['Magnitud'], y=df_pareto['Efecto'], orientation='h', marker_color=colors))
            fig_pareto.add_vline(x=limite, line_dash="dash", line_color="red")
            fig_pareto.update_layout(xaxis_title="Efecto (|E|)", height=400, showlegend=False)
            st.plotly_chart(fig_pareto, use_container_width=True)
        with col2:
            st.markdown("**Gráfico de Daniel**")
            idx = np.argsort(effects)
            sorted_effects = effects[idx]
            n = len(sorted_effects)
            z = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            fig_daniel = go.Figure()
            fig_daniel.add_trace(go.Scatter(x=sorted_effects, y=z, mode='markers+text', text=[names[i] for i in idx], textposition="top center"))
            mask = np.abs(sorted_effects) < limite
            slope, intercept = np.polyfit(sorted_effects[mask], z[mask], 1) if sum(mask) > 1 else np.polyfit(sorted_effects, z, 1)
            fig_daniel.add_trace(go.Scatter(x=[min(sorted_effects), max(sorted_effects)], y=[min(sorted_effects) * slope + intercept, max(sorted_effects) * slope + intercept], mode='lines', line_color='blue'))
            fig_daniel.update_layout(xaxis_title="Efectos Estandarizados", yaxis_title="Proporción", height=400, showlegend=False)
            st.plotly_chart(fig_daniel, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudieron generar los gráficos: {e}")

def ui_excluir_efectos(df, resp, anova, tipo_anova, key_suffix):
    st.markdown("---")
    st.subheader("🛠️ Refinamiento de Efectos (Excluir No Significativos)")
    todos_efectos = [idx for idx in anova.index if idx != 'Residual']
    efectos_seleccionados = st.multiselect("Efectos a MANTENER en el modelo final:", todos_efectos, default=todos_efectos, key=f"excluir_efectos_{key_suffix}")

    if st.button("🔄 Recalcular Modelo Reducido", key=f"btn_recalcular_{key_suffix}"):
        if len(efectos_seleccionados) == 0:
            st.warning("⚠️ Debes mantener al menos un efecto.")
        else:
            formula_reducida = f"{resp} ~ " + " + ".join(efectos_seleccionados)
            analizar_anova_mejorada(df, formula_reducida, "Modelo Reducido (Mejor ANOVA)", f"{key_suffix}_red", "reducido", tipo_anova)
            st.rerun()

# ============================================================
# CARGA DE DATOS, DISEÑO Y CONCLUSIONES
# ============================================================

def crear_diseno_completo(factores_dict, respuestas_dict, n_replicas=1, randomize=True):
    try:
        niveles_factores = [factores_dict[f]['niveles'] for f in factores_dict]
        nombres_factores = list(factores_dict.keys())
        combinaciones = list(product(*niveles_factores))
        df = pd.DataFrame(combinaciones, columns=nombres_factores)
        if n_replicas > 1: df = pd.concat([df] * n_replicas, ignore_index=True)
        if randomize: df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        for resp_name, resp_data in respuestas_dict.items():
            df[resp_name] = np.nan
        return df
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def generar_conclusiones_anova(anova_table, modelo, df, factor, respuesta, alpha=0.05):
    conclusiones = []
    p_valor = None
    for col in anova_table.columns:
        if 'P-Value' in col: p_valor = anova_table[col].iloc[0]; break
    if p_valor is None: return ["⚠️ No se obtuvo p-valor"], None
    if p_valor < alpha: conclusiones.append(f"✅ **Conclusión:** Diferencia significativa (p = {p_valor:.6f})")
    else: conclusiones.append(f"❌ **Conclusión:** Sin diferencia significativa (p = {p_valor:.6f})")
    medias = df.groupby(factor)[respuesta].agg(['mean', 'std', 'count']).reset_index()
    medias.columns = [factor, 'Media', 'Desv', 'n']
    medias_ordenadas = medias.sort_values('Media', ascending=False)
    for _, row in medias_ordenadas.iterrows():
        conclusiones.append(f"  • {row[factor]}: Media = {row['Media']:.4f} ± {row['Desv']:.4f} (n={int(row['n'])})")
    mejor, peor = medias_ordenadas.iloc[0], medias_ordenadas.iloc[-1]
    conclusiones.append(f"🏆 **Mejor:** {mejor[factor]} (Media = {mejor['Media']:.4f})")
    conclusiones.append(f"📉 **Peor:** {peor[factor]} (Media = {peor['Media']:.4f})")
    cv = (df[respuesta].std() / df[respuesta].mean()) * 100
    conclusiones.append(f"📈 **CV:** {cv:.2f}%")
    conclusiones.append(f"📐 **R² = {modelo.rsquared:.6f}**")
    return conclusiones, medias_ordenadas

def generar_conclusiones_factorial(anova_table, modelo, df, factores, respuesta, alpha=0.05):
    conclusiones = []
    for i, row in anova_table.iterrows():
        p_valor = None
        for col in anova_table.columns:
            if 'P-Value' in col: p_valor = row[col]; break
        if p_valor is None: continue
        if p_valor < alpha: conclusiones.append(f"✅ **{i}:** Efecto significativo (p = {p_valor:.6f})")
        else: conclusiones.append(f"❌ **{i}:** No significativo (p = {p_valor:.6f})")
    conclusiones.append(f"📐 **R² = {modelo.rsquared:.6f}**")
    return conclusiones

def detectar_separador(texto):
    lineas = [l.strip() for l in texto.split('\n') if l.strip()]
    if not lineas: return ','
    conteos = {sep: sum(l.count(sep) for l in lineas) for sep in [',', ';', '\t', '|', ' ']}
    mejor_sep = max(conteos, key=conteos.get)
    return mejor_sep if conteos[mejor_sep] > 0 else ','

def limpiar_nombres_columnas(df):
    df.columns = [c.strip().replace(' ', '_').replace(';', '').replace(',', '') for c in df.columns]
    return df

def convertir_tipos(df):
    for col in df.columns:
        try: df[col] = pd.to_numeric(df[col])
        except: df[col] = df[col].astype('category')
    return df

def cargar_datos_desde_csv(archivo, separador=None):
    try:
        contenido = archivo.getvalue().decode('utf-8')
        if separador is None: separador = detectar_separador(contenido)
        try: df = pd.read_csv(io.StringIO(contenido), sep=separador)
        except:
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(contenido), sep=sep)
                    if len(df.columns) > 1: break
                except: continue
        return convertir_tipos(limpiar_nombres_columnas(df))
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def cargar_datos_desde_url(url, separador=None):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        if separador is None: separador = detectar_separador(content)
        try: df = pd.read_csv(io.StringIO(content), sep=separador)
        except:
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep)
                    if len(df.columns) > 1: break
                except: continue
        return convertir_tipos(limpiar_nombres_columnas(df))
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def procesar_datos_texto(texto, n_columnas=None):
    try:
        separador = detectar_separador(texto)
        lineas = [l.strip() for l in texto.split('\n') if l.strip()]
        if not lineas: return None
        valores = lineas[0].split(separador)
        tiene_encabezado = False
        try:
            for v in valores: float(v.strip())
        except: tiene_encabezado = True
        datos, columnas = [], None
        if tiene_encabezado:
            columnas = [v.strip() for v in valores]
            lineas = lineas[1:]
        for linea in lineas:
            valores = [v.strip() for v in linea.split(separador) if v.strip()]
            if n_columnas is None or len(valores) == n_columnas:
                try: datos.append([float(v) for v in valores])
                except: datos.append(valores)
        if not datos: return None
        df = pd.DataFrame(datos)
        if columnas and len(columnas) == len(df.columns): df.columns = columnas
        else: df.columns = [f'V{i+1}' for i in range(len(df.columns))]
        return convertir_tipos(limpiar_nombres_columnas(df))
    except: return None

def obtener_datos_generales(key_suffix, n_columnas_esperadas=None):
    st.subheader("📥 Carga de Datos")
    tipo_carga = st.radio("Método:", ["✏️ Manual", "📁 CSV", "🔗 URL"], horizontal=True, key=f"tipo_carga_{key_suffix}")
    df_resultado = None
    if tipo_carga == "✏️ Manual":
        if n_columnas_esperadas: st.caption(f"Formato: {n_columnas_esperadas} columnas")
        datos_texto = st.text_area("Datos:", value="", height=200, key=f"text_area_{key_suffix}", placeholder="A,23\nA,25\nB,30")
        if datos_texto and st.button("📊 Procesar", key=f"procesar_text_{key_suffix}"):
            df_resultado = procesar_datos_texto(datos_texto, n_columnas_esperadas)
            if df_resultado is not None:
                st.success(f"✅ {len(df_resultado)} filas cargadas")
                st.dataframe(df_resultado.head())
                st.session_state.datos_cargados[key_suffix] = df_resultado
    elif tipo_carga == "📁 CSV":
        archivo = st.file_uploader("Archivo:", type=['csv', 'txt'], key=f"csv_upload_{key_suffix}")
        if archivo is not None and st.button("📊 Cargar", key=f"cargar_csv_{key_suffix}"):
            df_resultado = cargar_datos_desde_csv(archivo)
            if df_resultado is not None:
                st.success(f"✅ {len(df_resultado)} filas")
                st.dataframe(df_resultado.head())
                st.session_state.datos_cargados[key_suffix] = df_resultado
    elif tipo_carga == "🔗 URL":
        url = st.text_input("URL:", key=f"url_input_{key_suffix}")
        if url and st.button("📊 Cargar", key=f"cargar_url_{key_suffix}"):
            df_resultado = cargar_datos_desde_url(url)
            if df_resultado is not None:
                st.success(f"✅ {len(df_resultado)} filas")
                st.dataframe(df_resultado.head())
                st.session_state.datos_cargados[key_suffix] = df_resultado
    if key_suffix in st.session_state.datos_cargados:
        df_guardado = st.session_state.datos_cargados[key_suffix]
        st.info(f"📊 Datos previos: {len(df_guardado)} filas")
        return df_guardado
    return df_resultado

def mostrar_informacion_datos(df, key_suffix):
    if df is not None:
        col1, col2 = st.columns(2)
        with col1: st.metric("Filas", len(df))
        with col2: st.metric("Columnas", len(df.columns))

# ============================================================


# ============================================================

# ============================================================
# DISEÑOS FACTORIALES FRACCIONADOS 2^(k-p) — MÓDULO CRIBADO
# ============================================================

def rango_generadores_gf2(vectores):
    """Rango de vectores binarios representados como enteros (álgebra GF(2))."""
    pivotes = {}
    for valor in vectores:
        x = int(valor)
        while x:
            pivote = x.bit_length() - 1
            if pivote in pivotes:
                x ^= pivotes[pivote]
            else:
                pivotes[pivote] = x
                break
    return len(pivotes)


def sugerir_generadores_fraccion(k, p):
    """Busca generadores regulares linealmente independientes y prioriza resolución."""
    import itertools
    if not (1 <= p <= min(4, k // 2)):
        raise ValueError(f"Para k={k}, el orden p debe estar entre 1 y {min(4, k // 2)}.")
    m = k - p
    base = list("ABCDEFGH"[:m])
    extra = list("ABCDEFGH"[m:k])
    candidatos = [
        sum(1 << i for i in c)
        for r in range(1, m + 1)
        for c in itertools.combinations(range(m), r)
    ]
    mejor = None
    for elegidos in itertools.combinations(candidatos, p):
        # Los p generadores deben ser independientes; si no, la fracción
        # tendría menos de 2^(k-p) corridas distintas.
        if rango_generadores_gf2(elegidos) != p:
            continue
        longitudes = []
        for mascara in range(1, 1 << p):
            palabra = 0
            for j in range(p):
                if (mascara >> j) & 1:
                    palabra ^= elegidos[j]
            longitudes.append(palabra.bit_count() + mascara.bit_count())
        puntaje = (min(longitudes), sum(longitudes), tuple(-x for x in sorted(longitudes)))
        if mejor is None or puntaje > mejor[0]:
            mejor = (puntaje, elegidos)
    if mejor is None:
        raise ValueError("No fue posible construir generadores independientes para esta fracción.")
    elegidos = mejor[1]
    generadores = [
        f"{extra[j]}=" + "".join(base[i] for i in range(m) if (elegidos[j] >> i) & 1)
        for j in range(p)
    ]
    return generadores, mejor[0][0]


def analizar_generadores_fraccion(generadores_txt, k, p):
    """Valida exactamente p generadores, aceptando saltos reales o '\\\\n' literales."""
    import re
    letras = list("ABCDEFGH"[:k])
    base = set(letras[:k-p])
    extras = set(letras[k-p:])

    # Streamlit puede conservar en el textarea los dos caracteres '\\\\n'
    # cuando una versión previa construyó el valor sugerido incorrectamente.
    texto = str(generadores_txt or "")
    texto = texto.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    lineas = [x.strip().upper().replace(" ", "").replace("\\t", "")
              for x in re.split(r"[\r\n;]+", texto) if x.strip()]
    generadores = {}
    if len(lineas) != p:
        raise ValueError(
            f"Se requieren exactamente {p} generadores independientes; "
            f"se detectaron {len(lineas)}. Escribe uno por línea."
        )
    vectores = []
    for linea in lineas:
        if linea.count("=") != 1:
            raise ValueError(f"Generador inválido: {linea}. Usa la forma E=ABCD.")
        lhs, rhs = linea.split("=", 1)
        if lhs not in extras:
            raise ValueError(f"{lhs} debe ser uno de los factores generados: {', '.join(sorted(extras))}.")
        if lhs in generadores:
            raise ValueError(f"El generador {lhs} está repetido.")
        if not rhs or any(ch not in base for ch in rhs):
            raise ValueError(f"El lado derecho de {lhs} debe contener solo factores básicos ({''.join(sorted(base))}).")
        if len(set(rhs)) != len(rhs):
            raise ValueError(f"El generador {lhs} repite letras.")
        rhs_ordenado = "".join(sorted(rhs, key=letras.index))
        generadores[lhs] = rhs_ordenado
        vectores.append(sum(1 << letras.index(ch) for ch in rhs_ordenado))
    if set(generadores) != extras:
        faltan = sorted(extras - set(generadores))
        raise ValueError(f"Faltan generadores para: {', '.join(faltan)}.")
    if rango_generadores_gf2(vectores) != p:
        raise ValueError(
            "Los generadores no son independientes en GF(2). "
            "Modifica las combinaciones para obtener exactamente 2^(k-p) corridas distintas."
        )
    return generadores


def palabras_definitorias_fraccion(generadores, k):
    """Devuelve el subgrupo definitorio como máscaras binarias."""
    letras = list("ABCDEFGH"[:k])
    relaciones = []
    for lhs, rhs in generadores.items():
        mascara = 1 << letras.index(lhs)
        for ch in rhs:
            mascara ^= (1 << letras.index(ch))
        relaciones.append(mascara)
    palabras = {0}
    for rel in relaciones:
        palabras |= {x ^ rel for x in list(palabras)}
    return sorted(palabras)


def texto_efecto_mascara(mask, letras):
    return "".join(letras[i] for i in range(len(letras)) if (mask >> i) & 1) or "I"


def resolucion_fraccion(palabras):
    longitudes = [int(x).bit_count() for x in palabras if x]
    return min(longitudes) if longitudes else None


def construir_estructura_alias(generadores, k, max_orden=2):
    letras = list("ABCDEFGH"[:k])
    definitorias = palabras_definitorias_fraccion(generadores, k)
    efectos = []
    import itertools
    for orden in range(1, min(max_orden, k) + 1):
        for comb in itertools.combinations(range(k), orden):
            mask = sum(1 << i for i in comb)
            alias_masks = sorted({mask ^ d for d in definitorias})
            alias_txt = " = ".join(texto_efecto_mascara(x, letras) for x in alias_masks)
            efectos.append({"Efecto": texto_efecto_mascara(mask, letras),
                            "Estructura de alias": alias_txt,
                            "Orden": orden})
    return pd.DataFrame(efectos), definitorias


def generar_diseno_fraccionado(factores, respuesta, p, generadores_txt,
                               replicas=1, puntos_centro=0, randomizar=True, semilla=123):
    """Genera la matriz 2^(k-p) y la traduce a niveles reales."""
    import itertools
    nombres = list(factores.keys())
    k = len(nombres)
    if k < 3 or k > 8:
        raise ValueError("Para el módulo fraccionado se admiten entre 3 y 8 factores.")
    if p < 1 or p > min(4, k // 2):
        raise ValueError(f"Para k={k}, el orden p debe estar entre 1 y {min(4, k // 2)}; se requieren al menos p factores básicos.")
    generadores = analizar_generadores_fraccion(generadores_txt, k, p)
    letras = list("ABCDEFGH"[:k])
    base_letras = letras[:k-p]
    extras = letras[k-p:]
    matriz_base = list(itertools.product([-1, 1], repeat=k-p))
    filas_cod = []
    for fila in matriz_base:
        cod = dict(zip(base_letras, fila))
        for extra in extras:
            cod[extra] = int(np.prod([cod[x] for x in generadores[extra]]))
        filas_cod.append(cod)

    # Repeticiones completas del diseño base
    filas = []
    for rep in range(replicas):
        for cod in filas_cod:
            fila = {}
            for j, nombre in enumerate(nombres):
                niveles = factores[nombre]["niveles"]
                bajo, alto = niveles[0], niveles[1]
                valor = bajo if cod[letras[j]] == -1 else alto
                try:
                    valor = float(valor)
                    if float(valor).is_integer(): valor = int(valor)
                except Exception:
                    pass
                fila[nombre] = valor
            fila[respuesta] = np.nan
            filas.append(fila)

    # Los puntos centrales se agregan solo para factores continuos numéricos.
    if puntos_centro:
        no_continuos = [n for n in nombres if factores[n].get("tipo") != "Continuo"]
        no_numericos = []
        for n in nombres:
            try:
                float(factores[n]["niveles"][0]); float(factores[n]["niveles"][1])
            except Exception:
                no_numericos.append(n)
        if no_continuos or no_numericos:
            raise ValueError("Los puntos centrales requieren que todos los factores sean continuos y tengan niveles numéricos.")
        for _ in range(puntos_centro):
            centro = {}
            for n in nombres:
                lo, hi = map(float, factores[n]["niveles"][:2])
                centro[n] = (lo + hi) / 2
            centro[respuesta] = np.nan
            filas.append(centro)

    df = pd.DataFrame(filas)
    if randomizar:
        df = df.sample(frac=1, random_state=semilla).reset_index(drop=True)
    df.insert(0, "Corrida", np.arange(1, len(df) + 1))
    meta = {
        "k": k, "p": p, "n_base": 2 ** (k-p), "generadores": generadores,
        "resolucion": resolucion_fraccion(palabras_definitorias_fraccion(generadores, k)),
        "letras": letras, "base_letras": base_letras, "extras": extras,
        "n_replicas": replicas, "puntos_centro": puntos_centro,
        "randomizar": randomizar, "factores": nombres
    }
    return df, meta


def codificar_factores_fraccion(df, factores_def, meta):
    """Codifica niveles reales en -1/+1 y centros en 0."""
    salida = pd.DataFrame(index=df.index)
    letras = meta["letras"]
    for j, nombre in enumerate(meta["factores"]):
        lo, hi = factores_def[nombre]["niveles"][:2]
        s = df[nombre]
        try:
            lo_n, hi_n = float(lo), float(hi)
            vals = pd.to_numeric(s, errors="coerce")
            cod = 2 * (vals - lo_n) / (hi_n - lo_n) - 1
            cod = cod.where(vals.notna(), np.nan)
        except Exception:
            cod = s.map({str(lo): -1.0, str(hi): 1.0, lo: -1.0, hi: 1.0})
        salida[letras[j]] = pd.to_numeric(cod, errors="coerce")
    return salida


def matriz_efectos_fraccion(codigos, max_orden=2):
    import itertools
    cols = {}
    letras = list(codigos.columns)
    for orden in range(1, min(max_orden, len(letras)) + 1):
        for comb in itertools.combinations(letras, orden):
            nombre = "".join(comb)
            v = np.ones(len(codigos))
            for letra in comb:
                v = v * codigos[letra].to_numpy(dtype=float)
            cols[nombre] = v
    return pd.DataFrame(cols, index=codigos.index)


def seleccionar_efectos_estimables(matriz, tolerancia=1e-9):
    """Evita ajustar simultáneamente columnas perfectamente aliasadas."""
    elegidos, descartados = [], []
    for col in matriz.columns:
        v = matriz[col].to_numpy(dtype=float)
        if not np.isfinite(v).all() or np.std(v) < tolerancia:
            descartados.append((col, "Sin variación o con valores faltantes"))
            continue
        duplicado = None
        for prev in elegidos:
            u = matriz[prev].to_numpy(dtype=float)
            corr = np.corrcoef(v, u)[0, 1]
            if np.isfinite(corr) and abs(abs(corr) - 1) < 1e-8:
                duplicado = (prev, corr)
                break
        if duplicado:
            descartados.append((col, f"Alias con {duplicado[0]} (r={duplicado[1]:.0f})"))
        else:
            elegidos.append(col)
    return elegidos, descartados


def mostrar_analisis_fraccionado(df_diseno, respuesta, factores_def, meta, alpha=0.05):
    """Análisis tipo STATGRAPHICS: efectos, alias, correlaciones, ANOVA, Pareto y diagnóstico."""
    st.markdown("## 1. Atributos del diseño")
    res = meta["resolucion"]
    etiqueta_res = {3:"III", 4:"IV", 5:"V", 6:"VI", 7:"VII", 8:"VIII"}.get(res, str(res))
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Diseño", f"2^{meta['k']}-{meta['p']}")
    c2.metric("Corridas base", meta["n_base"])
    c3.metric("Resolución", etiqueta_res)
    c4.metric("Corridas ejecutadas", len(df_diseno))
    st.write("**Generadores:** " + "; ".join(f"{k}={v}" for k,v in meta["generadores"].items()))
    st.caption("La resolución se obtiene como la longitud mínima de las palabras no identidad de la relación definitoria.")

    st.markdown("## 2. Estructura de alias")
    alias_df, definitorias = construir_estructura_alias(meta["generadores"], meta["k"], 2)
    st.dataframe(alias_df[["Efecto","Estructura de alias"]], use_container_width=True, hide_index=True)
    palabras = [texto_efecto_mascara(x, meta["letras"]) for x in definitorias]
    st.write("**Relación definitoria:** I = " + " = ".join(palabras[1:]))
    st.caption("Los efectos unidos por el signo igual no pueden separarse con este diseño sin corridas adicionales o supuestos sobre efectos despreciables.")

    codigos = codificar_factores_fraccion(df_diseno, factores_def, meta)
    efectos = matriz_efectos_fraccion(codigos, 2)
    st.markdown("## 3. Matriz de correlaciones de efectos")
    corr = efectos.corr()
    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Correlaciones entre columnas de efectos")
    fig_corr.update_layout(height=max(450, min(900, 24*len(corr.columns))))
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Valores cercanos a +1 o -1 indican columnas idénticas o confundidas; valores cercanos a 0 indican ortogonalidad entre columnas.")

    st.markdown("## 4. Ajuste del modelo y tabla ANOVA")
    datos = df_diseno.copy()
    y = pd.to_numeric(datos[respuesta], errors="coerce")
    validos = y.notna()
    cod_validos = codigos.loc[validos].reset_index(drop=True)
    y_validos = y.loc[validos].reset_index(drop=True)
    if len(y_validos) < 3:
        st.warning("Ingresa al menos tres respuestas observadas para ajustar el modelo.")
        return
    efectos_validos = matriz_efectos_fraccion(cod_validos, 2)
    estimables, descartados = seleccionar_efectos_estimables(efectos_validos)
    # Se reservan al menos 3 g.l. para error cuando el tamaño muestral lo permite,
    # siguiendo la recomendación del manual de STATGRAPHICS.
    max_terminos = max(0, len(y_validos) - 4)
    if len(estimables) > max_terminos:
        descartados.extend((e, "No incluido: se reservan grados de libertad para estimar el error") for e in estimables[max_terminos:])
        estimables = estimables[:max_terminos]
    if not estimables:
        st.warning("No hay grados de libertad suficientes para estimar efectos y error. Agrega réplicas del diseño o puntos centrales y vuelve a ejecutar el análisis.")
        return
    df_modelo = pd.DataFrame({"Y_resp": y_validos})
    nombres_modelo = {}
    for j, efecto in enumerate(estimables):
        nombre_col = f"E{j+1}"
        df_modelo[nombre_col] = efectos_validos[efecto].to_numpy()
        nombres_modelo[nombre_col] = efecto
    formula = "Y_resp ~ " + " + ".join(nombres_modelo.keys())
    modelo, anova = analizar_anova_mejorada(df_modelo, formula, "ANOVA del diseño fraccionado",
                                            "fraccionado", "screening", 2)
    if modelo is None:
        return
    anova_display = anova.copy()
    anova_display.index = [nombres_modelo.get(str(x), str(x)) for x in anova_display.index]
    st.dataframe(anova_display, use_container_width=True)
    st.write(f"**R²:** {modelo.rsquared:.4f} · **R² ajustado:** {modelo.rsquared_adj:.4f} · **Desv. estándar residual:** {np.sqrt(modelo.mse_resid):.4f}")
    if descartados:
        with st.expander("Efectos no estimados separadamente"):
            st.dataframe(pd.DataFrame(descartados, columns=["Efecto","Motivo"]), hide_index=True)

    st.markdown("## 5. Efectos estimados y gráfico de Pareto")
    filas_ef = []
    gl = modelo.df_resid
    tcrit = stats.t.ppf(1-alpha/2, gl) if gl > 0 else np.nan
    for col, efecto in nombres_modelo.items():
        beta = modelo.params.get(col, np.nan)
        se = modelo.bse.get(col, np.nan)
        tval = modelo.tvalues.get(col, np.nan)
        pval = modelo.pvalues.get(col, np.nan)
        filas_ef.append({"Efecto":efecto, "Estimación del efecto":2*beta,
                         "Error estándar":2*se, "t":tval, "Valor-p":pval,
                         "Significativo":bool(pval < alpha) if np.isfinite(pval) else False})
    efectos_tabla = pd.DataFrame(filas_ef).sort_values("Valor-p")
    st.dataframe(efectos_tabla, use_container_width=True, hide_index=True)
    fig_p = px.bar(efectos_tabla.sort_values("Estimación del efecto", key=lambda s: s.abs()),
                   x="Estimación del efecto", y="Efecto", orientation="h",
                   color="Significativo", title="Pareto de efectos estimados (magnitud y signo)")
    fig_p.update_layout(height=max(400, 28*len(efectos_tabla)))
    st.plotly_chart(fig_p, use_container_width=True)
    if gl > 0:
        fig_t = px.bar(efectos_tabla.assign(Abs_t=efectos_tabla["t"].abs()).sort_values("Abs_t"),
                       x="Abs_t", y="Efecto", orientation="h",
                       title=f"Pareto estandarizado |t| (referencia t crítico = {tcrit:.3f})")
        fig_t.add_vline(x=tcrit, line_dash="dash", line_color="red",
                        annotation_text=f"α={alpha:.3f}")
        st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("## 6. Diagnóstico del modelo")
    resid = modelo.resid
    ajust = modelo.fittedvalues
    col1,col2 = st.columns(2)
    with col1:
        fig_r = px.scatter(x=ajust, y=resid, labels={"x":"Valores ajustados","y":"Residuales"},
                           title="Residuales contra valores ajustados")
        fig_r.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_r, use_container_width=True)
    with col2:
        qq = stats.probplot(resid, dist="norm")
        fig_q = px.scatter(x=qq[0][0], y=qq[0][1], labels={"x":"Cuantiles normales","y":"Residuales ordenados"},
                           title="Gráfico de probabilidad normal")
        slope, intercept, _ = qq[1]
        xx = np.asarray(qq[0][0])
        fig_q.add_scatter(x=xx, y=intercept+slope*xx, mode="lines", name="Referencia")
        st.plotly_chart(fig_q, use_container_width=True)

    st.markdown("## 7. Gráficos de efectos principales")
    for letra, nombre in zip(meta["letras"], meta["factores"]):
        if letra in codigos.columns:
            lo, hi = factores_def[nombre]["niveles"][:2]
            vals = []
            for nivel in [-1,1]:
                fila = {x:0.0 for x in meta["letras"]}
                fila[letra] = nivel
                for col, efecto in nombres_modelo.items():
                    fila[col] = np.prod([fila[x] for x in efecto])
                pred = float(modelo.predict(pd.DataFrame([{col:fila[col] for col in nombres_modelo}]))[0])
                vals.append({"Nivel":str(lo if nivel==-1 else hi),"Respuesta predicha":pred})
            fig_m = px.line(pd.DataFrame(vals), x="Nivel", y="Respuesta predicha",
                            markers=True, title=f"Efecto principal: {nombre}")
            st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("## 8. Optimización exploratoria de la respuesta")
    meta_obj = st.selectbox("Objetivo", ["Maximizar","Minimizar","Alcanzar un objetivo"], key="fr_objetivo")
    objetivo = None
    if meta_obj == "Alcanzar un objetivo":
        objetivo = st.number_input("Valor objetivo", value=float(y_validos.mean()), key="fr_target")
    import itertools
    candidatos = []
    for esquina in itertools.product([-1,1], repeat=meta["k"]):
        fila_cod = dict(zip(meta["letras"], esquina))
        xrow = {}
        for col, efecto in nombres_modelo.items():
            xrow[col] = float(np.prod([fila_cod[x] for x in efecto]))
        pred = float(modelo.predict(pd.DataFrame([xrow]))[0])
        candidatos.append((pred, esquina))
    if meta_obj == "Maximizar": elegido = max(candidatos, key=lambda z:z[0])
    elif meta_obj == "Minimizar": elegido = min(candidatos, key=lambda z:z[0])
    else: elegido = min(candidatos, key=lambda z:abs(z[0]-objetivo))
    st.metric("Respuesta predicha en el mejor vértice", f"{elegido[0]:.4f}")
    opt_rows = []
    for j,nombre in enumerate(meta["factores"]):
        niveles = factores_def[nombre]["niveles"][:2]
        opt_rows.append({"Factor":nombre,"Nivel seleccionado":niveles[0] if elegido[1][j]==-1 else niveles[1]})
    st.dataframe(pd.DataFrame(opt_rows), use_container_width=True, hide_index=True)
    st.warning("La optimización se realiza en los vértices del rango experimental y es exploratoria. No extrapoles fuera de los niveles estudiados; confirma experimentalmente la condición recomendada.")


# WIZARD INTERACTIVO — APLICACIÓN INDEPENDIENTE
# ============================================================

st.header("🎯 Wizard de Diseño Experimental (Estilo Statgraphics)")
st.markdown("Crea tu diseño paso a paso, define los factores, y la aplicación analizará automáticamente.")

if 'wiz_paso' not in st.session_state: st.session_state.wiz_paso = 1
if 'wiz_tipo_diseno' not in st.session_state: st.session_state.wiz_tipo_diseno = None
if 'wiz_factores' not in st.session_state: st.session_state.wiz_factores = {}
if 'wiz_respuesta' not in st.session_state: st.session_state.wiz_respuesta = None
if 'wiz_datos' not in st.session_state: st.session_state.wiz_datos = None
if 'wiz_config' not in st.session_state: st.session_state.wiz_config = {}

st.progress((st.session_state.wiz_paso - 1) / 6)
st.caption(f"**Paso {st.session_state.wiz_paso} de 7**")

# PASO 1
if st.session_state.wiz_paso == 1:
    st.subheader("📋 Paso 1: Tipo de Diseño")
    opciones_diseno = {
        "🔍 Cribado — Factorial completo 2^k": "screening_full",
        "🔬 Cribado — Factorial fraccionado 2^(k-p)": "screening_fractional",
        "📈 Superficie de Respuesta (MSR)": "rsm",
        "🔢 Factorial Multinivel": "multilevel", "🎯 Un Solo Factor (DCA)": "single_factor",
        "📊 Un Solo Factor con Bloques (DBCA)": "single_factor_block",
        "🔲 Cuadrado Latino (DCL)": "latin_square", "🔳 Cuadrado Greco-Latino (DCGL)": "greco_latin",
    }
    st.markdown("### 🔍 Opciones de cribado")
    st.caption("Elige entre conservar todas las combinaciones 2^k o reducir corridas mediante una fracción regular 2^(k-p).")
    tipo = st.radio("**Clase de diseño:**", list(opciones_diseno.keys()), key="wiz_tipo_radio")
    if st.button("➡️ Siguiente Paso", key="wiz_next_1"):
        st.session_state.wiz_tipo_diseno = opciones_diseno[tipo]
        st.session_state.wiz_paso = 2
        st.rerun()

# PASO 2
elif st.session_state.wiz_paso == 2:
    st.subheader("📋 Paso 2: Definición de Factores")
    tipo = st.session_state.wiz_tipo_diseno
    es_cribado = tipo in ("screening", "screening_full", "screening_fractional")
    n_factores = st.number_input("**Número de Factores:**", min_value=3 if es_cribado else 1, max_value=8, value=3 if es_cribado else (1 if tipo in ['single_factor', 'single_factor_block', 'latin_square', 'greco_latin'] else 3), step=1, key="wiz_n_factores")
    factores_def = {}
    for i in range(n_factores):
        with st.expander(f"📌 Factor {chr(65+i)}", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1: nombre = st.text_input(f"Nombre del Factor {chr(65+i)}:", value=f"Factor_{chr(65+i)}", key=f"wiz_nombre_{i}")
            with col2:
                max_niveles = 2 if es_cribado else 10
                n_niveles = st.number_input(
                    "Niveles:", min_value=2, max_value=max_niveles,
                    value=2 if es_cribado or tipo == "rsm" else 3,
                    step=1, key=f"wiz_niveles_{i}"
                )
            with col3: tipo_var = st.selectbox("Tipo:", ["Continuo", "Categórico"], key=f"wiz_tipo_{i}")
            niveles = []
            if tipo_var == "Continuo" and n_niveles == 2:
                col_low, col_high = st.columns(2)
                with col_low: low = st.text_input("Low:", value="-1.0", key=f"wiz_low_{i}")
                with col_high: high = st.text_input("High:", value="1.0", key=f"wiz_high_{i}")
                niveles = [low, high]
            else:
                cols = st.columns(min(n_niveles, 5))
                for j in range(n_niveles):
                    with cols[j % len(cols)]: niveles.append(st.text_input(f"Nivel {j+1}:", value=str(j+1), key=f"wiz_nivel_{i}_{j}"))
            factores_def[nombre] = {'niveles': niveles, 'tipo': tipo_var, 'n_niveles': n_niveles}
    if tipo == 'single_factor_block':
        with st.expander("📌 Factor de Bloque", expanded=True):
            nombre_bloque = st.text_input("Nombre del Bloque:", value="Bloque", key="wiz_nombre_bloque")
            n_niveles_bloque = st.number_input("Niveles del Bloque:", min_value=2, max_value=10, value=3, step=1, key="wiz_niveles_bloque")
            niveles_bloque = []
            cols = st.columns(min(n_niveles_bloque, 5))
            for j in range(n_niveles_bloque):
                with cols[j % len(cols)]: niveles_bloque.append(st.text_input(f"Bloque {j+1}:", value=f"B{j+1}", key=f"wiz_bloque_{j}"))
            factores_def[nombre_bloque] = {'niveles': niveles_bloque, 'tipo': 'Categórico', 'n_niveles': n_niveles_bloque, 'es_bloque': True}
    if tipo == 'latin_square':
        with st.expander("📌 Filas y Columnas (Bloqueos)", expanded=True):
            col1, col2 = st.columns(2)
            with col1: n_filas = st.number_input("Filas:", min_value=2, max_value=10, value=3, step=1, key="wiz_n_filas")
            with col2: n_cols = st.number_input("Columnas:", min_value=2, max_value=10, value=3, step=1, key="wiz_n_cols")
    if tipo == 'greco_latin':
        with st.expander("📌 Filas, Columnas y Greco (Bloqueos)", expanded=True):
            col1, col2 = st.columns(2)
            with col1: n_filas = st.number_input("Filas:", min_value=2, max_value=10, value=3, step=1, key="wiz_n_filas_g")
            with col2: n_cols = st.number_input("Columnas:", min_value=2, max_value=10, value=3, step=1, key="wiz_n_cols_g")
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("⬅️ Atrás", key="wiz_back_2"): st.session_state.wiz_paso = 1; st.rerun()
    with col_btn2:
        if st.button("➡️ Siguiente Paso", key="wiz_next_2"):
            st.session_state.wiz_factores = factores_def; st.session_state.wiz_paso = 3; st.rerun()

# PASO 3
elif st.session_state.wiz_paso == 3:
    st.subheader("📋 Paso 3: Definición de la Respuesta")
    nombre_resp = st.text_input("**Nombre de la Variable Respuesta:**", value="Respuesta", key="wiz_nombre_resp")
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("⬅️ Atrás", key="wiz_back_3"): st.session_state.wiz_paso = 2; st.rerun()
    with col_btn2:
        if st.button("➡️ Siguiente Paso", key="wiz_next_3"):
            st.session_state.wiz_respuesta = nombre_resp; st.session_state.wiz_paso = 4; st.rerun()

# PASO 4
elif st.session_state.wiz_paso == 4:
    st.subheader("📋 Paso 4: Opciones del Diseño")
    col1, col2 = st.columns(2)
    with col1:
        n_replicas = st.number_input("**Réplicas:**", min_value=1, max_value=10, value=1, step=1, key="wiz_replicas")
        puntos_centro = st.number_input("**Puntos Centrales:**", min_value=0, max_value=10, value=0, step=1, key="wiz_puntos_centro")
    with col2:
        randomizar = st.checkbox("**Randomizar el orden de corrida**", value=True, key="wiz_randomizar")
    if st.session_state.wiz_tipo_diseno in ("screening", "screening_fractional"):
        st.markdown("### 🔬 Configuración de la fracción 2^(k-p)")
        k_actual = len(st.session_state.wiz_factores)
        # Para p generadores independientes se requieren al menos p factores básicos:
        # p <= k-p. Así, p=4 solo es admisible desde k=8.
        p_max = min(4, k_actual // 2)
        if p_max < 1:
            st.warning("Define al menos dos factores adicionales en el paso anterior.")
            p_fraccion = 1
        else:
            p_fraccion = st.selectbox("Orden de fracción (p)", list(range(1, p_max + 1)),
                                      format_func=lambda p: f"2^(k-{p}) — {2**(k_actual-p)} corridas base",
                                      key="wiz_p_fraccion")
        sugeridos, res_sugerida = sugerir_generadores_fraccion(k_actual, p_fraccion)
        texto_sugerido = "\n".join(sugeridos)
        key_generadores = f"wiz_generadores_{k_actual}_{p_fraccion}"
        if key_generadores in st.session_state:
            st.session_state[key_generadores] = str(st.session_state[key_generadores]).replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
        generadores_txt = st.text_area("Generadores (un generador por línea)", value=texto_sugerido,
                                       help="Se requieren exactamente p generadores independientes. Ejemplo: para p=2 escribe C=AB y D=AC, cada uno en una línea.",
                                       key=key_generadores)
        st.info(f"Generadores sugeridos para resolución {res_sugerida}. La resolución definitiva se calculará al generar el diseño.")
        st.caption("La fracción reduce el número de corridas, pero introduce aliasado. Revisa la estructura de alias antes de ejecutar el experimento.")
    elif st.session_state.wiz_tipo_diseno == "screening_full":
        st.markdown("### 🔍 Cribado factorial completo 2^k")
        k_completo = len(st.session_state.wiz_factores)
        st.info(f"Se generarán las {2**k_completo} combinaciones completas de los {k_completo} factores, antes de réplicas y puntos centrales. No se aplicará fraccionamiento ni aliasado estructural.")
        p_fraccion, generadores_txt = 0, ""
    else:
        p_fraccion, generadores_txt = 0, ""
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("⬅️ Atrás", key="wiz_back_4"): st.session_state.wiz_paso = 3; st.rerun()
    with col_btn2:
        if st.button("🎲 Generar Diseño", key="wiz_generar"):
            st.session_state.wiz_config = {'n_replicas': n_replicas, 'puntos_centro': puntos_centro, 'randomizar': randomizar, 'p_fraccion': p_fraccion, 'generadores_txt': generadores_txt}
            st.session_state.wiz_datos = None; st.session_state.wiz_paso = 5; st.rerun()

# PASO 5
elif st.session_state.wiz_paso == 5:
    st.subheader("📋 Paso 5: Tabla del Diseño")
    if st.session_state.wiz_datos is None:
        factores, resp, config, tipo = st.session_state.wiz_factores, st.session_state.wiz_respuesta, st.session_state.wiz_config, st.session_state.wiz_tipo_diseno
        try:
            if tipo in ('screening', 'screening_fractional'):
                df_diseno, meta_fr = generar_diseno_fraccionado(
                    factores, resp, config.get('p_fraccion', 1), config.get('generadores_txt', ''),
                    replicas=config['n_replicas'], puntos_centro=config['puntos_centro'],
                    randomizar=config['randomizar'])
                st.session_state.wiz_fr_meta = meta_fr
            elif tipo == 'screening_full':
                if any(len(v.get('niveles', [])) != 2 for v in factores.values()):
                    raise ValueError("El cribado factorial completo 2^k requiere exactamente dos niveles por factor.")
                df_diseno = crear_diseno_completo(
                    factores, {resp: {'valores': None}},
                    n_replicas=config['n_replicas'], randomize=config['randomizar']
                )
                st.session_state.wiz_fr_meta = None
            elif tipo in ['multilevel', 'rsm']:
                df_diseno = crear_diseno_completo(factores, {resp: {'valores': None}}, n_replicas=config['n_replicas'], randomize=config['randomizar'])
            elif tipo in ['single_factor', 'single_factor_block']:
                factor_principal = [f for f, v in factores.items() if not v.get('es_bloque', False)]
                if factor_principal:
                    f = factor_principal[0]; niveles_f = factores[f]['niveles']; n_bloque, niveles_bloque = None, None
                    if tipo == 'single_factor_block':
                        factores_bloque = [f for f, v in factores.items() if v.get('es_bloque', False)]
                        if factores_bloque: n_bloque, niveles_bloque = factores_bloque[0], factores[factores_bloque[0]]['niveles']
                    filas = []
                    for nivel in niveles_f:
                        if n_bloque:
                            for blq in niveles_bloque: filas.append({f: nivel, n_bloque: blq, resp: None})
                        else:
                            for _ in range(config['n_replicas']): filas.append({f: nivel, resp: None})
                    df_diseno = pd.DataFrame(filas)
                    if config['randomizar']: df_diseno = df_diseno.sample(frac=1, random_state=42).reset_index(drop=True)
            elif tipo in ['latin_square', 'greco_latin']:
                factor_principal = [f for f, v in factores.items() if not v.get('es_bloque', False)]
                if factor_principal:
                    f = factor_principal[0]; niveles_f = factores[f]['niveles']; n = len(niveles_f); filas = []
                    for i in range(n):
                        for j in range(n):
                            fila = {'Fila': f'F{i+1}', 'Columna': f'C{j+1}', f: niveles_f[(i + j) % n], resp: None}
                            if tipo == 'greco_latin': fila['Greco'] = chr(65 + (i + 2*j) % n)
                            filas.append(fila)
                    df_diseno = pd.DataFrame(filas)
            else:
                df_diseno = crear_diseno_completo(factores, {resp: {'valores': None}}, n_replicas=config['n_replicas'], randomize=config['randomizar'])
            if config.get('puntos_centro', 0) > 0 and tipo not in ('screening', 'screening_fractional'):
                factores_centro = {f: v for f, v in factores.items() if not v.get('es_bloque', False)}
                fila_centro = {}
                for f, v in factores_centro.items():
                    if v.get('tipo') != 'Continuo' or len(v.get('niveles', [])) != 2:
                        raise ValueError("Los puntos centrales requieren factores continuos con dos niveles numéricos.")
                    try:
                        bajo, alto = map(float, v['niveles'][:2])
                    except Exception:
                        raise ValueError(f"Los niveles de {f} deben ser numéricos para calcular puntos centrales.")
                    fila_centro[f] = (bajo + alto) / 2
                fila_centro[resp] = np.nan
                centros = pd.DataFrame([fila_centro] * int(config['puntos_centro']))
                df_diseno = pd.concat([df_diseno, centros], ignore_index=True)
            st.session_state.wiz_datos = df_diseno
        except Exception as e: st.error(f"❌ Error al generar el diseño: {str(e)}")

    if st.session_state.wiz_datos is not None:
        df_diseno = st.session_state.wiz_datos
        st.success(f"✅ Diseño generado: {len(df_diseno)} corridas")
        tipo_actual = st.session_state.wiz_tipo_diseno
        if tipo_actual in ("screening", "screening_fractional") and st.session_state.get("wiz_fr_meta"):
            meta_fr = st.session_state.wiz_fr_meta
            st.info(f"Diseño 2^({meta_fr['k']}-{meta_fr['p']}) · {meta_fr['n_base']} corridas base · Resolución {meta_fr['resolucion']} · {meta_fr['p']} generadores: " + "; ".join(f"{a}={b}" for a,b in meta_fr['generadores'].items()))
        elif tipo_actual == "screening_full":
            k_completo = len(st.session_state.wiz_factores)
            st.info(f"Diseño factorial completo 2^{k_completo} · {2**k_completo} combinaciones base · Sin fraccionamiento.")
        st.dataframe(df_diseno, use_container_width=True)
        st.download_button("📥 Descargar Diseño (vacío)", data=df_diseno.to_csv(index=False).encode('utf-8'), file_name="diseno_vacio.csv", mime="text/csv")
        st.subheader("✏️ Editar datos manualmente")
        df_editado = st.data_editor(df_diseno, num_rows="fixed", key="wiz_editor")

        st.markdown("---")
        st.subheader("📁 O cargar datos completos desde CSV / URL")
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            archivo = st.file_uploader("Sube tu CSV:", type=['csv', 'txt'], key="wiz_upload_csv")
            if archivo is not None:
                df_subido = cargar_datos_desde_csv(archivo)
                if df_subido is not None:
                    if st.button("✅ Usar CSV", key="wiz_usar_csv"):
                        st.session_state.wiz_datos = df_subido
                        st.session_state.wiz_paso = 6
                        st.rerun()
        with col_up2:
            url = st.text_input("O ingresa una URL:", key="wiz_upload_url")
            if url and st.button("✅ Usar URL", key="wiz_usar_url"):
                df_subido = cargar_datos_desde_url(url)
                if df_subido is not None:
                    st.session_state.wiz_datos = df_subido
                    st.session_state.wiz_paso = 6
                    st.rerun()

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("⬅️ Atrás", key="wiz_back_5"): st.session_state.wiz_paso = 4; st.rerun()
        with col_btn2:
            if st.button("➡️ Analizar Datos", key="wiz_analizar"):
                st.session_state.wiz_datos = df_editado; st.session_state.wiz_paso = 6; st.rerun()

# PASO 6
elif st.session_state.wiz_paso == 6:
    st.subheader("📋 Paso 6: Configuración del Análisis")
    df_diseno = st.session_state.wiz_datos
    resp = st.session_state.wiz_respuesta
    if df_diseno is None or len(df_diseno) == 0:
        st.error("❌ No hay datos. Vuelve al paso anterior.")
        if st.button("⬅️ Volver"): st.session_state.wiz_paso = 5; st.rerun()
    elif df_diseno[resp].isnull().all():
        st.warning("⚠️ La columna de respuesta está vacía.")
        if st.button("⬅️ Volver"): st.session_state.wiz_paso = 5; st.rerun()
    else:
        tipo_anova = selector_tipo_anova("wizard")
        st.session_state.wiz_tipo_anova = tipo_anova
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("⬅️ Atrás", key="wiz_back_6"): st.session_state.wiz_paso = 5; st.rerun()
        with col_btn2:
            if st.button("🚀 Ejecutar Análisis", key="wiz_ejecutar"): st.session_state.wiz_paso = 7; st.rerun()

# PASO 7 (CON PARETO, DANIEL Y MODELO REDUCIDO)
elif st.session_state.wiz_paso == 7:
    st.subheader("📊 Paso 7: Análisis y Resultados")
    df_diseno, resp, tipo, factores_def = st.session_state.wiz_datos, st.session_state.wiz_respuesta, st.session_state.wiz_tipo_diseno, st.session_state.wiz_factores
    factores_reales = [f for f in factores_def.keys() if f in df_diseno.columns]
    factor_principal = [f for f in factores_reales if not factores_def[f].get('es_bloque', False)]
    df_analisis = df_diseno.copy().reset_index(drop=True)
    try:
        tipo_anova = st.session_state.get('wiz_tipo_anova', 1)
        if tipo in ('screening', 'screening_fractional'):
            meta_fr = st.session_state.get('wiz_fr_meta')
            if not meta_fr:
                st.error("No se encontró la configuración del diseño fraccionado. Regresa al paso 4 y genera nuevamente el diseño.")
            else:
                alpha_fr = st.slider("Nivel de significancia α", 0.01, 0.10, 0.05, 0.01, key="fr_alpha")
                mostrar_analisis_fraccionado(df_analisis, resp, factores_def, meta_fr, alpha_fr)
        elif tipo in ['multilevel', 'rsm', 'screening_full']:
            for f in factores_reales: df_analisis[f] = df_analisis[f].astype('category')
            if tipo == 'rsm':
                vars_list = factor_principal
                formula = f'{resp} ~ ' + ' + '.join(vars_list) + ' + ' + ' + '.join([f'I({v}**2)' for v in vars_list])
                if len(vars_list) >= 2: formula += ' + ' + ' + '.join([f'{vars_list[i]}:{vars_list[j]}' for i in range(len(vars_list)) for j in range(i+1, len(vars_list))])
            elif tipo == 'screening_full':
                formula = f'{resp} ~ ' + ' * '.join([f'C({v})' for v in factor_principal])
            else:
                formula = f'{resp} ~ ' + ' * '.join(factor_principal)
            modelo, anova = analizar_anova_mejorada(df_analisis, formula, "ANOVA Wizard", "wizard", "wizard", tipo_anova)

            if modelo is not None:
                mostrar_tabla_anova("wizard")
                mostrar_summary_statistics(modelo, df_analisis, factor_principal, resp, "wizard", tipo_anova)
                fig_sc = graficar_sumas_cuadrados(anova)
                if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)

                # ⭐ NUEVO: Gráficos de Efectos e Interacciones + Pareto/Daniel
                graficar_efectos_e_interacciones(df_analisis, factor_principal, resp)
                mostrar_pareto_y_daniel(modelo, anova, resp, len(df_analisis))
                ui_excluir_efectos(df_analisis, resp, anova, tipo_anova, "wizard")

                # ⭐ MOSTRAR MEJOR ANOVA
                if 'wizard_red' in st.session_state.resultados_anova:
                    st.markdown("### 🌟 Mejor ANOVA (Modelo Reducido)")
                    mostrar_tabla_anova("wizard_red")

                mostrar_graficos_residuos_directo(modelo, df_analisis, factor_principal[0], resp, "wizard", "Gráficos de Residuos")
                if len(factor_principal) >= 1:
                    st.subheader("📊 Boxplots por Factor")
                    fig_box = graficar_boxplot_por_factor(df_analisis, factor_principal, resp)
                    if fig_box: st.plotly_chart(fig_box, use_container_width=True)
                if len(factor_principal) >= 2:
                    st.subheader("📊 Gráficos de Interacción")
                    fig_int = graficos_interaccion_multifactorial(df_analisis, factor_principal, resp, max_pares=4)
                    if fig_int: st.plotly_chart(fig_int, use_container_width=True)
                    fig_heat = graficos_interaccion_heatmap(df_analisis, factor_principal, resp)
                    if fig_heat: st.plotly_chart(fig_heat, use_container_width=True)
                st.subheader("🔬 Pruebas Múltiples")
                mse_c, gl_c = extraer_mse_gl(anova)
                for factor in factor_principal:
                    n_niveles = len(df_analisis[factor].unique())
                    if n_niveles > 2:
                        st.markdown(f"#### 📌 {factor} ({n_niveles} niveles)")
                        with st.expander(f"Tukey HSD para {factor}", expanded=False):
                            tukey = realizar_tukey(df_analisis, factor, resp)
                            if tukey is not None: st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
                        with st.expander(f"LSD para {factor}", expanded=False):
                            lsd_df = realizar_lsd(df_analisis, factor, resp, mse_c, gl_c)
                            if lsd_df is not None: st.dataframe(lsd_df)
                    else:
                        st.markdown(f"#### 📌 {factor} (2 niveles)")
                        with st.expander(f"LSD para {factor}", expanded=False):
                            lsd_df = realizar_lsd(df_analisis, factor, resp, mse_c, gl_c)
                            if lsd_df is not None: st.dataframe(lsd_df)
                st.subheader("📝 Conclusiones del Análisis")
                for c in generar_conclusiones_factorial(anova, modelo, df_analisis, factor_principal, resp):
                    if c.startswith("✅"): st.success(c)
                    elif c.startswith("❌"): st.error(c)
                    else: st.info(c)
        elif tipo == 'single_factor':
            f = factor_principal[0]
            df_analisis[f] = df_analisis[f].astype('category')
            modelo, anova = analizar_anova_mejorada(df_analisis, f'{resp} ~ C({f})', "DCA Wizard", "wizard", "wizard", tipo_anova)
            if modelo is not None:
                mostrar_tabla_anova("wizard")
                mostrar_summary_statistics(modelo, df_analisis, [f], resp, "wizard")
                fig_sc = graficar_sumas_cuadrados(anova)
                if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                mostrar_graficos_residuos_directo(modelo, df_analisis, f, resp, "wizard", "Gráficos de Residuos")
                fig = px.box(df_analisis, x=f, y=resp, color=f); fig.update_layout(showlegend=False, height=350); st.plotly_chart(fig, use_container_width=True)
                medias = df_analisis.groupby(f)[resp].mean().reset_index()
                fig2 = px.bar(medias, x=f, y=resp, color=f, text_auto=True); fig2.update_layout(showlegend=False, height=300); st.plotly_chart(fig2, use_container_width=True)
                st.subheader("🔬 Pruebas Múltiples")
                mse_c, gl_c = extraer_mse_gl(anova)
                with st.expander("Tukey HSD", expanded=False):
                    tukey = realizar_tukey(df_analisis, f, resp)
                    if tukey is not None: st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
                with st.expander("LSD", expanded=False):
                    lsd_df = realizar_lsd(df_analisis, f, resp, mse_c, gl_c)
                    if lsd_df is not None: st.dataframe(lsd_df)
                st.subheader("📝 Conclusiones del Análisis")
                for c in generar_conclusiones_anova(anova, modelo, df_analisis, f, resp)[0]:
                    if c.startswith("✅") or c.startswith("🏆") or c.startswith("📈"): st.success(c)
                    elif c.startswith("❌"): st.error(c)
                    else: st.info(c)
        elif tipo == 'single_factor_block':
            f = factor_principal[0]; factor_bloque = [f for f in factores_reales if factores_def[f].get('es_bloque', False)]
            df_analisis[f] = df_analisis[f].astype('category')
            if factor_bloque:
                blq = factor_bloque[0]; df_analisis[blq] = df_analisis[blq].astype('category'); formula = f'{resp} ~ C({f}) + C({blq})'
            else: formula = f'{resp} ~ C({f})'
            modelo, anova = analizar_anova_mejorada(df_analisis, formula, "DBCA Wizard", "wizard", "wizard", tipo_anova)
            if modelo is not None:
                mostrar_tabla_anova("wizard")
                mostrar_summary_statistics(modelo, df_analisis, [f] + (factor_bloque if factor_bloque else []), resp, "wizard")
                fig_sc = graficar_sumas_cuadrados(anova)
                if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                mostrar_graficos_residuos_directo(modelo, df_analisis, f, resp, "wizard", "Gráficos de Residuos")
                fig = px.box(df_analisis, x=f, y=resp, color=f); fig.update_layout(showlegend=False, height=350); st.plotly_chart(fig, use_container_width=True)
                if factor_bloque:
                    fig_blq = px.box(df_analisis, x=blq, y=resp, color=blq); fig_blq.update_layout(showlegend=False, height=350); st.plotly_chart(fig_blq, use_container_width=True)
                st.subheader("🔬 Pruebas Múltiples")
                mse_c, gl_c = extraer_mse_gl(anova)
                with st.expander("Tukey HSD", expanded=False):
                    tukey = realizar_tukey(df_analisis, f, resp)
                    if tukey is not None: st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
                with st.expander("LSD", expanded=False):
                    lsd_df = realizar_lsd(df_analisis, f, resp, mse_c, gl_c)
                    if lsd_df is not None: st.dataframe(lsd_df)
                st.subheader("📝 Conclusiones del Análisis")
                for c in generar_conclusiones_anova(anova, modelo, df_analisis, f, resp)[0]:
                    if c.startswith("✅") or c.startswith("🏆") or c.startswith("📈"): st.success(c)
                    elif c.startswith("❌"): st.error(c)
                    else: st.info(c)
        elif tipo in ['latin_square', 'greco_latin']:
            f = factor_principal[0]
            for col in ['Fila', 'Columna', f]:
                if col in df_analisis.columns: df_analisis[col] = df_analisis[col].astype('category')
            if tipo == 'latin_square': formula = f'{resp} ~ C(Fila) + C(Columna) + C({f})'
            else:
                if 'Greco' in df_analisis.columns:
                    df_analisis['Greco'] = df_analisis['Greco'].astype('category')
                    formula = f'{resp} ~ C(Fila) + C(Columna) + C({f}) + C(Greco)'
                else: formula = f'{resp} ~ C(Fila) + C(Columna) + C({f})'
            modelo, anova = analizar_anova_mejorada(df_analisis, formula, "CL/GL Wizard", "wizard", "wizard", tipo_anova)
            if modelo is not None:
                mostrar_tabla_anova("wizard")
                factores_ss = ['Fila', 'Columna', f] + (['Greco'] if tipo == 'greco_latin' and 'Greco' in df_analisis.columns else [])
                mostrar_summary_statistics(modelo, df_analisis, factores_ss, resp, "wizard")
                fig_sc = graficar_sumas_cuadrados(anova)
                if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                mostrar_graficos_residuos_directo(modelo, df_analisis, f, resp, "wizard", "Gráficos de Residuos")
                fig = px.box(df_analisis, x=f, y=resp, color=f); fig.update_layout(showlegend=False, height=350); st.plotly_chart(fig, use_container_width=True)
                if 'Fila' in df_analisis.columns and 'Columna' in df_analisis.columns:
                    pivot = df_analisis.pivot_table(index='Fila', columns='Columna', values=resp, aggfunc='mean')
                    fig_heat = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='Blues'); st.plotly_chart(fig_heat, use_container_width=True)
                st.subheader("🔬 Pruebas Múltiples")
                mse_c, gl_c = extraer_mse_gl(anova)
                with st.expander(f"Tukey HSD para {f}", expanded=False):
                    tukey = realizar_tukey(df_analisis, f, resp)
                    if tukey is not None: st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
                with st.expander(f"LSD para {f}", expanded=False):
                    lsd_df = realizar_lsd(df_analisis, f, resp, mse_c, gl_c)
                    if lsd_df is not None: st.dataframe(lsd_df)
                if tipo == 'greco_latin' and 'Greco' in df_analisis.columns:
                    with st.expander("Tukey HSD para Greco", expanded=False):
                        tukey_g = realizar_tukey(df_analisis, 'Greco', resp)
                        if tukey_g is not None: st.dataframe(pd.DataFrame(data=tukey_g.summary().data[1:], columns=tukey_g.summary().data[0]))
                    with st.expander("LSD para Greco", expanded=False):
                        lsd_g = realizar_lsd(df_analisis, 'Greco', resp, mse_c, gl_c)
                        if lsd_g is not None: st.dataframe(lsd_g)
                st.subheader("📝 Conclusiones del Análisis")
                for c in generar_conclusiones_anova(anova, modelo, df_analisis, f, resp)[0]:
                    if c.startswith("✅") or c.startswith("🏆") or c.startswith("📈"): st.success(c)
                    elif c.startswith("❌"): st.error(c)
                    else: st.info(c)
    except Exception as e:
        st.error(f"❌ Error durante el análisis: {str(e)}")
        st.exception(e)
    st.markdown("---")
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("⬅️ Volver al Paso 6", key="wiz_back_7"): st.session_state.wiz_paso = 6; st.rerun()
    with col_btn2:
        if st.button("🔄 Nuevo Diseño", key="wiz_nuevo"):
            st.session_state.wiz_paso = 1; st.session_state.wiz_tipo_diseno = None; st.session_state.wiz_factores = {}
            st.session_state.wiz_respuesta = None; st.session_state.wiz_datos = None; st.session_state.wiz_config = {}; st.session_state.wiz_fr_meta = None
            st.rerun()

# ============================================================
