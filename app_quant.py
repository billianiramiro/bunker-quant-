import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import urllib.request
import xml.etree.ElementTree as ET

# Descargar el diccionario para la IA
nltk.download('vader_lexicon', quiet=True)

# ==========================================
# CONFIGURACIÓN DE LA APP
# ==========================================
st.set_page_config(page_title="Bunker Quant", layout="wide")

# Estética general de la App (Fondo oscuro profesional)
st.markdown("""
    <style>
    .main { background-color: #131722; }
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 10px; border: 1px solid #2a2e39; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Tu Bunker Cuantitativo")
st.markdown("Plataforma integral: Proyección estadística, Análisis Técnico, Radiografía Fundamental y Sentimiento con IA.")

tab1, tab2, tab3 = st.tabs(["📈 Predicción y Análisis Técnico", "🏥 Radiografía Fundamental", "🧠 Sentimiento de Mercado"])

# ==========================================
# PESTAÑA 1: MONTE CARLO Y ANALISTA TÉCNICO
# ==========================================
with tab1:
    st.header("Simulador y Contexto de Mercado")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ticker_mc = st.text_input("Ticker de la Acción", value="GOOGL").upper()
    with col_b:
        meses_proyeccion = st.slider("Meses a Proyectar", 1, 60, 24)
    with col_c:
        simulaciones = st.slider("Escenarios", 1000, 10000, 5000)

    if st.button("🚀 Ejecutar Análisis"):
        with st.spinner('Procesando algoritmos de predicción y análisis técnico...'):
            try:
                df = yf.download(ticker_mc, period='5y', progress=False)
                close_prices = df['Close'].squeeze().dropna().values
                S0 = close_prices[-1]
                
                # --- LÓGICA MONTE CARLO ---
                log_returns = np.log(close_prices[1:] / close_prices[:-1])
                drift = np.mean(log_returns) - (0.5 * np.var(log_returns))
                stdev = np.std(log_returns)
                
                dias_totales = meses_proyeccion * 21
                Z = np.random.normal(0, 1, (dias_totales, simulaciones))
                daily_returns = np.exp(drift + stdev * Z)
                price_paths = S0 * np.vstack([np.ones(simulaciones), np.cumprod(daily_returns, axis=0)])
                
                p50 = np.percentile(price_paths, 50, axis=1)
                p5 = np.percentile(price_paths, 5, axis=1)
                p95 = np.percentile(price_paths, 95, axis=1)
                
                # --- DISEÑO ESTILO TRADINGVIEW ---
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.set_facecolor('#131722')
                fig.patch.set_facecolor('#131722')
                ax.grid(True, color='#2a2e39', linestyle='-', alpha=0.3)
                
                ax.plot(price_paths[:, :50], color='#2a2e39', alpha=0.15) 
                ax.plot(p50, label='Precio Esperado (P50)', color='#3179f5', linewidth=3)
                ax.plot(p95, label='Optimista (P95)', color='#00ff88', linestyle='--', linewidth=1.5)
                ax.plot(p5, label='Pesimista (P5)', color='#ff3a33', linestyle='--', linewidth=1.5)
                ax.axhline(S0, color='white', linestyle=':', alpha=0.6, label=f'Actual: ${S0:.2f}')
                
                ax.spines['bottom'].set_color('#2a2e39')
                ax.spines['left'].set_color('#2a2e39')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                ax.set_title(f'PROYECCIÓN ESTRATÉGICA: {ticker_mc}', color='white', fontsize=14, pad=20)
                ax.legend(facecolor='#131722', edgecolor='#2a2e39')
                st.pyplot(fig)
                
                # --- MÉTRICAS DE RESULTADO ---
                st.subheader("Objetivos Proyectados")
                m1, m2, m3 = st.columns(3)
                m1.metric("Escenario Bajo", f"${p5[-1]:,.2f}", delta=f"{((p5[-1]/S0)-1)*100:.1f}%", delta_color="inverse")
                m2.metric("Escenario Base", f"${p50[-1]:,.2f}", delta=f"{((p50[-1]/S0)-1)*100:.1f}%")
                m3.metric("Escenario Alto", f"${p95[-1]:,.2f}", delta=f"{((p95[-1]/S0)-1)*100:.1f}%")
                
                # =========================================================
                # EL ANALISTA ALGORTÍMICO (Contexto de Mercado)
                # =========================================================
                st.divider()
                st.subheader("📋 Contexto de Mercado (El Analista Algorítmico)")
                
                # Cálculos de pisos, techos y medias móviles
                historial_6m = df['Close'][-126:]
                soporte_6m = historial_6m.min()
                resistencia_6m = historial_6m.max()
                sma_50 = df['Close'][-50:].mean()
                sma_200 = df['Close'][-200:].mean()
                
                # Cálculo de Volatilidad reciente (últimos 30 días)
                retornos_30d = np.log(df['Close'][-30:] / df['Close'][-30:].shift(1)).dropna()
                vol_30d = retornos_30d.std() * np.sqrt(252) * 100
                
                # 1. Lógica de Tendencia
                if S0 > sma_50 and sma_50 > sma_200:
                    st.success("🔥 **FUERTE ALCISTA:** El mercado está pagando cada vez más por este activo. La acción cotiza por encima de sus promedios de corto y largo plazo, indicando gran confianza e interés comprador en los últimos meses.")
                elif S0 > sma_50 and sma_50 < sma_200:
                    st.info("📈 **RECUPERACIÓN:** El activo viene de una mala racha histórica, pero en las últimas semanas la gente ha comenzado a acumularlo nuevamente. Está intentando revertir su tendencia a positiva.")
                elif S0 < sma_50 and sma_50 > sma_200:
                    st.warning("⚠️ **CORRECCIÓN:** Históricamente venía muy bien, pero recientemente los inversores han estado tomando ganancias (vendiendo). El sentimiento a corto plazo se ha enfriado y perdió su promedio de 50 días.")
                else:
                    st.error("❄️ **FUERTE BAJISTA:** El sentimiento actual es muy negativo. El mercado se ha estado deshaciendo de este activo consistentemente y cotiza por debajo de todos sus promedios importantes.")

                # 2. Lógica de Niveles (Pisos y Techos)
                dist_soporte = ((S0 - soporte_6m) / S0) * 100
                dist_resistencia = ((resistencia_6m - S0) / S0) * 100
                
                if dist_resistencia < 4:
                    st.write(f"🧱 **ZONA DE TECHO:** El precio actual (${S0:.2f}) está **muy cerca de su resistencia máxima de los últimos 6 meses** (${resistencia_6m:.2f}). Atención: el mercado suele dudar en comprar aquí por miedo a un rebote a la baja.")
                elif dist_soporte < 4:
                    st.write(f"🛏️ **ZONA DE PISO:** El precio (${S0:.2f}) está **apoyado sobre su soporte clave de 6 meses** (${soporte_6m