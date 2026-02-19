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

# Estética general de la App
st.markdown("""
    <style>
    .main { background-color: #131722; }
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 10px; border: 1px solid #2a2e39; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Tu Bunker Cuantitativo")
st.markdown("Plataforma integral: Proyección estadística, Radiografía Fundamental y Sentimiento con IA.")

tab1, tab2, tab3 = st.tabs(["📈 Predicción (Estilo TV)", "🏥 Radiografía Fundamental", "🧠 Sentimiento de Mercado"])

# ==========================================
# PESTAÑA 1: MONTE CARLO (ESTILO TRADINGVIEW)
# ==========================================
with tab1:
    st.header("Simulador de Precios Futuros")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ticker_mc = st.text_input("Ticker de la Acción", value="GOOGL").upper()
    with col_b:
        meses_proyeccion = st.slider("Meses a Proyectar", 1, 60, 24)
    with col_c:
        simulaciones = st.slider("Escenarios", 1000, 10000, 5000)

    if st.button("🚀 Ejecutar Análisis"):
        with st.spinner('Procesando algoritmos...'):
            try:
                df = yf.download(ticker_mc, period='5y', progress=False)
                close_prices = df['Close'].squeeze().dropna().values
                S0 = close_prices[-1]
                
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
                
                # Fondo y Rejilla
                ax.set_facecolor('#131722')
                fig.patch.set_facecolor('#131722')
                ax.grid(True, color='#2a2e39', linestyle='-', alpha=0.3)
                
                # Dibujar Caminos y Líneas Maestras
                ax.plot(price_paths[:, :50], color='#2a2e39', alpha=0.15) 
                ax.plot(p50, label='Precio Esperado (P50)', color='#3179f5', linewidth=3)
                ax.plot(p95, label='Optimista (P95)', color='#00ff88', linestyle='--', linewidth=1.5)
                ax.plot(p5, label='Pesimista (P5)', color='#ff3a33', linestyle='--', linewidth=1.5)
                ax.axhline(S0, color='white', linestyle=':', alpha=0.6, label=f'Actual: ${S0:.2f}')
                
                # Bordes y Títulos
                ax.spines['bottom'].set_color('#2a2e39')
                ax.spines['left'].set_color('#2a2e39')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                ax.set_title(f'PROYECCIÓN ESTRATÉGICA: {ticker_mc}', color='white', fontsize=14, pad=20)
                ax.legend(facecolor='#131722', edgecolor='#2a2e39')
                
                st.pyplot(fig)
                
                # Métricas con estilo
                st.subheader("Objetivos Proyectados")
                m1, m2, m3 = st.columns(3)
                m1.metric("Escenario Bajo", f"${p5[-1]:,.2f}", delta=f"{((p5[-1]/S0)-1)*100:.1f}%", delta_color="inverse")
                m2.metric("Escenario Base", f"${p50[-1]:,.2f}", delta=f"{((p50[-1]/S0)-1)*100:.1f}%")
                m3.metric("Escenario Alto", f"${p95[-1]:,.2f}", delta=f"{((p95[-1]/S0)-1)*100:.1f}%")
                
            except Exception as e:
                st.error(f"Error técnico: {e}")

# ==========================================
# PESTAÑA 2: RADIOGRAFÍA FUNDAMENTAL
# ==========================================
with tab2:
    st.header("Radiografía Fundamental")
    tickers_input = st.text_input("Tickers (separados por comas)", value="AAPL, GOOGL, KO, MELI, SPY")
    
    if st.button("📊 Generar Reporte"):
        lista_tickers = [t.strip().upper() for t in tickers_input.split(',')]
        with st.spinner('Analizando balances...'):
            resultados = []
            for ticker in lista_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    def fmt(val, p=False, m=False):
                        if val is None or val == "N/A": return "N/A"
                        if p: return f"{val * 100:.2f}%"
                        if m: return f"${val:,.2f}"
                        return f"{val:.2f}"

                    peg = info.get('trailingPegRatio', info.get('pegRatio'))
                    
                    resultados.append({
                        "Ticker": ticker,
                        "Precio": fmt(info.get('currentPrice', info.get('previousClose')), m=True),
                        "P/E Ratio": fmt(info.get('trailingPE')),
                        "PEG Ratio": f"{peg:.2f}" if peg else "N/A",
                        "P/B Ratio": fmt(info.get('priceToBook')),
                        "Margen Neto": fmt(info.get('profitMargins'), p=True),
                        "ROE": fmt(info.get('returnOnEquity'), p=True),
                        "Deuda/Cap": fmt(info.get('debtToEquity')),
                        "Div. Yield": fmt(info.get('dividendYield'), p=True)
                    })
                except: pass
            
            if resultados:
                st.dataframe(pd.DataFrame(resultados), use_container_width=True)
            else:
                st.warning("No hay datos disponibles.")

    with st.expander("📚 Ayuda de Métricas"):
        st.write("PEG < 1: Crecimiento barato. ROE > 15%: Gestión eficiente. Margen > 20%: Ventaja competitiva.")

# ==========================================
# PESTAÑA 3: SENTIMIENTO DE MERCADO (RSS)
# ==========================================
with tab3:
    st.header("Análisis de Sentimiento (IA)")
    ticker_news = st.text_input("Ticker para noticias", value="AAPL").upper()
    
    if st.button("📰 Escanear Noticias"):
        with st.spinner('Leyendo titulares RSS...'):
            try:
                url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_news}&region=US&lang=en-US"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                xml_data = urllib.request.urlopen(req).read()
                items = ET.fromstring(xml_data).findall('./channel/item')
                
                if not items:
                    st.warning("Sin noticias recientes.")
                else:
                    sia = SentimentIntensityAnalyzer()
                    rows = []
                    total_score = 0
                    
                    for item in items:
                        tit = item.find('title').text
                        date = item.find('pubDate').text.replace(" +0000", "")
                        score = sia.polarity_scores(tit)['compound']
                        total_score += score
                        
                        impacto = "🟢 Positivo" if score > 0.15 else "🔴 Negativo" if score < -0.15 else "⚪ Neutral"
                        rows.append({"Fecha": date, "Impacto": impacto, "Titular": tit, "Score": round(score, 3)})
                    
                    # Dashboard de Sentimiento
                    avg = ( (total_score / len(rows)) + 1 ) * 50
                    st.subheader(f"Temperatura: {avg:.1f}/100")
                    st.progress(int(avg))
                    
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e:
                st.error(f"Error en feed: {e}")