import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET

# Descargar el diccionario para la IA
nltk.download('vader_lexicon', quiet=True)

# ==========================================
# CONFIGURACIÓN DE LA APP
# ==========================================
st.set_page_config(page_title="Bunker Quant", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #131722; }
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 10px; border: 1px solid #2a2e39; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# BARRA LATERAL: BUSCADOR DE TICKERS
# ==========================================
with st.sidebar:
    st.header("🔍 Buscador de Tickers")
    st.markdown("¿No sabes el símbolo oficial? Escribe el nombre de la empresa:")
    busqueda = st.text_input("Ej: Mercado Libre, Banco Frances, SPY")
    
    if st.button("🔎 Buscar"):
        if busqueda:
            with st.spinner("Buscando en Wall Street..."):
                try:
                    # Conexión directa a la API de búsqueda de Yahoo Finance
                    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(busqueda)}&quotesCount=5"
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    respuesta = urllib.request.urlopen(req)
                    datos = json.loads(respuesta.read())
                    
                    if 'quotes' in datos and len(datos['quotes']) > 0:
                        st.markdown("### Resultados:")
                        for q in datos['quotes']:
                            ticker_res = q.get('symbol', 'N/A')
                            nombre = q.get('shortname', q.get('longname', 'Desconocido'))
                            tipo = q.get('quoteType', 'Activo')
                            bolsa = q.get('exchange', '')
                            # Agregamos la bolsa para diferenciar fácilmente Argentina de EE.UU.
                            st.success(f"**{ticker_res}** ({nombre}) - {tipo} | {bolsa}")
                    else:
                        st.warning("No se encontraron resultados.")
                except Exception as e:
                    st.error("Error al buscar. Intenta de nuevo.")

# Título Principal
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
                # 1. Obtener el nombre oficial de la empresa primero
                stock_data = yf.Ticker(ticker_mc)
                nombre_empresa = stock_data.info.get('longName', stock_data.info.get('shortName', ticker_mc))
                
                # Imprimir el nombre de la empresa como un título destacado Neón
                st.markdown(f"<h2 style='text-align: center; color: #3179f5; border-bottom: 1px solid #2a2e39; padding-bottom: 10px;'>🏢 {nombre_empresa} ({ticker_mc})</h2>", unsafe_allow_html=True)

                # 2. Descargar los datos históricos
                df = yf.download(ticker_mc, period='5y', progress=False)
                
                # MATRIZ LIMPIA
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
                
                # =========================================================
                # DISEÑO: ESTILO TRADINGVIEW / TECNOLÓGICO
                # =========================================================
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.set_facecolor('#131722')
                fig.patch.set_facecolor('#131722')
                ax.grid(True, color='#2a2e39', linestyle='-', alpha=0.4)
                
                ax.plot(price_paths[:, :30], color='#2a2e39', alpha=0.3, linewidth=0.8) 
                ax.fill_between(range(len(p50)), p5, p95, color='#3179f5', alpha=0.08)
                
                ax.plot(p50, color='#3179f5', linewidth=6, alpha=0.15)
                ax.plot(p95, color='#00ff88', linewidth=4, alpha=0.1)
                ax.plot(p5, color='#ff3a33', linewidth=4, alpha=0.1)
                
                ax.plot(p50, label='Precio Esperado (P50)', color='#3179f5', linewidth=2)
                ax.plot(p95, label='Optimista (P95)', color='#00ff88', linewidth=1.5)
                ax.plot(p5, label='Pesimista (P5)', color='#ff3a33', linewidth=1.5)
                
                ax.axhline(S0, color='white', linestyle='--', alpha=0.5, linewidth=1, label=f'Actual: ${S0:.2f}')
                
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.spines['bottom'].set_color('#2a2e39')
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(colors='#787b86') 
                
                ax.set_title(f'PROYECCIÓN ESTRATÉGICA', color='white', fontsize=14, pad=20, loc='left', fontweight='bold')
                ax.legend(facecolor='#131722', edgecolor='#2a2e39', loc='upper left', fontsize=10)
                
                st.pyplot(fig)
                
                # --- MÉTRICAS DE RESULTADO ---
                st.subheader("Objetivos Proyectados")
                m1, m2, m3 = st.columns(3)
                m1.metric("Escenario Pesimista (P5)", f"${p5[-1]:,.2f}", delta=f"{((p5[-1]/S0)-1)*100:.1f}%")
                m2.metric("Escenario Base (P50)", f"${p50[-1]:,.2f}", delta=f"{((p50[-1]/S0)-1)*100:.1f}%")
                m3.metric("Escenario Optimista (P95)", f"${p95[-1]:,.2f}", delta=f"{((p95[-1]/S0)-1)*100:.1f}%")
                
                # =========================================================
                # EL ANALISTA ALGORTÍMICO
                # =========================================================
                st.divider()
                st.subheader("📋 Contexto de Mercado (El Analista Algorítmico)")
                
                historial_6m = close_prices[-126:] if len(close_prices) >= 126 else close_prices
                soporte_6m = historial_6m.min()
                resistencia_6m = historial_6m.max()
                
                sma_50 = close_prices[-50:].mean() if len(close_prices) >= 50 else close_prices.mean()
                sma_200 = close_prices[-200:].mean() if len(close_prices) >= 200 else close_prices.mean()
                
                precios_30d = close_prices[-31:] if len(close_prices) >= 31 else close_prices
                retornos_30d = np.log(precios_30d[1:] / precios_30d[:-1])
                vol_30d = retornos_30d.std() * np.sqrt(252) * 100
                
                if S0 > sma_50 and sma_50 > sma_200:
                    st.success("🔥 **FUERTE ALCISTA:** El mercado está pagando cada vez más por este activo. La acción cotiza por encima de sus promedios de corto y largo plazo, indicando gran confianza e interés comprador en los últimos meses.")
                elif S0 > sma_50 and sma_50 < sma_200:
                    st.info("📈 **RECUPERACIÓN:** El activo viene de una mala racha histórica, pero en las últimas semanas la gente ha comenzado a acumularlo nuevamente. Está intentando revertir su tendencia a positiva.")
                elif S0 < sma_50 and sma_50 > sma_200:
                    st.warning("⚠️ **CORRECCIÓN:** Históricamente venía muy bien, pero recientemente los inversores han estado tomando ganancias (vendiendo). El sentimiento a corto plazo se ha enfriado y perdió su promedio de 50 días.")
                else:
                    st.error("❄️ **FUERTE BAJISTA:** El sentimiento actual es muy negativo. El mercado se ha estado deshaciendo de este activo consistentemente y cotiza por debajo de todos sus promedios importantes.")

                dist_soporte = ((S0 - soporte_6m) / S0) * 100
                dist_resistencia = ((resistencia_6m - S0) / S0) * 100
                
                if dist_resistencia < 4:
                    st.write(f"🧱 **ZONA DE TECHO:** El precio actual (${S0:.2f}) está **muy cerca de su resistencia máxima de los últimos 6 meses** (${resistencia_6m:.2f}). Atención: el mercado suele dudar en comprar aquí por miedo a un rebote a la baja.")
                elif dist_soporte < 4:
                    st.write(f"🛏️ **ZONA DE PISO:** El precio (${S0:.2f}) está **apoyado sobre su soporte clave de 6 meses** (${soporte_6m:.2f}). Históricamente, cuando cae a este nivel, los inversores lo perciben barato y entran a comprar.")
                else:
                    st.write(f"🧭 **PUNTO MEDIO:** El activo navega en zona neutral. Su piso histórico reciente (donde suelen entrar a rescatarlo) está en **${soporte_6m:.2f}**, y su techo psicológico (donde suelen vender) está en **${resistencia_6m:.2f}**.")

                if vol_30d < 15:
                    st.write(f"🌊 **VOLATILIDAD - Calma Chicha ({vol_30d:.1f}%):** La acción se está moviendo con extrema tranquilidad. Ideal para perfiles conservadores; no se esperan movimientos bruscos de un día para el otro.")
                elif vol_30d < 30:
                    st.write(f"📊 **VOLATILIDAD - Normal ({vol_30d:.1f}%):** El activo presenta fluctuaciones estándar y muy predecibles, saludables para el mercado de acciones.")
                elif vol_30d < 50:
                    st.write(f"🎢 **VOLATILIDAD - Alta ({vol_30d:.1f}%):** El precio está dando sacudidas fuertes. Hay mucha indecisión técnica o noticias impactando el activo. Hay grandes oportunidades de rebote, pero requiere estómago.")
                else:
                    st.write(f"⚡ **VOLATILIDAD - Extrema ({vol_30d:.1f}%):** ¡Cuidado! El activo está en una montaña rusa salvaje. El riesgo es muy alto y es probable ver movimientos de dos dígitos en muy pocos días.")

            except Exception as e:
                st.error(f"Error técnico o Ticker no válido: {e}")

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

    with st.expander("📚 Ayuda de Métricas (Glosario)"):
        st.write("""
        * **P/E Ratio:** Años que tardarías en recuperar la inversión con las ganancias actuales.
        * **PEG Ratio:** P/E ajustado al crecimiento. < 1: Crecimiento muy barato.
        * **P/B Ratio:** Precio contra los activos físicos contables.
        * **Margen Neto:** Porcentaje de ganancia limpia sobre lo vendido. > 20% es excelente.
        * **ROE:** Eficiencia gerencial. > 15% histórico es señal de gran empresa.
        * **Deuda/Cap:** Qué tan endeudada está frente a su dinero propio.
        """)

# ==========================================
# PESTAÑA 3: SENTIMIENTO DE MERCADO (RSS)
# ==========================================
with tab3:
    st.header("Análisis de Sentimiento (IA)")
    ticker_news = st.text_input("Ticker para noticias", value="AAPL").upper()
    
    if st.button("📰 Escanear Noticias"):
        with st.spinner('Leyendo titulares RSS oficiales...'):
            try:
                url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_news}&region=US&lang=en-US"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                xml_data = urllib.request.urlopen(req).read()
                items = ET.fromstring(xml_data).findall('./channel/item')
                
                if not items:
                    st.warning("Sin noticias recientes en el feed.")
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
                    
                    if len(rows) > 0:
                        avg = ( (total_score / len(rows)) + 1 ) * 50
                        st.subheader(f"Temperatura Actual: {avg:.1f}/100")
                        
                        if avg < 40:
                            st.error("PÁNICO / MIEDO EXTREMO")
                        elif avg > 60:
                            st.success("EUFORIA / CODICIA")
                        else:
                            st.info("SENTIMIENTO NEUTRAL")
                            
                        st.progress(int(avg))
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e:
                st.error(f"Error en lectura de noticias: {e}")