import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime

# Descargar el diccionario de palabras para la IA (solo lo hace la primera vez)
nltk.download('vader_lexicon', quiet=True)

# ==========================================
# CONFIGURACIÓN DE LA APP
# ==========================================
st.set_page_config(page_title="Bunker Quant", layout="wide")
st.title("📊 Tu Bunker Cuantitativo")
st.markdown("Plataforma integral: Proyección estadística, Radiografía Fundamental y Análisis de Sentimiento con IA.")

# Las 3 pestañas de tu mesa de trabajo
tab1, tab2, tab3 = st.tabs(["📈 Predicción (Precios)", "🏥 Radiografía Fundamental", "🧠 Sentimiento de Mercado"])

# ==========================================
# PESTAÑA 1: MONTE CARLO (PRECIOS PUROS)
# ==========================================
with tab1:
    st.header("Simulador de Precios Futuros")
    st.markdown("Proyección pura del precio futuro de un activo aislando el ruido del mercado.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker_mc = st.text_input("Ticker de la Acción (Monte Carlo)", value="GOOGL").upper()
    with col2:
        meses_proyeccion = st.slider("Meses a Proyectar", min_value=1, max_value=60, value=24)
    with col3:
        simulaciones = st.slider("Cantidad de Escenarios", 1000, 10000, 10000)

    if st.button("🚀 Ejecutar Predicción"):
        with st.spinner(f'Analizando el histórico de {ticker_mc}...'):
            try:
                df = yf.download(ticker_mc, period='5y', progress=False)
                close_prices = df['Close'].squeeze().dropna().values
                S0 = close_prices[-1]
                
                log_returns = np.log(close_prices[1:] / close_prices[:-1])
                u = np.mean(log_returns)
                v = np.var(log_returns)
                drift = u - (0.5 * v)
                stdev = np.std(log_returns)
                
                dias_totales = meses_proyeccion * 21
                Z = np.random.normal(0, 1, (dias_totales, simulaciones))
                daily_returns = np.exp(drift + stdev * Z)
                
                price_paths = S0 * np.vstack([np.ones(simulaciones), np.cumprod(daily_returns, axis=0)])
                
                p50 = np.percentile(price_paths, 50, axis=1)
                p5 = np.percentile(price_paths, 5, axis=1)
                p95 = np.percentile(price_paths, 95, axis=1)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(price_paths[:, :50], color='grey', alpha=0.05) 
                ax.plot(p50, label='Precio Esperado (P50)', color='#1f77b4', linewidth=3)
                ax.plot(p95, label='Precio Optimista (P95)', color='#2ca02c', linestyle='--', linewidth=2)
                ax.plot(p5, label='Precio Pesimista (P5)', color='#d62728', linestyle='--', linewidth=2)
                ax.axhline(S0, color='black', linestyle=':', label=f'Precio Actual (${S0:.2f})')
                
                ax.set_title(f'Predicción de Precio: {ticker_mc} a {meses_proyeccion} meses', fontsize=14)
                ax.set_xlabel('Días de Mercado Proyectados')
                ax.set_ylabel('Precio de la Acción (USD)')
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                st.subheader(f"Precio proyectado en {meses_proyeccion} meses")
                m1, m2, m3 = st.columns(3)
                m1.metric("Escenario Pesimista (P5)", f"${p5[-1]:,.2f}")
                m2.metric("Escenario Esperado (P50)", f"${p50[-1]:,.2f}")
                m3.metric("Escenario Optimista (P95)", f"${p95[-1]:,.2f}")
                
            except Exception as e:
                st.error("Error al procesar los datos. Verifica el Ticker.")

# ==========================================
# PESTAÑA 2: RADIOGRAFÍA FUNDAMENTAL
# ==========================================
with tab2:
    st.header("Radiografía Fundamental de la Empresa")
    st.markdown("Evalúa la salud financiera, rentabilidad y múltiplos de valoración real de las compañías.")
    
    tickers_input = st.text_input("Ingresa los Tickers separados por comas", value="AAPL, GOOGL, KO, MELI, SPY")
    
    if st.button("📊 Generar Radiografía"):
        lista_tickers = [t.strip().upper() for t in tickers_input.split(',')]
        
        with st.spinner('Extrayendo balances contables de la base de datos...'):
            resultados = []
            for ticker in lista_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    def obtener_metrica(llave, tipo="num"):
                        valor = info.get(llave)
                        if valor is None:
                            return "N/A"
                        if tipo == "pct":
                            return f"{valor * 100:.2f}%"
                        if tipo == "moneda":
                            return f"${valor:,.2f}"
                        return f"{valor:.2f}"
                    
                    peg_bruto = info.get('trailingPegRatio', info.get('pegRatio'))
                    peg_formateado = f"{peg_bruto:.2f}" if peg_bruto is not None else "N/A"

                    resultados.append({
                        "Ticker": ticker,
                        "Precio": obtener_metrica('currentPrice', 'moneda') if info.get('currentPrice') else obtener_metrica('previousClose', 'moneda'),
                        "P/E Ratio": obtener_metrica('trailingPE'),
                        "P/E (Futuro)": obtener_metrica('forwardPE'),
                        "PEG Ratio": peg_formateado,
                        "P/B (Libros)": obtener_metrica('priceToBook'),
                        "Margen Neto": obtener_metrica('profitMargins', 'pct'),
                        "ROE (Capital)": obtener_metrica('returnOnEquity', 'pct'),
                        "Deuda/Capital": obtener_metrica('debtToEquity'),
                        "Dividend Yield": obtener_metrica('dividendYield', 'pct')
                    })
                        
                except Exception as e:
                    pass # Ignora si hay un error crítico con un ticker y sigue con el resto
            
            if resultados:
                df_resultados = pd.DataFrame(resultados)
                st.dataframe(df_resultados, use_container_width=True)
            else:
                st.warning("No se pudieron obtener datos para los tickers ingresados.")

    st.divider()
    with st.expander("📚 Glosario: ¿Cómo leer esta radiografía fundamental?"):
        st.markdown("""
        * **P/E Ratio (Price-to-Earnings):** Cuánto estás pagando por cada dólar que la empresa gana en la actualidad. 
        * **P/E (Futuro):** Lo mismo, pero usando las proyecciones para el próximo año. 
        * **PEG Ratio:** Relaciona el P/E con la velocidad a la que crecen las ganancias. Menor a 1.0 indica oportunidad.
        * **P/B (Price-to-Book):** Compara el precio con el valor físico contable.
        * **Margen Neto:** De todo lo que vende, ¿qué porcentaje es ganancia limpia?
        * **ROE (Return on Equity):** Mide la eficiencia de los directivos para sacarle jugo al dinero de los accionistas.
        * **Deuda/Capital:** Nivel de endeudamiento.
        * **Dividend Yield:** Porcentaje anual que te pagan en efectivo por tener la acción.
        """)

# ==========================================
# PESTAÑA 3: SENTIMIENTO DE MERCADO (IA)
# ==========================================
# ==========================================
# PESTAÑA 3: SENTIMIENTO DE MERCADO (IA)
# ==========================================
# ==========================================
# PESTAÑA 3: SENTIMIENTO DE MERCADO (IA)
# ==========================================
with tab3:
    st.header("Termómetro de Miedo y Codicia (IA)")
    st.markdown("Analiza los últimos titulares usando la fuente RSS directa de Yahoo Finance y NLP.")
    
    ticker_news = st.text_input("Ticker a analizar (Ej: AAPL, TSLA, MSFT)", value="AAPL").upper()
    
    if st.button("📰 Analizar Sentimiento"):
        with st.spinner(f'Leyendo noticias en tiempo real de {ticker_news}...'):
            try:
                # 1. Usar urllib y XML nativos para leer el RSS oficial (¡A prueba de fallos!)
                import urllib.request
                import xml.etree.ElementTree as ET
                
                # URL del canal RSS directo de Yahoo
                url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_news}&region=US&lang=en-US"
                # Nos hacemos pasar por un navegador web para que Yahoo no nos bloquee
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                
                try:
                    respuesta = urllib.request.urlopen(req)
                    xml_data = respuesta.read()
                    root = ET.fromstring(xml_data)
                    noticias = root.findall('./channel/item')
                except Exception as e:
                    noticias = []
                
                if not noticias:
                    st.warning("No se encontraron noticias recientes para este ticker en el feed oficial.")
                else:
                    sia = SentimentIntensityAnalyzer()
                    titulares_procesados = []
                    puntaje_total = 0
                    
                    for item in noticias:
                        # Extraer los datos limpios del XML
                        titulo = item.find('title').text if item.find('title') is not None else ''
                        fecha_raw = item.find('pubDate').text if item.find('pubDate') is not None else 'Fecha desconocida'
                        
                        # Limpiar un poco la fecha para que se vea mejor en la tabla
                        fecha = fecha_raw.replace(" +0000", "").replace(" GMT", "")
                        
                        # Analizar el titular con IA
                        analisis = sia.polarity_scores(titulo)
                        score = analisis['compound']
                        puntaje_total += score
                        
                        if score > 0.15:
                            estado = "🟢 Positivo"
                        elif score < -0.15:
                            estado = "🔴 Negativo"
                        else:
                            estado = "⚪ Neutral"
                            
                        titulares_procesados.append({
                            "Fecha / Hora": fecha,
                            "Impacto": estado,
                            "Titular": titulo,
                            "Score IA": round(score, 3)
                        })
                    
                    if len(titulares_procesados) > 0:
                        promedio = puntaje_total / len(titulares_procesados)
                        termometro = (promedio + 1) * 50 
                        
                        st.subheader("Temperatura Actual del Activo")
                        
                        if termometro < 40:
                            st.error(f"PÁNICO / MIEDO EXTREMO (Nivel: {termometro:.1f}/100)")
                            st.progress(int(termometro))
                            st.markdown("*El mercado está pesimista.*")
                        elif termometro > 60:
                            st.success(f"EUFORIA / CODICIA (Nivel: {termometro:.1f}/100)")
                            st.progress(int(termometro))
                            st.markdown("*El mercado está eufórico.*")
                        else:
                            st.info(f"NEUTRAL (Nivel: {termometro:.1f}/100)")
                            st.progress(int(termometro))
                            st.markdown("*Flujo de noticias normal.*")
                        
                        st.divider()
                        st.markdown("### Desglose de Titulares Analizados")
                        df_noticias = pd.DataFrame(titulares_procesados)
                        st.dataframe(df_noticias, use_container_width=True)
                    else:
                        st.warning("Se descargaron noticias, pero no se pudo extraer el texto.")
                    
            except Exception as e:
                st.error(f"Ocurrió un error de conexión o procesamiento. Detalle técnico: {e}")