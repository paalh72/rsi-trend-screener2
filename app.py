import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- KONFIGURASJON ---
st.set_page_config(page_title="RSI Trend Screener", layout="wide")

# Initier session state for å huske resultater mellom interaksjoner
if 'results' not in st.session_state:
    st.session_state.results = None
if 'analyzed_tickers' not in st.session_state:
    st.session_state.analyzed_tickers = []

# --- TICKER HENTING ---
@st.cache_data(ttl=24*3600)
def get_sp500_tickers():
    """Scraper Wikipedia for oppdatert S&P 500 liste."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        st.error(f"Kunne ikke hente S&P 500 liste: {e}")
        return []

def get_oslo_tickers():
    """Returnerer en liste med de mest likvide aksjene på Oslo Børs."""
    return [
        "EQNR.OL", "DNB.OL", "NHY.OL", "TEL.OL", "ORK.OL", "MOWI.OL", "YAR.OL", 
        "TOM.OL", "GJF.OL", "STB.OL", "SALM.OL", "AKRBP.OL", "SUBC.OL", "KOG.OL",
        "NAS.OL", "FRO.OL", "MPCC.OL", "VAR.OL", "PGS.OL", "TGS.OL", "LSG.OL",
        "BAKKA.OL", "ENTRA.OL", "SCHA.OL", "SCHB.OL", "AFG.OL", "ATEA.OL", 
        "BON.OL", "BWLPG.OL", "DNO.OL", "ELK.OL", "EPR.OL", "FLNG.OL", "GOGL.OL",
        "HAFNI.OL", "HEX.OL", "IDEX.OL", "KIT.OL", "NOD.OL", "OTOVO.OL", "PHEL.OL",
        "RECSI.OL", "SRBNK.OL", "VEI.OL", "VOW.OL", "XXL.OL", "ADE.OL", "BOUV.OL"
    ]

# --- DATABEHANDLING ---
@st.cache_data(ttl=3600) 
def get_stock_data(tickers, period="5y"):
    data = {}
    infos = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_tickers = len(tickers)
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Laster data {i+1}/{total_tickers}: {ticker}")
        progress_bar.progress((i + 1) / total_tickers)
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty or len(hist) < 200:
                continue

            data[ticker] = hist
            try:
                infos[ticker] = stock.info
            except:
                infos[ticker] = {} 
                
        except Exception as e:
            continue
        
    status_text.empty()
    progress_bar.empty()
    return data, infos

def calculate_technical_indicators(df):
    df = df.copy()
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # SMA
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    return df

def analyze_rsi_cycles(df, rsi_low=30, rsi_high=70):
    df = df.dropna(subset=['RSI'])
    in_uptrend = False
    entry_price = 0
    successful_hits = 0
    total_cycles = 0
    cycle_history = [] 
    
    for i in range(1, len(df) - 65): 
        current_rsi = df['RSI'].iloc[i]
        prev_rsi = df['RSI'].iloc[i-1]
        current_price = df['Close'].iloc[i]
        current_date = df.index[i]
        
        # SJEKK BUNN
        if prev_rsi < rsi_low and current_rsi >= rsi_low and not in_uptrend:
            in_uptrend = True
            entry_price = current_price
            total_cycles += 1
            
            # Sjekk Hit Rate (10% oppgang innen 3 mnd)
            future_window = df['High'].iloc[i+1 : i+64]
            if not future_window.empty:
                max_price = future_window.max()
                if max_price >= entry_price * 1.10:
                    successful_hits += 1
                    cycle_history.append({'Type': 'Hit', 'Date': current_date, 'Price': entry_price})
                else:
                    cycle_history.append({'Type': 'Miss', 'Date': current_date, 'Price': entry_price})

        # SJEKK TOPP (Bruker nå variabel for rsi_high)
        elif current_rsi > rsi_high and in_uptrend:
            in_uptrend = False
    
    hit_rate = (successful_hits / total_cycles) * 100 if total_cycles > 0 else 0
    return hit_rate, total_cycles, cycle_history

def filter_dataframe_by_period(df, period):
    """Filtrerer dataframe basert på valgt tidsperiode."""
    if period == "5 år": return df
    
    end_date = df.index[-1]
    start_date = end_date
    
    if period == "3 år":
        start_date = end_date - timedelta(days=3*365)
    elif period == "1 år":
        start_date = end_date - timedelta(days=365)
    elif period == "6 mnd":
        start_date = end_date - timedelta(days=180)
        
    return df[df.index >= start_date]

# --- GUI ---

st.title("📈 RSI Trend Screener Pro")

# 1. SIDEBAR SETUP
st.sidebar.header("1. Velg Aksjeunivers")
universe_choice = st.sidebar.radio(
    "Marked:",
    ("Oslo Børs (Topp 50)", "S&P 500 (USA)", "Egen liste")
)

tickers_list = []
if universe_choice == "S&P 500 (USA)":
    tickers_list = get_sp500_tickers()
    limit = st.sidebar.slider("Begrens antall (hastighet)", 10, 500, 50)
    tickers_list = tickers_list[:limit]
elif universe_choice == "Oslo Børs (Topp 50)":
    tickers_list = get_oslo_tickers()
else:
    raw_input = st.sidebar.text_area("Lim inn tickere (komma-separert)", "EQNR.OL, NHY.OL, AAPL, TSLA")
    tickers_list = [t.strip() for t in raw_input.split(',')]

st.sidebar.header("2. Kriterier")
# MCAP: Tillater nå helt ned til 0.0 (f.eks 0.1 mrd = 100 mill)
min_mcap_bn = st.sidebar.number_input("Min. Market Cap (Mrd)", min_value=0.0, value=0.1, step=0.1, format="%.1f")
min_pe = st.sidebar.number_input("Min P/E", 0.0)
max_pe = st.sidebar.number_input("Maks P/E", 100.0)

st.sidebar.subheader("RSI Innstillinger")
col_rsi1, col_rsi2 = st.sidebar.columns(2)
with col_rsi1:
    rsi_low_lim = st.number_input("RSI Bunn", value=30, step=1)
with col_rsi2:
    rsi_high_lim = st.number_input("RSI Topp", value=70, step=1)

# 3. ANALYSE KNAPP
if st.sidebar.button("🚀 Start Analyse"):
    st.write(f"Starter analyse av {len(tickers_list)} selskaper...")
    
    # Nullstill tidligere resultater
    st.session_state.results = None
    
    data, infos = get_stock_data(tickers_list)
    results = []
    
    progress_scan = st.progress(0)
    analyzed_count = 0
    
    for ticker, df in data.items():
        analyzed_count += 1
        progress_scan.progress(analyzed_count / len(data))
        
        info = infos.get(ticker, {})
        mcap = info.get('marketCap', 0)
        pe = info.get('trailingPE', 0)
        
        # MCAP Sjekk (milliarder)
        if (mcap / 1_000_000_000) < min_mcap_bn: continue
        
        # P/E Sjekk
        if pe is not None and (pe < min_pe or pe > max_pe): continue
        
        # Teknisk
        df = calculate_technical_indicators(df)
        
        # Sender med brukerens RSI grenser
        hit_rate, n_cycles, cycles = analyze_rsi_cycles(df, rsi_low=rsi_low_lim, rsi_high=rsi_high_lim)
        
        if n_cycles < 3: continue 
        
        results.append({
            'Ticker': ticker,
            'Navn': info.get('shortName', ticker),
            'Hit Rate': round(hit_rate, 1),
            'Sykluser': n_cycles,
            'Siste Pris': round(df['Close'].iloc[-1], 2),
            'P/E': round(pe, 2) if pe else None,
            '_df': df,
            '_cycles': cycles
        })
            
    progress_scan.empty()
    
    if results:
        # Lagre til session state slik at vi ikke mister data ved klikk
        st.session_state.results = pd.DataFrame(results).sort_values(by='Hit Rate', ascending=False)
        st.success(f"Fant {len(st.session_state.results)} aksjer!")
    else:
        st.warning("Ingen aksjer funnet med disse kriteriene.")

# --- RESULTATVISNING ---
# Sjekk om vi har resultater i minnet
if st.session_state.results is not None:
    res_df = st.session_state.results
    
    # Vis tabell
    st.dataframe(res_df[['Ticker', 'Navn', 'Hit Rate', 'Sykluser', 'Siste Pris', 'P/E']])
    
    st.markdown("---")
    st.subheader("🔎 Detaljvisning")
    
    col_sel, col_per = st.columns([3, 1])
    with col_sel:
        # Selectbox vil nå fungere uten å resette appen fordi data ligger i session_state
        selected_ticker = st.selectbox("Velg aksje for analyse", res_df['Ticker'].tolist())
    with col_per:
        time_period = st.selectbox("Tidsperiode", ["6 mnd", "1 år", "3 år", "5 år"], index=1)
    
    if selected_ticker:
        # Hent data fra session state
        row = res_df[res_df['Ticker'] == selected_ticker].iloc[0]
        full_df = row['_df']
        
        # Filtrer basert på tidsperiode
        plot_df = filter_dataframe_by_period(full_df, time_period)
        
        # Lag subplot med 2 rader (Pris øverst, RSI nederst)
        # share_x=True gjør at zooming i den ene påvirker den andre
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=(f"Kursutvikling - {row['Navn']}", "RSI Momentum"))

        # 1. Kursgraf (Candlestick eller Line - bruker Line for renhet her)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], name='Kurs', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA50'], name='SMA50', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA200'], name='SMA200', line=dict(color='red', width=1)), row=1, col=1)

        # Legg til signaler på kursgrafen
        # Må filtrere signaler så vi bare viser de som er innenfor valgt tidsperiode
        start_date_plot = plot_df.index[0]
        visible_cycles = [c for c in row['_cycles'] if c['Date'] >= start_date_plot]
        
        for cy in visible_cycles:
            col = 'green' if cy['Type'] == 'Hit' else 'red'
            fig.add_trace(go.Scatter(
                x=[cy['Date']], y=[cy['Price']], 
                mode='markers', 
                marker=dict(color=col, size=10, symbol='triangle-up'),
                name=f"Signal ({cy['Type']})",
                showlegend=False
            ), row=1, col=1)

        # 2. RSI Graf
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        
        # Legg til linjer for 30 og 70 (eller brukerens valgte grenser)
        fig.add_hline(y=rsi_high_lim, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=rsi_low_lim, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Fyll området mellom 30 og 70 for visuell effekt
        # Plotly har ikke enkel "fill between horizontal lines" direkte, men grensene hjelper.
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_yaxes(title_text="Pris", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Nøkkeltall under graf
        st.info(f"RSI Parametere brukt: Bunn < {rsi_low_lim}, Topp > {rsi_high_lim}. Hit Rate basert på {row['Sykluser']} sykluser siste 5 år.")

elif st.session_state.results is None:
    st.info("👈 Velg innstillinger i menyen og trykk 'Start Analyse'")
