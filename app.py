import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- KONFIGURASJON ---
st.set_page_config(page_title="RSI Trend Screener", layout="wide")

# Initier session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'analyzed_tickers' not in st.session_state:
    st.session_state.analyzed_tickers = []

# --- HJELPEFUNKSJONER ---
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
        return []

def get_oslo_tickers():
    return [
        "EQNR.OL", "DNB.OL", "NHY.OL", "TEL.OL", "ORK.OL", "MOWI.OL", "YAR.OL", 
        "TOM.OL", "GJF.OL", "STB.OL", "SALM.OL", "AKRBP.OL", "SUBC.OL", "KOG.OL",
        "NAS.OL", "FRO.OL", "MPCC.OL", "VAR.OL", "PGS.OL", "TGS.OL", "LSG.OL",
        "BAKKA.OL", "ENTRA.OL", "SCHA.OL", "SCHB.OL", "AFG.OL", "ATEA.OL", 
        "BON.OL", "BWLPG.OL", "DNO.OL", "ELK.OL", "EPR.OL", "FLNG.OL", "GOGL.OL",
        "HAFNI.OL", "HEX.OL", "IDEX.OL", "KIT.OL", "NOD.OL", "OTOVO.OL", "PHEL.OL",
        "RECSI.OL", "SRBNK.OL", "VEI.OL", "VOW.OL", "XXL.OL", "ADE.OL", "BOUV.OL"
    ]

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
            if hist.empty or len(hist) < 200: continue
            data[ticker] = hist
            try: infos[ticker] = stock.info
            except: infos[ticker] = {} 
        except: continue
        
    status_text.empty()
    progress_bar.empty()
    return data, infos

def calculate_technical_indicators(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
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
    
    # Krav: Minst 10% økning for å kalle det en "Hit"
    min_gain_req = 0.10
    # Tidsvindu for gevinst: 3 mnd (ca 63 handelsdager)
    look_ahead_days = 63 
    
    for i in range(1, len(df) - look_ahead_days - 2): 
        current_rsi = df['RSI'].iloc[i]
        prev_rsi = df['RSI'].iloc[i-1]
        current_price = df['Close'].iloc[i]
        current_date = df.index[i]
        
        # SJEKK BUNN (KJØPSSIGNAL)
        if prev_rsi < rsi_low and current_rsi >= rsi_low and not in_uptrend:
            in_uptrend = True
            entry_price = current_price
            total_cycles += 1
            
            # Validering: Sjekk om prisen stiger 10% de neste 3 mnd
            future_window = df['High'].iloc[i+1 : i+look_ahead_days]
            
            if not future_window.empty:
                max_price = future_window.max()
                if max_price >= entry_price * (1 + min_gain_req):
                    successful_hits += 1
                    cycle_history.append({'Type': 'Hit', 'Date': current_date, 'Price': entry_price})
                else:
                    cycle_history.append({'Type': 'Miss', 'Date': current_date, 'Price': entry_price})
            else:
                 # Hvis vi er helt på slutten av datasettet og ikke kan se 3 mnd frem
                 cycle_history.append({'Type': 'Pending', 'Date': current_date, 'Price': entry_price})

        # SJEKK TOPP (SALGSSIGNAL)
        elif current_rsi > rsi_high and in_uptrend:
            in_uptrend = False
    
    # Beregn Hit Rate (Ignorerer 'Pending' sykluser som er for ferske)
    finished_cycles = [c for c in cycle_history if c['Type'] != 'Pending']
    hits = len([c for c in finished_cycles if c['Type'] == 'Hit'])
    total = len(finished_cycles)
    
    hit_rate = (hits / total) * 100 if total > 0 else 0
    return hit_rate, total, cycle_history

def filter_dataframe_by_period(df, period):
    if period == "5 år": return df
    end_date = df.index[-1]
    days_map = {"3 år": 3*365, "1 år": 365, "6 mnd": 180}
    start_date = end_date - timedelta(days=days_map.get(period, 365))
    return df[df.index >= start_date]

# --- GUI ---

st.title("📈 RSI Trend Screener Pro")

# --- INFO BOKS (Sidebar) ---
with st.sidebar.expander("ℹ️ Slik virker modellen", expanded=False):
    st.markdown("""
    **Faste kriterier (kan ikke endres):**
    
    1.  **Hit Rate Krav:** En syklus regnes som en "Hit" (suksess) kun hvis aksjekursen stiger minst **10 %** innen **3 måneder (63 handelsdager)** etter kjøpssignalet.
    2.  **Miss:** Hvis aksjen gir signal, men ikke klarer 10 % stigning før tiden er ute, regnes det som en "Miss". Dette trekker Hit Rate ned.
    3.  **Kjøpssignal:** Når RSI krysser *opp* gjennom bunn-grensen (f.eks 30).
    4.  **Nullstilling:** Syklusen avsluttes når RSI går over topp-grensen (f.eks 70).
    5.  **Data:** Bruker daglige sluttkurser justert for utbytte (Adjusted Close).
    """)

# 1. SIDEBAR MENY
st.sidebar.header("1. Innstillinger")
universe_choice = st.sidebar.radio("Marked:", ("Oslo Børs (Topp 50)", "S&P 500 (USA)", "Egen liste"))

tickers_list = []
if universe_choice == "S&P 500 (USA)":
    tickers_list = get_sp500_tickers()
    limit = st.sidebar.slider("Antall aksjer (S&P 500)", 10, 500, 50)
    tickers_list = tickers_list[:limit]
elif universe_choice == "Oslo Børs (Topp 50)":
    tickers_list = get_oslo_tickers()
else:
    raw_input = st.sidebar.text_area("Tickere (komma-separert)", "EQNR.OL, NHY.OL, AAPL, TSLA")
    tickers_list = [t.strip() for t in raw_input.split(',')]

st.sidebar.subheader("Filtrering")
min_mcap_bn = st.sidebar.number_input("Min. Market Cap (Mrd)", min_value=0.0, value=0.1, step=0.1, format="%.1f")
min_pe = st.sidebar.number_input("Min P/E", 0.0)
max_pe = st.sidebar.number_input("Maks P/E", 100.0)

st.sidebar.subheader("RSI Grenser")
c1, c2 = st.sidebar.columns(2)
rsi_low_lim = c1.number_input("Bunn", value=30)
rsi_high_lim = c2.number_input("Topp", value=70)

# 2. KJØR ANALYSE
if st.sidebar.button("🚀 Start Analyse"):
    st.write(f"Analyserer {len(tickers_list)} aksjer...")
    st.session_state.results = None # Reset
    data, infos = get_stock_data(tickers_list)
    results = []
    
    prog = st.progress(0)
    for i, (ticker, df) in enumerate(data.items()):
        prog.progress((i+1)/len(data))
        info = infos.get(ticker, {})
        mcap = info.get('marketCap', 0) / 1e9
        pe = info.get('trailingPE', None)
        
        if mcap < min_mcap_bn: continue
        if pe and (pe < min_pe or pe > max_pe): continue
        
        df = calculate_technical_indicators(df)
        hit, n_cyc, cyc_det = analyze_rsi_cycles(df, rsi_low_lim, rsi_high_lim)
        
        if n_cyc < 3: continue 
        
        results.append({
            'Ticker': ticker, 'Navn': info.get('shortName', ticker),
            'Hit Rate': round(hit, 1), 'Sykluser': n_cyc,
            'Pris': round(df['Close'].iloc[-1], 2), 'P/E': round(pe, 2) if pe else None,
            '_df': df, '_cycles': cyc_det
        })
    prog.empty()
    
    if results:
        st.session_state.results = pd.DataFrame(results).sort_values('Hit Rate', ascending=False)
        st.success(f"Fant {len(results)} aksjer.")
    else:
        st.warning("Ingen treff.")

# 3. VIS RESULTATER
if st.session_state.results is not None:
    res = st.session_state.results
    st.dataframe(res[['Ticker', 'Navn', 'Hit Rate', 'Sykluser', 'Pris', 'P/E']])
    
    st.markdown("---")
    c_sel, c_per = st.columns([3, 1])
    sel_ticker = c_sel.selectbox("Velg aksje:", res['Ticker'].tolist())
    period = c_per.selectbox("Tid:", ["6 mnd", "1 år", "3 år", "5 år"], index=1)
    
    if sel_ticker:
        row = res[res['Ticker'] == sel_ticker].iloc[0]
        df_plot = filter_dataframe_by_period(row['_df'], period)
        
        # Plotting
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=(f"{row['Navn']} - Kurs", "RSI (14)"))

        # Kurs
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], name='Pris', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA50'], name='SMA50', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA200'], name='SMA200', line=dict(color='red', width=1)), row=1, col=1)

        # Signaler (kun synlige i perioden)
        vis_cycles = [c for c in row['_cycles'] if c['Date'] >= df_plot.index[0]]
        for c in vis_cycles:
            col = 'green' if c['Type'] == 'Hit' else ('red' if c['Type'] == 'Miss' else 'gray')
            fig.add_trace(go.Scatter(x=[c['Date']], y=[c['Price']], mode='markers', 
                                     marker=dict(color=col, size=12, symbol='triangle-up'),
                                     name=c['Type'], showlegend=False), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=rsi_high_lim, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=rsi_low_lim, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20))
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"Grønn trekant = Pris steg >10% innen 3 mnd. Rød trekant = Pris steg ikke nok.")

elif st.session_state.results is None:
    st.info("👈 Konfigurer søket i menyen til venstre.")
