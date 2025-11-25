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

# --- HJELPEFUNKSJONER ---
@st.cache_data(ttl=24*3600)
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        tickers = tables[0]['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except: return []

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
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Scanner {ticker} ({i+1}/{len(tickers)})")
        progress_bar.progress((i + 1) / len(tickers))
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

def find_trend_patterns(df, rsi_low, rsi_high, min_gain_pct, required_streak):
    """
    Finner mønster: Bunn -> Topp -> Bunn -> Topp
    Kriterier:
    1. Pris Topp > Pris Bunn * (1 + 10%)
    2. Pris Bunn(Nå) > Pris Bunn(Forrige)
    3. Pris Topp(Nå) > Pris Topp(Forrige)
    """
    df = df.dropna(subset=['RSI'])
    
    # Identifiser soner
    is_bottom_zone = df['RSI'] < rsi_low
    is_top_zone = df['RSI'] > rsi_high
    
    # Finn ekstrempunkter (lokale bunner og topper i sonene)
    cycles = []
    
    state = 'wait_for_bottom' # Tilstander: wait_for_bottom, in_bottom, wait_for_top, in_top
    
    current_cycle = {}
    local_min_price = float('inf')
    local_min_date = None
    local_max_price = 0
    local_max_date = None
    
    # Enkel tilstandsmaskin for å finne B->T sekvenser
    for date, row in df.iterrows():
        price = row['Close']
        rsi = row['RSI']
        
        if state == 'wait_for_bottom':
            if rsi < rsi_low:
                state = 'in_bottom'
                local_min_price = price
                local_min_date = date
        
        elif state == 'in_bottom':
            if rsi < rsi_low:
                # Oppdater laveste punkt i denne bunn-sonen
                if price < local_min_price:
                    local_min_price = price
                    local_min_date = date
            else:
                # RSI har gått over bunn-grensen igjen, bunnen er satt
                current_cycle['BottomDate'] = local_min_date
                current_cycle['BottomPrice'] = local_min_price
                state = 'wait_for_top'
                local_max_price = 0 # Reset maks
        
        elif state == 'wait_for_top':
            if rsi > rsi_high:
                state = 'in_top'
                local_max_price = price
                local_max_date = date
            # Hvis vi ser en ny bunn før vi ser en topp, oppdater bunnen (lavere lav) eller start på nytt?
            # Enklest: Hvis RSI går under low igjen før high, betyr det at trenden nedover fortsatte. 
            # Vi resetter til ny bunn.
            if rsi < rsi_low:
                state = 'in_bottom'
                local_min_price = price
                local_min_date = date
                
        elif state == 'in_top':
            if rsi > rsi_high:
                if price > local_max_price:
                    local_max_price = price
                    local_max_date = date
            else:
                # Toppen er ferdig
                current_cycle['TopDate'] = local_max_date
                current_cycle['TopPrice'] = local_max_price
                
                # Sjekk kravet om 10% stigning fra bunn til topp
                if current_cycle['TopPrice'] >= current_cycle['BottomPrice'] * (1 + min_gain_pct):
                    cycles.append(current_cycle.copy())
                
                # Gjør klar for neste syklus
                state = 'wait_for_bottom'
                current_cycle = {}
                local_min_price = float('inf')

    # Nå har vi en liste med gyldige sykluser (som alle har klart 10% kravet)
    # Nå må vi sjekke REKKEFØLGEN (Trend: Stigende bunner og stigende topper)
    
    if len(cycles) < required_streak:
        return 0, []

    # Sjekk bakover fra siste syklus for å se hvor lang trenden er
    current_streak = 1
    trend_cycles = [cycles[-1]]
    
    for i in range(len(cycles) - 2, -1, -1):
        curr = cycles[i+1] # Den nyere
        prev = cycles[i]   # Den eldre
        
        # Sjekk Trend-krav:
        # Nåværende Bunn > Forrige Bunn  OG  Nåværende Topp > Forrige Topp
        is_rising_bottom = curr['BottomPrice'] > prev['BottomPrice']
        is_rising_top = curr['TopPrice'] > prev['TopPrice']
        
        if is_rising_bottom and is_rising_top:
            current_streak += 1
            trend_cycles.insert(0, prev) # Legg til foran
        else:
            break # Trenden er brutt
            
    if current_streak >= required_streak:
        return current_streak, trend_cycles
    else:
        return 0, []

def filter_dataframe_by_period(df, period):
    if period == "5 år": return df
    end_date = df.index[-1]
    days_map = {"3 år": 3*365, "1 år": 365, "6 mnd": 180}
    start_date = end_date - timedelta(days=days_map.get(period, 365))
    return df[df.index >= start_date]

# --- GUI ---

st.title("📈 RSI Trend Screener: Higher Highs & Lows")

# --- INFO BOKS ---
with st.sidebar.expander("ℹ️ Slik virker algoritmen", expanded=True):
    st.markdown("""
    Appen leter etter en **vedvarende opptrend** definert ved:
    
    1.  **Syklus:** Prisen går fra RSI-bunn til RSI-topp.
    2.  **Minimumsvekst:** Hver RSI-topp må være minst **10 %** høyere enn bunnen i samme syklus.
    3.  **Stigende Bunner:** Neste RSI-bunn må ha *høyere kurs* enn forrige bunn.
    4.  **Stigende Topper:** Neste RSI-topp må ha *høyere kurs* enn forrige topp.
    5.  **Gjentakelse:** Dette mønsteret må ha skjedd **X ganger på rad** (som du velger).
    """)

# 1. MENY
st.sidebar.header("1. Innstillinger")
universe_choice = st.sidebar.radio("Marked:", ("Oslo Børs (Topp 50)", "S&P 500 (USA)", "Egen liste"))

tickers_list = []
if universe_choice == "S&P 500 (USA)":
    tickers_list = get_sp500_tickers()
    limit = st.sidebar.slider("Begrens antall (S&P 500)", 10, 500, 100)
    tickers_list = tickers_list[:limit]
elif universe_choice == "Oslo Børs (Topp 50)":
    tickers_list = get_oslo_tickers()
else:
    raw_input = st.sidebar.text_area("Tickere", "EQNR.OL, NHY.OL, AAPL, NVDA, TSLA")
    tickers_list = [t.strip() for t in raw_input.split(',')]

st.sidebar.subheader("Filter")
min_mcap_bn = st.sidebar.number_input("Min. Market Cap (Mrd)", value=0.1, step=0.1)
trend_streak = st.sidebar.number_input("Antall repeterende mønstre (X)", value=2, min_value=1)

st.sidebar.subheader("RSI Grenser")
col1, col2 = st.sidebar.columns(2)
rsi_low_lim = col1.number_input("Bunn", value=30)
rsi_high_lim = col2.number_input("Topp", value=70)

# 2. START
if st.sidebar.button("🚀 Start Trend Analyse"):
    st.write(f"Analyserer {len(tickers_list)} aksjer for trendmønstre...")
    st.session_state.results = None
    data, infos = get_stock_data(tickers_list)
    results = []
    
    prog = st.progress(0)
    for i, (ticker, df) in enumerate(data.items()):
        prog.progress((i+1)/len(data))
        info = infos.get(ticker, {})
        mcap = info.get('marketCap', 0) / 1e9
        
        if mcap < min_mcap_bn: continue
        
        df = calculate_technical_indicators(df)
        
        # Kjører den nye trend-algoritmen
        streak, trend_cycles = find_trend_patterns(
            df, 
            rsi_low=rsi_low_lim, 
            rsi_high=rsi_high_lim, 
            min_gain_pct=0.10, # Fast 10% krav
            required_streak=trend_streak
        )
        
        if streak >= trend_streak:
            results.append({
                'Ticker': ticker, 
                'Navn': info.get('shortName', ticker),
                'Trend Lengde': streak,
                'Siste Pris': round(df['Close'].iloc[-1], 2),
                '_df': df, 
                '_cycles': trend_cycles
            })
    prog.empty()
    
    if results:
        st.session_state.results = pd.DataFrame(results).sort_values('Trend Lengde', ascending=False)
        st.balloons() # Litt feiring når vi finner noe!
        st.success(f"Fant {len(results)} aksjer med sterk trend!")
    else:
        st.warning(f"Ingen aksjer oppfylte kravet om {trend_streak} repeterende mønstre med 10% gevinst.")

# 3. VISNING
if st.session_state.results is not None:
    res = st.session_state.results
    st.dataframe(res[['Ticker', 'Navn', 'Trend Lengde', 'Siste Pris']])
    
    st.markdown("---")
    c_sel, c_per = st.columns([3, 1])
    sel_ticker = c_sel.selectbox("Velg aksje:", res['Ticker'].tolist())
    period = c_per.selectbox("Tid:", ["6 mnd", "1 år", "3 år", "5 år"], index=1)
    
    if sel_ticker:
        row = res[res['Ticker'] == sel_ticker].iloc[0]
        df_plot = filter_dataframe_by_period(row['_df'], period)
        trend_cycles = row['_cycles']
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=(f"{row['Navn']} - Trend Analyse", "RSI"))

        # PRIS GRAF
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], name='Kurs', line=dict(color='gray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA50'], name='SMA50', line=dict(color='orange', width=1)), row=1, col=1)

        # TEGN TREND-LINJER OG PUNKTER
        # Vi tegner linjer mellom bunnene for å vise "Trendgulvet"
        valid_dates = [c['BottomDate'] for c in trend_cycles if c['BottomDate'] >= df_plot.index[0]]
        valid_prices = [c['BottomPrice'] for c in trend_cycles if c['BottomDate'] >= df_plot.index[0]]
        
        if valid_dates:
             fig.add_trace(go.Scatter(x=valid_dates, y=valid_prices, mode='lines+markers', 
                                      line=dict(color='green', width=2, dash='dash'),
                                      marker=dict(size=10, symbol='triangle-up'),
                                      name='Stigende Bunner'), row=1, col=1)

        # Marker topper
        top_dates = [c['TopDate'] for c in trend_cycles if c['TopDate'] >= df_plot.index[0]]
        top_prices = [c['TopPrice'] for c in trend_cycles if c['TopDate'] >= df_plot.index[0]]
        
        if top_dates:
             fig.add_trace(go.Scatter(x=top_dates, y=top_prices, mode='markers', 
                                      marker=dict(color='red', size=10, symbol='triangle-down'),
                                      name='Stigende Topper'), row=1, col=1)

        # RSI GRAF
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=rsi_high_lim, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=rsi_low_lim, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Marker RSI-punktene som korresponderer med trenden
        rsi_bottoms = df_plot.loc[valid_dates]['RSI'] if valid_dates else []
        if len(rsi_bottoms) > 0:
            fig.add_trace(go.Scatter(x=valid_dates, y=rsi_bottoms, mode='markers', marker=dict(color='green'), showlegend=False), row=2, col=1)

        fig.update_layout(height=600, margin=dict(t=40, b=20))
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.results is None:
    st.info("👈 Velg marked og trykk start.")
