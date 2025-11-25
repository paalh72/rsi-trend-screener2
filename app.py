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
    """Henter S&P 500 eller bruker backup."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except Exception:
        # Fallback liste
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
            "XOM", "JPM", "PG", "V", "LLY", "HD", "MA", "CVX", "MRK", "ABBV", 
            "PEP", "KO", "BAC", "AVGO", "COST", "PFE", "TMO", "WMT", "CSCO", "MCD", "DIS"
        ]

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
    
    if not tickers: return {}, {}

    for i, ticker in enumerate(tickers):
        status_text.text(f"Laster data {i+1}/{len(tickers)}: {ticker}")
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
    """
    Beregner RSI med Wilder's Smoothing (Standard for TradingView/Nordnet).
    Dette gir mer korrekte signaler enn vanlig glidende snitt.
    """
    df = df.copy()
    
    # RSI Beregning (Wilder's Smoothing)
    delta = df['Close'].diff()
    
    # Separer oppgang og nedgang
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Bruk ewm (Exponential Weighted Moving Average) for å etterligne Wilder's
    # com = window - 1. For RSI 14 er com=13.
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # SMA
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    return df

def find_trend_patterns(df, rsi_low, rsi_high, min_gain_pct, required_streak):
    """
    Finner mønster: Bunn -> Topp -> Bunn -> Topp
    """
    df = df.dropna(subset=['RSI'])
    
    cycles = []
    state = 'wait_for_bottom' 
    current_cycle = {}
    local_min_price = float('inf')
    local_min_date = None
    local_max_price = 0
    local_max_date = None
    
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
                # Vi er fortsatt i bunn-sonen, se etter lavere bunn
                if price < local_min_price:
                    local_min_price = price
                    local_min_date = date
            else:
                # RSI brøt opp fra bunn
                current_cycle['BottomDate'] = local_min_date
                current_cycle['BottomPrice'] = local_min_price
                state = 'wait_for_top'
                local_max_price = 0 
        
        elif state == 'wait_for_top':
            if rsi > rsi_high:
                state = 'in_top'
                local_max_price = price
                local_max_date = date
            
            # Reset hvis RSI faller ned igjen uten å ha nådd toppen (falsk start)
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
                # Toppen er satt
                current_cycle['TopDate'] = local_max_date
                current_cycle['TopPrice'] = local_max_price
                
                # Sjekk gevinstkravet
                if current_cycle['TopPrice'] >= current_cycle['BottomPrice'] * (1 + min_gain_pct):
                    cycles.append(current_cycle.copy())
                
                state = 'wait_for_bottom'
                current_cycle = {}
                local_min_price = float('inf')

    total_valid_cycles = len(cycles)

    # Hvis brukeren kun ber om 1 syklus (ingen trend-sammenligning kreves)
    if required_streak == 0:
        return 0, cycles, total_valid_cycles

    if len(cycles) < 2:
        # Har ikke nok sykluser til å sammenligne A mot B
        return (1 if len(cycles)==1 else 0), cycles, total_valid_cycles

    # Sjekk trend (sammenlign bakover)
    current_streak = 0
    trend_cycles = []
    
    # Vi starter fra siste syklus og går bakover
    # Hvis siste syklus er del av en trend, teller vi den
    
    # Logikk: Finn lengste sammenhengende kjede på slutten
    temp_cycles = [cycles[-1]]
    temp_streak = 1
    
    for i in range(len(cycles) - 2, -1, -1):
        curr = cycles[i+1] # Nyere
        prev = cycles[i]   # Eldre
        
        # Trend krav: Høyere bunn OG Høyere topp
        is_rising_bottom = curr['BottomPrice'] > prev['BottomPrice']
        is_rising_top = curr['TopPrice'] > prev['TopPrice']
        
        if is_rising_bottom and is_rising_top:
            temp_streak += 1
            temp_cycles.insert(0, prev)
        else:
            break # Trend brutt
            
    # Hvis kravet er 1 repetisjon (altså 1 trend-hopp: A -> B), må streak være >= 2 (som betyr 2 sykluser involvert)
    # Men for å være snill med visningen: Hvis bruker velger "1", viser vi alt som har minst 1 syklus?
    # Nei, bruker spurte om "repeterende". 
    # La oss si: Streak tallet er antall "hopp".
    # Egentlig er "Streak" i koden nå antall Sykluser i trenden.
    
    return temp_streak, temp_cycles, total_valid_cycles

def filter_dataframe_by_period(df, period):
    if period == "5 år": return df
    end_date = df.index[-1]
    days_map = {"3 år": 3*365, "1 år": 365, "6 mnd": 180}
    start_date = end_date - timedelta(days=days_map.get(period, 365))
    return df[df.index >= start_date]

# --- GUI ---

st.title("📈 RSI Trend Screener Pro")

# 1. MENY
st.sidebar.header("1. Innstillinger")
universe_choice = st.sidebar.radio("Marked:", ("Oslo Børs (Topp 50)", "S&P 500 (USA)", "Egen liste"))

tickers_list = []
if universe_choice == "S&P 500 (USA)":
    tickers_list = get_sp500_tickers()
    limit = st.sidebar.slider("Begrens antall aksjer", 10, len(tickers_list), 50)
    tickers_list = tickers_list[:limit]
elif universe_choice == "Oslo Børs (Topp 50)":
    tickers_list = get_oslo_tickers()
else:
    raw_input = st.sidebar.text_area("Tickere", "EQNR.OL, NHY.OL, AAPL, NVDA, TSLA")
    tickers_list = [t.strip() for t in raw_input.split(',')]

st.sidebar.subheader("Filtrering")
min_mcap_bn = st.sidebar.number_input("Min. Market Cap (Mrd)", value=0.1, step=0.1)

# Gevinst og Streak
st.sidebar.markdown("---")
st.sidebar.caption("Trend Kriterier")
min_gain_input = st.sidebar.number_input("Min. % oppgang i hver syklus", value=5.0, step=1.0) # Satt ned default for test
min_gain_decimal = min_gain_input / 100.0

trend_streak_req = st.sidebar.number_input("Minimum antall sykluser i trend", value=1, min_value=1, help="1 = Vis alle med minst én godkjent syklus. 2 = Krever at syklus 2 er høyere enn syklus 1.")

st.sidebar.subheader("RSI Grenser")
col1, col2 = st.sidebar.columns(2)
rsi_low_lim = col1.number_input("Bunn (<)", value=30)
rsi_high_lim = col2.number_input("Topp (>)", value=70)

# 2. START
if st.sidebar.button("🚀 Start Trend Analyse"):
    st.write(f"Analyserer {len(tickers_list)} aksjer...")
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
        
        # Kjører algoritmen
        streak, trend_cycles, total_cycles = find_trend_patterns(
            df, 
            rsi_low=rsi_low_lim, 
            rsi_high=rsi_high_lim, 
            min_gain_pct=min_gain_decimal,
            required_streak=trend_streak_req
        )
        
        # Hvis bruker setter streak=1, vil vi bare se at vi fant noe (streak >= 1)
        if streak >= trend_streak_req:
            results.append({
                'Ticker': ticker, 
                'Navn': info.get('shortName', ticker),
                'Trend Lengde': streak,
                'Fant Sykluser': total_cycles, # Ny kolonne for debugging
                'Siste Pris': round(df['Close'].iloc[-1], 2),
                '_df': df, 
                '_cycles': trend_cycles
            })
    prog.empty()
    
    if results:
        st.session_state.results = pd.DataFrame(results).sort_values('Trend Lengde', ascending=False)
        st.success(f"Fant {len(results)} aksjer!")
    else:
        st.error("Ingen aksjer funnet. Tips: Prøv å justere RSI-grensene (f.eks 35 og 65) eller senk gevinstkravet.")

# 3. VISNING
if st.session_state.results is not None:
    res = st.session_state.results
    
    # Vis tabell med ny kolonne
    st.dataframe(res[['Ticker', 'Navn', 'Trend Lengde', 'Fant Sykluser', 'Siste Pris']])
    
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
                            subplot_titles=(f"{row['Navn']} - Trend (Viser sykluser i trenden)", "RSI (Wilder's)"))

        # PRIS GRAF
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], name='Kurs', line=dict(color='gray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA50'], name='SMA50', line=dict(color='orange', width=1)), row=1, col=1)

        valid_dates = [c['BottomDate'] for c in trend_cycles if c['BottomDate'] >= df_plot.index[0]]
        valid_prices = [c['BottomPrice'] for c in trend_cycles if c['BottomDate'] >= df_plot.index[0]]
        
        if valid_dates:
             # Grønn stiplet linje som binder bunnene sammen
             fig.add_trace(go.Scatter(x=valid_dates, y=valid_prices, mode='lines+markers', 
                                      line=dict(color='green', width=2, dash='dash'),
                                      marker=dict(size=10, symbol='triangle-up'),
                                      name='Stigende Bunner'), row=1, col=1)

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
        
        fig.update_layout(height=600, margin=dict(t=40, b=20))
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.results is None:
    st.info("👈 Velg marked, juster parametere og trykk start.")
