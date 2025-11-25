import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- KONFIGURASJON ---
st.set_page_config(page_title="RSI Trend Screener", layout="wide")

# --- TICKER HENTING ---
@st.cache_data(ttl=24*3600)
def get_sp500_tickers():
    """Scraper Wikipedia for oppdatert S&P 500 liste."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Yahoo Finance bruker bindestrek for noen aksjer (BRK.B), Wikipedia bruker punktum
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        st.error(f"Kunne ikke hente S&P 500 liste: {e}")
        return []

def get_oslo_tickers():
    """Returnerer en liste med de mest likvide aksjene på Oslo Børs (OBX + utvalgte)."""
    # Hardkodet liste da det er vanskelig å scrape Oslo Børs stabilt uten API-nøkkel
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

@st.cache_data(ttl=3600) # Cacher i 1 time
def get_stock_data(tickers, period="5y"):
    """Henter data. Returnerer dataframes og info."""
    data = {}
    infos = {}
    
    # Progress bar setup
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_tickers = len(tickers)
    
    # Batch processing er vanskelig med yfinance.Ticker().info, så vi looper.
    # Dette tar tid, så vi begrenser brukeren i UI hvis listen er for lang.
    
    for i, ticker in enumerate(tickers):
        # Vis progress
        status_text.text(f"Analyserer {i+1}/{total_tickers}: {ticker}")
        progress_bar.progress((i + 1) / total_tickers)
        
        try:
            stock = yf.Ticker(ticker)
            
            # Hent historikk (raskt)
            hist = stock.history(period=period)
            
            if hist.empty:
                continue

            # Hent info (TREG OPERASJON - flaskehalsen)
            # For å gjøre det raskere hopper vi over info hvis historikk er veldig kort
            if len(hist) < 200: 
                continue

            # Lagre data
            data[ticker] = hist
            
            # Prøv å hente info, men ikke krasj hvis det feiler
            try:
                infos[ticker] = stock.info
            except:
                infos[ticker] = {} # Tom dict hvis info feiler
                
        except Exception as e:
            continue # Hopp over feilende aksjer
        
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

        # SJEKK TOPP
        elif current_rsi > rsi_high and in_uptrend:
            in_uptrend = False
    
    hit_rate = (successful_hits / total_cycles) * 100 if total_cycles > 0 else 0
    return hit_rate, total_cycles, cycle_history

def run_backtest(df):
    initial = 100000
    capital = initial
    position = 0
    equity_curve = []
    in_pos = False
    
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        rsi = df['RSI'].iloc[i]
        
        if rsi < 30 and not in_pos:
            position = capital / price
            capital = 0
            in_pos = True
        elif rsi > 70 and in_pos:
            capital = position * price
            position = 0
            in_pos = False
            
        curr_eq = capital + (position * price)
        equity_curve.append({'Date': df.index[i], 'Equity': curr_eq})
        
    eq_df = pd.DataFrame(equity_curve).set_index('Date')
    tot_ret = (eq_df['Equity'].iloc[-1] - initial) / initial
    
    roll_max = eq_df['Equity'].cummax()
    dd = (eq_df['Equity'] - roll_max) / roll_max
    max_dd = dd.min()
    
    return eq_df, tot_ret, max_dd

# --- GUI ---

st.title("📈 RSI Trend Screener (Auto)")

# 1. VELG UNIVERS
st.sidebar.header("1. Velg Aksjeunivers")
universe_choice = st.sidebar.radio(
    "Hvilke aksjer vil du scanne?",
    ("Oslo Børs (Topp 50)", "S&P 500 (USA)", "Egen liste")
)

tickers_list = []
if universe_choice == "S&P 500 (USA)":
    st.sidebar.info("Henter automatisk listen over S&P 500 fra Wikipedia.")
    tickers_list = get_sp500_tickers()
    # Begrensning for MVP hastighet (kan fjernes hvis du har tid)
    limit = st.sidebar.slider("Begrens antall aksjer (for hastighet)", 10, 500, 50)
    tickers_list = tickers_list[:limit]
    
elif universe_choice == "Oslo Børs (Topp 50)":
    tickers_list = get_oslo_tickers()
    
else:
    raw_input = st.sidebar.text_area("Lim inn tickere (komma-separert)", "EQNR.OL, NHY.OL, AAPL")
    tickers_list = [t.strip() for t in raw_input.split(',')]

st.sidebar.markdown(f"**Valgt antall:** {len(tickers_list)} aksjer")

# 2. FILTERE
st.sidebar.header("2. Kriterier")
min_mcap_bn = st.sidebar.number_input("Min. Market Cap (Mrd)", 1.0)
min_pe = st.sidebar.number_input("Min P/E", 0.0)
max_pe = st.sidebar.number_input("Maks P/E", 100.0)
rsi_lim = st.sidebar.slider("RSI Bunn", 10, 40, 30)

if st.sidebar.button("🚀 Start Analyse"):
    st.write(f"Starter analyse av {len(tickers_list)} selskaper. Dette kan ta litt tid...")
    
    data, infos = get_stock_data(tickers_list)
    results = []
    
    progress_scan = st.progress(0)
    
    # Loop gjennom nedlastede data
    analyzed_count = 0
    for ticker, df in data.items():
        analyzed_count += 1
        progress_scan.progress(analyzed_count / len(data))
        
        info = infos.get(ticker, {})
        
        # Fundamental filtrering
        mcap = info.get('marketCap', 0)
        pe = info.get('trailingPE', 0)
        
        # Sjekk Market Cap (konverterer alt til "enheter" for enkelhet, reell valuta kan variere)
        if (mcap / 1_000_000_000) < min_mcap_bn: continue
        
        # Sjekk PE (hvis PE finnes)
        if pe is not None and (pe < min_pe or pe > max_pe): continue
        
        # Teknisk analyse
        df = calculate_technical_indicators(df)
        hit_rate, n_cycles, cycles = analyze_rsi_cycles(df, rsi_low=rsi_lim)
        
        if n_cycles < 3: continue # Minstekrav
        
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
        res_df = pd.DataFrame(results).sort_values(by='Hit Rate', ascending=False)
        
        st.success(f"Fant {len(res_df)} aksjer som matcher kriteriene!")
        
        # Vis tabell
        st.dataframe(res_df[['Ticker', 'Navn', 'Hit Rate', 'Sykluser', 'Siste Pris', 'P/E']])
        
        # Detaljvisning
        st.markdown("---")
        st.subheader("🔎 Detaljvisning")
        sel = st.selectbox("Velg aksje", res_df['Ticker'].tolist())
        
        row = res_df[res_df['Ticker'] == sel].iloc[0]
        sub_df = row['_df']
        
        # Grafer (samme som før)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sub_df.index, y=sub_df['Close'], name='Kurs'))
            # Tegn signaler
            for cy in row['_cycles']:
                col = 'green' if cy['Type'] == 'Hit' else 'red'
                fig.add_trace(go.Scatter(x=[cy['Date']], y=[cy['Price']], mode='markers', marker=dict(color=col, size=8)))
            fig.update_layout(title=f"{sel} - Signaler", height=350)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            eq, ret, dd = run_backtest(sub_df)
            st.metric("Backtest Avkastning", f"{ret*100:.1f}%", f"Drawdown: {dd*100:.1f}%")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=eq.index, y=eq['Equity'], fill='tozeroy'))
            fig_eq.update_layout(title="Equity Curve", height=300)
            st.plotly_chart(fig_eq, use_container_width=True)
            
    else:
        st.warning("Ingen aksjer funnet. Prøv å senke kravene eller endre univers.")
