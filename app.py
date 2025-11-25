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
                    successful
