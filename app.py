import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. 페이지 설정 (가장 윗줄에 있어야 함)
# ---------------------------------------------------------
st.set_page_config(page_title="Whale Hunter", layout="wide", page_icon="🐋")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 2. 모바일 스타일 (CSS) 설정
# ---------------------------------------------------------
# f-string 충돌 방지를 위해 일반 문자열로 작성
GLOBAL_CSS = """
<style>
    /* 전체 레이아웃 패딩 제거 */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    header, footer { visibility: hidden; }
    
    /* 다크모드 배경 */
    .stApp { background-color: #000000; color: #f2f2f7; }
    
    /* 입력창 스타일 */
    div[data-baseweb="input"] > div {
        background-color: #1c1c1e !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
    input { color: white !important; }
    
    /* 버튼 스타일 */
    button {
        background-color: #0A84FF !important;
        color: white !important;
        border-radius: 8px !important;
        height: 3rem !important;
        font-weight: 700 !important;
    }

    /* 결과 리포트 컨테이너 */
    .mobile-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #000000;
        color: #f2f2f7;
        padding: 0 4px 40px 4px;
        max-width: 100%;
    }
    .header-box {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid #333; padding: 12px 4px; margin-bottom: 20px;
    }
    .ticker-txt { font-size: 28px; font-weight: 800; line-height: 1.1; margin: 0; }
    .sub-txt { font-size: 12px; color: #8e8e93; margin-top: 4px; }
    
    .score-box {
        background: #1c1c1e; border: 1px solid #333; border-radius: 12px;
        padding: 8px 16px; text-align: center; min-width: 80px;
    }
    .score-num { font-size: 26px; font-weight: 800; line-height: 1; }
    .score-label { font-size: 10px; color: #8e8e93; font-weight: 600; margin-top: 4px; }

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }
    .stat-card {
        background: #1c1c1e; border-radius: 12px; padding: 12px; text-align: center;
    }
    .stat-val { font-size: 18px; font-weight: 800; margin-top: 2px; }
    .stat-lbl { font-size: 11px; color: #8e8e93; }

    .main-card {
        background: #1c1c1e; border-radius: 14px; padding: 16px;
        margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    .row-flex { display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 14px; }
    .row-lbl { color: #8e8e93; }
    .row-val { font-weight: 600; }
    
    .section-head { font-size: 16px; font-weight: 700; margin: 30px 0 10px 0; color: #f2f2f7; }
    
    .driver-card {
        background: #2c2c2e; padding: 12px 10px; border-radius: 10px;
        display: flex; flex-direction: column; justify-content: center;
    }
    
    /* 가로 스크롤 테이블 */
    .scroll-table {
        overflow-x: auto; -webkit-overflow-scrolling: touch;
        margin-top: 10px; padding-bottom: 10px;
    }
    table { width: 100%; border-collapse: collapse; white-space: nowrap; }
    td { font-size: 12px; padding: 6px 8px; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. 상수 및 설정
# ---------------------------------------------------------
WINDOW_SIZE = 60
FORECAST_DAYS = 30

# Colors
C_BULL = "#30D158" # Green
C_BEAR = "#FF453A" # Red
C_NEUT = "#8E8E93" # Gray
C_WARN = "#FFD60A" # Yellow
C_CYAN = "#64D2FF" # Blue
C_PURP = "#BF5AF2" # Purple

# ---------------------------------------------------------
# 4. 데이터 함수 (캐싱 적용)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_stock_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'mkt_cap': info.get('marketCap', 0),
            'per': info.get('trailingPE', None),
            'roe': info.get('returnOnEquity', None),
            'name': info.get('longName', ticker)
        }
    except:
        return {'mkt_cap': 0, 'per': None, 'roe': None, 'name': ticker}

@st.cache_data(ttl=1800)
def get_clean_data(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        
        if len(df) < 100: return None

        # 지표 계산
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA120'] = df['Close'].rolling(120).mean()
        
        # MACD
        df['EMA12'] = df['Close'].ewm(span=12).mean()
        df['EMA26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger
        std = df['Close'].rolling(20).std()
        df['BB_Up'] = df['MA20'] + (std * 2)
        df['BB_Lo'] = df['MA20'] - (std * 2)
        
        # ATR
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        
        # OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # MFI
        typ = (df['High'] + df['Low'] + df['Close']) / 3
        mf = typ * df['Volume']
        df['MFI'] = 100 - (100 / (1 + mf.rolling(14).sum() / mf.rolling(14).sum())) # Simplified

        # Volatility
        df['Vol'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        df.dropna(inplace=True)
        return df
    except: return None

# ---------------------------------------------------------
# 5. 분석 로직
# ---------------------------------------------------------
def run_monte_carlo(df):
    np.random.seed(42)
    last = df['Close'].iloc[-1]
    vol = df['Vol'].iloc[-30:].mean() / 100 # 일일 변동성 근사치
    
    sims = []
    days = 120
    for _ in range(500): # 시뮬레이션 횟수
        prices = [last]
        p = last
        for _ in range(days):
            p = p * (1 + np.random.normal(0, vol))
            prices.append(p)
        sims.append(prices)
    
    sims = np.array(sims)
    final_prices = sims[:, -1]
    max_prices = np.max(sims, axis=1)
    
    target = last * 1.3
    win_prob = (max_prices >= target).mean() * 100
    peak_yield = (np.median(max_prices) - last) / last * 100
    
    # 예상 도달일 (단순화)
    try:
        hit_indices = np.argmax(sims >= target, axis=1)
        valid_hits = hit_indices[hit_indices > 0]
        if len(valid_hits) > 0:
            avg_days = int(np.mean(valid_hits))
            expected_date = (datetime.now() + timedelta(days=avg_days)).strftime("%Y-%m-%d")
        else:
            expected_date = "N/A"
    except: expected_date = "N/A"

    return win_prob, peak_yield, expected_date

def analyze_stock(df, info, win_prob):
    last = df.iloc[-1]
    score = 50
    cards = []
    
    # 1. Trend
    if last['Close'] > last['MA20']:
        score += 15
        cards.append(('Trend', 'Uptrend', C_BULL))
    else:
        score -= 15
        cards.append(('Trend', 'Downtrend', C_BEAR))
        
    # 2. Whale (OBV Divergence simplified)
    price_change = (last['Close'] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100
    obv_change = (last['OBV'] - df['OBV'].iloc[-20]) / (abs(df['OBV'].iloc[-20])+1) * 100
    
    if obv_change > price_change + 5:
        score += 20
        cards.append(('Whale', 'Buying', C_BULL))
    elif obv_change < price_change - 5:
        score -= 20
        cards.append(('Whale', 'Selling', C_BEAR))
    else:
        cards.append(('Whale', 'Neutral', C_NEUT))

    # 3. Probability
    if win_prob >= 40: score += 15
    elif win_prob <= 10: score -= 15

    # 4. Volatility
    if last['Vol'] > 3.0: cards.append(('Vol', 'High', C_PURP))
    else: cards.append(('Vol', 'Normal', C_NEUT))

    score = max(0, min(100, int(score)))
    
    # Target/Stop
    atr = last['ATR']
    stop = last['Close'] - (atr * 2)
    target = last['Close'] + (atr * 3)
    
    # Color Determination
    main_color = C_BULL if score >= 70 else (C_WARN if score >= 40 else C_BEAR)
    
    return {
        'score': score, 'color': main_color, 'cards': cards,
        'stop': stop, 'target': target, 'close': last['Close'],
        'atr': atr, 'rsi': last['RSI']
    }

# ---------------------------------------------------------
# 6. HTML 생성 (오류 방지 위해 f-string 최소화)
# ---------------------------------------------------------
def make_html(ticker, info, anal, monte):
    win_prob, peak_yield, exp_date = monte
    
    # 시가총액 포맷팅
    cap = info['mkt_cap']
    if cap > 1e12: cap_str = f"{cap/1e12:.1f}T"
    elif cap > 1e9: cap_str = f"{cap/1e9:.1f}B"
    else: cap_str = "-"

    # 색상
    peak_col = C_PURP if peak_yield > 40 else C_BULL
    
    # 카드 HTML 생성
    cards_html = ""
    for title, val, col in anal['cards']:
        cards_html += f"""
        <div class="driver-card" style="border-left: 3px solid {col};">
            <div style="font-size:11px; color:#8e8e93;">{title}</div>
            <div style="font-size:13px; font-weight:700; color:#fff;">{val}</div>
        </div>
        """

    html = f"""
    <div class="mobile-container">
        <div class="header-box">
            <div>
                <div class="ticker-txt">{ticker}</div>
                <div class="sub-txt">{info['name'][:15]}.. • {cap_str}</div>
            </div>
            <div class="score-box">
                <div class="score-num" style="color: {anal['color']};">{anal['score']}</div>
                <div class="score-label">AI SCORE</div>
            </div>
        </div>

        <div class="grid-2">
            <div class="stat-card">
                <div class="stat-lbl">Win Prob</div>
                <div class="stat-val" style="color: {C_BULL if win_prob > 30 else '#fff'}">{win_prob:.0f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-lbl">Peak Upside</div>
                <div class="stat-val" style="color: {peak_col}">+{peak_yield:.0f}%</div>
            </div>
        </div>

        <div class="main-card" style="border: 1px solid {anal['color']}44;">
            <div style="font-size:15px; font-weight:700; margin-bottom:15px; color:#fff;">🎯 Trading Setup</div>
            
            <div class="row-flex">
                <span class="row-lbl">Current Price</span>
                <span class="row-val">${anal['close']:.2f}</span>
            </div>
            <div class="row-flex">
                <span class="row-lbl">Target (TP)</span>
                <span class="row-val" style="color:{C_BULL}">${anal['target']:.2f}</span>
            </div>
            <div class="row-flex">
                <span class="row-lbl">Stop Loss (SL)</span>
                <span class="row-val" style="color:{C_BEAR}">${anal['stop']:.2f}</span>
            </div>
            
            <div style="border-top: 1px dashed #333; margin: 10px 0;"></div>
            
            <div class="row-flex">
                <span class="row-lbl">Expected Hit</span>
                <span class="row-val" style="color:{C_CYAN}">{exp_date}</span>
            </div>
        </div>

        <div class="section-head">📊 Key Drivers</div>
        <div class="grid-2">
            {cards_html}
        </div>

        <div class="section-head">📑 AI Comment</div>
        <div class="main-card" style="font-size: 13px; color: #d1d1d6; line-height: 1.6;">
            The AI analysis indicates a <b>{anal['cards'][0][1]}</b> scenario. 
            Whale activity is currently <b>{anal['cards'][1][1]}</b>. 
            With a volatility of {anal['cards'][3][1]}, the target price could be reached around <b>{exp_date}</b>.
        </div>
        
        <div style="height: 50px;"></div>
    </div>
    """
    return html

# ---------------------------------------------------------
# 7. 실행부
# ---------------------------------------------------------
st.title("🐋 Whale Hunter")
st.caption("Mobile Optimized AI Analyst")

ticker_input = st.text_input("", placeholder="Ticker (e.g. NVDA)", value="NVDA")

if st.button("ANALYZE", use_container_width=True):
    if not ticker_input:
        st.error("Please enter a ticker.")
    else:
        ticker = ticker_input.upper().split(',')[0].strip()
        status = st.empty()
        status.info(f"Scanning {ticker}...")
        
        try:
            # 데이터 로드
            df = get_clean_data(ticker)
            info = get_stock_info(ticker)
            
            if df is None or df.empty:
                status.error("Data not found.")
            else:
                # 분석 실행
                monte = run_monte_carlo(df)
                anal = analyze_stock(df, info, monte[0])
                
                # 결과 출력
                status.empty()
                html_code = make_html(ticker, info, anal, monte)
                st.markdown(html_code, unsafe_allow_html=True)
                
        except Exception as e:
            status.error(f"Error: {e}")
