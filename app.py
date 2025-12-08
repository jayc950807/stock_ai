import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import logging
import warnings
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. 설정 및 초기화 (반드시 맨 위에 있어야 함)
# ---------------------------------------------------------
st.set_page_config(page_title="Whale Hunter", layout="wide", page_icon="🐋")

logger = logging.getLogger('yfinance')
logger.setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 2. 모바일 최적화 CSS (전역 적용)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Streamlit 기본 패딩 제거 및 모바일 꽉 찬 화면 */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    
    /* 헤더, 푸터 숨김 */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 다크모드 강제 및 입력창 스타일 */
    .stApp {
        background-color: #000000;
        color: #f2f2f7;
    }
    input {
        color: #ffffff !important;
        background-color: #1c1c1e !important; 
        border: 1px solid #333 !important;
    }
    
    /* 버튼 스타일 */
    div.stButton > button {
        background-color: #0A84FF !important;
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. 데이터 및 상수
# ---------------------------------------------------------
REF_DATA = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'TSLA': 'Tesla',
    'JPM': 'JPMorgan', 'JNJ': 'Johnson&Johnson', 'KO': 'CocaCola',
    'PLTR': 'Palantir', 'SOFI': 'SoFi', 'COIN': 'Coinbase', 'AMC': 'AMC',
    'IWM': 'Russell2000', 'SPY': 'S&P500', 'QQQ': 'Nasdaq'
}
REFERENCE_TICKERS = list(REF_DATA.keys())

WINDOW_SIZE = 60
FORECAST_DAYS = 30
TOP_N = 5

C_BULL = "#30D158" # iOS Green
C_BEAR = "#FF453A" # iOS Red
C_NEUT = "#8E8E93" # iOS Gray
C_WARN = "#FFD60A" # iOS Yellow
C_CYAN = "#64D2FF" # iOS Cyan
C_PURP = "#BF5AF2" # iOS Purple

# ---------------------------------------------------------
# 4. 데이터 엔진 함수
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_stock_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        data = {
            'mkt_cap': info.get('marketCap', 0),
            'per': info.get('trailingPE', None),
            'pbr': info.get('priceToBook', None),
            'roe': info.get('returnOnEquity', None),
            'name': info.get('longName', ticker)
        }
        return data
    except:
        return {'mkt_cap': 0, 'per': None, 'pbr': None, 'roe': None, 'name': ticker}

@st.cache_data(ttl=1800)
def get_clean_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if df.empty or len(df) < WINDOW_SIZE + FORECAST_DAYS: return None

        # --- [기술적 지표 계산] ---
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA120'] = df['Close'].rolling(120).mean()

        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['Stoch_K'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

        df['WillR'] = ((high_14 - df['Close']) / (high_14 - low_14)) * -100

        std_20 = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['MA20'] + (std_20 * 2)
        df['BB_Lower'] = df['MA20'] - (std_20 * 2)

        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['KC_Upper'] = df['MA20'] + (df['ATR'] * 1.5)
        df['KC_Lower'] = df['MA20'] - (df['ATR'] * 1.5)

        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        ad_factor = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, 1)
        df['AD_Line'] = (ad_factor * df['Volume']).fillna(0).cumsum()

        typical = (df['High'] + df['Low'] + df['Close']) / 3
        mf = typical * df['Volume']
        df['MFI'] = 100 - (100 / (1 + (mf.where(typical > typical.shift(1), 0).rolling(14).sum() / mf.where(typical < typical.shift(1), 0).rolling(14).sum())))

        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).rolling(20).sum() / df['Volume'].rolling(20).sum()

        df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100

        nine_high = df['High'].rolling(window=9).max()
        nine_low = df['Low'].rolling(window=9).min()
        df['Tenkan'] = (nine_high + nine_low) / 2
        twenty_six_high = df['High'].rolling(window=26).max()
        twenty_six_low = df['Low'].rolling(window=26).min()
        df['Kijun'] = (twenty_six_high + twenty_six_low) / 2
        df['Senkou_Span_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
        fifty_two_high = df['High'].rolling(window=52).max()
        fifty_two_low = df['Low'].rolling(window=52).min()
        df['Senkou_Span_B'] = ((fifty_two_high + fifty_two_low) / 2).shift(26)

        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = (df['High'] - df['Low']) / df['Close'] * 100

        df.dropna(inplace=True)
        return df
    except: return None

@st.cache_data(ttl=3600)
def get_benchmark(mode):
    ticker = "SPY" if mode == "SAFE" else "IWM"
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

@st.cache_resource
def load_reference_cache():
    cache = {}
    for ticker in REFERENCE_TICKERS[:5]: 
        try:
            df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.get_level_values(0)
                except: pass
            if not df.empty and len(df) > 100:
                cache[ticker] = df
        except: pass
    return cache

# ---------------------------------------------------------
# 5. 분석 로직 함수들
# ---------------------------------------------------------
def check_ttm_squeeze(df):
    last = df.iloc[-1]
    bb_width = last['BB_Upper'] - last['BB_Lower']
    kc_width = last['KC_Upper'] - last['KC_Lower']
    return bb_width < kc_width

def check_candle_pattern(df):
    last = df.iloc[-1]
    open_p, close_p = last['Open'], last['Close']
    high_p, low_p = last['High'], last['Low']
    body = abs(close_p - open_p)
    total_range = high_p - low_p
    if total_range == 0: return None
    
    upper_shadow = high_p - max(open_p, close_p)
    lower_shadow = min(open_p, close_p) - low_p
    
    if (lower_shadow > body * 2) and (upper_shadow < body * 0.5) and (lower_shadow > upper_shadow * 2): return "Hammer"
    if body <= (total_range * 0.1): return "Doji"
    return None

def check_rsi_divergence(df, window=10):
    if len(df) < window * 2: return None
    current = df.iloc[-window:]
    prev = df.iloc[-window*2:-window]

    curr_low_price = current['Close'].min()
    prev_low_price = prev['Close'].min()
    curr_low_rsi = current.loc[current['Close'].idxmin()]['RSI']
    prev_low_rsi = prev.loc[prev['Close'].idxmin()]['RSI']

    curr_high_price = current['Close'].max()
    prev_high_price = prev['Close'].max()
    curr_high_rsi = current.loc[current['Close'].idxmax()]['RSI']
    prev_high_rsi = prev.loc[prev['Close'].idxmax()]['RSI']

    if curr_low_price < prev_low_price and curr_low_rsi > prev_low_rsi: return "REG_BULL"
    if curr_high_price > prev_high_price and curr_high_rsi < prev_high_rsi: return "REG_BEAR"
    return None

def get_18_tech_signals(df):
    last = df.iloc[-1]
    signals = []

    # SMA
    signals.append(("SMA 20", f"{last['MA20']:.2f}", "Bull" if last['Close'] > last['MA20'] else "Bear"))
    signals.append(("SMA 60", f"{last['MA60']:.2f}", "Bull" if last['Close'] > last['MA60'] else "Bear"))
    signals.append(("SMA 120", f"{last['MA120']:.2f}", "Bull" if last['Close'] > last['MA120'] else "Bear"))

    # Momentum
    rsi = last['RSI']
    bias = "Bear" if rsi > 70 else ("Bull" if rsi < 30 else "Neutral")
    signals.append(("RSI (14)", f"{rsi:.1f}", bias))

    macd = last['MACD']
    sig = last['MACD_Signal']
    signals.append(("MACD", f"{macd:.2f}", "Bull" if macd > sig else "Bear"))

    k = last['Stoch_K']
    d = last['Stoch_D']
    signals.append(("Stoch K/D", f"{k:.0f}/{d:.0f}", "Bull" if k > d else "Bear"))

    cci = last['CCI']
    bias = "Bear" if cci > 100 else ("Bull" if cci < -100 else "Neutral")
    signals.append(("CCI", f"{cci:.0f}", bias))

    wr = last['WillR']
    bias = "Bull" if wr < -80 else ("Bear" if wr > -20 else "Neutral")
    signals.append(("Will %R", f"{wr:.0f}", bias))

    # Volatility
    pos, bias = ("Mid", "Neutral")
    if last['Close'] > last['BB_Upper']: pos, bias = "Upper", "Bear"
    elif last['Close'] < last['BB_Lower']: pos, bias = "Lower", "Bull"
    signals.append(("Bollinger", pos, bias))

    signals.append(("ATR", f"{last['ATR']:.2f}", "Neutral"))

    obv_ma = df['OBV'].rolling(20).mean().iloc[-1]
    signals.append(("OBV Trend", "Up" if last['OBV'] > obv_ma else "Down", "Bull" if last['OBV'] > obv_ma else "Bear"))

    mfi = last['MFI']
    bias = "Bear" if mfi > 80 else ("Bull" if mfi < 20 else "Neutral")
    signals.append(("MFI", f"{mfi:.0f}", bias))

    signals.append(("VWAP", f"{last['VWAP']:.2f}", "Bull" if last['Close'] > last['VWAP'] else "Bear"))

    roc = last['ROC']
    signals.append(("ROC", f"{roc:.2f}%", "Bull" if roc > 0 else "Bear"))

    cloud_top = max(last['Senkou_Span_A'], last['Senkou_Span_B'])
    cloud_bot = min(last['Senkou_Span_A'], last['Senkou_Span_B'])
    ichi, bias = "In Cloud", "Neutral"
    if last['Close'] > cloud_top: ichi, bias = "Above", "Bull"
    elif last['Close'] < cloud_bot: ichi, bias = "Below", "Bear"
    signals.append(("Ichimoku", ichi, bias))

    sqz = check_ttm_squeeze(df)
    signals.append(("TTM Squeeze", "ON" if sqz else "OFF", "Bull" if sqz else "Neutral"))

    pat = check_candle_pattern(df)
    signals.append(("Candle", pat if pat else "-", "Bull" if pat == "Hammer" else "Neutral"))

    vol = last['Volatility']
    signals.append(("Vol Ratio", f"{vol:.2f}%", "Neutral"))

    return signals

def run_monte_carlo(df, num_simulations=1000, days=120):
    np.random.seed(42) 
    last_price = df['Close'].iloc[-1]
    target_price = last_price * 1.30
    
    if len(df) < 30: daily_vol = df['Log_Ret'].std()
    else: daily_vol = df['Log_Ret'].tail(30).std()
    
    sim_df = pd.DataFrame()
    max_peaks = []

    # 벡터화 연산 대신 루프로 간단 구현 (데모용)
    for x in range(num_simulations):
        price_series = [last_price]
        price = last_price
        for y in range(days):
            price = price * (1 + daily_vol * np.random.normal())
            price_series.append(price)
        sim_df[x] = price_series
        max_peaks.append(np.max(price_series))
    
    sim_maxes = sim_df.max() 
    win_count = (sim_maxes >= target_price).sum()
    win_prob = (win_count / num_simulations) * 100
    
    hit_days = []
    winning_peaks = []

    for col in sim_df.columns:
        if sim_df[col].max() >= target_price:
            hits = sim_df.index[sim_df[col] >= target_price].tolist()
            if hits: hit_days.append(hits[0])
            winning_peaks.append(sim_df[col].max())
            
    if hit_days:
        avg_days_needed = int(np.mean(hit_days))
        future_date = datetime.now() + timedelta(days=avg_days_needed)
        expected_date_str = future_date.strftime("%Y-%m-%d")
    else:
        expected_date_str = "N/A"

    if winning_peaks: target_peak_price = np.median(winning_peaks)
    else: target_peak_price = np.median(max_peaks)
        
    peak_yield = (target_peak_price - last_price) / last_price * 100
    ending = sim_df.iloc[-1, :]
    
    return sim_df, np.percentile(ending, 90), np.percentile(ending, 10), np.mean(ending), win_prob, expected_date_str, peak_yield

def calculate_kelly(win_rate, reward_risk_ratio):
    p = win_rate / 100
    q = 1 - p
    b = reward_risk_ratio
    if b <= 0: return 0
    kelly_fraction = p - (q / b)
    safe_kelly = max(0, kelly_fraction * 0.5)
    return safe_kelly * 100

def analyze_whale_mode(df, benchmark_df, win_rate, avg_return, stock_info, monte_prob):
    last = df.iloc[-1]
    close = last['Close']
    atr = last['ATR']
    volatility = last['Volatility']
    mkt_cap = stock_info['mkt_cap']

    # --- Metrics ---
    recent_20 = df.iloc[-20:]
    price_rank = (close - recent_20['Close'].min()) / (recent_20['Close'].max() - recent_20['Close'].min() + 1e-9) * 100
    obv_rank = (last['OBV'] - recent_20['OBV'].min()) / (recent_20['OBV'].max() - recent_20['OBV'].min() + 1e-9) * 100
    whale_gap = obv_rank - price_rank

    ad_trend = df['AD_Line'].diff(20).iloc[-1]
    price_trend_val = df['Close'].diff(20).iloc[-1]
    ad_signal = "Neut"
    if price_trend_val < 0 and ad_trend > 0: ad_signal = "Bull"
    elif price_trend_val > 0 and ad_trend < 0: ad_signal = "Bear"

    vp_window = df.iloc[-60:]
    hist, bins = np.histogram(vp_window['Close'], bins=30, weights=vp_window['Volume'])
    poc_idx = hist.argmax()
    poc_price = (bins[poc_idx] + bins[poc_idx+1]) / 2
    
    poc_signal = "Supp"
    if close > poc_price * 1.02: poc_signal = "Bull"
    elif close < poc_price * 0.98: poc_signal = "Bear"

    mfi_val = last['MFI']
    mfi_signal = "Neut"
    if mfi_val < 20: mfi_signal = "Oversold"
    elif mfi_val > 80: mfi_signal = "Overbot"

    # --- Scoring ---
    score = 50
    cards = []
    red_flags = 0

    # 1. Fundamentals
    per, roe = stock_info['per'], stock_info['roe']
    if per and roe:
        if per < 25 and roe > 0.10: score += 15; cards.append({'title':'Fundamentals','stat':'Undervalued','col':C_CYAN})
        elif roe > 0.15: score += 10; cards.append({'title':'Fundamentals','stat':'High Growth','col':C_BULL})
        elif per > 80: score -= 10; cards.append({'title':'Fundamentals','stat':'Overvalued','col':C_WARN})
        else: cards.append({'title':'Fundamentals','stat':'Normal','col':C_NEUT})
    else: cards.append({'title':'Fundamentals','stat':'No Data','col':C_NEUT})

    # 2. Whale Gap
    if whale_gap > 30: score += 20; cards.append({'title':'Whale Gap','stat':'Strong Buy','col':C_BULL})
    elif whale_gap > 10: score += 10; cards.append({'title':'Whale Gap','stat':'Accumulation','col':C_CYAN})
    elif whale_gap < -10: 
        score -= 15; red_flags += 1
        cards.append({'title':'Whale Gap','stat':'Distribution','col':C_BEAR})
    else: cards.append({'title':'Whale Gap','stat':'Neutral','col':C_NEUT})

    # 3. Squeeze
    if check_ttm_squeeze(df): score += 15; cards.append({'title':'Volatility','stat':'Squeeze ON','col':C_PURP})
    else: cards.append({'title':'Volatility','stat':'Normal','col':C_NEUT})
    
    # 4. Divergence
    div_status = check_rsi_divergence(df)
    if div_status == "REG_BULL": score += 20; cards.append({'title':'Divergence','stat':'Bullish','col':C_BULL})
    elif div_status == "REG_BEAR": score -= 20; cards.append({'title':'Divergence','stat':'Bearish','col':C_BEAR})
    else: cards.append({'title':'Divergence','stat':'None','col':C_NEUT})

    # 5. Candle
    pat = check_candle_pattern(df)
    if pat == "Hammer": score += 10; cards.append({'title':'Pattern','stat':'Hammer','col':C_WARN})
    else: cards.append({'title':'Pattern','stat':'-','col':C_NEUT})

    # 6. Ichimoku
    c_top = max(last['Senkou_Span_A'], last['Senkou_Span_B'])
    c_bot = min(last['Senkou_Span_A'], last['Senkou_Span_B'])
    if close > c_top: score += 10; cards.append({'title':'Ichimoku','stat':'Above Cloud','col':C_CYAN})
    elif close < c_bot: score -= 10; cards.append({'title':'Ichimoku','stat':'Below Cloud','col':C_BEAR})
    else: cards.append({'title':'Ichimoku','stat':'In Cloud','col':C_NEUT})

    # 7. Trend
    if close > last['MA20']: score += 10; cards.append({'title':'Trend (MA20)','stat':'Up','col':C_BULL})
    else: score -= 15; cards.append({'title':'Trend (MA20)','stat':'Down','col':C_BEAR})

    # 8. Monte Carlo
    if monte_prob >= 40: score += 10; cards.append({'title':'Probability','stat':'High (>40%)','col':C_BULL})
    elif monte_prob <= 10: score -= 10; cards.append({'title':'Probability','stat':'Low (<10%)','col':C_BEAR})
    else: cards.append({'title':'Probability','stat':'Medium','col':C_NEUT})

    if ad_signal == "Bull": score += 15
    elif ad_signal == "Bear": score -= 15; red_flags += 1
    if poc_signal == "Bull": score += 10
    elif poc_signal == "Bear": score -= 10; red_flags += 1
    if mfi_signal == "Oversold": score += 10

    if red_flags > 0: score = min(score, 65)
    score = max(0, min(100, int(score)))

    # Mode & Targets
    if mkt_cap < 10_000_000_000 or volatility > 3.0:
        mode_txt, theme_col = "BEAST", C_PURP
        stop_mult, target_mult = 2.5, 5.0
    else:
        mode_txt, theme_col = "SAFE", C_CYAN
        stop_mult, target_mult = 2.0, 3.0

    stop = close - (atr * stop_mult)
    target = close + (atr * target_mult)

    if score >= 80: t, c = "STRONG BUY", C_BULL
    elif score >= 60: 
        if red_flags > 0: t, c = "CAUTION", C_WARN
        else: t, c = "BUY", C_CYAN
    elif score <= 30: t, c = "SELL", C_BEAR
    else: t, c = "HOLD", C_NEUT

    vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
    vol_ratio = (last['Volume'] / vol_avg) * 100

    return {
        'mode': mode_txt, 'theme': theme_col, 'score': score,
        'title': t, 'color': c,
        'cards': cards, 'tech_signals': get_18_tech_signals(df),
        'stop': stop, 'target': target, 'close': close,
        'vol_data': {'last': last['Volume'], 'avg': vol_avg, 'ratio': vol_ratio},
        'adv_features': {'whale_gap': whale_gap, 'ad_signal': ad_signal, 'poc_signal': poc_signal, 'mfi_signal': mfi_signal, 'poc_price': poc_price},
        'monte_prob': monte_prob 
    }

def get_score_history(df, bench_df, win_rate, avg_ret, stock_info):
    history = []
    for i in range(9, -1, -1):
        if i == 0: sliced_df = df
        else: sliced_df = df.iloc[:-i]
        
        sim_res = run_monte_carlo(sliced_df, num_simulations=100, days=120) 
        res = analyze_whale_mode(sliced_df, bench_df, win_rate, avg_ret, stock_info, sim_res[4])
        label = sliced_df.index[-1].strftime('%m-%d')
        history.append({'day': label, 'score': res['score'], 'adv': res['adv_features']})
    return history

def generate_ai_report_text(ticker, analysis, stock_info, score_history, expected_date_str, peak_yield):
    try: days_str = f"{(datetime.strptime(expected_date_str, '%Y-%m-%d') - datetime.now()).days} days"
    except: days_str = "Unknown"
    
    html = f"""
    <ul style="margin:0; padding-left:15px;">
        <li>Current AI Score is <b>{analysis['score']}</b>.</li>
        <li>Volatility indicates potential target hit in <b>{days_str}</b>.</li>
        <li>Max upside potential estimated at <b>+{peak_yield:.1f}%</b>.</li>
        <li>Volume is <b>{analysis['vol_data']['ratio']:.0f}%</b> of average.</li>
    </ul>
    """
    return html

# ---------------------------------------------------------
# 6. 모바일 UI 렌더링 (HTML 생성)
# ---------------------------------------------------------
def get_render_html(ticker, mkt_cap, analysis, monte_res, score_history, stock_info):
    sim_df, opt, pes, mean, win_prob, expected_date_str, peak_yield = monte_res

    # 시가총액
    if mkt_cap > 0:
        val_won = mkt_cap * 1400 
        if val_won > 100_000_000_000_000: cap_str = f"{val_won/100_000_000_000_000:.1f}T KRW"
        elif val_won > 1_000_000_000_000: cap_str = f"{val_won/1_000_000_000_000:.1f}T KRW"
        else: cap_str = f"{val_won/100_000_000_000:.0f}B KRW"
    else: cap_str = "-"

    peak_color = C_PURP if peak_yield > 50 else (C_BULL if peak_yield > 0 else C_BEAR)
    peak_str = f"MAX +{peak_yield:.0f}%" if peak_yield > 0 else f"MAX {peak_yield:.0f}%"

    # History Table
    dates = [item['day'] for item in score_history]
    scores = [item['score'] for item in score_history]
    gaps = [item['adv']['whale_gap'] for item in score_history]
    ads = [item['adv']['ad_signal'] for item in score_history]
    mfis = [item['adv']['mfi_signal'] for item in score_history]

    def get_style_content(label, v):
        txt, col, bg, fw = "-", "#666", "transparent", "normal"
        if label.startswith("Date"): return v, "#888", "transparent", "normal"
        elif label == "Score":
            txt = str(v)
            if v >= 80: col, fw = C_BULL, "bold"
            elif v >= 60: col, fw = C_CYAN, "bold"
            elif v <= 40: col, fw = C_BEAR, "bold"
            else: col = C_NEUT
        elif label == "Whale":
            if v > 10: txt, col, bg, fw = "BUY", C_BULL, "rgba(48, 209, 88, 0.1)", "bold"
            elif v < -10: txt, col, bg, fw = "SELL", C_BEAR, "rgba(255, 69, 58, 0.1)", "bold"
            else: txt = "WAIT"
        elif label == "Smart":
            if v == "Bull": txt, col, bg, fw = "BUY", C_BULL, "rgba(48, 209, 88, 0.1)", "bold"
            elif v == "Bear": txt, col, bg, fw = "SELL", C_BEAR, "rgba(255, 69, 58, 0.1)", "bold"
            else: txt = "WAIT"
        elif label == "MFI":
            if v == "Oversold": txt, col, bg, fw = "BUY", C_BULL, "rgba(48, 209, 88, 0.1)", "bold"
            elif v == "Overbot": txt, col, bg, fw = "SELL", C_BEAR, "rgba(255, 69, 58, 0.1)", "bold"
            else: txt = "WAIT"
        return txt, col, bg, fw

    def make_row_html(label, values, is_header=False):
        row_html = f"<tr><td style='position:sticky; left:0; background:#1c1c1e; z-index:10; text-align:left; color:#999; font-size:12px; padding:8px 10px; border-right:1px solid #333; width:60px;'>{label}</td>"
        for v in values:
            txt, col, bg, fw = get_style_content(label, v)
            if is_header: row_html += f"<td style='color:#bbb; font-size:11px; padding:6px 4px; min-width:35px;'>{txt}</td>"
            else: row_html += f"<td style='color:{col}; background:{bg}; font-weight:{fw}; font-size:11px; padding:6px 4px; border-radius:4px;'>{txt}</td>"
        row_html += "</tr>"
        return row_html

    hist_table = """<div style="overflow-x:auto; -webkit-overflow-scrolling:touch; margin-top:10px; padding-bottom:5px;">
    <table style="width:100%; border-collapse:collapse; text-align:center; white-space:nowrap;">"""
    hist_table += make_row_html("Date", dates, is_header=True)
    hist_table += make_row_html("Score", scores)
    hist_table += make_row_html("Whale", gaps)
    hist_table += make_row_html("Smart", ads)
    hist_table += make_row_html("MFI", mfis)
    hist_table += "</table></div>"
    
    # Cards
    cards_html = "<div style='display:grid; grid-template-columns: 1fr 1fr; gap:8px;'>"
    for c in analysis['cards']:
        cards_html += f"""
        <div style="background:#2c2c2e; padding:12px 10px; border-radius:12px; border-left:3px solid {c['col']}; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:11px; color:#8e8e93; margin-bottom:2px;">{c['title']}</div>
            <div style="font-size:13px; font-weight:600; color:#f2f2f7;">{c['stat']}</div>
        </div>"""
    cards_html += "</div>"

    # Tech Signals
    def make_tech_table(start, end):
        t_html = "<table style='width:100%; border-collapse:collapse;'>"
        for i in range(start, end):
            name, val, bias = analysis['tech_signals'][i]
            tc = C_BULL if bias == "Bull" else (C_BEAR if bias == "Bear" else C_NEUT)
            fw = "700" if bias != "Neutral" else "400"
            t_html += f"<tr><td style='padding:6px 0; color:#8e8e93; font-size:12px;'>{name}</td><td style='text-align:right; color:{tc}; font-weight:{fw}; font-size:12px;'>{val}</td></tr>"
        t_html += "</table>"
        return t_html

    tech_html = f"""
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; background:#2c2c2e; padding:15px; border-radius:12px;">
        <div>{make_tech_table(0, 9)}</div>
        <div style="border-left:1px solid #48484a; padding-left:15px;">{make_tech_table(9, 18)}</div>
    </div>
    """

    report_text = generate_ai_report_text(ticker, analysis, stock_info, score_history, expected_date_str, peak_yield)
    
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;600;800&display=swap');
        .mobile-container {{
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #000000;
            color: #f2f2f7;
            padding: 0px 5px 40px 5px;
            max-width: 100%;
            overflow-x: hidden;
        }}
        .header-section {{
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding: 10px 5px; border-bottom: 1px solid #333;
        }}
        .ticker-title {{ font-size: 32px; font-weight: 800; letter-spacing: -0.5px; margin: 0; line-height: 1.1; }}
        .ticker-info {{ font-size: 13px; color: #8e8e93; margin-top: 4px; }}
        .score-badge {{
            text-align: center; background: #1c1c1e; padding: 8px 16px; border-radius: 16px; border: 1px solid #333;
        }}
        .score-val {{ font-size: 28px; font-weight: 800; color: {analysis['color']}; line-height: 1; }}
        .score-lbl {{ font-size: 10px; color: #636366; font-weight: 600; margin-top: 2px; }}
        .section-title {{
            font-size: 17px; font-weight: 700; color: #f2f2f7; margin: 25px 0 10px 0; letter-spacing: -0.3px;
        }}
        .card-box {{
            background: #1c1c1e; border-radius: 14px; padding: 16px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .strategy-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 14px; }}
        .strategy-lbl {{ color: #8e8e93; }}
        .strategy-val {{ font-weight: 600; }}
        .ai-report li {{ margin-bottom: 8px; font-size: 14px; color: #d1d1d6; line-height: 1.5; }}
        .ai-report b {{ color: #fff; font-weight: 600; }}
    </style>
    """

    html = f"""
    {css}
    <div class="mobile-container">
        <div class="header-section">
            <div>
                <div class="ticker-title">{ticker}</div>
                <div class="ticker-info">{stock_info.get('name', '')}<br>{cap_str} • <span style="color:{analysis['theme']}">{analysis['mode']}</span></div>
            </div>
            <div class="score-badge">
                <div class="score-val">{analysis['score']}</div>
                <div class="score-lbl">AI SCORE</div>
            </div>
        </div>

        <div style="display:flex; gap:10px; margin-bottom:15px;">
            <div style="flex:1; background:#1c1c1e; padding:12px; border-radius:12px; text-align:center;">
                <div style="font-size:11px; color:#8e8e93;">Win Prob</div>
                <div style="font-size:20px; font-weight:800; color:{C_BULL if win_prob>=40 else '#fff'};">{win_prob:.0f}%</div>
            </div>
            <div style="flex:1; background:#1c1c1e; padding:12px; border-radius:12px; text-align:center;">
                <div style="font-size:11px; color:#8e8e93;">Peak Potential</div>
                <div style="font-size:20px; font-weight:800; color:{peak_color};">{peak_str}</div>
            </div>
        </div>

        <div class="card-box" style="border: 1px solid {analysis['color']}44;">
            <div style="font-size:15px; font-weight:700; margin-bottom:15px; color:#fff;">🎯 Trading Setup</div>
            <div class="strategy-row">
                <span class="strategy-lbl">Entry (Current)</span>
                <span class="strategy-val">${analysis['close']:.2f}</span>
            </div>
            <div class="strategy-row">
                <span class="strategy-lbl">Target Price</span>
                <span class="strategy-val" style="color:{C_BULL}">${analysis['target']:.2f}</span>
            </div>
            <div class="strategy-row">
                <span class="strategy-lbl">Stop Loss</span>
                <span class="strategy-val" style="color:{C_BEAR}">${analysis['stop']:.2f}</span>
            </div>
             <div style="margin-top:15px; padding-top:12px; border-top:1px dashed #3a3a3c;">
                <div class="strategy-row" style="margin-bottom:0;">
                    <span class="strategy-lbl">Expected Date</span>
                    <span class="strategy-val" style="color:{C_CYAN}">{expected_date_str}</span>
                </div>
            </div>
        </div>

        <div class="section-title">📊 8 Key Drivers</div>
        {cards_html}

        <div class="section-title">📑 AI Analysis</div>
        <div class="card-box ai-report">
            {report_text}
        </div>

        <div class="section-title">📈 Momentum Trend</div>
        {hist_table}
        <div style="font-size:10px; color:#666; text-align:right; margin-top:4px;">*Scroll horizontally</div>

        <div class="section-title">🎛 Technical Details</div>
        {tech_html}
        
        <div style="height:30px;"></div>
    </div>
    """
    return html

# ---------------------------------------------------------
# 7. 실행 (Main)
# ---------------------------------------------------------
st.title("🐋 Whale Hunter")
st.caption("AI-Powered Technical Analysis")

ref_cache = load_reference_cache()
ticker_input = st.text_input("", placeholder="Ticker (e.g., NVDA, TSLA)", value="NVDA")

if st.button("Analyze", type="primary", use_container_width=True):
    if not ticker_input:
        st.warning("Please enter a ticker.")
    else:
        spy_df = get_benchmark("SAFE")
        iwm_df = get_benchmark("GROWTH")
        
        tickers = [t.strip().upper() for t in ticker_input.split(',')]
        
        for ticker in tickers:
            if not ticker: continue
            
            status_text = st.empty()
            status_text.info(f"Analyzing {ticker}...")
            
            try:
                stock_info = get_stock_info(ticker)
                mkt_cap = stock_info['mkt_cap']
                target_df = get_clean_data(ticker)
                
                if target_df is None:
                    status_text.error(f"Failed to load data for {ticker}")
                    continue
                    
                target_df.name = ticker
                volatility = target_df['Volatility'].iloc[-1]
                bench_df = iwm_df if (mkt_cap < 10_000_000_000 or volatility > 3.0) else spy_df
                
                avg_ret = 0
                win_rate = 50 

                monte_res = run_monte_carlo(target_df)
                analysis = analyze_whale_mode(target_df, bench_df, win_rate, avg_ret, stock_info, monte_res[4])
                score_history = get_score_history(target_df, bench_df, win_rate, avg_ret, stock_info)
                
                html_out = get_render_html(ticker, mkt_cap, analysis, monte_res, score_history, stock_info)
                status_text.empty() 
                
                st.markdown(html_out, unsafe_allow_html=True)
                st.markdown("---")
                
            except Exception as e:
                status_text.error(f"Error: {str(e)}")
