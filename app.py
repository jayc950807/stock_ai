import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime, timedelta

# 1. 설정 및 초기화
st.set_page_config(page_title="WQA | Market Intelligence", layout="wide", page_icon="📊", initial_sidebar_state="collapsed")
logger = logging.getLogger('yfinance')
logger.setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# --- [공통 디자인 CSS] ---
# 이 CSS는 화면 출력과 다운로드 파일 양쪽에 모두 적용됩니다.
COMMON_CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        background-color: #050505 !important; 
        color: #E5E7EB;
        margin: 0;
        padding: 0;
    }
    
    .report-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #050505;
    }
    
    /* 헤더 스타일 */
    .header-main {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 25px;
        border-bottom: 1px solid #333;
        padding-bottom: 15px;
    }
    
    /* 카드 박스 스타일 */
    .box-dark {
        background: #121212;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
    }
    
    /* 그리드 시스템 */
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .grid-responsive { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; }
    
    /* 텍스트 유틸리티 */
    .text-bull { color: #00E676; font-weight: bold; }
    .text-bear { color: #FF5252; font-weight: bold; }
    .text-cyan { color: #00B0FF; font-weight: bold; }
    .text-purp { color: #D500F9; font-weight: bold; }
    .text-warn { color: #FFD740; font-weight: bold; }
    .text-neut { color: #78909C; font-weight: normal; }
    
    /* 스크롤 테이블 */
    .scroll-x {
        display: flex;
        overflow-x: auto;
        padding-bottom: 5px;
    }
    .scroll-x::-webkit-scrollbar { height: 4px; }
    .scroll-x::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

    /* Streamlit 고유 UI 덮어쓰기 */
    .stApp { background-color: #050505; }
    header { visibility: hidden; }
    .block-container { padding-top: 1rem; padding-bottom: 5rem; max-width: 1000px; }
    .stTextInput > div > div > input {
        background-color: #1A1A1A; color: #fff; border: 1px solid #333; border-radius: 8px; height: 50px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0066FF 0%, #00B0FF 100%);
        color: white; border: none; height: 50px; font-weight: 700; border-radius: 8px; width: 100%;
    }
"""

st.markdown(f"<style>{COMMON_CSS}</style>", unsafe_allow_html=True)

# 2. 참조 데이터 및 상수
WINDOW_SIZE = 60
FORECAST_DAYS = 30

C_BULL = "#00E676"
C_BEAR = "#FF5252"
C_NEUT = "#78909C"
C_WARN = "#FFD740"
C_CYAN = "#00B0FF"
C_PURP = "#D500F9"
C_DARK = "#121212"

# 3. 데이터 엔진
@st.cache_data(ttl=3600)
def get_stock_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'mkt_cap': info.get('marketCap', 0),
            'per': info.get('trailingPE', None),
            'pbr': info.get('priceToBook', None),
            'roe': info.get('returnOnEquity', None),
            'name': info.get('longName', ticker)
        }
    except: return {'mkt_cap': 0, 'per': None, 'pbr': None, 'roe': None, 'name': ticker}

@st.cache_data(ttl=1800)
def get_clean_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if df.empty or len(df) < WINDOW_SIZE + FORECAST_DAYS: return None

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

        # 일목균형표
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

# 분석 로직
def get_18_tech_signals(df):
    last = df.iloc[-1]
    signals = []
    
    signals.append(("SMA 20", f"{last['MA20']:.2f}", "Bull" if last['Close'] > last['MA20'] else "Bear"))
    signals.append(("SMA 60", f"{last['MA60']:.2f}", "Bull" if last['Close'] > last['MA60'] else "Bear"))
    signals.append(("SMA 120", f"{last['MA120']:.2f}", "Bull" if last['Close'] > last['MA120'] else "Bear"))
    
    rsi = last['RSI']
    bias = "Bear" if rsi > 70 else ("Bull" if rsi < 30 else "Neutral")
    signals.append(("RSI (14)", f"{rsi:.1f}", bias))

    macd = last['MACD']
    sig = last['MACD_Signal']
    signals.append(("MACD", f"{macd:.2f}/{sig:.2f}", "Bull" if macd > sig else "Bear"))

    k = last['Stoch_K']
    d = last['Stoch_D']
    signals.append(("Stoch", f"{k:.0f}/{d:.0f}", "Bull" if k > d else "Bear"))

    cci = last['CCI']
    bias = "Bear" if cci > 100 else ("Bull" if cci < -100 else "Neutral")
    signals.append(("CCI", f"{cci:.0f}", bias))

    wr = last['WillR']
    bias = "Bull" if wr < -80 else ("Bear" if wr > -20 else "Neutral")
    signals.append(("Will %R", f"{wr:.0f}", bias))

    pos, bias = "Mid", "Neutral"
    if last['Close'] > last['BB_Upper']: pos, bias = "High", "Bear"
    elif last['Close'] < last['BB_Lower']: pos, bias = "Low", "Bull"
    signals.append(("Bollinger", pos, bias))

    signals.append(("ATR", f"{last['ATR']:.2f}", "Neutral"))
    
    obv_ma = df['OBV'].rolling(20).mean().iloc[-1]
    signals.append(("OBV", "Up" if last['OBV'] > obv_ma else "Down", "Bull" if last['OBV'] > obv_ma else "Bear"))

    mfi = last['MFI']
    bias = "Bear" if mfi > 80 else ("Bull" if mfi < 20 else "Neutral")
    signals.append(("MFI", f"{mfi:.0f}", bias))

    signals.append(("VWAP", f"{last['VWAP']:.2f}", "Bull" if last['Close'] > last['VWAP'] else "Bear"))
    
    roc = last['ROC']
    signals.append(("ROC", f"{roc:.2f}%", "Bull" if roc > 0 else "Bear"))

    c_top = max(last['Senkou_Span_A'], last['Senkou_Span_B'])
    c_bot = min(last['Senkou_Span_A'], last['Senkou_Span_B'])
    ichi, bias = "Cloud", "Neutral"
    if last['Close'] > c_top: ichi, bias = "Above", "Bull"
    elif last['Close'] < c_bot: ichi, bias = "Below", "Bear"
    signals.append(("Ichimoku", ichi, bias))

    sqz = check_ttm_squeeze(df)
    signals.append(("Squeeze", "ON" if sqz else "OFF", "Bull" if sqz else "Neutral"))

    pat = check_candle_pattern(df)
    signals.append(("Candle", pat if pat else "None", "Bull" if pat == "Hammer" else "Neutral"))

    vol = last['Volatility']
    signals.append(("Vol Ratio", f"{vol:.2f}%", "Neutral"))

    return signals

def check_rsi_divergence(df, window=10):
    if len(df) < window * 2: return None
    current = df.iloc[-window:]
    prev = df.iloc[-window*2:-window]
    
    if current.empty or prev.empty: return None

    curr_low_price = current['Close'].min()
    prev_low_price = prev['Close'].min()
    curr_low_rsi = current.loc[current['Close'].idxmin()]['RSI']
    prev_low_rsi = prev.loc[prev['Close'].idxmin()]['RSI']

    if curr_low_price < prev_low_price and curr_low_rsi > prev_low_rsi: return "REG_BULL"
    
    curr_high_price = current['Close'].max()
    prev_high_price = prev['Close'].max()
    curr_high_rsi = current.loc[current['Close'].idxmax()]['RSI']
    prev_high_rsi = prev.loc[prev['Close'].idxmax()]['RSI']

    if curr_high_price > prev_high_price and curr_high_rsi < prev_high_rsi: return "REG_BEAR"
    return None

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
    upper_shadow = high_p - max(open_p, close_p)
    lower_shadow = min(open_p, close_p) - low_p
    total_range = high_p - low_p
    if total_range == 0: return None
    if (lower_shadow > body * 2) and (upper_shadow < body * 0.5): return "Hammer"
    if body <= (total_range * 0.1): return "Doji"
    return None

def run_monte_carlo(df, num_simulations=1000, days=120):
    np.random.seed(42) 
    last_price = df['Close'].iloc[-1]
    target_price = last_price * 1.30
    
    if len(df) < 30: daily_vol = df['Log_Ret'].std()
    else: daily_vol = df['Log_Ret'].tail(30).std()
    
    sim_df = pd.DataFrame()
    max_peaks = []

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
        expected_date_str = "도달 불가"

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

def analyze_whale_mode(df, benchmark_df, stock_info, monte_prob):
    last = df.iloc[-1]
    close = last['Close']
    atr = last['ATR']
    volatility = last['Volatility']
    mkt_cap = stock_info['mkt_cap']

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

    score = 50
    cards = []
    red_flags = 0

    per, roe = stock_info['per'], stock_info['roe']
    if per and roe:
        if per < 25 and roe > 0.10: score += 15; cards.append({'title':'FUNDAMENTAL','stat':'UNDERVALUED','desc':f'PER {per:.1f}', 'col':C_CYAN})
        elif roe > 0.15: score += 10; cards.append({'title':'FUNDAMENTAL','stat':'HIGH ROE','desc':f'ROE {roe*100:.1f}%', 'col':C_BULL})
        elif per > 80: score -= 10; cards.append({'title':'FUNDAMENTAL','stat':'OVERVALUED','desc':f'PER {per:.1f}', 'col':C_WARN})
        else: cards.append({'title':'FUNDAMENTAL','stat':'FAIR','desc':'No Signal', 'col':C_NEUT})
    else: cards.append({'title':'FUNDAMENTAL','stat':'NO DATA','desc':'Missing Info', 'col':C_NEUT})

    if whale_gap > 30: score += 20; cards.append({'title':'WHALE FLOW','stat':'ACCUMULATION','desc':'Smart Money In', 'col':C_BULL})
    elif whale_gap > 10: score += 10; cards.append({'title':'WHALE FLOW','stat':'INFLOW','desc':'Buying Detected', 'col':C_CYAN})
    elif whale_gap < -10: 
        score -= 15; red_flags += 1
        cards.append({'title':'WHALE FLOW','stat':'DISTRIBUTION','desc':'Selling Pressure', 'col':C_BEAR})
    else: cards.append({'title':'WHALE FLOW','stat':'NEUTRAL','desc':'Balanced', 'col':C_NEUT})

    if check_ttm_squeeze(df): score += 15; cards.append({'title':'VOLATILITY','stat':'SQUEEZE ON','desc':'Explosion Soon', 'col':C_PURP})
    else: cards.append({'title':'VOLATILITY','stat':'NORMAL','desc':'Building Up', 'col':C_NEUT})
    
    div_status = check_rsi_divergence(df)
    if div_status == "REG_BULL": score += 20; cards.append({'title':'DIVERGENCE','stat':'BULLISH','desc':'Trend Reversal', 'col':C_BULL})
    elif div_status == "REG_BEAR": score -= 20; cards.append({'title':'DIVERGENCE','stat':'BEARISH','desc':'Top Sign', 'col':C_BEAR})
    else: cards.append({'title':'DIVERGENCE','stat':'NONE','desc':'Synced', 'col':C_NEUT})

    pat = check_candle_pattern(df)
    if pat == "Hammer": score += 10; cards.append({'title':'PATTERN','stat':'HAMMER','desc':'Bounce Likely', 'col':C_WARN})
    elif pat == "Doji": cards.append({'title':'PATTERN','stat':'DOJI','desc':'Indecision', 'col':C_NEUT})
    else: cards.append({'title':'PATTERN','stat':'NONE','desc':'No Pattern', 'col':C_NEUT})

    c_top = max(last['Senkou_Span_A'], last['Senkou_Span_B'])
    c_bot = min(last['Senkou_Span_A'], last['Senkou_Span_B'])
    if close > c_top: score += 10; cards.append({'title':'ICHIMOKU','stat':'ABOVE CLOUD','desc':'Bull Trend', 'col':C_CYAN})
    elif close < c_bot: score -= 10; cards.append({'title':'ICHIMOKU','stat':'BELOW CLOUD','desc':'Resistance', 'col':C_BEAR})
    else: cards.append({'title':'ICHIMOKU','stat':'IN CLOUD','desc':'Choppy', 'col':C_NEUT})

    if close > last['MA20']: 
        score += 10
        cards.append({'title':'TREND','stat':'UPTREND','desc':'Above MA20', 'col':C_BULL})
    else: 
        score -= 15
        cards.append({'title':'TREND','stat':'DOWNTREND','desc':'Below MA20', 'col':C_BEAR})

    if monte_prob >= 40: score += 10; cards.append({'title':'PROBABILITY','stat':f'{monte_prob:.0f}% (>30%)','desc':'High Chance', 'col':C_BULL})
    elif monte_prob <= 10: score -= 10; cards.append({'title':'PROBABILITY','stat':f'{monte_prob:.0f}% (>30%)','desc':'Low Chance', 'col':C_BEAR})
    else: cards.append({'title':'PROBABILITY','stat':f'{monte_prob:.0f}% (>30%)','desc':'Moderate', 'col':C_NEUT})

    if ad_signal == "Bull": score += 15
    elif ad_signal == "Bear": score -= 15; red_flags += 1
    if poc_signal == "Bull": score += 10
    elif poc_signal == "Bear": score -= 10; red_flags += 1
    if mfi_signal == "Oversold": score += 10

    if red_flags > 0: score = min(score, 65)
    score = max(0, min(100, int(score)))

    if mkt_cap < 10_000_000_000 or volatility > 3.0:
        mode_txt, theme_col = "AGGRESSIVE", C_PURP
        stop_mult, target_mult = 2.5, 5.0
    else:
        mode_txt, theme_col = "DEFENSIVE", C_CYAN
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
        'kelly': calculate_kelly(monte_prob, (target-close)/(close-stop) if close>stop else 1),
        'vol_data': {'last': last['Volume'], 'avg': vol_avg, 'ratio': vol_ratio},
        'adv_features': {'whale_gap': whale_gap, 'ad_signal': ad_signal, 'poc_signal': poc_signal, 'mfi_signal': mfi_signal, 'poc_price': poc_price},
        'monte_prob': monte_prob 
    }

def get_score_history(df, bench_df, stock_info):
    history = []
    for i in range(9, -1, -1):
        if i == 0: sliced_df, sliced_bench = df, bench_df
        else: sliced_df, sliced_bench = df.iloc[:-i], bench_df.iloc[:-i]

        if len(sliced_bench) > len(sliced_df): sliced_bench = sliced_bench.iloc[-len(sliced_df):]
        label = sliced_df.index[-1].strftime('%m-%d')
        sim_res = run_monte_carlo(sliced_df, num_simulations=100, days=120) 
        res = analyze_whale_mode(sliced_df, sliced_bench, stock_info, sim_res[4])
        history.append({'day': label, 'score': res['score'], 'adv': res['adv_features']})
    return history

def generate_ai_report_text(ticker, analysis, stock_info, expected_date_str, peak_yield):
    prob = analysis['monte_prob']
    try:
        vol_score = analysis['tech_signals'][-1][1] 
        vol_val = float(vol_score.replace('%',''))
    except: vol_val = 0.0

    html = f"""
    <div style="font-size:0.95rem; color:#ddd; line-height:1.6;">
        <div style="margin-bottom:12px;">
            <strong style="color:#fff;">AI SUMMARY</strong><br>
            Current Score: <strong style="color:{analysis['color']}">{analysis['score']} ({analysis['title']})</strong>
        </div>
        <div style="margin-bottom:12px;">
            <strong style="color:#fff;">PREDICTION</strong>
            <ul style="padding-left:15px; margin-top:4px; color:#bbb;">
                <li>Vol ({vol_val}%) implies move within <b>{expected_date_str.split('-')[-1]} days</b>.</li>
                <li>Potential Peak: <b>{peak_yield:+.1f}%</b>.</li>
                <li>Position Size: <b>{analysis['kelly']:.1f}%</b> (Win Prob: {prob:.0f}%)</li>
            </ul>
        </div>
    </div>
    """
    return html

# --------------------------
# UI 렌더링 및 HTML 생성 (다운로드용 + 화면표시용 통합)
# --------------------------
def generate_full_html(ticker, mkt_cap, analysis, monte_res, score_history, stock_info):
    sim_df, opt, pes, mean, win_prob, expected_date_str, peak_yield = monte_res

    if mkt_cap > 0:
        val_won = mkt_cap * 1350
        if val_won > 100_000_000_000_000: cap_str = f"{val_won/100_000_000_000_000:.1f}T KRW"
        elif val_won > 1_000_000_000_000: cap_str = f"{val_won/1_000_000_000_000:.1f}T KRW"
        else: cap_str = f"{val_won/100_000_000_000:.0f}B KRW"
    else: cap_str = "-"

    peak_color = C_PURP if peak_yield > 50 else (C_BULL if peak_yield > 0 else C_BEAR)
    
    dates = [item['day'] for item in score_history]
    scores = [item['score'] for item in score_history]
    gaps = [item['adv']['whale_gap'] for item in score_history]
    
    def get_cell_style(val, type='score'):
        color, bg, weight = "#888", "transparent", "normal"
        if type == 'score':
            if val >= 80: color, bg, weight = C_BULL, "#00E67611", "bold"
            elif val >= 60: color, bg, weight = C_CYAN, "#00B0FF11", "bold"
            elif val <= 40: color, bg, weight = C_BEAR, "#FF525211", "bold"
        elif type == 'gap':
            if val > 10: color, weight = C_BULL, "bold"
            elif val < -10: color, weight = C_BEAR, "bold"
        return f"color:{color}; background:{bg}; font-weight:{weight};"

    report_html = generate_ai_report_text(ticker, analysis, stock_info, expected_date_str, peak_yield)
    
    cards_html = "<div class='grid-2'>"
    for c in analysis['cards']:
        cards_html += f"""
        <div style="background:#222; padding:10px; border-radius:6px; border-left:3px solid {c['col']};">
            <div style="font-size:0.7rem; color:#888; font-weight:700;">{c['title']}</div>
            <div style="font-size:0.85rem; color:#eee; font-weight:600; margin-top:2px;">{c['stat']}</div>
        </div>"""
    cards_html += "</div>"

    tech_rows = ""
    for i in range(18):
        name, val, bias = analysis['tech_signals'][i]
        c = C_BULL if bias == "Bull" else (C_BEAR if bias == "Bear" else C_NEUT)
        tech_rows += f"""
        <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #333; font-size:0.8rem;">
            <span style="color:#aaa;">{name.split('(')[0]}</span>
            <span style="color:{c}; font-weight:600;">{val}</span>
        </div>"""
    
    # 2열 분리 로직 (HTML 문자열 처리)
    half = len(analysis['tech_signals']) // 2
    rows_list = tech_rows.split('</div>')
    # 단순 split으로 div 태그가 깨질 수 있으므로, 재조립보다는 loop에서 직접 분리
    col1, col2 = "", ""
    for i in range(18):
        name, val, bias = analysis['tech_signals'][i]
        c = C_BULL if bias == "Bull" else (C_BEAR if bias == "Bear" else C_NEUT)
        row = f"""<div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #2A2A2A; font-size:0.8rem;">
            <span style="color:#999;">{name.split('(')[0]}</span>
            <span style="color:{c}; font-weight:600;">{val}</span>
        </div>"""
        if i < 9: col1 += row
        else: col2 += row

    mom_cells = ""
    for d, s, g in zip(dates, scores, gaps):
        mom_cells += f"""
        <div style="display:flex; flex-direction:column; align-items:center; min-width:55px; margin-right:10px;">
            <span style="font-size:0.7rem; color:#666; margin-bottom:4px;">{d}</span>
            <span style="font-size:0.9rem; {get_cell_style(s, 'score')} padding:4px 8px; border-radius:4px; border:1px solid #333;">{s}</span>
            <span style="font-size:0.7rem; {get_cell_style(g, 'gap')} margin-top:4px;">{int(g)}</span>
        </div>"""

    # --- HTML 조립 ---
    full_html = f"""
    <div class="report-container">
        <div class="header-main">
            <div>
                <h1 style="font-size:3rem; font-weight:900; margin:0; line-height:1; letter-spacing:-1px;">{ticker}</h1>
                <div style="font-size:0.9rem; color:#888; margin-top:5px; font-weight:500;">
                    {stock_info.get('name','')} | {cap_str}
                    <span style="background:{analysis['theme']}22; color:{analysis['theme']}; padding:2px 6px; border-radius:4px; font-size:0.75rem; font-weight:700; margin-left:8px;">{analysis['mode']}</span>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="display:inline-block; margin-left:15px;">
                    <div style="font-size:0.75rem; color:#666; font-weight:700;">PROB</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{C_BULL if win_prob>=40 else '#888'};">{win_prob:.0f}%</div>
                </div>
                <div style="display:inline-block; margin-left:15px;">
                    <div style="font-size:0.75rem; color:#666; font-weight:700;">SCORE</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{analysis['color']};">{analysis['score']}</div>
                </div>
            </div>
        </div>

        <div class="box-dark">
            {report_html}
            <div style="margin-top:15px; padding-top:15px; border-top:1px dashed #333;">
                <div style="font-size:0.75rem; color:#666; font-weight:700; margin-bottom:8px;">KEY DRIVERS</div>
                {cards_html}
            </div>
        </div>

        <div class="grid-responsive">
            <div class="box-dark">
                <div style="font-size:0.9rem; color:#fff; font-weight:700; margin-bottom:12px; display:flex; align-items:center;">
                    <span style="width:4px; height:16px; background:{analysis['color']}; margin-right:8px; border-radius:2px;"></span>TRADING PLAN
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.9rem;">
                    <span style="color:#888;">ENTRY</span> <span style="color:#fff; font-weight:600;">${analysis['close']:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.9rem;">
                    <span style="color:#888;">TARGET</span> <span style="color:{C_BULL}; font-weight:600;">${analysis['target']:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:12px; font-size:0.9rem;">
                    <span style="color:#888;">STOP</span> <span style="color:{C_BEAR}; font-weight:600;">${analysis['stop']:.2f}</span>
                </div>
                <div style="background:#1a1a1a; padding:10px; border-radius:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#888; font-size:0.8rem;">Expected</span>
                        <span style="color:{C_CYAN}; font-weight:700; font-size:1rem;">{expected_date_str}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
                        <span style="color:#888; font-size:0.8rem;">Max Peak</span>
                        <span style="color:{peak_color}; font-weight:700; font-size:0.9rem;">{peak_yield:+.1f}%</span>
                    </div>
                </div>
            </div>

            <div class="box-dark" style="overflow:hidden;">
                <div style="font-size:0.9rem; color:#fff; font-weight:700; margin-bottom:12px;">MOMENTUM (10D)</div>
                <div class="scroll-x">
                    {mom_cells}
                </div>
                <div style="font-size:0.7rem; color:#555; margin-top:5px; text-align:right;">*Scroll right</div>
            </div>
        </div>

        <div class="box-dark">
            <div style="font-size:0.9rem; color:#fff; font-weight:700; margin-bottom:12px;">TECHNICAL INDICATORS</div>
            <div class="grid-2" style="gap:20px;">
                <div>{col1}</div>
                <div>{col2}</div>
            </div>
        </div>
        
        <div style="text-align:center; color:#555; font-size:0.8rem; margin-top:20px;">
            WQA Generated Report • {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
    """
    return full_html

# --------------------------
# Main App Execution
# --------------------------
st.markdown("""
<div style='text-align: center; padding: 20px 0 30px 0;'>
    <div style='font-size: 1.5rem; font-weight: 900; letter-spacing: 1px; color: #fff;'>WHALE QUANT <span style='color:#00B0FF'>ANALYTICS</span></div>
    <div style='font-size: 0.8rem; color: #666; margin-top: 5px; font-weight: 500;'>Institutional Grade Market Intelligence</div>
</div>
""", unsafe_allow_html=True)

ticker_input = st.text_input("", placeholder="Ticker Symbol (e.g. NVDA, TSLA, PLTR)")

if st.button("RUN ANALYSIS"):
    if not ticker_input:
        st.warning("Please enter a ticker.")
    else:
        tickers = [t.strip().upper() for t in ticker_input.split(',')]
        spy_df = get_benchmark("SAFE")
        iwm_df = get_benchmark("GROWTH")
        
        for ticker in tickers:
            if not ticker: continue
            
            with st.spinner(f"Analyzing {ticker}..."):
                try:
                    stock_info = get_stock_info(ticker)
                    df = get_clean_data(ticker)
                    
                    if df is None:
                        st.error(f"❌ {ticker}: Data Unavailable.")
                        continue
                        
                    mkt_cap = stock_info['mkt_cap']
                    volatility = df['Volatility'].iloc[-1]
                    bench_df = iwm_df if (mkt_cap < 10000000000 or volatility > 3.0) else spy_df
                    
                    monte_res = run_monte_carlo(df)
                    analysis = analyze_whale_mode(df, bench_df, stock_info, monte_res[4])
                    score_history = get_score_history(df, bench_df, stock_info)
                    
                    # HTML 내용 생성
                    report_content = generate_full_html(ticker, mkt_cap, analysis, monte_res, score_history, stock_info)
                    
                    # 1. 화면에 표시 (단순 렌더링)
                    st.markdown(report_content, unsafe_allow_html=True)
                    
                    # 2. 다운로드용 HTML 생성 (CSS 포함하여 독립 실행 가능하도록)
                    # 중요: 이 HTML 파일은 인터넷 연결 없이도 스타일이 유지됩니다.
                    download_html = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>{ticker} Report - WQA</title>
                        <style>{COMMON_CSS}</style>
                    </head>
                    <body>
                        {report_content}
                    </body>
                    </html>
                    """
                    
                    # 3. 다운로드 버튼 (종목별 고유 키 할당)
                    st.download_button(
                        label=f"📥 {ticker} 리포트 다운로드 (HTML)",
                        data=download_html,
                        file_name=f"{ticker}_Report.html",
                        mime="text/html",
                        key=f"btn_{ticker}"
                    )
                    
                    st.markdown("---")
                    
                except Exception as e:
                    st.error(f"System Error ({ticker}): {str(e)}")
