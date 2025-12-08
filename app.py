import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import logging
import warnings
from datetime import datetime, timedelta

# 1. 설정 및 초기화 (모바일 최적화 레이아웃)
st.set_page_config(page_title="Whale Hunter AI", layout="wide", page_icon="🐋", initial_sidebar_state="collapsed")
logger = logging.getLogger('yfinance')
logger.setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# --- [모바일 최적화 CSS] ---
# 핸드폰 화면(768px 이하)일 때 UI를 강제로 앱처럼 변환
st.markdown("""
    <style>
    /* 전체 배경 및 폰트 */
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;500;700;900&display=swap');
    
    .stApp {
        background-color: #0E1117;
        font-family: 'Pretendard', sans-serif;
    }
    
    /* 상단 여백 제거 (모바일에서 넓게 쓰기 위함) */
    .block-container {
        padding-top: 1rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
        padding-bottom: 3rem;
    }

    /* 입력창 스타일 */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #fff;
        border: 1px solid #444;
        border-radius: 12px;
        height: 50px;
        font-size: 1.1rem;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        font-size: 1rem;
    }

    /* 반응형 컨테이너 (모바일용 미디어 쿼리) */
    @media (max-width: 768px) {
        .mobile-header {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 10px !important;
        }
        .header-stats {
            width: 100%;
            display: flex;
            justify-content: space-between;
            background: #1f2937;
            padding: 15px;
            border-radius: 12px;
            margin-top: 10px;
        }
        .stat-item {
            text-align: center !important;
            border: none !important;
            padding: 0 !important;
        }
        .report-section {
            padding: 10px !important;
        }
        .grid-layout {
            grid-template-columns: 1fr !important; /* 무조건 한 줄로 */
        }
        .ticker-name {
            font-size: 2.5rem !important;
        }
        /* 테이블 스크롤 가능하게 */
        .scrollable-table {
            display: block;
            overflow-x: auto;
            white-space: nowrap;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# 2. 참조 데이터 및 전역 캐시
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

# --- [COLOR PALETTE] ---
C_BULL = "#00E676" # Green
C_BEAR = "#FF5252" # Red
C_NEUT = "#B0BEC5" # Grey
C_WARN = "#FFD740" # Yellow
C_CYAN = "#00B0FF" # Blue
C_PURP = "#E040FB" # Purple

# 3. 데이터 엔진
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

        # [NEW] A/D Line 추가
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

def get_18_tech_signals(df):
    last = df.iloc[-1]
    signals = []

    # SMA
    signals.append(("SMA 20 (단기)", f"{last['MA20']:.2f}", "Bull" if last['Close'] > last['MA20'] else "Bear"))
    signals.append(("SMA 60 (중기)", f"{last['MA60']:.2f}", "Bull" if last['Close'] > last['MA60'] else "Bear"))
    signals.append(("SMA 120 (장기)", f"{last['MA120']:.2f}", "Bull" if last['Close'] > last['MA120'] else "Bear"))

    # Momentum
    rsi = last['RSI']
    bias = "Bear" if rsi > 70 else ("Bull" if rsi < 30 else "Neutral")
    signals.append(("RSI (14)", f"{rsi:.1f}", bias))

    macd = last['MACD']
    sig = last['MACD_Signal']
    signals.append(("MACD", f"{macd:.2f}/{sig:.2f}", "Bull" if macd > sig else "Bear"))

    k = last['Stoch_K']
    d = last['Stoch_D']
    signals.append(("Stochastic", f"K{k:.0f}/D{d:.0f}", "Bull" if k > d else "Bear"))

    cci = last['CCI']
    bias = "Bear" if cci > 100 else ("Bull" if cci < -100 else "Neutral")
    signals.append(("CCI", f"{cci:.1f}", bias))

    wr = last['WillR']
    bias = "Bull" if wr < -80 else ("Bear" if wr > -20 else "Neutral")
    signals.append(("Williams %R", f"{wr:.1f}", bias))

    # Volatility & Volume
    pos, bias = ("중간", "Neutral")
    if last['Close'] > last['BB_Upper']: pos, bias = "상단 저항", "Bear"
    elif last['Close'] < last['BB_Lower']: pos, bias = "하단 지지", "Bull"
    signals.append(("Bollinger", pos, bias))

    signals.append(("ATR (변동폭)", f"{last['ATR']:.2f}", "Neutral"))

    obv_ma = df['OBV'].rolling(20).mean().iloc[-1]
    signals.append(("OBV (수급)", "상승" if last['OBV'] > obv_ma else "하락", "Bull" if last['OBV'] > obv_ma else "Bear"))

    mfi = last['MFI']
    bias = "Bear" if mfi > 80 else ("Bull" if mfi < 20 else "Neutral")
    signals.append(("MFI (자금)", f"{mfi:.1f}", bias))

    signals.append(("VWAP (평단)", f"{last['VWAP']:.2f}", "Bull" if last['Close'] > last['VWAP'] else "Bear"))

    roc = last['ROC']
    signals.append(("ROC (등락)", f"{roc:.2f}%", "Bull" if roc > 0 else "Bear"))

    cloud_top = max(last['Senkou_Span_A'], last['Senkou_Span_B'])
    cloud_bot = min(last['Senkou_Span_A'], last['Senkou_Span_B'])
    ichi, bias = "구름대 안", "Neutral"
    if last['Close'] > cloud_top: ichi, bias = "구름대 위", "Bull"
    elif last['Close'] < cloud_bot: ichi, bias = "구름대 아래", "Bear"
    signals.append(("일목균형표", ichi, bias))

    sqz = check_ttm_squeeze(df)
    signals.append(("TTM Squeeze", "ON" if sqz else "OFF", "Bull" if sqz else "Neutral"))

    pat = check_candle_pattern(df)
    signals.append(("캔들 패턴", pat if pat else "일반", "Bull" if pat == "Hammer" else "Neutral"))

    vol = last['Volatility']
    signals.append(("변동성 Ratio", f"{vol:.2f}%", "Neutral"))

    return signals

def z_score_normalize(series):
    return (series - series.mean()) / series.std()

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
    if curr_low_price > prev_low_price and curr_low_rsi < prev_low_rsi: return "HID_BULL"
    if curr_high_price < prev_high_price and curr_high_rsi > prev_high_rsi: return "HID_BEAR"
    return None

def check_ttm_squeeze(df):
    last = df.iloc[-1]
    bb_width = last['BB_Upper'] - last['BB_Lower']
    kc_width = last['KC_Upper'] - last['KC_Lower']
    if bb_width < kc_width: return True
    return False

def check_candle_pattern(df):
    last = df.iloc[-1]
    open_p, close_p = last['Open'], last['Close']
    high_p, low_p = last['High'], last['Low']
    body = abs(close_p - open_p)
    upper_shadow = high_p - max(open_p, close_p)
    lower_shadow = min(open_p, close_p) - low_p
    total_range = high_p - low_p
    if total_range == 0: return None
    if (lower_shadow > body * 2) and (upper_shadow < body * 0.5) and (lower_shadow > upper_shadow * 2): return "Hammer"
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

    if winning_peaks:
        target_peak_price = np.median(winning_peaks)
    else:
        target_peak_price = np.median(max_peaks)
        
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
    
    # 0. Risk Flags
    red_flags = 0

    # 1. Fundamentals
    per, roe = stock_info['per'], stock_info['roe']
    if per and roe:
        if per < 25 and roe > 0.10: score += 15; cards.append({'title':'펀더멘털','stat':'저평가 우량','desc':f'PER {per:.1f}', 'col':C_CYAN})
        elif roe > 0.15: score += 10; cards.append({'title':'펀더멘털','stat':'고수익성','desc':f'ROE {roe*100:.1f}%', 'col':C_BULL})
        elif per > 80: score -= 10; cards.append({'title':'펀더멘털','stat':'고평가 주의','desc':f'PER {per:.1f}', 'col':C_WARN})
        else: cards.append({'title':'펀더멘털','stat':'적정/보통','desc':'특이사항 없음', 'col':C_NEUT})
    else: cards.append({'title':'펀더멘털','stat':'정보 없음','desc':'데이터 부족', 'col':C_NEUT})

    # 2. Whale Gap
    if whale_gap > 30: score += 20; cards.append({'title':'고래 수급','stat':'강력 매집','desc':'개미 털고 매집 중', 'col':C_BULL})
    elif whale_gap > 10: score += 10; cards.append({'title':'고래 수급','stat':'매집 의심','desc':'자금 유입 포착', 'col':C_CYAN})
    elif whale_gap < -10: 
        score -= 15; red_flags += 1
        cards.append({'title':'고래 수급','stat':'세력 이탈','desc':'매도 시그널', 'col':C_BEAR})
    else: cards.append({'title':'고래 수급','stat':'중립','desc':'수급 특이점 없음', 'col':C_NEUT})

    # 3. Squeeze
    if check_ttm_squeeze(df): score += 15; cards.append({'title':'변동성','stat':'스퀴즈 ON','desc':'에너지 폭발 임박', 'col':C_PURP})
    else: cards.append({'title':'변동성','stat':'일반','desc':'에너지 축적 필요', 'col':C_NEUT})
    
    # 4. Divergence
    div_status = check_rsi_divergence(df)
    if div_status == "REG_BULL": score += 20; cards.append({'title':'다이버전스','stat':'상승 반전','desc':'추세 전환 신호', 'col':C_BULL})
    elif div_status == "REG_BEAR": score -= 20; cards.append({'title':'다이버전스','stat':'하락 반전','desc':'고점 징후 포착', 'col':C_BEAR})
    else: cards.append({'title':'다이버전스','stat':'없음','desc':'지표와 주가 동행', 'col':C_NEUT})

    # 5. Candle
    pat = check_candle_pattern(df)
    if pat == "Hammer": score += 10; cards.append({'title':'캔들 패턴','stat':'망치형 (Bull)','desc':'바닥권 반등 암시', 'col':C_WARN})
    elif pat == "Doji": cards.append({'title':'캔들 패턴','stat':'도지 (Doji)','desc':'추세 고민 중', 'col':C_NEUT})
    else: cards.append({'title':'캔들 패턴','stat':'일반','desc':'특이 패턴 없음', 'col':C_NEUT})

    # 6. Ichimoku
    c_top = max(last['Senkou_Span_A'], last['Senkou_Span_B'])
    c_bot = min(last['Senkou_Span_A'], last['Senkou_Span_B'])
    if close > c_top: score += 10; cards.append({'title':'일목균형표','stat':'구름대 위','desc':'상승 추세 지지', 'col':C_CYAN})
    elif close < c_bot: score -= 10; cards.append({'title':'일목균형표','stat':'구름대 아래','desc':'강한 저항 구간', 'col':C_BEAR})
    else: cards.append({'title':'일목균형표','stat':'구름대 안','desc':'방향성 탐색 중', 'col':C_NEUT})

    # 7. Trend
    if close > last['MA20']: 
        score += 10
        cards.append({'title':'추세 (MA)','stat':'단기 상승','desc':'20일선 위', 'col':C_BULL})
    else: 
        score -= 15
        cards.append({'title':'추세 (MA)','stat':'단기 하락','desc':'20일선 붕괴', 'col':C_BEAR})

    # 8. Hit Rate (Monte Carlo +30% Touch)
    if monte_prob >= 40: score += 10; cards.append({'title':'대박 확률','stat':f'{monte_prob:.0f}% (>30%)','desc':'120일 내 유력', 'col':C_BULL})
    elif monte_prob <= 10: score -= 10; cards.append({'title':'대박 확률','stat':f'{monte_prob:.0f}% (>30%)','desc':'희박', 'col':C_BEAR})
    else: cards.append({'title':'대박 확률','stat':f'{monte_prob:.0f}% (>30%)','desc':'보통', 'col':C_NEUT})

    # --- Extra Signals ---
    if ad_signal == "Bull": score += 15
    elif ad_signal == "Bear": score -= 15; red_flags += 1
    
    if poc_signal == "Bull": score += 10
    elif poc_signal == "Bear": score -= 10; red_flags += 1
    
    if mfi_signal == "Oversold": score += 10

    if red_flags > 0:
        score = min(score, 65)

    score = max(0, min(100, int(score)))

    # Mode
    if mkt_cap < 10_000_000_000 or volatility > 3.0:
        mode_txt, theme_col = "🦄 야수형", C_PURP
        stop_mult, target_mult = 2.5, 5.0
    else:
        mode_txt, theme_col = "🛡️ 안전형", C_CYAN
        stop_mult, target_mult = 2.0, 3.0

    stop = close - (atr * stop_mult)
    target = close + (atr * target_mult)

    if score >= 80: t, c = "Strong Buy", C_BULL
    elif score >= 60: 
        if red_flags > 0: t, c = "Caution", C_WARN
        else: t, c = "Buy", C_CYAN
    elif score <= 30: t, c = "Sell", C_BEAR
    else: t, c = "Hold", C_NEUT

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

def get_score_history(df, bench_df, win_rate, avg_ret, stock_info):
    history = []
    # 0~9일 전 데이터 순회 (최근 10일)
    for i in range(9, -1, -1):
        if i == 0:
            sliced_df = df
            sliced_bench = bench_df
        else:
            sliced_df = df.iloc[:-i]
            sliced_bench = bench_df.iloc[:-i]

        if len(sliced_bench) > len(sliced_df):
            sliced_bench = sliced_bench.iloc[-len(sliced_df):]

        label = sliced_df.index[-1].strftime('%m-%d')
        sim_res = run_monte_carlo(sliced_df, num_simulations=100, days=120) 
        
        res = analyze_whale_mode(sliced_df, sliced_bench, win_rate, avg_ret, stock_info, sim_res[4])
        history.append({'day': label, 'score': res['score'], 'adv': res['adv_features']})
    return history

def generate_ai_report_text(ticker, analysis, stock_info, score_history, expected_date_str, peak_yield):
    prob = analysis['monte_prob']
    
    try:
        target_dt = datetime.strptime(expected_date_str, "%Y-%m-%d")
        days_left = (target_dt - datetime.now()).days
        days_str = f"{days_left}일"
    except:
        days_str = "-"

    try:
        vol_score = analysis['tech_signals'][-1][1] 
        vol_val = float(vol_score.replace('%',''))
    except:
        vol_val = 0.0

    reason_html = ""
    
    if vol_val > 4.0:
        reason_html += f"<li><b>기간:</b> 변동성({vol_val}%)이 커서 <b>{days_str}</b>만에 목표 도달 가능성 있음.</li>"
    elif vol_val > 2.0:
        reason_html += f"<li><b>기간:</b> <b>{days_str}</b> 정도 꾸준한 상승 필요.</li>"
    else:
        reason_html += f"<li><b>기간:</b> <b>{days_str}</b> 이상 소요될 수 있음.</li>"

    if peak_yield > 40:
        reason_html += f"<li><b>수익:</b> 통계적으로 <b>+{peak_yield:.1f}%</b>까지 슈팅 가능.</li>"
    elif peak_yield > 20:
        reason_html += f"<li><b>수익:</b> <b>+{peak_yield:.1f}%</b> 부근 고점 예상.</li>"
    else:
        reason_html += f"<li><b>수익:</b> <b>+{peak_yield:.1f}%</b> 부근 횡보 예상.</li>"

    html = f"""
    <div style="line-height:1.6; color:#e0e0e0; font-size:0.95em;">
        <div style="margin-bottom:15px; padding:10px; background:#2A2A2A; border-radius:8px;">
            <div style="color:#aaa; font-size:0.8em; margin-bottom:5px;">AI 종합 의견</div>
            <div style="font-size:1.1em; font-weight:bold;">현재 점수는 <span style="color:{analysis['color']}">{analysis['score']}점 ({analysis['title']})</span> 입니다.</div>
        </div>
        <div style="margin-bottom:15px;">
            <div style="color:#aaa; font-size:0.8em; margin-bottom:5px;">분석 근거 (Why?)</div>
            <ul style="margin:0 0 0 20px; padding:0; color:#ccc; font-size:0.95em;">
                {reason_html}
            </ul>
        </div>
        <div>
            <div style="color:#aaa; font-size:0.8em; margin-bottom:5px;">투자 제안</div>
            비중 <b>{analysis['kelly']:.1f}%</b> 추천 (승률 {prob:.1f}%)
        </div>
    </div>
    """
    return html

# [UI 렌더링 HTML 생성 함수 - 모바일 최적화]
def get_render_html(ticker, mkt_cap, analysis, monte_res, score_history, stock_info):
    sim_df, opt, pes, mean, win_prob, expected_date_str, peak_yield = monte_res

    if mkt_cap > 0:
        val_won = mkt_cap * 1350
        if val_won > 100_000_000_000_000: cap_str = f"{val_won/100_000_000_000_000:.1f}조"
        elif val_won > 1_000_000_000_000: cap_str = f"{val_won/1_000_000_000_000:.1f}조"
        else: cap_str = f"{val_won/100_000_000_000:.0f}천억"
    else: cap_str = "-"

    peak_color = C_PURP if peak_yield > 50 else (C_BULL if peak_yield > 0 else C_BEAR)
    peak_str = f"MAX +{peak_yield:.0f}%" if peak_yield > 0 else f"MAX {peak_yield:.0f}%"

    sorted_history = score_history 
    dates = [item['day'] for item in sorted_history]
    scores = [item['score'] for item in sorted_history]
    gaps = [item['adv']['whale_gap'] for item in sorted_history]
    ads = [item['adv']['ad_signal'] for item in sorted_history]
    mfis = [item['adv']['mfi_signal'] for item in sorted_history]
    pocs = [item['adv']['poc_signal'] for item in sorted_history]

    def get_style_content(label, v):
        txt, col, bg, fw = "WAIT", "#666", "transparent", "normal"
        if label.startswith("Date"): return v, "#bbb", "transparent", "normal"
        elif label == "Score":
            txt = str(v)
            if v >= 80: col, bg, fw = C_BULL, "#00E67611", "bold"
            elif v >= 60: col, bg, fw = C_CYAN, "#00B0FF11", "bold"
            elif v <= 40: col, bg, fw = C_BEAR, "#FF525211", "bold"
            else: col = C_NEUT
        elif label == "Whale":
            if v > 10: txt, col, bg, fw = "BUY", C_BULL, "#00E67622", "900"
            elif v < -10: txt, col, bg, fw = "SELL", C_BEAR, "#FF525222", "900"
        elif label == "Smart":
            if v == "Bull": txt, col, bg, fw = "BUY", C_BULL, "#00E67622", "900"
            elif v == "Bear": txt, col, bg, fw = "SELL", C_BEAR, "#FF525222", "900"
        elif label == "RSI":
            if v == "Oversold": txt, col, bg, fw = "BUY", C_BULL, "#00E67622", "900"
            elif v == "Overbot": txt, col, bg, fw = "SELL", C_BEAR, "#FF525222", "900"
        elif label == "POC":
            if v == "Bull": txt, col, bg, fw = "BUY", C_BULL, "#00E67622", "900"
            elif v == "Bear": txt, col, bg, fw = "SELL", C_BEAR, "#FF525222", "900"
        return txt, col, bg, fw

    def make_row_html(label, values, is_header=False):
        row_html = f"<tr><td style='position:sticky; left:0; z-index:1; text-align:left; color:#999; font-size:0.8rem; padding:12px 8px; border-right:1px solid #333; background:#1A1A1A; width:60px;'>{label}</td>"
        for v in values:
            txt, col, bg, fw = get_style_content(label, v)
            if is_header: row_html += f"<td style='color:#bbb; font-size:0.75rem; padding:8px 6px; background:#222; border-bottom:1px solid #444;'>{txt}</td>"
            else: row_html += f"<td style='color:{col}; background:{bg}; font-weight:{fw}; font-size:0.8rem; padding:8px 6px; border:1px solid #222;'>{txt}</td>"
        row_html += "</tr>"
        return row_html

    hist_table = """<div class="scrollable-table" style="margin-top:10px; border:1px solid #333; border-radius:6px; overflow-x:auto;"><table style="width:100%; border-collapse:collapse; text-align:center; white-space:nowrap;">"""
    hist_table += make_row_html("Date", dates, is_header=True)
    hist_table += make_row_html("Score", scores)
    hist_table += make_row_html("Whale", gaps)
    hist_table += make_row_html("Smart", ads)
    hist_table += make_row_html("RSI", mfis)
    hist_table += make_row_html("POC", pocs)
    hist_table += "</table></div>"
    
    legend_html = f"""
    <div style="display:flex; justify-content:flex-end; gap:10px; font-size:0.7em; color:#888; margin-top:8px; margin-bottom:15px;">
        <span style="display:flex; align-items:center;"><span style="width:8px; height:8px; background:{C_BULL}; margin-right:4px; border-radius:2px;"></span>BUY</span>
        <span style="display:flex; align-items:center;"><span style="width:8px; height:8px; background:{C_BEAR}; margin-right:4px; border-radius:2px;"></span>SELL</span>
    </div>
    """

    cards_html = "<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:10px;'>"
    for c in analysis['cards']:
        cards_html += f"""
        <div style="background:#262626; padding:12px; border-radius:12px; border-left:4px solid {c['col']}; position:relative; overflow:hidden;">
            <div style="position:absolute; top:0; left:0; width:100%; height:100%; background:{c['col']}; opacity:0.05;"></div>
            <div style="font-size:0.8rem; font-weight:bold; color:#eee; margin-bottom:4px;">{c['title']}</div>
            <div style="font-size:0.9rem; color:{c['col']}; font-weight:bold;">{c['stat']}</div>
        </div>"""
    cards_html += "</div>"

    tech_html = "<div style='display:grid; grid-template-columns: 1fr 1fr; gap:10px; font-size:0.8em;'>"
    tech_html += "<div><table style='width:100%; border-collapse:collapse;'>"
    for i in range(9):
        name, val, bias = analysis['tech_signals'][i]
        tc = C_BULL if bias == "Bull" else (C_BEAR if bias == "Bear" else C_NEUT)
        weight = "bold" if bias != "Neutral" else "normal"
        tech_html += f"<tr><td style='padding:6px 0; color:#bbb;'>{name.split('(')[0]}</td><td style='text-align:right; color:{tc}; font-weight:{weight};'>{val}</td></tr>"
    tech_html += "</table></div>"
    tech_html += "<div><table style='width:100%; border-collapse:collapse;'>"
    for i in range(9, 18):
        name, val, bias = analysis['tech_signals'][i]
        tc = C_BULL if bias == "Bull" else (C_BEAR if bias == "Bear" else C_NEUT)
        weight = "bold" if bias != "Neutral" else "normal"
        tech_html += f"<tr><td style='padding:6px 0; color:#bbb;'>{name.split('(')[0]}</td><td style='text-align:right; color:{tc}; font-weight:{weight};'>{val}</td></tr>"
    tech_html += "</table></div></div>"

    report_text = generate_ai_report_text(ticker, analysis, stock_info, score_history, expected_date_str, peak_yield)
    prob_c = C_BULL if win_prob >= 40 else (C_BEAR if win_prob <= 10 else "#ccc")

    html = f"""
    <div class="container" style="font-family: sans-serif; background: #121212; color: #E5E7EB; border-radius: 16px;">
        <div class="header-row mobile-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div>
                <h1 class="ticker-name" style="font-size: 3.5rem; font-weight: 900; margin: 0; line-height: 1; letter-spacing:-2px;">{ticker}</h1>
                <div style="font-size: 0.9rem; color: #9CA3AF; margin-top: 5px;">{cap_str} <span class="badge badge-mode" style="background: {analysis['theme']}20; color: {analysis['theme']}; padding: 2px 6px; border-radius: 4px; font-size:0.8em;">{analysis['mode']}</span></div>
            </div>
            <div class="header-stats" style="display: flex; gap: 20px;">
                <div class="stat-item">
                    <div style="font-size: 1.8rem; font-weight: 800; color:{prob_c}; line-height:1;">{win_prob:.0f}%</div>
                    <div style="font-size: 0.7rem; color: #6B7280;">HIT PROB</div>
                </div>
                <div class="stat-item" style="border-left:1px solid #333; padding-left:20px;">
                    <div style="font-size: 1.8rem; font-weight: 800; color:{analysis['color']}; line-height:1;">{analysis['score']}</div>
                    <div style="font-size: 0.7rem; color: #6B7280;">AI SCORE</div>
                </div>
            </div>
        </div>

        <div class="report-section" style="background: #1A1A1A; padding: 20px; border-radius: 16px; margin-bottom: 20px;">
            <div class="report-top" style="display:flex; gap:20px; flex-wrap:wrap;">
                <div class="report-left" style="flex: 1; min-width: 280px;">
                    <div style="font-size: 1rem; font-weight: bold; color: #fff; margin-bottom: 15px; padding-bottom:10px; border-bottom:1px solid #333;">📑 AI 요약 리포트</div>
                    {report_text}
                </div>
            </div>
             <div style="margin-top:20px;">
                  <div style="font-size: 0.9rem; font-weight: bold; color: #ddd; margin-bottom: 10px;">📊 핵심 팩터 8 (Key Drivers)</div>
                  {cards_html}
            </div>
            <div style="margin-top:25px;">
                <div style="font-size: 0.9rem; font-weight: bold; color: #ddd; margin-bottom: 10px;">📈 최근 10일 모멘텀 (좌우 스크롤)</div>
                {hist_table}
                {legend_html}
            </div>
        </div>

        <div class="grid-layout" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
            <div>
                <div class="card" style="background: #1E1E1E; border-radius: 12px; padding: 20px; border: 1px solid #333;">
                    <div style="font-size: 1rem; font-weight: bold; color: #fff; margin-bottom: 15px; border-bottom:1px solid #333; padding-bottom:10px;">🎯 매매 전략 (Strategy)</div>
                    <div style="display:flex; justify-content:space-between; font-size:1rem; margin-bottom:10px;">
                        <span style="color:#bbb;">진입가 (현재)</span> <b>${analysis['close']:.2f}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:1rem; margin-bottom:10px;">
                        <span style="color:#bbb;">목표가 (TP)</span> <b style="color:{C_BULL}">${analysis['target']:.2f}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:1rem; margin-bottom:10px;">
                        <span style="color:#bbb;">손절가 (SL)</span> <b style="color:{C_BEAR}">${analysis['stop']:.2f}</b>
                    </div>
                   
                    <div style="margin-top:15px; padding-top:15px; border-top:1px dashed #444; background:#222; border-radius:8px; padding:15px;">
                         <div style="font-size:0.9rem; font-weight:bold; color:#ddd; margin-bottom:10px;">🚀 시뮬레이션 예측</div>
                         <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="color:#aaa; font-size:0.85rem;">예상 도달일</span>
                            <div style="text-align:right;">
                                <div style="color:{C_CYAN}; font-weight:bold; font-size:1.2rem;">{expected_date_str}</div>
                                <div style="color:{peak_color}; font-size:0.9rem; margin-top:4px; font-weight:bold;">{peak_str}</div>
                            </div>
                         </div>
                    </div>
                </div>
            </div>

            <div>
                <div class="card" style="background: #1E1E1E; border-radius: 12px; padding: 20px; border: 1px solid #333;">
                    <div style="font-size: 1rem; font-weight: bold; color: #fff; margin-bottom: 15px; border-bottom:1px solid #333; padding-bottom:10px;">🎛 기술적 지표 (Indicators)</div>
                    {tech_html}
                </div>
            </div>
        </div>
    </div>
    """
    return html

# --------------------------
# Main Streamlit App Layout
# --------------------------
st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>🐋 Whale Hunter AI</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #888; margin-bottom: 30px; font-size: 0.9rem;'>모바일 최적화 버전 (Mobile-First)</div>", unsafe_allow_html=True)

# 입력창
ticker_input = st.text_input("", placeholder="티커 입력 (예: NVDA, TSLA)", value="NVDA")

if st.button("🚀 분석 시작"):
    if not ticker_input:
        st.warning("티커를 입력해주세요!")
    else:
        spy_df = get_benchmark("SAFE")
        iwm_df = get_benchmark("GROWTH")
        
        tickers = [t.strip().upper() for t in ticker_input.split(',')]
        
        for ticker in tickers:
            if not ticker: continue
            
            status_text = st.empty()
            status_text.info(f"⏳ {ticker} 분석 중...")
            
            try:
                stock_info = get_stock_info(ticker)
                mkt_cap = stock_info['mkt_cap']
                target_df = get_clean_data(ticker)
                
                if target_df is None:
                    status_text.error(f"❌ {ticker}: 데이터 부족")
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
                st.components.v1.html(html_out, height=1600, scrolling=False)
                
            except Exception as e:
                status_text.error(f"에러 발생: {str(e)}")
