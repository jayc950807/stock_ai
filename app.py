import warnings
# [설정] 시스템 경고 메시지 완벽 차단
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr
from IPython.display import clear_output, HTML, display
import io
import base64
import logging
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# yfinance 로거 차단
logger = logging.getLogger('yfinance')
logger.setLevel(logging.CRITICAL)
plt.style.use('dark_background')

# [NEW] DuckDuckGo 검색 라이브러리
try:
    from duckduckgo_search import DDGS
except ImportError:
    class DDGS:
        def news(self, keywords, max_results=5): return []

# [NEW] 번역 라이브러리
try:
    from deep_translator import GoogleTranslator
except ImportError:
    class GoogleTranslator:
        def __init__(self, source='auto', target='ko'): pass
        def translate(self, text): return text

# 2. 참조 데이터 및 전역 캐시
REF_DATA = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'TSLA': 'Tesla',
    'GOOGL': 'Google', 'AMZN': 'Amazon', 'META': 'Meta', 'AMD': 'AMD',
    'NFLX': 'Netflix', 'INTC': 'Intel', 'QCOM': 'Qualcomm', 'AVGO': 'Broadcom',
    'JPM': 'JPMorgan', 'BAC': 'BoA', 'GS': 'GoldmanSachs', 'V': 'Visa',
    'JNJ': 'Johnson&Johnson', 'LLY': 'EliLilly', 'PFE': 'Pfizer', 'UNH': 'UnitedHealth',
    'KO': 'CocaCola', 'PEP': 'Pepsi', 'MCD': 'McDonalds', 'WMT': 'Walmart',
    'PLTR': 'Palantir', 'SOFI': 'SoFi', 'COIN': 'Coinbase', 'AMC': 'AMC', 'GME': 'GameStop',
    'XOM': 'Exxon', 'CVX': 'Chevron',
    'IWM': 'Russell2000', 'SPY': 'S&P500', 'QQQ': 'Nasdaq', 'SOXX': 'Semiconductor'
}
REFERENCE_TICKERS = list(REF_DATA.keys())
GLOBAL_REF_CACHE = {}

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
C_BG   = "#121212" # Background

# 3. 데이터 엔진
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

def get_clean_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        if df.empty or len(df) < WINDOW_SIZE + FORECAST_DAYS: return None

        # 기술적 지표 계산
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
        rs = gain / loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))

        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['Stoch_K'] = ((df['Close'] - low_14) / (high_14 - low_14).replace(0, 1)) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std()).replace(0, 0.001)
        df['WillR'] = ((high_14 - df['Close']) / (high_14 - low_14).replace(0, 1)) * -100

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
        df['MFI'] = 100 - (100 / (1 + (mf.where(typical > typical.shift(1), 0).rolling(14).sum() / mf.where(typical < typical.shift(1), 0).rolling(14).sum().replace(0, 1))))

        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).rolling(20).sum() / df['Volume'].rolling(20).sum().replace(0, 1)
        df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12).replace(0, 1)) * 100

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
    except Exception as e:
        print(f"Data Error: {e}")
        return None

def get_market_macro():
    try:
        df = yf.download(['^VIX', '^TNX'], period='5d', progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        vix = df['^VIX'].iloc[-1]
        tnx = df['^TNX'].iloc[-1]
        status = "Normal"
        score_adj = 0
        if vix > 25: 
            status = "FEAR (위험)"
            score_adj = -15
        elif vix < 14:
            status = "GREED (안정)"
            score_adj = +5
        return {'vix': vix, 'tnx': tnx, 'status': status, 'score_adj': score_adj}
    except:
        return {'vix': 0, 'tnx': 0, 'status': 'Unknown', 'score_adj': 0}

def get_google_news_rss(ticker):
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            titles = []
            for item in root.findall('.//item')[:3]:
                title = item.find('title')
                if title is not None: titles.append(title.text)
            return titles
    except: return []
    return []

def get_sentiment_and_short_data(ticker, df):
    data = {'short_pct': 0, 'short_signal': 'N/A', 'upside_pot': 0, 'analyst_signal': 'N/A', 'news_score': 0, 'news_signal': 'Neutral', 'headlines': []}
    t = yf.Ticker(ticker)
    try:
        info = t.info
        short_float = info.get('shortPercentOfFloat', 0)
        if short_float is None: short_float = 0
        short_pct = short_float * 100
        short_signal = "Neutral"
        if short_pct > 30: short_signal = "Squeeze Possibility"
        elif short_pct > 10: short_signal = "High Short (Bad)"
        current_price = df['Close'].iloc[-1]
        target_mean = info.get('targetMeanPrice', current_price)
        if target_mean is None: target_mean = current_price
        upside_pot = ((target_mean - current_price) / current_price) * 100
        analyst_signal = "Bull" if upside_pot > 10 else ("Bear" if upside_pot < -10 else "Neutral")
        data['short_pct'] = short_pct
        data['short_signal'] = short_signal
        data['upside_pot'] = upside_pot
        data['analyst_signal'] = analyst_signal
    except: pass 

    raw_headlines = []
    try:
        yf_news = t.news
        if yf_news:
            for item in yf_news[:3]:
                title = item.get('title', '')
                if title: raw_headlines.append(title)
    except: pass

    if len(raw_headlines) < 3:
        try:
            ddgs = DDGS()
            ddg_res = ddgs.news(keywords=f"{ticker} stock", max_results=3)
            if ddg_res:
                for item in ddg_res:
                    title = item.get('title', '')
                    if title: raw_headlines.append(title)
        except: pass

    if len(raw_headlines) < 3:
        try:
            g_news = get_google_news_rss(ticker)
            if g_news: raw_headlines.extend(g_news)
        except: pass

    unique_headlines = list(set(raw_headlines))[:5]
    sentiment_score = 0
    bull_words = ['up', 'surge', 'jump', 'beat', 'growth', 'gain', 'buy', 'strong', 'profit', 'partnership', 'merger', 'record', 'soar', 'bull', 'upgrade']
    bear_words = ['down', 'drop', 'fall', 'miss', 'loss', 'sell', 'weak', 'lawsuit', 'investigation', 'inflation', 'cut', 'crash', 'plunge', 'bear', 'downgrade']
    final_headlines = []
    try: translator = GoogleTranslator(source='auto', target='ko')
    except: translator = None

    for title in unique_headlines:
        title_lower = title.lower()
        for w in bull_words:
            if w in title_lower: sentiment_score += 1
        for w in bear_words:
            if w in title_lower: sentiment_score -= 1
        translated_title = title
        if translator:
            try: translated_title = translator.translate(title)
            except: pass
        final_headlines.append(translated_title)

    news_signal = "Positive" if sentiment_score > 0 else ("Negative" if sentiment_score < 0 else "Neutral")
    data['news_score'] = sentiment_score
    data['news_signal'] = news_signal
    data['headlines'] = final_headlines
    return data

def get_benchmark(mode):
    ticker = "SPY" if mode == "SAFE" else "IWM"
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        return df
    except: return None

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
    signals.append(("MACD", f"{macd:.2f}", "Bull" if macd > sig else "Bear"))
    
    k = last['Stoch_K']
    d = last['Stoch_D']
    signals.append(("Stoch", f"{k:.0f}/{d:.0f}", "Bull" if k > d else "Bear"))
    
    cci = last['CCI']
    bias = "Bear" if cci > 100 else ("Bull" if cci < -100 else "Neutral")
    signals.append(("CCI", f"{cci:.0f}", bias))
    
    wr = last['WillR']
    bias = "Bull" if wr < -80 else ("Bear" if wr > -20 else "Neutral")
    signals.append(("Will%R", f"{wr:.0f}", bias))
    
    pos, bias = ("Mid", "Neutral")
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
    
    cloud_top = max(last['Senkou_Span_A'], last['Senkou_Span_B'])
    cloud_bot = min(last['Senkou_Span_A'], last['Senkou_Span_B'])
    ichi, bias = "In", "Neutral"
    if last['Close'] > cloud_top: ichi, bias = "Above", "Bull"
    elif last['Close'] < cloud_bot: ichi, bias = "Below", "Bear"
    signals.append(("Ichimoku", ichi, bias))
    
    sqz = check_ttm_squeeze(df)
    signals.append(("Squeeze", "ON" if sqz else "OFF", "Bull" if sqz else "Neutral"))
    pat = check_candle_pattern(df)
    signals.append(("Candle", pat if pat else "-", "Bull" if pat == "Hammer" else "Neutral"))
    vol = last['Volatility']
    signals.append(("Vol Ratio", f"{vol:.2f}%", "Neutral"))
    return signals

def z_score_normalize(series):
    return (series - series.mean()) / (series.std() + 1e-9)

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

def find_top_matches(target_df, ref_tickers, window, top_n=5):
    target_series = target_df['Close'].tail(window)
    target_z = z_score_normalize(target_series)
    matches = []
    for ref_ticker in ref_tickers:
        if ref_ticker not in GLOBAL_REF_CACHE: continue
        ref_df = GLOBAL_REF_CACHE[ref_ticker]
        limit = len(ref_df) - window - FORECAST_DAYS
        for i in range(0, limit, 5):
            if hasattr(target_df, 'name') and ref_ticker == target_df.name and i > limit - 20: continue
            past_series = ref_df['Close'].iloc[i : i + window]
            past_z = z_score_normalize(past_series)
            if len(target_z) != len(past_z): continue
            corr, _ = pearsonr(target_z, past_z)
            if corr > 0.65:
                future = ref_df['Close'].iloc[i + window : i + window + FORECAST_DAYS]
                s_p = future.iloc[0].item(); e_p = future.iloc[-1].item()
                ret = (e_p - s_p) / s_p * 100
                matches.append({'ticker': ref_ticker, 'score': corr, 'future_return': ret})
    return sorted(matches, key=lambda x: x['score'], reverse=True)[:top_n]

def run_monte_carlo(df, num_simulations=5000, days=120):
    last_price = df['Close'].iloc[-1]
    target_percents = [0.3, 0.5, 0.7, 1.0, 1.5] # 30%, 50%, 70%, 100%, 150%

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

    # 1. 메인(+30%) 결과
    main_target = last_price * 1.30
    sim_maxes = sim_df.max()
    win_count = (sim_maxes >= main_target).sum()
    win_prob = (win_count / num_simulations) * 100

    hit_days = []
    winning_peaks = []
    for col in sim_df.columns:
        if sim_df[col].max() >= main_target:
            hits = sim_df.index[sim_df[col] >= main_target].tolist()
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

    # 2. 추가 목표별 통계
    extra_scenarios = []
    for pct in target_percents:
        tgt_price = last_price * (1 + pct)
        count = (sim_maxes >= tgt_price).sum()
        prob = (count / num_simulations) * 100
        
        tgt_hit_days = []
        for col in sim_df.columns:
            if sim_df[col].max() >= tgt_price:
                hits = sim_df.index[sim_df[col] >= tgt_price].tolist()
                if hits: tgt_hit_days.append(hits[0])
        
        if tgt_hit_days:
            avg_d = int(np.mean(tgt_hit_days))
            f_date = (datetime.now() + timedelta(days=avg_d)).strftime("%Y-%m-%d")
        else:
            f_date = "-"
            
        extra_scenarios.append({'pct': int(pct*100), 'prob': prob, 'date': f_date})

    ending_values = sim_df.iloc[-1, :]
    worst_case_price = np.percentile(ending_values, 10)
    min_yield = (worst_case_price - last_price) / last_price * 100

    forecast_median = sim_df.median(axis=1)
    forecast_upper = sim_df.quantile(0.9, axis=1)
    forecast_lower = sim_df.quantile(0.1, axis=1)
    forecast_data = {'median': forecast_median, 'upper': forecast_upper, 'lower': forecast_lower}

    return sim_df, np.percentile(ending_values, 90), np.percentile(ending_values, 10), np.mean(ending_values), win_prob, expected_date_str, peak_yield, forecast_data, min_yield, extra_scenarios

def calculate_kelly(win_rate, reward_risk_ratio):
    p = win_rate / 100
    q = 1 - p
    b = reward_risk_ratio
    if b <= 0: return 0
    kelly_fraction = p - (q / b)
    safe_kelly = max(0, kelly_fraction * 0.5)
    return safe_kelly * 100

def analyze_whale_mode(ticker, df, benchmark_df, win_rate, avg_return, stock_info, monte_prob, macro_data):
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

    score += macro_data['score_adj']
    if macro_data['status'] == 'FEAR (위험)':
        cards.append({'title':'0. 시장 상황','stat':'공포(VIX↑)','desc':'변동성 주의', 'col':C_BEAR})
    elif macro_data['status'] == 'GREED (안정)':
        cards.append({'title':'0. 시장 상황','stat':'안정(VIX↓)','desc':'투자 심리 호조', 'col':C_BULL})
    else:
        cards.append({'title':'0. 시장 상황','stat':'보통','desc':'특이사항 없음', 'col':C_NEUT})

    per, roe = stock_info['per'], stock_info['roe']
    if per and roe:
        if per < 25 and roe > 0.10: score += 15; cards.append({'title':'1. 펀더멘털','stat':'저평가 우량','desc':f'PER {per:.1f}', 'col':C_CYAN})
        elif roe > 0.15: score += 10; cards.append({'title':'1. 펀더멘털','stat':'고수익성','desc':f'ROE {roe*100:.1f}%', 'col':C_BULL})
        elif per > 80: score -= 10; cards.append({'title':'1. 펀더멘털','stat':'고평가 주의','desc':f'PER {per:.1f}', 'col':C_WARN})
        else: cards.append({'title':'1. 펀더멘털','stat':'적정/보통','desc':'특이사항 없음', 'col':C_NEUT})
    else: cards.append({'title':'1. 펀더멘털','stat':'정보 없음','desc':'데이터 부족', 'col':C_NEUT})

    if whale_gap > 30: score += 20; cards.append({'title':'2. 고래 수급','stat':'강력 매집','desc':'개미 털고 매집 중', 'col':C_BULL})
    elif whale_gap > 10: score += 10; cards.append({'title':'2. 고래 수급','stat':'매집 의심','desc':'자금 유입 포착', 'col':C_CYAN})
    elif whale_gap < -10:
        score -= 15; red_flags += 1
        cards.append({'title':'2. 고래 수급','stat':'세력 이탈','desc':'매도 시그널', 'col':C_BEAR})
    else: cards.append({'title':'2. 고래 수급','stat':'중립','desc':'수급 특이점 없음', 'col':C_NEUT})

    if check_ttm_squeeze(df): score += 15; cards.append({'title':'3. 변동성','stat':'스퀴즈 ON','desc':'에너지 폭발 임박', 'col':C_PURP})
    else: cards.append({'title':'3. 변동성','stat':'일반','desc':'에너지 축적 필요', 'col':C_NEUT})

    div_status = check_rsi_divergence(df)
    if div_status == "REG_BULL": score += 20; cards.append({'title':'4. 다이버전스','stat':'상승 반전','desc':'추세 전환 신호', 'col':C_BULL})
    elif div_status == "REG_BEAR": score -= 20; cards.append({'title':'4. 다이버전스','stat':'하락 반전','desc':'고점 징후 포착', 'col':C_BEAR})
    else: cards.append({'title':'4. 다이버전스','stat':'없음','desc':'지표와 주가 동행', 'col':C_NEUT})

    pat = check_candle_pattern(df)
    if pat == "Hammer": score += 10; cards.append({'title':'5. 캔들 패턴','stat':'망치형 (Bull)','desc':'바닥권 반등 암시', 'col':C_WARN})
    elif pat == "Doji": cards.append({'title':'5. 캔들 패턴','stat':'도지 (Doji)','desc':'추세 고민 중', 'col':C_NEUT})
    else: cards.append({'title':'5. 캔들 패턴','stat':'일반','desc':'특이 패턴 없음', 'col':C_NEUT})

    c_top = max(last['Senkou_Span_A'], last['Senkou_Span_B'])
    c_bot = min(last['Senkou_Span_A'], last['Senkou_Span_B'])
    if close > c_top: score += 10; cards.append({'title':'6. 일목균형표','stat':'구름대 위','desc':'상승 추세 지지', 'col':C_CYAN})
    elif close < c_bot: score -= 10; cards.append({'title':'6. 일목균형표','stat':'구름대 아래','desc':'강한 저항 구간', 'col':C_BEAR})
    else: cards.append({'title':'6. 일목균형표','stat':'구름대 안','desc':'방향성 탐색 중', 'col':C_NEUT})

    if close > last['MA20']:
        score += 10
        cards.append({'title':'7. 추세 (MA)','stat':'단기 상승','desc':'20일선 위', 'col':C_BULL})
    else:
        score -= 15
        cards.append({'title':'7. 추세 (MA)','stat':'단기 하락','desc':'20일선 붕괴', 'col':C_BEAR})

    if monte_prob >= 40: score += 10; cards.append({'title':'8. 대박 확률','stat':f'{monte_prob:.0f}% (>30%)','desc':'120일 내 +30% 유력', 'col':C_BULL})
    elif monte_prob <= 10: score -= 10; cards.append({'title':'8. 대박 확률','stat':f'{monte_prob:.0f}% (>30%)','desc':'대시세 희박', 'col':C_BEAR})
    else: cards.append({'title':'8. 대박 확률','stat':f'{monte_prob:.0f}% (>30%)','desc':'보통', 'col':C_NEUT})

    sent_data = get_sentiment_and_short_data(ticker, df) 
    sp = sent_data['short_pct']
    if sent_data['short_signal'] == "Squeeze Possibility":
        score += 10 
        cards.append({'title':'9. 공매도(Short)','stat':f'{sp:.1f}% (폭발적)','desc':'⚠️ 숏 스퀴즈 가능성!', 'col':C_PURP})
    elif sent_data['short_signal'] == "High Short (Bad)":
        score -= 15
        cards.append({'title':'9. 공매도(Short)','stat':f'{sp:.1f}% (위험)','desc':'하락 베팅 세력 많음', 'col':C_BEAR})
    else:
        cards.append({'title':'9. 공매도(Short)','stat':f'{sp:.1f}% (양호)','desc':'특이사항 없음', 'col':C_NEUT})
    
    if sent_data['news_signal'] == "Positive":
        score += 10
        cards.append({'title':'10. 뉴스 심리','stat':'긍정적','desc':'호재성 키워드 포착', 'col':C_BULL})
    elif sent_data['news_signal'] == "Negative":
        score -= 10
        cards.append({'title':'10. 뉴스 심리','stat':'부정적','desc':'악재성 키워드 주의', 'col':C_BEAR})
    
    if sent_data['upside_pot'] > 30: score += 5
    if ad_signal == "Bull": score += 15
    elif ad_signal == "Bear": score -= 15; red_flags += 1
    if poc_signal == "Bull": score += 10
    elif poc_signal == "Bear": score -= 10; red_flags += 1
    if mfi_signal == "Oversold": score += 10

    if red_flags > 0: score = min(score, 65)
    score = max(0, min(100, int(score)))

    if mkt_cap < 10_000_000_000 or volatility > 3.0:
        mode_txt, theme_col = "🦄 야수 (고위험)", C_PURP
        stop_mult, target_mult = 2.5, 5.0
    else:
        mode_txt, theme_col = "🛡️ 우량 (안전형)", C_CYAN
        stop_mult, target_mult = 2.0, 3.0

    stop = close - (atr * stop_mult)
    target = close + (atr * target_mult)

    if score >= 80: t, c = "강력 매수", C_BULL
    elif score >= 60:
        if red_flags > 0: t, c = "주의 (혼조세)", C_WARN
        else: t, c = "매수", C_CYAN
    elif score <= 30: t, c = "매도 / 관망", C_BEAR
    else: t, c = "관망 / 중립", C_NEUT
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
        'monte_prob': monte_prob,
        'sent_data': sent_data 
    }

def get_score_history(ticker, df, bench_df, win_rate, avg_ret, stock_info, macro_data):
    history = []
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
        res = analyze_whale_mode(ticker, sliced_df, sliced_bench, win_rate, avg_ret, stock_info, sim_res[4], macro_data)
        history.append({'day': label, 'score': res['score'], 'adv': res['adv_features']})
    return history

def generate_ai_report_text(ticker, analysis, stock_info, score_history, expected_date_str, peak_yield, min_yield):
    prob = analysis['monte_prob']
    try:
        target_dt = datetime.strptime(expected_date_str, "%Y-%m-%d")
        days_left = (target_dt - datetime.now()).days
        days_str = f"{days_left}일"
    except: days_str = "불확실"

    try:
        vol_score = analysis['tech_signals'][-1][1]
        vol_val = float(vol_score.replace('%',''))
    except: vol_val = 0.0

    reason_html = ""
    if vol_val > 4.0:
        reason_html += f"<li><b>기간 예측:</b> 현재 일일 변동성({vol_val}%)이 매우 높아, 추세 형성 시 <b>{days_str}</b> 만에 목표가 도달이 가능한 에너지를 보유하고 있습니다.</li>"
    elif vol_val > 2.0:
        reason_html += f"<li><b>기간 예측:</b> 평균적인 변동성({vol_val}%)을 보이고 있어, 목표 달성까지 약 <b>{days_str}</b> 간의 꾸준한 상승 흐름이 필요합니다.</li>"
    else:
        reason_html += f"<li><b>기간 예측:</b> 낮은 변동성({vol_val}%)으로 인해 급등보다는 완만한 우상향이 예상되며, 도달까지 <b>{days_str}</b> 이상 소요될 수 있습니다.</li>"

    if peak_yield > 40:
        reason_html += f"<li><b>수익률 예측(Upside):</b> 높은 변동성은 돌파 시 강한 <b>오버슈팅(Over-shoot)</b>을 유발하며, 통계적으로 <b>+{peak_yield:.1f}%</b> 구간까지 순간 급등할 확률이 존재합니다.</li>"
    elif peak_yield > 20:
        reason_html += f"<li><b>수익률 예측(Upside):</b> 상승 모멘텀이 유지될 경우, 1차 목표 돌파 후 <b>+{peak_yield:.1f}%</b> 수준에서 고점을 형성할 가능성이 높습니다.</li>"
    else:
        reason_html += f"<li><b>수익률 예측(Upside):</b> 강력한 저항선이나 매물대로 인해, 목표 달성 후 추가 상승보다는 <b>+{peak_yield:.1f}%</b> 부근에서의 횡보나 조정이 예상됩니다.</li>"

    reason_html += f"<li><b>리스크 경고(Downside):</b> 반대로 악재 발생 시 시뮬레이션 하위 10% 최악의 경우 <b>{min_yield:.1f}%</b>까지 하락할 위험이 통계적으로 존재합니다.</li>"

    html = f"""
    <div style="line-height:1.6; color:#e0e0e0; font-size:0.9em;">
        <ul style="margin:0; padding-left:20px; color:#ccc;">
            {reason_html}
        </ul>
    </div>
    """
    return html

def get_action_strategy(ticker, analysis, monte_res):
    score = analysis['score']
    win_prob = monte_res[4]
    peak_yield = monte_res[6]
    min_yield = monte_res[8]
    kelly = analysis['kelly']
    
    downside = abs(min_yield) if min_yield < 0 else 1.0
    if downside == 0: downside = 1.0
    rr_ratio = peak_yield / downside
    
    whale_gap = analysis['adv_features']['whale_gap']
    is_squeeze = any(c['title'] == '3. 변동성' and '스퀴즈 ON' in c['stat'] for c in analysis['cards'])
    
    decision = "HOLD"
    reason = "판단 보류"
    color = "#aaa"
    
    if score < 60:
        decision = "DROP (관심 삭제)"
        reason = "AI 점수가 60점 미만입니다. 상승 모멘텀이 부족합니다."
        color = C_BEAR
    elif win_prob < 50:
        decision = "DROP (관심 삭제)"
        reason = "시뮬레이션 승률이 50% 미만입니다. 기회 비용이 큽니다."
        color = C_BEAR
    elif rr_ratio < 2.0:
        decision = "WAIT (관망)"
        reason = f"손익비가 {rr_ratio:.1f}배로 낮습니다. (목표수익 대비 리스크가 큼)"
        color = C_WARN
    else:
        if whale_gap > 10 or is_squeeze:
            decision = "BUY (진입 추천)"
            reason = "점수/승률/손익비 합격 + 고래 매집/스퀴즈 신호 포착됨."
            color = C_BULL
        else:
            decision = "WATCH (타이밍 대기)"
            reason = "조건은 훌륭하나, 결정적인 매수 트리거(고래/변동성)가 아직 없습니다."
            color = C_CYAN
            
    html = f"""
    <div style="background:#1A1A1A; border:1px solid #333; border-radius:8px; padding:15px; grid-column: span 2; margin-top:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #333; padding-bottom:10px; margin-bottom:10px;">
            <div style="font-size:1rem; font-weight:700; color:#fff; display:flex; align-items:center;">
                <span style="background:{color}; width:8px; height:8px; border-radius:50%; margin-right:8px;"></span>
                AI 트레이딩 액션 가이드
            </div>
            <div style="font-size:0.8rem; color:#888;">5단계 필터링 결과</div>
        </div>
        
        <div style="display:flex; align-items:center; gap:20px; margin-bottom:15px;">
            <div style="font-size:1.6rem; font-weight:900; color:{color}; white-space:nowrap;">{decision}</div>
            <div style="background:#252525; padding:8px 12px; border-radius:6px; font-size:0.85rem; color:#ccc; flex-grow:1; line-height:1.4;">
                <b>💡 판단 근거:</b> {reason}
            </div>
        </div>
        
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px;">
            <div style="background:#262626; padding:8px; border-radius:6px; text-align:center;">
                <div style="font-size:0.75rem; color:#999;">진입 비중</div>
                <div style="font-size:1.1rem; font-weight:700; color:{C_CYAN};">{kelly:.1f}%</div>
            </div>
             <div style="background:#262626; padding:8px; border-radius:6px; text-align:center;">
                <div style="font-size:0.75rem; color:#999;">손익비 (R/R)</div>
                <div style="font-size:1.1rem; font-weight:700; color:{C_WARN if rr_ratio < 2 else C_BULL};">{rr_ratio:.1f}배</div>
            </div>
             <div style="background:#262626; padding:8px; border-radius:6px; text-align:center;">
                <div style="font-size:0.75rem; color:#999;">손절가 (SL)</div>
                <div style="font-size:1.1rem; font-weight:700; color:{C_BEAR};">${analysis['stop']:.2f}</div>
            </div>
        </div>
    </div>
    """
    return html

def render_whale_ui(ticker, mkt_cap, analysis, monte_res, score_history, stock_info):
    sim_df, opt, pes, mean, win_prob, expected_date_str, peak_yield, forecast_data, min_yield, extra_scenarios = monte_res
    sd = analysis.get('sent_data', {})

    if mkt_cap > 0:
        val_won = mkt_cap * 1350
        if val_won > 100_000_000_000_000: cap_str = f"{val_won/100_000_000_000_000:.1f}조원"
        elif val_won > 1_000_000_000_000: cap_str = f"{val_won/1_000_000_000_000:.1f}조원"
        else: cap_str = f"{val_won/100_000_000_000:.0f}천억원"
    else: cap_str = "-"

    peak_color = C_PURP if peak_yield > 50 else (C_BULL if peak_yield > 0 else C_BEAR)
    peak_str = f"+{peak_yield:.1f}%" if peak_yield > 0 else f"{peak_yield:.1f}%"
    min_color = C_BEAR if min_yield < -10 else C_WARN
    min_str = f"{min_yield:.1f}%"

    cards_html = "<div style='display:grid; grid-template-columns: 1fr; gap:8px;'>"
    for c in analysis['cards']:
        cards_html += f"""
        <div style="background:#252525; padding:8px 12px; border-radius:6px; border-left:3px solid {c['col']}; display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:0.8rem; font-weight:600; color:#ddd;">{c['title']}</div>
            <div style="text-align:right;">
                <div style="font-size:0.8rem; font-weight:bold; color:{c['col']};">{c['stat']}</div>
            </div>
        </div>"""
    cards_html += "</div>"

    headlines = sd.get('headlines', [])
    news_html = ""
    if headlines:
        for title in headlines[:3]:
            news_html += f"<li style='color:#bbb; font-size:0.85rem; margin-bottom:6px; line-height:1.4;'>• {title}</li>"
    else: news_html = "<li style='color:#666;'>뉴스 데이터 없음</li>"

    extra_table_html = "<table style='width:100%; border-collapse:collapse; font-size:0.85rem;'>"
    extra_table_html += f"<tr style='color:#888; border-bottom:1px solid #333;'><th style='padding:4px; text-align:left;'>Target</th><th style='padding:4px; text-align:right;'>Prob</th><th style='padding:4px; text-align:right;'>ETA</th></tr>"
    for row in extra_scenarios:
        pct = row['pct']
        prob = row['prob']
        date = row['date']
        p_col = C_BULL if prob >= 50 else (C_WARN if prob >= 20 else "#555")
        extra_table_html += f"<tr style='border-bottom:1px solid #222;'>"
        extra_table_html += f"<td style='padding:4px; color:#ddd;'>+{pct}%</td>"
        extra_table_html += f"<td style='padding:4px; text-align:right; color:{p_col}; font-weight:bold;'>{prob:.0f}%</td>"
        extra_table_html += f"<td style='padding:4px; text-align:right; color:#888;'>{date}</td></tr>"
    extra_table_html += "</table>"
    
    action_guide_html = get_action_strategy(ticker, analysis, monte_res)

    # [수정] 18개 전체 지표 표시 (2단 배열)
    tech_mid = (len(analysis['tech_signals']) + 1) // 2
    left_signals = analysis['tech_signals'][:tech_mid]
    right_signals = analysis['tech_signals'][tech_mid:]

    def make_tech_col(signals):
        rows = ""
        for name, val, bias in signals:
            c = C_BULL if bias=="Bull" else (C_BEAR if bias=="Bear" else "#777")
            rows += f"<div style='display:flex; justify-content:space-between; margin-bottom:4px;'><span style='color:#888;'>{name}</span><span style='color:{c}; font-weight:bold;'>{val}</span></div>"
        return rows

    tech_left = make_tech_col(left_signals)
    tech_right = make_tech_col(right_signals)

    # [복구] 모멘텀 트렌드 (History Table)
    def get_style_content(label, v):
        txt, col, bg, fw = "Wait", "#666", "transparent", "normal"
        if label.startswith("Date"): return v, "#bbb", "transparent", "normal"
        elif label == "AI Score":
            txt = str(v)
            if v >= 80: col, bg, fw = C_BULL, "#00E67611", "bold"
            elif v >= 60: col, bg, fw = C_CYAN, "#00B0FF11", "bold"
            elif v <= 40: col, bg, fw = C_BEAR, "#FF525211", "bold"
            else: col = C_NEUT
        elif label == "Whale":
            if v > 10: txt, col, bg, fw = "Buy", C_BULL, "#00E67622", "900"
            elif v < -10: txt, col, bg, fw = "Sell", C_BEAR, "#FF525222", "900"
        elif label == "Money":
            if v == "Bull": txt, col, bg, fw = "Buy", C_BULL, "#00E67622", "900"
            elif v == "Bear": txt, col, bg, fw = "Sell", C_BEAR, "#FF525222", "900"
        elif label == "RSI":
            if v == "Oversold": txt, col, bg, fw = "Buy", C_BULL, "#00E67622", "900"
            elif v == "Overbot": txt, col, bg, fw = "Sell", C_BEAR, "#FF525222", "900"
        elif label == "POC":
            if v == "Bull": txt, col, bg, fw = "Buy", C_BULL, "#00E67622", "900"
            elif v == "Bear": txt, col, bg, fw = "Sell", C_BEAR, "#FF525222", "900"
        return txt, col, bg, fw

    def make_row_html(label, values, is_header=False):
        row_html = f"<tr><td style='text-align:left; color:#999; font-size:0.75rem; padding:6px 6px; border-right:1px solid #333; background:#1A1A1A; width:80px;'>{label}</td>"
        for v in values:
            txt, col, bg, fw = get_style_content(label, v)
            if is_header: row_html += f"<td style='color:#bbb; font-size:0.65rem; padding:4px 2px; background:#222; border-bottom:1px solid #444;'>{txt}</td>"
            else: row_html += f"<td style='color:{col}; background:{bg}; font-weight:{fw}; font-size:0.7rem; padding:4px 2px; border:1px solid #222;'>{txt}</td>"
        row_html += "</tr>"
        return row_html

    dates = [item['day'] for item in score_history]
    scores = [item['score'] for item in score_history]
    gaps = [item['adv']['whale_gap'] for item in score_history]
    ads = [item['adv']['ad_signal'] for item in score_history]
    mfis = [item['adv']['mfi_signal'] for item in score_history]
    pocs = [item['adv']['poc_signal'] for item in score_history]

    hist_table = """<div style="overflow-x:auto; margin-top:5px; border:1px solid #333; border-radius:6px;"><table style="width:100%; border-collapse:collapse; text-align:center; table-layout:fixed; white-space:nowrap;">"""
    hist_table += make_row_html("Date", dates, is_header=True)
    hist_table += make_row_html("AI Score", scores)
    hist_table += make_row_html("Whale", gaps)
    hist_table += make_row_html("Money", ads)
    hist_table += make_row_html("RSI", mfis)
    hist_table += make_row_html("POC", pocs)
    hist_table += "</table></div>"

    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        .db-container {{ font-family: 'Inter', sans-serif; background: #121212; color: #eee; padding: 30px; border-radius: 12px; max-width: 1100px; margin: 0 auto; box-sizing: border-box; }}
        .db-header {{ display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 20px; border-bottom: 1px solid #333; margin-bottom: 20px; }}
        .db-grid {{ display: grid; grid-template-columns: 1.8fr 1.2fr; gap: 25px; }}
        .t-ticker {{ font-size: 3rem; font-weight: 900; line-height: 1; letter-spacing: -1px; margin: 0; }}
        .t-sub {{ font-size: 0.95rem; color: #888; margin-top: 5px; font-weight: 400; }}
        .t-score {{ font-size: 3rem; font-weight: 800; line-height: 1; }}
        .db-card {{ background: #1E1E1E; border-radius: 10px; padding: 20px; border: 1px solid #2A2A2A; margin-bottom: 20px; }}
        .db-card-h {{ font-size: 0.9rem; font-weight: 700; color: #ccc; margin-bottom: 15px; display: flex; justify-content: space-between; border-bottom: 1px solid #333; padding-bottom: 8px; }}
        .stat-box {{ background: #252525; border-radius: 8px; padding: 15px; text-align: center; }}
        .stat-label {{ font-size: 0.75rem; color: #888; margin-bottom: 5px; }}
        .stat-val {{ font-size: 1.4rem; font-weight: 700; }}
    </style>

    <div class="db-container">
        <div class="db-header">
            <div>
                <h1 class="t-ticker">{ticker}</h1>
                <div class="t-sub">{stock_info.get('name','')} · {cap_str} · <span style="color:{analysis['theme']}; border:1px solid {analysis['theme']}; padding:2px 6px; border-radius:4px; font-size:0.8em;">{analysis['mode']}</span></div>
            </div>
            <div style="text-align:right; display:flex; gap:30px;">
                <div>
                    <div style="font-size:0.8rem; color:#888;">AI Score</div>
                    <div class="t-score" style="color:{analysis['color']}">{analysis['score']}</div>
                </div>
                <div>
                    <div style="font-size:0.8rem; color:#888;">Win Rate (>30%)</div>
                    <div class="t-score" style="color:{C_BULL if win_prob>=50 else '#666'}">{win_prob:.0f}<span style="font-size:0.5em;">%</span></div>
                </div>
            </div>
        </div>

        <div class="db-grid">
            
            <div>
                <div class="db-card">
                    <div class="db-card-h">📑 AI 정밀 분석 리포트</div>
                    <div style="font-size:0.95rem; line-height:1.6; color:#ccc;">
                        <div style="margin-bottom:12px;"><b>종합 의견:</b> {ticker}의 현재 상태는 <span style="color:{analysis['color']}; font-weight:bold;">{analysis['title']}</span> 단계입니다.</div>
                        {generate_ai_report_text(ticker, analysis, stock_info, score_history, expected_date_str, peak_yield, min_yield)}
                    </div>
                </div>

                <div class="db-card" style="padding: 15px;">
                    <div class="db-card-h" style="margin-bottom: 10px;">🚀 시뮬레이션: 리스크 & 리워드 (120일)</div>
                    <div style="display:flex; gap:10px; margin-bottom:10px;">
                        <div style="background:#252525; flex:1; padding:8px; border-radius:6px; text-align:center;">
                            <div style="font-size:0.7rem; color:#888;">🔥 Max Peak</div>
                            <div style="font-size:1.1rem; font-weight:bold; color:{peak_color}">{peak_str}</div>
                        </div>
                        <div style="background:#252525; flex:1; padding:8px; border-radius:6px; text-align:center;">
                            <div style="font-size:0.7rem; color:#888;">🥶 Worst Case</div>
                            <div style="font-size:1.1rem; font-weight:bold; color:{min_color}">{min_str}</div>
                        </div>
                    </div>
                    {extra_table_html}
                </div>

                <div class="db-card">
                    <div class="db-card-h">📈 모멘텀 트렌드 (최근 10일)</div>
                    {hist_table}
                </div>

                <div class="db-card">
                    <div class="db-card-h">📰 주요 뉴스 헤드라인 ({sd.get('news_signal','-')})</div>
                    <ul style="list-style:none; padding:0; margin:0;">{news_html}</ul>
                </div>
            </div>

            <div>
                <div class="db-card">
                    <div class="db-card-h">📊 8대 핵심 요인</div>
                    {cards_html}
                </div>

                <div class="db-card">
                    <div class="db-card-h">🎯 매매 전략 가이드</div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.9rem;">
                        <span style="color:#888;">진입가 (Entry)</span> <span>${analysis['close']:.2f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.9rem;">
                        <span style="color:#888;">목표가 (Target)</span> <span style="color:{C_BULL}; font-weight:bold;">${analysis['target']:.2f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:15px; font-size:0.9rem;">
                        <span style="color:#888;">손절가 (Stop)</span> <span style="color:{C_BEAR}; font-weight:bold;">${analysis['stop']:.2f}</span>
                    </div>
                    <div style="border-top:1px dashed #333; padding-top:10px; margin-top:10px;">
                        <div style="font-size:0.8rem; color:#888; margin-bottom:5px;">월가 괴리율 (Upside)</div>
                        <div style="font-size:1.2rem; font-weight:bold; color:{C_CYAN};">{sd.get('upside_pot',0):.1f}%</div>
                    </div>
                </div>

                <div class="db-card">
                    <div class="db-card-h">🎛 기술적 지표 (18종)</div>
                    <div style="font-size:0.8rem; display:grid; grid-template-columns: 1fr 1fr; gap:15px;">
                        <div>{tech_left}</div>
                        <div>{tech_right}</div>
                    </div>
                </div>
            </div>
        </div>

        {action_guide_html}
    </div>
    """
    display(HTML(html))

def init_cache():
    global GLOBAL_REF_CACHE
    if GLOBAL_REF_CACHE: return 
    print("⏳ 참조 데이터(대형주/ETF) 초기화 중... (약 10~20초 소요)")
    valid_count = 0
    total_count = len(REFERENCE_TICKERS)
    for idx, t in enumerate(REFERENCE_TICKERS):
        try:
            df = get_clean_data(t, period="2y")
            if df is not None:
                df.name = t
                GLOBAL_REF_CACHE[t] = df
                valid_count += 1
        except: pass
    print(f"✅ 초기화 완료: {valid_count}/{total_count} 개 종목 캐싱됨.\n")

def run_v32_final():
    if not GLOBAL_REF_CACHE: init_cache()
    spy_df = get_benchmark("SAFE")
    iwm_df = get_benchmark("GROWTH")
    macro_data = get_market_macro()

    while True:
        try: ticker = input("\n🔎 종목코드 입력 (예: NVDA) : ").strip().upper()
        except: break
        if ticker in ['Q', 'QUIT']: break
        if not ticker: continue
        clear_output(wait=True)

        stock_info = get_stock_info(ticker)
        mkt_cap = stock_info['mkt_cap']
        target_df = get_clean_data(ticker)

        if target_df is None: print("❌ 데이터 없음"); continue
        target_df.name = ticker

        volatility = target_df['Volatility'].iloc[-1]
        bench_df = iwm_df if (mkt_cap < 10_000_000_000 or volatility > 3.0) else spy_df

        top_matches = find_top_matches(target_df, REFERENCE_TICKERS, WINDOW_SIZE, TOP_N)
        returns = [m['future_return'] for m in top_matches] if top_matches else [0]
        avg_ret = np.mean(returns)
        win_rate = sum(r > 0 for r in returns) / len(returns) * 100 if returns else 0

        monte_res = run_monte_carlo(target_df)
        
        analysis = analyze_whale_mode(ticker, target_df, bench_df, win_rate, avg_ret, stock_info, monte_res[4], macro_data)
        score_history = get_score_history(ticker, target_df, bench_df, win_rate, avg_ret, stock_info, macro_data)

        render_whale_ui(ticker, mkt_cap, analysis, monte_res, score_history, stock_info)

run_v32_final()
