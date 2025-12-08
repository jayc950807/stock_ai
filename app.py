import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Whale Hunter Pro", layout="wide", page_icon="🐋")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 2. CSS 스타일 (아이폰/모바일 최적화 + 코드 노출 방지)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* 모바일 전체화면 설정 */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 5rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    header, footer { visibility: hidden; }
    
    /* OLED 블랙 테마 */
    .stApp { background-color: #000000; color: #f2f2f7; }
    
    /* 입력창 & 버튼 */
    div[data-baseweb="input"] > div {
        background-color: #1c1c1e !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 12px !important;
    }
    input { color: white !important; }
    button {
        background-color: #0A84FF !important;
        color: white !important;
        border-radius: 12px !important;
        height: 3.5rem !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        border: none !important;
    }

    /* 커스텀 UI 클래스 정의 */
    .app-container { font-family: -apple-system, sans-serif; padding: 0 4px 40px 4px; color: #f2f2f7; }
    
    .header-row { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding: 10px 4px; margin-bottom: 15px; }
    .ticker-title { font-size: 32px; font-weight: 900; line-height: 1; margin: 0; letter-spacing: -0.5px; }
    .ticker-sub { font-size: 13px; color: #888; margin-top: 4px; }
    .score-badge { background: #1c1c1e; padding: 8px 12px; border-radius: 14px; border: 1px solid #333; text-align: center; min-width: 75px; }
    .score-val { font-size: 26px; font-weight: 800; line-height: 1; }
    .score-lbl { font-size: 10px; color: #666; font-weight: 700; margin-top: 2px; }
    
    .stat-row { display: flex; gap: 8px; margin-bottom: 15px; }
    .stat-box { flex: 1; background: #1c1c1e; padding: 12px; border-radius: 12px; text-align: center; }
    .stat-lbl { font-size: 11px; color: #888; margin-bottom: 2px; }
    .stat-val { font-size: 18px; font-weight: 800; }
    
    .card { background: #1c1c1e; border-radius: 16px; padding: 16px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
    .card-title { font-size: 15px; font-weight: 700; color: #fff; margin-bottom: 12px; display: flex; align-items: center; gap: 6px; }
    
    .info-flex { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-size: 14px; }
    .lbl { color: #888; } .val { font-weight: 600; }
    
    /* 8 Key Drivers Grid */
    .driver-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 20px; }
    .driver-item { background: #2c2c2e; padding: 12px 10px; border-radius: 10px; display: flex; flex-direction: column; justify-content: center; position: relative; overflow: hidden; }
    
    /* Tech Indicators */
    .tech-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 12px; }
    .tech-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #333; }
    
    /* Scroll Table */
    .scroll-x { overflow-x: auto; -webkit-overflow-scrolling: touch; margin-top: 5px; padding-bottom: 5px; }
    table { width: 100%; border-collapse: collapse; white-space: nowrap; }
    th { color: #888; font-size: 11px; text-align: center; padding: 6px; border-bottom: 1px solid #444; }
    td { font-size: 12px; text-align: center; padding: 8px 6px; border-bottom: 1px solid #333; }
    .sticky-col { position: sticky; left: 0; background: #1c1c1e; z-index: 5; border-right: 1px solid #333; color: #ccc; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. 데이터 엔진 (삭제했던 18개 지표 및 로직 전체 복구)
# ---------------------------------------------------------
WINDOW_SIZE = 60
FORECAST_DAYS = 30

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
    except: return {'mkt_cap': 0, 'per': None, 'roe': None, 'name': ticker}

@st.cache_data(ttl=1800)
def get_full_data(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
            
        if len(df) < WINDOW_SIZE + FORECAST_DAYS: return None

        # [모두 복구된 기술적 지표들]
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA120'] = df['Close'].rolling(120).mean()
        
        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Stochastic
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['Stoch_K'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        # CCI
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

        # Bollinger & Keltner (Squeeze)
        std_20 = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['MA20'] + (std_20 * 2)
        df['BB_Lower'] = df['MA20'] - (std_20 * 2)
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['KC_Upper'] = df['MA20'] + (df['ATR'] * 1.5)
        df['KC_Lower'] = df['MA20'] - (df['ATR'] * 1.5)

        # OBV & A/D
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        ad_factor = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, 1)
        df['AD_Line'] = (ad_factor * df['Volume']).fillna(0).cumsum()

        # MFI
        mf = tp * df['Volume']
        df['MFI'] = 100 - (100 / (1 + (mf.where(tp > tp.shift(1), 0).rolling(14).sum() / mf.where(tp < tp.shift(1), 0).rolling(14).sum())))
        
        # VWAP
        df['VWAP'] = (df['Volume'] * tp).rolling(20).sum() / df['Volume'].rolling(20).sum()

        # Ichimoku
        high_9 = df['High'].rolling(9).max()
        low_9 = df['Low'].rolling(9).min()
        df['Tenkan'] = (high_9 + low_9) / 2
        high_26 = df['High'].rolling(26).max()
        low_26 = df['Low'].rolling(26).min()
        df['Kijun'] = (high_26 + low_26) / 2
        df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
        high_52 = df['High'].rolling(52).max()
        low_52 = df['Low'].rolling(52).min()
        df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)

        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol'] = (df['High'] - df['Low']) / df['Close'] * 100

        df.dropna(inplace=True)
        return df
    except: return None

# ---------------------------------------------------------
# 4. 분석 로직 (복구됨)
# ---------------------------------------------------------
C_UP = "#30D158"
C_DOWN = "#FF453A"
C_NEUT = "#8E8E93"
C_WARN = "#FFD60A"
C_BLUE = "#64D2FF"
C_PURP = "#BF5AF2"

def get_tech_signals(df):
    last = df.iloc[-1]
    sigs = []
    
    # 1. SMA
    sigs.append(("SMA 20", f"{last['MA20']:.2f}", "Bull" if last['Close'] > last['MA20'] else "Bear"))
    sigs.append(("SMA 60", f"{last['MA60']:.2f}", "Bull" if last['Close'] > last['MA60'] else "Bear"))
    
    # 2. RSI
    rsi = last['RSI']
    bias = "Bear" if rsi > 70 else ("Bull" if rsi < 30 else "Neut")
    sigs.append(("RSI (14)", f"{rsi:.1f}", bias))
    
    # 3. MACD
    sigs.append(("MACD", f"{last['MACD']:.2f}", "Bull" if last['MACD'] > last['MACD_Signal'] else "Bear"))
    
    # 4. Stoch
    k, d = last['Stoch_K'], last['Stoch_D']
    sigs.append(("Stoch", f"{k:.0f}/{d:.0f}", "Bull" if k > d else "Bear"))
    
    # 5. CCI
    cci = last['CCI']
    bias = "Bear" if cci > 100 else ("Bull" if cci < -100 else "Neut")
    sigs.append(("CCI", f"{cci:.0f}", bias))
    
    # 6. Bollinger
    pos = "Mid"
    if last['Close'] > last['BB_Upper']: pos = "High"
    elif last['Close'] < last['BB_Lower']: pos = "Low"
    sigs.append(("Bollinger", pos, "Bear" if pos=="High" else ("Bull" if pos=="Low" else "Neut")))
    
    # 7. OBV
    obv_ma = df['OBV'].rolling(20).mean().iloc[-1]
    sigs.append(("OBV", "Up" if last['OBV'] > obv_ma else "Down", "Bull" if last['OBV'] > obv_ma else "Bear"))
    
    # 8. MFI
    mfi = last['MFI']
    bias = "Bear" if mfi > 80 else ("Bull" if mfi < 20 else "Neut")
    sigs.append(("MFI", f"{mfi:.0f}", bias))
    
    # 9. Ichimoku
    cloud_top = max(last['Senkou_A'], last['Senkou_B'])
    cloud_bot = min(last['Senkou_A'], last['Senkou_B'])
    ichi = "In Cloud"
    if last['Close'] > cloud_top: ichi = "Above"
    elif last['Close'] < cloud_bot: ichi = "Below"
    sigs.append(("Ichimoku", ichi, "Bull" if ichi=="Above" else ("Bear" if ichi=="Below" else "Neut")))
    
    # 10. Squeeze
    bb_w = last['BB_Upper'] - last['BB_Lower']
    kc_w = last['KC_Upper'] - last['KC_Lower']
    sqz = bb_w < kc_w
    sigs.append(("Squeeze", "ON" if sqz else "OFF", "Bull" if sqz else "Neut"))
    
    return sigs

def run_monte_carlo(df):
    np.random.seed(42)
    last_price = df['Close'].iloc[-1]
    daily_vol = df['Log_Ret'].tail(30).std()
    
    sims = []
    # 1000회 시뮬레이션 복구
    for _ in range(1000):
        prices = [last_price]
        p = last_price
        for _ in range(120):
            p *= (1 + daily_vol * np.random.normal())
            prices.append(p)
        sims.append(prices)
    
    sims = np.array(sims)
    max_peaks = np.max(sims, axis=1)
    target = last_price * 1.3
    
    win_prob = (max_peaks >= target).mean() * 100
    peak_yield = (np.median(max_peaks) - last_price) / last_price * 100
    
    # 예상 도달일
    hit_indices = np.argmax(sims >= target, axis=1)
    valid_hits = hit_indices[hit_indices > 0]
    days_needed = int(np.mean(valid_hits)) if len(valid_hits) > 0 else 0
    date_str = (datetime.now() + timedelta(days=days_needed)).strftime("%Y-%m-%d") if days_needed else "N/A"
    
    return win_prob, peak_yield, date_str

def analyze_whale_mode(df, info, monte_prob):
    last = df.iloc[-1]
    score = 50
    cards = [] # (Title, Value, Color)
    
    # 1. Whale Gap (OBV Rank - Price Rank)
    rec = df.iloc[-20:]
    p_rank = (last['Close'] - rec['Close'].min()) / (rec['Close'].max() - rec['Close'].min() + 1e-9) * 100
    o_rank = (last['OBV'] - rec['OBV'].min()) / (rec['OBV'].max() - rec['OBV'].min() + 1e-9) * 100
    gap = o_rank - p_rank
    
    if gap > 20: 
        score += 20
        cards.append(("Whale Gap", "Strong Buy", C_UP))
    elif gap < -20: 
        score -= 20
        cards.append(("Whale Gap", "Selling", C_DOWN))
    else: 
        cards.append(("Whale Gap", "Neutral", C_NEUT))

    # 2. Fundamentals
    per = info['per']
    if per and per < 25: 
        score += 10
        cards.append(("Fund.", "Undervalued", C_BLUE))
    elif per and per > 60: 
        score -= 5
        cards.append(("Fund.", "High Val", C_WARN))
    else: 
        cards.append(("Fund.", "Normal", C_NEUT))

    # 3. Squeeze
    bb_w = last['BB_Upper'] - last['BB_Lower']
    kc_w = last['KC_Upper'] - last['KC_Lower']
    if bb_w < kc_w:
        score += 15
        cards.append(("Squeeze", "Ready", C_PURP))
    else:
        cards.append(("Squeeze", "None", C_NEUT))

    # 4. Trend (MA20)
    if last['Close'] > last['MA20']:
        score += 10
        cards.append(("Trend", "Uptrend", C_UP))
    else:
        score -= 15
        cards.append(("Trend", "Downtrend", C_DOWN))

    # 5. Ichimoku Cloud
    cloud_top = max(last['Senkou_A'], last['Senkou_B'])
    if last['Close'] > cloud_top:
        score += 10
        cards.append(("Cloud", "Above", C_BLUE))
    else:
        cards.append(("Cloud", "Below/In", C_NEUT))
        
    # 6. RSI Divergence (간이)
    if last['RSI'] < 30: 
        score += 10
        cards.append(("RSI", "Oversold", C_UP))
    elif last['RSI'] > 70: 
        score -= 10
        cards.append(("RSI", "Overbought", C_DOWN))
    else:
        cards.append(("RSI", "Neutral", C_NEUT))

    # 7. Probability
    if monte_prob >= 40: 
        score += 10
        cards.append(("Prob", "High", C_UP))
    elif monte_prob <= 10: 
        score -= 10
        cards.append(("Prob", "Low", C_DOWN))
    else:
        cards.append(("Prob", "Mid", C_NEUT))
        
    # 8. Volatility
    if last['Vol'] > 3.0:
        cards.append(("Vol", "High", C_PURP))
    else:
        cards.append(("Vol", "Normal", C_NEUT))
        
    score = max(0, min(100, int(score)))
    
    # Target Logic
    stop = last['Close'] - (last['ATR'] * 2)
    target = last['Close'] + (last['ATR'] * 3.5)
    
    # Main Color
    if score >= 80: main_c = C_UP
    elif score >= 50: main_c = C_BLUE
    elif score >= 30: main_c = C_WARN
    else: main_c = C_DOWN
    
    return {
        'score': score, 'main_c': main_c, 'cards': cards,
        'stop': stop, 'target': target, 'close': last['Close'],
        'whale_gap': gap, 'ad_sig': "Bull" if df['AD_Line'].iloc[-1] > df['AD_Line'].iloc[-2] else "Bear"
    }

def get_history(df, info):
    hist = []
    # 10일치 루프 복구
    for i in range(9, -1, -1):
        if i == 0: sub = df
        else: sub = df.iloc[:-i]
        
        prob, _, _ = run_monte_carlo(sub)
        res = analyze_whale_mode(sub, info, prob)
        
        hist.append({
            'date': sub.index[-1].strftime("%m-%d"),
            'score': res['score'],
            'gap': res['whale_gap'],
            'ad': res['ad_sig']
        })
    return hist

# ---------------------------------------------------------
# 5. UI 렌더링 (HTML String 조합 방식 - 코드 노출 원천 차단)
# ---------------------------------------------------------
def render_full_report(t, i, res, monte, tech, hist):
    win_prob, peak, date_str = monte
    
    # 1. Header HTML
    cap = i['mkt_cap']
    cap_str = f"{cap/1e9:.1f}B" if cap > 1e9 else "-"
    
    html = '<div class="app-container">'
    
    # Header
    html += '<div class="header-row">'
    html += f'<div><div class="ticker-title">{t}</div><div class="ticker-sub">{i["name"][:15]}.. • {cap_str}</div></div>'
    html += f'<div class="score-badge"><div class="score-val" style="color:{res["main_c"]}">{res["score"]}</div><div class="score-lbl">SCORE</div></div>'
    html += '</div>'
    
    # Stats
    html += '<div class="stat-row">'
    html += f'<div class="stat-box"><div class="stat-lbl">Win Prob</div><div class="stat-val" style="color:{C_UP if win_prob>40 else "#fff"}">{win_prob:.0f}%</div></div>'
    html += f'<div class="stat-box"><div class="stat-lbl">Max Peak</div><div class="stat-val" style="color:{C_PURP}">+{peak:.0f}%</div></div>'
    html += '</div>'
    
    # Trading Card
    html += f'<div class="card" style="border:1px solid {res["main_c"]}44">'
    html += '<div class="card-title">🎯 Trading Setup</div>'
    html += f'<div class="info-flex"><span class="lbl">Entry</span><span class="val">${res["close"]:.2f}</span></div>'
    html += f'<div class="info-flex"><span class="lbl">Target</span><span class="val" style="color:{C_BLUE}">${res["target"]:.2f}</span></div>'
    html += f'<div class="info-flex"><span class="lbl">Stop Loss</span><span class="val" style="color:{C_DOWN}">${res["stop"]:.2f}</span></div>'
    html += '<div style="margin-top:10px; padding-top:10px; border-top:1px dashed #333;">'
    html += f'<div class="info-flex" style="margin-bottom:0;"><span class="lbl">Expected Date</span><span class="val" style="color:{C_WARN}">{date_str}</span></div>'
    html += '</div></div>'
    
    # 8 Drivers
    html += '<div class="section-title" style="margin:20px 0 10px; font-weight:700;">📊 8 Key Drivers</div>'
    html += '<div class="driver-grid">'
    for title, val, col in res['cards']:
        html += f'<div class="driver-item" style="border-left:3px solid {col}">'
        html += f'<div style="font-size:10px; color:#888;">{title}</div>'
        html += f'<div style="font-weight:700; font-size:13px;">{val}</div>'
        html += '</div>'
    html += '</div>'
    
    # History Table (Scrollable)
    html += '<div class="section-title" style="margin:20px 0 10px; font-weight:700;">📈 10-Day Momentum</div>'
    html += '<div class="scroll-x"><table>'
    # Thead
    html += '<tr><th class="sticky-col">Date</th><th>Score</th><th>Whale</th><th>Smart</th></tr>'
    # Tbody
    for row in hist:
        s_col = C_UP if row['score']>=60 else (C_DOWN if row['score']<=40 else "#ccc")
        w_txt = "Buy" if row['gap']>10 else ("Sell" if row['gap']<-10 else "-")
        w_col = C_UP if row['gap']>10 else (C_DOWN if row['gap']<-10 else "#888")
        
        html += '<tr>'
        html += f'<td class="sticky-col">{row["date"]}</td>'
        html += f'<td style="color:{s_col}; font-weight:700;">{row["score"]}</td>'
        html += f'<td style="color:{w_col}">{w_txt}</td>'
        html += f'<td style="color:{C_UP if row["ad"]=="Bull" else C_DOWN}">{row["ad"]}</td>'
        html += '</tr>'
    html += '</table></div>'
    
    # Tech Indicators
    html += '<div class="section-title" style="margin:20px 0 10px; font-weight:700;">🎛 18 Tech Indicators</div>'
    html += '<div class="card"><div class="tech-grid">'
    # Left Col
    html += '<div>'
    for i in range(5):
        name, val, bias = tech[i]
        tc = C_UP if bias=="Bull" else (C_DOWN if bias=="Bear" else "#888")
        html += f'<div class="tech-row"><span style="color:#888">{name}</span><span style="color:{tc}; font-weight:600;">{val}</span></div>'
    html += '</div>'
    # Right Col
    html += '<div>'
    for i in range(5, 10):
        name, val, bias = tech[i]
        tc = C_UP if bias=="Bull" else (C_DOWN if bias=="Bear" else "#888")
        html += f'<div class="tech-row"><span style="color:#888">{name}</span><span style="color:{tc}; font-weight:600;">{val}</span></div>'
    html += '</div>'
    html += '</div></div>'

    html += '<div style="height:50px;"></div></div>'
    
    return html

# ---------------------------------------------------------
# 6. 메인 실행
# ---------------------------------------------------------
st.title("🐋 Whale Hunter")
st.caption("Pro Mobile Edition")

txt = st.text_input("", value="NVDA", placeholder="Ticker...")

if st.button("ANALYZE", use_container_width=True):
    if not txt:
        st.error("Ticker Required")
    else:
        t = txt.split(',')[0].strip().upper()
        status = st.empty()
        status.info(f"Scanning {t}...")
        
        try:
            df = get_full_data(t)
            if df is None:
                status.error("Data Not Found (Need > 90 days)")
            else:
                info = get_stock_info(t)
                
                # 분석 실행
                monte = run_monte_carlo(df)
                res = analyze_whale_mode(df, info, monte[0])
                tech = get_tech_signals(df)
                hist = get_history(df, info)
                
                # HTML 생성 및 출력
                html_code = render_full_report(t, info, res, monte, tech, hist)
                status.empty()
                st.markdown(html_code, unsafe_allow_html=True)
                
        except Exception as e:
            status.error(f"Error: {str(e)}")
