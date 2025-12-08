import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. 설정 (가장 윗줄)
# ---------------------------------------------------------
st.set_page_config(page_title="Whale Hunter", layout="wide", page_icon="🐋")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 2. 스타일 CSS (변수 충돌 방지를 위해 일반 문자열로 분리)
# ---------------------------------------------------------
# 이 부분은 f-string을 쓰지 않아 절대 깨지지 않습니다.
MOBILE_CSS = """
<style>
    /* 전체 레이아웃 패딩 제거 (폰 화면 꽉 차게) */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 5rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    
    /* UI 요소 숨김 및 다크모드 배경 */
    header, footer { visibility: hidden; }
    .stApp { background-color: #000000; color: #f2f2f7; }
    
    /* 입력창 및 버튼 디자인 */
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
        height: 3rem !important;
        font-weight: 700 !important;
        border: none !important;
    }

    /* 결과화면 컨테이너 */
    .mobile-wrapper {
        font-family: -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        padding: 0 5px 40px 5px;
        color: #f2f2f7;
    }

    /* 헤더 섹션 */
    .header-row {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid #333; padding: 15px 5px; margin-bottom: 20px;
    }
    .ticker-name { font-size: 30px; font-weight: 800; line-height: 1; margin: 0; }
    .ticker-sub { font-size: 13px; color: #8e8e93; margin-top: 5px; }
    
    .score-badge {
        background: #1c1c1e; padding: 10px 15px; border-radius: 14px;
        border: 1px solid #333; text-align: center; min-width: 80px;
    }
    .score-num { font-size: 26px; font-weight: 900; line-height: 1; }
    .score-txt { font-size: 10px; color: #8e8e93; font-weight: 600; margin-top: 3px; }

    /* 통계 그리드 */
    .stat-grid { display: flex; gap: 10px; margin-bottom: 15px; }
    .stat-box {
        flex: 1; background: #1c1c1e; padding: 12px; border-radius: 12px; text-align: center;
    }
    .stat-label { font-size: 11px; color: #8e8e93; margin-bottom: 2px; }
    .stat-value { font-size: 20px; font-weight: 800; }

    /* 메인 카드 */
    .main-card {
        background: #1c1c1e; border-radius: 16px; padding: 16px;
        margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .card-title { font-size: 15px; font-weight: 700; color: #fff; margin-bottom: 15px; }
    
    .info-row {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 10px; font-size: 14px;
    }
    .lbl { color: #8e8e93; }
    .val { font-weight: 600; color: #fff; }

    /* 드라이버 카드 그리드 */
    .driver-grid {
        display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 20px;
    }
    .driver-item {
        background: #2c2c2e; padding: 10px 4px; border-radius: 10px; text-align: center;
    }
    
    /* 가로 스크롤 테이블 */
    .scroll-box {
        overflow-x: auto; -webkit-overflow-scrolling: touch;
        margin-top: 10px; padding-bottom: 10px;
    }
    table { width: 100%; border-collapse: collapse; white-space: nowrap; }
    td { font-size: 12px; padding: 8px 10px; border-bottom: 1px solid #333; }
    .sticky-col {
        position: sticky; left: 0; background: #1c1c1e; z-index: 5;
        border-right: 1px solid #333; font-weight: 600; color: #999;
    }
    
    .section-head { font-size: 16px; font-weight: 800; margin: 30px 0 10px 0; color: #f2f2f7; }
</style>
"""
st.markdown(MOBILE_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. 데이터 로직 (안전한 함수들)
# ---------------------------------------------------------
# 컬러 상수 (iOS 스타일)
C_UP = "#30D158"   # 초록
C_DOWN = "#FF453A" # 빨강
C_MID = "#8E8E93"  # 회색
C_HIGH = "#BF5AF2" # 보라
C_BLUE = "#64D2FF" # 파랑
C_WARN = "#FFD60A" # 노랑

@st.cache_data(ttl=1800)
def get_data(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if len(df) < 60: return None
        
        # 지표 계산
        df['MA20'] = df['Close'].rolling(20).mean()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        
        # OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Volatility
        df['Vol'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        df.dropna(inplace=True)
        return df
    except: return None

@st.cache_data(ttl=3600)
def get_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'name': info.get('longName', ticker),
            'cap': info.get('marketCap', 0)
        }
    except: return {'name': ticker, 'cap': 0}

def run_analysis(df):
    last = df.iloc[-1]
    
    # 1. 몬테카를로 (단순화)
    vol = df['Vol'].tail(30).mean() / 100
    target_price = last['Close'] * 1.3
    
    sims = []
    np.random.seed(42)
    for _ in range(300):
        p = last['Close']
        prices = []
        for _ in range(120): # 120일
            p *= (1 + np.random.normal(0, vol))
            prices.append(p)
        sims.append(max(prices))
        
    win_prob = np.mean([1 if s >= target_price else 0 for s in sims]) * 100
    peak_yield = (np.median(sims) - last['Close']) / last['Close'] * 100
    
    # 2. 스코어링
    score = 50
    drivers = [] # (Title, Value, Color)
    
    # Trend
    if last['Close'] > last['MA20']:
        score += 15
        drivers.append(('Trend', 'UP', C_UP))
    else:
        score -= 15
        drivers.append(('Trend', 'DOWN', C_DOWN))
        
    # Whale (OBV)
    obv_change = df['OBV'].diff(20).iloc[-1]
    if obv_change > 0:
        score += 15
        drivers.append(('Whale', 'BUY', C_UP))
    else:
        score -= 10
        drivers.append(('Whale', 'SELL', C_DOWN))
        
    # Volatility
    if last['Vol'] > 3.5:
        drivers.append(('Vol', 'HIGH', C_HIGH))
    else:
        drivers.append(('Vol', 'NORMAL', C_MID))
        
    score = max(0, min(100, int(score)))
    
    # Target Setup
    stop = last['Close'] - (last['ATR'] * 2)
    target = last['Close'] + (last['ATR'] * 3.5)
    
    # Color 결정
    main_color = C_UP if score >= 70 else (C_WARN if score >= 40 else C_DOWN)
    
    return {
        'score': score, 'main_color': main_color,
        'prob': win_prob, 'peak': peak_yield,
        'drivers': drivers,
        'close': last['Close'], 'stop': stop, 'target': target,
        'vol_ratio': (last['Volume'] / df['Volume'].tail(20).mean()) * 100
    }

# ---------------------------------------------------------
# 4. HTML 생성 (CSS 충돌 방지를 위해 인라인 스타일 사용)
# ---------------------------------------------------------
def make_html_output(ticker, info, res):
    # 시가총액
    cap = info['cap']
    if cap > 1e12: cap_str = f"{cap/1e12:.1f}T"
    elif cap > 1e9: cap_str = f"{cap/1e9:.1f}B"
    else: cap_str = "-"
    
    # 날짜 계산
    est_days = "45-60d" # 단순화

    # 드라이버 카드 HTML 조립
    drivers_html = ""
    for title, val, col in res['drivers']:
        drivers_html += f"""
        <div class="driver-item" style="border-top: 3px solid {col};">
            <div style="font-size:10px; color:#8e8e93;">{title}</div>
            <div style="font-size:13px; font-weight:800; color:#fff;">{val}</div>
        </div>
        """
        
    # 남은 칸 채우기 (UI 깨짐 방지)
    while len(res['drivers']) < 3:
        drivers_html += """<div class="driver-item"><div style="color:#333">-</div></div>"""
        res['drivers'].append(("", "", ""))

    # 최종 HTML 문자열 (f-string 사용하되 CSS 블록 없음)
    html = f"""
    <div class="mobile-wrapper">
        <div class="header-row">
            <div>
                <div class="ticker-name">{ticker}</div>
                <div class="ticker-sub">{info['name'][:15]}.. • {cap_str}</div>
            </div>
            <div class="score-badge">
                <div class="score-num" style="color: {res['main_color']};">{res['score']}</div>
                <div class="score-txt">AI SCORE</div>
            </div>
        </div>
        
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-label">Win Prob</div>
                <div class="stat-value" style="color: {C_UP if res['prob']>40 else '#fff'}">{res['prob']:.0f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Max Peak</div>
                <div class="stat-value" style="color: {C_HIGH}">+{res['peak']:.0f}%</div>
            </div>
        </div>
        
        <div class="main-card" style="border: 1px solid {res['main_color']}44;">
            <div class="card-title">🎯 Trading Setup</div>
            <div class="info-row">
                <span class="lbl">Entry Price</span>
                <span class="val">${res['close']:.2f}</span>
            </div>
            <div class="info-row">
                <span class="lbl">Target (TP)</span>
                <span class="val" style="color: {C_UP}">${res['target']:.2f}</span>
            </div>
            <div class="info-row">
                <span class="lbl">Stop Loss (SL)</span>
                <span class="val" style="color: {C_DOWN}">${res['stop']:.2f}</span>
            </div>
            <div style="margin-top:15px; pt-top:10px; border-top:1px dashed #333;">
                <div class="info-row" style="margin-bottom:0;">
                    <span class="lbl">Expected Time</span>
                    <span class="val" style="color: {C_BLUE}">{est_days}</span>
                </div>
            </div>
        </div>
        
        <div class="section-head">📊 Key Drivers</div>
        <div class="driver-grid">
            {drivers_html}
        </div>
        
        <div class="section-head">📑 AI Comment</div>
        <div class="main-card" style="font-size: 13px; line-height: 1.6; color: #ccc;">
            System detected <b>{res['drivers'][0][1]}</b> trend with <b>{res['drivers'][1][1]}</b> whale volume.
            Current volatility is <b>{res['drivers'][2][1]}</b>.
            Volume ratio is <b>{res['vol_ratio']:.0f}%</b> compared to average.
        </div>
        
        <div style="height: 50px;"></div>
    </div>
    """
    return html

# ---------------------------------------------------------
# 5. 실행부
# ---------------------------------------------------------
st.title("🐋 Whale Hunter")
st.caption("Mobile Optimized")

ticker_input = st.text_input("", placeholder="Ticker (e.g. NVDA)", value="NVDA")

if st.button("ANALYZE", use_container_width=True):
    if not ticker_input:
        st.error("Please enter a ticker")
    else:
        # 콤마 처리
        t = ticker_input.split(',')[0].strip().upper()
        
        with st.spinner(f"Analyzing {t}..."):
            try:
                df = get_data(t)
                info = get_info(t)
                
                if df is None:
                    st.error("Data Load Failed")
                else:
                    res = run_analysis(df)
                    # HTML 렌더링 (코드가 보이지 않음)
                    st.markdown(make_html_output(t, info, res), unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
