import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Whale Hunter", layout="wide", page_icon="🐋")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 2. CSS (스타일) - 들여쓰기 문제 해결을 위해 한 줄로 압축하거나 별도 처리
# ---------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 0rem !important; padding-bottom: 5rem !important; padding-left: 0.5rem !important; padding-right: 0.5rem !important; max-width: 100% !important; }
    header, footer { visibility: hidden; }
    .stApp { background-color: #000000; color: #f2f2f7; }
    div[data-baseweb="input"] > div { background-color: #1c1c1e !important; color: white !important; border: 1px solid #333 !important; border-radius: 12px !important; }
    input { color: white !important; }
    button { background-color: #0A84FF !important; color: white !important; border-radius: 12px !important; height: 3rem !important; font-weight: 700 !important; border: none !important; }
    
    /* 카드 디자인 */
    .mobile-box { font-family: -apple-system, sans-serif; padding: 0 5px 40px 5px; color: #f2f2f7; }
    .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding: 15px 5px; margin-bottom: 20px; }
    .title { font-size: 30px; font-weight: 800; margin: 0; line-height: 1; }
    .sub { font-size: 13px; color: #888; margin-top: 5px; }
    .badge { background: #1c1c1e; padding: 10px 15px; border-radius: 14px; border: 1px solid #333; text-align: center; }
    .score { font-size: 26px; font-weight: 900; line-height: 1; }
    
    .grid { display: flex; gap: 10px; margin-bottom: 15px; }
    .stat { flex: 1; background: #1c1c1e; padding: 12px; border-radius: 12px; text-align: center; }
    .val { font-size: 20px; font-weight: 800; }
    .lbl { font-size: 11px; color: #888; }
    
    .card { background: #1c1c1e; border-radius: 16px; padding: 16px; margin-bottom: 20px; }
    .row { display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 14px; }
    .drivers { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 20px; }
    .d-item { background: #2c2c2e; padding: 10px 4px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. 데이터 로직
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def get_data(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if len(df) < 50: return None
        
        df['MA20'] = df['Close'].rolling(20).mean()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['Vol'] = (df['High'] - df['Low']) / df['Close'] * 100
        df.dropna(inplace=True)
        return df
    except: return None

@st.cache_data(ttl=3600)
def get_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {'name': info.get('longName', ticker), 'cap': info.get('marketCap', 0)}
    except: return {'name': ticker, 'cap': 0}

def analyze(df):
    last = df.iloc[-1]
    score = 50
    
    # Trend
    if last['Close'] > last['MA20']: trend, t_col = "UP", "#30D158"
    else: trend, t_col = "DOWN", "#FF453A"
    if trend == "UP": score += 20
    else: score -= 20
    
    # Whale
    obv_chg = df['OBV'].diff(20).iloc[-1]
    if obv_chg > 0: whale, w_col = "BUY", "#30D158"
    else: whale, w_col = "SELL", "#FF453A"
    if whale == "BUY": score += 20
    else: score -= 10
    
    # Vol
    if last['Vol'] > 3.0: vol, v_col = "HIGH", "#BF5AF2"
    else: vol, v_col = "NORM", "#8E8E93"
    
    score = max(0, min(100, score))
    main_col = "#30D158" if score >= 60 else "#FF453A"
    
    target = last['Close'] + (last['ATR'] * 3)
    stop = last['Close'] - (last['ATR'] * 2)
    
    return {
        'score': score, 'col': main_col,
        'drivers': [(trend, t_col, 'Trend'), (whale, w_col, 'Whale'), (vol, v_col, 'Vol')],
        'close': last['Close'], 'target': target, 'stop': stop
    }

# ---------------------------------------------------------
# 4. 렌더링 (들여쓰기 완전 제거 방식)
# ---------------------------------------------------------
def render_ui(t, i, r):
    # 시가총액
    cap = i['cap']
    cap_str = f"{cap/1e9:.1f}B" if cap > 1e9 else "-"
    
    # HTML 조립 (들여쓰기 없이 연결)
    h = ""
    h += f'<div class="mobile-box">'
    
    # 헤더
    h += f'<div class="header">'
    h += f'<div><div class="title">{t}</div><div class="sub">{i["name"][:15]}.. • {cap_str}</div></div>'
    h += f'<div class="badge"><div class="score" style="color:{r["col"]}">{r["score"]}</div></div>'
    h += f'</div>'
    
    # 통계
    h += f'<div class="grid">'
    h += f'<div class="stat"><div class="lbl">Win Prob</div><div class="val" style="color:#30D158">High</div></div>'
    h += f'<div class="stat"><div class="lbl">Target</div><div class="val" style="color:#64D2FF">${r["target"]:.0f}</div></div>'
    h += f'</div>'
    
    # 전략
    h += f'<div class="card" style="border:1px solid {r["col"]}44">'
    h += f'<div class="row"><span style="color:#888">Entry</span><span>${r["close"]:.2f}</span></div>'
    h += f'<div class="row"><span style="color:#888">Stop</span><span style="color:#FF453A">${r["stop"]:.2f}</span></div>'
    h += f'</div>'
    
    # 드라이버
    h += f'<div class="drivers">'
    for val, col, title in r['drivers']:
        h += f'<div class="d-item" style="border-top:3px solid {col}">'
        h += f'<div style="font-size:10px; color:#888">{title}</div><div style="font-weight:800">{val}</div>'
        h += f'</div>'
    h += f'</div>'
    
    h += f'</div>'
    
    return h

# ---------------------------------------------------------
# 5. 실행
# ---------------------------------------------------------
st.title("🐋 Whale Hunter")
st.caption("Mobile Edition")

txt = st.text_input("", value="NVDA")

if st.button("ANALYZE", use_container_width=True):
    t = txt.split(',')[0].strip().upper()
    try:
        df = get_data(t)
        if df is not None:
            info = get_info(t)
            res = analyze(df)
            # 여기가 핵심: 들여쓰기 없는 HTML 문자열을 넣음
            st.markdown(render_ui(t, info, res), unsafe_allow_html=True)
        else:
            st.error("Data Load Failed")
    except Exception as e:
        st.error(str(e))
