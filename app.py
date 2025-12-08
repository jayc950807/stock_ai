# ... (앞부분의 import 및 데이터 처리 함수들은 기존과 동일하게 유지) ...

# [UI 렌더링 HTML 생성 함수 - 모바일 최적화 버전]
def get_render_html(ticker, mkt_cap, analysis, monte_res, score_history, stock_info):
    sim_df, opt, pes, mean, win_prob, expected_date_str, peak_yield = monte_res

    # 시가총액 문자열
    if mkt_cap > 0:
        val_won = mkt_cap * 1400 # 환율 대략 적용
        if val_won > 100_000_000_000_000: cap_str = f"{val_won/100_000_000_000_000:.1f}조"
        elif val_won > 1_000_000_000_000: cap_str = f"{val_won/1_000_000_000_000:.1f}조"
        else: cap_str = f"{val_won/100_000_000_000:.0f}천억"
    else: cap_str = "-"

    peak_color = C_PURP if peak_yield > 50 else (C_BULL if peak_yield > 0 else C_BEAR)
    peak_str = f"MAX +{peak_yield:.0f}%" if peak_yield > 0 else f"MAX {peak_yield:.0f}%"

    # --- 데이터 준비 (기존 로직 유지) ---
    sorted_history = score_history 
    dates = [item['day'] for item in sorted_history]
    scores = [item['score'] for item in sorted_history]
    gaps = [item['adv']['whale_gap'] for item in sorted_history]
    ads = [item['adv']['ad_signal'] for item in sorted_history]
    mfis = [item['adv']['mfi_signal'] for item in sorted_history]
    pocs = [item['adv']['poc_signal'] for item in sorted_history]

    def get_style_content(label, v):
        txt, col, bg, fw = "-", "#666", "transparent", "normal"
        if label.startswith("Date"): return v, "#888", "transparent", "normal"
        elif label == "AI Score":
            txt = str(v)
            if v >= 80: col, fw = C_BULL, "bold"
            elif v >= 60: col, fw = C_CYAN, "bold"
            elif v <= 40: col, fw = C_BEAR, "bold"
            else: col = C_NEUT
        elif label == "Whale Gap":
            if v > 10: txt, col, bg, fw = "BUY", C_BULL, "rgba(0, 230, 118, 0.1)", "bold"
            elif v < -10: txt, col, bg, fw = "SELL", C_BEAR, "rgba(255, 82, 82, 0.1)", "bold"
            else: txt = "WAIT"
        elif label == "Smart Money":
            if v == "Bull": txt, col, bg, fw = "BUY", C_BULL, "rgba(0, 230, 118, 0.1)", "bold"
            elif v == "Bear": txt, col, bg, fw = "SELL", C_BEAR, "rgba(255, 82, 82, 0.1)", "bold"
            else: txt = "WAIT"
        elif label == "RSI/MFI":
            if v == "Oversold": txt, col, bg, fw = "BUY", C_BULL, "rgba(0, 230, 118, 0.1)", "bold"
            elif v == "Overbot": txt, col, bg, fw = "SELL", C_BEAR, "rgba(255, 82, 82, 0.1)", "bold"
            else: txt = "WAIT"
        elif label == "POC Level":
            if v == "Bull": txt, col, bg, fw = "BUY", C_BULL, "rgba(0, 230, 118, 0.1)", "bold"
            elif v == "Bear": txt, col, bg, fw = "SELL", C_BEAR, "rgba(255, 82, 82, 0.1)", "bold"
            else: txt = "-"
        return txt, col, bg, fw

    def make_row_html(label, values, is_header=False):
        row_html = f"<tr><td style='position:sticky; left:0; background:#1c1c1e; z-index:10; text-align:left; color:#999; font-size:12px; padding:8px 10px; border-right:1px solid #333; width:80px;'>{label}</td>"
        for v in values:
            txt, col, bg, fw = get_style_content(label, v)
            if is_header: row_html += f"<td style='color:#bbb; font-size:11px; padding:6px 4px; min-width:40px;'>{txt}</td>"
            else: row_html += f"<td style='color:{col}; background:{bg}; font-weight:{fw}; font-size:11px; padding:6px 4px; border-radius:4px;'>{txt}</td>"
        row_html += "</tr>"
        return row_html

    hist_table = """<div style="overflow-x:auto; -webkit-overflow-scrolling:touch; margin-top:10px; padding-bottom:5px;">
    <table style="width:100%; border-collapse:collapse; text-align:center; white-space:nowrap;">"""
    hist_table += make_row_html("Date", dates, is_header=True)
    hist_table += make_row_html("AI Score", scores)
    hist_table += make_row_html("Whale Gap", gaps)
    hist_table += make_row_html("Smart Money", ads)
    hist_table += make_row_html("RSI/MFI", mfis)
    hist_table += "</table></div>"
    
    # 카드형 UI (Key Drivers)
    cards_html = "<div style='display:grid; grid-template-columns: 1fr 1fr; gap:8px;'>"
    for c in analysis['cards']:
        cards_html += f"""
        <div style="background:#2c2c2e; padding:12px 10px; border-radius:12px; border-left:3px solid {c['col']}; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:11px; color:#8e8e93; margin-bottom:2px;">{c['title']}</div>
            <div style="font-size:13px; font-weight:600; color:#f2f2f7;">{c['stat']}</div>
        </div>"""
    cards_html += "</div>"

    # 기술적 지표 (2컬럼 -> 모바일에서는 컴팩트 리스트)
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
    
    # CSS: 아이폰 스타일 (San Francisco font, Rounded corners, Deep Dark)
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;600;800&display=swap');
        
        /* 전체 컨테이너: 패딩 줄이고 배경색 OLED Black */
        .mobile-container {{
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #000000;
            color: #f2f2f7;
            padding: 0px 5px 40px 5px; /* 모바일 좌우 여백 최소화 */
            max-width: 100%;
            overflow-x: hidden;
        }}

        /* 헤더 영역 */
        .header-section {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 10px 5px;
            border-bottom: 1px solid #333;
        }}
        .ticker-title {{ font-size: 32px; font-weight: 800; letter-spacing: -0.5px; margin: 0; line-height: 1.1; }}
        .ticker-info {{ font-size: 13px; color: #8e8e93; margin-top: 4px; }}
        
        /* 핵심 스코어 뱃지 */
        .score-badge {{
            text-align: center;
            background: #1c1c1e;
            padding: 8px 16px;
            border-radius: 16px;
            border: 1px solid #333;
        }}
        .score-val {{ font-size: 28px; font-weight: 800; color: {analysis['color']}; line-height: 1; }}
        .score-lbl {{ font-size: 10px; color: #636366; font-weight: 600; margin-top: 2px; }}

        /* 섹션 공통 */
        .section-title {{
            font-size: 17px; font-weight: 700; color: #f2f2f7; margin: 25px 0 10px 0; letter-spacing: -0.3px;
        }}
        .card-box {{
            background: #1c1c1e;
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}

        /* 트레이딩 전략 박스 */
        .strategy-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 14px; }}
        .strategy-lbl {{ color: #8e8e93; }}
        .strategy-val {{ font-weight: 600; }}

        /* 리포트 텍스트 스타일 재정의 */
        .ai-report li {{ margin-bottom: 8px; font-size: 14px; color: #d1d1d6; line-height: 1.5; }}
        .ai-report b {{ color: #fff; font-weight: 600; }}
    </style>
    """

    # HTML 구조 조립 (모바일 레이아웃)
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
                <div style="font-size:11px; color:#8e8e93;">Win Probability</div>
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

        <div class="section-title">📈 Recent Momentum</div>
        {hist_table}
        <div style="font-size:10px; color:#666; text-align:right; margin-top:4px;">*Scroll horizontally</div>

        <div class="section-title">🎛 Technical Details</div>
        {tech_html}
        
        <div style="height:30px;"></div> </div>
    """
    return html

# --------------------------
# Main Streamlit App Layout (수정됨)
# --------------------------
# 1. layout="centered" 대신 "wide"를 쓰되 CSS로 폭 제어 (모바일은 꽉 차게)
# st.set_page_config 호출부는 맨 위에 있으므로 그대로 두되, 아래 CSS를 추가.

# 전체 스타일 조정 (Streamlit 기본 패딩 제거)
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🐋 Whale Hunter")
st.caption("AI-Powered Technical Analysis") # 모바일용으로 간결하게

# 캐시 로딩 (백그라운드)
ref_cache = load_reference_cache()

# 입력창
ticker_input = st.text_input("", placeholder="Ticker (e.g., NVDA, TSLA)", value="NVDA")

if st.button("Analyze", type="primary", use_container_width=True): # 버튼 꽉 차게
    if not ticker_input:
        st.warning("Please enter a ticker.")
    else:
        # 벤치마크 로딩
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
                
                # HTML 렌더링 (Iframe 제거 -> st.markdown 사용)
                html_out = get_render_html(ticker, mkt_cap, analysis, monte_res, score_history, stock_info)
                status_text.empty() 
                
                # [핵심 수정] components.v1.html 대신 markdown(unsafe_allow_html=True) 사용
                # 이렇게 하면 iframe 프레임이 생기지 않고, 페이지의 일부처럼 렌더링되어 스크롤이 자연스럽습니다.
                st.markdown(html_out, unsafe_allow_html=True)
                st.markdown("---") # 종목 간 구분선
                
            except Exception as e:
                status_text.error(f"Error: {str(e)}")
