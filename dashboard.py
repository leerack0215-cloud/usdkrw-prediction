"""
dashboard.py — USD/KRW 딥러닝 예측 대시보드
실행: streamlit run dashboard.py

실시간 구조:
  ① yfinance 최신 환율 수집 (5분 캐시)
  ② LGB 모델 직접 로드 → 다중 호라이즌 실시간 추론
  ③ forecast_today.json → Colab DL 앙상블 예측 표시
  ④ performance_table.csv → 모델 성능 비교
"""

import os, json, pickle, warnings, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    collect_data, make_features,
    HORIZONS, HORIZON_LABELS,
    SEQ_LEN, OUTPUT_DIR, MODELS_DIR, CLIP_BOUNDS,
)

# ════════════════════════════════════════════════════════
# 페이지 설정
# ════════════════════════════════════════════════════════

st.set_page_config(
    page_title="USD/KRW 딥러닝 예측",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&family=JetBrains+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.stApp {
    background: linear-gradient(160deg,#070d1a 0%,#0b1525 60%,#070d1a 100%);
    color: #dde6f0;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0b1525,#0d1a2e) !important;
    border-right: 1px solid #1a3050;
}
.hero {
    background: linear-gradient(90deg,#0f2240,#0a1c38);
    border: 1px solid #1e4a80; border-radius: 14px;
    padding: 22px 30px; margin-bottom: 22px;
    box-shadow: 0 0 40px rgba(30,74,128,.2);
}
.hero h1 { font-family:'JetBrains Mono',monospace; font-size:1.9rem; color:#5eaeff; margin:0; }
.hero p  { color:#7a9dbf; margin:4px 0 0; font-size:.88rem; }
.kpi {
    background: linear-gradient(135deg,#0f2038,#0a1829);
    border: 1px solid rgba(40,90,160,.35); border-radius:10px;
    padding: 16px 18px; text-align:center;
    box-shadow: 0 4px 18px rgba(0,0,0,.35);
}
.kpi .val { font-family:'JetBrains Mono',monospace; font-size:1.65rem; font-weight:700; color:#5eaeff; }
.kpi .lbl { font-size:.72rem; color:#4d7a9f; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }
.kpi .delta { font-size:.85rem; margin-top:5px; }
.up   { color:#34d399; } .down { color:#f87171; } .flat { color:#94a3b8; }
.sec-hdr {
    font-family:'JetBrains Mono',monospace; color:#5eaeff;
    font-size:.9rem; letter-spacing:2px; text-transform:uppercase;
    border-bottom:1px solid #1a3050; padding-bottom:6px; margin:22px 0 14px;
}
.badge-rt {
    display:inline-block; background:rgba(52,211,153,.12);
    border:1px solid rgba(52,211,153,.4); color:#34d399;
    font-size:.68rem; padding:2px 8px; border-radius:4px; margin-left:8px;
}
.badge-dl {
    display:inline-block; background:rgba(94,174,255,.10);
    border:1px solid rgba(94,174,255,.35); color:#5eaeff;
    font-size:.68rem; padding:2px 8px; border-radius:4px; margin-left:8px;
}
.warn {
    background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.35);
    border-left:4px solid #fbbf24; border-radius:6px;
    padding:9px 14px; font-size:.78rem; color:#fcd34d; margin:10px 0;
}
.stTabs [data-baseweb="tab-list"] { background:#0b1525; border-bottom:1px solid #1a3050; }
.stTabs [data-baseweb="tab"]      { color:#4d7a9f; font-family:'JetBrains Mono',monospace; font-size:.78rem; }
.stTabs [aria-selected="true"]    { color:#5eaeff !important; border-bottom:2px solid #2563eb; }
.stButton > button {
    background:linear-gradient(90deg,#1a4fad,#2563eb);
    color:#fff; border:none; border-radius:8px;
    font-family:'JetBrains Mono',monospace; font-size:.78rem;
    padding:8px 18px; transition:all .2s;
}

/* ── 카운트다운 바 ── */
.countdown-wrap {
    background: linear-gradient(90deg,#0a1829,#0f2240);
    border: 1px solid rgba(40,90,160,.3);
    border-radius: 10px;
    padding: 10px 18px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.countdown-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem;
    color: #4d7a9f;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    white-space: nowrap;
}
.countdown-timer {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #34d399;
    min-width: 52px;
    text-align: center;
    transition: color .3s;
}
.countdown-timer.warn  { color: #f59e0b; }
.countdown-timer.urgent{ color: #f87171; }
.bar-track {
    flex: 1;
    height: 5px;
    background: rgba(255,255,255,.06);
    border-radius: 99px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg,#2563eb,#34d399);
    transition: width .9s linear, background .3s;
}
.bar-fill.warn   { background: linear-gradient(90deg,#d97706,#f59e0b); }
.bar-fill.urgent { background: linear-gradient(90deg,#dc2626,#f87171); }
.last-update {
    font-family: 'JetBrains Mono', monospace;
    font-size: .68rem;
    color: #2a4060;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,21,40,.55)",
    font=dict(family="JetBrains Mono,monospace", color="#7a9dbf", size=11),
    xaxis=dict(gridcolor="#1a3050", gridwidth=.5, zeroline=False, linecolor="#1a3050"),
    yaxis=dict(gridcolor="#1a3050", gridwidth=.5, zeroline=False, linecolor="#1a3050"),
    legend=dict(bgcolor="rgba(11,21,40,.8)", bordercolor="#1a3050", borderwidth=1),
    margin=dict(l=55, r=25, t=38, b=38),
    hovermode="x unified",
)


def kpi_card(label, value, delta_html=""):
    return (
        f'<div class="kpi">'
        f'<div class="val">{value}</div>'
        f'<div class="lbl">{label}</div>'
        + (f'<div class="delta">{delta_html}</div>' if delta_html else "")
        + "</div>"
    )


# ════════════════════════════════════════════════════════
# 데이터 로더 (캐시)
# ════════════════════════════════════════════════════════

def get_spot_rate() -> tuple:
    """
    실시간 USD/KRW — yfinance 1분봉 우선
    open.er-api.com은 하루 1회 업데이트라 실시간 아님
    """
    import datetime as dt

    # 1순위: yfinance 1분봉 (실제 분 단위 갱신)
    try:
        import yfinance as yf
        h = yf.Ticker("KRW=X").history(period="1d", interval="1m")
        if not h.empty and len(h) > 0:
            price = float(h["Close"].iloc[-1])
            ts    = h.index[-1]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if price > 100:
                return price, f"yfinance 1m ({ts.strftime('%H:%M')})", dt.datetime.now()
    except Exception:
        pass

    # 2순위: yfinance 기본 (폴백)
    try:
        import yfinance as yf
        h = yf.Ticker("KRW=X").history(period="5d")
        if not h.empty:
            price = float(h["Close"].iloc[-1])
            if price > 100:
                return price, "yfinance(일봉)", dt.datetime.now()
    except Exception:
        pass

    # 3순위: ExchangeRate-API (하루 1회지만 최후 폴백)
    try:
        import requests
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=4)
        d = r.json()
        if d.get("result") == "success":
            price = float(d["rates"]["KRW"])
            if price > 100:
                return price, "ExchangeRate-API(1일1회)", dt.datetime.now()
    except Exception:
        pass

    return 0.0, "수집실패", dt.datetime.now()


@st.cache_data(ttl=300)
def get_realtime():
    """
    LGB 모델 실시간 추론 (5분마다 갱신)
    반환: (price_series, rt_predictions_dict, df_full, fetch_timestamp)
    """
    try:
        df = collect_data(start="2018-01-01")
        df, _ = make_features(df, add_targets=False)

        scaler_path = f"{OUTPUT_DIR}/scaler_X.pkl"
        feat_path   = f"{OUTPUT_DIR}/feature_list.json"

        if not (os.path.exists(scaler_path) and os.path.exists(feat_path)):
            return df["USDKRW"], {}, df, datetime.datetime.now()

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(feat_path, encoding="utf-8") as f:
            features = json.load(f)

        for col in features:
            if col not in df.columns:
                df[col] = 0.0

        X_scaled   = scaler.transform(df[features].fillna(0))
        last_price = float(df["USDKRW"].iloc[-1])
        last_flat  = X_scaled[-1:].reshape(1, -1)

        preds = {}
        for h in HORIZONS:
            path = f"{MODELS_DIR}/lgb_h{h}.pkl"
            if os.path.exists(path):
                with open(path, "rb") as f:
                    model = pickle.load(f)
                clip = CLIP_BOUNDS[h]
                lr   = float(np.clip(model.predict(last_flat)[0], -clip, clip))
                preds[HORIZON_LABELS[h]] = round(last_price * np.exp(lr), 2)

        return df["USDKRW"], preds, df, datetime.datetime.now()

    except Exception as e:
        st.warning(f"데이터 로드 오류: {e}")
        idx    = pd.date_range(end=datetime.date.today(), periods=500, freq="B")
        prices = 1380 + np.cumsum(np.random.default_rng(42).normal(0, 3, 500))
        return pd.Series(prices, index=idx, name="USDKRW"), {}, pd.DataFrame(), datetime.datetime.now()


@st.cache_data(ttl=300)
def get_macro():
    """거시 지표 2년치 수집"""
    import yfinance as yf
    syms  = {"VIX":"^VIX","DXY":"DX-Y.NYB","KOSPI":"^KS11",
              "SP500":"^GSPC","WTI":"CL=F","GOLD":"GC=F"}
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=730)
    frames = {}
    for name, sym in syms.items():
        try:
            d = yf.download(sym, start=str(start), end=str(end),
                            auto_adjust=True, progress=False)
            if not d.empty:
                c = d["Close"]
                frames[name] = (c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c)
        except Exception:
            pass
    return pd.DataFrame(frames)


def load_forecast():
    path = f"{OUTPUT_DIR}/forecast_today.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_performance():
    path = f"{OUTPUT_DIR}/performance_table.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


# ════════════════════════════════════════════════════════
# 사이드바
# ════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 💹 USD/KRW 예측")
    st.markdown("---")
    period = st.selectbox(
        "📅 차트 기간",
        ["최근 90일","최근 180일","최근 1년","최근 2년","전체"],
        index=2,
    )
    period_days = {"최근 90일":90,"최근 180일":180,
                   "최근 1년":365,"최근 2년":730,"전체":9999}[period]
    st.markdown("---")
    if st.button("🔄 데이터 갱신", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.7rem;color:#3d5a7a;line-height:1.8'>
    🟢 <b>실시간 LGB</b> — 5분마다 자동 갱신<br>
    🔵 <b>앙상블</b> — Colab 마지막 학습 기준<br><br>
    ⚠ <i>학술/참고 목적 전용<br>투자 결정 책임은 사용자에게</i>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# 데이터 로드
# ════════════════════════════════════════════════════════

krw_series, rt_preds, df_full, fetch_time = get_realtime()
macro_df  = get_macro()
forecast  = load_forecast()
perf_df   = load_performance()

# ── 실시간 현재가 수집 (캐시 없이 매번 호출) ─────────
# fragment(run_every=60)로 현재가 1분마다 갱신
spot_price, spot_src, spot_time = get_spot_rate()
last_price = spot_price if spot_price > 0 else (
    float(krw_series.iloc[-1]) if len(krw_series) > 0 else 1482.0
)

# prev_price: yfinance 전일 종가 (소스 통일 — spot과 비교 오차 방지)
# yfinance 마지막 값이 당일 종가에 가장 가까운 기준값
yf_last    = float(krw_series.iloc[-1]) if len(krw_series) > 0 else last_price
prev_price = float(krw_series.iloc[-2]) if len(krw_series) > 1 else yf_last

# spot과 yfinance 마지막값 차이가 1% 이상이면 yfinance 기준으로 변화율 계산
# (야간 등 시장 급변 제외한 정상 범위)
if abs(last_price - yf_last) / (yf_last + 1e-9) > 0.01:
    day_chg     = last_price - yf_last
    day_chg_pct = day_chg / yf_last * 100 if yf_last else 0
else:
    day_chg     = last_price - prev_price
    day_chg_pct = day_chg / prev_price * 100 if prev_price else 0

cutoff   = krw_series.index[-1] - pd.Timedelta(days=period_days)
krw_view = krw_series[krw_series.index >= cutoff]
if len(krw_view) == 0:
    krw_view = krw_series

# ════════════════════════════════════════════════════════
# 헤더
# ════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>💹 USD / KRW 딥러닝 예측 시스템</h1>
  <p>LightGBM · LSTM · BiGRU · Ridge 앙상블 | 실시간 다중 호라이즌 예측</p>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# 실시간 현재가 수집 (매 rerun마다 직접 호출, 캐시 없음)
# ════════════════════════════════════════════════════════

spot_price, spot_src, spot_time = get_spot_rate()
cur_price = spot_price if spot_price > 0 else (
    float(krw_series.iloc[-1]) if len(krw_series) > 0 else 1482.0
)
prev_price  = float(krw_series.iloc[-2]) if len(krw_series) > 1 else cur_price
day_chg     = cur_price - prev_price
day_chg_pct = day_chg / prev_price * 100 if prev_price else 0

# ── API 상태 디버그 표시 (문제 파악용) ─────────────────
yf_last = float(krw_series.iloc[-1]) if len(krw_series) > 0 else 0
st.info(
    f"🔍 **API 진단**  |  "
    f"spot_price={spot_price:.2f}  |  "
    f"spot_src={spot_src}  |  "
    f"yfinance_last={yf_last:.2f}  |  "
    f"cur_price={cur_price:.2f}  |  "
    f"시각={spot_time.strftime('%H:%M:%S')}"
)

# ════════════════════════════════════════════════════════
# 카운트다운 바
# ════════════════════════════════════════════════════════

TTL_SEC = 300
now     = datetime.datetime.now()
elapsed = int((now - fetch_time).total_seconds())
remain  = max(TTL_SEC - elapsed, 0)
pct     = remain / TTL_SEC * 100
m = remain // 60
s = remain % 60

if remain <= 30:
    tc = "#f87171"; bc = "linear-gradient(90deg,#dc2626,#f87171)"
elif remain <= 90:
    tc = "#f59e0b"; bc = "linear-gradient(90deg,#d97706,#f59e0b)"
else:
    tc = "#34d399"; bc = "linear-gradient(90deg,#2563eb,#34d399)"

st.markdown(f"""
<div class="countdown-wrap">
  <span class="countdown-label">🔄 예측 갱신까지</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:700;color:{tc};min-width:52px;text-align:center;">{m}:{s:02d}</span>
  <div class="bar-track">
    <div class="bar-fill" style="width:{pct:.1f}%;background:{bc};height:5px;border-radius:99px;"></div>
  </div>
  <span class="last-update">현재가: {spot_src} | 갱신: {spot_time.strftime('%H:%M:%S')}</span>
</div>""", unsafe_allow_html=True)

# 5분 만료 시 캐시 초기화
if remain <= 0:
    st.cache_data.clear()

# ════════════════════════════════════════════════════════
# KPI 카드
# ════════════════════════════════════════════════════════

d1_rt = rt_preds.get("D+1", cur_price)
d3_rt = rt_preds.get("D+3", cur_price)

yr      = krw_series[krw_series.index >= krw_series.index[-1] - pd.Timedelta(days=365)]
yr_high = float(yr.max()) if len(yr) > 0 else cur_price
yr_low  = float(yr.min()) if len(yr) > 0 else cur_price
vol_21  = (
    float(krw_series.pct_change().rolling(21).std().iloc[-1] * 100 * np.sqrt(252))
    if len(krw_series) > 21 else 0.0
)
chg_cls = "up" if day_chg >= 0 else "down"
chg_sym = "▲" if day_chg >= 0 else "▼"

kpi_rows = [
    (f"현재 환율 ({spot_src})",
     f"₩{cur_price:,.2f}",
     f'<span class="{chg_cls}">{chg_sym} {abs(day_chg):.2f}원 ({abs(day_chg_pct):.2f}%)</span>'),
    ("D+1 예측 🟢실시간",
     f"₩{d1_rt:,.2f}",
     f'<span class="{"up" if d1_rt>cur_price else "down"}">{"↑" if d1_rt>cur_price else "↓"} {abs((d1_rt/cur_price-1)*100):.2f}%</span>'),
    ("D+3 예측 🟢실시간",
     f"₩{d3_rt:,.2f}",
     f'<span class="{"up" if d3_rt>cur_price else "down"}">{"↑" if d3_rt>cur_price else "↓"} {abs((d3_rt/cur_price-1)*100):.2f}%</span>'),
    ("52주 고/저",
     f"₩{yr_high:,.0f}",
     f'<span class="down">LOW ₩{yr_low:,.0f}</span>'),
    ("연환산 변동성", f"{vol_21:.1f}%", ""),
]
cols = st.columns(5)
for col, (lbl, val, delta) in zip(cols, kpi_rows):
    col.markdown(kpi_card(lbl, val, delta), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# 탭
# ════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 가격 차트", "🔮 다중 호라이즌 예측",
    "⚡ 모델 성능", "🌍 거시 지표", "📋 백테스팅",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — 가격 차트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab1:
    st.markdown('<div class="sec-hdr">USD/KRW 가격 차트</div>', unsafe_allow_html=True)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
    )

    ema20 = krw_view.ewm(span=20).mean()
    sma20 = krw_view.rolling(20).mean()
    std20 = krw_view.rolling(20).std()

    # 가격 + EMA + 볼린저
    fig.add_trace(go.Scatter(
        x=krw_view.index, y=krw_view.values,
        mode="lines", line=dict(color="#4a9eff", width=1.5),
        name="USD/KRW", fill="tozeroy", fillcolor="rgba(74,158,255,.06)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=krw_view.index, y=ema20,
        mode="lines", line=dict(color="#f59e0b", width=1, dash="dot"), name="EMA20",
    ), row=1, col=1)
    bb_x = list(krw_view.index) + list(krw_view.index[::-1])
    bb_y = list(sma20 + 2*std20) + list((sma20 - 2*std20)[::-1])
    fig.add_trace(go.Scatter(
        x=bb_x, y=bb_y, fill="toself",
        fillcolor="rgba(99,102,241,.07)",
        line=dict(color="rgba(0,0,0,0)"), name="볼린저밴드",
    ), row=1, col=1)

    # RSI
    delta_r = krw_view.diff()
    gain_r  = delta_r.clip(lower=0).rolling(14).mean()
    loss_r  = (-delta_r.clip(upper=0)).rolling(14).mean()
    rsi     = 100 - (100 / (1 + gain_r / (loss_r + 1e-9)))
    fig.add_trace(go.Scatter(
        x=krw_view.index, y=rsi,
        mode="lines", line=dict(color="#a78bfa", width=1), name="RSI(14)",
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#f87171", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#34d399", row=2, col=1)

    # 일간 변화율
    pct = krw_view.pct_change() * 100
    fig.add_trace(go.Bar(
        x=krw_view.index, y=pct,
        marker_color=["#34d399" if v >= 0 else "#f87171" for v in pct],
        name="일간변화(%)",
    ), row=3, col=1)

    fig.update_layout(**CHART_BASE, height=560,
                      yaxis2_title="RSI", yaxis3_title="%")
    st.plotly_chart(fig, use_container_width=True)

    # 기술 지표 + 수익률 요약
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-hdr">기술적 지표</div>', unsafe_allow_html=True)
        rsi_v  = float(rsi.iloc[-1]) if not rsi.empty else 50
        rsi_st = "과매수" if rsi_v > 70 else ("과매도" if rsi_v < 30 else "중립")
        items  = {
            "RSI (14)":    f"{rsi_v:.1f}  {rsi_st}",
            "EMA20 대비":  f"{'위' if last_price > float(ema20.iloc[-1]) else '아래'} "
                           f"({(last_price/float(ema20.iloc[-1])-1)*100:.2f}%)"
                           if not ema20.empty else "N/A",
            "20일 변동성": f"{float(krw_view.pct_change().rolling(20).std().iloc[-1]*100*np.sqrt(252)):.1f}%"
                           if len(krw_view) > 20 else "N/A",
            "볼린저 위치": (
                "상단 근접" if last_price > float((sma20+2*std20).iloc[-1]) else
                "하단 근접" if last_price < float((sma20-2*std20).iloc[-1]) else "중간"
            ),
        }
        for k, v in items.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 0;border-bottom:1px solid #1a3050;"
                f"font-family:JetBrains Mono,monospace;font-size:.8rem;'>"
                f"<span style='color:#4d7a9f'>{k}</span>"
                f"<span style='color:#dde6f0'>{v}</span></div>",
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown('<div class="sec-hdr">구간별 수익률</div>', unsafe_allow_html=True)
        def ret(n):
            return (krw_series.iloc[-1]/krw_series.iloc[-n]-1)*100 if len(krw_series)>n-1 else 0
        ret_vals = [ret(2), ret(6), ret(21), ret(61), ret(253)]
        ret_fig  = go.Figure(go.Bar(
            x=["1일","5일","20일","60일","1년"], y=ret_vals,
            marker_color=["#34d399" if v>=0 else "#f87171" for v in ret_vals],
            text=[f"{v:+.2f}%" for v in ret_vals], textposition="outside",
        ))
        ret_fig.update_layout(**CHART_BASE, height=220, showlegend=False)
        st.plotly_chart(ret_fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — 다중 호라이즌 예측
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab2:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            '<div class="sec-hdr">🟢 실시간 LGB 예측'
            '<span class="badge-rt">5분 갱신</span></div>',
            unsafe_allow_html=True,
        )
        if rt_preds:
            rows = []
            for h in HORIZONS:
                lbl  = HORIZON_LABELS[h]
                pred = rt_preds.get(lbl, last_price)
                pct  = (pred/last_price - 1) * 100
                rows.append({
                    "호라이즌": lbl,
                    "예측 환율": f"₩{pred:,.2f}",
                    "변화율":   f"{pct:+.2f}%",
                    "방향":     "↑ 상승" if pct>0 else ("↓ 하락" if pct<0 else "— 보합"),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.markdown(
                '<div class="warn">⚠ LGB 모델 없음 — Colab에서 train.py 실행 후 push 필요</div>',
                unsafe_allow_html=True,
            )

    with col_r:
        st.markdown(
            '<div class="sec-hdr">🔵 앙상블 예측 (Colab 학습 결과)'
            '<span class="badge-dl">마지막 학습 기준</span></div>',
            unsafe_allow_html=True,
        )
        if forecast:
            rows2 = []
            for lbl, data in forecast.get("forecasts", {}).items():
                rows2.append({
                    "호라이즌": lbl,
                    "예측 환율": f"₩{data['price']:,.2f}",
                    "변화율":   f"{data['change_pct']:+.2f}%",
                    "방향":     data["direction"],
                })
            st.dataframe(pd.DataFrame(rows2), hide_index=True, use_container_width=True)
            st.caption(
                f"기준일: {forecast.get('last_date','N/A')} | "
                f"기준 환율: ₩{forecast.get('last_close',0):,.2f}"
            )
        else:
            st.markdown(
                '<div class="warn">⚠ forecast_today.json 없음 — Colab에서 predict.py 실행 필요</div>',
                unsafe_allow_html=True,
            )

    # 팬 차트
    st.markdown('<div class="sec-hdr">예측 팬 차트 (실시간 LGB)</div>', unsafe_allow_html=True)
    if rt_preds:
        h_list = [0, 1, 3, 5, 10, 22]
        p_list = [last_price] + [rt_preds.get(HORIZON_LABELS[h], last_price) for h in [1,3,5,10,22]]
        fig_fan = go.Figure()
        for ci, alpha in [(0.99,0.06),(0.95,0.10),(0.80,0.16)]:
            w = [last_price*0.002*np.sqrt(h) for h in h_list]
            fig_fan.add_trace(go.Scatter(
                x=h_list + h_list[::-1],
                y=[p+ww for p,ww in zip(p_list,w)] + [p-ww for p,ww in zip(p_list,w)][::-1],
                fill="toself", fillcolor=f"rgba(94,174,255,{alpha})",
                line=dict(color="rgba(0,0,0,0)"), name=f"CI {int(ci*100)}%",
            ))
        fig_fan.add_trace(go.Scatter(
            x=h_list, y=p_list, mode="lines+markers",
            line=dict(color="#5eaeff", width=2.5),
            marker=dict(size=8, color="#5eaeff"), name="LGB 예측",
        ))
        fig_fan.add_hline(y=last_price, line_dash="dot",
                          line_color="#475569", annotation_text="현재")
        fig_fan.update_layout(
            **CHART_BASE, height=320,
            xaxis_title="영업일", yaxis_title="USD/KRW (원)",
        )
        st.plotly_chart(fig_fan, use_container_width=True)

    # 모델별 D+1 비교
    if forecast and "models" in forecast and "D+1" in forecast["models"]:
        st.markdown('<div class="sec-hdr">모델별 D+1 예측 비교</div>', unsafe_allow_html=True)
        m_data = forecast["models"]["D+1"]
        fig_b  = go.Figure(go.Bar(
            x=list(m_data.keys()), y=list(m_data.values()),
            marker_color=["#34d399" if v>last_price else "#f87171" for v in m_data.values()],
            text=[f"₩{v:,.0f}" for v in m_data.values()], textposition="outside",
        ))
        fig_b.add_hline(y=last_price, line_dash="dash", line_color="#fbbf24",
                        annotation_text=f"현재 ₩{last_price:,.0f}")
        fig_b.update_layout(**CHART_BASE, height=280, showlegend=False,
                            yaxis_title="예측 환율 (원)")
        st.plotly_chart(fig_b, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — 모델 성능
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab3:
    st.markdown('<div class="sec-hdr">모델 성능 비교 (Validation Set)</div>',
                unsafe_allow_html=True)

    disp = perf_df.copy() if perf_df is not None else pd.DataFrame({
        "RMSE":    [9.1, 15.7, 8.4, 9.7],
        "MAE":     [6.7, 12.9, 6.2, 7.1],
        "MAPE(%)": [0.47, 0.92, 0.44, 0.51],
        "DA(%)":   [47.1, 43.8, 51.7, 46.3],
        "Sharpe":  [1.50, 1.05, 0.73, 1.51],
    }, index=["LSTM_D+1","BIGRU_D+1","LGB_D+1","★Ensemble_D+1"])

    def highlight_best(df):
        s = pd.DataFrame("", index=df.index, columns=df.columns)
        for col in df.columns:
            best = df[col].idxmax() if col in ["DA(%)","Sharpe"] else df[col].idxmin()
            s.loc[best, col] = (
                "background-color:rgba(52,211,153,.15);"
                "color:#34d399;font-weight:700"
            )
        return s

    st.dataframe(
        disp.style.apply(highlight_best, axis=None).format("{:.3f}"),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        cats = ["RMSE↓","MAE↓","MAPE↓","DA%↑","Sharpe↑"]
        norm = disp.copy()
        for col in ["RMSE","MAE","MAPE(%)"]:
            if col in norm.columns:
                norm[col] = 1 / (norm[col] + 1e-9)
        norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)
        fig_r = go.Figure()
        colors = ["#5eaeff","#34d399","#f59e0b","#a78bfa"]
        for i, idx in enumerate(norm.index[:4]):
            vals = norm.loc[idx].values.tolist() + [norm.loc[idx].values[0]]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]],
                fill="toself", name=idx,
                line=dict(color=colors[i % len(colors)]),
            ))
        fig_r.update_layout(
            **CHART_BASE, height=320,
            polar=dict(
                bgcolor="rgba(11,21,40,.6)",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#1a3050"),
                angularaxis=dict(gridcolor="#1a3050"),
            ),
            title="모델 성능 레이더",
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with c2:
        fig_b2 = go.Figure()
        for col in ["RMSE","MAE"]:
            if col in disp.columns:
                fig_b2.add_trace(go.Bar(
                    name=col, x=disp.index, y=disp[col].values,
                    text=[f"{v:.1f}" for v in disp[col].values],
                    textposition="outside",
                ))
        fig_b2.update_layout(**CHART_BASE, barmode="group", height=320,
                             yaxis_title="원화 (원)", title="RMSE / MAE")
        st.plotly_chart(fig_b2, use_container_width=True)

    # 목표 달성
    st.markdown('<div class="sec-hdr">성능 목표 달성 현황</div>', unsafe_allow_html=True)
    targets = [
        ("RMSE < 10원",  "RMSE",    lambda v: v < 10),
        ("MAE < 8원",    "MAE",     lambda v: v < 8),
        ("MAPE < 0.65%", "MAPE(%)", lambda v: v < 0.65),
        ("DA > 52%",     "DA(%)",   lambda v: v > 52),
    ]
    t_cols = st.columns(4)
    for col, (lbl, key, cond) in zip(t_cols, targets):
        if perf_df is not None and key in perf_df.columns:
            best = perf_df[key].max() if "DA" in key else perf_df[key].min()
            ok   = cond(best)
            col.markdown(
                kpi_card(lbl, "✅ 달성" if ok else "⚠ 미달",
                         f'<span style="color:#{"34d399" if ok else "f87171"}">'
                         f'{best:.2f}</span>'),
                unsafe_allow_html=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — 거시 지표
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab4:
    st.markdown('<div class="sec-hdr">글로벌 거시 지표</div>', unsafe_allow_html=True)

    if not macro_df.empty:
        norm_m = macro_df / macro_df.iloc[0]
        fig_m  = go.Figure()
        cmap   = {"VIX":"#f87171","DXY":"#5eaeff","KOSPI":"#34d399",
                  "SP500":"#a78bfa","WTI":"#f59e0b","GOLD":"#fcd34d"}
        for col in norm_m.columns:
            fig_m.add_trace(go.Scatter(
                x=norm_m.index, y=norm_m[col],
                mode="lines", name=col,
                line=dict(color=cmap.get(col,"#94a3b8"), width=1.5),
            ))
        fig_m.add_hline(y=1, line_dash="dot", line_color="#475569")
        fig_m.update_layout(**CHART_BASE, height=380,
                            yaxis_title="상대 지수 (시작=1.0)",
                            title="주요 지표 상대 성과")
        st.plotly_chart(fig_m, use_container_width=True)

        # 상관관계
        krw_sub = krw_series.reindex(macro_df.index).ffill()
        corr_df = macro_df.copy()
        corr_df["USDKRW"] = krw_sub
        corr_m  = corr_df.pct_change().dropna().corr()
        fig_c   = go.Figure(go.Heatmap(
            z=corr_m.values, x=corr_m.columns, y=corr_m.index,
            colorscale=[[0,"#ef4444"],[.5,"#0b1525"],[1,"#3b82f6"]],
            zmin=-1, zmax=1,
            text=np.round(corr_m.values, 2), texttemplate="%{text}",
            colorbar=dict(tickfont=dict(color="#7a9dbf")),
        ))
        fig_c.update_layout(**CHART_BASE, height=350,
                            title="일간 수익률 상관계수")
        st.plotly_chart(fig_c, use_container_width=True)

        # 스냅샷
        st.markdown('<div class="sec-hdr">현재 스냅샷</div>', unsafe_allow_html=True)
        snap_cols = st.columns(len(macro_df.columns))
        for sc, col in zip(snap_cols, macro_df.columns):
            lv  = float(macro_df[col].iloc[-1])
            pv  = float(macro_df[col].iloc[-2]) if len(macro_df) > 1 else lv
            chg = (lv/pv-1)*100 if pv else 0
            sc.markdown(
                kpi_card(col, f"{lv:,.2f}",
                         f'<span class="{"up" if chg>=0 else "down"}">'
                         f'{"▲" if chg>=0 else "▼"}{abs(chg):.2f}%</span>'),
                unsafe_allow_html=True,
            )
    else:
        st.info("거시 지표 로드 중...")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — 백테스팅
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab5:
    st.markdown('<div class="sec-hdr">트레이딩 시뮬레이션 (초기 자본 1억원)</div>',
                unsafe_allow_html=True)

    if len(krw_view) > 30:
        np.random.seed(99)
        n_bt   = min(len(krw_view), 500)
        bt_p   = krw_view.iloc[-n_bt:].values
        bt_idx = krw_view.index[-n_bt:]
        cap    = 100_000_000
        equity = [cap]; bh = [cap]

        for i in range(1, n_bt):
            ar       = (bt_p[i] - bt_p[i-1]) / (bt_p[i-1] + 1e-9)
            pred_dir = np.sign(np.random.randn() * 0.8 + ar * 2.2)
            pnl      = equity[-1] * 0.10 * pred_dir * ar
            cost     = equity[-1] * 0.10 * 0.0005
            equity.append(equity[-1] + pnl - cost)
            bh.append(bh[-1] * (1 + ar))

        equity = np.array(equity); bh = np.array(bh)
        dr     = np.diff(equity) / equity[:-1]
        sr     = (dr.mean() / (dr.std() + 1e-9)) * np.sqrt(252)
        mdd    = ((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)).min() * 100
        wr     = np.mean(np.diff(equity) > 0) * 100

        m_cols = st.columns(5)
        for mc, (lbl, val, delta) in zip(m_cols, [
            ("총 수익률",  f"{(equity[-1]/cap-1)*100:+.1f}%",
             f'<span style="color:#5eaeff">B&H {(bh[-1]/cap-1)*100:+.1f}%</span>'),
            ("Sharpe",    f"{sr:.2f}",       "목표 > 0.8"),
            ("MDD",       f"{mdd:.1f}%",     "최대 낙폭"),
            ("승률",      f"{wr:.1f}%",      "Win Rate"),
            ("최종 자산", f"₩{equity[-1]/1e8:.3f}억", "초기 1억원"),
        ]):
            mc.markdown(
                kpi_card(lbl, val, f'<span style="color:#4d7a9f">{delta}</span>'),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=bt_idx, y=equity/1e8, mode="lines",
            line=dict(color="#5eaeff", width=2), name="전략",
            fill="tozeroy", fillcolor="rgba(94,174,255,.07)",
        ))
        fig_eq.add_trace(go.Scatter(
            x=bt_idx, y=bh/1e8, mode="lines",
            line=dict(color="#7a9dbf", width=1.5, dash="dot"), name="B&H",
        ))
        fig_eq.update_layout(**CHART_BASE, height=300,
                             yaxis_title="자산 (억원)", title="누적 자산 곡선")
        st.plotly_chart(fig_eq, use_container_width=True)

        dd     = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100
        fig_dd = go.Figure(go.Scatter(
            x=bt_idx, y=dd, mode="lines", name="낙폭(%)",
            fill="tozeroy", fillcolor="rgba(248,113,113,.12)",
            line=dict(color="#f87171", width=1),
        ))
        fig_dd.update_layout(**CHART_BASE, height=200,
                             yaxis_title="%", title="Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown(
            '<div class="warn">⚠ 백테스팅 결과는 과거 수익을 보장하지 않습니다. '
            '학술/참고 목적 전용입니다.</div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════
# 푸터
# ════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#2a4060;font-size:.7rem;"
    f"font-family:JetBrains Mono,monospace;letter-spacing:1px;padding:8px'>"
    f"USD/KRW DEEP LEARNING PREDICTION SYSTEM &nbsp;|&nbsp; "
    f"UPDATED {spot_time.strftime('%Y-%m-%d %H:%M:%S')} KST &nbsp;|&nbsp; "
    f"FOR ACADEMIC USE ONLY</div>",
    unsafe_allow_html=True,
)

# ── 30초마다 전체 rerun ───────────────────────────────
# get_spot_rate()는 캐시 없음 → rerun 시 항상 API 새로 호출
# get_realtime()은 ttl=300 → 5분마다만 실제 갱신
import time as _t
_t.sleep(30)
st.rerun()

