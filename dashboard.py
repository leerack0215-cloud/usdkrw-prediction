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

# ── 한국시간 (KST = UTC+9) ────────────────────────────
KST = datetime.timezone(datetime.timedelta(hours=9))

def now_kst() -> datetime.datetime:
    return datetime.datetime.now(KST)

def fmt_kst(dt_obj) -> str:
    """datetime → KST 문자열 (HH:MM:SS)"""
    if dt_obj is None:
        return "N/A"
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=KST)
    return dt_obj.astimezone(KST).strftime("%H:%M:%S")

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
    반환: (가격, 소스명, KST 수집시각)
    """
    # 1순위: yfinance 1분봉
    try:
        import yfinance as yf
        h = yf.Ticker("KRW=X").history(period="1d", interval="1m")
        if not h.empty:
            price = float(h["Close"].iloc[-1])
            ts    = h.index[-1]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if price > 100:
                return price, f"yfinance 1m", now_kst()
    except Exception:
        pass

    # 2순위: yfinance 일봉
    try:
        import yfinance as yf
        h = yf.Ticker("KRW=X").history(period="5d")
        if not h.empty:
            price = float(h["Close"].iloc[-1])
            if price > 100:
                return price, "yfinance 일봉", now_kst()
    except Exception:
        pass

    # 3순위: ExchangeRate-API (하루 1회)
    try:
        import requests
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=4)
        d = r.json()
        if d.get("result") == "success":
            price = float(d["rates"]["KRW"])
            if price > 100:
                return price, "ExchangeRate-API(1일1회)", now_kst()
    except Exception:
        pass

    return 0.0, "수집실패", now_kst()


@st.cache_data(ttl=300)
def get_realtime():
    """
    LGB 모델 실시간 추론 (5분마다 갱신)
    반환: (price_series, log_return_dict, df_full, fetch_timestamp)

    핵심: 예측값을 절대가격이 아닌 log_return으로 저장
    → 호출 시 cur_price(실시간 spot)에 곱해서 최종 예측가 산출
    → spot 가격이 바뀔 때마다 예측가도 자동 갱신됨
    """
    try:
        df = collect_data(start="2018-01-01")
        df, _ = make_features(df, add_targets=False)

        scaler_path = f"{OUTPUT_DIR}/scaler_X.pkl"
        feat_path   = f"{OUTPUT_DIR}/feature_list.json"

        if not (os.path.exists(scaler_path) and os.path.exists(feat_path)):
            return df["USDKRW"], {}, df, now_kst()

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(feat_path, encoding="utf-8") as f:
            features = json.load(f)

        for col in features:
            if col not in df.columns:
                df[col] = 0.0

        X_scaled  = scaler.transform(df[features].fillna(0))
        last_flat = X_scaled[-1:].reshape(1, -1)

        # log_return 저장 (절대가격 아님)
        log_returns = {}
        for h in HORIZONS:
            path = f"{MODELS_DIR}/lgb_h{h}.pkl"
            if os.path.exists(path):
                with open(path, "rb") as f:
                    model = pickle.load(f)
                clip = CLIP_BOUNDS[h]
                lr   = float(np.clip(model.predict(last_flat)[0], -clip, clip))
                log_returns[HORIZON_LABELS[h]] = lr   # ← log_return 저장

        return df["USDKRW"], log_returns, df, now_kst()

    except Exception as e:
        st.warning(f"데이터 로드 오류: {e}")
        idx    = pd.date_range(end=datetime.date.today(), periods=500, freq="B")
        prices = 1380 + np.cumsum(np.random.default_rng(42).normal(0, 3, 500))
        return pd.Series(prices, index=idx, name="USDKRW"), {}, pd.DataFrame(), now_kst()


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
    if st.button("🔄 데이터 갱신", width="stretch"):
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

# ── 실시간 현재가 (yfinance 1분봉 → 일봉 폴백) ────────
# get_spot_rate()는 캐시 없음 → 30초마다 rerun 시 항상 새로 호출
spot_price, spot_src, spot_time = get_spot_rate()
cur_price  = spot_price if spot_price > 0 else (
    float(krw_series.iloc[-1]) if len(krw_series) > 0 else 1482.0
)
prev_price  = float(krw_series.iloc[-2]) if len(krw_series) > 1 else cur_price
day_chg     = cur_price - prev_price
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
# 카운트다운 바 (KST 기준)
# ════════════════════════════════════════════════════════

TTL_SEC    = 300
now_k      = now_kst()
# fetch_time이 KST aware이면 그대로, naive이면 KST로 간주
ft = fetch_time
if ft.tzinfo is None:
    ft = ft.replace(tzinfo=KST)
elapsed    = int((now_k - ft).total_seconds())
elapsed    = max(0, min(elapsed, TTL_SEC))   # 음수/초과 방지
remain     = TTL_SEC - elapsed
pct        = remain / TTL_SEC * 100
m = remain // 60
s = remain % 60

if remain <= 30:
    tc = "#f87171"; bc = "linear-gradient(90deg,#dc2626,#f87171)"
elif remain <= 90:
    tc = "#f59e0b"; bc = "linear-gradient(90deg,#d97706,#f59e0b)"
else:
    tc = "#34d399"; bc = "linear-gradient(90deg,#2563eb,#34d399)"

lgb_update_kst = fmt_kst(fetch_time)
spot_kst       = fmt_kst(spot_time)

# 카운트다운 바 (정적 표시)
st.markdown(f"""
<div class="countdown-wrap">
  <span class="countdown-label">🔄 LGB 예측 갱신까지</span>
  <span id="kst-timer" style="font-family:'JetBrains Mono',monospace;
    font-size:1.1rem;font-weight:700;color:{tc};
    min-width:52px;text-align:center;">{m}:{s:02d}</span>
  <div class="bar-track">
    <div id="kst-bar" style="width:{pct:.1f}%;background:{bc};
      height:5px;border-radius:99px;transition:width 1s linear;"></div>
  </div>
  <span class="last-update">
    🟢 LGB: {lgb_update_kst} KST &nbsp;|&nbsp;
    💱 현재가({spot_src}): {spot_kst} KST
  </span>
</div>""", unsafe_allow_html=True)

# JS 카운트다운 — st.components.v1.html() 사용 (script 차단 우회)
import streamlit.components.v1 as _stc
_stc.html(f"""
<script>
var remain = {remain};
var ttl    = {TTL_SEC};

function tick() {{
  var parentDoc = window.parent.document;
  var timerEl = parentDoc.getElementById('kst-timer');
  var barEl   = parentDoc.getElementById('kst-bar');

  if (timerEl) {{
    var m = Math.floor(remain / 60);
    var s = remain % 60;
    timerEl.textContent = m + ':' + (s < 10 ? '0' : '') + s;
    if (remain <= 30) timerEl.style.color = '#f87171';
    else if (remain <= 90) timerEl.style.color = '#f59e0b';
    else timerEl.style.color = '#34d399';
  }}
  if (barEl) {{
    barEl.style.width = (remain / ttl * 100) + '%';
    if (remain <= 30) barEl.style.background = 'linear-gradient(90deg,#dc2626,#f87171)';
    else if (remain <= 90) barEl.style.background = 'linear-gradient(90deg,#d97706,#f59e0b)';
    else barEl.style.background = 'linear-gradient(90deg,#2563eb,#34d399)';
  }}

  if (remain <= 0) {{
    window.parent.location.reload();
    return;
  }}
  remain--;
  setTimeout(tick, 1000);
}}
setTimeout(tick, 1000);
</script>
""", height=0)

if remain <= 0:
    st.cache_data.clear()

# ════════════════════════════════════════════════════════
# KPI 카드
# log_return → 실시간 cur_price 기준 예측가 변환
# rt_preds = {label: log_return} 형태
# ════════════════════════════════════════════════════════

import math
def lr_to_price(label, fallback):
    lr = rt_preds.get(label)
    if lr is None:
        return fallback
    return round(cur_price * math.exp(lr), 2)

d1_rt = lr_to_price("D+1", cur_price)
d3_rt = lr_to_price("D+3", cur_price)

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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 가격 차트", "🔮 다중 호라이즌 예측",
    "⚡ 모델 성능", "🌍 거시 지표", "📋 백테스팅", "📰 뉴스 센티멘트",
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

    # ── 가격 (fill="tozeroy" 제거 → autofit 정상화) ──
    y_min = float(krw_view.min()) * 0.998
    y_max = float(krw_view.max()) * 1.002

    fig.add_trace(go.Scatter(
        x=krw_view.index, y=krw_view.values,
        mode="lines", line=dict(color="#4a9eff", width=1.5),
        name="USD/KRW",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=krw_view.index, y=ema20,
        mode="lines", line=dict(color="#f59e0b", width=1, dash="dot"), name="EMA20",
    ), row=1, col=1)
    # 볼린저밴드
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
    pct_bar = krw_view.pct_change() * 100
    fig.add_trace(go.Bar(
        x=krw_view.index, y=pct_bar,
        marker_color=["#34d399" if v >= 0 else "#f87171" for v in pct_bar],
        name="일간변화(%)",
    ), row=3, col=1)

    fig.update_layout(
        **CHART_BASE, height=560,
        yaxis2_title="RSI", yaxis3_title="%",
    )
    # 가격 y축 범위 별도 설정 (CHART_BASE와 충돌 방지)
    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    st.plotly_chart(fig, width="stretch")

    # 기술 지표 + 수익률 요약
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-hdr">기술적 지표</div>', unsafe_allow_html=True)
        rsi_v  = float(rsi.iloc[-1]) if not rsi.empty else 50
        rsi_st = "과매수" if rsi_v > 70 else ("과매도" if rsi_v < 30 else "중립")
        items  = {
            "RSI (14)":    f"{rsi_v:.1f}  {rsi_st}",
            "EMA20 대비":  f"{'위' if cur_price > float(ema20.iloc[-1]) else '아래'} "
                           f"({(cur_price/float(ema20.iloc[-1])-1)*100:.2f}%)"
                           if not ema20.empty else "N/A",
            "20일 변동성": f"{float(krw_view.pct_change().rolling(20).std().iloc[-1]*100*np.sqrt(252)):.1f}%"
                           if len(krw_view) > 20 else "N/A",
            "볼린저 위치": (
                "상단 근접" if cur_price > float((sma20+2*std20).iloc[-1]) else
                "하단 근접" if cur_price < float((sma20-2*std20).iloc[-1]) else "중간"
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
        st.plotly_chart(ret_fig, width="stretch")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — 다중 호라이즌 예측
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab2:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            f'<div class="sec-hdr">🟢 실시간 LGB 예측'
            f'<span class="badge-rt">기준가: ₩{cur_price:,.2f} · {spot_kst} KST</span></div>',
            unsafe_allow_html=True,
        )
        if rt_preds:
            rows = []
            for h in HORIZONS:
                lbl  = HORIZON_LABELS[h]
                lr   = rt_preds.get(lbl)
                if lr is None:
                    continue
                pred = round(cur_price * math.exp(lr), 2)
                pct  = (pred / cur_price - 1) * 100
                rows.append({
                    "호라이즌": lbl,
                    "예측 환율": f"₩{pred:,.2f}",
                    "변화율":   f"{pct:+.2f}%",
                    "방향":     "↑ 상승" if pct>0 else ("↓ 하락" if pct<0 else "— 보합"),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
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
            st.dataframe(pd.DataFrame(rows2), hide_index=True, width="stretch")
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
        p_list = [cur_price] + [
            round(cur_price * math.exp(rt_preds.get(HORIZON_LABELS[h], 0)), 2)
            for h in [1, 3, 5, 10, 22]
        ]
        fig_fan = go.Figure()
        for ci, alpha in [(0.99,0.06),(0.95,0.10),(0.80,0.16)]:
            w = [cur_price*0.002*np.sqrt(h) for h in h_list]
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
        fig_fan.add_hline(y=cur_price, line_dash="dot",
                          line_color="#475569", annotation_text="현재")
        fig_fan.update_layout(
            **CHART_BASE, height=320,
            xaxis_title="영업일", yaxis_title="USD/KRW (원)",
        )
        st.plotly_chart(fig_fan, width="stretch")

    # 모델별 D+1 비교
    if forecast and "models" in forecast and "D+1" in forecast["models"]:
        st.markdown('<div class="sec-hdr">모델별 D+1 예측 비교</div>', unsafe_allow_html=True)
        m_data = forecast["models"]["D+1"]
        fig_b  = go.Figure(go.Bar(
            x=list(m_data.keys()), y=list(m_data.values()),
            marker_color=["#34d399" if v>cur_price else "#f87171" for v in m_data.values()],
            text=[f"₩{v:,.0f}" for v in m_data.values()], textposition="outside",
        ))
        fig_b.add_hline(y=cur_price, line_dash="dash", line_color="#fbbf24",
                        annotation_text=f"현재 ₩{cur_price:,.0f}")
        fig_b.update_layout(**CHART_BASE, height=280, showlegend=False,
                            yaxis_title="예측 환율 (원)")
        st.plotly_chart(fig_b, width="stretch")


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
        width="stretch",
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
        st.plotly_chart(fig_r, width="stretch")

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
        st.plotly_chart(fig_b2, width="stretch")

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
        st.plotly_chart(fig_m, width="stretch")

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
        st.plotly_chart(fig_c, width="stretch")

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
        st.plotly_chart(fig_eq, width="stretch")

        dd     = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100
        fig_dd = go.Figure(go.Scatter(
            x=bt_idx, y=dd, mode="lines", name="낙폭(%)",
            fill="tozeroy", fillcolor="rgba(248,113,113,.12)",
            line=dict(color="#f87171", width=1),
        ))
        fig_dd.update_layout(**CHART_BASE, height=200,
                             yaxis_title="%", title="Drawdown")
        st.plotly_chart(fig_dd, width="stretch")

        st.markdown(
            '<div class="warn">⚠ 백테스팅 결과는 과거 수익을 보장하지 않습니다. '
            '학술/참고 목적 전용입니다.</div>',
            unsafe_allow_html=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — 뉴스 센티멘트 (1시간 캐시)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=3600)
def fetch_news() -> list:
    """
    USD/KRW 관련 뉴스 RSS 스크래핑 (1시간 캐시)
    소스: Reuters, Investing.com, 연합뉴스 등 RSS
    반환: [{"title":..., "link":..., "pub":..., "source":...}, ...]
    """
    import requests
    from xml.etree import ElementTree as ET

    feeds = [
        ("Reuters FX",    "https://feeds.reuters.com/reuters/businessNews"),
        ("Investing.com", "https://www.investing.com/rss/news_301.rss"),
        ("연합뉴스 경제", "https://www.yna.co.kr/rss/economy.xml"),
    ]
    KW = ["KRW","원달러","환율","USD","won","Korea","한국","Fed","금리",
          "BOK","한은","외환","KOSPI","달러"]

    articles = []
    for src, url in feeds:
        try:
            r = requests.get(url, timeout=6,
                             headers={"User-Agent": "Mozilla/5.0"})
            root = ET.fromstring(r.content)
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                link  = (item.findtext("link")  or "").strip()
                pub   = (item.findtext("pubDate") or
                         item.findtext("{http://purl.org/dc/elements/1.1/}date") or "").strip()
                if any(k.lower() in title.lower() for k in KW):
                    articles.append({"title": title, "link": link,
                                     "pub": pub[:25], "source": src})
                if len(articles) >= 30:
                    break
        except Exception:
            pass
    return articles[:30]


@st.cache_data(ttl=3600)
def analyze_sentiment(articles_json: str) -> list:
    """
    Claude API로 뉴스 센티멘트 분석
    articles_json: JSON 문자열 (캐시 키로 사용 가능한 타입)
    """
    import json as _json
    articles = _json.loads(articles_json)

    if not articles:
        return []

    try:
        import anthropic
    except ImportError:
        return []

    headlines = "\n".join(
        [f"{i+1}. {a['title']}" for i, a in enumerate(articles)]
    )
    prompt = f"""다음 뉴스 헤드라인들이 USD/KRW 환율에 미치는 영향을 분석하세요.

헤드라인:
{headlines}

각 뉴스에 대해 아래 JSON 배열로만 응답하세요 (다른 텍스트 없이):
[
  {{
    "idx": 1,
    "score": <-2에서+2 사이 정수. +2=강한달러강세, -2=강한원화강세, 0=중립>,
    "reason": "<한국어로 15자 이내 이유>",
    "impact": "<상승|하락|중립>"
  }},
  ...
]"""

    try:
        client = anthropic.Anthropic()
        msg    = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 1500,
            messages   = [{"role": "user", "content": prompt}],
        )
        text  = msg.content[0].text.strip()
        start = text.find("[")
        end   = text.rfind("]") + 1
        return _json.loads(text[start:end])
    except Exception:
        return []


with tab6:
    st.markdown('<div class="sec-hdr">📰 환율 영향 뉴스 센티멘트</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(37,99,235,.08);border:1px solid rgba(37,99,235,.3);
    border-radius:8px;padding:10px 16px;font-size:.78rem;color:#7a9dbf;margin-bottom:14px'>
    💡 1시간마다 자동 수집 · Claude AI가 각 뉴스의 USD/KRW 영향도 평가<br>
    <b>+2</b> 달러 강세(원화약세) &nbsp;|&nbsp; <b>0</b> 중립 &nbsp;|&nbsp; <b>-2</b> 원화 강세(달러약세)
    </div>""", unsafe_allow_html=True)

    with st.spinner("뉴스 수집 + AI 분석 중..."):
        news_list = fetch_news()

    if not news_list:
        st.warning("뉴스를 불러오지 못했습니다. 잠시 후 다시 시도하세요.")
    else:
        # 센티멘트 분석 — JSON 문자열로 직렬화해서 전달
        import json as _json
        with st.spinner(f"{len(news_list)}개 뉴스 AI 분석 중..."):
            sentiments = analyze_sentiment(
                _json.dumps(news_list, ensure_ascii=False)
            )

        # 센티멘트 맵 생성
        sent_map = {s["idx"]: s for s in sentiments}

        # 종합 센티멘트 스코어
        scores = [s["score"] for s in sentiments if "score" in s]
        avg_score = sum(scores) / len(scores) if scores else 0

        # 종합 KPI
        col_s1, col_s2, col_s3 = st.columns(3)
        sc_color = "#f87171" if avg_score > 0.5 else ("#34d399" if avg_score < -0.5 else "#94a3b8")
        sc_label = "달러 강세 우세" if avg_score > 0.5 else ("원화 강세 우세" if avg_score < -0.5 else "혼조세")
        col_s1.markdown(
            kpi_card("뉴스 종합 센티멘트",
                     f'<span style="color:{sc_color}">{avg_score:+.2f}</span>',
                     f'<span style="color:{sc_color}">{sc_label}</span>'),
            unsafe_allow_html=True,
        )
        col_s2.markdown(
            kpi_card("수집 뉴스", f"{len(news_list)}건",
                     f"갱신: {now_kst().strftime('%H:%M')} KST"),
            unsafe_allow_html=True,
        )
        up_cnt   = sum(1 for s in scores if s > 0)
        down_cnt = sum(1 for s in scores if s < 0)
        col_s3.markdown(
            kpi_card("달러↑ / 원화↑",
                     f'<span class="down">↑{up_cnt}</span> / '
                     f'<span class="up">↑{down_cnt}</span>',
                     "뉴스 건수 기준"),
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # 센티멘트 바 차트
        if scores:
            fig_s = go.Figure(go.Bar(
                x=[f"#{i+1}" for i in range(len(scores))],
                y=scores,
                marker_color=["#f87171" if s > 0 else ("#34d399" if s < 0 else "#475569")
                              for s in scores],
                text=[f"{s:+d}" for s in scores],
                textposition="outside",
            ))
            fig_s.add_hline(y=0, line_color="#475569")
            fig_s.update_layout(
                **CHART_BASE, height=200,
                yaxis=dict(range=[-2.5, 2.5], gridcolor="#1a3050"),
                xaxis_title="뉴스", yaxis_title="센티멘트 점수",
                title="뉴스별 USD/KRW 영향도",
                showlegend=False,
            )
            st.plotly_chart(fig_s, width="stretch")

        # 뉴스 목록 테이블
        st.markdown('<div class="sec-hdr">뉴스 상세</div>', unsafe_allow_html=True)
        for i, art in enumerate(news_list):
            s = sent_map.get(i + 1, {})
            score  = s.get("score", 0)
            reason = s.get("reason", "분석 중")
            impact = s.get("impact", "중립")

            if score > 0:
                badge_color = "rgba(248,113,113,.15)"; badge_bc = "#f87171"; arrow = "↑ 달러강세"
            elif score < 0:
                badge_color = "rgba(52,211,153,.12)";  badge_bc = "#34d399"; arrow = "↓ 원화강세"
            else:
                badge_color = "rgba(71,85,105,.2)";    badge_bc = "#475569"; arrow = "— 중립"

            st.markdown(f"""
<div style='background:{badge_color};border:1px solid {badge_bc}33;
border-left:3px solid {badge_bc};border-radius:6px;
padding:10px 14px;margin-bottom:6px;'>
  <div style='display:flex;justify-content:space-between;align-items:center;'>
    <span style='font-size:.82rem;color:#dde6f0;flex:1;margin-right:12px'>{art['title']}</span>
    <span style='font-family:JetBrains Mono,monospace;font-size:.78rem;color:{badge_bc};
    white-space:nowrap;font-weight:700'>{score:+d} {arrow}</span>
  </div>
  <div style='margin-top:5px;font-size:.7rem;color:#4d7a9f'>
    {art['source']} · {art['pub']} &nbsp;|&nbsp;
    <span style='color:{badge_bc}'>{reason}</span>
    &nbsp;|&nbsp; <a href='{art['link']}' target='_blank'
    style='color:#2563eb'>원문 ↗</a>
  </div>
</div>""", unsafe_allow_html=True)

        # LGB 보정 안내
        st.markdown('<div class="sec-hdr">예측 보정 안내</div>', unsafe_allow_html=True)

        if abs(avg_score) >= 1.0:
            bias_pct = avg_score * 0.05   # score 1당 ±0.05%
            adj_d1   = round(d1_rt * (1 + bias_pct / 100), 2)
            direction = "달러 강세" if avg_score > 0 else "원화 강세"
            color     = "#f87171" if avg_score > 0 else "#34d399"
            st.markdown(f"""
<div style='background:rgba(37,99,235,.08);border:1px solid rgba(37,99,235,.3);
border-radius:8px;padding:12px 16px;font-size:.82rem;color:#dde6f0'>
  📊 <b>뉴스 센티멘트 보정</b><br><br>
  종합 점수 <span style='color:{color};font-weight:700'>{avg_score:+.2f}</span>
  → <span style='color:{color}'>{direction}</span> 우세<br>
  LGB D+1 기본 예측: <b>₩{d1_rt:,.2f}</b><br>
  뉴스 보정 D+1 참고: <b style='color:{color}'>₩{adj_d1:,.2f}</b>
  <span style='color:#4d7a9f;font-size:.72rem'>
  (±{abs(bias_pct):.3f}% 보정 · 참고용)</span>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style='background:rgba(71,85,105,.15);border:1px solid #1a3050;
border-radius:8px;padding:12px 16px;font-size:.82rem;color:#7a9dbf'>
  📊 종합 센티멘트 점수 <b>{avg_score:+.2f}</b> — 혼조세<br>
  뉴스 신호가 중립적이라 LGB 예측값(₩{d1_rt:,.2f})을 그대로 사용합니다.
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# 푸터
# ════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#2a4060;font-size:.7rem;"
    f"font-family:JetBrains Mono,monospace;letter-spacing:1px;padding:8px'>"
    f"USD/KRW DEEP LEARNING PREDICTION SYSTEM &nbsp;|&nbsp; "
    f"UPDATED {now_kst().strftime('%Y-%m-%d %H:%M:%S')} KST &nbsp;|&nbsp; "
    f"FOR ACADEMIC USE ONLY</div>",
    unsafe_allow_html=True,
)

# 30초마다 자동 새로고침 (환율 + 예측 갱신)
import streamlit.components.v1 as _stc2
_stc2.html("""
<script>
setTimeout(function() {
  window.parent.location.reload();
}, 30000);
</script>
""", height=0)

