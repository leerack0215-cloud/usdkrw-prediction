"""
utils.py — USD/KRW 예측 시스템 v3
멀티스케일 데이터 + 강화 피처 + ARIMAX 지원

설계 원칙:
  1. Lookahead Bias 완전 차단
  2. 멀티스케일: 일봉 + 1h + 5m + 1m 통합
  3. 강화 피처: 금리차 / 크로스통화 / 한국특화 / 레짐감지
  4. 모든 파일의 단일 상수 소스
"""

import os, warnings, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ════════════════════════════════════════════════════════
# 공용 상수
# ════════════════════════════════════════════════════════

KST = datetime.timezone(datetime.timedelta(hours=9))

def now_kst() -> datetime.datetime:
    return datetime.datetime.now(KST)

# 티커 정의
TICKERS_DAILY = {
    "USDKRW": "KRW=X",
    "DXY":    "DX-Y.NYB",
    "VIX":    "^VIX",
    "US10Y":  "^TNX",
    "US2Y":   "^IRX",
    "WTI":    "CL=F",
    "GOLD":   "GC=F",
    "KOSPI":  "^KS11",
    "SP500":  "^GSPC",
    "USDJPY": "JPY=X",
    "USDCNY": "CNY=X",   # 위안화 (한국 무역 1위국)
    "USDTWD": "TWD=X",   # 대만달러 (반도체 경쟁국)
    "SOX":    "^SOX",    # 필라델피아 반도체 지수
    "EEM":    "EEM",     # 신흥국 ETF
}

HORIZONS       = [1, 3, 5, 10, 22]
HORIZON_LABELS = {1:"D+1", 3:"D+3", 5:"D+5", 10:"D+10", 22:"D+22"}
SEQ_LEN        = 30
OUTPUT_DIR     = "outputs"
MODELS_DIR     = os.path.join(OUTPUT_DIR, "models")
DATA_DIR       = os.path.join(OUTPUT_DIR, "data")   # 분봉 누적 저장

CLIP_BOUNDS = {1:0.025, 3:0.040, 5:0.055, 10:0.080, 22:0.120}

for d in [OUTPUT_DIR, MODELS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)


# ════════════════════════════════════════════════════════
# 데이터 수집 — 단일 인터벌
# ════════════════════════════════════════════════════════

def _download(ticker: str, start: str, end: str,
              interval: str = "1d") -> pd.Series:
    """단일 티커 다운로드 → Close 반환. 실패 시 빈 Series"""
    try:
        raw = yf.download(
            ticker, start=start, end=end,
            interval=interval,
            auto_adjust=True, progress=False,
        )
        if raw is None or raw.empty:
            return pd.Series(dtype=float)
        c = raw["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c = c.dropna()
        c.index = pd.to_datetime(c.index).tz_localize(None)
        return c
    except Exception:
        return pd.Series(dtype=float)


# ════════════════════════════════════════════════════════
# 멀티스케일 데이터 수집
# ════════════════════════════════════════════════════════

def collect_multiscale(
    start_daily: str = "2015-01-01",
) -> pd.DataFrame:
    """
    멀티스케일 USD/KRW 수집 후 일봉으로 통합

    레이어:
      일봉:   start_daily ~ 현재  (전체 기간)
      1시간봉: 최근 730일          (yfinance 제공 한계)
      5분봉:   최근 60일
      1분봉:   최근 7일

    반환: 일봉 DataFrame (OHLCV + 멀티스케일 통계 피처)
    """
    today = datetime.date.today()
    end   = today.strftime("%Y-%m-%d")

    print("  [멀티스케일] 일봉 수집...")
    # ── 일봉 전체 ─────────────────────────────────────
    frames_d = {}
    for name, sym in TICKERS_DAILY.items():
        s = _download(sym, start_daily, end, "1d")
        if len(s) > 100:
            frames_d[name] = s.rename(name)   # ← 컬럼명 명시적 지정
            print(f"    ✓ {name}: {len(s)}행")
        else:
            print(f"    ⚠ {name}: 데이터 부족 — 스킵")

    if "USDKRW" not in frames_d:
        raise RuntimeError("USDKRW 일봉 수집 실패")

    df = pd.concat(frames_d.values(), axis=1)
    df.columns = list(frames_d.keys())        # ← 컬럼명 재보장
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().ffill().bfill()

    # ── 1시간봉 통계 → 일봉으로 집계 ─────────────────
    print("  [멀티스케일] 1시간봉 수집...")
    h1_start = (today - datetime.timedelta(days=729)).strftime("%Y-%m-%d")
    krw_1h   = _download("KRW=X", h1_start, end, "1h")

    if len(krw_1h) > 100:
        krw_1h_daily = krw_1h.resample("D").agg(
            h1_open  = ("first"),
            h1_high  = ("max"),
            h1_low   = ("min"),
            h1_close = ("last"),
            h1_std   = ("std"),
            h1_range = (lambda x: x.max() - x.min()),
        )
        krw_1h_daily.columns = ["h1_open","h1_high","h1_low",
                                  "h1_close","h1_std","h1_range"]
        df = df.join(krw_1h_daily, how="left")
        print(f"    ✓ 1시간봉: {len(krw_1h)}행 → 일집계 {len(krw_1h_daily)}행")
    else:
        print("    ⚠ 1시간봉: 데이터 부족 — 스킵")

    # ── 5분봉 통계 → 일봉으로 집계 ───────────────────
    print("  [멀티스케일] 5분봉 수집...")
    m5_start = (today - datetime.timedelta(days=59)).strftime("%Y-%m-%d")
    krw_5m   = _download("KRW=X", m5_start, end, "5m")

    if len(krw_5m) > 100:
        krw_5m_daily = krw_5m.resample("D").agg(
            m5_std   = ("std"),
            m5_range = (lambda x: x.max() - x.min()),
            m5_n     = ("count"),
        )
        krw_5m_daily.columns = ["m5_std","m5_range","m5_n"]
        df = df.join(krw_5m_daily, how="left")
        print(f"    ✓ 5분봉: {len(krw_5m)}행 → 일집계 {len(krw_5m_daily)}행")
    else:
        print("    ⚠ 5분봉: 데이터 부족 — 스킵")

    # ── 1분봉 통계 → 일봉으로 집계 ───────────────────
    print("  [멀티스케일] 1분봉 수집...")
    m1_start = (today - datetime.timedelta(days=6)).strftime("%Y-%m-%d")
    krw_1m   = _download("KRW=X", m1_start, end, "1m")

    if len(krw_1m) > 100:
        krw_1m_daily = krw_1m.resample("D").agg(
            m1_std      = ("std"),
            m1_range    = (lambda x: x.max() - x.min()),
            m1_n        = ("count"),
            m1_open_ret = (lambda x: (x.iloc[-1]/x.iloc[0]-1)*100
                           if len(x)>1 else 0),
        )
        krw_1m_daily.columns = ["m1_std","m1_range","m1_n","m1_open_ret"]
        df = df.join(krw_1m_daily, how="left")
        print(f"    ✓ 1분봉: {len(krw_1m)}행 → 일집계 {len(krw_1m_daily)}행")
    else:
        print("    ⚠ 1분봉: 데이터 부족 — 스킵")

    df = df.ffill().bfill()
    print(f"  멀티스케일 통합 완료: {len(df)}행 × {len(df.columns)}컬럼")
    return df


# ════════════════════════════════════════════════════════
# 레거시 단일스케일 수집 (predict.py 경량 모드용)
# ════════════════════════════════════════════════════════

def collect_data(start: str = "2015-01-01") -> pd.DataFrame:
    """
    일봉 전용 경량 수집 (predict.py / dashboard.py 용)
    collect_multiscale()의 간소화 버전
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    frames = {}
    for name, sym in TICKERS_DAILY.items():
        s = _download(sym, start, today, "1d")
        if len(s) > 100:
            frames[name] = s.rename(name)   # ← 컬럼명 명시적 지정
    if "USDKRW" not in frames:
        raise RuntimeError("USDKRW 수집 실패")
    df = pd.concat(frames.values(), axis=1)
    df.columns = list(frames.keys())        # ← 컬럼명 재보장
    df.index = pd.to_datetime(df.index)
    return df.sort_index().ffill().bfill()


# ════════════════════════════════════════════════════════
# 강화 피처 엔지니어링
# ════════════════════════════════════════════════════════

def make_features(
    df: pd.DataFrame,
    add_targets: bool = True,
) -> tuple:
    """
    강화 피처 엔지니어링 (Lookahead Bias 완전 차단)

    피처 그룹:
      G1. 추세 (EMA/SMA/MACD)
      G2. 모멘텀 (RSI/Stoch/CCI/ADX)
      G3. 변동성 (BB/ATR/HV/레짐)
      G4. 수익률 (다구간 + 래그)
      G5. 금리차 (한미 스프레드)
      G6. 크로스통화 (CNY/JPY/TWD 상대강도)
      G7. 한국특화 (SOX/EEM/WTI_KRW)
      G8. 멀티스케일 통계
      G9. 캘린더 + 지정학
    """
    df = df.copy()
    c  = df["USDKRW"]

    # ── G1. 추세 ──────────────────────────────────────
    for w in [5, 10, 20, 60, 120]:
        df[f"ema_{w}"]    = c.ewm(span=w, adjust=False).mean()
        df[f"sma_{w}"]    = c.rolling(w).mean()
        df[f"vs_ema_{w}"] = (c / (df[f"ema_{w}"] + 1e-9) - 1) * 100
        df[f"vs_sma_{w}"] = (c / (df[f"sma_{w}"] + 1e-9) - 1) * 100

    ema12           = c.ewm(span=12, adjust=False).mean()
    ema26           = c.ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    df["macd_cross"]= (df["macd"] > df["macd_sig"]).astype(int)

    # ── G2. 모멘텀 ────────────────────────────────────
    delta  = c.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]     = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df["rsi_7"]   = 100 - (100 / (1 + delta.clip(lower=0).rolling(7).mean() /
                                   (-delta.clip(upper=0)).rolling(7).mean().clip(lower=1e-9)))
    df["rsi_ob"]  = (df["rsi"] > 70).astype(int)
    df["rsi_os"]  = (df["rsi"] < 30).astype(int)

    low14          = c.rolling(14).min()
    high14         = c.rolling(14).max()
    df["stoch_k"]  = 100*(c-low14)/(high14-low14+1e-9)
    df["stoch_d"]  = df["stoch_k"].rolling(3).mean()
    df["stoch_sig"]= (df["stoch_k"] > df["stoch_d"]).astype(int)
    df["willr"]    = -100*(high14-c)/(high14-low14+1e-9)

    sma20_cci  = c.rolling(20).mean()
    mad20      = c.rolling(20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"]  = (c - sma20_cci) / (0.015 * mad20 + 1e-9)

    # ADX (추세 강도)
    tr   = pd.concat([
        (high14 - low14),
        (high14 - c.shift(1)).abs(),
        (low14  - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["adx"] = tr.rolling(14).mean() / (c + 1e-9) * 100

    # ── G3. 변동성 ────────────────────────────────────
    sma20          = c.rolling(20).mean()
    std20          = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (4 * std20) / (sma20 + 1e-9)
    df["bb_pos"]   = (c - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["bb_break_up"]  = (c > df["bb_upper"]).astype(int)
    df["bb_break_dn"]  = (c < df["bb_lower"]).astype(int)

    df["atr"]    = tr.rolling(14).mean()
    df["hv_5"]   = c.pct_change().rolling(5).std()  * np.sqrt(252) * 100
    df["hv_21"]  = c.pct_change().rolling(21).std() * np.sqrt(252) * 100
    df["hv_60"]  = c.pct_change().rolling(60).std() * np.sqrt(252) * 100

    # 변동성 레짐
    df["hv_ratio"]   = df["hv_5"] / (df["hv_21"] + 1e-9)
    df["vol_regime"] = pd.cut(df["hv_ratio"],
                               bins=[0, 0.7, 1.3, 99],
                               labels=[0, 1, 2]).astype(float)

    # 평균회귀 z-score
    df["zscore_20"] = (c - sma20) / (std20 + 1e-9)
    df["zscore_60"] = (c - c.rolling(60).mean()) / (
        c.rolling(60).std() + 1e-9)

    # ── G4. 수익률 피처 ────────────────────────────────
    for d in [1, 2, 3, 5, 10, 20, 60]:
        df[f"ret_{d}d"] = c.pct_change(d) * 100

    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_ret_{lag}"] = c.pct_change().shift(lag) * 100

    # 가속도 (수익률 변화)
    df["ret_accel"] = df["ret_1d"] - df["lag_ret_1"]

    # 갭 (시가 - 전일 종가 대용)
    df["overnight_gap"] = c.diff()

    # ── G5. 금리차 (한미 스프레드) ────────────────────
    if "US10Y" in df.columns:
        df["us10y"]           = df["US10Y"]
        df["us10y_ret5"]      = df["US10Y"].pct_change(5) * 100
        df["us10y_ma20"]      = df["US10Y"].rolling(20).mean()
        df["us10y_vs_ma"]     = df["US10Y"] - df["us10y_ma20"]

    if "US2Y" in df.columns:
        df["us2y"]            = df["US2Y"]
        df["yield_curve"]     = df["US10Y"] - df["US2Y"]  # 장단기 스프레드
        df["yield_curve_chg"] = df["yield_curve"].diff(5)

    # 한국 기준금리 (고정값 사용 — 한국은행 API 없이)
    # 2024-10-16 기준 3.25% → 2025-02-25 3.00%
    kr_rate_map = {
        "2023-01-01": 3.50,
        "2024-01-01": 3.50,
        "2024-10-16": 3.25,
        "2025-02-25": 3.00,
    }
    kr_rates = pd.Series(kr_rate_map, dtype=float)
    kr_rates.index = pd.to_datetime(kr_rates.index)
    kr_rate_series = kr_rates.reindex(
        df.index.union(kr_rates.index)).ffill().reindex(df.index)
    df["kr_base_rate"] = kr_rate_series.values

    if "US10Y" in df.columns:
        df["rate_spread"]     = df["US10Y"] - df["kr_base_rate"]
        df["rate_spread_ma5"] = df["rate_spread"].rolling(5).mean()
        df["rate_spread_chg"] = df["rate_spread"].diff(5)

    # ── G6. 크로스통화 상대강도 ───────────────────────
    if "USDJPY" in df.columns:
        df["usdjpy_ret5"]     = df["USDJPY"].pct_change(5) * 100
        df["krw_vs_jpy"]      = df["USDKRW"] / (df["USDJPY"] + 1e-9)
        df["krw_vs_jpy_ret5"] = df["krw_vs_jpy"].pct_change(5) * 100

    if "USDCNY" in df.columns:
        df["usdcny_ret5"]     = df["USDCNY"].pct_change(5) * 100
        df["krw_vs_cny"]      = df["USDKRW"] / (df["USDCNY"] + 1e-9)
        df["krw_vs_cny_ret5"] = df["krw_vs_cny"].pct_change(5) * 100
        # 위안화 동조화 지수
        df["cny_corr_20"]     = (
            df["USDKRW"].pct_change()
            .rolling(20)
            .corr(df["USDCNY"].pct_change())
        )

    if "USDTWD" in df.columns:
        df["krw_vs_twd"]      = df["USDKRW"] / (df["USDTWD"] + 1e-9)
        df["krw_vs_twd_ret5"] = df["krw_vs_twd"].pct_change(5) * 100

    # 아시아 통화 평균 대비 KRW 위치
    asia_cols = [col for col in ["USDJPY","USDCNY","USDTWD"]
                 if col in df.columns]
    if len(asia_cols) >= 2:
        asia_norm = pd.DataFrame({
            c: df[c] / df[c].rolling(60).mean() for c in asia_cols
        }).mean(axis=1)
        krw_norm          = df["USDKRW"] / df["USDKRW"].rolling(60).mean()
        df["krw_vs_asia"] = krw_norm - asia_norm

    # ── G7. 한국 특화 피처 ────────────────────────────
    if "DXY" in df.columns:
        df["dxy_ret5"]    = df["DXY"].pct_change(5) * 100
        df["dxy_ret20"]   = df["DXY"].pct_change(20) * 100
        df["dxy_vs_ma20"] = (df["DXY"] / (
            df["DXY"].rolling(20).mean() + 1e-9) - 1) * 100

    if "VIX" in df.columns:
        df["vix"]         = df["VIX"]
        df["vix_ma5"]     = df["VIX"].rolling(5).mean()
        df["vix_ret5"]    = df["VIX"].pct_change(5) * 100
        df["vix_regime"]  = (df["VIX"] > 25).astype(int)
        df["vix_spike"]   = (df["VIX"] > df["VIX"].rolling(20).mean() * 1.5).astype(int)

    if "WTI" in df.columns:
        df["wti_ret5"]    = df["WTI"].pct_change(5) * 100
        # 실질 에너지 수입 부담 (WTI × USDKRW)
        df["wti_krw"]     = df["WTI"] * df["USDKRW"]
        df["wti_krw_ret5"]= df["wti_krw"].pct_change(5) * 100

    if "GOLD" in df.columns:
        df["gold_ret5"]   = df["GOLD"].pct_change(5) * 100
        df["gold_vix_ratio"] = df["GOLD"] / (df["VIX"] + 1e-9) \
            if "VIX" in df.columns else 0

    if "KOSPI" in df.columns:
        df["kospi_ret5"]  = df["KOSPI"].pct_change(5) * 100
        df["kospi_ret20"] = df["KOSPI"].pct_change(20) * 100
        df["kospi_hv"]    = df["KOSPI"].pct_change().rolling(21).std() * np.sqrt(252) * 100

    if "SP500" in df.columns:
        df["sp500_ret5"]  = df["SP500"].pct_change(5) * 100
        df["sp500_ret20"] = df["SP500"].pct_change(20) * 100

    # SOX (반도체 → 한국 수출 선행지표)
    if "SOX" in df.columns:
        df["sox_ret5"]    = df["SOX"].pct_change(5) * 100
        df["sox_ret20"]   = df["SOX"].pct_change(20) * 100
        df["sox_vs_sp"]   = df["SOX"].pct_change(20) - df["SP500"].pct_change(20) \
            if "SP500" in df.columns else 0

    # EEM (신흥국 리스크 선호)
    if "EEM" in df.columns:
        df["eem_ret5"]    = df["EEM"].pct_change(5) * 100
        df["eem_vs_sp"]   = df["EEM"].pct_change(20) - df["SP500"].pct_change(20) \
            if "SP500" in df.columns else 0

    # ── G8. 멀티스케일 통계 ────────────────────────────
    ms_cols = ["h1_std","h1_range","m5_std","m5_range",
               "m1_std","m1_range","m1_n","m1_open_ret"]
    for col in ms_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
            # 정규화 (rolling z-score)
            df[f"{col}_z"] = (
                (df[col] - df[col].rolling(20).mean()) /
                (df[col].rolling(20).std() + 1e-9)
            )

    # 분봉 변동성 vs 일봉 변동성 비율
    if "h1_std" in df.columns:
        df["ms_vol_ratio"] = df["h1_std"] / (df["atr"] + 1e-9)

    # ── G9. 캘린더 + 지정학 ───────────────────────────
    df["day_of_week"]    = df.index.dayofweek
    df["month"]          = df.index.month
    df["month_end"]      = df.index.is_month_end.astype(int)
    df["quarter_end"]    = df.index.is_quarter_end.astype(int)
    df["week_of_year"]   = df.index.isocalendar().week.astype(int)

    # 월말 환율 압력 (결제 수요 — 영업일 기준 마지막 5일)
    df["month_end_5d"]   = 0
    for i, date in enumerate(df.index):
        month_end = df.index[df.index.month == date.month].max()
        if (month_end - date).days <= 7:
            df.loc[date, "month_end_5d"] = 1

    # 지정학적 리스크
    df["geo_risk"] = 0
    try:
        df.loc["2020-02-20":"2020-04-15", "geo_risk"] = 1
        df.loc["2022-02-24":"2022-06-30", "geo_risk"] = 1
        df.loc["2024-12-03":"2024-12-15", "geo_risk"] = 1
    except Exception:
        pass

    # ── 피처 목록 확정 ────────────────────────────────
    raw_cols = list(TICKERS_DAILY.keys())
    tgt_cols = [f"y_h{h}" for h in HORIZONS]
    features = [
        col for col in df.columns
        if col not in raw_cols + tgt_cols
        and df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]

    # ── 타겟 변수 ──────────────────────────────────────
    if add_targets:
        for h in HORIZONS:
            df[f"y_h{h}"] = np.log(c.shift(-h) / c)
        df.dropna(subset=[f"y_h{h}" for h in HORIZONS], inplace=True)
    else:
        feat_avail = [f for f in features if f in df.columns]
        df = df.dropna(subset=feat_avail[:10])  # 핵심 피처만 체크

    return df, features


# ════════════════════════════════════════════════════════
# ARIMAX 외생변수 준비
# ════════════════════════════════════════════════════════

ARIMAX_EXOG = ["rate_spread", "dxy_ret5", "wti_ret5",
               "vix", "krw_vs_cny_ret5", "yield_curve"]

def get_arimax_exog(df: pd.DataFrame) -> pd.DataFrame:
    """ARIMAX 외생변수 선택 + 누락 컬럼 0 대체"""
    cols = [c for c in ARIMAX_EXOG if c in df.columns]
    exog = df[cols].copy().ffill().fillna(0)
    # 누락 컬럼 0으로 채움
    for c in ARIMAX_EXOG:
        if c not in exog.columns:
            exog[c] = 0.0
    return exog[ARIMAX_EXOG]


# ════════════════════════════════════════════════════════
# 시퀀스 생성 (DL용)
# ════════════════════════════════════════════════════════

def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> tuple:
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len: i])
        ys.append(y[i])
    return (
        np.array(Xs, dtype=np.float32),
        np.array(ys, dtype=np.float32),
    )


# ════════════════════════════════════════════════════════
# 평가 지표
# ════════════════════════════════════════════════════════

def compute_metrics(
    y_true_prices: np.ndarray,
    y_pred_prices: np.ndarray,
    label: str = "",
) -> dict:
    y_true = np.array(y_true_prices).ravel()
    y_pred = np.array(y_pred_prices).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(
        np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))
    ) * 100)
    da   = (
        float(np.mean(
            np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
        ) * 100) if len(y_true) > 1 else 0.0
    )
    ret  = np.diff(y_pred)
    sr   = (
        float((ret.mean() / (ret.std() + 1e-9)) * np.sqrt(252))
        if len(ret) > 1 else 0.0
    )

    result = {
        "RMSE":    round(rmse, 3),
        "MAE":     round(mae,  3),
        "MAPE(%)": round(mape, 3),
        "DA(%)":   round(da,   2),
        "Sharpe":  round(sr,   3),
    }
    if label:
        print(
            f"  [{label:22s}] "
            f"RMSE={rmse:7.2f}원  MAE={mae:7.2f}원  "
            f"MAPE={mape:5.2f}%  DA={da:5.1f}%"
        )
    return result
