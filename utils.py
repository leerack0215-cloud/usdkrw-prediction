"""
utils.py — USD/KRW 예측 시스템 공유 유틸리티
모든 파일의 단일 소스:
  - 공용 상수 (HORIZONS / SEQ_LEN / CLIP_BOUNDS / 경로)
  - 데이터 수집 / 피처 엔지니어링 / 시퀀스 생성 / 평가 지표

설계 원칙:
  - Lookahead Bias 완전 차단
  - 학습/추론 공통 사용 (일관성 보장)
  - 누락 티커 안전 처리
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ════════════════════════════════════════════════════════
# 공용 상수 — 모든 파일은 여기서만 import
# ════════════════════════════════════════════════════════

TICKERS = {
    "USDKRW": "KRW=X",
    "DXY":    "DX-Y.NYB",
    "VIX":    "^VIX",
    "US10Y":  "^TNX",
    "WTI":    "CL=F",
    "GOLD":   "GC=F",
    "KOSPI":  "^KS11",
    "SP500":  "^GSPC",
    "USDJPY": "JPY=X",
}

HORIZONS       = [1, 3, 5, 10, 22]
HORIZON_LABELS = {1: "D+1", 3: "D+3", 5: "D+5", 10: "D+10", 22: "D+22"}
SEQ_LEN        = 30
OUTPUT_DIR     = "outputs"
MODELS_DIR     = os.path.join(OUTPUT_DIR, "models")

# 호라이즌별 로그수익률 클리핑 한계
CLIP_BOUNDS = {1: 0.025, 3: 0.040, 5: 0.055, 10: 0.080, 22: 0.120}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════
# 데이터 수집
# ════════════════════════════════════════════════════════

def collect_data(start: str = "2015-01-01") -> pd.DataFrame:
    """
    yfinance 시장 데이터 수집
    - USDKRW 필수 / 나머지 실패 시 경고 후 스킵
    - 결측치 forward/backward fill
    """
    end    = datetime.date.today().strftime("%Y-%m-%d")
    frames = {}

    for name, ticker in TICKERS.items():
        try:
            raw = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False,
            )
            if raw is None or raw.empty:
                print(f"  ⚠ {name}: 빈 데이터 — 스킵")
                continue

            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = close.dropna()
            if len(close) < 100:
                print(f"  ⚠ {name}: 데이터 부족({len(close)}행) — 스킵")
                continue

            frames[name] = close.rename(name)
            print(f"  ✓ {name}: {len(close)}행")

        except Exception as e:
            print(f"  ✗ {name}: {e}")

    if "USDKRW" not in frames:
        raise RuntimeError("USDKRW 데이터 수집 실패 — 인터넷 연결을 확인하세요.")

    df = pd.concat(frames.values(), axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().ffill().bfill()
    return df


# ════════════════════════════════════════════════════════
# 피처 엔지니어링
# ════════════════════════════════════════════════════════

def make_features(
    df: pd.DataFrame,
    add_targets: bool = True,
) -> tuple:
    """
    피처 엔지니어링 (Lookahead Bias 완전 차단)

    Parameters
    ----------
    df          : collect_data() 반환 DataFrame
    add_targets : True  → 학습용 (y_h1~h22 추가, dropna)
                  False → 추론용 (타겟 없음, 최신 행 보존)

    Returns
    -------
    (df_processed, feature_names)
    """
    df = df.copy()
    c  = df["USDKRW"]

    # ── 추세 지표 ──────────────────────────────────────
    for w in [5, 10, 20, 60]:
        df[f"ema_{w}"]    = c.ewm(span=w, adjust=False).mean()
        df[f"sma_{w}"]    = c.rolling(w).mean()
        df[f"vs_ema_{w}"] = (c / (df[f"ema_{w}"] + 1e-9) - 1) * 100
        df[f"vs_sma_{w}"] = (c / (df[f"sma_{w}"] + 1e-9) - 1) * 100

    # MACD
    ema12           = c.ewm(span=12, adjust=False).mean()
    ema26           = c.ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # ── 모멘텀 ────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    low14         = c.rolling(14).min()
    high14        = c.rolling(14).max()
    df["stoch_k"] = 100 * (c - low14)  / (high14 - low14  + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["willr"]   = -100 * (high14 - c) / (high14 - low14 + 1e-9)

    # CCI
    sma20_cci = c.rolling(20).mean()
    mad20     = c.rolling(20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    df["cci"] = (c - sma20_cci) / (0.015 * mad20 + 1e-9)

    # ── 변동성 ────────────────────────────────────────
    sma20          = c.rolling(20).mean()
    std20          = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (4 * std20) / (sma20 + 1e-9)
    df["bb_pos"]   = (c - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"] + 1e-9
    )
    df["atr"]   = c.diff().abs().rolling(14).mean()
    df["hv_5"]  = c.pct_change().rolling(5).std()  * np.sqrt(252) * 100
    df["hv_21"] = c.pct_change().rolling(21).std() * np.sqrt(252) * 100

    # ── 수익률 피처 ────────────────────────────────────
    for d in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{d}d"] = c.pct_change(d) * 100

    # 래그 수익률 (과거값만 — Lookahead 없음)
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_ret_{lag}"] = c.pct_change().shift(lag) * 100

    # ── 거시 지표 ──────────────────────────────────────
    if "VIX" in df.columns:
        df["vix"]        = df["VIX"]
        df["vix_ma5"]    = df["VIX"].rolling(5).mean()
        df["vix_ret5"]   = df["VIX"].pct_change(5) * 100
        df["vix_regime"] = (df["VIX"] > 25).astype(int)

    if "DXY" in df.columns:
        df["dxy_ret5"]    = df["DXY"].pct_change(5)  * 100
        df["dxy_ret20"]   = df["DXY"].pct_change(20) * 100
        df["dxy_vs_ma20"] = (
            df["DXY"] / (df["DXY"].rolling(20).mean() + 1e-9) - 1
        ) * 100

    if "US10Y" in df.columns:
        df["us10y"]      = df["US10Y"]
        df["us10y_ret5"] = df["US10Y"].pct_change(5) * 100

    if "WTI"    in df.columns: df["wti_ret5"]    = df["WTI"].pct_change(5)    * 100
    if "GOLD"   in df.columns: df["gold_ret5"]   = df["GOLD"].pct_change(5)   * 100
    if "KOSPI"  in df.columns: df["kospi_ret5"]  = df["KOSPI"].pct_change(5)  * 100
    if "SP500"  in df.columns:
        df["sp500_ret5"]  = df["SP500"].pct_change(5)  * 100
        df["sp500_ret20"] = df["SP500"].pct_change(20) * 100
    if "USDJPY" in df.columns: df["usdjpy_ret5"] = df["USDJPY"].pct_change(5) * 100

    # ── 캘린더 피처 ────────────────────────────────────
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month
    df["month_end"]   = df.index.is_month_end.astype(int)
    df["quarter_end"] = df.index.is_quarter_end.astype(int)

    # ── 지정학적 리스크 더미 ───────────────────────────
    df["geo_risk"] = 0
    try:
        df.loc["2020-02-20":"2020-04-15", "geo_risk"] = 1  # 코로나
        df.loc["2022-02-24":"2022-06-30", "geo_risk"] = 1  # 우크라이나
        df.loc["2024-12-03":"2024-12-15", "geo_risk"] = 1  # 계엄
    except Exception:
        pass

    # ── 피처 목록 확정 (원본 시세·타겟 제외) ──────────
    raw_cols = list(TICKERS.keys())
    tgt_cols = [f"y_h{h}" for h in HORIZONS]
    features = [
        col for col in df.columns
        if col not in raw_cols + tgt_cols
    ]

    # ── 타겟 변수 ──────────────────────────────────────
    # y_h{n} = log(P(t+n) / P(t))  — n일 직접 로그수익률
    # 누적 아님! 각 호라이즌을 독립적으로 직접 예측
    if add_targets:
        for h in HORIZONS:
            df[f"y_h{h}"] = np.log(c.shift(-h) / c)
        df.dropna(inplace=True)
    else:
        # 추론용: 최신 행 보존 (타겟 NaN 제거 없음)
        feat_available = [f for f in features if f in df.columns]
        df = df.dropna(subset=feat_available)

    return df, features


# ════════════════════════════════════════════════════════
# 시퀀스 생성 (DL용)
# ════════════════════════════════════════════════════════

def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> tuple:
    """시계열 시퀀스 (batch, seq_len, n_feat) 생성"""
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
    """
    실제 가격 단위(원화)로 평가
    RMSE / MAE / MAPE / DA / Sharpe
    """
    y_true = np.array(y_true_prices).ravel()
    y_pred = np.array(y_pred_prices).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    )
    da = (
        float(np.mean(
            np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
        ) * 100)
        if len(y_true) > 1 else 0.0
    )
    ret = np.diff(y_pred)
    sr  = (
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
            f"  [{label:20s}] "
            f"RMSE={rmse:7.2f}원  MAE={mae:7.2f}원  "
            f"MAPE={mape:5.2f}%  DA={da:5.1f}%"
        )
    return result
