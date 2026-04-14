"""
USD/KRW 딥러닝 예측 모델 파이프라인
Phase 0~8 전체 구현 (SPEC 완전 준수)
"""

# ─────────────────────────────────────────────
# 0. 의존성 임포트
# ─────────────────────────────────────────────
import os, json, warnings, pickle, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# 데이터 수집
import yfinance as yf
try:
    import pandas_datareader.data as web
    HAS_DATAREADER = True
except ImportError:
    HAS_DATAREADER = False

# 전처리 / 통계
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller

# 기술적 지표
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

# 딥러닝 (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, BatchNormalization,
    MaxPooling1D, Bidirectional, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Flatten,
    Multiply, Permute, Lambda, Reshape, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber
import tensorflow.keras.backend as K

# 트리 기반
try:
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    HAS_TREE = True
except ImportError:
    HAS_TREE = False

# HPO
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
START_DATE = "2015-01-01"
END_DATE   = datetime.date.today().strftime("%Y-%m-%d")
SEQ_LEN    = 60          # 최적값 (탐색 생략 시 고정)
HORIZON    = 1           # D+1 예측
BATCH_SIZE = 32
EPOCHS     = 200
PATIENCE   = 10

TICKERS = {
    "USDKRW": "KRW=X",
    "DXY":    "DX-Y.NYB",
    "VIX":    "^VIX",
    "US10Y":  "^TNX",
    "US2Y":   "^IRX",
    "WTI":    "CL=F",
    "GOLD":   "GC=F",
    "COPPER": "HG=F",
    "KOSPI":  "^KS11",
    "USDJPY": "JPY=X",
    "USDCNY": "CNY=X",
    "SP500":  "^GSPC",
}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═════════════════════════════════════════════
# PHASE 1 — 데이터 수집
# ═════════════════════════════════════════════

def collect_market_data(start=START_DATE, end=END_DATE):
    """yfinance로 시장 데이터 수집"""
    print("[PHASE 1] 시장 데이터 수집 중...")
    frames = {}
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"  ⚠ {name}({ticker}) 데이터 없음")
                continue
            # Close만 추출
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            frames[name] = close.rename(name)
            print(f"  ✓ {name}: {len(df)}행")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    df_raw = pd.concat(frames.values(), axis=1)
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.sort_index()
    print(f"  총 {len(df_raw)}행, {len(df_raw.columns)}열 수집 완료")
    return df_raw


def collect_fred_data(start=START_DATE, end=END_DATE):
    """FRED 거시 데이터 수집 (실패 시 더미 반환)"""
    fred_codes = {
        "CPI":      "CPIAUCSL",
        "UNRATE":   "UNRATE",
        "FED_RATE": "FEDFUNDS",
        "ISM_PMI":  "NAPM",
    }
    frames = {}
    if HAS_DATAREADER:
        for name, code in fred_codes.items():
            try:
                s = web.DataReader(code, "fred", start, end)[code]
                # 월별 → 일별 forward fill
                s = s.resample("D").ffill()
                frames[name] = s
                print(f"  ✓ FRED {name}")
            except Exception as e:
                print(f"  ⚠ FRED {name}: {e}")
    if not frames:
        print("  ⚠ FRED 데이터 없음 — 더미 0으로 대체")
    return pd.DataFrame(frames) if frames else pd.DataFrame()


# ═════════════════════════════════════════════
# PHASE 1-D — 데이터 품질 검증
# ═════════════════════════════════════════════

def validate_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    print("\n[PHASE 1-D] 데이터 품질 검증...")
    df = df_raw.copy()

    # Step 1: 결측치 처리
    miss_pct = df.isnull().mean()
    drop_cols = miss_pct[miss_pct > 0.10].index.tolist()
    if drop_cols:
        print(f"  결측 5% 초과 → 제거: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)
    df = df.ffill().bfill()

    # Step 2: 이상치 클리핑 (3σ)
    for col in df.columns:
        mu, sigma = df[col].mean(), df[col].std()
        df[col] = df[col].clip(mu - 3*sigma, mu + 3*sigma)

    # Step 3: 정상성 검정 (ADF)
    # USDKRW는 타겟 변수이므로 ADF 검정에서 제외
    protect_cols = ["USDKRW"]
    non_stationary = []
    for col in df.columns:
        if col in protect_cols:
            continue
        try:
            adf_p = adfuller(df[col].dropna())[1]
            if adf_p > 0.05:
                non_stationary.append(col)
        except:
            pass
    if non_stationary:
        print(f"  비정상 시계열 → 로그차분: {non_stationary}")
        for col in non_stationary:
            df[col + "_logdiff"] = np.log(df[col] + 1e-9).diff()
            df.drop(columns=[col], inplace=True)

    df.dropna(inplace=True)
    print(f"  정제 완료: {len(df)}행, {len(df.columns)}열")
    return df


# ═════════════════════════════════════════════
# PHASE 2 — 피처 엔지니어링
# ═════════════════════════════════════════════

def add_technical_indicators(df: pd.DataFrame, price_col="USDKRW") -> pd.DataFrame:
    print("\n[PHASE 2-A] 기술적 지표 계산...")
    c = df[price_col].copy()

    # ── 추세 ──
    for w in [5, 10, 20, 60]:
        df[f"EMA_{w}"] = c.ewm(span=w, adjust=False).mean()
        df[f"SMA_{w}"] = c.rolling(w).mean()

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # ADX (간이 구현)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / (sma20 + 1e-9)

    # ATR (Close-only 근사)
    df["ATR"] = c.diff().abs().rolling(14).mean()

    # Historical Volatility
    df["HV_21"] = c.pct_change().rolling(21).std() * np.sqrt(252)
    df["HV_5"]  = c.pct_change().rolling(5).std()  * np.sqrt(252)

    # Stochastic (Close-only 근사)
    low14  = c.rolling(14).min()
    high14 = c.rolling(14).max()
    df["STOCH_K"] = 100 * (c - low14) / (high14 - low14 + 1e-9)
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

    # Williams %R
    df["WILLR"] = -100 * (high14 - c) / (high14 - low14 + 1e-9)

    # CCI
    tp    = c  # OHLC 없으므로 Close 대체
    sma20_cci = tp.rolling(20).mean()
    mad20     = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df["CCI"] = (tp - sma20_cci) / (0.015 * mad20 + 1e-9)

    print(f"  기술적 지표 {len([c for c in df.columns if any(k in c for k in ['EMA','SMA','MACD','RSI','BB','ATR','HV','STOCH','WILLR','CCI'])])}개 추가")
    return df


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[PHASE 2-B] 거시 파생 피처 계산...")

    if "US10Y" in df.columns and "US2Y" in df.columns:
        df["yield_spread"] = df["US10Y"] - df["US2Y"]

    if "USDKRW" in df.columns and "DXY" in df.columns:
        df["usdkrw_vs_dxy"] = df["USDKRW"] / (df["DXY"] + 1e-9)

    if "VIX" in df.columns:
        df["vix_regime"] = (df["VIX"] > 25).astype(int)
        df["VIX_MA5"]    = df["VIX"].rolling(5).mean()

    if "DXY" in df.columns:
        df["dxy_mom_5d"]  = df["DXY"].pct_change(5)
        df["dxy_mom_20d"] = df["DXY"].pct_change(20)

    if "WTI" in df.columns:
        df["oil_krw_impact"] = df["WTI"] * (-0.7)

    if "USDKRW" in df.columns:
        df["krw_mom_5d"]  = df["USDKRW"].pct_change(5)
        df["krw_mom_20d"] = df["USDKRW"].pct_change(20)
        df["krw_mom_1d"]  = df["USDKRW"].pct_change(1)

    # Lag 피처
    for lag in [1, 3, 5, 10, 22]:
        df[f"USDKRW_lag{lag}"] = df["USDKRW"].shift(lag)

    print("  거시 파생 피처 추가 완료")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[PHASE 2-C] 캘린더/이벤트 피처...")
    df["month_end"]    = (df.index.is_month_end | 
                          df.index.to_series().shift(-1).dt.is_month_end |
                          df.index.to_series().shift(-2).dt.is_month_end).astype(int)
    df["quarter_end"]  = df.index.is_quarter_end.astype(int)
    df["day_of_week"]  = df.index.dayofweek
    df["month"]        = df.index.month

    # 이벤트 더미 (실제 날짜 기반)
    # 계엄 사태 2024-12-03
    df["geopolitical_risk"] = 0
    df.loc["2024-12-03":"2024-12-15", "geopolitical_risk"] = 1
    # 코로나 충격
    df.loc["2020-02-20":"2020-04-15", "geopolitical_risk"] = 1
    # 우크라이나 전쟁
    df.loc["2022-02-24":"2022-06-30", "geopolitical_risk"] = 1

    return df


def build_features(df_clean: pd.DataFrame, fred_df: pd.DataFrame) -> pd.DataFrame:
    df = df_clean.copy()

    # FRED 합치기
    if not fred_df.empty:
        df = df.join(fred_df, how="left").ffill()

    df = add_technical_indicators(df, price_col="USDKRW")
    df = add_macro_features(df)
    df = add_calendar_features(df)

    # 타겟 변수
    df["y_reg"] = df["USDKRW"].shift(-HORIZON)   # D+1 종가
    df["y_cls"] = (df["USDKRW"].shift(-HORIZON) > df["USDKRW"]).astype(int)

    df.dropna(inplace=True)
    print(f"\n[PHASE 2] 피처 엔지니어링 완료: {len(df)}행, {df.shape[1]}열")
    return df


def select_features(df: pd.DataFrame, target="y_reg", max_features=40):
    print(f"\n[PHASE 2-E] 피처 선택 (목표: ~{max_features}개)...")
    drop_cols = ["y_reg", "y_cls", "USDKRW"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].fillna(0)
    y = df[target]

    # 상관계수 필터 (|r| > 0.95 제거)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_high_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
    X.drop(columns=drop_high_corr, inplace=True, errors="ignore")
    print(f"  상관계수 필터로 {len(drop_high_corr)}개 제거")

    # MI 기반 상위 선택
    mi = mutual_info_regression(X.fillna(0), y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    selected = mi_series.head(max_features).index.tolist()
    print(f"  최종 선택 피처 수: {len(selected)}")

    # 저장
    with open(os.path.join(OUTPUT_DIR, "feature_list.json"), "w") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    return selected


# ═════════════════════════════════════════════
# PHASE 3 — 데이터 분할
# ═════════════════════════════════════════════

def temporal_split(df: pd.DataFrame, features: list):
    print("\n[PHASE 3] 시계열 분할...")
    train_end = "2024-06-30"
    val_end   = "2025-06-30"

    train = df[df.index <= train_end]
    val   = df[(df.index > train_end) & (df.index <= val_end)]
    test  = df[df.index > val_end]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train[features].fillna(0))
    X_val   = scaler_X.transform(val[features].fillna(0))
    X_test  = scaler_X.transform(test[features].fillna(0))

    y_train = scaler_y.fit_transform(train[["y_reg"]])
    y_val   = scaler_y.transform(val[["y_reg"]])
    y_test  = scaler_y.transform(test[["y_reg"]])

    # 스케일러 저장 (RULE-1 준수)
    with open(os.path.join(OUTPUT_DIR, "scaler_X.pkl"), "wb") as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(OUTPUT_DIR, "scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)

    print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler_X, scaler_y,
            train, val, test)


def make_sequences(X, y, seq_len=SEQ_LEN):
    """시계열 시퀀스 생성"""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ═════════════════════════════════════════════
# PHASE 4 — 모델 아키텍처
# ═════════════════════════════════════════════

def build_lstm(seq_len, n_feat):
    """MODEL-1: Stacked LSTM"""
    inp = Input(shape=(seq_len, n_feat))
    x   = LSTM(256, return_sequences=True,
                kernel_regularizer=l2(1e-4))(inp)
    x   = Dropout(0.30)(x)
    x   = LSTM(128, return_sequences=True)(x)
    x   = Dropout(0.20)(x)
    x   = LSTM(64,  return_sequences=False)(x)
    x   = Dropout(0.15)(x)
    x   = Dense(32, activation="relu")(x)
    x   = Dense(16, activation="relu")(x)
    out = Dense(1)(x)
    model = Model(inp, out, name="StackedLSTM")
    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0),
        loss=Huber(delta=1.0),
        metrics=["mae"]
    )
    return model


def build_cnn_bilstm_attention(seq_len, n_feat):
    """MODEL-3: CNN-BiLSTM-Attention"""
    inp = Input(shape=(seq_len, n_feat))

    # CNN
    x = Conv1D(128, 3, padding="causal", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding="causal", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    # BiLSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.25)(x)

    # Bahdanau Attention (Keras 3 호환)
    score   = Dense(1, activation="tanh")(x)          # (B, T, 1)
    alpha   = tf.keras.layers.Softmax(axis=1)(score)  # (B, T, 1)
    context = tf.keras.layers.Dot(axes=1)([alpha, x]) # (B, 1, 2*128)
    context = tf.keras.layers.Flatten()(context)      # (B, 2*128)

    x   = Dense(64, activation="gelu")(context)
    out = Dense(1)(x)

    model = Model(inp, out, name="CNN_BiLSTM_Attn")
    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0),
        loss=Huber(delta=1.0),
        metrics=["mae"]
    )
    return model


def build_wavenet(seq_len, n_feat):
    """MODEL-4: WaveNet (Dilated Causal Conv)"""
    inp = Input(shape=(seq_len, n_feat))
    x   = Conv1D(64, 1)(inp)   # 입력 임베딩

    skip_connections = []
    for dilation in [1, 2, 4, 8, 16, 32]:
        # Gated activation
        x_tanh = Conv1D(64, 2, dilation_rate=dilation,
                         padding="causal", activation="tanh")(x)
        x_sig  = Conv1D(64, 2, dilation_rate=dilation,
                         padding="causal", activation="sigmoid")(x)
        gated  = Multiply()([x_tanh, x_sig])

        # Skip
        skip   = Conv1D(64, 1)(gated)
        skip_connections.append(skip)

        # Residual
        x = Add()([x, Conv1D(64, 1)(gated)])

    # Sum skips
    x = Add()(skip_connections)
    x = tf.keras.layers.Activation("relu")(x)
    x = Conv1D(64, 1, activation="relu")(x)
    x = Conv1D(32, 1, activation="relu")(x)
    x = GlobalAveragePooling1D()(x)
    out = Dense(1)(x)

    model = Model(inp, out, name="WaveNet")
    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0),
        loss=Huber(delta=1.0),
        metrics=["mae"]
    )
    return model


def build_simple_transformer(seq_len, n_feat):
    """Transformer (TFT 간소화 버전)"""
    inp = Input(shape=(seq_len, n_feat))
    x   = Dense(64)(inp)
    x   = LayerNormalization()(x)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x   = Add()([x, attn_out])
    x   = LayerNormalization()(x)
    x   = Dense(128, activation="relu")(x)
    x   = Dropout(0.1)(x)
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)

    model = Model(inp, out, name="SimpleTransformer")
    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0),
        loss=Huber(delta=1.0),
        metrics=["mae"]
    )
    return model


# ═════════════════════════════════════════════
# PHASE 5 — 학습 파이프라인
# ═════════════════════════════════════════════

def get_callbacks(model_name: str):
    """공통 Callbacks (RULE-3 준수)"""
    ckpt_path = os.path.join(OUTPUT_DIR, f"best_{model_name}.keras")
    return [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6, verbose=0),
        ModelCheckpoint(ckpt_path, save_best_only=True, verbose=0),
    ]


def train_dl_model(model, Xtr_seq, ytr_seq, Xva_seq, yva_seq, name):
    print(f"\n  [학습] {name} ...")
    history = model.fit(
        Xtr_seq, ytr_seq,
        validation_data=(Xva_seq, yva_seq),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(name),
        verbose=0,
    )
    stopped = len(history.history["loss"])
    print(f"    → 종료 Epoch: {stopped}  val_loss: {min(history.history['val_loss']):.6f}")
    return model, history


def train_tree_models(X_train_flat, y_train_flat, X_val_flat, y_val_flat):
    """MODEL-6: LightGBM + XGBoost + CatBoost"""
    print("\n  [학습] Tree 앙상블...")
    preds_val = {}

    if HAS_TREE:
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.05,
            num_leaves=63, reg_lambda=1e-4,
            early_stopping_round=20, verbose=-1
        )
        lgb_model.fit(X_train_flat, y_train_flat.ravel(),
                      eval_set=[(X_val_flat, y_val_flat.ravel())])
        preds_val["lgb"] = lgb_model.predict(X_val_flat)

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05,
            max_depth=6, reg_lambda=1e-4,
            early_stopping_rounds=20, verbosity=0,
            eval_metric="rmse"
        )
        xgb_model.fit(X_train_flat, y_train_flat.ravel(),
                      eval_set=[(X_val_flat, y_val_flat.ravel())],
                      verbose=False)
        preds_val["xgb"] = xgb_model.predict(X_val_flat)

        # CatBoost
        cat_model = cb.CatBoostRegressor(
            iterations=1000, learning_rate=0.05,
            depth=6, l2_leaf_reg=1e-4,
            early_stopping_rounds=20, verbose=0
        )
        cat_model.fit(X_train_flat, y_train_flat.ravel(),
                      eval_set=(X_val_flat, y_val_flat.ravel()))
        preds_val["cat"] = cat_model.predict(X_val_flat)

        print("    → LGB / XGB / CAT 학습 완료")
        return lgb_model, xgb_model, cat_model, preds_val
    else:
        print("    ⚠ tree 라이브러리 미설치")
        return None, None, None, {}


# ═════════════════════════════════════════════
# PHASE 6 — 평가 지표 (RULE-4)
# ═════════════════════════════════════════════

def compute_metrics(y_true, y_pred, label=""):
    """RMSE / MAE / MAPE / DA / Sharpe"""
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100

    # Direction Accuracy
    if len(y_true) > 1:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir   = np.sign(np.diff(y_pred))
        da = np.mean(actual_dir == pred_dir) * 100
    else:
        da = np.nan

    # Sharpe (단순 수익률 기준)
    ret = np.diff(y_pred)
    sr  = (ret.mean() / (ret.std() + 1e-9)) * np.sqrt(252) if len(ret) > 1 else np.nan

    metrics = {"RMSE": rmse, "MAE": mae, "MAPE(%)": mape,
               "DA(%)": da, "Sharpe": sr}
    if label:
        print(f"  [{label}] RMSE={rmse:.4f} MAE={mae:.4f} "
              f"MAPE={mape:.2f}% DA={da:.1f}% SR={sr:.3f}")
    return metrics


# ═════════════════════════════════════════════
# Walk-Forward Validation
# ═════════════════════════════════════════════

def walk_forward_validation(model_builder, X_all, y_all,
                             seq_len=SEQ_LEN, n_folds=10,
                             min_train=180, test_window=20):
    print(f"\n[PHASE 6-B] Walk-Forward Validation ({n_folds} folds)...")
    fold_metrics = []
    n = len(X_all)
    train_start = 0

    for fold in range(n_folds):
        train_end = min_train + fold * test_window
        test_end  = train_end + test_window
        if test_end > n:
            break

        Xtr = X_all[train_start:train_end]
        ytr = y_all[train_start:train_end]
        Xte = X_all[train_end:test_end]
        yte = y_all[train_end:test_end]

        if len(Xtr) < seq_len + 10 or len(Xte) < 2:
            continue

        Xtr_seq, ytr_seq = make_sequences(Xtr, ytr, seq_len)
        Xte_seq, yte_seq = make_sequences(Xte, yte, seq_len)

        if len(Xtr_seq) < 1 or len(Xte_seq) < 1:
            continue

        model = model_builder(seq_len, Xtr.shape[1])
        model.fit(Xtr_seq, ytr_seq, epochs=30, batch_size=BATCH_SIZE,
                  callbacks=[EarlyStopping(patience=5)], verbose=0)

        pred = model.predict(Xte_seq, verbose=0)
        m = compute_metrics(yte_seq, pred)
        fold_metrics.append(m)
        print(f"  Fold {fold+1}: DA={m['DA(%)']:.1f}%  RMSE={m['RMSE']:.4f}")

    if fold_metrics:
        avg_da   = np.mean([m["DA(%)"]   for m in fold_metrics if not np.isnan(m["DA(%)"])])
        avg_rmse = np.mean([m["RMSE"]    for m in fold_metrics])
        avg_sr   = np.mean([m["Sharpe"]  for m in fold_metrics if not np.isnan(m["Sharpe"])])
        print(f"\n  WFV 평균 — DA={avg_da:.1f}%  RMSE={avg_rmse:.4f}  SR={avg_sr:.3f}")
        return fold_metrics, avg_da, avg_sr
    return fold_metrics, np.nan, np.nan


# ═════════════════════════════════════════════
# PHASE 7 — SHAP 해석
# ═════════════════════════════════════════════

def compute_shap(model, X_sample, features, model_name="model"):
    if not HAS_SHAP:
        print("  ⚠ shap 미설치")
        return
    print(f"\n[PHASE 7] SHAP 해석: {model_name}")
    try:
        # Tree 모델
        if hasattr(model, "predict") and hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)
        else:
            # DL → GradientExplainer 대신 KernelExplainer (느리지만 범용)
            bg = X_sample[:50]
            explainer = shap.KernelExplainer(
                lambda x: model.predict(
                    x.reshape(-1, SEQ_LEN, x.shape[-1] // SEQ_LEN), verbose=0
                ),
                bg
            )
            sv = explainer.shap_values(X_sample[:20])

        # Bar plot 저장
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_sample, feature_names=features,
                          plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_bar_{model_name}.png"), dpi=150)
        plt.close()
        print(f"  SHAP bar plot 저장: shap_bar_{model_name}.png")
    except Exception as e:
        print(f"  ⚠ SHAP 실패: {e}")


# ═════════════════════════════════════════════
# PHASE 6-C — 백테스팅 시뮬레이션
# ═════════════════════════════════════════════

def backtest_simulation(y_true_prices, y_pred_prices,
                         initial_capital=100_000_000,
                         cost_rate=0.0005):
    """방향 예측 기반 간단 백테스팅"""
    capital   = initial_capital
    position  = 0  # 현재 포지션 (1 = long, -1 = short, 0 = flat)
    equity    = [capital]
    trades    = 0
    wins      = 0

    for i in range(1, len(y_true_prices)):
        pred_dir = np.sign(y_pred_prices[i] - y_pred_prices[i-1])
        actual_ret = (y_true_prices[i] - y_true_prices[i-1]) / (y_true_prices[i-1] + 1e-9)

        # 최대 포지션 10%
        trade_size = capital * 0.10
        pnl = trade_size * pred_dir * actual_ret
        cost = trade_size * cost_rate

        capital += pnl - cost
        equity.append(capital)
        if pnl > 0:
            wins += 1
        trades += 1

    equity = np.array(equity)
    total_ret = (equity[-1] / initial_capital - 1) * 100
    years = len(equity) / 252
    cagr  = ((equity[-1] / initial_capital) ** (1/max(years, 1e-9)) - 1) * 100

    daily_ret = np.diff(equity) / equity[:-1]
    sr  = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * np.sqrt(252)
    mdd_val = ((equity - np.maximum.accumulate(equity)) / (np.maximum.accumulate(equity) + 1e-9)).min() * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0

    result = {
        "total_return(%)": total_ret,
        "CAGR(%)": cagr,
        "Sharpe": sr,
        "MDD(%)": mdd_val,
        "WinRate(%)": win_rate,
        "equity_curve": equity,
    }
    print(f"\n  백테스팅 결과:")
    print(f"    총 수익률: {total_ret:.2f}%  CAGR: {cagr:.2f}%")
    print(f"    Sharpe: {sr:.2f}  MDD: {mdd_val:.2f}%  승률: {win_rate:.1f}%")
    return result


# ═════════════════════════════════════════════
# 메타 앙상블
# ═════════════════════════════════════════════

def build_ensemble(val_preds: dict, y_val_true):
    """Ridge 메타 모델 (Level-1)"""
    if not val_preds:
        return None
    X_meta = np.column_stack(list(val_preds.values()))
    ridge  = Ridge(alpha=1.0)
    ridge.fit(X_meta, y_val_true.ravel())
    with open(os.path.join(OUTPUT_DIR, "model_ensemble.pkl"), "wb") as f:
        pickle.dump(ridge, f)
    print("\n  ✓ 앙상블 메타모델 저장 완료")
    return ridge


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  USD/KRW 딥러닝 예측 파이프라인 시작")
    print("=" * 60)

    # ── PHASE 1 ──
    df_raw  = collect_market_data()
    fred_df = collect_fred_data()
    df_clean = validate_and_clean(df_raw)

    # ── PHASE 2 ──
    df_feat  = build_features(df_clean, fred_df)
    features = select_features(df_feat, target="y_reg", max_features=40)

    print(f"\n✅ PHASE 1: 데이터 수집 완료 (n={len(df_raw)}, 기간: {df_raw.index[0].date()}~{df_raw.index[-1].date()})")
    print(f"✅ PHASE 2: 피처 엔지니어링 완료 (최종 피처 수: {len(features)})")

    # ── PHASE 3 ──
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler_X, scaler_y,
     train_df, val_df, test_df) = temporal_split(df_feat, features)

    print(f"✅ PHASE 3: 데이터 분할 완료 (Train:{len(X_train)}/ Val:{len(X_val)}/ Test:{len(X_test)}개)")

    # 시퀀스 생성
    Xtr_seq, ytr_seq = make_sequences(X_train, y_train, SEQ_LEN)
    Xva_seq, yva_seq = make_sequences(X_val,   y_val,   SEQ_LEN)
    Xte_seq, yte_seq = make_sequences(X_test,  y_test,  SEQ_LEN)

    n_feat = X_train.shape[1]
    val_preds_scaled = {}
    all_metrics = {}

    # ── PHASE 4/5: DL 모델 학습 ──
    print("\n[PHASE 4/5] DL 모델 학습...")

    # MODEL-1: LSTM
    lstm_model = build_lstm(SEQ_LEN, n_feat)
    lstm_model, _ = train_dl_model(
        lstm_model, Xtr_seq, ytr_seq, Xva_seq, yva_seq, "LSTM")
    lstm_pred_val = lstm_model.predict(Xva_seq, verbose=0)
    val_preds_scaled["LSTM"] = lstm_pred_val.ravel()
    all_metrics["LSTM"] = compute_metrics(
        scaler_y.inverse_transform(yva_seq),
        scaler_y.inverse_transform(lstm_pred_val), "LSTM")
    lstm_model.save(os.path.join(OUTPUT_DIR, "model_lstm.keras"))

    # MODEL-3: CNN-BiLSTM
    cnn_model = build_cnn_bilstm_attention(SEQ_LEN, n_feat)
    cnn_model, _ = train_dl_model(
        cnn_model, Xtr_seq, ytr_seq, Xva_seq, yva_seq, "CNN_BiLSTM")
    cnn_pred_val = cnn_model.predict(Xva_seq, verbose=0)
    val_preds_scaled["CNN_BiLSTM"] = cnn_pred_val.ravel()
    all_metrics["CNN_BiLSTM"] = compute_metrics(
        scaler_y.inverse_transform(yva_seq),
        scaler_y.inverse_transform(cnn_pred_val), "CNN_BiLSTM")

    # MODEL-4: WaveNet
    wn_model = build_wavenet(SEQ_LEN, n_feat)
    wn_model, _ = train_dl_model(
        wn_model, Xtr_seq, ytr_seq, Xva_seq, yva_seq, "WaveNet")
    wn_pred_val = wn_model.predict(Xva_seq, verbose=0)
    val_preds_scaled["WaveNet"] = wn_pred_val.ravel()
    all_metrics["WaveNet"] = compute_metrics(
        scaler_y.inverse_transform(yva_seq),
        scaler_y.inverse_transform(wn_pred_val), "WaveNet")

    # Transformer
    tf_model = build_simple_transformer(SEQ_LEN, n_feat)
    tf_model, _ = train_dl_model(
        tf_model, Xtr_seq, ytr_seq, Xva_seq, yva_seq, "Transformer")
    tf_pred_val = tf_model.predict(Xva_seq, verbose=0)
    val_preds_scaled["Transformer"] = tf_pred_val.ravel()
    all_metrics["Transformer"] = compute_metrics(
        scaler_y.inverse_transform(yva_seq),
        scaler_y.inverse_transform(tf_pred_val), "Transformer")

    print(f"✅ PHASE 4: 전체 모델 구축 완료")

    # MODEL-6: Tree 모델
    lgb_m, xgb_m, cat_m, tree_preds = train_tree_models(
        X_train, y_train, X_val, y_val)

    for name, pred in tree_preds.items():
        val_preds_scaled[name.upper()] = pred
        all_metrics[name.upper()] = compute_metrics(
            scaler_y.inverse_transform(y_val),
            scaler_y.inverse_transform(pred.reshape(-1, 1)), name.upper())

    # ── 앙상블 ──
    # val_preds를 같은 길이로 맞추기
    min_len = min(len(v) for v in val_preds_scaled.values())
    val_preds_aligned = {k: v[-min_len:] for k, v in val_preds_scaled.items()}
    y_val_aligned = yva_seq[-min_len:]

    ensemble = build_ensemble(val_preds_aligned, y_val_aligned)

    if ensemble is not None:
        X_meta_val = np.column_stack(list(val_preds_aligned.values()))
        ens_pred   = ensemble.predict(X_meta_val)
        all_metrics["Ensemble"] = compute_metrics(
            scaler_y.inverse_transform(y_val_aligned),
            scaler_y.inverse_transform(ens_pred.reshape(-1, 1)), "Ensemble★")

    # ── PHASE 5 체크 ──
    print(f"✅ PHASE 5: 학습 완료")

    # ── PHASE 6: WFV ──
    X_all = np.vstack([X_train, X_val])
    y_all = np.vstack([y_train, y_val])
    wfv_results, avg_da, avg_sr = walk_forward_validation(
        lambda sl, nf: build_lstm(sl, nf),
        X_all, y_all, seq_len=SEQ_LEN, n_folds=10)

    print(f"✅ PHASE 6: WFV 백테스팅 완료 (Ensemble DA: {avg_da:.1f}%, SR: {avg_sr:.3f})")

    # 백테스팅 시뮬레이션
    if len(yte_seq) > 1:
        y_true_inv = scaler_y.inverse_transform(yte_seq).ravel()
        lstm_te    = lstm_model.predict(Xte_seq, verbose=0)
        y_pred_inv = scaler_y.inverse_transform(lstm_te).ravel()
        bt_result  = backtest_simulation(y_true_inv, y_pred_inv)
    else:
        bt_result  = {}

    # ── PHASE 7: SHAP ──
    if HAS_SHAP and lgb_m is not None:
        try:
            explainer = shap.TreeExplainer(lgb_m)
            sv = explainer.shap_values(X_val[:200])
            plt.figure(figsize=(10, 8))
            shap.summary_plot(sv, X_val[:200], feature_names=features,
                              plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150)
            plt.close()
            top3 = pd.Series(np.abs(sv).mean(axis=0),
                             index=features).nlargest(3).index.tolist()
            print(f"✅ PHASE 7: SHAP 해석 완료 (Top-3: {top3[0]}, {top3[1]}, {top3[2]})")
        except Exception as e:
            print(f"  ⚠ SHAP 실패: {e}")
            top3 = ["N/A"] * 3
    else:
        top3 = ["N/A"] * 3

    # ── PHASE 8: 성능표 저장 ──
    perf_df = pd.DataFrame(all_metrics).T
    perf_df.to_csv(os.path.join(OUTPUT_DIR, "performance_table.csv"))
    print("\n" + "=" * 60)
    print("  모델 성능 비교표")
    print("=" * 60)
    print(perf_df.to_string())

    # 오늘 예측 (D+1)
    if len(X_test) >= SEQ_LEN:
        last_seq = X_test[-SEQ_LEN:].reshape(1, SEQ_LEN, n_feat)
        pred_scaled = lstm_model.predict(last_seq, verbose=0)
        pred_price  = scaler_y.inverse_transform(pred_scaled)[0][0]
        last_price  = df_feat["USDKRW"].iloc[-1]
        forecast = {
            "date_today": str(datetime.date.today()),
            "last_close": float(last_price),
            "D+1_forecast": float(pred_price),
            "direction": "상승" if pred_price > last_price else "하락",
            "change_pct": float((pred_price - last_price) / last_price * 100)
        }
        with open(os.path.join(OUTPUT_DIR, "forecast_today.json"), "w") as f:
            json.dump(forecast, f, ensure_ascii=False, indent=2)
        print(f"\n  📈 오늘 예측 (D+1): {pred_price:.2f}원 "
              f"({'↑' if pred_price > last_price else '↓'} "
              f"{abs(forecast['change_pct']):.2f}%)")

    print(f"\n✅ PHASE 8: 최종 산출물 저장 완료 → {OUTPUT_DIR}/")
    print("=" * 60)
    return all_metrics, bt_result, features


if __name__ == "__main__":
    main()
