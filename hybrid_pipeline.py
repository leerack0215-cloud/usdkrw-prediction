
import os, json, pickle, datetime, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, GlobalAveragePooling1D,
    Bidirectional, BatchNormalization, Add, Multiply,
    LayerNormalization, MultiHeadAttention, Flatten,
    GRU, Concatenate, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import lightgbm as lgb

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEQ_LEN    = 30
HORIZON    = 1
BATCH_SIZE = 32
EPOCHS     = 100

TICKERS = {
    "USDKRW": "KRW=X",
    "DXY":    "DX-Y.NYB",
    "VIX":    "^VIX",
    "US10Y":  "^TNX",
    "WTI":    "CL=F",
    "GOLD":   "GC=F",
    "KOSPI":  "^KS11",
    "SP500":  "^GSPC",
}

# ══════════════════════════════════
# 데이터 수집
# ══════════════════════════════════
def collect_data():
    print("[1] 데이터 수집...")
    frames = {}
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start="2015-01-01",
                             auto_adjust=True, progress=False)
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            frames[name] = close.rename(name)
            print(f"  ✓ {name}: {len(df)}행")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    df = pd.concat(frames.values(), axis=1)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

# ══════════════════════════════════
# 피처 엔지니어링
# ══════════════════════════════════
def make_features(df):
    print("[2] 피처 엔지니어링...")
    c = df["USDKRW"].copy()

    # 추세
    for w in [5, 10, 20, 60]:
        df[f"EMA_{w}"] = c.ewm(span=w).mean()
        df[f"SMA_{w}"] = c.rolling(w).mean()

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100/(1+gain/(loss+1e-9)))

    # 볼린저
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BB_upper"] = sma20 + 2*std20
    df["BB_lower"] = sma20 - 2*std20
    df["BB_width"] = (4*std20)/(sma20+1e-9)

    # ATR
    df["ATR"] = c.diff().abs().rolling(14).mean()

    # 모멘텀
    for d in [1, 3, 5, 10, 20]:
        df[f"mom_{d}"] = c.pct_change(d)

    # 변동성
    df["HV_10"] = c.pct_change().rolling(10).std()*np.sqrt(252)
    df["HV_21"] = c.pct_change().rolling(21).std()*np.sqrt(252)

    # 거시
    if "VIX" in df.columns:
        df["vix_regime"] = (df["VIX"] > 25).astype(int)
        df["VIX_ma5"]    = df["VIX"].rolling(5).mean()
    if "DXY" in df.columns:
        df["dxy_mom5"]  = df["DXY"].pct_change(5)
        df["dxy_mom20"] = df["DXY"].pct_change(20)
    if "US10Y" in df.columns:
        df["US10Y_ma5"] = df["US10Y"].rolling(5).mean()

    # 래그
    for lag in [1, 3, 5, 10]:
        df[f"lag_{lag}"] = c.shift(lag)

    # 캘린더
    df["month_end"]    = df.index.is_month_end.astype(int)
    df["quarter_end"]  = df.index.is_quarter_end.astype(int)
    df["day_of_week"]  = df.index.dayofweek
    df["month"]        = df.index.month

    # 지정학적 리스크
    df["geo_risk"] = 0
    df.loc["2020-02-20":"2020-04-15", "geo_risk"] = 1
    df.loc["2022-02-24":"2022-06-30", "geo_risk"] = 1
    df.loc["2024-12-03":"2024-12-15", "geo_risk"] = 1

    # 타겟: 절대값 대신 로그수익률 예측 (스케일 문제 해결)
    df["y_reg"] = np.log(c.shift(-1) / c)   # D+1 로그수익률
    df["y_d3"]  = np.log(c.shift(-3) / c)   # D+3
    df["y_d5"]  = np.log(c.shift(-5) / c)   # D+5
    df["y_d10"] = np.log(c.shift(-10) / c)  # D+10
    df["y_d22"] = np.log(c.shift(-22) / c)  # D+22 (월)
    df.dropna(inplace=True)

    y_cols   = ["y_reg","y_d3","y_d5","y_d10","y_d22"]
    features = [col for col in df.columns
                if col not in y_cols + ["USDKRW"]]
    print(f"  피처 수: {len(features)}")
    return df, features

# ══════════════════════════════════
# 시퀀스 생성
# ══════════════════════════════════
def make_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# ══════════════════════════════════
# 평가
# ══════════════════════════════════
def metrics(y_true, y_pred, label=""):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true-y_pred)/(np.abs(y_true)+1e-9)))*100
    da   = np.mean(np.sign(np.diff(y_true))==np.sign(np.diff(y_pred)))*100 if len(y_true)>1 else 0
    ret  = np.diff(y_pred)
    sr   = (ret.mean()/(ret.std()+1e-9))*np.sqrt(252) if len(ret)>1 else 0
    if label:
        print(f"  [{label}] RMSE={rmse:.2f} MAE={mae:.2f} "
              f"MAPE={mape:.2f}% DA={da:.1f}% SR={sr:.3f}")
    return {"RMSE":rmse,"MAE":mae,"MAPE(%)":mape,"DA(%)":da,"Sharpe":sr}

cb = [EarlyStopping(patience=10, restore_best_weights=True),
      ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=0)]

# ══════════════════════════════════
# MODEL 1: Stacked LSTM
# ══════════════════════════════════
def build_lstm(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x   = LSTM(128, return_sequences=True)(inp)
    x   = Dropout(0.2)(x)
    x   = LSTM(64, return_sequences=True)(x)
    x   = Dropout(0.15)(x)
    x   = LSTM(32)(x)
    x   = Dropout(0.1)(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m   = Model(inp, out, name="LSTM")
    m.compile(optimizer=Adam(1e-3), loss=Huber())
    return m

# ══════════════════════════════════
# MODEL 2: GRU (LSTM 보완)
# ══════════════════════════════════
def build_gru(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x   = Bidirectional(GRU(64, return_sequences=True))(inp)
    x   = Dropout(0.2)(x)
    x   = Bidirectional(GRU(32))(x)
    x   = Dropout(0.1)(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m   = Model(inp, out, name="BiGRU")
    m.compile(optimizer=Adam(1e-3), loss=Huber())
    return m

# ══════════════════════════════════
# MODEL 3: Transformer
# ══════════════════════════════════
def build_transformer(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x   = Dense(64)(inp)
    x   = LayerNormalization()(x)
    # Multi-Head Attention
    attn = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = Add()([x, attn])
    x    = LayerNormalization()(x)
    # FFN
    ffn  = Dense(128, activation="relu")(x)
    ffn  = Dense(64)(ffn)
    x    = Add()([x, ffn])
    x    = LayerNormalization()(x)
    x    = GlobalAveragePooling1D()(x)
    x    = Dense(32, activation="relu")(x)
    x    = Dropout(0.1)(x)
    out  = Dense(1)(x)
    m    = Model(inp, out, name="Transformer")
    m.compile(optimizer=Adam(1e-3), loss=Huber())
    return m

# ══════════════════════════════════
# MODEL 4: WaveNet (Dilated CNN)
# ══════════════════════════════════
def build_wavenet(seq_len, n_feat):
    inp  = Input(shape=(seq_len, n_feat))
    x    = Conv1D(32, 1)(inp)
    skip_list = []
    for d in [1, 2, 4, 8]:
        x_t = Conv1D(32, 2, dilation_rate=d,
                     padding="causal", activation="tanh")(x)
        x_s = Conv1D(32, 2, dilation_rate=d,
                     padding="causal", activation="sigmoid")(x)
        gated = Multiply()([x_t, x_s])
        skip  = Conv1D(32, 1)(gated)
        skip_list.append(skip)
        res   = Conv1D(32, 1)(gated)
        x     = Add()([x, res])
    x   = Add()(skip_list)
    x   = tf.keras.layers.Activation("relu")(x)
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m   = Model(inp, out, name="WaveNet")
    m.compile(optimizer=Adam(1e-3), loss=Huber())
    return m

# ══════════════════════════════════
# MODEL 5: CNN-LSTM 하이브리드
# (CNN으로 패턴 추출 → LSTM으로 시계열 학습)
# ══════════════════════════════════
def build_cnn_lstm(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    # CNN 블록
    x   = Conv1D(64, 3, padding="causal", activation="relu")(inp)
    x   = BatchNormalization()(x)
    x   = Conv1D(64, 3, padding="causal", activation="relu")(x)
    x   = BatchNormalization()(x)
    # LSTM 블록
    x   = LSTM(64, return_sequences=True)(x)
    x   = Dropout(0.2)(x)
    x   = LSTM(32)(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m   = Model(inp, out, name="CNN_LSTM")
    m.compile(optimizer=Adam(1e-3), loss=Huber())
    return m

# ══════════════════════════════════
# MAIN
# ══════════════════════════════════
def main():
    print("="*55)
    print("  하이브리드 딥러닝 앙상블 파이프라인")
    print("="*55)

    # 데이터
    df_raw = collect_data()
    df, features = make_features(df_raw.copy())

    # 분할
    train = df[df.index <= "2024-06-30"]
    val   = df[(df.index > "2024-06-30") & (df.index <= "2025-06-30")]
    test  = df[df.index > "2025-06-30"]

    # X 스케일러만 사용 (y는 로그수익률이라 스케일 불필요)
    scaler_X = MinMaxScaler()

    X_train = scaler_X.fit_transform(train[features].fillna(0))
    X_val   = scaler_X.transform(val[features].fillna(0))
    X_test  = scaler_X.transform(test[features].fillna(0))

    # 다중 호라이즌 타겟
    horizons = {
        "D+1":"y_reg","D+3":"y_d3","D+5":"y_d5",
        "D+10":"y_d10","D+22":"y_d22"
    }
    y_data = {}
    for h, col in horizons.items():
        y_data[h] = {
            "train": train[[col]].values,
            "val":   val[[col]].values,
            "test":  test[[col]].values,
        }

    with open(f"{OUTPUT_DIR}/scaler_X.pkl","wb") as f:
        pickle.dump(scaler_X, f)
    with open(f"{OUTPUT_DIR}/feature_list.json","w") as f:
        json.dump(features, f, ensure_ascii=False)

    # 실제 가격 저장 (예측값 복원용)
    last_prices = {
        "train": train["USDKRW"].values,
        "val":   val["USDKRW"].values,
        "test":  test["USDKRW"].values,
    }

    print(f"  Train:{len(train)} Val:{len(val)} Test:{len(test)}")

    # D+1 기준으로 모델 학습
    y_train = y_data["D+1"]["train"]
    y_val   = y_data["D+1"]["val"]
    y_test  = y_data["D+1"]["test"]

    Xtr,ytr = make_sequences(X_train, y_train)
    Xva,yva = make_sequences(X_val,   y_val)
    Xte,yte = make_sequences(X_test,  y_test)
    n_feat  = X_train.shape[1]

    all_metrics = {}
    val_preds   = {}
    model_store = {}

    # ── 딥러닝 모델 5개 학습 ──
    dl_models = {
        "LSTM":      build_lstm(SEQ_LEN, n_feat),
        "BiGRU":     build_gru(SEQ_LEN, n_feat),
        "Transformer": build_transformer(SEQ_LEN, n_feat),
        "WaveNet":   build_wavenet(SEQ_LEN, n_feat),
        "CNN_LSTM":  build_cnn_lstm(SEQ_LEN, n_feat),
    }

    for name, model in dl_models.items():
        print(f"\n[{name} 학습]")
        model.fit(Xtr, ytr,
                  validation_data=(Xva, yva),
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  callbacks=cb, verbose=0)
        p = model.predict(Xva, verbose=0)
        val_preds[name] = p.ravel()
        # 로그수익률 → 실제 가격으로 변환해서 평가
        val_prices = last_prices["val"][SEQ_LEN:]
        y_true_price = val_prices * np.exp(yva.ravel())
        y_pred_price = val_prices * np.exp(p.ravel())
        all_metrics[name] = metrics(y_true_price, y_pred_price, name)
        model.save(f"{OUTPUT_DIR}/model_{name.lower()}.keras")
        size = os.path.getsize(
            f"{OUTPUT_DIR}/model_{name.lower()}.keras")/1e6
        print(f"  저장: {size:.1f}MB")
        model_store[name] = model

    # ── LightGBM (트리 보완) ──
    print("\n[LightGBM 학습]")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, verbose=-1)
    lgb_model.fit(X_train, y_train.ravel(),
                  eval_set=[(X_val, y_val.ravel())],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
    p_lgb = lgb_model.predict(X_val)
    val_preds["LGB"] = p_lgb
    val_prices_flat = last_prices["val"]
    y_true_lgb = val_prices_flat * np.exp(y_val.ravel())
    y_pred_lgb = val_prices_flat * np.exp(p_lgb.ravel())
    all_metrics["LGB"] = metrics(y_true_lgb, y_pred_lgb, "LGB")
    with open(f"{OUTPUT_DIR}/model_lgb.pkl","wb") as f:
        pickle.dump(lgb_model, f)

    # ══════════════════════════════
    # Level-1: 메타 앙상블
    # 성능 기반 가중치 자동 계산
    # ══════════════════════════════
    print("\n[메타 앙상블 구성]")

    # RMSE 역수로 가중치 계산 (성능 좋을수록 높은 가중치)
    rmse_dict = {k: all_metrics[k]["RMSE"] for k in val_preds}
    inv_rmse  = {k: 1/(v+1e-9) for k,v in rmse_dict.items()}
    total_inv = sum(inv_rmse.values())
    weights   = {k: v/total_inv for k,v in inv_rmse.items()}

    print("  모델별 가중치 (RMSE 역수 기반):")
    for k,w in sorted(weights.items(), key=lambda x:-x[1]):
        print(f"    {k}: {w:.3f} (RMSE={rmse_dict[k]:.2f})")

    # Ridge 메타 모델
    min_len = min(len(v) for v in val_preds.values())
    aligned = {k: v[-min_len:] for k,v in val_preds.items()}
    yva_a   = yva[-min_len:]

    X_meta  = np.column_stack(list(aligned.values()))
    ridge   = Ridge(alpha=0.5)
    ridge.fit(X_meta, yva_a.ravel())
    p_ridge = ridge.predict(X_meta)

    # MLP 메타 모델
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        max_iter=500, random_state=42,
        early_stopping=True)
    mlp.fit(X_meta, yva_a.ravel())
    p_mlp = mlp.predict(X_meta)

    # 최종 앙상블 = Ridge 60% + MLP 40%
    p_final = 0.6 * p_ridge + 0.4 * p_mlp
    val_prices_a = last_prices["val"][SEQ_LEN:][-min_len:]
    y_true_ens = val_prices_a * np.exp(yva_a.ravel())
    y_pred_ens = val_prices_a * np.exp(p_final.ravel())
    all_metrics["★Hybrid_Ensemble"] = metrics(
        y_true_ens, y_pred_ens, "★Hybrid_Ensemble")

    with open(f"{OUTPUT_DIR}/model_ensemble.pkl","wb") as f:
        pickle.dump({"ridge":ridge,"mlp":mlp,"weights":weights}, f)

    # ══════════════════════════════
    # 성능표 출력
    # ══════════════════════════════
    print("\n" + "="*55)
    print("  모델 성능 비교표")
    print("="*55)
    perf_df = pd.DataFrame(all_metrics).T
    print(perf_df.round(2).to_string())
    perf_df.to_csv(f"{OUTPUT_DIR}/performance_table.csv")

    # ══════════════════════════════
    # 오늘 예측 — 전 모델
    # ══════════════════════════════
    print("\n[다중 호라이즌 예측]")
    last_price = float(df["USDKRW"].iloc[-1])
    print(f"  현재 환율: {last_price:,.2f}원")

    # 각 호라이즌별 로그수익률 예측 → 실제 가격 변환
    horizon_days = {"D+1":1,"D+3":3,"D+5":5,"D+10":10,"D+22":22}
    forecasts_by_model  = {}   # {모델명: {호라이즌: 예측가격}}
    hybrid_by_horizon   = {}   # {호라이즌: 하이브리드 예측가격}

    if len(X_test) >= SEQ_LEN:
        last_seq  = X_test[-SEQ_LEN:].reshape(1, SEQ_LEN, n_feat)
        last_flat = X_test[-1:].reshape(1, -1)

        # 호라이즌별 LGB 모델 별도 학습 (더 안정적)
        lgb_horizon_models = {}
        horizon_col_map = {
            "D+1": "y_reg", "D+3": "y_d3",
            "D+5": "y_d5",  "D+10": "y_d10", "D+22": "y_d22"
        }
        print("  호라이즌별 LGB 학습 중...")
        for h_label, y_col in horizon_col_map.items():
            ytr_h = y_data[h_label]["train"].ravel()
            yva_h = y_data[h_label]["val"].ravel()
            m_h = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05,
                num_leaves=15, verbose=-1,
                reg_lambda=0.1)
            m_h.fit(X_train, ytr_h,
                    eval_set=[(X_val, yva_h)],
                    callbacks=[lgb.early_stopping(20,verbose=False)])
            lgb_horizon_models[h_label] = m_h
            print(f"    {h_label} 완료")

        # 각 호라이즌 예측
        for h_label, h_days in horizon_days.items():
            meta_inputs = []

            for name, model in model_store.items():
                # D+1 로그수익률로 단기 예측
                log_ret = float(model.predict(last_seq, verbose=0)[0][0])
                # 클리핑: 일별 수익률 ±3% 이내로 제한
                log_ret = np.clip(log_ret, -0.03, 0.03)
                pred_price = last_price * np.exp(log_ret * h_days)
                if name not in forecasts_by_model:
                    forecasts_by_model[name] = {}
                forecasts_by_model[name][h_label] = round(float(pred_price), 2)
                meta_inputs.append(log_ret)

            # 호라이즌 전용 LGB
            lgb_h = lgb_horizon_models[h_label]
            lgb_log_h = float(lgb_h.predict(last_flat)[0])
            lgb_log_h = np.clip(lgb_log_h, -0.03, 0.03)
            lgb_price_h = last_price * np.exp(lgb_log_h)
            if "LGB" not in forecasts_by_model:
                forecasts_by_model["LGB"] = {}
            forecasts_by_model["LGB"][h_label] = round(float(lgb_price_h), 2)
            meta_inputs.append(lgb_log_h)

            # 하이브리드 앙상블
            meta_arr  = np.array(meta_inputs).reshape(1,-1)
            p_r       = float(ridge.predict(meta_arr)[0])
            p_m       = float(mlp.predict(meta_arr)[0])
            hybrid_lr = np.clip(0.6*p_r + 0.4*p_m, -0.03, 0.03)
            hybrid_price = last_price * np.exp(hybrid_lr)
            hybrid_by_horizon[h_label] = round(float(hybrid_price), 2)

        # 출력
        print(f"\n  {'호라이즌':8s} {'★Hybrid':>10s} {'LSTM':>10s} "
              f"{'BiGRU':>10s} {'LGB':>10s}")
        print("  " + "-"*52)
        for h in horizon_days:
            hyb  = hybrid_by_horizon[h]
            lstm = forecasts_by_model.get("LSTM",{}).get(h, 0)
            gru  = forecasts_by_model.get("BiGRU",{}).get(h, 0)
            lgbv = forecasts_by_model.get("LGB",{}).get(h, 0)
            arr  = "↑" if hyb > last_price else "↓"
            pct  = (hyb/last_price-1)*100
            print(f"  {h:8s} {hyb:>10,.2f} {lstm:>10,.2f} "
                  f"{gru:>10,.2f} {lgbv:>10,.2f}  {arr}{pct:+.2f}%")

    # JSON 저장
    best_d1 = hybrid_by_horizon.get("D+1", last_price)
    forecast_out = {
        "date_today":        str(datetime.date.today()),
        "last_close":        last_price,
        "D+1_forecast":      hybrid_by_horizon.get("D+1", last_price),
        "D+3_forecast":      hybrid_by_horizon.get("D+3", last_price),
        "D+5_forecast":      hybrid_by_horizon.get("D+5", last_price),
        "D+10_forecast":     hybrid_by_horizon.get("D+10", last_price),
        "D+22_forecast":     hybrid_by_horizon.get("D+22", last_price),
        "direction":         "상승" if best_d1 > last_price else "하락",
        "change_pct":        round((best_d1/last_price-1)*100, 3),
        "models":            forecasts_by_model,
        "hybrid_forecasts":  hybrid_by_horizon,
        "weights":           {k:round(float(v),4) for k,v in weights.items()},
    }
    with open(f"{OUTPUT_DIR}/forecast_today.json","w") as f:
        json.dump(forecast_out, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ forecast_today.json 저장 완료")

    # 파일 크기 확인
    print("\n[파일 크기]")
    total = 0
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)/1e6
            total += size
            print(f"  {fname}: {size:.1f}MB")
    print(f"  총합: {total:.1f}MB")

    print("\n✅ 하이브리드 앙상블 완료!")
    return all_metrics, forecasts_by_model

if __name__ == "__main__":
    main()
