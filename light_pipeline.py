
import os, json, pickle, datetime, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, GlobalAveragePooling1D, Bidirectional, BatchNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import lightgbm as lgb

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN    = 30   # 60→30 (속도 향상)
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

# ── 데이터 수집 ──
def collect_data():
    print("[1] 데이터 수집 중...")
    frames = {}
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start="2018-01-01",
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

# ── 피처 엔지니어링 ──
def make_features(df):
    print("[2] 피처 엔지니어링...")
    c = df["USDKRW"].copy()

    # 기술적 지표
    for w in [5, 10, 20]:
        df[f"EMA_{w}"] = c.ewm(span=w).mean()
        df[f"SMA_{w}"] = c.rolling(w).mean()

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain/(loss+1e-9)))

    # 볼린저
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BB_width"] = (4 * std20) / (sma20 + 1e-9)

    # 모멘텀
    for d in [1, 5, 10, 20]:
        df[f"mom_{d}"] = c.pct_change(d)

    # 변동성
    df["HV_10"] = c.pct_change().rolling(10).std() * np.sqrt(252)

    # 거시 파생
    if "US10Y" in df.columns:
        df["US10Y_ma5"] = df["US10Y"].rolling(5).mean()
    if "VIX" in df.columns:
        df["vix_regime"] = (df["VIX"] > 25).astype(int)
    if "DXY" in df.columns:
        df["dxy_mom5"] = df["DXY"].pct_change(5)

    # 래그
    for lag in [1, 3, 5]:
        df[f"lag_{lag}"] = c.shift(lag)

    # 캘린더
    df["month_end"]   = df.index.is_month_end.astype(int)
    df["day_of_week"] = df.index.dayofweek

    # 타겟
    df["y_reg"] = c.shift(-HORIZON)
    df.dropna(inplace=True)

    features = [col for col in df.columns if col not in ["y_reg", "USDKRW"]]
    print(f"  피처 수: {len(features)}")
    return df, features

# ── 시퀀스 생성 ──
def make_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# ── 경량 LSTM ──
def build_light_lstm(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x   = LSTM(64, return_sequences=True)(inp)   # 256→64
    x   = Dropout(0.2)(x)
    x   = LSTM(32)(x)                             # 128→32
    x   = Dropout(0.1)(x)
    x   = Dense(16, activation="relu")(x)
    out = Dense(1)(x)
    model = Model(inp, out, name="LightLSTM")
    model.compile(optimizer=Adam(1e-3), loss=Huber())
    return model

# ── 경량 CNN ──
def build_light_cnn(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x   = Conv1D(32, 3, padding="causal", activation="relu")(inp)  # 128→32
    x   = BatchNormalization()(x)
    x   = Conv1D(32, 3, padding="causal", activation="relu")(x)
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(16, activation="relu")(x)
    out = Dense(1)(x)
    model = Model(inp, out, name="LightCNN")
    model.compile(optimizer=Adam(1e-3), loss=Huber())
    return model

# ── 평가 ──
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
        print(f"  [{label}] RMSE={rmse:.2f} MAE={mae:.2f} MAPE={mape:.2f}% DA={da:.1f}% SR={sr:.3f}")
    return {"RMSE":rmse,"MAE":mae,"MAPE(%)":mape,"DA(%)":da,"Sharpe":sr}

# ── MAIN ──
def main():
    print("="*50)
    print("  경량 USD/KRW 예측 파이프라인")
    print("="*50)

    # 데이터
    df_raw  = collect_data()
    df, features = make_features(df_raw.copy())

    # 분할
    train = df[df.index <= "2024-06-30"]
    val   = df[(df.index > "2024-06-30") & (df.index <= "2025-06-30")]
    test  = df[df.index > "2025-06-30"]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train[features].fillna(0))
    X_val   = scaler_X.transform(val[features].fillna(0))
    X_test  = scaler_X.transform(test[features].fillna(0))
    y_train = scaler_y.fit_transform(train[["y_reg"]])
    y_val   = scaler_y.transform(val[["y_reg"]])
    y_test  = scaler_y.transform(test[["y_reg"]])

    # 스케일러 저장
    with open(f"{OUTPUT_DIR}/scaler_X.pkl","wb") as f: pickle.dump(scaler_X,f)
    with open(f"{OUTPUT_DIR}/scaler_y.pkl","wb") as f: pickle.dump(scaler_y,f)
    with open(f"{OUTPUT_DIR}/feature_list.json","w") as f:
        json.dump(features, f, ensure_ascii=False)

    print(f"  Train:{len(train)} Val:{len(val)} Test:{len(test)}")

    # 시퀀스
    Xtr,ytr = make_sequences(X_train, y_train)
    Xva,yva = make_sequences(X_val,   y_val)
    Xte,yte = make_sequences(X_test,  y_test)
    n_feat  = X_train.shape[1]

    cb = [EarlyStopping(patience=10, restore_best_weights=True),
          ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)]

    all_metrics = {}
    val_preds   = {}

    # LSTM
    print("\n[LSTM 학습]")
    lstm = build_light_lstm(SEQ_LEN, n_feat)
    lstm.fit(Xtr,ytr,validation_data=(Xva,yva),
             epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=cb,verbose=0)
    p = lstm.predict(Xva,verbose=0)
    val_preds["LSTM"] = p.ravel()
    all_metrics["LSTM"] = metrics(
        scaler_y.inverse_transform(yva),
        scaler_y.inverse_transform(p), "LSTM")
    lstm.save(f"{OUTPUT_DIR}/model_lstm.keras")
    print(f"  저장 크기: {os.path.getsize(OUTPUT_DIR+'/model_lstm.keras')/1e6:.1f}MB")

    # CNN
    print("\n[CNN 학습]")
    cnn = build_light_cnn(SEQ_LEN, n_feat)
    cnn.fit(Xtr,ytr,validation_data=(Xva,yva),
            epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=cb,verbose=0)
    p = cnn.predict(Xva,verbose=0)
    val_preds["CNN"] = p.ravel()
    all_metrics["CNN"] = metrics(
        scaler_y.inverse_transform(yva),
        scaler_y.inverse_transform(p), "CNN")
    cnn.save(f"{OUTPUT_DIR}/model_cnn.keras")
    print(f"  저장 크기: {os.path.getsize(OUTPUT_DIR+'/model_cnn.keras')/1e6:.1f}MB")

    # LightGBM
    print("\n[LightGBM 학습]")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, verbose=-1)
    lgb_model.fit(X_train, y_train.ravel(),
                  eval_set=[(X_val, y_val.ravel())],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
    p_lgb = lgb_model.predict(X_val)
    val_preds["LGB"] = p_lgb
    all_metrics["LGB"] = metrics(
        scaler_y.inverse_transform(y_val),
        scaler_y.inverse_transform(p_lgb.reshape(-1,1)), "LGB")
    with open(f"{OUTPUT_DIR}/model_lgb.pkl","wb") as f:
        pickle.dump(lgb_model, f)
    print(f"  저장 크기: {os.path.getsize(OUTPUT_DIR+'/model_lgb.pkl')/1e6:.1f}MB")

    # 앙상블
    print("\n[앙상블]")
    min_len = min(len(v) for v in val_preds.values())
    aligned = {k: v[-min_len:] for k,v in val_preds.items()}
    yva_a   = yva[-min_len:]
    X_meta  = np.column_stack(list(aligned.values()))
    ridge   = Ridge(alpha=1.0)
    ridge.fit(X_meta, yva_a.ravel())
    ens_p   = ridge.predict(X_meta)
    all_metrics["Ensemble"] = metrics(
        scaler_y.inverse_transform(yva_a),
        scaler_y.inverse_transform(ens_p.reshape(-1,1)), "Ensemble★")
    with open(f"{OUTPUT_DIR}/model_ensemble.pkl","wb") as f:
        pickle.dump(ridge, f)

    # 성능표 저장
    perf_df = pd.DataFrame(all_metrics).T
    perf_df.to_csv(f"{OUTPUT_DIR}/performance_table.csv")
    print("\n" + perf_df.to_string())

    # 오늘 예측 — 모델별 전부 저장
    print("\n[오늘 예측]")
    last_price = df["USDKRW"].iloc[-1]
    forecasts_by_model = {}

    if len(X_test) >= SEQ_LEN:
        last_seq = X_test[-SEQ_LEN:].reshape(1, SEQ_LEN, n_feat)

        # LSTM
        p_lstm = scaler_y.inverse_transform(
            lstm.predict(last_seq, verbose=0))[0][0]
        forecasts_by_model["LSTM"] = float(p_lstm)

        # CNN
        p_cnn = scaler_y.inverse_transform(
            cnn.predict(last_seq, verbose=0))[0][0]
        forecasts_by_model["CNN"] = float(p_cnn)

        # LGB
        p_lgb_today = scaler_y.inverse_transform(
            lgb_model.predict(X_test[-1:]).reshape(-1,1))[0][0]
        forecasts_by_model["LGB"] = float(p_lgb_today)

        # 앙상블
        meta_input = np.array([[
            (p_lstm - last_price)/last_price,
            (p_cnn  - last_price)/last_price,
            (p_lgb_today - last_price)/last_price,
        ]])
        p_ens = float(np.mean([p_lstm, p_cnn, p_lgb_today]))
        forecasts_by_model["Ensemble"] = p_ens

    best_pred = forecasts_by_model.get("Ensemble", last_price)
    forecast_out = {
        "date_today":    str(datetime.date.today()),
        "last_close":    float(last_price),
        "D+1_forecast":  float(best_pred),
        "direction":     "상승" if best_pred > last_price else "하락",
        "change_pct":    float((best_pred-last_price)/last_price*100),
        "models":        forecasts_by_model,
    }
    with open(f"{OUTPUT_DIR}/forecast_today.json","w") as f:
        json.dump(forecast_out, f, ensure_ascii=False, indent=2)

    print(f"  오늘 예측:")
    for m, v in forecasts_by_model.items():
        arrow = "↑" if v > last_price else "↓"
        print(f"    {m}: {v:,.2f}원 {arrow}")

    print(f"\n✅ 완료! outputs/ 폴더 확인하세요")
    return all_metrics, forecasts_by_model

if __name__ == "__main__":
    main()
