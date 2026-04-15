"""
train.py — USD/KRW 하이브리드 앙상블 학습
실행: python train.py

Phase 1: 데이터 수집
Phase 2: 피처 엔지니어링
Phase 3: 비율 기반 분할 (70 / 15 / 15)
Phase 4: 호라이즌별 LightGBM (D+1~D+22)
Phase 5: D+1 딥러닝 (LSTM + BiGRU)
Phase 6: D+1 메타 앙상블 (Ridge)
Phase 7: 성능 평가 + 저장
Phase 8: predict.py 자동 호출
"""

import os, sys, json, pickle, datetime, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Bidirectional,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

from utils import (
    collect_data, make_features, make_sequences, compute_metrics,
    HORIZONS, SEQ_LEN, OUTPUT_DIR, MODELS_DIR, CLIP_BOUNDS,
)

# ════════════════════════════════════════════════════════
# 모델 정의
# ════════════════════════════════════════════════════════

def build_lstm(seq_len: int, n_feat: int) -> Model:
    inp = Input(shape=(seq_len, n_feat))
    x   = LSTM(128, return_sequences=True)(inp)
    x   = Dropout(0.20)(x)
    x   = LSTM(64,  return_sequences=True)(x)
    x   = Dropout(0.15)(x)
    x   = LSTM(32)(x)
    x   = Dropout(0.10)(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m   = Model(inp, out, name="LSTM")
    m.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss=Huber(delta=0.01),
    )
    return m


def build_bigru(seq_len: int, n_feat: int) -> Model:
    inp = Input(shape=(seq_len, n_feat))
    x   = Bidirectional(GRU(64, return_sequences=True))(inp)
    x   = Dropout(0.20)(x)
    x   = Bidirectional(GRU(32))(x)
    x   = Dropout(0.10)(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m   = Model(inp, out, name="BiGRU")
    m.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss=Huber(delta=0.01),
    )
    return m


CALLBACKS = [
    EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6, verbose=0),
]


# ════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════

def main():
    t0 = datetime.datetime.now()
    print("=" * 62)
    print("  USD/KRW 하이브리드 앙상블 학습")
    print(f"  시작: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    # ── Phase 1: 데이터 수집 ─────────────────────────
    print("\n[Phase 1] 데이터 수집...")
    df_raw = collect_data(start="2015-01-01")
    print(
        f"  완료: {len(df_raw)}행  "
        f"({df_raw.index[0].date()} ~ {df_raw.index[-1].date()})"
    )

    # ── Phase 2: 피처 엔지니어링 ─────────────────────
    print("\n[Phase 2] 피처 엔지니어링...")
    df, features = make_features(df_raw, add_targets=True)
    print(f"  피처: {len(features)}개  /  유효 행: {len(df)}행")

    # ── Phase 3: 비율 기반 분할 (70 / 15 / 15) ───────
    print("\n[Phase 3] 시계열 분할 (70 / 15 / 15)...")
    n      = len(df)
    i_val  = int(n * 0.70)
    i_test = int(n * 0.85)
    train  = df.iloc[:i_val]
    val    = df.iloc[i_val:i_test]
    test   = df.iloc[i_test:]

    if len(train) < 200 or len(val) < 30:
        print("❌ 데이터 부족 — 수집 시작일을 앞당기세요.")
        sys.exit(1)

    print(f"  Train: {len(train)}행  ({train.index[0].date()} ~ {train.index[-1].date()})")
    print(f"  Val:   {len(val)}행  ({val.index[0].date()} ~ {val.index[-1].date()})")
    print(f"  Test:  {len(test)}행  ({test.index[0].date()} ~ {test.index[-1].date()})")

    # RobustScaler — Train에만 fit (Lookahead Bias 차단)
    scaler  = RobustScaler()
    X_train = scaler.fit_transform(train[features].fillna(0))
    X_val   = scaler.transform(val[features].fillna(0))
    X_test  = scaler.transform(test[features].fillna(0)) if len(test) > 0 else np.empty((0, len(features)))

    price_train = train["USDKRW"].values
    price_val   = val["USDKRW"].values
    price_test  = test["USDKRW"].values if len(test) > 0 else np.array([])

    # 스케일러 + 피처 목록 저장
    with open(f"{OUTPUT_DIR}/scaler_X.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{OUTPUT_DIR}/feature_list.json", "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    print("\n  ✅ scaler_X.pkl + feature_list.json 저장")

    # ── Phase 4: 호라이즌별 LightGBM ────────────────
    print("\n[Phase 4] LightGBM 학습 (전 호라이즌)...")
    lgb_models  = {}
    all_metrics = {}

    for h in HORIZONS:
        print(f"\n  ─ D+{h} LGB ─")
        ytr  = train[f"y_h{h}"].values
        yva  = val[f"y_h{h}"].values
        clip = CLIP_BOUNDS[h]

        model = lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.02,
            num_leaves=31, max_depth=5,
            min_child_samples=20,
            reg_alpha=0.1, reg_lambda=0.2,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        model.fit(
            X_train, ytr,
            eval_set=[(X_val, yva)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        p_lr     = np.clip(model.predict(X_val), -clip, clip)
        y_true_p = price_val * np.exp(val[f"y_h{h}"].values)
        y_pred_p = price_val * np.exp(p_lr)
        all_metrics[f"LGB_D+{h}"] = compute_metrics(y_true_p, y_pred_p, f"LGB D+{h}")

        lgb_models[h] = model
        path = f"{MODELS_DIR}/lgb_h{h}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"    저장: {path} ({os.path.getsize(path)/1e6:.2f}MB)")

    # ── Phase 5: D+1 딥러닝 (LSTM + BiGRU) ──────────
    print("\n[Phase 5] D+1 딥러닝 학습...")
    ytr_d1 = train["y_h1"].values
    yva_d1 = val["y_h1"].values
    clip1  = CLIP_BOUNDS[1]

    Xtr_seq, ytr_seq = make_sequences(X_train, ytr_d1)
    Xva_seq, yva_seq = make_sequences(X_val,   yva_d1)
    price_val_seq    = price_val[SEQ_LEN:]
    n_feat           = X_train.shape[1]

    dl_preds_val_lr = {}

    for name, builder in [("lstm", build_lstm), ("bigru", build_bigru)]:
        print(f"\n  ─ {name.upper()} ─")
        model = builder(SEQ_LEN, n_feat)
        model.fit(
            Xtr_seq, ytr_seq,
            validation_data=(Xva_seq, yva_seq),
            epochs=200, batch_size=32,
            callbacks=CALLBACKS, verbose=0,
        )
        stopped = len(model.history.history["loss"])
        print(f"    종료 Epoch: {stopped}")

        p_lr = np.clip(model.predict(Xva_seq, verbose=0).ravel(), -clip1, clip1)
        dl_preds_val_lr[name] = p_lr

        y_true_p = price_val_seq * np.exp(yva_seq)
        y_pred_p = price_val_seq * np.exp(p_lr)
        all_metrics[f"{name.upper()}_D+1"] = compute_metrics(
            y_true_p, y_pred_p, f"{name.upper()} D+1"
        )

        path = f"{MODELS_DIR}/{name}_h1.keras"
        model.save(path)
        print(f"    저장: {path} ({os.path.getsize(path)/1e6:.2f}MB)")

    # ── Phase 6: D+1 Ridge 메타 앙상블 ──────────────
    print("\n[Phase 6] D+1 Ridge 앙상블 학습...")
    lgb_d1_lr = np.clip(
        lgb_models[1].predict(X_val[SEQ_LEN:]), -clip1, clip1
    )
    X_meta = np.column_stack([
        lgb_d1_lr,
        dl_preds_val_lr.get("lstm",  lgb_d1_lr),
        dl_preds_val_lr.get("bigru", lgb_d1_lr),
    ])
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_meta, yva_seq)

    p_ens    = np.clip(ridge.predict(X_meta), -clip1, clip1)
    y_true_p = price_val_seq * np.exp(yva_seq)
    y_pred_p = price_val_seq * np.exp(p_ens)
    all_metrics["★Ensemble_D+1"] = compute_metrics(y_true_p, y_pred_p, "★Ensemble D+1")

    path = f"{MODELS_DIR}/ensemble_h1.pkl"
    with open(path, "wb") as f:
        pickle.dump(ridge, f)
    print(f"  저장: {path}")

    # ── Phase 7: 성능표 출력 + 저장 ─────────────────
    print("\n" + "=" * 62)
    print("  모델 성능 (Validation Set)")
    print("=" * 62)
    perf_df = pd.DataFrame(all_metrics).T
    print(perf_df.to_string())
    perf_df.to_csv(f"{OUTPUT_DIR}/performance_table.csv")
    print(f"\n  ✅ performance_table.csv 저장")

    # 파일 크기 확인
    print("\n[파일 크기 확인]")
    total = 0.0
    for root, _, files in os.walk(OUTPUT_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size  = os.path.getsize(fpath) / 1e6
            total += size
            flag  = "⚠️" if size >= 95 else "✅"
            print(f"  {flag} {fpath}: {size:.2f}MB")
    print(f"  총합: {total:.2f}MB")

    # ── Phase 8: 예측 자동 실행 ──────────────────────
    print("\n[Phase 8] 예측 실행 (predict.py)...")
    os.system(f"{sys.executable} predict.py")

    elapsed = (datetime.datetime.now() - t0).seconds
    print(f"\n{'='*62}")
    print(f"✅ 완료!  소요: {elapsed//60}분 {elapsed%60}초")
    print(f"   다음 단계: git add outputs/ && git push")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
