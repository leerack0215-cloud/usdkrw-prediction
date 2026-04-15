"""
train.py — USD/KRW v3 통합 학습 파이프라인
실행: python train.py

Phase 1: 멀티스케일 데이터 수집
Phase 2: 강화 피처 엔지니어링
Phase 3: 비율 기반 분할 (70/15/15)
Phase 4: ARIMAX 학습 (외생변수 기반 선형 시계열)
Phase 5: LGB 학습 (호라이즌별)
Phase 6: DL 학습 (LSTM + BiGRU, D+1)
Phase 7: 4모델 Ridge 메타 앙상블
Phase 8: 성능 평가 + 저장
Phase 9: predict.py 자동 호출
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
    collect_multiscale, make_features, make_sequences,
    compute_metrics, get_arimax_exog, ARIMAX_EXOG,
    HORIZONS, SEQ_LEN, OUTPUT_DIR, MODELS_DIR, CLIP_BOUNDS,
    now_kst,
)

# ════════════════════════════════════════════════════════
# DL 모델 정의
# ════════════════════════════════════════════════════════

def build_lstm(seq_len: int, n_feat: int) -> Model:
    inp = Input(shape=(seq_len, n_feat))
    x   = LSTM(128, return_sequences=True)(inp)
    x   = Dropout(0.20)(x)
    x   = LSTM(64, return_sequences=True)(x)
    x   = Dropout(0.15)(x)
    x   = LSTM(32)(x)
    x   = Dropout(0.10)(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m   = Model(inp, out, name="LSTM")
    m.compile(optimizer=Adam(1e-3, clipnorm=1.0), loss=Huber(0.01))
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
    m.compile(optimizer=Adam(1e-3, clipnorm=1.0), loss=Huber(0.01))
    return m


CALLBACKS = [
    EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6, verbose=0),
]


# ════════════════════════════════════════════════════════
# ARIMAX 학습
# ════════════════════════════════════════════════════════

def train_arimax(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    y_val: pd.Series,
    exog_val: pd.DataFrame,
) -> tuple:
    """
    ARIMAX(p,d,q) 학습
    pmdarima auto_arima로 최적 차수 자동 선택
    반환: (model, val_log_returns_pred)
    """
    try:
        import pmdarima as pm
        print("  auto_arima 차수 탐색 중...")
        model = pm.auto_arima(
            y_train,
            exogenous   = exog_train.values,
            start_p=1, start_q=1,
            max_p=3,   max_q=3,
            d=None,            # ADF 검정으로 자동
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            n_fits=20,
        )
        order = model.order
        print(f"  최적 차수: ARIMA{order}")

        # Val 예측
        # return_conf_int=False → 배열 1개만 반환 (tuple 아님)
        n_val = len(y_val)
        raw   = model.predict(
            n_periods  = n_val,
            exogenous  = exog_val.values,
            return_conf_int=False,
        )
        # pmdarima 버전에 따라 (array,) 또는 array 반환 — 안전하게 처리
        preds = raw[0] if isinstance(raw, tuple) else raw
        return model, np.array(preds)

    except ImportError:
        print("  ⚠ pmdarima 없음 — ARIMAX 스킵 (pip install pmdarima)")
        return None, None
    except Exception as e:
        print(f"  ⚠ ARIMAX 실패: {e}")
        return None, None


# ════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════

def main():
    t0 = now_kst()
    print("=" * 65)
    print("  USD/KRW v3 — 멀티스케일 + ARIMAX + 강화피처 학습")
    print(f"  시작: {t0.strftime('%Y-%m-%d %H:%M:%S')} KST")
    print("=" * 65)

    # ── Phase 1: 멀티스케일 데이터 수집 ─────────────
    print("\n[Phase 1] 멀티스케일 데이터 수집...")
    df_raw = collect_multiscale(start_daily="2015-01-01")
    print(
        f"  완료: {len(df_raw)}행 × {len(df_raw.columns)}컬럼  "
        f"({df_raw.index[0].date()} ~ {df_raw.index[-1].date()})"
    )

    # ── Phase 2: 강화 피처 엔지니어링 ───────────────
    print("\n[Phase 2] 강화 피처 엔지니어링...")
    df, features = make_features(df_raw, add_targets=True)
    print(f"  피처: {len(features)}개  /  유효 행: {len(df)}행")

    # 피처 목록 저장
    with open(f"{OUTPUT_DIR}/feature_list.json", "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    # ── Phase 3: 비율 기반 분할 (70/15/15) ──────────
    print("\n[Phase 3] 시계열 분할 (70/15/15)...")
    n      = len(df)
    i_val  = int(n * 0.70)
    i_test = int(n * 0.85)
    train  = df.iloc[:i_val]
    val    = df.iloc[i_val:i_test]
    test   = df.iloc[i_test:]

    if len(train) < 200 or len(val) < 30:
        print("❌ 데이터 부족")
        sys.exit(1)

    print(f"  Train: {len(train)}행  ({train.index[0].date()} ~ {train.index[-1].date()})")
    print(f"  Val:   {len(val)}행  ({val.index[0].date()} ~ {val.index[-1].date()})")
    print(f"  Test:  {len(test)}행  ({test.index[0].date()} ~ {test.index[-1].date()})")

    # RobustScaler — Train에만 fit (Lookahead Bias 차단)
    scaler  = RobustScaler()
    X_train = scaler.fit_transform(train[features].fillna(0))
    X_val   = scaler.transform(val[features].fillna(0))
    X_test  = scaler.transform(test[features].fillna(0)) if len(test) > 0 else None

    price_train = train["USDKRW"].values
    price_val   = val["USDKRW"].values

    with open(f"{OUTPUT_DIR}/scaler_X.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("\n  ✅ scaler_X.pkl 저장")

    all_metrics = {}

    # ── Phase 4: ARIMAX 학습 ─────────────────────────
    print("\n[Phase 4] ARIMAX 학습...")
    arimax_model = None
    arimax_val_lr = None

    y_train_d1_lr = train["y_h1"].values  # log-return D+1
    y_val_d1_lr   = val["y_h1"].values
    exog_train    = get_arimax_exog(train)
    exog_val      = get_arimax_exog(val)

    arimax_model, arimax_val_preds = train_arimax(
        y_train_d1_lr, exog_train,
        y_val_d1_lr,   exog_val,
    )

    if arimax_model is not None and arimax_val_preds is not None:
        clip1 = CLIP_BOUNDS[1]
        arimax_val_lr = np.clip(arimax_val_preds, -clip1, clip1)
        y_true_p = price_val * np.exp(y_val_d1_lr)
        y_pred_p = price_val * np.exp(arimax_val_lr)
        all_metrics["ARIMAX_D+1"] = compute_metrics(
            y_true_p, y_pred_p, "ARIMAX D+1"
        )
        path = f"{MODELS_DIR}/arimax_h1.pkl"
        with open(path, "wb") as f:
            pickle.dump(arimax_model, f)
        print(f"  저장: {path}")
    else:
        print("  ARIMAX 스킵 — 앙상블에서 제외")

    # ── Phase 5: LGB 학습 (호라이즌별) ──────────────
    print("\n[Phase 5] LGB 학습 (전 호라이즌)...")
    lgb_models = {}

    for h in HORIZONS:
        print(f"\n  ─ D+{h} LGB ─")
        ytr  = train[f"y_h{h}"].values
        yva  = val[f"y_h{h}"].values
        clip = CLIP_BOUNDS[h]

        model = lgb.LGBMRegressor(
            n_estimators     = 3000,
            learning_rate    = 0.02,
            num_leaves       = 63,
            max_depth        = 6,
            min_child_samples= 20,
            reg_alpha        = 0.1,
            reg_lambda       = 0.2,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = 42,
            verbose          = -1,
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
        all_metrics[f"LGB_D+{h}"] = compute_metrics(
            y_true_p, y_pred_p, f"LGB D+{h}"
        )

        lgb_models[h] = model
        path = f"{MODELS_DIR}/lgb_h{h}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"    저장: {path} ({os.path.getsize(path)/1e6:.2f}MB)")

    # ── Phase 6: DL 학습 (LSTM + BiGRU, D+1) ────────
    print("\n[Phase 6] DL 학습 (LSTM + BiGRU)...")
    ytr_d1 = train["y_h1"].values
    yva_d1 = val["y_h1"].values
    clip1  = CLIP_BOUNDS[1]

    Xtr_seq, ytr_seq = make_sequences(X_train, ytr_d1)
    Xva_seq, yva_seq = make_sequences(X_val,   yva_d1)
    price_val_seq    = price_val[SEQ_LEN:]
    n_feat           = X_train.shape[1]

    dl_val_lr = {}

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

        p_lr = np.clip(
            model.predict(Xva_seq, verbose=0).ravel(), -clip1, clip1
        )
        dl_val_lr[name] = p_lr

        y_true_p = price_val_seq * np.exp(yva_seq)
        y_pred_p = price_val_seq * np.exp(p_lr)
        all_metrics[f"{name.upper()}_D+1"] = compute_metrics(
            y_true_p, y_pred_p, f"{name.upper()} D+1"
        )

        path = f"{MODELS_DIR}/{name}_h1.keras"
        model.save(path)
        print(f"    저장: {path} ({os.path.getsize(path)/1e6:.2f}MB)")

    # ── Phase 7: 4모델 Ridge 메타 앙상블 ─────────────
    print("\n[Phase 7] 4모델 Ridge 앙상블 (ARIMAX+LGB+LSTM+BiGRU)...")

    lgb_d1_lr = np.clip(
        lgb_models[1].predict(X_val[SEQ_LEN:]), -clip1, clip1
    )

    # 앙상블 입력 구성 (ARIMAX 있으면 포함)
    meta_inputs = [lgb_d1_lr,
                   dl_val_lr.get("lstm",  lgb_d1_lr),
                   dl_val_lr.get("bigru", lgb_d1_lr)]

    if arimax_val_lr is not None:
        # ARIMAX는 전체 val 구간 → seq_len 이후 슬라이싱
        arimax_seq = arimax_val_lr[SEQ_LEN:] \
            if len(arimax_val_lr) > SEQ_LEN else arimax_val_lr
        # 길이 맞추기
        min_len = min(len(lgb_d1_lr), len(arimax_seq))
        meta_inputs_aligned = [x[-min_len:] for x in meta_inputs]
        meta_inputs_aligned.append(arimax_seq[-min_len:])
        X_meta  = np.column_stack(meta_inputs_aligned)
        y_meta  = yva_seq[-min_len:]
        pv_meta = price_val_seq[-min_len:]
        n_models = 4
    else:
        min_len = len(lgb_d1_lr)
        X_meta  = np.column_stack(meta_inputs)
        y_meta  = yva_seq
        pv_meta = price_val_seq
        n_models = 3

    ridge = Ridge(alpha=0.1)
    ridge.fit(X_meta, y_meta)

    p_ens    = np.clip(ridge.predict(X_meta), -clip1, clip1)
    y_true_p = pv_meta * np.exp(y_meta)
    y_pred_p = pv_meta * np.exp(p_ens)
    all_metrics[f"★Ensemble_D+1({n_models}모델)"] = compute_metrics(
        y_true_p, y_pred_p, f"★Ensemble ({n_models}모델)"
    )

    # 메타 정보 저장 (predict.py에서 ARIMAX 유무 판단용)
    meta_info = {
        "n_models":    n_models,
        "has_arimax":  arimax_model is not None,
        "arimax_exog": ARIMAX_EXOG,
        "trained_at":  now_kst().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{OUTPUT_DIR}/meta_info.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    path = f"{MODELS_DIR}/ensemble_h1.pkl"
    with open(path, "wb") as f:
        pickle.dump(ridge, f)
    print(f"  저장: {path}")

    # ── Phase 8: 성능 평가 ───────────────────────────
    print("\n" + "=" * 65)
    print("  모델 성능 비교 (Validation Set)")
    print("=" * 65)
    perf_df = pd.DataFrame(all_metrics).T
    print(perf_df.to_string())
    perf_df.to_csv(f"{OUTPUT_DIR}/performance_table.csv")

    # 파일 크기 확인
    print("\n[파일 크기]")
    total = 0.0
    for root, _, files in os.walk(OUTPUT_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size  = os.path.getsize(fpath) / 1e6
            total += size
            flag  = "⚠️" if size >= 95 else "✅"
            print(f"  {flag} {fpath}: {size:.2f}MB")
    print(f"  총합: {total:.2f}MB")

    # ── Phase 9: 예측 자동 실행 ──────────────────────
    print("\n[Phase 9] predict.py 실행...")
    os.system(f"{sys.executable} predict.py")

    elapsed = int((now_kst() - t0).total_seconds())
    print(f"\n{'='*65}")
    print(f"✅ 완료!  소요: {elapsed//60}분 {elapsed%60}초")
    print(f"   다음: git add outputs/ && git push")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
