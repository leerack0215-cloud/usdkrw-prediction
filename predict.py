"""
predict.py — USD/KRW v3 예측 파이프라인
실행: python predict.py

흐름:
  1. 모델 파일 확인
  2. 최신 데이터 수집 + 강화 피처
  3. LGB 전 호라이즌 예측
  4. ARIMAX D+1 예측 (모델 있을 때)
  5. DL D+1 예측 (LSTM + BiGRU)
  6. 4모델 Ridge 앙상블
  7. forecast_today.json 저장
"""

import os, sys, json, pickle, datetime, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
except ImportError:
    HAS_TF = False

from utils import (
    collect_data, make_features, make_sequences,
    get_arimax_exog, ARIMAX_EXOG,
    HORIZONS, HORIZON_LABELS, SEQ_LEN,
    OUTPUT_DIR, MODELS_DIR, CLIP_BOUNDS,
    now_kst,
)


# ════════════════════════════════════════════════════════
# 모델 로더
# ════════════════════════════════════════════════════════

def load_models() -> dict:
    """저장된 모든 모델 로드"""
    models = {
        "lgb":      {},
        "dl":       {},
        "ensemble": None,
        "arimax":   None,
    }

    # LGB (전 호라이즌)
    for h in HORIZONS:
        path = f"{MODELS_DIR}/lgb_h{h}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                models["lgb"][h] = pickle.load(f)
        else:
            print(f"  ⚠ LGB D+{h}: 없음")

    # DL (D+1)
    if HAS_TF:
        for name in ["lstm", "bigru"]:
            path = f"{MODELS_DIR}/{name}_h1.keras"
            if os.path.exists(path):
                try:
                    models["dl"][name] = tf.keras.models.load_model(
                        path, compile=False
                    )
                except Exception as e:
                    print(f"  ⚠ {name.upper()} 로드 실패: {e}")

    # Ridge 앙상블
    path = f"{MODELS_DIR}/ensemble_h1.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            models["ensemble"] = pickle.load(f)

    # ARIMAX
    path = f"{MODELS_DIR}/arimax_h1.pkl"
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                models["arimax"] = pickle.load(f)
            print("  ✓ ARIMAX 로드")
        except Exception as e:
            print(f"  ⚠ ARIMAX 로드 실패: {e}")

    print(
        f"  LGB: {len(models['lgb'])}개 | "
        f"DL: {len(models['dl'])}개 | "
        f"ARIMAX: {'있음' if models['arimax'] else '없음'} | "
        f"앙상블: {'있음' if models['ensemble'] else '없음'}"
    )
    return models


# ════════════════════════════════════════════════════════
# 예측 실행
# ════════════════════════════════════════════════════════

def predict_today(
    df: pd.DataFrame,
    features: list,
    scaler,
    models: dict,
    meta_info: dict,
) -> dict:
    """
    최신 데이터로 다중 호라이즌 예측
    log_return 형태로 저장 (dashboard에서 실시간 cur_price에 곱함)
    """
    last_price = float(df["USDKRW"].iloc[-1])
    last_date  = str(df.index[-1].date())

    X_scaled  = scaler.transform(df[features].fillna(0))
    last_flat = X_scaled[-1:].reshape(1, -1)

    results = {
        "date_today":  now_kst().strftime("%Y-%m-%d"),
        "last_close":  last_price,
        "last_date":   last_date,
        "trained_at":  meta_info.get("trained_at", "N/A"),
        "n_models":    meta_info.get("n_models", 3),
        "forecasts":   {},
        "log_returns": {},       # dashboard 실시간 연동용
        "models":      {v: {} for v in HORIZON_LABELS.values()},
    }

    clip1 = CLIP_BOUNDS[1]

    # ── LGB 예측 (전 호라이즌) ──────────────────────
    lgb_lr_d1 = None
    for h in HORIZONS:
        if h not in models["lgb"]:
            continue
        clip = CLIP_BOUNDS[h]
        lr   = float(np.clip(
            models["lgb"][h].predict(last_flat)[0], -clip, clip
        ))
        pred = round(last_price * np.exp(lr), 2)
        hlabel = HORIZON_LABELS[h]
        results["models"][hlabel]["LGB"] = pred
        results["log_returns"][hlabel]   = lr    # log_return 저장
        if h == 1:
            lgb_lr_d1 = lr

    # ── ARIMAX D+1 ──────────────────────────────────
    arimax_lr_d1 = None
    if models["arimax"] is not None:
        try:
            exog_last   = get_arimax_exog(df).iloc[[-1]].values
            raw         = models["arimax"].predict(
                n_periods=1, exogenous=exog_last
            )
            # pmdarima 버전에 따라 (array,) 또는 array 반환 — 안전하게 처리
            arimax_pred  = raw[0] if isinstance(raw, tuple) else raw
            arimax_lr_d1 = float(np.clip(arimax_pred[0], -clip1, clip1))
            pred = round(last_price * np.exp(arimax_lr_d1), 2)
            results["models"]["D+1"]["ARIMAX"] = pred
        except Exception as e:
            print(f"  ⚠ ARIMAX 예측 실패: {e}")

    # ── DL 예측 (D+1) ───────────────────────────────
    dl_lr_d1 = {}
    if HAS_TF and len(models["dl"]) > 0:
        last_seq = X_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, -1).astype(np.float32)
        for name, model in models["dl"].items():
            try:
                lr   = float(np.clip(
                    model.predict(last_seq, verbose=0)[0][0], -clip1, clip1
                ))
                pred = round(last_price * np.exp(lr), 2)
                dl_lr_d1[name] = lr
                results["models"]["D+1"][name.upper()] = pred
            except Exception as e:
                print(f"  ⚠ {name.upper()} 예측 실패: {e}")

    # ── Ridge 앙상블 D+1 ─────────────────────────────
    if models["ensemble"] is not None and lgb_lr_d1 is not None:
        try:
            n_models = meta_info.get("n_models", 3)
            # 앙상블 입력 순서: lgb, lstm, bigru, [arimax]
            meta_row = [
                lgb_lr_d1,
                dl_lr_d1.get("lstm",  lgb_lr_d1),
                dl_lr_d1.get("bigru", lgb_lr_d1),
            ]
            if n_models >= 4 and arimax_lr_d1 is not None:
                meta_row.append(arimax_lr_d1)
            elif n_models >= 4:
                meta_row.append(lgb_lr_d1)  # ARIMAX 없으면 LGB 대체

            ens_lr = float(np.clip(
                models["ensemble"].predict(np.array([meta_row]))[0],
                -clip1, clip1,
            ))
            pred = round(last_price * np.exp(ens_lr), 2)
            results["models"]["D+1"]["★Ensemble"] = pred
            # 앙상블 log_return으로 D+1 대표값 덮어씀
            results["log_returns"]["D+1"] = ens_lr
        except Exception as e:
            print(f"  ⚠ 앙상블 예측 실패: {e}")

    # ── 최종 예측값 결정 (is not None 가드) ─────────
    for h in HORIZONS:
        hlabel = HORIZON_LABELS[h]
        m      = results["models"].get(hlabel, {})
        lr     = results["log_returns"].get(hlabel, 0.0)

        if h == 1:
            ens = m.get("★Ensemble")
            lgb = m.get("LGB")
            best = ens if ens is not None else (
                lgb if lgb is not None else last_price
            )
            best_lr = results["log_returns"].get("D+1", lgb_lr_d1 or 0.0)
        else:
            lgb = m.get("LGB")
            best = lgb if lgb is not None else last_price
            best_lr = lr

        pct = round((best / last_price - 1) * 100, 3)
        results["forecasts"][hlabel] = {
            "price":      best,
            "log_return": best_lr,
            "change_pct": pct,
            "direction":  "상승 ↑" if pct > 0 else (
                "하락 ↓" if pct < 0 else "보합 —"
            ),
        }

    # 편의 키
    d1 = results["forecasts"]["D+1"]
    results["D+1_forecast"] = d1["price"]
    results["direction"]    = "상승" if d1["price"] > last_price else "하락"
    results["change_pct"]   = d1["change_pct"]

    return results


# ════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════

def main():
    print("=" * 58)
    print("  USD/KRW v3 실시간 예측")
    print(f"  {now_kst().strftime('%Y-%m-%d %H:%M:%S')} KST")
    print("=" * 58)

    # 필수 파일 확인
    required = [
        f"{OUTPUT_DIR}/scaler_X.pkl",
        f"{OUTPUT_DIR}/feature_list.json",
        f"{MODELS_DIR}/lgb_h1.pkl",
    ]
    for path in required:
        if not os.path.exists(path):
            print(f"❌ {path} 없음 → train.py를 먼저 실행하세요!")
            sys.exit(1)

    # 스케일러 + 피처 로드
    with open(f"{OUTPUT_DIR}/scaler_X.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{OUTPUT_DIR}/feature_list.json", encoding="utf-8") as f:
        features = json.load(f)

    # 메타 정보
    meta_info = {}
    meta_path = f"{OUTPUT_DIR}/meta_info.json"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta_info = json.load(f)

    # 데이터 수집 (경량 일봉)
    print("\n[1] 최신 데이터 수집...")
    df = collect_data(start="2018-01-01")
    print("[2] 피처 엔지니어링...")
    df, _ = make_features(df, add_targets=False)

    # 피처 컬럼 맞추기
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    # 모델 로드
    print("[3] 모델 로드...")
    models = load_models()

    # 예측
    print("[4] 예측 실행...")
    results = predict_today(df, features, scaler, models, meta_info)

    # 저장
    out_path = f"{OUTPUT_DIR}/forecast_today.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 결과 출력
    lp = results["last_close"]
    print(f"\n  현재(일봉): ₩{lp:,.2f}  ({results['last_date']})")
    print(f"  학습 기준:  {results['trained_at']}")
    print(f"\n  {'호라이즌':7s}  {'예측':>12s}  {'변화율':>8s}  방향")
    print(f"  {'─'*46}")
    for lbl, data in results["forecasts"].items():
        print(
            f"  {lbl:7s}  ₩{data['price']:>11,.2f}  "
            f"{data['change_pct']:>+7.2f}%  {data['direction']}"
        )

    print(f"\n  📊 D+1 모델별:")
    for name, pred in results["models"].get("D+1", {}).items():
        print(f"    {name:14s}: ₩{pred:,.2f}")

    print(f"\n✅ {out_path} 저장 완료!")


if __name__ == "__main__":
    main()
