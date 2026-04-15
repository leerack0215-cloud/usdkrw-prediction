"""
predict.py — 경량 일일 예측 (학습된 모델 로드 → JSON 저장)
실행: python predict.py

흐름:
  1. 모델 파일 존재 확인 (없으면 sys.exit)
  2. 최신 데이터 수집 + 피처 엔지니어링 (추론 모드)
  3. LGB 전 호라이즌 예측
  4. DL D+1 예측 (LSTM + BiGRU)
  5. Ridge 앙상블 D+1
  6. forecast_today.json 저장
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
    collect_data, make_features,
    make_sequences, compute_metrics,
    HORIZONS, HORIZON_LABELS, SEQ_LEN,
    OUTPUT_DIR, MODELS_DIR, CLIP_BOUNDS,
)


# ════════════════════════════════════════════════════════
# 모델 로더
# ════════════════════════════════════════════════════════

def load_all_models() -> dict:
    """저장된 모델 전체 로드. 없으면 경고 후 스킵."""
    models = {"lgb": {}, "dl": {}, "ensemble": None}

    for h in HORIZONS:
        path = f"{MODELS_DIR}/lgb_h{h}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                models["lgb"][h] = pickle.load(f)
        else:
            print(f"  ⚠ LGB D+{h}: {path} 없음")

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

    ens_path = f"{MODELS_DIR}/ensemble_h1.pkl"
    if os.path.exists(ens_path):
        with open(ens_path, "rb") as f:
            models["ensemble"] = pickle.load(f)

    print(
        f"  LGB: {len(models['lgb'])}개 | "
        f"DL: {len(models['dl'])}개 | "
        f"앙상블: {'있음' if models['ensemble'] else '없음'}"
    )
    return models


# ════════════════════════════════════════════════════════
# 예측 실행
# ════════════════════════════════════════════════════════

def predict_today(df, features, scaler, models) -> dict:
    """
    최신 데이터로 다중 호라이즌 예측

    반환 JSON 구조:
    {
      "date_today":  "2026-04-15",
      "last_close":  1482.18,
      "last_date":   "2026-04-14",
      "forecasts": {
        "D+1": {"price": 1485.0, "change_pct": +0.19, "direction": "상승 ↑"},
        ...
      },
      "models": {
        "D+1": {"LGB": 1485.0, "LSTM": 1483.0, "BIGRU": 1486.0, "★Ensemble": 1484.5},
        ...
      },
      "D+1_forecast": 1484.5,
      "direction":    "상승",
      "change_pct":   +0.19
    }
    """
    last_price = float(df["USDKRW"].iloc[-1])
    last_date  = str(df.index[-1].date())

    X_scaled  = scaler.transform(df[features].fillna(0))
    last_flat = X_scaled[-1:].reshape(1, -1)

    results = {
        "date_today": str(datetime.date.today()),
        "last_close": last_price,
        "last_date":  last_date,
        "forecasts":  {},
        "models":     {v: {} for v in HORIZON_LABELS.values()},
    }

    clip1 = CLIP_BOUNDS[1]

    # ── LGB 예측 (전 호라이즌) ──────────────────────
    lgb_lr_d1 = None
    for h in HORIZONS:
        if h not in models["lgb"]:
            continue
        clip = CLIP_BOUNDS[h]
        lr   = float(np.clip(models["lgb"][h].predict(last_flat)[0], -clip, clip))
        pred = round(last_price * np.exp(lr), 2)
        results["models"][HORIZON_LABELS[h]]["LGB"] = pred
        if h == 1:
            lgb_lr_d1 = lr

    # ── DL 예측 (D+1 전용) ──────────────────────────
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

    # ── D+1 앙상블 ──────────────────────────────────
    ens_d1_price = None
    if (
        models["ensemble"] is not None
        and lgb_lr_d1 is not None
        and len(dl_lr_d1) == 2
    ):
        try:
            meta = np.array([[
                lgb_lr_d1,
                dl_lr_d1.get("lstm",  lgb_lr_d1),
                dl_lr_d1.get("bigru", lgb_lr_d1),
            ]])
            ens_lr       = float(np.clip(models["ensemble"].predict(meta)[0], -clip1, clip1))
            ens_d1_price = round(last_price * np.exp(ens_lr), 2)
            results["models"]["D+1"]["★Ensemble"] = ens_d1_price
        except Exception as e:
            print(f"  ⚠ 앙상블 예측 실패: {e}")

    # ── 최종 예측값 결정 ─────────────────────────────
    # is not None 체크 — 예측값 0.0 이 falsy 취급되는 버그 방지
    for h in HORIZONS:
        hlabel = HORIZON_LABELS[h]
        m      = results["models"].get(hlabel, {})

        if h == 1:
            ens  = m.get("★Ensemble")
            lgb  = m.get("LGB")
            best = ens if ens is not None else (lgb if lgb is not None else last_price)
        else:
            lgb  = m.get("LGB")
            best = lgb if lgb is not None else last_price

        pct = round((best / last_price - 1) * 100, 3)
        results["forecasts"][hlabel] = {
            "price":      best,
            "change_pct": pct,
            "direction":  "상승 ↑" if pct > 0 else ("하락 ↓" if pct < 0 else "보합 —"),
        }

    # 편의 키 (대시보드용)
    d1_price = results["forecasts"]["D+1"]["price"]
    results["D+1_forecast"] = d1_price
    results["direction"]    = "상승" if d1_price > last_price else "하락"
    results["change_pct"]   = results["forecasts"]["D+1"]["change_pct"]
    return results


# ════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  USD/KRW 실시간 예측 업데이트")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # 필수 파일 확인
    for path in [
        f"{OUTPUT_DIR}/scaler_X.pkl",
        f"{OUTPUT_DIR}/feature_list.json",
        f"{MODELS_DIR}/lgb_h1.pkl",
    ]:
        if not os.path.exists(path):
            print(f"❌ {path} 없음 → train.py를 먼저 실행하세요!")
            sys.exit(1)

    # 스케일러 + 피처 로드
    with open(f"{OUTPUT_DIR}/scaler_X.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{OUTPUT_DIR}/feature_list.json", encoding="utf-8") as f:
        features = json.load(f)

    # 데이터 수집 + 피처 엔지니어링 (추론 모드)
    print("\n[1] 최신 데이터 수집...")
    df = collect_data(start="2018-01-01")
    print("[2] 피처 엔지니어링 (추론 모드)...")
    df, _ = make_features(df, add_targets=False)

    # 학습 피처와 컬럼 일치
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    # 모델 로드
    print("[3] 모델 로드...")
    models = load_all_models()

    # 예측
    print("[4] 예측 실행...")
    results = predict_today(df, features, scaler, models)

    # 저장
    out_path = f"{OUTPUT_DIR}/forecast_today.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 결과 출력
    lp = results["last_close"]
    print(f"\n  현재: ₩{lp:,.2f}  ({results['last_date']})")
    print(f"\n  {'호라이즌':7s}  {'예측':>12s}  {'변화율':>8s}  방향")
    print(f"  {'─'*46}")
    for lbl, data in results["forecasts"].items():
        print(
            f"  {lbl:7s}  ₩{data['price']:>11,.2f}  "
            f"{data['change_pct']:>+7.2f}%  {data['direction']}"
        )

    print(f"\n  📊 D+1 모델별:")
    for model_name, pred in results["models"].get("D+1", {}).items():
        print(f"    {model_name:12s}: ₩{pred:,.2f}")

    print(f"\n✅ {out_path} 저장 완료!")


if __name__ == "__main__":
    main()
