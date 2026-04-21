"""
collect_residuals.py — 잔차 데이터 수집 스크립트
EC2 cron: 매일 KST 06:10 (UTC 21:10) 자동 실행

동작:
  1. S3에서 d1_history.json 다운로드
  2. 24시간 전 예측 스냅샷 추출
  3. yfinance 1분봉으로 실제가격 매칭
  4. 잔차(오차) 계산 후 residual_data.csv 누적
  5. S3에 업로드

EC2 cron 등록:
  crontab -e
  10 21 * * * python3 /home/ec2-user/usdkrw/collect_residuals.py >> /home/ec2-user/usdkrw/residual.log 2>&1
"""

import os
import json
import datetime
import boto3
import pandas as pd
import yfinance as yf

# ── 설정 ──────────────────────────────────────────────
KST        = datetime.timezone(datetime.timedelta(hours=9))
S3_BUCKET  = os.environ.get("S3_BUCKET", "usdkrw-prediction-jason")
WORK_DIR   = os.path.expanduser("~/usdkrw")
HIST_PATH  = f"{WORK_DIR}/outputs/d1_history.json"
RESID_PATH = f"{WORK_DIR}/outputs/residual_data.csv"

s3 = boto3.client("s3")


def now_kst():
    return datetime.datetime.now(KST)


def log(msg):
    print(f"[{now_kst().strftime('%Y-%m-%d %H:%M:%S')} KST] {msg}")


# ── Step 1: S3에서 d1_history.json 다운로드 ──────────
def download_history():
    log("S3에서 d1_history.json 다운로드...")
    try:
        os.makedirs(f"{WORK_DIR}/outputs", exist_ok=True)
        s3.download_file(S3_BUCKET, "outputs/d1_history.json", HIST_PATH)
        with open(HIST_PATH, encoding="utf-8") as f:
            hist = json.load(f)
        log(f"  로드 완료: {len(hist)}개 레코드")
        return hist
    except Exception as e:
        log(f"  ❌ 실패: {e}")
        return []


# ── Step 2: 24시간 전 스냅샷 추출 ────────────────────
def get_target_snapshots(hist):
    """
    현재 시각 기준 23~25시간 전 예측값 추출
    (±1시간 허용: 정확히 24h 분봉이 없을 수 있으므로)
    """
    now = now_kst()
    window_start = now - datetime.timedelta(hours=25)
    window_end   = now - datetime.timedelta(hours=23)

    targets = []
    for rec in hist:
        try:
            dt = datetime.datetime.strptime(
                f"{rec['date']} {rec['ts']}", "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=KST)
            if window_start <= dt <= window_end:
                targets.append({**rec, "pred_dt": dt})
        except Exception:
            pass

    log(f"  타겟 스냅샷: {len(targets)}개 (23~25시간 전)")
    return targets


# ── Step 3: yfinance 1분봉으로 실제가격 매칭 ─────────
def get_actual_prices(targets):
    """
    각 예측 시점 + 24h 의 실제 1분봉 가격 수집
    주말/공휴일이면 다음 거래일 첫 번째 분봉 사용
    """
    if not targets:
        return []

    log("yfinance 1분봉 수집...")

    # 최근 7일치 1분봉 (yfinance 최대)
    try:
        df_1m = yf.download(
            "KRW=X", period="7d", interval="1m",
            auto_adjust=True, progress=False
        )
        if isinstance(df_1m.columns, pd.MultiIndex):
            df_1m = df_1m["Close"].iloc[:, 0].to_frame("Close")
        else:
            df_1m = df_1m[["Close"]]
        df_1m.index = pd.to_datetime(df_1m.index).tz_convert(KST)
        log(f"  1분봉 수집: {len(df_1m)}행")
    except Exception as e:
        log(f"  ❌ 1분봉 수집 실패: {e}")
        return []

    results = []
    for t in targets:
        pred_dt  = t["pred_dt"]
        target_dt = pred_dt + datetime.timedelta(hours=24)

        # target_dt 이후 첫 번째 유효한 1분봉 찾기 (최대 ±60분)
        actual = None
        try:
            future = df_1m[df_1m.index >= target_dt]
            if not future.empty:
                closest = future.index[0]
                diff_min = (closest - target_dt).total_seconds() / 60
                if diff_min <= 60:  # 60분 이내
                    actual = round(float(future.iloc[0]["Close"]), 2)
        except Exception:
            pass

        if actual is None:
            log(f"  ⚠ 실제가격 없음: {target_dt.strftime('%m-%d %H:%M')} (주말/공휴일)")
            continue

        lgb_pred = t.get("lgb")
        if lgb_pred is None:
            continue

        error     = round(actual - lgb_pred, 4)
        error_pct = round((actual / lgb_pred - 1) * 100, 4)

        row = {
            "pred_date":   t["date"],
            "pred_ts":     t["ts"],
            "actual_dt":   target_dt.strftime("%Y-%m-%d %H:%M"),
            "cur_price":   t.get("cur_price"),
            "lgb_pred":    lgb_pred,
            "arimax_pred": t.get("arimax"),
            "actual":      actual,
            "error":       error,
            "error_pct":   error_pct,
        }
        # 피처 스냅샷 포함
        for k, v in (t.get("features") or {}).items():
            row[f"feat_{k}"] = v

        results.append(row)
        log(f"  매칭: {t['ts']} → 실제 ₩{actual:,.2f} | 오차 {error:+.2f} ({error_pct:+.3f}%)")

    return results


# ── Step 4: residual_data.csv 누적 저장 ──────────────
def save_residuals(new_rows):
    if not new_rows:
        log("  새 잔차 데이터 없음")
        return

    new_df = pd.DataFrame(new_rows)

    if os.path.exists(RESID_PATH):
        existing = pd.read_csv(RESID_PATH)
        # 중복 제거 (pred_date + pred_ts 기준)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["pred_date", "pred_ts"], keep="last"
        )
    else:
        combined = new_df

    combined = combined.sort_values(["pred_date", "pred_ts"]).reset_index(drop=True)
    combined.to_csv(RESID_PATH, index=False)
    log(f"  residual_data.csv 저장: 총 {len(combined)}행")
    return combined


# ── Step 5: S3 업로드 ─────────────────────────────────
def upload_to_s3():
    log("S3 업로드...")
    files = {
        RESID_PATH: "outputs/residual_data.csv",
    }
    for local, s3_key in files.items():
        if os.path.exists(local):
            try:
                s3.upload_file(local, S3_BUCKET, s3_key)
                log(f"  ✅ {s3_key}")
            except Exception as e:
                log(f"  ❌ {s3_key}: {e}")


# ── Step 6: 간단한 잔차 통계 출력 ────────────────────
def print_stats():
    if not os.path.exists(RESID_PATH):
        return
    try:
        df = pd.read_csv(RESID_PATH)
        log(f"\n=== 잔차 통계 ({len(df)}개 샘플) ===")
        log(f"  평균 오차:  {df['error'].mean():+.3f} KRW")
        log(f"  RMSE:       {(df['error']**2).mean()**0.5:.3f} KRW")
        log(f"  MAE:        {df['error'].abs().mean():.3f} KRW")
        log(f"  방향 정확도: {(df['error'] * (df['lgb_pred'] - df['cur_price']) > 0).mean()*100:.1f}%")
        log(f"  데이터 기간: {df['pred_date'].min()} ~ {df['pred_date'].max()}")

        # 잔차 학습 가능 여부
        if len(df) >= 30:
            log(f"\n  ✅ 잔차 보정 학습 가능! (30개 이상)")
            log(f"     → train_residual.py 실행하세요")
        else:
            log(f"\n  ⏳ 아직 {30 - len(df)}일치 더 필요 (현재 {len(df)}일)")
    except Exception as e:
        log(f"  통계 계산 실패: {e}")


# ── 메인 ─────────────────────────────────────────────
if __name__ == "__main__":
    log("=== 잔차 수집 시작 ===")

    hist    = download_history()
    targets = get_target_snapshots(hist)
    rows    = get_actual_prices(targets)
    save_residuals(rows)
    upload_to_s3()
    print_stats()

    log("=== 완료 ===")
