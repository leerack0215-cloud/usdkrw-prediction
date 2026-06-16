# USD/KRW 환율 예측 시스템

> ARIMAX · LightGBM · LSTM · BiGRU + Ridge 앙상블  
> 멀티스케일 피처(137개) · AWS 자동화 파이프라인 · Streamlit 실시간 대시보드

성균관대학교 정보통신대학원 빅데이터학과 · 2026년 1학기 데이터마이닝  
이용재 (2025720228)

---

## 개요

D+1 ~ D+22(영업일 기준) 다중 호라이즌으로 USD/KRW 환율을 예측하는 앙상블 딥러닝 시스템입니다.

- **Lookahead Bias 완전 차단** — 모든 피처는 현재 이전 데이터만 사용
- **멀티스케일 피처** — 일봉 + 1H + 5M + 1M 분봉 통계를 일봉에 통합
- **4모델 앙상블** — ARIMAX + LightGBM + LSTM + BiGRU → Ridge 메타러너
- **실시간 운영** — 매일 KST 06:00 자동 예측 → Streamlit Cloud 대시보드

---

## 성능 (Validation Set, 단순 분할 기준)

| 모델 | RMSE(↓) | MAE(↓) | DA%(↑) | Sharpe(↑) |
|------|---------|--------|--------|-----------|
| Persistence (Baseline) | 8.83원 | 6.65원 | 49.0% | — |
| ARIMAX | 8.45원 | 6.38원 | 47.5% | -0.095 |
| LightGBM D+1 | 7.85원 | 5.87원 | **57.3%** | -0.102 |
| LSTM D+1 | 9.26원 | 6.96원 | 48.5% | 0.348 |
| BiGRU D+1 | 29.55원 | 27.20원 | 48.8% | 0.268 |
| **★ Ridge 앙상블 (4모델)** | **7.45원** | **5.81원** | 48.5% | **0.367** |

> ※ Walk-Forward Validation 미적용 결과 — 적용 시 수치 변동 가능  
> ※ `DA_p`: 이항검정(H₀: DA=50%) p-value — 재학습 후 `performance_table.csv`에 자동 생성

---

## 프로젝트 구조

```
usdkrw-prediction/
├── utils.py                 # 데이터 수집·피처 엔지니어링·평가 지표
├── train.py                 # 9단계 학습 파이프라인 (Colab GPU, 4~8시간)
├── predict.py               # 일일 다중 호라이즌 예측 (~1분)
├── dashboard.py             # Streamlit 실시간 대시보드
├── collect_residuals.py     # EC2 cron — 24h 전 예측 vs 실제 오차 수집
├── colab_master.ipynb       # Google Colab 8셀 자동화 노트북
├── make_ppt.py              # 기말 보고서 PPT 생성 스크립트
├── requirements.txt         # 배포용 (Streamlit Cloud, TF 미포함)
├── requirements_train.txt   # 학습용 (Colab, TF + pmdarima 포함)
└── outputs/
    ├── models/              # 학습된 모델 파일 (.pkl, .keras)
    ├── scaler_X.pkl         # RobustScaler (Train에만 fit)
    ├── feature_list.json    # 피처 순서 목록 (137개)
    ├── meta_info.json       # 앙상블 구성 메타 정보
    ├── forecast_today.json  # 오늘 예측값 (대시보드용)
    └── performance_table.csv # 모델 성능 비교표 (DA_p 포함)
```

---

## 빠른 시작

### 1. 환경 설치

```bash
# 대시보드만 실행
pip install -r requirements.txt

# 학습까지 실행 (Colab 권장)
pip install -r requirements_train.txt
```

### 2. 학습 (Colab GPU)

```bash
python train.py
# Phase 1~9 자동 실행 후 predict.py 자동 호출
# 소요 시간: 4~8시간 (T4 GPU 기준)
```

### 3. 예측 (일일)

```bash
python predict.py
# outputs/forecast_today.json 생성
# 소요 시간: ~1분
```

### 4. 대시보드 실행

```bash
streamlit run dashboard.py
```

---

## 데이터 수집

| 구분 | 기간 | 집계 방식 |
|------|------|----------|
| 일봉 | 2015-01-01 ~ 현재 | 원본 OHLCV |
| 1시간봉 | 최근 730일 | OHLC + std + range → 일봉 집계 |
| 5분봉 | 최근 60일 | std + range + count → 일봉 집계 |
| 1분봉 | 최근 7일 | std + range + 오프닝 리턴 → 일봉 집계 |

**수집 티커 (14개)**: KRW=X, DX-Y.NYB, ^VIX, ^TNX, ^IRX, CL=F, GC=F, ^GSPC, SOXX, USDJPY=X, USDCNY=X, USDTWD=X, ^KS11

---

## 피처 그룹 (총 137개)

| 그룹 | 내용 | 피처 수 |
|------|------|--------|
| G1 추세 | EMA·SMA·MACD | 16 |
| G2 모멘텀 | RSI·Stoch·CCI·ADX | 11 |
| G3 변동성 | BB·ATR·HV·레짐 | 15 |
| G4 수익률 | 다구간 리턴 + 래그 | 12 |
| G5 금리차 | 한미 스프레드·수익률곡선 | 8 |
| G6 크로스통화 | CNY·JPY·TWD 상대강도 | 9 |
| G7 한국특화 | SOX·EEM·WTI(KRW) | 12 |
| G8 멀티스케일 | 분봉 통계 정규화 | 17 |
| G9 캘린더 | 요일·월·지정학 이벤트 | 6 |

---

## 모델 파이프라인 (train.py 9단계)

```
Phase 1  멀티스케일 데이터 수집
Phase 2  강화 피처 엔지니어링 (137개)
Phase 3  시계열 분할 70 / 15 / 15
Phase 4  ARIMAX 학습 (외생변수 6개)
Phase 5  LightGBM 학습 (D+1 ~ D+22)
Phase 6  LSTM + BiGRU 학습 (D+1)
Phase 7  Ridge 메타 앙상블
Phase 8  성능 평가 + Ablation Table + Persistence Baseline + 이항검정
Phase 9  predict.py 자동 호출
```

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| RMSE | 절대 오차 크기 (원화) |
| MAE | 평균 절대 오차 |
| MAPE | 상대 오차율 (%) |
| DA | 방향 정확도 (%) |
| **DA_p** | 이항검정 p-value (H₀: DA=50%, 단측) — `p<0.05` 시 유의 |
| Sharpe | 리스크 조정 수익률 (연환산 √252) |

---

## 배포 아키텍처

```
Google Colab (주 1회 수동)
  └─ train.py 실행 → outputs/ 생성 → git push
         │
         ▼
      GitHub (main)
    ┌────┴────────────────┐
    ▼                     ▼
Streamlit Cloud      GitHub Actions (UTC 21:30)
 dashboard.py          └─ S3 sync → git commit/push
    ▲
    │
AWS S3  ←  EC2/Lambda (KST 06:00 cron)
              ├─ predict.py → forecast_today.json
              └─ collect_residuals.py → residual_data.csv
```

---

## Colab 자동화 (colab_master.ipynb)

| 셀 | 역할 |
|----|------|
| 1 | GitHub Secrets 로드 |
| 2 | Drive 마운트 + 레포 클론 |
| 3 | 패키지 설치 (`requirements_train.txt`) |
| 4 | `git pull` — 최신 코드 동기화 |
| 5 | `train.py` or `predict.py` 자동 판단 실행 |
| 6 | Streamlit 로컬 테스트 (선택) |
| 7 | GitHub push (`outputs/` 포함) |
| 8 | 스케줄러 (매일 KST 06:00 반복) |

---

## 향후 계획

- [ ] **Walk-Forward Validation** — Expanding window 기반 일반화 성능 검증
- [ ] **VIF 다중공선성 진단** — 137개 피처 상관 구조 분석 및 정제
- [ ] **레짐별 앙상블** — 고변동/저변동 레짐 분리 후 별도 모델 적용 (H3 가설)
- [ ] **잔차 보정 모델** — `residual_data.csv` 누적 후 24h 예측 오차 패턴 학습

---

## 면책 고지

> ⚠ 본 시스템의 예측 결과는 **학술·참고 목적 전용**입니다.  
> 실제 투자 결정에 따른 책임은 전적으로 사용자에게 있습니다.
