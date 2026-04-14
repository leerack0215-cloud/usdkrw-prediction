# USD/KRW 딥러닝 예측 시스템

## 파일 구성

| 파일 | 역할 |
|------|------|
| `model_pipeline.py` | 데이터 수집 → 피처 엔지니어링 → 모델 학습 → 평가 전체 파이프라인 |
| `dashboard.py` | Streamlit 대시보드 (실시간 시각화) |
| `requirements.txt` | 의존성 목록 |

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 모델 파이프라인 실행 (4~8시간, GPU 권장)
python model_pipeline.py

# 3. 대시보드 실행
streamlit run dashboard.py
```

## 대시보드만 먼저 보기

모델 학습 없이도 대시보드를 실행하면  
실시간 yfinance 데이터로 차트 및 거시지표가 표시됩니다.  
예측값은 더미(시뮬레이션)로 표시되며, `model_pipeline.py` 실행 후 실제 예측으로 전환됩니다.

## 출력 파일 (`outputs/` 폴더)

```
outputs/
├── model_lstm.keras        ← LSTM 가중치
├── best_*.keras            ← 각 모델 최적 체크포인트
├── model_ensemble.pkl      ← 앙상블 메타 모델
├── scaler_X.pkl            ← 피처 스케일러
├── scaler_y.pkl            ← 타겟 스케일러
├── feature_list.json       ← 최종 선택 피처 목록
├── performance_table.csv   ← 전체 모델 성능 지표
├── forecast_today.json     ← 오늘 D+1 예측
└── shap_summary.png        ← SHAP 피처 중요도
```

## PHASE 별 구현 현황

| PHASE | 내용 | 상태 |
|-------|------|------|
| 0 | 설계 원칙 (Lookahead Bias 차단, WFV, 3중 과적합 방지) | ✅ |
| 1 | 데이터 수집 (yfinance 12종 + FRED 거시) + 품질 검증 | ✅ |
| 2 | 피처 엔지니어링 (기술적 지표 20+ · 거시 파생 · 캘린더) | ✅ |
| 3 | 시계열 분할 (Temporal Split + WFV + 시퀀스 생성) | ✅ |
| 4 | 모델 구축 (LSTM · CNN-BiLSTM · WaveNet · Transformer · Tree) | ✅ |
| 5 | 학습 파이프라인 (EarlyStopping · LR Schedule · OOF) | ✅ |
| 6 | 평가 (RMSE/MAE/MAPE/DA/SR + WFV + 백테스팅 시뮬레이션) | ✅ |
| 7 | SHAP 해석 (TreeExplainer / KernelExplainer) | ✅ |
| 8 | 산출물 저장 + Streamlit 대시보드 | ✅ |

## 권장 환경

- Python 3.10+
- GPU: NVIDIA RTX 3090+ (VRAM 24GB) 또는 Google Colab A100
- RAM: 32GB 이상
- 예상 학습 시간: 4~8시간 (전 모델, GPU 기준)

## 면책 고지

> ⚠ 본 시스템은 학술 및 참고 목적이며,  
> 실제 투자 결정의 책임은 전적으로 사용자에게 있습니다.
