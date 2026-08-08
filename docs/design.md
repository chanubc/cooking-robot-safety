# 궤적 예측 슬라이스 설계 (Phase 2 — 추적→궤적 예측)

작성 2026-08-08. 조리 로봇 안전 시스템의 "사람 궤적 예측 + 선제 감속/정지" 슬라이스.

## 목표
YOLO로 검출한 사람을 추적(ByteTrack)해 궤적을 만들고, 미래 위치를 예측해 로봇이 위험 시 선제적으로 감속/정지하게 한다. 파이썬으로 예측 모델을 개발·평가한 뒤 우승 모델을 브라우저 시뮬(v1)에 이식한다.

## 범위 결정 (YAGNI)
- **포함**: 등속(constant velocity), 칼만 필터, 평가 하네스(ADE/FDE + 안전지표), 검출-노이즈 비교.
- **연기**: 경량 LSTM / Trajectron++ — 현재 시뮬 궤적이 빈약해 학습형 모델은 헛배운다. **"현실적 궤적 데이터(시뮬 풍부화 or 실데이터)" 확보가 전제조건.** 인터페이스는 이들을 담을 수 있게 넓게 설계해 나중에 알맹이만 꽂는다.

## 배포 위치
- **기본: 온디바이스**(브라우저 WebGPU). 칼만은 JS 이식이 쉽다.
- **에스컬레이션(조건부): 온프레미스**(엣지 PC). 지표 미달 시 같은 인터페이스로 무거운 모델(Trajectron++ 등)로 교체. 클라우드는 지연/신뢰성 문제로 금지.
- 파라미터 규모 참고: 등속/칼만=학습 파라미터 0, 경량 LSTM≈0.0001B, Trajectron++≈0.002B. 전부 M 이하. Trajectron++가 온프레미스인 이유는 크기가 아니라 **브라우저 이식 불가한 소프트웨어 스택(PyTorch/GNN/샘플링)** 때문.

## 아키텍처

```
── 파이썬 오프라인 (개발·평가) ──
  궤적 데이터
   ├─ ① 시뮬 정답 궤적 (JSONL export)         : 깨끗
   └─ ③ 시뮬 BEV영상 → YOLO(best.onnx) → ByteTrack → Mapper : 노이즈
        ↓
  [예측기 인터페이스] predict(TrackScene) → Prediction
   ├─ 등속  ├─ 칼만  └─ (미래) LSTM / Trajectron++
        ↓
  [평가] ADE/FDE + 안전지표(선제정지·오정지·리드타임)
        ↓ 우승 선정
── 브라우저 (라이브 데모, v1 확장) ──
  BEV → YOLO(onnxruntime-web) → 간이추적 → [예측기] → SSM 3단계 → 로봇 제어
  (기존 person.position 정답입력 → 검출입력으로 교체)
```

## 모듈 (각 하나의 역할)
1. **Tracker** — 프레임별 검출 → ID·궤적. (ByteTrack)
2. **Mapper** — 이미지 픽셀좌표 → 바닥좌표(미터). 카메라 정보 기반(시뮬은 정확히 앎).
3. **Predictor** — 궤적 → 미래(여러 갈래). ★교체 지점.
4. **Evaluator** — ADE/FDE + 안전지표.
5. **SSMPolicy** — 미래 → 로봇 판정(정상/감속/정지). 보수적 종합.

## 인터페이스 계약 (되돌림 방지 — 넓게 정의)

입력 `TrackScene`:
```
TrackScene { now: float, horizon: float, agents: [Track], map: Optional[Map] }
Track { id: int, history: [ (t, x, z) ] }     # 바닥좌표 시계열
Map   { robot_zone: {x, z, r} }               # 선택
```

출력 `Prediction`:
```
Prediction { per_agent: { id: [Mode, ...] } }
Mode { prob: float, steps: [ (t, x, z, sigma) ] }   # multimodal + 불확실성
```

- 등속/칼만: 자기 트랙만 사용, map 무시, Mode 1개(prob=1.0). 등속은 sigma=0.
- Trajectron++(미래): agents 전원 + map + 여러 Mode 사용 → 인터페이스 무수정.
- **SSMPolicy 규칙**: 모든 agent×Mode×step 중 로봇 위험영역에 가장 빨리·확실히 진입하는 **최악 케이스**로 TTC 판정 → 단일/멀티모달 모두 동일 로직.

## 평가 지표
- **정확도**: ADE(전체 평균 위치오차), FDE(최종시점 오차). 단위 m, 낮을수록 좋음.
- **안전(핵심)**:
  - 선제정지 성공률 = 사람이 위험영역 진입 **전에** 감속/정지한 비율 (목표 ≥99%).
  - 오정지율 = 안 위험한데 멈춘 비율 (낮을수록).
  - 반응 리드타임 = 진입 몇 초 전에 반응했나.
  - 비대칭: 미탐(못 멈춤=사고) ≫ 오탐(괜히 멈춤=손실). 선제정지 성공률 최우선.
- **비교표 (모델2 × 입력2)**:

  | | ① 정답(깨끗) | ③ YOLO+추적(노이즈) |
  |---|---|---|
  | 등속 | ADE/FDE·안전 | ADE/FDE·안전 |
  | 칼만 | ADE/FDE·안전 | ADE/FDE·안전 |

  결론: (a) 등속 vs 칼만, (b) 검출 노이즈의 대가(①vs③), (c) 온디바이스 칼만으로 충분한가 = 에스컬레이션 판정.

## 테스트 전략
- 단위: Mapper(변환 정답 일치), 등속(직선→오차0), 칼만(노이즈 직선→등속보다 ADE↓, 무노이즈→등속과 동일), Evaluator(손계산 일치), SSMPolicy(진입 예측→stop, 이탈→run).
- 통합: 대본 시나리오(사람이 로봇으로 직진)로 정답 진입시각 vs 반응시각 → 안전지표 검증. "골든 시나리오" 픽스처 고정.

## 구현 단계
1. 자료구조(TrackScene/Prediction) + 시뮬 ① 정답 궤적 JSONL export.
2. Predictor: 등속(기존 이식) + 칼만, 인터페이스 뒤에.
3. Evaluator: ADE/FDE + 안전지표 → ①로 비교표 절반.
4. ③ 경로: BEV영상 → YOLO(best.onnx) → ByteTrack → Mapper → TrackScene.
5. ③로 비교표 완성 + 에스컬레이션 판정.
6. 우승 모델 JS 이식 → v1 배선 교체 → 라이브 데모.

1~5 파이썬 오프라인, 6 브라우저. 패키지는 uv 관리. Windows에서 학습/멀티프로세싱 코드는 `if __name__ == "__main__":` 가드 필수.
