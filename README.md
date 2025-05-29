# ResNet DDP 분산 학습 예제 (AstraGo 워크로드 테스트용)

이 코드는 AstraGo의 워크로드 생성 및 모니터링 기능을 테스트하기 위한 PyTorch 기반의 분산 학습(Distributed Data Parallel, DDP) 예제입니다.  
실제 데이터가 아닌, 메모리를 아끼는 가짜 데이터셋(LazyFakeDataset)을 사용하여 여러 GPU에서 ResNet-50 모델을 학습합니다.

## 주요 특징

- **PyTorch DDP(DistributedDataParallel) 기반 멀티 GPU 학습**
- **SyncBatchNorm** 적용
- **가짜 데이터셋(LazyFakeDataset)**: 실제 이미지 대신 무작위 텐서와 라벨을 생성하여 메모리 사용 최소화
- **100,000개 샘플, 100 에폭 학습**
- **각 GPU별로 학습 진행 상황 및 평균 Loss 출력**
- **AstraGo 환경에서 워크로드 생성/모니터링 테스트에 적합**

## 파일 구조

```
train.py  # 분산 학습 전체 로직이 포함된 메인 파일
```

## 실행 환경

- Python 3.8 이상
- PyTorch 1.9 이상
- torchvision
- CUDA가 지원되는 NVIDIA GPU 2개 이상

## 설치 방법

```bash
pip install torch torchvision
```

## 실행 방법

1. **2개 이상의 GPU가 필요합니다.**
2. 아래 명령어로 실행하세요.

```bash
python train.py
```

- GPU가 2개 미만일 경우, 에러 메시지가 출력되고 종료됩니다.
- GPU가 2개 이상이면, 각 GPU에서 프로세스가 생성되어 분산 학습이 시작됩니다.

## 코드 설명

- **LazyFakeDataset**: `__getitem__`에서 무작위 이미지(3x224x224)와 0~99 사이의 라벨을 생성합니다.
- **setup/cleanup**: DDP 환경 초기화 및 정리 함수입니다.
- **train**: 각 GPU별로 모델, 데이터로더, 옵티마이저, 손실함수 등을 생성하고 100 에폭 동안 학습을 수행합니다.
- **main**: 사용 가능한 GPU 개수를 확인하고, 2개 이상일 때만 `mp.spawn`으로 분산 학습을 시작합니다.

## 참고

- 이 코드는 실제 데이터가 아닌 가짜 데이터를 사용하므로, 모델의 성능이나 정확도는 의미가 없습니다.
- AstraGo의 워크로드 생성 및 모니터링 기능을 테스트하는 용도로 사용하세요.
