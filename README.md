# CUDA를 활용한 Tiled-based Matrix Multiplication 및 Cost Volume Aggregation

## 개요
본 프로젝트는 CUDA를 활용하여 Shared Memory 기반의 Tiled Matrix Multiplication 및 Cost Volume Aggregation CUDA 커널을 구현하고, 이를 통해 성능 향상을 실험하고 분석한 내용입니다.

---

## 1. Shared Memory를 활용한 Tile-based Matrix Multiplication CUDA 커널 작성
### Naive Matrix Multiplication CUDA Kernel

- <img src = "https://github.com/user-attachments/assets/c8734816-b5ee-4881-8a63-4212859bb17f" width="width size%" height="height size%">
- CUDA를 이용한 기본적인 행렬 곱셈 커널 구현.
- 과제 요구 사항: `α = 1`, `β = 0` 조건 반영.

---

### Tiled Matrix Multiplication CUDA Kernel

- <img src = "https://github.com/user-attachments/assets/50969cd3-c5ff-4949-ae5b-2a0cc7babd6f" width="width size%" height="height size%">
- Shared memory tiling 기법 적용.
- A, B 행렬의 타일 단위 데이터를 shared memory(`tile_A`, `tile_B`)에 저장.
- `__syncthreads()`를 통해 Race Condition 방지 및 thread 간 동기화 수행.
- Shared memory를 효율적으로 활용하여 메모리 접근 최적화.

---

### Matrix 연산 보조 함수

- <img src = "https://github.com/user-attachments/assets/91056cb5-d5e0-4fdc-ab14-8cca5ca1291c" width="width size%" height="height size%">
- `initialize_matrix(...)`: 행렬 초기화 함수.
- `compare_matrix(...)`: Naive, Tiled 연산 결과 비교. 부동 소수점 오차 허용 범위 `tol = 1e-3f` 적용.

- <img src = "https://github.com/user-attachments/assets/2a6ae811-ad20-4515-91ff-cf2523201263" width="width size%" height="height size%">
- `measure_and_run(...)`: 메모리 할당, 커널 호출, 연산 수행, 메모리 해제, 공정한 비교를 위해 naive matmul과 tiled matmul 실행 사이에서 위 사진에 명시된 커널을 호출하여 Cache 초기화 수행.

---

### 성능 비교 실험

- <img src = "https://github.com/user-attachments/assets/3d3a4ac2-0633-4a0e-ab35-c2a28f1a711b" width="width size%" height="height size%">
- **GPU**: V100 사용.
- **행렬 크기**: 1024 x 1024.
- **커널 반복 횟수**: 100 ~ 1000회.
- **성능 측정 방법**: `cudaEventElapsedTime()` API 사용.
- **결과**: Tiled kernel이 Naive kernel 대비 약 1.7배 성능 향상.

---

## 2. Cost Volume Aggregation CUDA 커널 작성
### Cost Volume Aggregation CUDA Kernel

- <img src = "https://github.com/user-attachments/assets/3c1f3678-9ca9-4003-ae58-58ad9451c1a5" width="width size%" height="height size%">
- Channel: 32로 고정.
- Block 내 Thread: 한 channel을 담당.
- 각 Thread Block: 한 줄의 Pixel 처리.

- <img src = "https://github.com/user-attachments/assets/f61b37dc-56de-4128-862e-20fecdd93dc6" width="width size%" height="height size%">
- Shared Memory에 각 View의 Weight 및 Feature 저장.

- <img src = "https://github.com/user-attachments/assets/7528d71a-544b-45cc-b844-53f92f580379" width="width size%" height="height size%">
- N개의 View의 가중 평균 Feature를 계산하여 Output에 저장.
- 분모가 0이 되는 상황 방지를 위해 최소값 `1e-6f` 설정.

---

## 3. 서술형 
### 필수-1. Shared Memory와 GPU 메모리 계층

- <img src = "https://github.com/user-attachments/assets/ed49cd67-bd04-4035-b12b-75d4669f4c1d" width="width size%" height="height size%">
- Shared memory는 하나의 thread block 내의 모든 thread들이 공유할 수 있는 메모리 공간.
- GPU 메모리 계층은 위 사진과 같은 구성.
  - SM 내에 Register memory와 L1 cache/Shared memory 등이 존재하며, off-chip 메모리로는 L2 cache, Global memory 등이 존재.
  - Shared memory는 GPU의 메모리 계층 중에서 global memory 보다 빠르고, 레지스터 메모리 보다는 느림.
- Shared memory는 비교적 작은 사이즈이지만 global memory에 비해 속도가 빠르기 때문에 tiling matrix multiplication과 같은 적절한 방법을 통해 global memory 접근 횟수를 줄이면 성능을 크게 향상 시킬수 있음.

---

### 필수-2. Coalesced Memory Access

- Coalesced Access: Warp 내 32개 thread의 memory access가 하나의 cache line을 access하여 한 개의 memory transaction으로 처리되는 경우.
- Coalesced Access를 하게 되는 경우, warp 내의 모든 thread의 memory instruction이 하나로 합쳐진 memory transaction이 생성되고 이로 인해 한 번의 memory access 만으로도 32개 thread의 memory request를 모두 처리 가능
- 만약 32개의 memory request가 연속적이긴 하나 두 개의 cache line에 걸쳐 접근하는 경우는 2번의 memory transaction이 발생하여 coalesced access 라고 보지 않음
- 반대로 32개의 memory request가 연속적이진 않으나 한 cache line을 access 하는 경우 CUDA compute capability에 따라 달라짐 (CUDA compute 2.0을 기준으로 달라짐)
- CUDA에서 global memory access는 memory transaction 단위로 처리하기 때문에 최대한 coalesced access로 global memory에 접근하는 것이 커널 성능 향상에 도움이 됨

---

