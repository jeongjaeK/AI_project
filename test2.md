# 통합 시험 환경 구성도

아래 구성도는 앞서 정리한 시험 1(IDF1), 시험 2(RAG+LLM), 시험 3(장비 인식 mAP@0.5), 시험 4(클라이언트 응답시간) 흐름과 추가로 제공한 RAG 검색 API / PostgreSQL + pgvector 구성을 하나로 통합한 것이다.

```mermaid
flowchart LR
    %% =====================================================
    %% 공통 시험 PC / 내부 네트워크
    %% =====================================================

    PC["시험 PC<br/>Apple M5 Pro / macOS 26.6.1<br/>시험 1 SSH 접속 및 스크립트 실행<br/>시험 1 결과 확인"]

    NET["내부 네트워크<br/>HTTP API / SSH<br/>시험 데이터 전송 및 결과 수신"]

    PC <--> NET

    %% =====================================================
    %% 시험 1. 객체 추적 정확도(IDF1)
    %% =====================================================

    subgraph TEST1["시험 1. 객체 추적 정확도 (IDF1)"]
        direction LR

        GPU1["GPU 서버 1<br/>Ubuntu 26.04.1 LTS<br/>NVIDIA GeForce RTX 5090<br/>SAM3 기반 다개체 추적 모듈<br/>IDF1 평가 모듈"]

        RUN1["평가 스크립트<br/>run_all.sh"]

        VIDEO1[("시험 영상 2개<br/>IR→RGB<br/>RGB→IR")]

        GT1[("정답 데이터 2개<br/>bbox · 지속 식별자")]

        TRACK1["SAM3 기반 다개체 추적 모듈<br/>Python 3.12<br/>4 fps 실시간 도착 재현"]

        CSV1[("추적 결과 CSV<br/>tracks.csv")]

        EVAL1["IDF1 평가 모듈<br/>Python 3.11<br/>TrackEval<br/>IoU 0.50"]

        OUT1["통합 IDF1 출력<br/>터미널 · summary.json"]

        GPU1 --> RUN1
        RUN1 --> TRACK1
        VIDEO1 --> TRACK1
        TRACK1 --> CSV1
        CSV1 --> EVAL1
        GT1 --> EVAL1
        EVAL1 --> OUT1
    end

    NET <--> GPU1
    PC -. "SSH 접속 · 스크립트 실행" .-> RUN1
    OUT1 -. "결과 확인" .-> PC

    %% =====================================================
    %% 시험 2. RAG + LLM 질의응답 정확도
    %% =====================================================

    subgraph TEST2["시험 2. 수요기업 평가 기반 AI Agent 질의응답 정확도 (Mean Evidence Recall@5)"]
        direction LR

        EVALDATA[("평가 데이터<br/>시험 질의 10개<br/>문항별 정답 근거 목록<br/>(manual_id + relevant_pages)")]

        GPU2["GPU 서버 2 · 공통 평가/클라이언트 서버<br/>Intel Core i9-10980XE / RTX 4090 × 2<br/>Ubuntu 20.04.6 LTS<br/>Python 3.10.20<br/>PyTorch 2.7.1 / CUDA 12.8<br/>시험 2 RAG 검색 API<br/>시험 3·4 전처리·후처리·평가 및 시간 측정"]

        PRE2["문서 · 질의 전처리"]

        EMB2["BGE-M3 임베딩 검색"]

        RERANK2["Cross-Encoder 재순위화"]

        DB2[("PostgreSQL + pgvector<br/>dataset-v3 문서 메타데이터<br/>청크 텍스트 및 임베딩<br/>manual_id 및 페이지 정보")]

        TOPK2["상위 검색 청크 반환<br/>문서 식별자 · 페이지 정보 포함"]

        EVAL2["Mean Evidence Recall@5 평가<br/>정답 근거와 검색 결과 비교"]

        OUT2["평가 결과 출력<br/>터미널 · 파일"]

        GPU2 --> PRE2
        PRE2 --> EMB2
        EMB2 --> DB2
        DB2 --> EMB2
        EMB2 --> RERANK2
        RERANK2 --> TOPK2
        EVALDATA -- "POST /search" --> EVAL2
        TOPK2 --> EVAL2
        EVAL2 --> OUT2
    end

    NET <--> GPU2
    OUT2 -. "GPU 서버 2에서 결과 확인" .-> GPU2

    %% =====================================================
    %% 시험 3 + 시험 4. 장비 인식 / 응답시간
    %% =====================================================

    subgraph TEST34["시험 3. 이미지 탐지 정확도 (mAP@0.5) + 시험 4. 클라이언트 응답시간"]
        direction LR

        GPUC["GPU 서버 2 · 평가용 클라이언트<br/>Intel Core i9-10980XE<br/>RTX 4090 × 2<br/>시험 2와 동일 서버<br/>전처리 · 후처리 · 평가<br/>시간 측정 주체"]

        IMG34[("시험 이미지 15장<br/>및 정답 데이터")]

        PRE34["원본 · 부분 이미지 생성"]

        REQ34["순차 API 요청<br/>HTTP POST /detect"]

        NPU["NPU 서버 : YOLO 장비 인식<br/>Rebellions RBLN-CA25 × 4<br/>클래스 · confidence · bbox 반환"]

        RESP34["탐지 결과 수신<br/>class · confidence · bbox"]

        POST34["좌표 복원 · 결과 결합"]

        MAP34["mAP@0.5 계산"]

        TIME34["클라이언트 응답시간 측정<br/>평가용 GPU 서버의 클라이언트 프로세스"]

        LOG34[("응답시간 별도 로그 파일<br/>회차별 측정값 저장")]

        AVG34["평가 스크립트 실행<br/>로그의 15회 측정값 평균 계산<br/>평균 응답시간 터미널 출력"]

        IMG34 --> PRE34
        PRE34 --> REQ34
        REQ34 --> NPU
        NPU --> RESP34
        RESP34 --> POST34
        POST34 --> MAP34

        REQ34 --> TIME34
        RESP34 --> TIME34
        TIME34 --> LOG34
        LOG34 --> AVG34
    end

    GPU2 --> IMG34

    %% =====================================================
    %% 공통 주석
    %% =====================================================

    NOTE1["※ 모델 추론: NPU 서버"]
    NOTE2["※ 시험 2·3·4는 동일한 GPU 서버 2에서 수행"]
    NOTE3["※ 응답시간 측정 주체: 평가용 GPU 서버의 클라이언트 프로세스"]
    NOTE4["※ RAG 검색: BGE-M3 임베딩 + Cross-Encoder 재순위화 + PostgreSQL/pgvector"]

    TEST34 --- NOTE1
    TEST34 --- NOTE2
    TEST34 --- NOTE3
    TEST2 --- NOTE4
```

## 흐름 요약

- **시험 PC**
  - IDF1 시험에서 SSH를 통해 GPU 서버 1에 접속하여 스크립트를 실행하고 터미널 결과를 확인한다.
  - 시험 2·3·4의 평가 및 결과 확인은 GPU 서버 2에서 수행한다.

- **시험 1: 객체 추적 정확도(IDF1)**
  - 시험 영상 2개를 SAM3 기반 다개체 추적 모듈에 입력한다.
  - 추적 결과를 `tracks.csv`로 저장한다.
  - 정답 데이터는 추적 모듈이 아니라 IDF1 평가 모듈에만 입력한다.
  - TrackEval에서 IoU 0.50 기준으로 평가하고 통합 IDF1을 출력한다.

- **시험 2: RAG+LLM 질의응답 정확도**
  - 시험 2는 GPU 서버 2에서 수행하며 결과도 GPU 서버 2에서 확인한다.
  - 평가 데이터의 시험 질의 10개와 문항별 정답 근거 목록을 사용해 `POST /search` 기반 검색 평가를 수행한다.
  - GPU 서버 2의 FastAPI 기반 RAG API가 질의를 전처리한다.
  - BGE-M3로 임베딩 검색을 수행하고 PostgreSQL + pgvector에서 관련 청크를 조회한다.
  - Cross-Encoder로 후보 청크를 재순위화한다.
  - 반환된 상위 검색 청크와 문항별 정답 근거 `(manual_id + relevant_pages)`를 비교하여 Mean Evidence Recall@5를 계산한다.

- **시험 3: 이미지 탐지 정확도(mAP@0.5)**
  - 시험 2와 동일한 GPU 서버 2가 평가용 클라이언트 서버로 동작하며 시험 이미지 15장과 정답 데이터를 관리한다.
  - 원본·부분 이미지를 생성하고 NPU 서버의 `/detect` API로 순차 요청한다.
  - NPU 서버는 YOLO 기반 추론 후 클래스, confidence, bbox를 반환한다.
  - 평가용 GPU 서버가 좌표 복원, 결과 결합, mAP@0.5 계산을 수행한다.

- **시험 4: 클라이언트 응답시간**
  - 시험 2·3과 동일한 GPU 서버 2의 클라이언트 프로세스가 요청·응답 구간의 응답시간을 측정한다.
  - 회차별 응답시간을 별도 로그 파일에 저장한다.
  - 평가 스크립트가 로그 파일의 15회 측정값을 읽어 평균 응답시간을 계산하고 터미널에 출력한다.

