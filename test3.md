# 통합 시험 환경 구성도

아래 구성도는 시험 1(IDF1), 시험 2(RAG 검색 정확도), 시험 3(장비 인식 mAP@0.5), 시험 4(클라이언트 응답시간)을 하나로 통합한 것이다.

특히 **GPU 서버 2 내부에 시험 2와 시험 3·4의 평가/클라이언트 프로세스를 함께 배치**하여, 시험 2의 RAG 검색 및 평가가 GPU 서버 2에서 수행된다는 점이 명확하게 보이도록 구성하였다.

```mermaid
flowchart LR

    %% =====================================================
    %% 시험 PC / 네트워크
    %% =====================================================

    PC["시험 PC<br/>Apple M5 Pro / macOS 26.6.1<br/>시험 1 SSH 접속 및 스크립트 실행<br/>시험 1 결과 확인"]

    NET["내부 네트워크<br/>HTTP API / SSH<br/>시험 데이터 전송 및 결과 수신"]

    PC <--> NET


    %% =====================================================
    %% 시험 1 : GPU 서버 1
    %% =====================================================

    subgraph TEST1["시험 1. 객체 추적 정확도 (IDF1)"]
        direction LR

        subgraph GPU1["GPU 서버 1<br/>Ubuntu 26.04.1 LTS · NVIDIA RTX 5090"]
            direction LR

            RUN1["평가 스크립트<br/>run_all.sh"]

            VIDEO1[("시험 영상 2개<br/>IR→RGB · RGB→IR")]

            TRACK1["SAM3 기반 다개체 추적 모듈<br/>Python 3.12<br/>4 fps 실시간 도착 재현"]

            CSV1[("추적 결과 CSV<br/>tracks.csv")]

            GT1[("정답 데이터 2개<br/>bbox · 지속 식별자")]

            EVAL1["IDF1 평가 모듈<br/>Python 3.11 · TrackEval<br/>IoU 0.50"]

            OUT1["통합 IDF1 출력<br/>터미널 · summary.json"]

            RUN1 --> TRACK1
            VIDEO1 --> TRACK1
            TRACK1 --> CSV1
            CSV1 --> EVAL1
            GT1 --> EVAL1
            EVAL1 --> OUT1
        end
    end

    NET <--> RUN1
    PC -. "SSH 접속 · 스크립트 실행" .-> RUN1
    OUT1 -. "결과 확인" .-> PC


    %% =====================================================
    %% GPU 서버 2 : 시험 2 / 시험 3 / 시험 4 공통 서버
    %% =====================================================

    subgraph GPU2ALL["GPU 서버 2 · 시험 2·3·4 공통 평가 / 클라이언트 서버<br/>Intel Core i9-10980XE · RTX 4090 × 2<br/>Ubuntu 20.04.6 LTS<br/>Python 3.10.20 · PyTorch 2.7.1 · CUDA 12.8"]
        direction TB

        %% -------------------------------------------------
        %% 시험 2 : RAG 검색 정확도
        %% -------------------------------------------------

        subgraph TEST2["시험 2. AI Agent 질의응답 검색 정확도 (Mean Evidence Recall@5)"]
            direction LR

            EVALDATA[("평가 데이터<br/>시험 질의 10개<br/>문항별 정답 근거 목록<br/>(manual_id + relevant_pages)")]

            API2["FastAPI 기반 RAG 검색 API<br/>/search"]

            PRE2["문서 · 질의 전처리"]

            EMB2["BGE-M3<br/>임베딩 검색"]

            DB2[("PostgreSQL + pgvector<br/>dataset-v3 문서 메타데이터<br/>청크 텍스트 및 임베딩<br/>manual_id · 페이지 정보")]

            RERANK2["Cross-Encoder<br/>재순위화"]

            TOPK2["상위 5개 검색 청크<br/>문서 식별자 · 페이지 정보"]

            EVAL2["Mean Evidence Recall@5 평가<br/>검색 결과와 정답 근거 비교"]

            OUT2["평가 결과 확인<br/>GPU 서버 2 터미널 · 결과 파일"]

            EVALDATA -- "POST /search" --> API2
            API2 --> PRE2
            PRE2 --> EMB2
            EMB2 <--> DB2
            EMB2 --> RERANK2
            RERANK2 --> TOPK2
            TOPK2 --> EVAL2
            EVALDATA --> EVAL2
            EVAL2 --> OUT2
        end


        %% -------------------------------------------------
        %% 시험 3 + 시험 4 : 장비 인식 및 응답시간
        %% -------------------------------------------------

        subgraph TEST34["시험 3. 이미지 탐지 정확도 (mAP@0.5) + 시험 4. 클라이언트 응답시간"]
            direction LR

            IMG34[("시험 이미지 15장<br/>및 정답 데이터")]

            PRE34["원본 · 부분 이미지 생성<br/>전처리"]

            REQ34["클라이언트 요청<br/>HTTP POST /detect"]

            RESP34["탐지 결과 수신<br/>class · confidence · bbox"]

            POST34["좌표 복원 · 결과 결합<br/>후처리"]

            MAP34["mAP@0.5 계산<br/>평가 결과 출력"]

            TIME34["클라이언트 응답시간 측정<br/>GPU 서버 2 클라이언트 프로세스"]

            LOG34[("응답시간 로그 파일<br/>15회 측정값 저장")]

            AVG34["평가 스크립트<br/>15회 평균 응답시간 계산<br/>터미널 출력"]

            IMG34 --> PRE34
            PRE34 --> REQ34
            RESP34 --> POST34
            POST34 --> MAP34

            REQ34 --> TIME34
            RESP34 --> TIME34
            TIME34 --> LOG34
            LOG34 --> AVG34
        end
    end

    NET <--> API2


    %% =====================================================
    %% NPU 서버 : 시험 3/4 추론 서버
    %% =====================================================

    subgraph NPUAREA["NPU 서버 · YOLO 장비 인식"]
        direction TB

        NPU["Rebellions RBLN-CA25 × 4<br/>YOLO 기반 장비 인식 API<br/>클래스 · confidence · bbox 반환"]
    end

    REQ34 -- "HTTP POST /detect" --> NPU
    NPU -- "탐지 결과 반환" --> RESP34


    %% =====================================================
    %% 공통 주석
    %% =====================================================

    NOTE1["※ 시험 2의 RAG 검색·재순위화·Recall@5 평가·결과 확인은 모두 GPU 서버 2에서 수행"]
    NOTE2["※ 시험 3·4의 전처리·후처리·mAP 평가·응답시간 측정도 GPU 서버 2에서 수행"]
    NOTE3["※ 시험 3·4의 YOLO 모델 추론만 NPU 서버에서 수행"]

    GPU2ALL --- NOTE1
    GPU2ALL --- NOTE2
    NPUAREA --- NOTE3
```

## 흐름 요약

### 시험 1 — 객체 추적 정확도(IDF1)

- 시험 PC에서 SSH로 GPU 서버 1에 접속한다.
- `run_all.sh`을 실행한다.
- 시험 영상 2개를 SAM3 기반 다개체 추적 모듈에 입력한다.
- 추적 결과를 `tracks.csv`로 저장한다.
- 정답 데이터는 추적 모듈이 아니라 IDF1 평가 모듈에만 입력한다.
- TrackEval에서 IoU 0.50 기준으로 통합 IDF1을 계산한다.
- 시험 PC에서 GPU 서버 1의 터미널 결과를 확인한다.

### 시험 2 — AI Agent 질의응답 검색 정확도

**시험 2의 전체 검색 및 평가 프로세스는 GPU 서버 2 내부에서 수행한다.**

- GPU 서버 2에 시험 질의 10개와 문항별 정답 근거 `(manual_id + relevant_pages)`를 준비한다.
- 평가 데이터에서 RAG 검색 API `/search`로 `POST /search` 요청을 수행한다.
- 질의를 전처리하고 BGE-M3 임베딩 검색을 수행한다.
- PostgreSQL + pgvector에서 dataset-v3의 관련 문서 및 청크를 조회한다.
- Cross-Encoder로 검색 후보를 재순위화한다.
- 상위 5개 검색 청크를 반환한다.
- 검색 결과와 문항별 정답 근거를 비교하여 Mean Evidence Recall@5를 계산한다.
- 평가 결과는 **GPU 서버 2의 터미널 또는 결과 파일에서 직접 확인한다.**

### 시험 3 — 이미지 탐지 정확도(mAP@0.5)

**GPU 서버 2가 평가용 클라이언트 역할을 수행한다.**

- 시험 이미지 15장과 정답 데이터를 사용한다.
- GPU 서버 2에서 원본·부분 이미지 생성 등의 전처리를 수행한다.
- `/detect` API를 통해 NPU 서버로 이미지를 전송한다.
- NPU 서버에서 YOLO 장비 인식 추론을 수행한다.
- 클래스, confidence, bbox 결과를 GPU 서버 2로 반환한다.
- GPU 서버 2에서 좌표 복원 및 결과 결합을 수행한다.
- 정답 데이터와 비교하여 mAP@0.5를 계산하고 결과를 출력한다.

### 시험 4 — 클라이언트 응답시간

**시험 3과 동일한 GPU 서버 2의 클라이언트 프로세스에서 측정한다.**

- GPU 서버 2에서 `/detect` 요청과 응답 사이의 클라이언트 응답시간을 측정한다.
- 각 측정값을 별도 로그 파일에 저장한다.
- 총 15회의 응답시간을 기록한다.
- 평가 스크립트가 로그 파일의 15개 측정값을 읽어 평균을 계산한다.
- 평균 클라이언트 응답시간을 GPU 서버 2 터미널에 출력한다.

## 서버 역할 정리

| 서버 | 시험 | 주요 역할 |
|---|---|---|
| 시험 PC | 시험 1 | SSH 접속, 시험 1 스크립트 실행 및 결과 확인 |
| GPU 서버 1 | 시험 1 | SAM3 추적, TrackEval IDF1 평가 |
| **GPU 서버 2** | **시험 2** | **RAG 검색 API, BGE-M3 검색, Cross-Encoder 재순위화, Mean Evidence Recall@5 평가 및 결과 확인** |
| **GPU 서버 2** | **시험 3** | **NPU 요청 클라이언트, 전처리·후처리, mAP@0.5 평가** |
| **GPU 서버 2** | **시험 4** | **클라이언트 응답시간 측정, 로그 저장, 평균 응답시간 계산** |
| NPU 서버 | 시험 3·4 | YOLO 장비 인식 모델 추론 |
