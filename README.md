# 📘 RAG Server

**RAG Server**는 사용자가 업로드한 PDF/TXT/MD 문서를 벡터화하여 저장하고, 자연어 질문에 대해 문서 기반 답변을 생성하는 LLM 기반 백엔드 서비스입니다.

이 프로젝트는 문서 파싱, 텍스트 청킹, 임베딩 생성, 로컬 Chroma 벡터 DB 저장, 그리고 LLM 질의응답까지 RAG 파이프라인을 직접 구성한 구조를 갖습니다.

---

## 🚀 핵심 기능

- PDF, TXT, Markdown 파일 업로드 지원
- `pdfplumber` 기반 문서 파싱 및 텍스트 추출
- `TextSplitter`를 이용한 오버랩 기반 청킹
- `Sentence-Transformers` 로컬 임베딩(`all-MiniLM-L6-v2`) 지원
- `ChromaDB` 로컬 영구 벡터 저장 및 유사도 검색
- `FastAPI` 기반 REST API
- OpenAI / Gemini / Upstage / Ollama LLM 연동
- 세션 기반 대화 히스토리 관리
- 파일 크기 제한 및 확장자 검증

---

## 🧠 아키텍처 요약

1. **문서 업로드**
   - 클라이언트가 PDF/TXT/MD 파일을 `/documents/upload`로 전송
   - 서버는 파일을 저장하고 문서 로더로 텍스트 추출
   - 추출된 텍스트를 청크로 나누고 임베딩 생성
   - 청크와 임베딩을 ChromaDB에 저장

2. **질의응답**
   - `/chat/query`로 사용자 질문 수신
   - 질문 임베딩 생성
   - ChromaDB에서 유사 문서 청크 검색
   - 검색된 문서를 컨텍스트로 LLM에 전달하여 답변 생성
   - 답변과 출처 메타데이터 반환

---

## 🛠️ 기술 스택

- Backend: `FastAPI`, `Pydantic`
- Vector DB: `ChromaDB`
- Embedding: `sentence-transformers` (`all-MiniLM-L6-v2`)
- PDF Parsing: `pdfplumber`
- LLM Clients: `openai` (OpenAI / Gemini / Upstage / Ollama 호환)
- Storage: 로컬 파일 시스템

---

## 📡 API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/` | 서비스 정보 반환 |
| `GET` | `/health/` | 헬스 체크 |
| `POST` | `/documents/upload` | 문서 업로드 및 처리 |
| `GET` | `/documents` | 등록된 문서 목록 조회 |
| `GET` | `/documents/{document_id}` | 단일 문서 메타데이터 조회 |
| `POST` | `/chat/query` | 질문 기반 답변 생성 |

> 실행 후 `http://localhost:8000/docs`에서 Swagger UI를 통해 API를 확인할 수 있습니다.

---

## ⚙️ 환경 설정

`app/core/config.py`에서 기본 설정값을 관리하며, `.env` 파일로 덮어쓸 수 있습니다.

주요 설정값:

- `LLM_PROVIDER`: `openai`, `upstage`, `gemini`, `ollama`
- `EMBEDDING_PROVIDER`: `huggingface`, `openai`
- `EMBEDDING_MODEL`: 기본 `all-MiniLM-L6-v2`
- `VECTOR_DB_PATH`: 로컬 Chroma 저장 경로 (`app/db/vector_db`)
- `UPLOAD_DIR`: 업로드 파일 저장 경로 (`data/uploads`)
- `MAX_UPLOAD_SIZE_MB`: 업로드 허용 최대 용량

---

## 🚀 실행 방법

Windows PowerShell 기준:

```powershell
cd C:\Users\ejs14\IdeaProjects\rag-server
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env` 파일에 필요한 값을 추가합니다:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
# 또는
# OPENAI_API_KEY=your_openai_api_key_here
# UPSTAGE_API_KEY=your_upstage_api_key_here
```

서버 실행:

```powershell
python -m uvicorn app.main:app --reload
```

필요 시 프론트엔드도 별도로 실행할 수 있습니다:

```powershell
cd frontend
npm install
npm run dev
```

---

## 📁 주요 파일 구조

- `app/main.py`: FastAPI 앱 엔트리 포인트
- `app/core/config.py`: 설정 및 환경 변수
- `app/api/routes/`: API 라우터 정의
  - `upload.py`, `documents.py`, `chat.py`, `health.py`
- `app/services/`: RAG 비즈니스 로직
  - `rag_pipeline.py`, `document_loader.py`, `text_splitter.py`, `embedding.py`, `vector_store.py`, `llm.py`, `conversation.py`
- `app/db/vector_db/`: ChromaDB 로컬 페르시스턴트 저장소
- `data/uploads/`: 업로드한 원본 문서 저장소

---

## 💡 참고

이 백엔드 서버는 프레임워크에 깊이 의존하지 않고, 문서 분해부터 벡터 검색, LLM 질의응답까지 자체 파이프라인을 구성한 점이 특징입니다.

필요한 경우 `LLM_PROVIDER`와 `EMBEDDING_PROVIDER` 설정만 변경하여 다양한 모델을 손쉽게 테스트할 수 있습니다.
