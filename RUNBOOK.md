# VisionRAG Runbook

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure environment

Copy `.env.example` to `.env` and set:
- `POSTGRES_DSN`
- `DEFAULT_S3_BUCKET` (or pass per request)
- `AWS_REGION`
- `GEMINI_API_KEY` (optional for retrieval-only mode)

## 3. Apply database migrations

```bash
python migrate.py
```

## 4. Start services

Query API:

```bash
python run_api.py
```

Worker:

```bash
python run_worker.py
```

## 5. Example API usage

Queue a document:

```bash
curl -X POST http://localhost:8000/v1/documents \
  -H "Content-Type: application/json" \
  -d "{\"s3_bucket\":\"my-bucket\",\"s3_key\":\"docs/my.pdf\"}"
```

Query:

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What does the revenue chart show?\",\"generate_answer\":true}"
```

