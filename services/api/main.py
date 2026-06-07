import os
import time
import logging
import httpx
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [api] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# Config
DATABASE_URL = os.environ["DATABASE_URL"]
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://ollama:11434")
LLM_MODEL    = os.environ.get("LLM_MODEL", "qwen3:8b")

app = FastAPI(title="AI Camera Guard API")


# DB
def get_conn():
    for attempt in range(10):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            log.info("Connected to Postgres")
            return conn
        except psycopg2.OperationalError:
            log.warning(f"Postgres not ready, retry {attempt+1}/10 …")
            time.sleep(3)
    raise RuntimeError("Cannot connect to Postgres")

conn = None

@app.on_event("startup")
def startup():
    global conn
    conn = get_conn()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/events")
def get_events(limit: int = 50, event_type: str = None):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if event_type:
            cur.execute(
                "SELECT * FROM events WHERE event_type = %s ORDER BY created_at DESC LIMIT %s",
                (event_type, limit)
            )
        else:
            cur.execute(
                "SELECT * FROM events ORDER BY created_at DESC LIMIT %s",
                (limit,)
            )
        return cur.fetchall()


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sql_used: str | None = None


def query_events_db(natural_query: str) -> tuple[str, str]:
    schema = """
    Table: events
    Columns:
      id            SERIAL PRIMARY KEY
      created_at    TIMESTAMPTZ
      camera_id     TEXT
      event_type    TEXT
      confidence    REAL
      description   TEXT
      snapshot_path TEXT
      raw_meta      JSONB
    """

    sql_prompt = f"""You are a PostgreSQL expert. Given this schema:
{schema}

Convert the following question to a single valid PostgreSQL SELECT query.
Return ONLY the SQL query, no explanation, no markdown, no semicolon.

Question: {natural_query}"""

    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": sql_prompt, "stream": False, "keep_alive": "10m", "options": {"think": False}},
        timeout=300.0
    )
    response.raise_for_status()
    sql = response.json()["response"].strip().rstrip(";")
    log.info(f"Generated SQL: {sql}")

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    except Exception as e:
        log.error(f"SQL execution failed: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"SQL error: {e}")

    answer_prompt = f"""You are a security monitoring assistant.
The user asked: "{natural_query}"

Database returned these results:
{rows}

Give a clear, concise answer in the same language the user asked.
Focus on facts: times, counts, event types. Be brief."""

    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": answer_prompt, "stream": False, "keep_alive": "10m", "options": {"think": False}},
        timeout=300.0
    )
    response.raise_for_status()
    answer = response.json()["response"].strip()

    return answer, sql


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    log.info(f"Chat request: {req.message}")
    answer, sql = query_events_db(req.message)
    return ChatResponse(answer=answer, sql_used=sql)