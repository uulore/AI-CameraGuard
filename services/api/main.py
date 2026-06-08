import os
import time
import logging
import threading
import httpx
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from psycopg2.extras import Json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [api] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# Config
DATABASE_URL = os.environ["DATABASE_URL"]
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
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


def insert_event_direct(event_type, confidence, description, meta, camera_id, track_id=None):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO events
                (camera_id, event_type, confidence, description, snapshot_path, raw_meta, track_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (camera_id, event_type, confidence, description, None, Json(meta), track_id),
        )
    conn.commit()


def run_background_checks():
    while True:
        time.sleep(1800)
        try:
            welfare = welfare_check(hours=48)
            for alert in welfare["alerts"]:
                insert_event_direct(
                    event_type  = "welfare_alert",
                    confidence  = 1.0,
                    description = f"Person {alert['track_id']} on {alert['camera_id']} inside for {alert['hours_inside']:.1f}h",
                    meta        = dict(alert),
                    camera_id   = alert["camera_id"],
                    track_id    = alert["track_id"],
                )
                log.warning(f"[welfare_alert] {alert}")

            traffic = traffic_check(window_minutes=60, threshold=5)
            for alert in traffic["alerts"]:
                insert_event_direct(
                    event_type  = "traffic_alert",
                    confidence  = 1.0,
                    description = f"{alert['unique_persons']} unique persons on {alert['camera_id']} in 60min",
                    meta        = dict(alert),
                    camera_id   = alert["camera_id"],
                )
                log.warning(f"[traffic_alert] {alert}")

        except Exception as e:
            log.error(f"Background check error: {e}")


@app.on_event("startup")
def startup():
    global conn
    conn = get_conn()
    thread = threading.Thread(target=run_background_checks, daemon=True)
    thread.start()
    log.info("Background checks started")


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


@app.get("/welfare/check")
def welfare_check(hours: int = 48):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT DISTINCT ON (camera_id, track_id)
                camera_id,
                track_id,
                created_at as entered_at,
                EXTRACT(EPOCH FROM (NOW() - created_at)) / 3600 as hours_inside
            FROM events
            WHERE event_type = 'person_entered'
              AND track_id NOT IN (
                SELECT track_id FROM events
                WHERE event_type = 'person_exited'
                  AND track_id IS NOT NULL
              )
              AND created_at < NOW() - INTERVAL '1 hour'
            ORDER BY camera_id, track_id, created_at DESC
        """)
        rows = cur.fetchall()
    alerts = [r for r in rows if r["hours_inside"] >= hours]
    return {"threshold_hours": hours, "alerts": alerts, "total": len(alerts)}


@app.get("/traffic/check")
def traffic_check(window_minutes: int = 60, threshold: int = 5):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                camera_id,
                COUNT(DISTINCT track_id) as unique_persons,
                MIN(created_at) as window_start,
                MAX(created_at) as window_end
            FROM events
            WHERE event_type = 'person_entered'
              AND created_at > NOW() - (%(minutes)s || ' minutes')::INTERVAL
              AND track_id IS NOT NULL
            GROUP BY camera_id
            HAVING COUNT(DISTINCT track_id) >= %(threshold)s
        """, {"minutes": window_minutes, "threshold": threshold})
        rows = cur.fetchall()
    return {
        "window_minutes": window_minutes,
        "threshold": threshold,
        "alerts": rows,
        "total": len(rows)
    }


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
      event_type    TEXT        -- values: person_detected, crowd_detected, person_entered,
                                --   person_exited, bulk_cargo_exit, weapon_detected,
                                --   smoke_detected, fire_detected, welfare_alert, traffic_alert
      confidence    REAL
      description   TEXT
      snapshot_path TEXT
      raw_meta      JSONB
      track_id      INTEGER     -- ByteTrack person ID
      direction     TEXT        -- 'in' or 'out'
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