import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import psycopg2
from psycopg2.extras import Json
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [vision] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# Config
DATABASE_URL   = os.environ["DATABASE_URL"]
VIDEO_SOURCE   = os.environ.get("VIDEO_SOURCE", "0")
CAMERA_ID      = os.environ.get("CAMERA_ID", "cam_01")
SNAPSHOT_DIR   = Path(os.environ.get("SNAPSHOT_DIR", "/app/snapshots"))
CONFIDENCE_MIN = float(os.environ.get("CONFIDENCE_MIN", "0.45"))
FRAME_SKIP     = int(os.environ.get("FRAME_SKIP", "5"))

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# COCO class ids to watch
WATCH_CLASSES = {
    0: "person",
}

def classify_event(detections: list[dict]) -> tuple[str, str]:
    persons = [d for d in detections if d["label"] == "person"]
    count   = len(persons)

    if count == 0:
        return "no_person", "No persons detected in frame"
    if count == 1:
        return "person_detected", "1 person detected"
    if count >= 5:
        return "crowd_detected", f"Crowd alert: {count} persons in frame"
    return "person_detected", f"{count} persons detected"


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
    raise RuntimeError("Cannot connect to Postgres after 10 attempts")


def insert_event(conn, event_type, confidence, description, snapshot_path, meta):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO events
                (camera_id, event_type, confidence, description, snapshot_path, raw_meta)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (CAMERA_ID, event_type, confidence, description, snapshot_path, Json(meta)),
        )
    conn.commit()


# Snapshot
def save_snapshot(frame, event_type: str) -> str:
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    name = f"{CAMERA_ID}_{event_type}_{ts}.jpg"
    path = SNAPSHOT_DIR / name
    cv2.imwrite(str(path), frame)
    return str(path)


# Main loop
def main():
    model = YOLO("yolov8n.pt")
    log.info(f"YOLO loaded, watching: {VIDEO_SOURCE}")

    conn = get_conn()

    while True:
        cap = cv2.VideoCapture(VIDEO_SOURCE if VIDEO_SOURCE != "0" else 0)
        if not cap.isOpened():
            log.error(f"Cannot open video source: {VIDEO_SOURCE}")
            time.sleep(5)
            continue

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_idx  = 0
        last_event = None

        log.info(f"Stream opened, FPS={fps:.1f}")

        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("End of video, restarting …")
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            results    = model(frame, verbose=False)[0]
            detections = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in WATCH_CLASSES:
                    continue
                detections.append({
                    "label": WATCH_CLASSES[cls_id],
                    "conf":  round(float(box.conf[0]), 3),
                    "bbox":  [round(float(v), 1) for v in box.xyxy[0].tolist()],
                })

            detections = [d for d in detections if d["conf"] >= CONFIDENCE_MIN]
            event_type, description = classify_event(detections)

            if event_type == "no_person":
                continue
            if event_type == last_event:
                continue
            last_event = event_type

            avg_conf = (
                sum(d["conf"] for d in detections) / len(detections)
                if detections else 0.0
            )

            snapshot_path = save_snapshot(frame, event_type)
            insert_event(
                conn,
                event_type    = event_type,
                confidence    = avg_conf,
                description   = description,
                snapshot_path = snapshot_path,
                meta = {
                    "frame_idx":  frame_idx,
                    "detections": detections,
                    "timestamp":  datetime.now(timezone.utc).isoformat(),
                },
            )
            log.info(f"[{event_type}] {description} | conf={avg_conf:.2f} | snap={snapshot_path}")

        cap.release()
        time.sleep(1)


if __name__ == "__main__":
    main()