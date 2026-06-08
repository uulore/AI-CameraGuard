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
ROI_Y          = float(os.environ.get("ROI_Y", "0.5"))

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

PERSON_CLASS = 0
CARGO_CLASSES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}


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


def insert_event(conn, event_type, confidence, description, snapshot_path, meta, track_id=None, direction=None):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO events
                (camera_id, event_type, confidence, description, snapshot_path, raw_meta, track_id, direction)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (CAMERA_ID, event_type, confidence, description, snapshot_path, Json(meta), track_id, direction),
        )
    conn.commit()


def save_snapshot(frame, event_type: str) -> str:
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    name = f"{CAMERA_ID}_{event_type}_{ts}.jpg"
    path = SNAPSHOT_DIR / name
    cv2.imwrite(str(path), frame)
    return str(path)


def load_models():
    model_main = YOLO("yolov8n.pt")
    log.info("Main COCO model loaded")

    model_weapon = None
    model_smoke  = None

    weapon_path = Path("models/weapon.pt")
    smoke_path  = Path("models/smoke.pt")

    if weapon_path.exists():
        model_weapon = YOLO(str(weapon_path))
        log.info("Weapon model loaded")
    else:
        log.warning("Weapon model not found — place model at models/weapon.pt to enable")

    if smoke_path.exists():
        model_smoke = YOLO(str(smoke_path))
        log.info("Smoke model loaded")
    else:
        log.warning("Smoke model not found — place model at models/smoke.pt to enable")

    return model_main, model_weapon, model_smoke


def run_detector(model, frame, conf_min):
    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        if float(box.conf[0]) >= conf_min:
            detections.append({
                "cls_id": int(box.cls[0]),
                "label":  model.names[int(box.cls[0])],
                "conf":   round(float(box.conf[0]), 3),
                "bbox":   [round(float(v), 1) for v in box.xyxy[0].tolist()],
            })
    return detections


def main():
    model_main, model_weapon, model_smoke = load_models()
    conn = get_conn()

    track_positions: dict[int, float] = {}
    track_logged:    dict[int, str]   = {}

    while True:
        cap = cv2.VideoCapture(VIDEO_SOURCE if VIDEO_SOURCE != "0" else 0)
        if not cap.isOpened():
            log.error(f"Cannot open video source: {VIDEO_SOURCE}")
            time.sleep(5)
            continue

        fps       = cap.get(cv2.CAP_PROP_FPS) or 25
        height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        roi_px    = int(height * ROI_Y)
        frame_idx = 0

        log.info(f"Stream opened, FPS={fps:.1f}, ROI y={roi_px}px")

        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("End of video, restarting …")
                track_positions.clear()
                track_logged.clear()
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            ts = datetime.now(timezone.utc).isoformat()

            results_main = model_main.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")[0]

            persons = []
            cargo   = []

            if results_main.boxes.id is not None:
                for box in results_main.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    if conf < CONFIDENCE_MIN:
                        continue

                    if cls_id == PERSON_CLASS:
                        track_id = int(box.id[0])
                        center_y = (float(box.xyxy[0][1]) + float(box.xyxy[0][3])) / 2
                        prev_y   = track_positions.get(track_id)

                        persons.append({"track_id": track_id, "conf": round(conf, 3)})

                        if prev_y is not None:
                            crossed_down = prev_y < roi_px <= center_y
                            crossed_up   = prev_y > roi_px >= center_y

                            if crossed_down and track_logged.get(track_id) != "in":
                                track_logged[track_id] = "in"
                                snap = save_snapshot(frame, "person_entered")
                                insert_event(conn, "person_entered", conf,
                                    f"Person {track_id} entered", snap,
                                    {"frame_idx": frame_idx, "timestamp": ts},
                                    track_id=track_id, direction="in")
                                log.info(f"[person_entered] track_id={track_id}")

                            elif crossed_up and track_logged.get(track_id) != "out":
                                track_logged[track_id] = "out"
                                snap = save_snapshot(frame, "person_exited")
                                insert_event(conn, "person_exited", conf,
                                    f"Person {track_id} exited", snap,
                                    {"frame_idx": frame_idx, "timestamp": ts},
                                    track_id=track_id, direction="out")
                                log.info(f"[person_exited] track_id={track_id}")

                        track_positions[track_id] = center_y

                    elif cls_id in CARGO_CLASSES:
                        cargo.append({
                            "label": CARGO_CLASSES[cls_id],
                            "conf":  round(conf, 3),
                            "bbox":  [round(float(v), 1) for v in box.xyxy[0].tolist()],
                        })

            if cargo and persons:
                snap = save_snapshot(frame, "bulk_cargo_exit")
                insert_event(conn, "bulk_cargo_exit",
                    max(c["conf"] for c in cargo),
                    f"{len(cargo)} item(s) carried: {', '.join(c['label'] for c in cargo)}",
                    snap,
                    {"frame_idx": frame_idx, "cargo": cargo, "persons": persons, "timestamp": ts})
                log.info(f"[bulk_cargo_exit] items={[c['label'] for c in cargo]}")
            last_presence_event = None
            if persons:
                count      = len(persons)
                event_type = "crowd_detected" if count >= 5 else "person_detected"
                avg_conf   = sum(p["conf"] for p in persons) / count

                if event_type != last_presence_event:
                    last_presence_event = event_type
                    snap = save_snapshot(frame, event_type)
                    insert_event(conn, event_type, avg_conf,
                        f"{count} person(s) detected", snap,
                        {"frame_idx": frame_idx, "detections": persons, "timestamp": ts})
                    log.info(f"[{event_type}] count={count} | conf={avg_conf:.2f}")

            if model_weapon:
                weapon_detections = run_detector(model_weapon, frame, CONFIDENCE_MIN)
                if weapon_detections:
                    snap = save_snapshot(frame, "weapon_detected")
                    insert_event(conn, "weapon_detected",
                        max(d["conf"] for d in weapon_detections),
                        f"Weapon detected: {weapon_detections[0]['label']}",
                        snap,
                        {"frame_idx": frame_idx, "detections": weapon_detections, "timestamp": ts})
                    log.warning(f"[weapon_detected] {weapon_detections}")

            if model_smoke:
                smoke_detections = run_detector(model_smoke, frame, CONFIDENCE_MIN)
                if smoke_detections:
                    label = smoke_detections[0]["label"]
                    event = "smoke_detected" if "smoke" in label.lower() else "fire_detected"
                    snap  = save_snapshot(frame, event)
                    insert_event(conn, event,
                        max(d["conf"] for d in smoke_detections),
                        f"{label} detected",
                        snap,
                        {"frame_idx": frame_idx, "detections": smoke_detections, "timestamp": ts})
                    log.warning(f"[{event}] {smoke_detections}")

        cap.release()
        time.sleep(1)


if __name__ == "__main__":
    main()