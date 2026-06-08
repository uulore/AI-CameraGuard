CREATE TABLE IF NOT EXISTS events (
    id            SERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    camera_id     TEXT NOT NULL DEFAULT 'cam_01',
    event_type    TEXT NOT NULL,
    confidence    REAL,
    description   TEXT,
    snapshot_path TEXT,
    raw_meta      JSONB,
    track_id      INTEGER,
    direction     TEXT
);

CREATE INDEX idx_events_created_at ON events(created_at DESC);
CREATE INDEX idx_events_event_type  ON events(event_type);
CREATE INDEX idx_events_camera_id   ON events(camera_id);
CREATE INDEX idx_events_track_id    ON events(track_id);