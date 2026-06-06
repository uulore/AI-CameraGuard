# AI Camera Guard

Backend for automated security camera monitoring.
Replaces manual surveillance with an AI layer that detects events in real time,
logs them as text, and answers natural language queries about camera history.

## Architecture

| Service  | Role                                      |
|----------|-------------------------------------------|
| vision   | Reads video stream, runs YOLO, logs events |
| api      | FastAPI — REST + /chat endpoint           |
| llm      | Ollama + Qwen3 — Text-to-SQL + answers    |
| postgres | Event log database                        |
| storage  | Snapshot frames (mounted volume)          |

## Quick Start

```bash
# 1. Pull Qwen3 model (once)
docker compose run --rm ollama ollama pull qwen3:8b

# 2. Put a demo video in ./videos/
cp your_video.mp4 videos/demo.mp4

# 3. Start everything
docker compose up --build
```

## Usage

```bash
# Check recent events
curl http://localhost:8000/events

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many people were detected in the last hour?"}'
```

## Event Types

| event_type       | Trigger                        |
|------------------|--------------------------------|
| person_detected  | 1–4 persons in frame           |
| crowd_detected   | 5+ persons in frame            |

## Stack

- [YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [Ollama](https://ollama.ai) + Qwen3:8b — local LLM
- FastAPI + PostgreSQL
- Runs fully local — no cloud required