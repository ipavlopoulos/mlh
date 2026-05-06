# Machine Learning for Humans

Flask app containing several small machine-learning and LLM demos:

- Alzheimer's image prediction UI backed by a Swin Transformer checkpoint
- Local LLM proxy UI
- Way To School and WalkFree pages using Gemini/Firebase integrations

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

The Swin checkpoint is intentionally not committed. Place `best_swin_nocurr_seed42.pth` in the project root, or set `MODEL_PATH`, before using the `/predict` endpoint.

## Configuration

All deploy-time settings are read from environment variables. See `.env.example` for the full list.

Set `URL_PREFIX=/vmc` when the app is served behind that public path.

Firebase config must be provided as JSON through:

- `WAY_TO_SCHOOL_FIREBASE_CONFIG`
- `WALKFREE_FIREBASE_CONFIG`

Gemini calls are proxied through Flask at `/api/gemini`, so `GEMINI_API_KEY` stays server-side.

## Production

```bash
gunicorn -w 2 -b 0.0.0.0:${PORT:-5000} wsgi:app
```

Health checks:

- `/health`
- `/health/model`

## Checks

```bash
python smoke_tests.py
```
