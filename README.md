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
export GEMINI_API_KEY="your-key"
python app.py
```

The Swin checkpoint is intentionally not committed. Place `best_swin_nocurr_seed42.pth` in the project root before using the `/predict` endpoint.
