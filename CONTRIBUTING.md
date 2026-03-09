# Contributing

## Setup

```bash
git clone https://github.com/CarlAmine/multimodal-rag-pipeline.git
cd multimodal-rag-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Start Qdrant locally:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Running Tests

```bash
pytest tests/ -v
```

## Conventions

- Follow PEP 8; use type hints on public functions.
- New modules belong in the appropriate package (`ingestion/`, `retrieval/`, `generation/`, `evaluation/`).
- Add a `tests/test_<module>.py` for any module with non-trivial logic.
- Commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.

## Submitting Changes

1. Fork and create a branch: `git checkout -b feat/my-feature`
2. Make changes and run tests.
3. Open a PR against `main` with a clear description.
