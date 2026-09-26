# Ability RAG

This project indexes each YAML ability as one LightRAG document and returns the
top-k complete ability records for keyword text. Search uses vector retrieval
only; it does not make a second LLM call to compose an answer.

## Setup

Install from the existing requirements file with uv:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env
```

Put an OpenAI API key in `.env`, then create the index:

```bash
.venv/bin/python ability_rag.py index
```

Indexing uses `gpt-6-luna` for LightRAG's entity extraction and
`text-embedding-3-small` for vectors. Both are configurable in `.env`.
If the embedding model is changed, set `OPENAI_EMBEDDING_DIMENSIONS` to a
dimension supported by that model and rebuild the index.

## Search

```bash
.venv/bin/python ability_rag.py search "powershell phishing attachment" -k 3
```

The result is JSON in rank order. Every item includes its source path and the
complete parsed YAML ability.

Re-running `index` adds newly created abilities. If an indexed ability changed
or was deleted, rebuild so stale vectors cannot survive:

```bash
.venv/bin/python ability_rag.py index --rebuild
```

The index lives in `.rag_storage/` by default. Change `RAG_WORKING_DIR` if a
different location is needed.
