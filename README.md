# Ability Search

Local top-k search for YAML ability records using SQLite FTS5 and BM25. It has
no model, API, service, or network dependency. YAML files remain the source of
truth, and their directory layout is ignored.

## Setup

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

## Index

Synchronize the SQLite index with every `.yml` and `.yaml` file beneath
`abilities/`:

```bash
.venv/bin/python ability_search.py index
```

New and changed records are indexed, and records whose YAML was deleted are
removed. Ability IDs must be unique across the corpus.

## Search

```bash
.venv/bin/python ability_search.py search "powershell phishing" -k 5
```

Keywords are combined with OR and ranked with BM25. Put a phrase in quotes when
its words must be adjacent:

```bash
.venv/bin/python ability_search.py search '"local account" powershell'
```

Filter by the exact YAML tactic, ATT&CK technique ID, or technique name:

```bash
.venv/bin/python ability_search.py search "attachment" \
  --tactic initial-access \
  --technique T1566.001
```

`--tactic` and `--technique` may each be repeated. Repeated values within one
filter are ORed; tactic and technique filters are ANDed together. The query is
optional, so filters can be used alone:

```bash
.venv/bin/python ability_search.py search --tactic reconnaissance
```

List the available filter values and counts:

```bash
.venv/bin/python ability_search.py facets
```

All commands output JSON. Use `--database PATH` on any command to select a
different index file. Python code can import `AbilitySearch` from
`ability_search.py` and call `index()`, `search()`, or `facets()` directly.
