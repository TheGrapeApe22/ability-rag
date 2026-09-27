"""Local full-text search for YAML ability records.

SQLite FTS5 provides the inverted index and BM25 ranking. The YAML files remain
the source of truth; directory names have no effect on tactics or techniques.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


DEFAULT_DATABASE = Path(".ability_search.sqlite")
SCHEMA_VERSION = 1
SEARCH_COLUMNS = (
    "name",
    "description",
    "tactic",
    "technique_id",
    "technique_name",
    "platforms",
    "commands",
)
# Name, description, tactic, technique ID/name, platforms, and commands.
BM25_WEIGHTS = (10.0, 5.0, 4.0, 8.0, 8.0, 3.0, 1.0)


@dataclass(frozen=True)
class AbilityRecord:
    ability_id: str
    source_path: str
    digest: str
    name: str
    description: str
    tactic: str
    technique_id: str
    technique_name: str
    platforms: str
    commands: str
    record: dict[str, Any]


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _ability_from_mapping(item: dict[str, Any], source: Path) -> AbilityRecord:
    ability_id = _text(item.get("id"))
    if not ability_id:
        raise ValueError(f"Ability in {source} has no id")

    technique = item.get("technique") or {}
    if not isinstance(technique, dict):
        raise ValueError(f"Ability {ability_id!r} in {source} has an invalid technique")

    platforms_value = item.get("platforms") or {}
    if not isinstance(platforms_value, dict):
        raise ValueError(f"Ability {ability_id!r} in {source} has invalid platforms")

    platform_terms: list[str] = []
    commands: list[str] = []
    for platform, executors in platforms_value.items():
        platform_terms.append(_text(platform))
        if not isinstance(executors, dict):
            continue
        for executor_name, executor in executors.items():
            platform_terms.append(_text(executor_name))
            if isinstance(executor, dict):
                for field in ("command", "cleanup"):
                    value = _text(executor.get(field))
                    if value:
                        commands.append(value)

    canonical = yaml.safe_dump(item, allow_unicode=True, sort_keys=True)
    return AbilityRecord(
        ability_id=ability_id,
        source_path=source.as_posix(),
        digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        name=_text(item.get("name")),
        description=_text(item.get("description")),
        tactic=_text(item.get("tactic")),
        technique_id=_text(technique.get("attack_id")),
        technique_name=_text(technique.get("name")),
        platforms=" ".join(term for term in platform_terms if term),
        commands="\n".join(commands),
        record=item,
    )


def load_abilities(directory: Path) -> list[AbilityRecord]:
    """Load ability mappings recursively, without deriving data from folders."""
    if not directory.is_dir():
        raise ValueError(f"Abilities directory does not exist: {directory}")

    paths = sorted((*directory.rglob("*.yml"), *directory.rglob("*.yaml")))
    records: list[AbilityRecord] = []
    seen: dict[str, str] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        items = document if isinstance(document, list) else [document]
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Expected an ability mapping in {path}")
            record = _ability_from_mapping(item, path)
            if record.ability_id in seen:
                raise ValueError(
                    f"Duplicate ability id {record.ability_id!r} in "
                    f"{path} and {seen[record.ability_id]}"
                )
            seen[record.ability_id] = path.as_posix()
            records.append(record)
    return records


def _normalize_filter(value: str) -> str:
    return value.strip().casefold()


def _fts_query(query: str) -> str:
    """Convert plain user text to a safe any-keyword FTS5 expression."""
    # Quoted text stays grouped as a phrase; other whitespace-delimited values
    # become individual terms. Every value is quoted, so FTS operators in user
    # input are treated as text rather than executable query syntax.
    parts = re.findall(r'"([^"]+)"|(\S+)', query.strip())
    terms = [quoted or bare for quoted, bare in parts if quoted or bare]
    escaped = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms]
    return " OR ".join(escaped)


class AbilitySearch:
    """Persistent SQLite-backed index and search interface."""

    def __init__(self, database: Path | str = DEFAULT_DATABASE):
        self.database = Path(database)

    def _connect(self, create_parent: bool = False) -> sqlite3.Connection:
        if create_parent:
            self.database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS abilities (
                rowid INTEGER PRIMARY KEY,
                ability_id TEXT NOT NULL UNIQUE,
                source_path TEXT NOT NULL,
                digest TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                tactic TEXT NOT NULL,
                tactic_norm TEXT NOT NULL,
                technique_id TEXT NOT NULL,
                technique_id_norm TEXT NOT NULL,
                technique_name TEXT NOT NULL,
                technique_name_norm TEXT NOT NULL,
                platforms TEXT NOT NULL,
                commands TEXT NOT NULL,
                record_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS abilities_tactic_idx
                ON abilities(tactic_norm);
            CREATE INDEX IF NOT EXISTS abilities_technique_id_idx
                ON abilities(technique_id_norm);
            CREATE INDEX IF NOT EXISTS abilities_technique_name_idx
                ON abilities(technique_name_norm);

            CREATE VIRTUAL TABLE IF NOT EXISTS abilities_fts USING fts5(
                name,
                description,
                tactic,
                technique_id,
                technique_name,
                platforms,
                commands,
                content='abilities',
                content_rowid='rowid',
                tokenize="unicode61 remove_diacritics 2 tokenchars '-_.'"
            );

            CREATE TRIGGER IF NOT EXISTS abilities_ai AFTER INSERT ON abilities BEGIN
                INSERT INTO abilities_fts(
                    rowid, name, description, tactic, technique_id,
                    technique_name, platforms, commands
                ) VALUES (
                    new.rowid, new.name, new.description, new.tactic,
                    new.technique_id, new.technique_name, new.platforms,
                    new.commands
                );
            END;

            CREATE TRIGGER IF NOT EXISTS abilities_ad AFTER DELETE ON abilities BEGIN
                INSERT INTO abilities_fts(
                    abilities_fts, rowid, name, description, tactic,
                    technique_id, technique_name, platforms, commands
                ) VALUES (
                    'delete', old.rowid, old.name, old.description, old.tactic,
                    old.technique_id, old.technique_name, old.platforms,
                    old.commands
                );
            END;

            CREATE TRIGGER IF NOT EXISTS abilities_au AFTER UPDATE ON abilities BEGIN
                INSERT INTO abilities_fts(
                    abilities_fts, rowid, name, description, tactic,
                    technique_id, technique_name, platforms, commands
                ) VALUES (
                    'delete', old.rowid, old.name, old.description, old.tactic,
                    old.technique_id, old.technique_name, old.platforms,
                    old.commands
                );
                INSERT INTO abilities_fts(
                    rowid, name, description, tactic, technique_id,
                    technique_name, platforms, commands
                ) VALUES (
                    new.rowid, new.name, new.description, new.tactic,
                    new.technique_id, new.technique_name, new.platforms,
                    new.commands
                );
            END;
            """
        )
        existing = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        if existing and int(existing["value"]) != SCHEMA_VERSION:
            raise ValueError(
                "The search database uses an incompatible schema. Delete it and re-index."
            )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    def index(self, abilities_dir: Path | str = "abilities") -> dict[str, int]:
        records = load_abilities(Path(abilities_dir))
        by_id = {record.ability_id: record for record in records}

        with self._connect(create_parent=True) as connection:
            self._create_schema(connection)
            existing = {
                row["ability_id"]: (row["digest"], row["source_path"])
                for row in connection.execute(
                    "SELECT ability_id, digest, source_path FROM abilities"
                )
            }
            inserted = sum(ability_id not in existing for ability_id in by_id)
            updated = sum(
                ability_id in existing
                and existing[ability_id] != (record.digest, record.source_path)
                for ability_id, record in by_id.items()
            )
            unchanged = len(records) - inserted - updated
            stale_ids = set(existing) - set(by_id)

            connection.executemany(
                """
                INSERT INTO abilities (
                    ability_id, source_path, digest, name, description,
                    tactic, tactic_norm, technique_id, technique_id_norm,
                    technique_name, technique_name_norm, platforms, commands,
                    record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ability_id) DO UPDATE SET
                    source_path=excluded.source_path,
                    digest=excluded.digest,
                    name=excluded.name,
                    description=excluded.description,
                    tactic=excluded.tactic,
                    tactic_norm=excluded.tactic_norm,
                    technique_id=excluded.technique_id,
                    technique_id_norm=excluded.technique_id_norm,
                    technique_name=excluded.technique_name,
                    technique_name_norm=excluded.technique_name_norm,
                    platforms=excluded.platforms,
                    commands=excluded.commands,
                    record_json=excluded.record_json
                WHERE abilities.digest != excluded.digest
                   OR abilities.source_path != excluded.source_path
                """,
                [
                    (
                        record.ability_id,
                        record.source_path,
                        record.digest,
                        record.name,
                        record.description,
                        record.tactic,
                        _normalize_filter(record.tactic),
                        record.technique_id,
                        _normalize_filter(record.technique_id),
                        record.technique_name,
                        _normalize_filter(record.technique_name),
                        record.platforms,
                        record.commands,
                        json.dumps(record.record, ensure_ascii=False, default=str),
                    )
                    for record in records
                ],
            )
            if stale_ids:
                placeholders = ",".join("?" for _ in stale_ids)
                connection.execute(
                    f"DELETE FROM abilities WHERE ability_id IN ({placeholders})",
                    tuple(stale_ids),
                )

        return {
            "inserted": inserted,
            "updated": updated,
            "unchanged": unchanged,
            "deleted": len(stale_ids),
            "total": len(records),
        }

    def _require_index(self) -> None:
        if not self.database.is_file():
            raise ValueError(f"Search index not found: {self.database}. Run index first.")

    def search(
        self,
        query: str = "",
        *,
        top_k: int = 10,
        tactics: Sequence[str] = (),
        techniques: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._require_index()

        fts_query = _fts_query(query)
        clauses: list[str] = []
        parameters: list[Any] = []
        normalized_tactics = [_normalize_filter(value) for value in tactics if value.strip()]
        normalized_techniques = [
            _normalize_filter(value) for value in techniques if value.strip()
        ]
        if normalized_tactics:
            marks = ",".join("?" for _ in normalized_tactics)
            clauses.append(f"a.tactic_norm IN ({marks})")
            parameters.extend(normalized_tactics)
        if normalized_techniques:
            marks = ",".join("?" for _ in normalized_techniques)
            clauses.append(
                f"(a.technique_id_norm IN ({marks}) "
                f"OR a.technique_name_norm IN ({marks}))"
            )
            parameters.extend(normalized_techniques)
            parameters.extend(normalized_techniques)

        if fts_query:
            weights = ", ".join(str(weight) for weight in BM25_WEIGHTS)
            sql = f"""
                SELECT a.source_path, a.record_json,
                       bm25(abilities_fts, {weights}) AS score
                FROM abilities_fts
                JOIN abilities AS a ON a.rowid = abilities_fts.rowid
                WHERE abilities_fts MATCH ?
            """
            query_parameters: list[Any] = [fts_query]
            if clauses:
                sql += " AND " + " AND ".join(clauses)
            sql += " ORDER BY score, a.name COLLATE NOCASE LIMIT ?"
            query_parameters.extend(parameters)
            query_parameters.append(top_k)
        else:
            sql = "SELECT source_path, record_json, NULL AS score FROM abilities AS a"
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY a.name COLLATE NOCASE LIMIT ?"
            query_parameters = [*parameters, top_k]

        with self._connect() as connection:
            rows = connection.execute(sql, query_parameters).fetchall()
        return [
            {
                "rank": rank,
                "score": None if row["score"] is None else -float(row["score"]),
                "source": row["source_path"],
                "ability": json.loads(row["record_json"]),
            }
            for rank, row in enumerate(rows, start=1)
        ]

    def facets(self) -> dict[str, list[dict[str, Any]]]:
        """Return available exact filter values and their record counts."""
        self._require_index()
        with self._connect() as connection:
            tactics = connection.execute(
                """
                SELECT tactic AS value, COUNT(*) AS count
                FROM abilities
                GROUP BY tactic_norm
                ORDER BY tactic COLLATE NOCASE
                """
            ).fetchall()
            techniques = connection.execute(
                """
                SELECT technique_id AS attack_id, technique_name AS name,
                       COUNT(*) AS count
                FROM abilities
                GROUP BY technique_id_norm, technique_name_norm
                ORDER BY technique_id COLLATE NOCASE, technique_name COLLATE NOCASE
                """
            ).fetchall()
        return {
            "tactics": [dict(row) for row in tactics],
            "techniques": [dict(row) for row in techniques],
        }


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database", type=Path, default=DEFAULT_DATABASE, help="SQLite index path"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index and search YAML abilities locally with SQLite FTS5."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser("index", help="Synchronize the search index")
    _add_common_paths(index_parser)
    index_parser.add_argument("--abilities-dir", type=Path, default=Path("abilities"))

    search_parser = commands.add_parser("search", help="Search indexed abilities")
    _add_common_paths(search_parser)
    search_parser.add_argument(
        "query", nargs="?", default="", help="Keywords; quote text for a phrase"
    )
    search_parser.add_argument("-k", "--top-k", type=int, default=10)
    search_parser.add_argument(
        "--tactic", action="append", default=[], help="Exact tactic; repeat for OR"
    )
    search_parser.add_argument(
        "--technique",
        action="append",
        default=[],
        help="Exact ATT&CK ID or technique name; repeat for OR",
    )

    facets_parser = commands.add_parser("facets", help="List tactic and technique filters")
    _add_common_paths(facets_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    search = AbilitySearch(args.database)
    try:
        if args.command == "index":
            result: Any = search.index(args.abilities_dir)
        elif args.command == "search":
            result = search.search(
                args.query,
                top_k=args.top_k,
                tactics=args.tactic,
                techniques=args.technique,
            )
        else:
            result = search.facets()
    except (OSError, sqlite3.Error, ValueError, yaml.YAMLError) as error:
        parser.exit(2, f"error: {error}\n")

    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
