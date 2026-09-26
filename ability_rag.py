"""Index and retrieve ability YAML records with LightRAG.

The retrieval path deliberately uses LightRAG's ``naive`` mode.  That mode
performs vector similarity search and returns structured chunks without asking
an LLM to write an answer, so a top-k search is cheap and deterministic.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import yaml
from dotenv import load_dotenv


DEFAULT_LLM_MODEL = "gpt-6-luna"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MANIFEST_NAME = "ability_manifest.json"
ABILITY_ID_PATTERN = re.compile(r"^ABILITY_ID:\s*(\S+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Settings:
    abilities_dir: Path
    working_dir: Path
    llm_model: str
    embedding_model: str
    embedding_dimensions: int

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            abilities_dir=Path(os.getenv("ABILITIES_DIR", "abilities")),
            working_dir=Path(os.getenv("RAG_WORKING_DIR", ".rag_storage")),
            llm_model=os.getenv("OPENAI_LLM_MODEL", DEFAULT_LLM_MODEL),
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            ),
            embedding_dimensions=int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")),
        )


@dataclass(frozen=True)
class AbilityRecord:
    ability_id: str
    source_path: Path
    data: dict[str, Any]
    digest: str


def _yaml_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Abilities directory does not exist: {directory}")
    return sorted((*directory.rglob("*.yml"), *directory.rglob("*.yaml")))


def load_abilities(directory: Path) -> list[AbilityRecord]:
    """Load one or more ability mappings from every YAML file."""
    records: list[AbilityRecord] = []
    seen_ids: dict[str, Path] = {}

    for path in _yaml_paths(directory):
        with path.open("r", encoding="utf-8") as stream:
            parsed = yaml.safe_load(stream)

        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Expected an ability mapping in {path}")
            ability_id = str(item.get("id", "")).strip()
            if not ability_id:
                raise ValueError(f"Ability in {path} has no id")
            if ability_id in seen_ids:
                raise ValueError(
                    f"Duplicate ability id {ability_id!r} in {path} and "
                    f"{seen_ids[ability_id]}"
                )
            seen_ids[ability_id] = path
            canonical = json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            digest_input = f"{path.as_posix()}\n{canonical}"
            records.append(
                AbilityRecord(
                    ability_id=ability_id,
                    source_path=path,
                    data=item,
                    digest=hashlib.sha256(digest_input.encode("utf-8")).hexdigest(),
                )
            )
    return records


def ability_to_document(record: AbilityRecord) -> str:
    """Create a search-oriented, single-chunk document for one ability."""
    ability = record.data
    technique = ability.get("technique") or {}
    platforms = ability.get("platforms") or {}
    platform_names = ", ".join(sorted(str(name) for name in platforms))

    commands: list[str] = []
    for executors in platforms.values():
        if not isinstance(executors, dict):
            continue
        for executor_name, executor in executors.items():
            if isinstance(executor, dict) and executor.get("command"):
                commands.append(f"{executor_name}: {executor['command']}")

    fields = [
        f"ABILITY_ID: {record.ability_id}",
        f"NAME: {ability.get('name', '')}",
        f"DESCRIPTION: {ability.get('description', '')}",
        f"TACTIC: {ability.get('tactic', '')}",
        f"TECHNIQUE_ID: {technique.get('attack_id', '') if isinstance(technique, dict) else ''}",
        f"TECHNIQUE_NAME: {technique.get('name', '') if isinstance(technique, dict) else ''}",
        f"PLATFORMS: {platform_names}",
        "COMMANDS:\n" + "\n".join(commands),
        "FULL_RECORD:\n" + yaml.safe_dump(ability, sort_keys=False, allow_unicode=True),
    ]
    return "\n".join(fields)


def _manifest_for(records: Iterable[AbilityRecord]) -> dict[str, str]:
    return {record.ability_id: record.digest for record in records}


def _read_manifest(working_dir: Path) -> dict[str, str]:
    path = working_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(digest, str)
        for key, digest in value.items()
    ):
        raise ValueError(f"Invalid index manifest: {path}")
    return value


def _write_manifest(working_dir: Path, manifest: dict[str, str]) -> None:
    working_dir.mkdir(parents=True, exist_ok=True)
    path = working_dir / MANIFEST_NAME
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _reset_index(working_dir: Path) -> None:
    resolved = working_dir.resolve()
    protected = {Path("/").resolve(), Path.cwd().resolve(), Path.home().resolve()}
    if resolved in protected:
        raise ValueError(f"Refusing to remove unsafe RAG_WORKING_DIR: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add a key."
        )


def _make_embedding_func(model: str, dimensions: int):
    # Import lazily so parsing/testing local YAML never initializes an API client.
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    async def embed(texts: list[str], **kwargs: Any):
        context = kwargs.get("context", "document")
        return await openai_embed.func(
            texts=texts,
            model=model,
            embedding_dim=dimensions,
            context=context,
        )

    return EmbeddingFunc(
        embedding_dim=dimensions,
        max_token_size=8192,
        func=embed,
        model_name=model,
        supports_asymmetric=True,
    )


@asynccontextmanager
async def open_rag(settings: Settings) -> AsyncIterator[Any]:
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete

    rag = LightRAG(
        working_dir=str(settings.working_dir),
        llm_model_func=openai_complete,
        llm_model_name=settings.llm_model,
        embedding_func=_make_embedding_func(
            settings.embedding_model, settings.embedding_dimensions
        ),
        # Every ability is far below this size, keeping each record in one chunk.
        chunk_token_size=8000,
        chunk_overlap_token_size=0,
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()
    try:
        yield rag
    finally:
        await rag.finalize_storages()


async def index_abilities(settings: Settings, rebuild: bool = False) -> dict[str, Any]:
    _require_api_key()
    records = load_abilities(settings.abilities_dir)
    current_manifest = _manifest_for(records)

    if rebuild:
        _reset_index(settings.working_dir)
    previous_manifest = _read_manifest(settings.working_dir)

    changed = sorted(
        ability_id
        for ability_id, digest in previous_manifest.items()
        if current_manifest.get(ability_id) != digest
    )
    if changed:
        preview = ", ".join(changed[:5])
        suffix = " ..." if len(changed) > 5 else ""
        raise ValueError(
            "The corpus changed or removed already-indexed abilities "
            f"({preview}{suffix}). Run index again with --rebuild."
        )

    pending = [
        record for record in records if record.ability_id not in previous_manifest
    ]
    if pending:
        documents = [ability_to_document(record) for record in pending]
        ids = [f"ability-{record.ability_id}" for record in pending]
        paths = [record.source_path.as_posix() for record in pending]
        async with open_rag(settings) as rag:
            await rag.ainsert(documents, ids=ids, file_paths=paths)
        _write_manifest(settings.working_dir, current_manifest)
    elif not (settings.working_dir / MANIFEST_NAME).exists():
        _write_manifest(settings.working_dir, current_manifest)

    return {
        "indexed": len(pending),
        "total": len(records),
        "working_dir": str(settings.working_dir),
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
    }


def _ability_id_from_chunk(chunk: dict[str, Any]) -> str | None:
    match = ABILITY_ID_PATTERN.search(str(chunk.get("content", "")))
    return match.group(1) if match else None


def ranked_abilities(
    raw_result: dict[str, Any], records: Iterable[AbilityRecord], limit: int
) -> list[dict[str, Any]]:
    by_id = {record.ability_id: record for record in records}
    chunks = raw_result.get("data", {}).get("chunks", [])
    results: list[dict[str, Any]] = []
    returned: set[str] = set()
    for chunk in chunks:
        ability_id = _ability_id_from_chunk(chunk)
        if not ability_id or ability_id in returned or ability_id not in by_id:
            continue
        returned.add(ability_id)
        record = by_id[ability_id]
        results.append(
            {
                "rank": len(results) + 1,
                "source": record.source_path.as_posix(),
                "ability": record.data,
            }
        )
        if len(results) == limit:
            break
    return results


async def search_abilities(
    settings: Settings, keywords: str, top_k: int
) -> list[dict[str, Any]]:
    _require_api_key()
    if top_k < 1:
        raise ValueError("top-k must be at least 1")
    records = load_abilities(settings.abilities_dir)
    if not (settings.working_dir / MANIFEST_NAME).exists():
        raise ValueError("No ability index found. Run the index command first.")

    from lightrag import QueryParam

    async with open_rag(settings) as rag:
        raw_result = await rag.aquery_data(
            keywords,
            QueryParam(
                mode="naive",
                top_k=top_k,
                chunk_top_k=top_k,
                enable_rerank=False,
            ),
        )
    return ranked_abilities(raw_result, records, top_k)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index and retrieve YAML abilities with LightRAG."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="Index ability YAML files")
    index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the configured RAG index",
    )
    search_parser = subparsers.add_parser(
        "search", help="Return the top-k abilities for keyword text"
    )
    search_parser.add_argument("keywords", help="Keywords or a short natural-language query")
    search_parser.add_argument("-k", "--top-k", type=int, default=5)
    return parser


async def _run(args: argparse.Namespace, settings: Settings) -> Any:
    if args.command == "index":
        return await index_abilities(settings, rebuild=args.rebuild)
    return await search_abilities(settings, args.keywords, args.top_k)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(_run(args, Settings.from_environment()))
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.exit(2, f"error: {error}\n")
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
