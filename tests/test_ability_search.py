from __future__ import annotations

import tempfile
import unittest
import sqlite3
from pathlib import Path

import yaml

from ability_search import AbilitySearch, _fts_query, load_abilities


def _write_ability(
    path: Path,
    *,
    ability_id: str,
    name: str,
    description: str,
    tactic: str,
    technique_id: str,
    technique_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "id": ability_id,
                    "name": name,
                    "description": description,
                    "tactic": tactic,
                    "technique": {"attack_id": technique_id, "name": technique_name},
                    "platforms": {
                        "windows": {"psh": {"command": "Get-LocalUser", "cleanup": ""}}
                    },
                }
            ],
            sort_keys=False,
        ),
        encoding="utf-8",
    )


class AbilitySearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.abilities = self.root / "abilities"
        self.database = self.root / "search.sqlite"
        _write_ability(
            self.abilities / "folders-do-not-matter" / "one.yml",
            ability_id="one",
            name="Download phishing attachment",
            description="Use PowerShell to retrieve a malicious Office document",
            tactic="initial-access",
            technique_id="T1566.001",
            technique_name="Phishing: Spearphishing Attachment",
        )
        _write_ability(
            self.abilities / "also-ignored" / "two.yaml",
            ability_id="two",
            name="Enumerate local accounts",
            description="Discover users with PowerShell",
            tactic="discovery",
            technique_id="T1087.001",
            technique_name="Account Discovery: Local Account",
        )
        self.search = AbilitySearch(self.database)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_indexes_and_ranks_keyword_results(self) -> None:
        summary = self.search.index(self.abilities)
        self.assertEqual(summary["inserted"], 2)
        results = self.search.search("phishing PowerShell", top_k=2)
        self.assertEqual([result["ability"]["id"] for result in results], ["one", "two"])

    def test_filters_by_yaml_tactic_not_folder(self) -> None:
        self.search.index(self.abilities)
        results = self.search.search("PowerShell", tactics=["INITIAL-ACCESS"])
        self.assertEqual([result["ability"]["id"] for result in results], ["one"])

    def test_filters_by_technique_id_or_exact_name(self) -> None:
        self.search.index(self.abilities)
        by_id = self.search.search(techniques=["t1087.001"])
        by_name = self.search.search(
            techniques=["Phishing: Spearphishing Attachment"]
        )
        self.assertEqual(by_id[0]["ability"]["id"], "two")
        self.assertEqual(by_name[0]["ability"]["id"], "one")

    def test_incremental_sync_updates_and_deletes(self) -> None:
        self.search.index(self.abilities)
        first = self.abilities / "folders-do-not-matter" / "one.yml"
        first.unlink()
        second = self.abilities / "also-ignored" / "two.yaml"
        content = second.read_text(encoding="utf-8").replace(
            "Enumerate local accounts", "List local accounts"
        )
        second.write_text(content, encoding="utf-8")
        summary = self.search.index(self.abilities)
        self.assertEqual(summary["updated"], 1)
        self.assertEqual(summary["deleted"], 1)
        self.assertEqual(self.search.search("phishing"), [])
        self.assertEqual(self.search.search("List")[0]["ability"]["id"], "two")

    def test_moving_a_file_updates_source_without_changing_categories(self) -> None:
        self.search.index(self.abilities)
        original = self.abilities / "folders-do-not-matter" / "one.yml"
        moved = self.abilities / "arbitrary" / "location" / "one.yml"
        moved.parent.mkdir(parents=True)
        original.rename(moved)

        summary = self.search.index(self.abilities)
        results = self.search.search(tactics=["initial-access"])
        self.assertEqual(summary["updated"], 1)
        self.assertEqual(results[0]["source"], moved.as_posix())
        self.assertEqual(results[0]["ability"]["tactic"], "initial-access")

    def test_facets_report_filter_values(self) -> None:
        self.search.index(self.abilities)
        facets = self.search.facets()
        self.assertEqual(
            [entry["value"] for entry in facets["tactics"]],
            ["discovery", "initial-access"],
        )
        self.assertEqual(len(facets["techniques"]), 2)

    def test_plain_text_cannot_inject_fts_operators(self) -> None:
        self.assertEqual(_fts_query('PowerShell OR "local accounts"'), '"PowerShell" OR "OR" OR "local accounts"')

    def test_duplicate_ids_are_rejected(self) -> None:
        _write_ability(
            self.abilities / "duplicate.yml",
            ability_id="one",
            name="Duplicate",
            description="Duplicate",
            tactic="other",
            technique_id="T0000",
            technique_name="Duplicate",
        )
        with self.assertRaisesRegex(ValueError, "Conflicting ability id"):
            load_abilities(self.abilities)

    def test_identical_duplicates_collapse_and_track_all_sources(self) -> None:
        original = self.abilities / "folders-do-not-matter" / "one.yml"
        duplicate = self.abilities / "another-source" / "one.yml"
        duplicate.parent.mkdir(parents=True)
        duplicate.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")

        summary = self.search.index(self.abilities)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["duplicates_collapsed"], 1)
        matches = self.search.search("phishing")
        self.assertEqual(len(matches), 1)
        self.assertEqual(set(matches[0]["sources"]), {original.as_posix(), duplicate.as_posix()})

        duplicate.unlink()
        summary = self.search.index(self.abilities)
        self.assertEqual(summary["updated"], 1)
        self.assertEqual(summary["duplicates_collapsed"], 0)
        self.assertEqual(self.search.search("phishing")[0]["sources"], [original.as_posix()])

    def test_existing_index_migrates_to_source_paths(self) -> None:
        self.search.index(self.abilities)
        with sqlite3.connect(self.database) as connection:
            connection.execute("ALTER TABLE abilities DROP COLUMN source_paths")
            connection.execute(
                "UPDATE metadata SET value = '1' WHERE key = 'schema_version'"
            )

        summary = self.search.index(self.abilities)
        self.assertEqual(summary["unchanged"], 2)
        result = self.search.search("phishing")[0]
        self.assertEqual(result["sources"], [result["source"]])


if __name__ == "__main__":
    unittest.main()
