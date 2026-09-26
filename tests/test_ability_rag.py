from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from ability_rag import (
    _ability_id_from_chunk,
    ability_to_document,
    load_abilities,
    ranked_abilities,
)


class AbilityLoadingTests(unittest.TestCase):
    def test_loads_list_yaml_and_formats_search_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "test.yml"
            path.write_text(
                yaml.safe_dump(
                    [
                        {
                            "id": "abc123",
                            "name": "PowerShell discovery",
                            "description": "Find local users",
                            "tactic": "discovery",
                            "technique": {"attack_id": "T1087", "name": "Account Discovery"},
                            "platforms": {
                                "windows": {"psh": {"command": "Get-LocalUser"}}
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            records = load_abilities(root)
            self.assertEqual(len(records), 1)
            document = ability_to_document(records[0])
            self.assertIn("ABILITY_ID: abc123", document)
            self.assertIn("PowerShell discovery", document)
            self.assertIn("Get-LocalUser", document)

    def test_duplicate_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("one.yml", "two.yaml"):
                (root / name).write_text("- id: duplicate\n  name: Example\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate ability id"):
                load_abilities(root)


class RetrievalMappingTests(unittest.TestCase):
    def test_maps_ranked_chunks_back_to_complete_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "abilities.yml").write_text(
                yaml.safe_dump(
                    [
                        {"id": "first", "name": "First"},
                        {"id": "second", "name": "Second"},
                    ]
                ),
                encoding="utf-8",
            )
            records = load_abilities(root)
            raw = {
                "data": {
                    "chunks": [
                        {"content": "ABILITY_ID: second\nNAME: Second"},
                        {"content": "ABILITY_ID: first\nNAME: First"},
                    ]
                }
            }
            results = ranked_abilities(raw, records, 2)
            self.assertEqual([item["ability"]["id"] for item in results], ["second", "first"])
            self.assertEqual(_ability_id_from_chunk(raw["data"]["chunks"][0]), "second")


if __name__ == "__main__":
    unittest.main()
