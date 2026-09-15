import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from closetscan.catalogue import load_catalogue
from closetscan.mcp_server import Wardrobe, handle


class CatalogueIntegrityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.write("manifest.json", {"garments": [
            {"id": i, "best_shot": f"shots/{i}.jpg", "timestamp_s": i + 1,
             "source_clip": "test.mp4", "observations": 3, "views": []}
            for i in range(2)
        ]})
        self.write("dedup.json", {"groups": [{"rows": [0, 1], "label": "grouped shirt"}],
                                  "junk_rows": []})
        self.write("attributes.json", {"garments": {"0": {"name": "old name", "category": "top"}}})
        self.write("narration.json", {"garments": {"0": {"summary": "old narration"}}})
        self.write("products/products.json", {"model": "test", "images": [
            {"group": 0, "view": "front", "file": "old.png"}
        ]})
        self.wardrobe = Wardrobe(str(self.directory))

    def write(self, name, data):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def call(self, tool, arguments):
        return handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": tool, "arguments": arguments}}, self.wardrobe)["result"]

    def test_invalid_ids_do_not_create_wear_log(self):
        arguments = [{}] + [{"garment_id": value} for value in
                            (None, "", "  ", "g", "G", "gg000", "g 0", "-1", "bad", 0, False, [], {})]
        before = {p: p.read_bytes() for p in self.directory.rglob("*.json")}
        for args in arguments:
            with self.subTest(arguments=args):
                self.assertTrue(self.call("log_wear", args)["isError"])
                self.assertTrue(self.call("get_garment", args)["isError"])
                self.assertFalse((self.directory / "wear_log.jsonl").exists())
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_invalid_id_does_not_append_to_existing_log(self):
        self.assertFalse(self.call("log_wear", {"garment_id": "g000", "date": "2026-09-15"})["isError"])
        log = self.directory / "wear_log.jsonl"
        before = log.read_bytes()
        self.assertTrue(self.call("log_wear", {})["isError"])
        self.assertEqual(log.read_bytes(), before)

    def test_documented_id_aliases_still_work(self):
        for gid in ("0", "g0", "G000", " g000 ", "000"):
            with self.subTest(gid=gid):
                result = self.call("get_garment", {"garment_id": gid})
                self.assertFalse(result["isError"])
                self.assertEqual(json.loads(result["content"][0]["text"])["id"], "g000")

    def test_corrupt_grouping_is_an_error_not_a_preview(self):
        path = self.directory / "dedup.json"
        for content in ("{", "null", "[]", "{}", '{"groups": null}',
                        '{"groups": [], "junk_rows": 1}'):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaisesRegex(ValueError, "dedup.json"):
                    load_catalogue(str(self.directory))
                self.assertTrue(self.call("list_garments", {})["isError"])
                self.assertTrue(self.call("log_wear", {"garment_id": "g000"})["isError"])
                self.assertFalse((self.directory / "wear_log.jsonl").exists())
                self.assertEqual(path.read_text(), content)

    def test_missing_grouping_ignores_old_enrichments(self):
        (self.directory / "dedup.json").unlink()
        garments = load_catalogue(str(self.directory))
        self.assertEqual(len(garments), 2)
        for garment in garments:
            self.assertEqual(garment.name, f"candidate row {garment.index}")
            self.assertEqual(garment.attributes, {})
            self.assertEqual(garment.narration, {})
            self.assertTrue(garment.views)
            self.assertTrue(all(view.kind == "frame" for view in garment.views))

    def test_missing_optional_files_remain_supported(self):
        for name in ("attributes.json", "narration.json", "products/products.json"):
            (self.directory / name).unlink()
        garments = load_catalogue(str(self.directory))
        self.assertEqual(len(garments), 1)
        self.assertEqual(garments[0].name, "grouped shirt")
        self.assertEqual(garments[0].attributes, {})

    def test_corrupt_optional_file_is_not_silently_ignored(self):
        (self.directory / "attributes.json").write_text("{")
        with self.assertRaisesRegex(ValueError, "attributes.json"):
            load_catalogue(str(self.directory))

    def test_unreadable_grouping_is_not_treated_as_missing(self):
        real_open = open
        def guarded_open(path, *args, **kwargs):
            if Path(path).name == "dedup.json":
                raise PermissionError("dedup.json is not readable")
            return real_open(path, *args, **kwargs)
        with patch("closetscan.catalogue.open", side_effect=guarded_open, create=True):
            with self.assertRaises(PermissionError):
                load_catalogue(str(self.directory))

    def test_cached_wardrobe_reports_corruption_and_recovers_after_repair(self):
        import os
        self.assertEqual(len(self.wardrobe.garments), 1)
        path = self.directory / "dedup.json"
        original = path.read_bytes()
        previous_mtime = path.stat().st_mtime
        path.write_text("{")
        os.utime(path, (previous_mtime + 2, previous_mtime + 2))
        self.assertTrue(self.call("list_garments", {})["isError"])
        path.write_bytes(original)
        os.utime(path, (previous_mtime + 4, previous_mtime + 4))
        self.assertFalse(self.call("list_garments", {})["isError"])


if __name__ == "__main__":
    unittest.main()
