"""Exercise paid-stage control flow without making API calls."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


class ProductFailureTests(unittest.TestCase):
    def setUp(self):
        try:
            from closetscan import product_shots
        except ImportError:
            self.skipTest("needs .[pipeline]")
        self.stage = product_shots
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        fixture = Path(__file__).resolve().parent / "fixtures/catalogue"
        for name in ("manifest.json", "dedup.json"):
            shutil.copy(fixture / name, self.out)
        shutil.copytree(fixture / "products", self.out / "products")

    def invoke(self, response):
        with patch.object(sys, "argv", ["product_shots", str(self.out), "--only", "0"]), \
             patch.object(self.stage, "read_key", return_value="test-only"), \
             patch.object(self.stage, "gather", return_value=[]), \
             patch.object(self.stage, "generate", return_value=response), \
             contextlib.redirect_stdout(io.StringIO()):
            self.stage.main()

    def test_failed_requests_stop_notebook(self):
        with self.assertRaises(SystemExit) as raised:
            self.invoke((None, 0.0, "http 401: test failure"))
        self.assertNotEqual(raised.exception.code, 0)
        records = json.loads((self.out / "products/products.json").read_text())["images"]
        failed = [r for r in records if r["group"] == 0]
        self.assertEqual(len(failed), 2)
        self.assertTrue(all(r["file"] is None and r["error"] for r in failed))

    def test_retry_preserves_other_garments_and_provenance(self):
        index = self.out / "products/products.json"
        before = json.loads(index.read_text())
        others = [r for r in before["images"] if r["group"] != 0]
        self.invoke((b"mock image bytes", 0.01, ""))
        after = json.loads(index.read_text())
        for old in others:
            new = next(r for r in after["images"]
                       if (r["group"], r["view"]) == (old["group"], old["view"]))
            self.assertEqual(new["file"], old["file"])
            self.assertEqual(new["model"], old.get("model", before["model"]))
        self.assertEqual(len(after["images"]), len(before["images"]))

    def test_image_only_groups_require_real_references(self):
        from closetscan.dedup import check_coverage
        result = {"groups": [{"rows": [0]}, {"rows": [], "sides": [{"image": "row_0_alt0"}]}],
                  "junk_rows": []}
        self.assertEqual(check_coverage(result, 1, {"row_0", "row_0_alt0"}), [])
        self.assertTrue(check_coverage(result, 1, {"row_0"}))
        result["groups"][1]["sides"] = []
        self.assertTrue(check_coverage(result, 1, {"row_0", "row_0_alt0"}))


if __name__ == "__main__":
    unittest.main()
