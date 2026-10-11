import json
import tempfile
import unittest
from pathlib import Path

from processing_scripts.subsample_hotpot_by_type import (
    convert_raw_instance,
    stratified_sample,
    write_manifest,
)


def raw_instance(question_id, question_type):
    return {
        "_id": question_id,
        "question": "Question " + question_id,
        "answer": "Answer " + question_id,
        "level": "medium",
        "type": question_type,
        "context": [["Title " + question_id, ["Sentence one.", "Sentence two."]]],
        "supporting_facts": [["Title " + question_id, 0]],
    }


class SubsampleHotpotByTypeTest(unittest.TestCase):
    def test_conversion_preserves_required_rag_fields(self):
        converted = convert_raw_instance(raw_instance("bridge-1", "bridge"))

        self.assertEqual(converted["question_id"], "bridge-1")
        self.assertEqual(converted["type"], "bridge")
        self.assertEqual(converted["answers_objects"][0]["spans"], ["Answer bridge-1"])
        self.assertEqual(converted["contexts"][0]["title"], "Title bridge-1")
        self.assertTrue(converted["contexts"][0]["is_supporting"])

    def test_stratified_sample_selects_each_type_deterministically(self):
        rows = [
            convert_raw_instance(raw_instance("bridge-" + str(i), "bridge"))
            for i in range(3)
        ] + [
            convert_raw_instance(raw_instance("comparison-" + str(i), "comparison"))
            for i in range(3)
        ]

        first = stratified_sample(rows, per_type=2, seed=13370)
        second = stratified_sample(rows, per_type=2, seed=13370)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(sum(row["type"] == "bridge" for row in first), 2)
        self.assertEqual(sum(row["type"] == "comparison" for row in first), 2)
        self.assertEqual(len({row["question_id"] for row in first}), 4)

    def test_stratified_sample_rejects_insufficient_type(self):
        rows = [convert_raw_instance(raw_instance("bridge-1", "bridge"))]
        with self.assertRaises(ValueError):
            stratified_sample(rows, per_type=1, seed=13370)

    def test_manifest_records_reproducibility_inputs_and_ids(self):
        rows = [convert_raw_instance(raw_instance("bridge-1", "bridge"))]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            write_manifest(rows, str(path), seed=13370, per_type=1)
            manifest = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["seed"], 13370)
        self.assertEqual(manifest["per_type"], 1)
        self.assertEqual(manifest["question_ids"], ["bridge-1"])


if __name__ == "__main__":
    unittest.main()
