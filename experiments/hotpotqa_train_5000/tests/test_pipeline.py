"""Failure-oriented checks for strict completion, data integrity and usage provenance."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

import pipeline
from common import atomic_json, atomic_text, sha256, text_hash
from sample import scan_source, validate_row


def sample_row(qid="q1"):
    return {"dataset": "hotpotqa", "question_id": qid, "question_text": "Example?", "level": "easy", "type": "bridge",
            "answers_objects": [{"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": ["yes"]}],
            "contexts": [{"idx": 0, "title": "Title", "paragraph_text": "Paragraph", "is_supporting": True}]}


class StrictCompletionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.prediction = self.directory / "prediction__example.json"
        self.manifest = {"datasets": {"smoke": {"question_ids": ["q1"]}}}
        self.binding = {"k": 1, "config_sha256": "x", "input_sha256": "y"}
        self.patches = [patch.object(pipeline, "binding", return_value=self.binding),
                        patch.object(pipeline, "prediction_path", return_value=self.prediction)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        text = "Wikipedia Title: Title\nParagraph"
        self.row = {"qid": "q1", "k": 1, "binding": self.binding, "prediction": "yes", "chain": "chain",
                    "raw_contexts": [{"title": "Title", "paragraph_text": "Paragraph"}], "reader_contexts": [text],
                    "effective_context_text": text, "context_sha256": text_hash(text), "chain_sha256": text_hash("chain"),
                    "prompt_sha256": "a" * 64, "call_sha256": "b" * 64,
                    "api": {"usage": None, "usage_source": "unavailable_legacy_diskcache", "finish_reasons": None}}
        self.predictions = {"q1": "yes"}
        self.write_outputs()

    def write_outputs(self):
        atomic_json(self.prediction, self.predictions)
        atomic_text(self.directory / "contexts.jsonl", json.dumps(self.row) + "\n")
        for suffix, value in (("_chains.txt", "chain\n"), ("_time_taken.txt", "1"),
                              ("_full_eval_path.txt", "input.jsonl"), ("_variable_replacements.json", "")):
            atomic_text(self.directory / (self.prediction.stem + suffix), value)
        atomic_text(self.directory / "config__example.jsonnet", "{}")
        files = [path for path in self.directory.iterdir() if path.name != "complete.json"]
        atomic_json(self.directory / "complete.json", {
            "status": "complete", "predict_exit_code": 0, "binding": self.binding,
            "output_hashes": {path.name: sha256(path) for path in files}})

    def validate(self, predictions=None, records=None):
        pipeline.validate_records(self.manifest, "smoke", 1, self.predictions if predictions is None else predictions,
                                  [self.row] if records is None else records)

    def test_valid_outputs_and_unknown_cached_usage_are_complete(self):
        self.assertTrue(pipeline.inspect_complete(self.manifest, "smoke", 1)[0])

    def test_missing_extra_or_whitespace_prediction_is_rejected(self):
        for predictions in ({}, {"q1": "yes", "extra": "no"}, {"q1": " "}, {"q1": None}, {"q1": ["yes"]}):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                self.validate(predictions=predictions)

    def test_duplicate_or_missing_context_is_rejected(self):
        for records in ([], [self.row, self.row]):
            with self.subTest(count=len(records)), self.assertRaises(ValueError):
                self.validate(records=records)

    def test_91_percent_predictions_are_still_incomplete(self):
        manifest = {"datasets": {"smoke": {"question_ids": [str(index) for index in range(100)]}}}
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            pipeline.validate_records(manifest, "smoke", 1, {str(index): "yes" for index in range(91)}, [])

    def test_wrong_prediction_binding_context_and_chain_are_rejected(self):
        mutations = [("prediction", "no"), ("k", 2), ("binding", {}),
                     ("effective_context_text", "other"), ("raw_contexts", []), ("chain", "other")]
        for field, value in mutations:
            row = dict(self.row, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(records=[row])

    def test_output_hash_change_and_missing_marker_are_incomplete(self):
        atomic_json(self.prediction, {"q1": "no"})
        self.assertFalse(pipeline.inspect_complete(self.manifest, "smoke", 1)[0])
        (self.directory / "complete.json").unlink()
        self.assertFalse(pipeline.inspect_complete(self.manifest, "smoke", 1)[0])

    def test_duplicate_prediction_json_keys_are_incomplete(self):
        atomic_text(self.prediction, '{"q1":"yes","q1":"yes"}')
        marker = self.directory / "complete.json"
        value = pipeline.read_json(marker)
        value["output_hashes"][self.prediction.name] = sha256(self.prediction)
        atomic_json(marker, value)
        self.assertFalse(pipeline.inspect_complete(self.manifest, "smoke", 1)[0])

    def test_running_or_failed_attempt_cannot_be_complete(self):
        for status in ("running", "failed"):
            marker = pipeline.read_json(self.directory / "complete.json")
            marker["status"] = status
            atomic_json(self.directory / "complete.json", marker)
            self.assertFalse(pipeline.inspect_complete(self.manifest, "smoke", 1)[0])

    def test_usage_requires_actual_api_provenance_and_consistent_totals(self):
        row = copy.deepcopy(self.row)
        row["api"] = {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                      "usage_source": "api_response", "finish_reasons": ["stop"]}
        self.validate(records=[row])
        for api in (dict(row["api"], usage_source="estimate"), dict(row["api"], finish_reasons=["length"]),
                    dict(row["api"], usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 13})):
            with self.subTest(api=api), self.assertRaises(ValueError):
                self.validate(records=[dict(row, api=api)])

    def test_full_inference_needs_explicit_flag_before_any_work(self):
        with patch.object(pipeline, "load_manifest", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ValueError, "allow-full-run"):
                pipeline.run_k("full", 1)


class SamplingIntegrityTest(unittest.TestCase):
    def test_missing_fields_or_broken_nested_context_are_rejected(self):
        row = sample_row()
        validate_row(row)
        del row["contexts"][0]["is_supporting"]
        with self.assertRaises(ValueError):
            validate_row(row)
        del row["answers_objects"]
        with self.assertRaises(ValueError):
            validate_row(row)

    def test_source_duplicates_are_rejected_and_fewshots_excluded_before_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            rows = [sample_row("fewshot"), sample_row("q1")]
            atomic_text(path, "".join(json.dumps(row) + "\n" for row in rows))
            candidates, excluded, metadata, digest = scan_source(path, {"fewshot"})
            self.assertEqual([qid for qid, _ in candidates], ["q1"])
            self.assertEqual(excluded, ["fewshot"])
            self.assertEqual(len(metadata), 2)
            self.assertEqual(digest, sha256(path))
            atomic_text(path, "".join(json.dumps(row) + "\n" for row in rows + rows))
            with self.assertRaisesRegex(ValueError, "Duplicate source ID"):
                scan_source(path, {"fewshot"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
