"""Run ONLY 6 coverage questions at k=1/8/15 and exercise interruption/restart.

Intended for initial acceptance in the isolated smoke set. Re-running an already
accepted smoke set verifies/uses strict skips, without launching the full set.
"""
import json
import os
import signal
import subprocess
import sys
import time

from common import SCRIPTS, SMOKE, RUNS, ROOT, atomic_json, environment, now, prediction_path, read_json, read_rows, require, sha256
from pipeline import inspect_complete, load_manifest


def command(k):
    return [sys.executable, str(SCRIPTS / "pipeline.py"), "run-k", "--mode", "smoke", "--k", str(k)]


def main():
    os.chdir(str(ROOT))
    os.environ.update(environment())
    manifest = load_manifest()
    started = now()
    report = {"started_at_utc": started, "mode": "smoke", "ks": [1, 8, 15],
              "questions": manifest["datasets"]["smoke"], "full_run_started": False}
    log = SMOKE / "smoke_acceptance.log"
    with log.open("a") as stream:
        subprocess.run(command(1), env=environment(), stdout=stream, stderr=subprocess.STDOUT, check=True)
        prediction = prediction_path(manifest, "smoke", 8)
        partial = []
        if not (prediction.parent / "attempts").exists():
            child = subprocess.Popen(command(8), env=environment(), stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 600
                while True:
                    paths = list((prediction.parent / "attempts").glob("*/records/*.json"))
                    if len(paths) >= 2:
                        partial = [read_json(path) for path in paths]
                        child.send_signal(signal.SIGTERM)
                        break
                    require(child.poll() is None, "Smoke child exited before interruption checkpoint")
                    require(time.monotonic() < deadline, "Smoke interruption test timed out")
                    time.sleep(0.2)
                code = child.wait(timeout=30)
                require(code != 0, "Interrupted attempt must exit nonzero")
                require(not inspect_complete(manifest, "smoke", 8)[0], "Interrupted k incorrectly marked complete")
                attempts = sorted((prediction.parent / "attempts").glob("*"))
                require(read_json(attempts[-1] / "attempt.json")["status"] == "failed", "Interruption status missing")
                report["interruption"] = {"signal": "SIGTERM", "exit_code": code,
                    "completed_qids_before_interrupt": [row["qid"] for row in partial],
                    "completion_marker_absent": not (prediction.parent / "complete.json").exists(),
                    "attempt": str(attempts[-1].relative_to(ROOT))}
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=30)
        subprocess.run(command(8), env=environment(), stdout=stream, stderr=subprocess.STDOUT, check=True)
        resumed = {row["qid"]: row for row in read_rows(prediction.parent / "contexts.jsonl")}
        for row in partial:
            current = resumed[row["qid"]]
            require(current["cache_hit"] and current["call_sha256"] == row["call_sha256"], "Recovery did not reuse the original DiskCache call")
            require(current["prediction"] == row["prediction"], "Recovery prediction changed")
            require(current["api"]["usage"] == row["api"]["usage"], "Saved actual usage not preserved on cache hit")
        if partial:
            report["resume"] = {"original_diskcache_reused_for_completed_questions": True,
                                "preserved_prediction_and_actual_usage": True}
        subprocess.run(command(15), env=environment(), stdout=stream, stderr=subprocess.STDOUT, check=True)
        results = {}
        for k in (1, 8, 15):
            done, reason = inspect_complete(manifest, "smoke", k)
            require(done, reason)
            pred = prediction_path(manifest, "smoke", k)
            rows = read_rows(pred.parent / "contexts.jsonl")
            results[str(k)] = {"count": len(rows), "cache_hits": sum(row["cache_hit"] for row in rows),
                "usage_available": sum(row["api"]["usage"] is not None for row in rows),
                "prompt_cropped": sum(row["prompt_cropped"] for row in rows),
                "actual_context_counts": [len(row["reader_contexts"]) for row in rows],
                "fewshot_counts": [row["fewshot_count_after_fit"] for row in rows],
                "prediction_sha256": sha256(pred), "context_sha256": sha256(pred.parent / "contexts.jsonl")}
            before = sha256(pred.parent / "complete.json")
            attempt_count = len(list((pred.parent / "attempts").glob("*")))
            result = subprocess.run(command(k), env=environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            stream.write(result.stdout)
            require("SKIP k=" in result.stdout, "Strict completed result was not skipped")
            require(sha256(pred.parent / "complete.json") == before, "Skipped marker was rewritten")
            require(len(list((pred.parent / "attempts").glob("*"))) == attempt_count, "Skip created an inference attempt")
        report.update(status="passed", ended_at_utc=now(), results=results,
                      completed_results_skipped_without_new_attempt=True,
                      prediction_and_effective_context_consistency=True,
                      run_manifest_sha256=sha256(RUNS / "run_manifest.json"))
        target = SMOKE / "smoke_acceptance_report.json"
        if target.exists():
            target = SMOKE / ("smoke_recheck_" + str(int(time.time())) + ".json")
        atomic_json(target, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
