"""Run the unchanged formal runner under a dedicated tmux server.

Refuse to start while a generation request has no durable response. This launcher
adds process supervision, without changing frozen inference/configuration code.
"""
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/dengxin/miniconda3/envs/adaptiverag/bin/python"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_event(directory, value):
    with (directory / "process_lifecycle.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value, recorded_at=now()), ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def missing_responses(set_name):
    missing = []
    for path in (ROOT / "predictions" / set_name).glob("*/request_events.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not (path.parent / "api_responses" / (row["qid"] + ".json")).exists():
                missing.append({"qid": row["qid"], "k": row["k"], "request_log": str(path)})
    return missing


def worker(set_name, directory):
    missing = missing_responses(set_name)
    if missing:
        write_event(directory, {"event": "start_blocked", "requests_without_responses": missing})
        raise SystemExit("Start blocked: unresolved generation response")
    child = None
    def interrupted(signum, frame):
        write_event(directory, {"event": "supervisor_signal", "signal": signum})
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    log = directory / ("runner_tmux_" + str(time.time_ns()) + ".log")
    with log.open("a", encoding="utf-8") as stream:
        child = subprocess.Popen([PYTHON, "-u", "analysis/hotpot_formal_runner/run_experiment.py", "--set-name", set_name, "--resume"],
            cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_event(directory, {"event": "runner_started", "pid": child.pid, "supervisor_pid": os.getpid(), "log": str(log), "manager": "tmux"})
        code = child.wait()
        write_event(directory, {"event": "runner_exited", "pid": child.pid, "exit_code": code})
    if code:
        # Handle external termination that bypasses the runner's Python guards.
        status_path = directory / "status.json"
        state = json.loads(status_path.read_text(encoding="utf-8"))
        if state["status"] not in ("complete", "stopped_on_error"):
            state.update(status="stopped_on_error", updated_at=now(), error="Runner exited with code {} (see process_lifecycle.jsonl)".format(code))
            tmp = status_path.with_suffix(".supervisor.tmp")
            with tmp.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(tmp), str(status_path))
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-name", default="hotpot_formal_1000_20261014")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--probe", action="store_true", help="Check detached lifetime without inference")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.set_name):
        raise ValueError("Invalid set name")
    directory = ROOT / "analysis" / (args.set_name + "_run")
    if args.worker:
        return worker(args.set_name, directory)
    socket = "kbqa_" + args.set_name
    if args.probe:
        session = "lifecycle_probe_" + str(time.time_ns())
        log = directory / (session + ".log")
        code = "import os,time; print('PROBE_STARTED',os.getpid(),os.getppid(),os.getsid(0),flush=True); time.sleep(12); print('PROBE_COMPLETE',flush=True)"
        command = " ".join(shlex.quote(part) for part in [PYTHON, "-u", "-c", code]) + " > " + shlex.quote(str(log)) + " 2>&1"
        subprocess.run(["tmux", "-L", socket, "new-session", "-d", "-s", session, "-c", str(ROOT), command], check=True)
        print(json.dumps({"socket": socket, "probe_session": session, "log": str(log)}))
        return 0
    pending = missing_responses(args.set_name)
    if pending:
        print(json.dumps({"can_start": False, "reason": "generation requests without durable API responses", "pending": pending}, ensure_ascii=False))
        return 2
    if args.check_only:
        print(json.dumps({"can_start": True, "socket": socket}))
        return 0
    exists = subprocess.run(["tmux", "-L", socket, "has-session", "-t", "runner"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if exists.returncode == 0:
        raise SystemExit("Runner session already exists; inspect it before starting again")
    command = " ".join(shlex.quote(part) for part in [PYTHON, "-u", str(Path(__file__).resolve()), "--set-name", args.set_name, "--worker"])
    subprocess.run(["tmux", "-L", socket, "new-session", "-d", "-s", "runner", "-n", "formal", "-c", str(ROOT), command], check=True)
    subprocess.run(["tmux", "-L", socket, "set-window-option", "-t", "runner", "remain-on-exit", "on"], check=True)
    print(json.dumps({"socket": socket, "session": "runner", "status": "started"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
