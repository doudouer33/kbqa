"""Small experiment utilities; inference and DiskCache remain repository owned."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
HERE = SCRIPTS.parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
RUNS = HERE / "runs"
SMOKE = RUNS / "smoke"
LOGS = RUNS / "logs"
sys.path.insert(0, str(ROOT))

from analysis.hotpot_formal_runner.storage import atomic_json, atomic_text, sha256, text_hash


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def settings():
    return read_json(HERE / "settings.json")


def artifact_path(path):
    """Resolve paths in preserved manifests without rewriting their identity."""
    path = Path(path)
    key = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
    migration_path = RUNS / "directory_migration.json"
    moves = read_json(migration_path)["moves"] if migration_path.exists() else {}
    return ROOT / moves.get(key, key)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ext_vars():
    from lib import get_retriever_address
    address = get_retriever_address()
    return {"RETRIEVER_HOST": str(address["host"]), "RETRIEVER_PORT": str(address["port"])}


def configured(path):
    import _jsonnet
    return json.loads(_jsonnet.evaluate_file(str(path), ext_vars=ext_vars()))


def environment():
    from lib import get_llm_server_address
    env = dict(os.environ)
    address = get_llm_server_address(settings()["llm_port"])
    env.update(
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
        OPENAI_API_KEY="EMPTY",
        OPENAI_BASE_URL=address["host"].rstrip("/") + ":" + address["port"] + "/v1",
        NO_PROXY="localhost,127.0.0.1", no_proxy="localhost,127.0.0.1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
    )
    # Do not accidentally activate the historical cache-bypassing runner.
    for key in list(env):
        if key.startswith("KBQA_FORMAL_") or key.startswith("HOTPOT5000_"):
            env.pop(key)
    env["PYTHONPATH"] = str(ROOT)
    return env


def prediction_path(manifest, mode, k):
    config = artifact_path(manifest["configs"][str(k)]["path"])
    data = artifact_path(manifest["datasets"][mode]["path"])
    name = "prediction__hotpotqa_to_hotpotqa__" + data.stem + ".json"
    return RUNS / "predictions" / manifest["set_names"][mode] / config.stem / name


def binding(manifest, mode, k):
    return {
        "manifest_sha256": sha256(RUNS / "run_manifest.json"),
        "input_sha256": manifest["datasets"][mode]["sha256"],
        "config_sha256": manifest["configs"][str(k)]["sha256"],
        "k": k, "set_name": manifest["set_names"][mode],
    }
