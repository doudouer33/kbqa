#!/usr/bin/env python3
"""Cache top-k retriever responses and serve smaller top-k prefixes locally."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, Dict, Tuple
from urllib import request


class RetrievalCache:
    def __init__(self, upstream_url: str, prefetch_count: int, timeout: float) -> None:
        self.upstream_url = upstream_url
        self.prefetch_count = prefetch_count
        self.timeout = timeout
        self._opener = request.build_opener(request.ProxyHandler({}))
        self._lock = threading.Lock()
        self._entries: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(payload: Dict[str, Any]) -> str:
        key_payload = dict(payload)
        key_payload.pop("max_hits_count", None)
        return json.dumps(key_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def retrieve(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        requested_count = int(payload.get("max_hits_count", 3))
        cache_key = self._key(payload)

        with self._lock:
            cached = self._entries.get(cache_key)
            if cached is not None and cached[0] >= requested_count:
                self.hits += 1
                result = dict(cached[1])
                result["retrieval"] = result["retrieval"][:requested_count]
                result["cache_hit"] = True
                return result

        upstream_count = max(requested_count, self.prefetch_count)
        upstream_payload = dict(payload)
        upstream_payload["max_hits_count"] = upstream_count
        encoded_payload = json.dumps(upstream_payload).encode("utf-8")
        upstream_request = request.Request(
            self.upstream_url,
            data=encoded_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener.open(upstream_request, timeout=self.timeout) as response:
            upstream_result = json.loads(response.read().decode("utf-8"))

        retrieval = upstream_result.get("retrieval")
        if not isinstance(retrieval, list):
            raise ValueError(f"Unexpected upstream retriever response: {upstream_result}")

        with self._lock:
            self._entries[cache_key] = (upstream_count, upstream_result)
            self.misses += 1

        result = dict(upstream_result)
        result["retrieval"] = retrieval[:requested_count]
        result["cache_hit"] = False
        return result

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
            }


def build_handler(cache: RetrievalCache):
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._write_json(200, {"status": "ok", "cache": cache.stats()})
            else:
                self._write_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/retrieve/":
                self._write_json(404, {"error": "not_found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object")
                self._write_json(200, cache.retrieve(payload))
            except Exception as exc:
                self._write_json(502, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8000/retrieve/")
    parser.add_argument("--prefetch-count", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    cache = RetrievalCache(args.upstream_url, args.prefetch_count, args.timeout)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(cache))
    print(
        f"Cached retriever proxy listening on http://{args.host}:{args.port}; "
        f"upstream={args.upstream_url}, prefetch_count={args.prefetch_count}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(f"Cached retriever proxy stopped; stats={cache.stats()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
