"""
Shared HTTP utilities: a small cached, retrying client used by all fetchers.

Design goals:
  * Reproducibility: every remote response is cached to disk keyed by a hash of
    the request. A second run reads from cache and makes zero network calls, so
    a reviewer can re-run the pipeline offline once the cache is populated.
  * Politeness: exponential backoff on 429/5xx, a fixed inter-request pause, and
    a descriptive User-Agent. Overpass and the Census API both throttle abusive
    clients; this keeps us well-behaved.
  * Transparency: the cache is plain files on disk, inspectable and diffable.

"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests


class CachedSession:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        user_agent: str,
        request_pause_s: float = 1.5,
        max_retries: int = 4,
        namespace: str = "http",
    ) -> None:
        self.cache_dir = Path(cache_dir) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_pause_s = request_pause_s
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._last_request_ts = 0.0

    # -- cache key -----------------------------------------------------------
    def _key(self, method: str, url: str, payload: Any) -> Path:
        raw = json.dumps(
            {"m": method, "u": url, "p": payload}, sort_keys=True, default=str
        ).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, key: Path) -> Any | None:
        if key.exists():
            try:
                return json.loads(key.read_text())
            except json.JSONDecodeError:
                key.unlink(missing_ok=True)   # corrupt cache entry; refetch
        return None

    def _write_cache(self, key: Path, value: Any) -> None:
        tmp = key.with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(key)                       # atomic on POSIX

    # -- polite pacing -------------------------------------------------------
    def _respect_pause(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.request_pause_s:
            time.sleep(self.request_pause_s - elapsed)
        self._last_request_ts = time.monotonic()

    # -- public API ----------------------------------------------------------
    def get_json(self, url: str, params: dict | None = None) -> Any:
        return self._request_json("GET", url, params=params)

    def post_json(self, url: str, data: str, timeout_s: int = 180) -> Any:
        return self._request_json("POST", url, data=data, timeout_s=timeout_s)

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        data: str | None = None,
        timeout_s: int = 60,
    ) -> Any:
        payload = params if params is not None else data
        key = self._key(method, url, payload)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            self._respect_pause()
            try:
                resp = self._session.request(
                    method, url, params=params, data=data, timeout=timeout_s
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = (2 ** attempt) * 2.0
                    time.sleep(wait)
                    last_err = RuntimeError(
                        f"{resp.status_code} from {url} (attempt {attempt + 1})"
                    )
                    continue
                resp.raise_for_status()
                try:
                    value = resp.json()
                except (ValueError, requests.exceptions.JSONDecodeError) as json_exc:
                    # Surface the raw body so the caller can diagnose the
                    # root cause (e.g. a Census API "Missing Key" HTML page
                    # returned with HTTP 200). Catches both stdlib ValueError
                    # and the requests.exceptions.JSONDecodeError subclass
                    # raised by requests >= 2.28 / Python 3.14.
                    body_preview = resp.text[:400].replace("\n", " ")
                    raise RuntimeError(
                        f"HTTP 200 from {url} but response is not JSON.\n"
                        f"  Parse error: {json_exc}\n"
                        f"  Response body (first 400 chars): {body_preview}"
                    ) from json_exc
                self._write_cache(key, value)
                return value
            except RuntimeError:
                # RuntimeError is raised by our own JSON-body check above;
                # propagate immediately without retrying (bad request, not a
                # transient error).
                raise
            except requests.RequestException as exc:
                last_err = exc
                time.sleep((2 ** attempt) * 2.0)

        raise RuntimeError(
            f"Failed to fetch {url} after {self.max_retries} attempts: {last_err}"
        )

    def cache_status(self) -> dict[str, int]:
        files = list(self.cache_dir.glob("*.json"))
        total_bytes = sum(f.stat().st_size for f in files)
        return {"entries": len(files), "bytes": total_bytes}

    def clear_namespace(self) -> int:
        """Delete all cached entries in this session's namespace directory.

        Returns the number of files deleted. Use when a previous run poisoned
        the cache with error responses (e.g. Census 'Invalid Key' HTML pages
        that were returned as HTTP 200 and cached before the key was active).
        """
        deleted = 0
        for f in list(self.cache_dir.glob("*.json")) + list(self.cache_dir.glob("*.tmp")):
            f.unlink(missing_ok=True)
            deleted += 1
        return deleted
