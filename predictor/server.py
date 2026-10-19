"""Small web server: serves web/ and lets the page trigger a data refresh."""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from . import config

log = logging.getLogger(__name__)


class RefreshState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.last_error: str | None = None
        self.last_finished: str | None = None
        self.started: str | None = None

    def snapshot(self) -> dict:
        meta_path = config.WEB_DATA_DIR / "meta.json"
        generated = None
        if meta_path.exists():
            try:
                generated = json.loads(meta_path.read_text(encoding="utf-8")).get("generated_at")
            except ValueError:
                pass
        return {"running": self.running, "started": self.started, "last_finished": self.last_finished,
                "last_error": self.last_error, "generated_at": generated}

    def start(self, force_fetch: bool = True) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.started = _now()
        threading.Thread(target=self._run, args=(force_fetch,), daemon=True).start()
        return True

    def _run(self, force_fetch: bool) -> None:
        from .build import build

        try:
            build(force_fetch=force_fetch)
            self.last_error = None
        except Exception:  # surfaced to the page through /api/status
            self.last_error = traceback.format_exc(limit=3)
            log.exception("refresh failed")
        finally:
            self.last_finished = _now()
            self.running = False


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


STATE = RefreshState()


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        if self.path.startswith("/data/") or self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        if self.path.split("?")[0] == "/api/status":
            return self._json(200, STATE.snapshot())
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] == "/api/refresh":
            started = STATE.start()
            return self._json(202 if started else 409, {"started": started, **STATE.snapshot()})
        self._json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args) -> None:
        if not self.path.startswith("/api/status"):
            log.debug("%s - %s", self.address_string(), fmt % args)


def _data_age_hours() -> float | None:
    generated = STATE.snapshot()["generated_at"]
    if not generated:
        return None
    ts = datetime.strptime(generated, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600


def _auto_refresh(every_hours: float) -> None:
    while True:
        age = _data_age_hours()
        if age is None or age >= every_hours:
            STATE.start()
        time.sleep(600)


def serve(host: str = "127.0.0.1", port: int = 8000, refresh_hours: float = 6.0) -> None:
    if refresh_hours > 0:
        threading.Thread(target=_auto_refresh, args=(refresh_hours,), daemon=True).start()
    elif not (config.WEB_DATA_DIR / "meta.json").exists():
        STATE.start()
    httpd = ThreadingHTTPServer((host, port), partial(Handler, directory=str(config.WEB_DIR)))
    print(f"Serving on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
