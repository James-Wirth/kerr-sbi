import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from records import Records, contained

STATIC = Path(__file__).resolve().parent / "static"


def handler_for(records: Records) -> type[BaseHTTPRequestHandler]:
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def allowed_host(self) -> bool:
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

        def do_POST(self) -> None:
            if (
                not self.allowed_host()
                or self.headers.get("Origin") != f"http://{self.headers.get('Host')}"
                or self.headers.get("X-Monitor-Token") != token
                or self.headers.get("Content-Type") != "application/json"
            ):
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object")
                if self.path == "/api/runs/trash":
                    result = records.trash.trash(payload["id"])
                elif self.path == "/api/trash/restore":
                    result = records.trash.restore(payload["id"])
                elif self.path == "/api/trash/purge":
                    if payload.get("confirm") is not True:
                        raise ValueError("permanent deletion requires confirmation")
                    result = records.trash.purge(payload["ids"])
                else:
                    self.send_error(404)
                    return
                records.storage_updated = 0
                self.respond(json.dumps(result).encode(), "application/json")
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                self.respond(json.dumps({"error": str(exc)}).encode(), "application/json", 409)

        def do_GET(self) -> None:
            if not self.allowed_host():
                self.send_error(403)
                return
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/api/overview":
                    value = records.overview()
                    value["mutation_token"] = token
                elif parsed.path == "/api/run":
                    value = records.run(query.get("id", [""])[0])
                elif parsed.path == "/api/preview":
                    value = records.preview(
                        query.get("id", [""])[0], int(query.get("index", ["0"])[0])
                    )
                elif parsed.path == "/api/figure":
                    identifier = query.get("id", [""])[0]
                    name = query.get("name", [""])[0]
                    if name not in ("posterior_means", "ranks"):
                        raise ValueError("unknown diagnostic figure")
                    directory = records.trash.source(identifier)
                    records.trash.validate_run(directory)
                    figure = contained(directory, directory / "diagnostics" / f"{name}.png")
                    self.respond(figure.read_bytes(), "image/png")
                    return
                elif parsed.path == "/favicon.ico":
                    self.respond(b"", "image/x-icon", 204)
                    return
                elif parsed.path in ("/", "/app.js", "/style.css"):
                    filename = "index.html" if parsed.path == "/" else parsed.path[1:]
                    payload = (STATIC / filename).read_bytes()
                    mime = {
                        "index.html": "text/html",
                        "app.js": "text/javascript",
                        "style.css": "text/css",
                    }[filename]
                    self.respond(payload, mime)
                    return
                else:
                    self.send_error(404)
                    return
                self.respond(json.dumps(value, allow_nan=False).encode(), "application/json")
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                self.respond(json.dumps({"error": str(exc)}).encode(), "application/json", 400)

        def respond(self, payload: bytes, mime: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", f"{mime}; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; "
                "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, message: str, *args: object) -> None:
            if len(args) < 2 or str(args[1]) not in ("200", "204"):
                super().log_message(message, *args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local training monitor with run trash and restore"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    records = Records(args.root, args.runs, args.data)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(records))
    print(f"Training monitor: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
