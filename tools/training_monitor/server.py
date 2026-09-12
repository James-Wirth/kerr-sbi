import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from records import Records

STATIC = Path(__file__).resolve().parent / "static"


def handler_for(records: Records) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            allowed_hosts = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            if self.headers.get("Host") not in allowed_hosts:
                self.send_error(403)
                return
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/api/overview":
                    value = records.overview()
                elif parsed.path == "/api/run":
                    value = records.run(query.get("id", [""])[0])
                elif parsed.path == "/api/preview":
                    value = records.preview(
                        query.get("id", [""])[0], int(query.get("index", ["0"])[0])
                    )
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
    parser = argparse.ArgumentParser(description="Read-only local training monitor")
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
