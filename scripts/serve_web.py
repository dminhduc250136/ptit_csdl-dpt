"""Static file server cho web/ - mo tab trinh duyet tu dong.

Chay: python -m scripts.serve_web

Truy cap: http://127.0.0.1:5500
Yeu cau: API uvicorn dang chay tren port 8000.
"""
from __future__ import annotations

import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path

PORT = 5500
ROOT = Path(__file__).resolve().parents[1] / "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        # Bot log
        pass


def main() -> int:
    if not ROOT.exists():
        print(f"Khong tim thay {ROOT}")
        return 1
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/"
        print(f"Serving {ROOT} at {url}")
        print("Yeu cau API uvicorn chay tren port 8000.")
        print("Bam Ctrl+C de dung.")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
