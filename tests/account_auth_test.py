"""device.account_auth_key(): the device's own account API, no TiinyOS needed.

    python3 tests/account_auth_test.py

Reverse-engineered by capturing a real TiinyOS login -- full writeup at
~/code/tiiny/tools/README-unlock.md. This checks only that device.py parses
the response the device actually sends; the discovery itself lives there.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import device  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + (("  " + str(extra)) if not cond else ""))
    if not cond:
        fails.append(name)


class _Handler(BaseHTTPRequestHandler):
    reply = {"status": "ok", "data_state": "unlocked", "auth_key": "the-real-key"}
    status = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = json.dumps(self.reply).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
addr = "127.0.0.1:%d" % server.server_port
threading.Thread(target=server.serve_forever, daemon=True).start()

got = device.account_auth_key(addr, "TNYM000", "correct horse")
check("returns the auth_key field", got == "the-real-key", got)

_Handler.reply, _Handler.status = {"error": "main_password_wrong"}, 403
got = device.account_auth_key(addr, "TNYM000", "wrong")
check("wrong password is empty string, not a crash", got == "", got)
_Handler.reply, _Handler.status = {"status": "ok", "auth_key": "the-real-key"}, 200

got = device.account_auth_key("127.0.0.1:1", "TNYM000", "x")
check("unreachable host is empty string", got == "", got)

server.shutdown()
print("\nFAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
