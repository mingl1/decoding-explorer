#!/usr/bin/env python3
"""Serve trace_viewer.html with auto-discovery of runs/."""

import http.server
import json
import os
import sys
import webbrowser

PORT = 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/runs":
            runs_dir = os.path.join(os.getcwd(), "runs")
            runs = []
            if os.path.isdir(runs_dir):
                for name in sorted(os.listdir(runs_dir), reverse=True):
                    trace = os.path.join(runs_dir, name, "trace.json")
                    if os.path.isfile(trace):
                        runs.append(name)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(runs).encode())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        # Quiet down request logs
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    url = f"http://localhost:{port}/trace_viewer.html"
    print(f"Serving at {url}")
    webbrowser.open(url)
    http.server.HTTPServer(("", port), Handler).serve_forever()
