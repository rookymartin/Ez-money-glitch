"""Simple HTTP server for the dashboard. Run: python dashboard/serve.py"""
import http.server
import socketserver
import os

PORT = 8080
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DASHBOARD_DIR)

with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"Dashboard running at: http://localhost:{PORT}/index.html")
    print("Press Ctrl+C to stop.")
    httpd.serve_forever()
