# This can be run in a conda environment. I have an environment for http:
#   $ conda activate http

import http.server, ssl
import socketserver
import os

PORT=22345

dir_path = os.path.dirname(os.path.realpath(__file__))
print("Path to serve:" + dir_path)

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=dir_path, **kwargs)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Serving at port", PORT)
    httpd.serve_forever()

