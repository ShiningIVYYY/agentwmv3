#!/usr/bin/env python3
"""CPU-only exact chat-template token counting for isolated environment runtimes."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from transformers import AutoTokenizer

TOKENIZERS = {}
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ready":true}')

    def do_POST(self):
        try:
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with LOCK:
                path = payload['path']
                if path not in TOKENIZERS:
                    TOKENIZERS[path] = AutoTokenizer.from_pretrained(path, local_files_only=True)
                kwargs = {'enable_thinking': False} if payload.get('disable_thinking') else {}
                # Transformers 5 may return BatchEncoding (len == number of
                # fields), not a list of token ids. Render then encode explicitly.
                rendered = TOKENIZERS[path].apply_chat_template(payload['messages'],
                            tokenize=False, add_generation_prompt=True, **kwargs)
                assert isinstance(rendered, str), 'Expected one rendered conversation'
                tokens = TOKENIZERS[path].encode(rendered, add_special_tokens=False)
            result = json.dumps({'count': len(tokens)}).encode()
            self.send_response(200)
        except Exception as e:
            result = json.dumps({'error': repr(e)}).encode()
            self.send_response(500)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(result)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8105)
    args = parser.parse_args()
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
