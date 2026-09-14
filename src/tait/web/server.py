"""Dependency-free local web chat using the Python standard library."""
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TAIT Web Chat</title><style>
:root{--blue:#2563eb;--yellow:#facc15;--bg:#0b1020;--panel:#121a2d;--text:#eef2ff;--muted:#9aa7c0}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(140deg,#0b1020,#14213d);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,sans-serif;min-height:100vh;display:flex;justify-content:center}
.app{width:min(900px,100%);display:flex;flex-direction:column;min-height:100vh}.top{padding:18px 20px;border-bottom:1px solid #263451;display:flex;justify-content:space-between;align-items:center}.brand{font-weight:800;font-size:20px}.flag{color:var(--yellow)}.chat{flex:1;padding:20px;display:flex;flex-direction:column;gap:14px;overflow:auto}.msg{max-width:82%;padding:12px 14px;border-radius:16px;white-space:pre-wrap;line-height:1.55}.user{align-self:flex-end;background:var(--blue)}.bot{align-self:flex-start;background:var(--panel);border:1px solid #273552}.composer{padding:14px;border-top:1px solid #263451;display:flex;gap:10px}.composer textarea{flex:1;resize:none;background:#0e1628;color:var(--text);border:1px solid #30415f;border-radius:14px;padding:12px;min-height:48px;font:inherit}.composer button{border:0;border-radius:14px;padding:0 18px;font-weight:700;background:var(--yellow);color:#111827;cursor:pointer}.status{font-size:12px;color:var(--muted);padding:0 20px 10px}
</style></head><body><div class="app"><div class="top"><div class="brand">TAIT <span class="flag">🇸🇩</span></div><div>Local Web Chat</div></div><div id="chat" class="chat"><div class="msg bot">Ready. Ask your local TAIT model something.</div></div><div id="status" class="status">127.0.0.1 • local only</div><div class="composer"><textarea id="input" placeholder="Write in Arabic, English, 中文…"></textarea><button id="send">Send</button></div></div>
<script>const chat=document.getElementById('chat'),input=document.getElementById('input'),send=document.getElementById('send'),status=document.getElementById('status');function add(text,c){const d=document.createElement('div');d.className='msg '+c;d.textContent=text;chat.appendChild(d);chat.scrollTop=chat.scrollHeight}async function go(){const text=input.value.trim();if(!text)return;input.value='';add(text,'user');send.disabled=true;status.textContent='Generating…';try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});const j=await r.json();if(!r.ok)throw new Error(j.error||'Request failed');add(j.reply,'bot')}catch(e){add('Error: '+e.message,'bot')}finally{send.disabled=false;status.textContent='127.0.0.1 • local only'}}send.onclick=go;input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();go()}});</script></body></html>'''


def run_server(model, tokenizer, host='127.0.0.1', port=7860, temperature=0.7, top_k=8, top_p=0.9, max_tokens=60, open_browser=True):
    from ..core.generation import generate_reply
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, content_type='application/json; charset=utf-8'):
            raw = body.encode('utf-8')
            self.send_response(code); self.send_header('Content-Type', content_type); self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
        def do_GET(self):
            if urlparse(self.path).path == '/': self._send(200, HTML, 'text/html; charset=utf-8')
            else: self._send(404, json.dumps({'error':'Not found'}))
        def do_POST(self):
            if urlparse(self.path).path != '/api/chat': return self._send(404, json.dumps({'error':'Not found'}))
            try:
                n=int(self.headers.get('Content-Length','0')); payload=json.loads(self.rfile.read(n) or '{}'); msg=str(payload.get('message','')).strip()
                if not msg: raise ValueError('message is required')
                reply=generate_reply(model,tokenizer,msg,max_new_tokens=max_tokens,temperature=temperature,top_k=top_k,top_p=top_p)
                self._send(200,json.dumps({'reply':reply},ensure_ascii=False))
            except Exception as exc: self._send(400,json.dumps({'error':str(exc)},ensure_ascii=False))
        def log_message(self, format, *args): pass
    server=ThreadingHTTPServer((host,port),Handler)
    url=f'http://{host}:{port}/'
    print(f"TAIT Web Chat: {url}")
    print("Press Ctrl+C to stop.")
    if open_browser: threading.Timer(0.4, lambda:webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nStopping web chat…")
    finally: server.server_close()
