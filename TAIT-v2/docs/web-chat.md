# Web Chat

Run:

```bash
tait chat --web --model model.npz
```

The server binds to `127.0.0.1` by default and exposes `/` and `/api/chat`. The frontend is plain HTML/CSS/vanilla JavaScript and the browser handles Unicode rendering naturally.

Use `--host`, `--port`, and `--no-browser` when needed. This is intentionally a local developer interface, not a public hosting layer.
