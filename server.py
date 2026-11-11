#!/usr/bin/env python3
"""Juego de rol y simulación con IA local.
Requisitos: Python 3.9+ y Ollama (https://ollama.com) en marcha. Sin dependencias extra.
Uso:  python server.py   ->  abre http://localhost:8000
"""
import json, os, re, sqlite3, threading, time, urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

BASE = Path(__file__).parent
DB = BASE / "partidas.db"
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
PORT = int(os.environ.get("PORT", 8000))
NUM_CTX = int(os.environ.get("NUM_CTX", 8192))
CAMPOS = ["rol", "edad", "aspecto", "personalidad", "historia", "relaciones", "estado", "notas"]


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS games(id INTEGER PRIMARY KEY, name TEXT, setup TEXT, rules TEXT,
            directives TEXT DEFAULT '[]', model TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, game_id INTEGER, kind TEXT, content TEXT, ts REAL);
        CREATE TABLE IF NOT EXISTS characters(id INTEGER PRIMARY KEY, game_id INTEGER, name TEXT, data TEXT,
            updated REAL, UNIQUE(game_id, name));
        """)


# ---------- Ollama ----------
def ollama(path, body=None, timeout=900):
    req = urllib.request.Request(
        OLLAMA + path,
        json.dumps(body).encode() if body is not None else None,
        {"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def chat(model, messages, stream=False, fmt=None):
    body = {"model": model, "messages": messages, "stream": stream, "options": {"num_ctx": NUM_CTX}}
    if fmt:
        body["format"] = fmt
        body["options"]["temperature"] = 0.2
    return ollama("/api/chat", body)


# ---------- Fichas ----------
def sheet_text(name, data):
    lines = [f"FICHA: {name}"]
    for k in CAMPOS:
        if data.get(k):
            lines.append(f"- {k}: {data[k]}")
    return "\n".join(lines)


def mentioned(chars, text):
    t = text.lower()
    out = []
    for ch in chars:
        parts = [ch["name"]] + ch["name"].split()
        if any(len(p) > 2 and re.search(r"\b" + re.escape(p.lower()) + r"\b", t) for p in parts):
            out.append(ch)
    return out


def find_char(chars, name):
    n = name.strip().lower()
    for ch in chars:
        if ch["name"].lower() == n:
            return ch
    for ch in chars:  # coincidencia por nombre de pila
        if n and n.split()[0] == ch["name"].lower().split()[0] and len(n.split()[0]) > 2:
            return ch
    return None


def load_chars(gid):
    with db() as c:
        rows = c.execute("SELECT * FROM characters WHERE game_id=?", (gid,)).fetchall()
    return [{"id": r["id"], "name": r["name"], "data": json.loads(r["data"])} for r in rows]


def save_char(gid, name, data, cid=None):
    with db() as c:
        if cid:
            c.execute("UPDATE characters SET name=?, data=?, updated=? WHERE id=? AND game_id=?",
                      (name, json.dumps(data, ensure_ascii=False), time.time(), cid, gid))
        else:
            c.execute("INSERT OR REPLACE INTO characters(game_id,name,data,updated) VALUES(?,?,?,?)",
                      (gid, name, json.dumps(data, ensure_ascii=False), time.time()))


# ---------- Prompts ----------
def system_prompt(g, directives, sheets):
    s = f"""Eres el NARRADOR y motor de simulación de un juego de rol por texto. No hay dados ni azar mecánico: tú decides las consecuencias de forma coherente con el mundo, las reglas y las fichas de los personajes.

Cómo jugar:
- El jugador escribe lo que dice o hace su personaje. Interprétalo con naturalidad y narra qué ocurre como consecuencia, según lo que diga puede pasar una cosa u otra.
- Nunca decidas por el jugador lo que dice, piensa o hace su personaje. Solo narra el mundo y a los demás.
- El mundo está vivo: los personajes tienen metas, personalidad y memoria propias, y actúan por iniciativa propia. Pueden pasar cosas aunque el jugador no las provoque.
- Respeta estrictamente las fichas: no mezcles características entre personajes ni contradigas datos ya establecidos.
- Los mensajes marcados como [DIRECTOR] son instrucciones del jugador sobre la historia o el mundo (cambiar algo, corregir un error, añadir un personaje). Aplícalas de inmediato y sin discutir, y continúa la escena ya corregida.
- Escribe en español, con prosa narrativa y diálogos. Extensión media (2-5 párrafos). Termina dejando margen para que el jugador responda.

MUNDO Y PROTAGONISTA (descrito por el jugador):
{g['setup']}"""
    if g["rules"].strip():
        s += f"\n\nREGLAS DE LA PARTIDA (impuestas por el jugador, obligatorias):\n{g['rules']}"
    if directives:
        s += "\n\nINSTRUCCIONES PERMANENTES DEL DIRECTOR (siguen vigentes):\n" + "\n".join(f"- {d}" for d in directives)
    if sheets:
        s += "\n\nFICHAS DE LOS PERSONAJES PRESENTES (datos oficiales):\n\n" + "\n\n".join(sheets)
    return s


EXTRACT_SYS = """Eres el archivista de un juego de rol. Mantienes las fichas de personajes a partir de lo ocurrido.
Responde SOLO con JSON con esta forma exacta:
{"personajes":[{"nombre":"...","nuevo":true,"campos":{"rol":"","edad":"","aspecto":"","personalidad":"","historia":"","relaciones":"","estado":""}}]}

Reglas:
- El PROTAGONISTA (el personaje del jugador) siempre debe tener ficha, con "rol":"protagonista". Si aún no existe, créala con lo que diga la descripción inicial.
- Crea ficha ("nuevo":true) solo de personajes con nombre que sean relevantes y vayan a volver a aparecer. No crees fichas de figurantes, multitudes ni personajes de un solo uso.
- Para personajes que ya tienen ficha ("nuevo":false), incluye únicamente los campos que hayan cambiado, con el valor completo ya actualizado.
- "aspecto" debe ser una descripción física concreta (pelo, ojos, ropa, rasgos) pensada para dibujar al personaje.
- "estado" = situación actual (dónde está, cómo se encuentra). "relaciones" = vínculos con otros personajes.
- No inventes datos que no se desprendan del texto, salvo el aspecto físico de un personaje nuevo.
- Si no hay nada que registrar, devuelve {"personajes":[]}. Escribe los valores en español."""


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"personajes": []}


def update_sheets(gid, g, context_text, new_reply):
    chars = load_chars(gid)
    known = ", ".join(c["name"] for c in chars) or "(ninguno todavía)"
    rel = mentioned(chars, context_text + "\n" + new_reply)
    sheets = "\n\n".join(sheet_text(c["name"], c["data"]) for c in rel) or "(ninguna)"
    user = f"""DESCRIPCIÓN INICIAL DEL JUEGO:
{g['setup']}

PERSONAJES CON FICHA: {known}

FICHAS RELEVANTES ACTUALES:
{sheets}

ÚLTIMOS ACONTECIMIENTOS:
{context_text}

RESPUESTA DEL NARRADOR:
{new_reply}"""
    r = chat(g["model"], [{"role": "system", "content": EXTRACT_SYS}, {"role": "user", "content": user}], fmt="json")
    out = parse_json(json.loads(r.read())["message"]["content"])
    touched = []
    for p in out.get("personajes", []):
        name = (p.get("nombre") or "").strip()
        campos = {k: v for k, v in (p.get("campos") or {}).items() if k in CAMPOS and v}
        if not name:
            continue
        ex = find_char(chars, name)
        if ex:
            ex["data"].update(campos)
            save_char(gid, ex["name"], ex["data"], ex["id"])
            touched.append(ex["name"])
        elif campos:
            save_char(gid, name, campos)
            chars.append({"id": None, "name": name, "data": campos})
            touched.append(name)
    return touched


# ---------- Turno ----------
def turn(gid, kind, text, emit):
    with db() as c:
        g = c.execute("SELECT * FROM games WHERE id=?", (gid,)).fetchone()
        directives = json.loads(g["directives"])
        if kind == "director":
            directives.append(text)
            c.execute("UPDATE games SET directives=? WHERE id=?", (json.dumps(directives, ensure_ascii=False), gid))
        if kind in ("player", "director"):
            c.execute("INSERT INTO messages(game_id,kind,content,ts) VALUES(?,?,?,?)", (gid, kind, text, time.time()))
        hist = [dict(r) for r in c.execute(
            "SELECT kind,content FROM messages WHERE game_id=? ORDER BY id DESC LIMIT 24", (gid,))][::-1]

    chars = load_chars(gid)
    recent = " ".join(h["content"] for h in hist[-6:]) + " " + text + " " + g["setup"] * (kind == "start")
    present = [c for c in chars if c["data"].get("rol") == "protagonista"]
    present += [c for c in mentioned(chars, recent) if c not in present]
    sheets = [sheet_text(c["name"], c["data"]) for c in present]

    msgs = [{"role": "system", "content": system_prompt(g, directives, sheets)}]
    for h in hist:
        if h["kind"] == "narrator":
            msgs.append({"role": "assistant", "content": h["content"]})
        elif h["kind"] == "director":
            msgs.append({"role": "user", "content": "[DIRECTOR] " + h["content"]})
        else:
            msgs.append({"role": "user", "content": h["content"]})
    if kind == "start":
        msgs.append({"role": "user", "content": "[DIRECTOR] Comienza la partida: presenta el mundo y la escena inicial en la que se encuentra el protagonista."})

    reply = ""
    for line in chat(g["model"], msgs, stream=True):
        d = json.loads(line)
        tok = d.get("message", {}).get("content", "")
        if tok:
            reply += tok
            emit({"t": "token", "v": tok})
    with db() as c:
        c.execute("INSERT INTO messages(game_id,kind,content,ts) VALUES(?,?,?,?)", (gid, "narrator", reply, time.time()))

    emit({"t": "status", "v": "Actualizando fichas…"})
    try:
        ctx = "\n".join(f"[{h['kind']}] {h['content']}" for h in hist[-4:])
        touched = update_sheets(gid, g, ctx, reply)
        emit({"t": "chars", "v": touched})
    except Exception as e:  # las fichas no deben romper el turno
        emit({"t": "status", "v": f"No se pudieron actualizar las fichas: {e}"})
    emit({"t": "done"})


# ---------- HTTP ----------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            b = (BASE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif p == "/api/models":
            try:
                tags = json.loads(ollama("/api/tags", timeout=5).read())
                self.send_json({"models": [m["name"] for m in tags["models"]]})
            except Exception:
                self.send_json({"models": [], "error": "No se pudo conectar con Ollama"})
        elif p == "/api/games":
            with db() as c:
                rows = c.execute("SELECT id,name,model,created FROM games ORDER BY created DESC").fetchall()
            self.send_json([dict(r) for r in rows])
        elif m := re.fullmatch(r"/api/games/(\d+)", p):
            gid = int(m[1])
            with db() as c:
                g = c.execute("SELECT * FROM games WHERE id=?", (gid,)).fetchone()
                msgs = c.execute("SELECT kind,content FROM messages WHERE game_id=? ORDER BY id", (gid,)).fetchall()
            if not g:
                return self.send_json({"error": "no existe"}, 404)
            self.send_json({"game": dict(g), "messages": [dict(x) for x in msgs], "characters": load_chars(gid)})
        else:
            self.send_json({"error": "no encontrado"}, 404)

    def do_POST(self):
        p, b = self.path, self.body()
        if p == "/api/games":
            with db() as c:
                cur = c.execute("INSERT INTO games(name,setup,rules,model,created) VALUES(?,?,?,?,?)",
                                (b["name"], b["setup"], b.get("rules", ""), b["model"], time.time()))
            return self.send_json({"id": cur.lastrowid})
        if m := re.fullmatch(r"/api/games/(\d+)/turn", p):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(ev):
                self.wfile.write((json.dumps(ev, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()
            try:
                turn(int(m[1]), b["kind"], b.get("text", ""), emit)
            except Exception as e:
                try:
                    emit({"t": "error", "v": f"Error con el modelo: {e}. ¿Está Ollama en marcha y el modelo descargado?"})
                except Exception:
                    pass
            return
        if m := re.fullmatch(r"/api/games/(\d+)/undo", p):  # borra el último turno completo
            gid = int(m[1])
            with db() as c:
                rows = c.execute("SELECT id,kind FROM messages WHERE game_id=? ORDER BY id DESC LIMIT 2", (gid,)).fetchall()
                if rows and rows[0]["kind"] == "narrator":
                    ids = [rows[0]["id"]] + ([rows[1]["id"]] if len(rows) > 1 and rows[1]["kind"] != "narrator" else [])
                    c.executemany("DELETE FROM messages WHERE id=?", [(i,) for i in ids])
            return self.send_json({"ok": True})
        if m := re.fullmatch(r"/api/games/(\d+)/characters", p):
            save_char(int(m[1]), b["name"], b["data"])
            return self.send_json({"ok": True})
        self.send_json({"error": "no encontrado"}, 404)

    def do_PUT(self):
        if m := re.fullmatch(r"/api/games/(\d+)/characters/(\d+)", self.path):
            b = self.body()
            save_char(int(m[1]), b["name"], b["data"], int(m[2]))
            return self.send_json({"ok": True})
        self.send_json({"error": "no encontrado"}, 404)

    def do_DELETE(self):
        with db() as c:
            if m := re.fullmatch(r"/api/games/(\d+)/characters/(\d+)", self.path):
                c.execute("DELETE FROM characters WHERE id=? AND game_id=?", (int(m[2]), int(m[1])))
            elif m := re.fullmatch(r"/api/games/(\d+)", self.path):
                g = int(m[1])
                for t in ("messages", "characters"):
                    c.execute(f"DELETE FROM {t} WHERE game_id=?", (g,))
                c.execute("DELETE FROM games WHERE id=?", (g,))
        self.send_json({"ok": True})


if __name__ == "__main__":
    init()
    print(f"Juego de rol listo en http://localhost:{PORT}  (Ollama: {OLLAMA})")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
