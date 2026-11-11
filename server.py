#!/usr/bin/env python3
"""Cronicas: juego de rol y simulacion con IA local.
Requisitos: Python 3.9+ y Ollama (https://ollama.com) en marcha. Sin dependencias extra.
Uso:  python server.py   ->  abre http://localhost:8000
Variables opcionales: OLLAMA_URL, PORT, NUM_CTX (contexto del modelo, por defecto 8192)
"""
import json, math, os, random, re, sqlite3, time, urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

BASE = Path(__file__).parent
DB = BASE / "partidas.db"
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
PORT = int(os.environ.get("PORT", 8000))
NUM_CTX = int(os.environ.get("NUM_CTX", 8192))
CAMPOS = ["rol", "edad", "aspecto", "personalidad", "historia", "relaciones", "estado", "notas"]
TIPOS_LUGAR = ["capital", "ciudad", "pueblo", "fortaleza", "ruina", "otro"]

# Tipos de acontecimiento que el motor puede pedir al narrador (se eligen 3 al azar en cada empuje)
EVENTOS = [
    "una complicacion o imprevisto que dificulte lo que el jugador esta haciendo",
    "una oportunidad inesperada que el jugador puede aprovechar o ignorar",
    "la llegada o aparicion de un personaje (nuevo o ya conocido) con su propia agenda",
    "una noticia, rumor o descubrimiento sobre el mundo o sobre alguien importante",
    "una consecuencia de algo que el jugador hizo o dijo antes",
    "una iniciativa de un personaje con ficha que actua por su cuenta, sin esperar al jugador",
    "un cambio en el entorno (clima, tension politica, conflicto, fiesta, peligro cercano)",
    "un avance o giro en una de las tramas abiertas",
    "un encuentro con un lugar o persona de otra parte del mundo",
]


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
        CREATE TABLE IF NOT EXISTS places(id INTEGER PRIMARY KEY, game_id INTEGER, name TEXT, kind TEXT, descr TEXT,
            x REAL, y REAL, UNIQUE(game_id, name));
        CREATE TABLE IF NOT EXISTS routes(game_id INTEGER, a INTEGER, b INTEGER, UNIQUE(game_id, a, b));
        """)
        for col, ddl in [("threads", "TEXT DEFAULT '[]'"), ("freq", "INTEGER DEFAULT 3"),
                         ("since_event", "INTEGER DEFAULT 0"), ("location", "TEXT DEFAULT ''")]:
            try:
                c.execute(f"ALTER TABLE games ADD COLUMN {col} {ddl}")
            except sqlite3.OperationalError:
                pass  # la columna ya existe


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


# ---------- Mapa ----------
DIRS = {"norte": (0, -1), "noreste": (.707, -.707), "este": (1, 0), "sureste": (.707, .707),
        "sur": (0, 1), "suroeste": (-.707, .707), "oeste": (-1, 0), "noroeste": (-.707, -.707)}
DIST = {"cerca": 16, "media": 26, "lejos": 40}


def norm_kind(t):
    t = (t or "").lower()
    for k in ("capital", "ciudad", "pueblo", "aldea", "fortaleza", "castillo", "ruina"):
        if k in t:
            return {"aldea": "pueblo", "castillo": "fortaleza"}.get(k, k)
    return "otro"


def load_places(gid):
    with db() as c:
        rows = c.execute("SELECT * FROM places WHERE game_id=?", (gid,)).fetchall()
    return [{"id": r["id"], "name": r["name"], "kind": r["kind"], "desc": r["descr"] or "", "x": r["x"], "y": r["y"]}
            for r in rows]


def load_routes(gid):
    with db() as c:
        return [{"a": r["a"], "b": r["b"]} for r in c.execute("SELECT a,b FROM routes WHERE game_id=?", (gid,))]


def find_place(places, name):
    n = (name or "").strip().lower()
    if not n:
        return None
    for p in places:
        if p["name"].lower() == n:
            return p
    for p in places:
        pn = p["name"].lower()
        if len(n) > 3 and (n in pn or pn in n):
            return p
    return None


def place_pos(places, ref=None, direction=None, dist=None):
    """Calcula coordenadas a partir de una posicion relativa (el modelo no maneja coordenadas fiables)."""
    if not places:
        return 0.0, 0.0
    r = find_place(places, ref) if ref else None
    if r:
        bx, by = r["x"], r["y"]
    else:
        bx = sum(p["x"] for p in places) / len(places)
        by = sum(p["y"] for p in places) / len(places)
    d = DIRS.get((direction or "").strip().lower())
    a0 = math.atan2(d[1], d[0]) if d else len(places) * 2.39996
    step = DIST.get((dist or "").strip().lower(), 26 if r else 34)
    first = None
    for off in (0, .5, -.5, 1, -1, 1.5, -1.5, 2, -2, 2.5, -2.5, 3):
        for mult in (1, 1.4, 1.8):
            x = bx + math.cos(a0 + off) * step * mult
            y = by + math.sin(a0 + off) * step * mult
            first = first or (x, y)
            if all(math.hypot(x - p["x"], y - p["y"]) >= 12 for p in places):
                return x, y
    return first


def apply_places(gid, lugares):
    places = load_places(gid)
    pending = []
    with db() as c:
        for p in lugares:
            if not isinstance(p, dict):
                continue
            name = (p.get("nombre") or "").strip()
            if not name:
                continue
            kind = norm_kind(p.get("tipo"))
            desc = (p.get("descripcion") or "").strip()
            ex = find_place(places, name)
            if ex and ex["name"].lower() == name.lower():
                if desc:
                    c.execute("UPDATE places SET descr=? WHERE id=?", (desc, ex["id"]))
                if p.get("tipo"):
                    c.execute("UPDATE places SET kind=? WHERE id=?", (kind, ex["id"]))
            else:
                x, y = place_pos(places, p.get("cerca_de"), p.get("direccion"), p.get("distancia"))
                cur = c.execute("INSERT OR IGNORE INTO places(game_id,name,kind,descr,x,y) VALUES(?,?,?,?,?,?)",
                                (gid, name, kind, desc, x, y))
                if cur.lastrowid:
                    places.append({"id": cur.lastrowid, "name": name, "kind": kind, "desc": desc, "x": x, "y": y})
            pending.append((name, p.get("conexiones") or []))
        for name, conns in pending:
            a = find_place(places, name)
            for cn in conns if isinstance(conns, list) else []:
                b = find_place(places, cn if isinstance(cn, str) else "")
                if a and b and a["id"] != b["id"]:
                    lo, hi = sorted((a["id"], b["id"]))
                    c.execute("INSERT OR IGNORE INTO routes(game_id,a,b) VALUES(?,?,?)", (gid, lo, hi))
    return [n for n, _ in pending]


# ---------- Prompts ----------
def system_prompt(g, directives, sheets, places, threads):
    s = f"""Eres el NARRADOR y motor de simulacion de un juego de rol por texto. No hay dados ni azar mecanico: tu decides las consecuencias de forma coherente con el mundo, las reglas y las fichas de los personajes.

Como jugar:
- El jugador escribe lo que dice o hace su personaje. Interpretalo con naturalidad y narra que ocurre como consecuencia; segun lo que diga puede pasar una cosa u otra.
- Nunca decidas por el jugador lo que dice, piensa o hace su personaje. Solo narras el mundo y a los demas.
- El mundo esta vivo: los personajes tienen metas, personalidad y memoria propias y pueden actuar por iniciativa propia. Tu trabajo es SIMULAR, no solo reaccionar.
- NO satures la historia de sucesos. Si ya hay algo en marcha (una conversacion, un conflicto, una busqueda, un viaje con sentido), deja que se desarrolle con calma y a su ritmo, sin anadir giros ni interrupciones.
- Solo cuando la escena este parada (nada ocurre, nadie quiere nada, el jugador no tiene un rumbo claro) toma tu la iniciativa: haz que pase algo concreto en lugar de preguntar al jugador que quiere hacer.
- Respeta estrictamente las fichas y la geografia: no mezcles caracteristicas entre personajes ni contradigas datos ya establecidos.
- Los mensajes marcados como [DIRECTOR] son instrucciones del jugador sobre la historia o el mundo (cambiar algo, corregir un error, anadir un personaje o lugar). Aplicalas de inmediato y sin discutir, y continua la escena ya corregida.
- Las notas marcadas como [NOTA DEL MOTOR] son ordenes internas: cumplelas sin mencionarlas ni nombrarlas.
- Escribe en espanol, con prosa narrativa y dialogos. Extension media (2-5 parrafos). Termina dejando margen para que el jugador responda.

MUNDO Y PROTAGONISTA (descrito por el jugador):
{g['setup']}"""
    if g["rules"].strip():
        s += f"\n\nREGLAS DE LA PARTIDA (impuestas por el jugador, obligatorias):\n{g['rules']}"
    if directives:
        s += "\n\nINSTRUCCIONES PERMANENTES DEL DIRECTOR (siguen vigentes):\n" + "\n".join(f"- {d}" for d in directives)
    if places:
        s += "\n\nLUGARES CONOCIDOS DEL MUNDO:\n" + "\n".join(places)
    if g["location"]:
        s += f"\nUbicacion actual del protagonista: {g['location']}"
    if threads:
        s += "\n\nTRAMAS ABIERTAS (hazlas avanzar, complicarlas o resolverlas a su ritmo):\n" + "\n".join(f"- {t}" for t in threads)
    if sheets:
        s += "\n\nFICHAS DE LOS PERSONAJES PRESENTES (datos oficiales):\n\n" + "\n\n".join(sheets)
    return s


def event_note():
    tipos = "; ".join(random.sample(EVENTOS, 3))
    return ("\n\n[NOTA DEL MOTOR - invisible para el jugador] La historia lleva unos turnos sin avanzar: no esta pasando nada. "
            f"Haz que ocurra ahora algo nuevo y concreto. Elige lo que mejor encaje de entre: {tipos}. Hazlo surgir con "
            "naturalidad del mundo, de los personajes y de las tramas abiertas, integralo en la narracion sin llamarlo "
            "'evento' y sin resolverlo del todo: dejalo abierto para que el jugador reaccione.")


EXTRACT_SYS = """Eres el archivista de un juego de rol. Mantienes las fichas de personajes, el mapa y las tramas a partir de lo ocurrido.
Responde SOLO con JSON con esta forma exacta:
{"personajes":[{"nombre":"...","nuevo":true,"campos":{"rol":"","edad":"","aspecto":"","personalidad":"","historia":"","relaciones":"","estado":""}}],
 "lugares":[{"nombre":"...","tipo":"capital|ciudad|pueblo|fortaleza|ruina|otro","descripcion":"...","cerca_de":"nombre de un lugar ya existente","direccion":"norte|noreste|este|sureste|sur|suroeste|oeste|noroeste","distancia":"cerca|media|lejos","conexiones":["nombres de lugares unidos por camino o ruta"]}],
 "hilos":["..."],
 "ubicacion_protagonista":"nombre de un lugar o vacio",
 "escena_estancada":false}

PERSONAJES
- El PROTAGONISTA (el personaje del jugador) siempre debe tener ficha, con "rol":"protagonista". Si aun no existe, crea la con la descripcion inicial.
- Crea ficha ("nuevo":true) solo de personajes con nombre que sean relevantes y vayan a volver a aparecer. No crees fichas de figurantes, multitudes ni personajes de un solo uso.
- Para personajes que ya tienen ficha ("nuevo":false), incluye unicamente los campos que hayan cambiado, con el valor completo ya actualizado.
- "aspecto" debe ser una descripcion fisica concreta (pelo, ojos, ropa, rasgos) pensada para dibujar al personaje.
- "estado" = situacion actual (donde esta, como se encuentra). "relaciones" = vinculos con otros personajes.
LUGARES
- Registra ciudades, pueblos, fortalezas, ruinas y enclaves importantes con nombre propio que existan en el mundo, aunque solo se mencionen (por ejemplo en la descripcion inicial). No registres habitaciones, tiendas ni sitios menores.
- Incluye solo lugares nuevos o con cambios. Para un lugar nuevo indica "cerca_de" (otro lugar que ya exista), "direccion" y "distancia" respecto a el cuando se pueda deducir; si es el primero del mapa, omitelos.
- "conexiones" solo con lugares ya existentes o presentes en esta misma lista.
HILOS Y UBICACION
- "hilos" = lista COMPLETA y actualizada de tramas abiertas (misterios, conflictos, amenazas, objetivos pendientes), frases cortas, maximo 6. Quita las resueltas y anade las nuevas.
- "ubicacion_protagonista" = lugar del mapa donde esta ahora el protagonista, o "" si no esta en ninguno.
ESTANCAMIENTO
- "escena_estancada" = true SOLO si, tras la respuesta del narrador, no esta pasando nada relevante: sin conflicto, amenaza, objetivo o conversacion con rumbo, y las tramas abiertas no avanzan (rutina, charla vacia, tiempo muerto). Es false si hay cualquier cosa en marcha, aunque sea tranquila, o si acaba de ocurrir algo que el jugador debe digerir o resolver.
GENERAL
- No inventes datos que no se desprendan del texto, salvo el aspecto fisico de un personaje nuevo y la posicion relativa de lugares.
- Si no hay nada que registrar en una seccion, deja su lista vacia. Escribe los valores en espanol."""


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
    return {}


def update_world(gid, g, context_text, new_reply):
    chars = load_chars(gid)
    places = load_places(gid)
    threads = json.loads(g["threads"] or "[]")
    known = ", ".join(c["name"] for c in chars) or "(ninguno todavia)"
    rel = mentioned(chars, context_text + "\n" + new_reply)
    sheets = "\n\n".join(sheet_text(c["name"], c["data"]) for c in rel) or "(ninguna)"
    plist = "\n".join(f"- {p['name']} ({p['kind']})" for p in places) or "(ninguno todavia)"
    user = f"""DESCRIPCION INICIAL DEL JUEGO:
{g['setup']}

PERSONAJES CON FICHA: {known}

FICHAS RELEVANTES ACTUALES:
{sheets}

LUGARES YA EN EL MAPA:
{plist}

TRAMAS ABIERTAS ACTUALES:
{json.dumps(threads, ensure_ascii=False)}

ULTIMOS ACONTECIMIENTOS:
{context_text}

RESPUESTA DEL NARRADOR:
{new_reply}"""
    r = chat(g["model"], [{"role": "system", "content": EXTRACT_SYS}, {"role": "user", "content": user}], fmt="json")
    out = parse_json(json.loads(r.read())["message"]["content"])

    touched = []
    for p in out.get("personajes") or []:
        if not isinstance(p, dict):
            continue
        name = (p.get("nombre") or "").strip()
        campos = {k: v for k, v in (p.get("campos") or {}).items() if k in CAMPOS and v and isinstance(v, str)}
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

    lugares = apply_places(gid, out.get("lugares") or [])

    with db() as c:
        hilos = out.get("hilos")
        if isinstance(hilos, list):
            hilos = [h.strip() for h in hilos if isinstance(h, str) and h.strip()][:8]
            c.execute("UPDATE games SET threads=? WHERE id=?", (json.dumps(hilos, ensure_ascii=False), gid))
        ub = find_place(load_places(gid), out.get("ubicacion_protagonista") or "")
        if ub:
            c.execute("UPDATE games SET location=? WHERE id=?", (ub["name"], gid))
        quiet = out.get("escena_estancada") is True
        c.execute("UPDATE games SET since_event=? WHERE id=?", ((g["since_event"] or 0) + 1 if quiet else 0, gid))
    return touched, lugares


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
        # el motor solo interviene si la historia lleva tantos turnos estancada como tolera la partida (1-3)
        fire = kind == "player" and (g["since_event"] or 0) >= max(1, min(3, g["freq"] or 2))
        if fire:
            c.execute("UPDATE games SET since_event=0 WHERE id=?", (gid,))
        hist = [dict(r) for r in c.execute(
            "SELECT kind,content FROM messages WHERE game_id=? ORDER BY id DESC LIMIT 24", (gid,))][::-1]
        g = c.execute("SELECT * FROM games WHERE id=?", (gid,)).fetchone()

    chars = load_chars(gid)
    places = load_places(gid)
    names = {p["id"]: p["name"] for p in places}
    conns = {}
    for r in load_routes(gid):
        conns.setdefault(r["a"], []).append(names.get(r["b"], "?"))
        conns.setdefault(r["b"], []).append(names.get(r["a"], "?"))
    plines = []
    for p in places:
        line = f"- {p['name']} ({p['kind']})"
        if p["desc"]:
            line += f": {p['desc']}"
        if conns.get(p["id"]):
            line += f" [conectado con: {', '.join(conns[p['id']])}]"
        plines.append(line)

    recent = " ".join(h["content"] for h in hist[-6:]) + " " + text + " " + g["setup"] * (kind == "start")
    present = [c for c in chars if c["data"].get("rol") == "protagonista"]
    present += [c for c in mentioned(chars, recent) if c not in present]
    sheets = [sheet_text(c["name"], c["data"]) for c in present]

    msgs = [{"role": "system", "content": system_prompt(g, directives, sheets, plines, json.loads(g["threads"] or "[]"))}]
    for h in hist:
        if h["kind"] == "narrator":
            msgs.append({"role": "assistant", "content": h["content"]})
        elif h["kind"] == "director":
            msgs.append({"role": "user", "content": "[DIRECTOR] " + h["content"]})
        else:
            msgs.append({"role": "user", "content": h["content"]})
    if kind == "start":
        msgs.append({"role": "user", "content": "[DIRECTOR] Comienza la partida: presenta el mundo y la escena inicial en la que se encuentra el protagonista. "
                     "La escena debe incluir ya un detonante, problema o gancho que invite a actuar."})
    if fire:
        msgs[-1]["content"] += event_note()

    reply = ""
    for line in chat(g["model"], msgs, stream=True):
        d = json.loads(line)
        tok = d.get("message", {}).get("content", "")
        if tok:
            reply += tok
            emit({"t": "token", "v": tok})
    with db() as c:
        c.execute("INSERT INTO messages(game_id,kind,content,ts) VALUES(?,?,?,?)", (gid, "narrator", reply, time.time()))

    emit({"t": "status", "v": "Actualizando fichas, mapa y tramas…"})
    try:
        ctx = "\n".join(f"[{h['kind']}] {h['content']}" for h in hist[-4:])
        touched, lugares = update_world(gid, g, ctx, reply)
        emit({"t": "world", "chars": touched, "places": lugares})
    except Exception as e:  # el archivo del mundo no debe romper el turno
        emit({"t": "status", "v": f"No se pudo actualizar el mundo: {e}"})
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
            game = dict(g)
            game["threads"] = json.loads(game.get("threads") or "[]")
            self.send_json({"game": game, "messages": [dict(x) for x in msgs], "characters": load_chars(gid),
                            "places": load_places(gid), "routes": load_routes(gid)})
        else:
            self.send_json({"error": "no encontrado"}, 404)

    def do_POST(self):
        p, b = self.path, self.body()
        if p == "/api/games":
            with db() as c:
                cur = c.execute("INSERT INTO games(name,setup,rules,model,created,freq) VALUES(?,?,?,?,?,?)",
                                (b["name"], b["setup"], b.get("rules", ""), b["model"], time.time(), int(b.get("freq", 2))))
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
                    emit({"t": "error", "v": f"Error con el modelo: {e}. ¿Esta Ollama en marcha y el modelo descargado?"})
                except Exception:
                    pass
            return
        if m := re.fullmatch(r"/api/games/(\d+)/undo", p):  # borra el ultimo turno completo
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
        if m := re.fullmatch(r"/api/games/(\d+)/places", p):
            gid = int(m[1])
            x, y = place_pos(load_places(gid))
            with db() as c:
                c.execute("INSERT OR IGNORE INTO places(game_id,name,kind,descr,x,y) VALUES(?,?,?,?,?,?)",
                          (gid, b["name"], norm_kind(b.get("kind")), b.get("desc", ""), x, y))
            return self.send_json({"ok": True})
        self.send_json({"error": "no encontrado"}, 404)

    def do_PUT(self):
        b = self.body()
        if m := re.fullmatch(r"/api/games/(\d+)/characters/(\d+)", self.path):
            save_char(int(m[1]), b["name"], b["data"], int(m[2]))
            return self.send_json({"ok": True})
        if m := re.fullmatch(r"/api/games/(\d+)/places/(\d+)", self.path):
            gid, pid = int(m[1]), int(m[2])
            with db() as c:
                old = c.execute("SELECT name FROM places WHERE id=? AND game_id=?", (pid, gid)).fetchone()
                if old:
                    if "name" in b and b["name"].strip():
                        c.execute("UPDATE places SET name=? WHERE id=?", (b["name"].strip(), pid))
                        c.execute("UPDATE games SET location=? WHERE id=? AND location=?", (b["name"].strip(), gid, old["name"]))
                    if "kind" in b:
                        c.execute("UPDATE places SET kind=? WHERE id=?", (norm_kind(b["kind"]), pid))
                    if "desc" in b:
                        c.execute("UPDATE places SET descr=? WHERE id=?", (b["desc"], pid))
                    if "x" in b and "y" in b:
                        c.execute("UPDATE places SET x=?, y=? WHERE id=?", (float(b["x"]), float(b["y"]), pid))
            return self.send_json({"ok": True})
        if m := re.fullmatch(r"/api/games/(\d+)", self.path):
            with db() as c:
                if "freq" in b:
                    c.execute("UPDATE games SET freq=? WHERE id=?", (max(1, min(3, int(b["freq"]))), int(m[1])))
            return self.send_json({"ok": True})
        self.send_json({"error": "no encontrado"}, 404)

    def do_DELETE(self):
        with db() as c:
            if m := re.fullmatch(r"/api/games/(\d+)/characters/(\d+)", self.path):
                c.execute("DELETE FROM characters WHERE id=? AND game_id=?", (int(m[2]), int(m[1])))
            elif m := re.fullmatch(r"/api/games/(\d+)/places/(\d+)", self.path):
                g, pid = int(m[1]), int(m[2])
                c.execute("DELETE FROM routes WHERE game_id=? AND (a=? OR b=?)", (g, pid, pid))
                c.execute("DELETE FROM places WHERE id=? AND game_id=?", (pid, g))
            elif m := re.fullmatch(r"/api/games/(\d+)", self.path):
                g = int(m[1])
                for t in ("messages", "characters", "places", "routes"):
                    c.execute(f"DELETE FROM {t} WHERE game_id=?", (g,))
                c.execute("DELETE FROM games WHERE id=?", (g,))
        self.send_json({"ok": True})


if __name__ == "__main__":
    init()
    print(f"Cronicas listo en http://localhost:{PORT}  (Ollama: {OLLAMA})")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
