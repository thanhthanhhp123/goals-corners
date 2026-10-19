"""Bundle web/ and web/data/ into one self-contained HTML file (no server needed).

Team logos are downloaded once (48px), cached in data/logos/ and inlined as
data: URIs, so the file also works where external images are blocked.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from . import config

log = logging.getLogger(__name__)

MODULES = ["model.js", "charts.js", "app.js"]  # dependency order
DATA_FILES = ["meta", "model", "teams", "fixtures", "results"]
LOGO_URL = "https://a.espncdn.com/combiner/i?img=/i/teamlogos/soccer/500/{id}.png&h=48&w=48"
LOGO_DIR = config.DATA_DIR / "logos"

EXPORT_RE = re.compile(r"^export (?:async )?(?:function\*?|const|let|class) (\w+)", re.M)
IMPORT_RE = re.compile(r"^import \{([^}]*)\} from '\./([\w.]+)';\s*$", re.M)


def _var(module: str) -> str:
    return "__mod_" + module.replace(".js", "").replace("-", "_")


def _wrap_module(name: str, src: str) -> str:
    """Give each ES module its own function scope; imports become destructuring."""
    exports = EXPORT_RE.findall(src)
    src = IMPORT_RE.sub(lambda m: f"const {{{m.group(1)}}} = {_var(m.group(2))};", src)
    src = re.sub(r"^export (?=(?:async )?(?:function|const|let|class))", "", src, flags=re.M)
    return f"const {_var(name)} = (() => {{\n{src}\nreturn {{ {', '.join(exports)} }};\n}})();"


def _logo_bytes(team_id: str) -> bytes | None:
    path = LOGO_DIR / f"{team_id}.png"
    if path.exists():
        return path.read_bytes() or None
    try:
        resp = requests.get(LOGO_URL.format(id=team_id), timeout=20)
        data = resp.content if resp.ok and resp.headers.get("content-type", "").startswith("image/") else b""
    except requests.RequestException:
        return None  # not cached, so a later export retries
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)  # an empty file records "no logo"
    return data or None


def embed_logos(teams: dict) -> None:
    ids = [tid for tid, t in teams.items() if t.get("logo")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for tid, data in zip(ids, pool.map(_logo_bytes, ids)):
            teams[tid]["logo"] = "data:image/png;base64," + base64.b64encode(data).decode() if data else ""


def export_html(out: Path, artifact: bool = False) -> Path:
    """Write the app as one file. `artifact` writes a page fragment for hosts that add their own <head>."""
    web = config.WEB_DIR
    html = (web / "index.html").read_text(encoding="utf-8")
    css = (web / "styles.css").read_text(encoding="utf-8")
    js = "\n".join(_wrap_module(m, (web / m).read_text(encoding="utf-8")) for m in MODULES)
    js = js.replace("</script", "<\\/script")
    data = {n: json.loads((config.WEB_DATA_DIR / f"{n}.json").read_text(encoding="utf-8")) for n in DATA_FILES}
    embed_logos(data["teams"])
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    scripts = "<script>window.__DATA__ = " + blob + ";</script>\n<script type=\"module\">\n" + js + "\n</script>"
    style = "<style>\n" + css + "\n</style>"

    if artifact:
        title = re.search(r"<title>.*?</title>", html, re.S).group(0)
        desc = re.search(r'<meta name="description"[^>]*>', html).group(0)
        body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
        body = body.replace('<script type="module" src="app.js"></script>', scripts)
        page = "\n".join([title, '<meta name="claude:viewport-fit-sides" content="cover">', desc, style, body])
    else:
        page = html.replace('<link rel="stylesheet" href="styles.css">', style)
        page = page.replace('<script type="module" src="app.js"></script>', scripts)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
