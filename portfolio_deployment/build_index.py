"""
Construye el `index.html` desplegable (autocontenido) a partir de la plantilla
`viewer.html`, incrustando:
  - la telemetria de UNA celula (`cell_internal.json`) para la pestaña CÉLULA, y
  - las trayectorias completas de la colonia (`scenario_ppo.json` y
    `scenario_extinction.json`) para la vista principal COLONIA.

El resultado funciona con doble clic (file://) y en Hugging Face sin usar `fetch`.

Uso:  python build_index.py
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def build():
    tpl = open(os.path.join(HERE, "viewer.html"), encoding="utf-8").read()

    cell = _load("cell_internal.json")
    ppo = _load("scenario_ppo.json")
    ext = _load("scenario_extinction.json")

    embed = {"cell": cell, "scenarios": {"ppo": ppo, "ext": ext}}
    embed_js = "<script>window.__EMBED__=" + json.dumps(embed, separators=(",", ":")) + ";</script>\n"

    marker = '<script>\n"use strict";'
    if marker not in tpl:
        raise SystemExit("No se encontro el punto de insercion del script principal en viewer.html")
    out = tpl.replace(marker, embed_js + marker, 1)

    dst = os.path.join(HERE, "index.html")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(out)

    mb = os.path.getsize(dst) / 1024 / 1024
    nppo = len(ppo["frames"]) if ppo else 0
    next_ = len(ext["frames"]) if ext else 0
    ncell = cell.get("n_frames") if cell else 0
    print(f"index.html generado (autocontenido) - {mb:.2f} MB - "
          f"colonia PPO:{nppo} frames - colapso:{next_} frames - celula:{ncell} frames")


if __name__ == "__main__":
    build()
