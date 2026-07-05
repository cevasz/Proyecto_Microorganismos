"""
Construye el `index.html` desplegable (autocontenido) a partir de la plantilla
`viewer.html`, incrustando la telemetria de la celula (`cell_internal.json`) y una
serie de poblacion de la colonia como contexto (extraida de `scenario_ppo.json`).

El resultado funciona con doble clic (file://) y en Hugging Face sin usar `fetch`.

Uso:  python build_index.py
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))


def load_pop(max_points=140):
    path = os.path.join(HERE, "scenario_ppo.json")
    if not os.path.exists(path):
        return []
    d = json.load(open(path, encoding="utf-8"))
    N = [fr["stats"]["N"] for fr in d["frames"]]
    if len(N) <= max_points:
        return N
    step = len(N) / max_points
    return [N[int(i * step)] for i in range(max_points)]


def build():
    tpl = open(os.path.join(HERE, "viewer.html"), encoding="utf-8").read()

    cell = json.load(open(os.path.join(HERE, "cell_internal.json"), encoding="utf-8"))
    pop = load_pop()

    embed = {"cell": cell, "pop": pop}
    embed_js = "<script>window.__EMBED__=" + json.dumps(embed, separators=(",", ":")) + ";</script>\n"

    marker = '<script>\n"use strict";'
    if marker not in tpl:
        raise SystemExit("No se encontro el punto de insercion del script principal en viewer.html")
    out = tpl.replace(marker, embed_js + marker, 1)

    dst = os.path.join(HERE, "index.html")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(out)
    mb = os.path.getsize(dst) / 1024 / 1024
    print(f"index.html generado (autocontenido) - {mb:.2f} MB - celula:{cell['n_frames']} fotogramas - poblacion:{len(pop)} puntos")

    # tambien exportar colony_pop.json para el modo fetch de viewer.html
    with open(os.path.join(HERE, "colony_pop.json"), "w") as f:
        json.dump(pop, f, separators=(",", ":"))


if __name__ == "__main__":
    build()
