---
title: NeuroColony-EC
emoji: 🦠
colorFrom: blue
colorTo: indigo
sdk: static
pinned: true
license: mit
short_description: E. coli como microfábrica · esquemático intracelular animado
---

# NeuroColony-EC — Esquemático intracelular de *E. coli*

Vista de **ingeniería** de una célula de *Escherichia coli* representada como una
**microfábrica** (estética de estructura tipo "Dyson sphere" / PCB). Un corte
esquemático anima, sobre la telemetría de la simulación, la maquinaria molecular:

- **Rutas metabólicas como circuitos impresos (PCB):** trazas con nodos-enzima y
  pulsos de sustrato que viajan hacia un banco de ATP.
- **Ribosomas como ensambladores industriales:** unen bloques de aminoácidos en
  cadenas polipeptídicas que se pliegan geométricamente.
- **Nucleoide / ADN central:** filamentos de datos que brillan durante la
  transcripción y muestran un eje de replicación en cada división.

La telemetría (energía/ATP, nutriente, transcripción, traducción) proviene de un
modelo metabólico de la célula, y la **P(división)** la calcula el **LSTM entrenado**
del proyecto (`models/lstm_division_best.pt`). Se incluye la población de la colonia
del escenario PPO como contexto.

Es un **visor estático**: `index.html` es autocontenido (datos embebidos), carga al
instante y **funciona con solo abrirlo** (doble clic).

## Contenido del Space

```
index.html   # visor autocontenido: HTML + CSS + JS + telemetría embebida
README.md    # esta tarjeta
```

Basta subir esos dos archivos a un Space de tipo **Static**.

### Fuentes (en el repositorio del proyecto)

```
viewer.html         # plantilla del visor (editable)
cell_internal.json  # telemetría de la célula (energía, ciclo, P(división) del LSTM)
generate_cell.py    # genera cell_internal.json desde el LSTM real
generate_data.py    # simulación de la colonia (para el contexto de población)
build_index.py      # incrusta la telemetría en la plantilla -> index.html
```

Regenerar:

```bash
python portfolio_deployment/generate_cell.py   # telemetría de la célula (LSTM real)
cd portfolio_deployment && python build_index.py
```

Ver `DEPLOY.md` para el despliegue paso a paso.
