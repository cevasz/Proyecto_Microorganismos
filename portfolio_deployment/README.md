---
title: NeuroColony-EC
emoji: 🦠
colorFrom: green
colorTo: yellow
sdk: static
pinned: true
license: mit
short_description: Colonia de E. coli · quimiotaxis PPO y división por LSTM
---

# NeuroColony-EC — Colonia de *E. coli*

Simulación **multiagente** de una colonia de *Escherichia coli*: múltiples bacterias
se desplazan continuamente por el medio buscando nutrientes, ascendiendo un gradiente
químico **real** (difusión FDM de EDPs) y proliferando por división celular. Estética
**maximalista** de laboratorio sobre fondo verde claro.

El visor tiene dos vistas:

- **Colonia (principal):** el plato de cultivo 100×100 µm con el campo de glucosa y las
  bacterias **fucsia** con textura de grumos, orientadas según su dirección (el tono
  fucsia se aclara u oscurece con la energía).
  Dos escenarios conmutables:
  - **Quimiotaxis inteligente (PPO):** las bacterias ascienden el gradiente hacia la
    fuente y proliferan hasta la capacidad de carga del medio.
  - **Colapso ecológico (random):** sin quimiotaxis, se dispersan, agotan su energía y
    la población colapsa.
- **Célula (secundaria):** corte esquemático de una célula (circuitos metabólicos,
  ribosomas, nucleoide) animado con la telemetría interna.

La **quimiotaxis** la decide una política **PPO** entrenada, la **P(división)** la
calcula el **LSTM entrenado** del proyecto (`models/lstm_division_best.pt`) y la
señalización interna procede de un **PI-NODE**.

Es un **visor estático**: `index.html` es autocontenido (datos embebidos), carga al
instante y **funciona con solo abrirlo** (doble clic). Nota: al reproducir un escenario
grabado, las bacterias no llevan estela por-agente (su identidad no se conserva entre
fotogramas serializados); en la simulación en vivo de escritorio sí.

## Contenido del Space

```
index.html   # visor autocontenido: HTML + CSS + JS + trayectorias embebidas
README.md    # esta tarjeta
```

Basta subir esos dos archivos a un Space de tipo **Static**.

### Fuentes (en el repositorio del proyecto)

```
viewer.html               # plantilla del visor (editable)
scenario_ppo.json         # trayectoria de la colonia PPO (posiciones, energía, glucosa)
scenario_extinction.json  # trayectoria del colapso ecológico
cell_internal.json        # telemetría de la célula (energía, ciclo, P(división) del LSTM)
generate_data.py          # corre la simulación real -> scenario_*.json + reconstruye index.html
generate_cell.py          # genera cell_internal.json desde el LSTM real
build_index.py            # incrusta las trayectorias en la plantilla -> index.html
```

Regenerar:

```bash
python portfolio_deployment/generate_cell.py   # telemetría de la célula (LSTM real)
python portfolio_deployment/generate_data.py    # colonia (PPO + colapso) + index.html
```

Solo reconstruir `index.html` tras editar `viewer.html`:

```bash
cd portfolio_deployment && python build_index.py
```

Parámetros de URL: `?view=colony|cell`, `?scenario=ppo|ext`, `?frame=N`, `?play=1`.

Ver `DEPLOY.md` para el despliegue paso a paso.
