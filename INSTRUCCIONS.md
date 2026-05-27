# NeuroColony-EC — Guía de Desarrollo con Vibe Coding
> Simulación Multiagente de *E. coli* | RL + Neural ODE + LSTM
> Departamento de IA · Tunja, Colombia · 2026

---

## ¿Qué es este documento?

Esta guía convierte el informe técnico IEEE de NeuroColony-EC en instrucciones accionables para desarrollo asistido por IA (*vibe coding*). Está diseñada para **dos desarrolladores trabajando en paralelo**, con responsabilidades divididas por módulo y una línea de prompts lista para copiar-pegar en Claude, Cursor, o cualquier asistente de código.

---

## Stack tecnológico recomendado

| Capa | Tecnología | Justificación |
|------|-----------|---------------|
| Lenguaje principal | **Python 3.11+** | Ecosistema ML/SciPy consolidado |
| RL (Módulo 1) | **Stable-Baselines3** (PPO/SAC) | API limpia, compatible con Gym |
| Neural ODE (Módulo 2) | **torchdiffeq** | Implementación de Chen et al. 2018 |
| LSTM (Módulo 3) | **PyTorch** | Control fino sobre arquitectura |
| Entorno / EDP | **NumPy + SciPy (FDM)** | Diferencias finitas 2D |
| Visualización | **Matplotlib + Pygame** (en tiempo real) | Trayectorias + campos escalares |
| Gestión de experimentos | **Weights & Biases** o **MLflow** | Tracking de métricas por módulo |
| Entorno virtual | **conda** o **uv** | Reproducibilidad |
| Control de versiones | **Git + GitHub** | Ramas por módulo |

---

## Estructura de carpetas del proyecto

```
neurocolony-ec/
├── env/                        # Entorno de simulación (EDP)
│   ├── grid.py                 # Campos escalares glucosa/O2
│   ├── diffusion.py            # Solver FDM
│   └── environment.py          # Gym-compatible wrapper
│
├── modules/
│   ├── rl_agent/               # Módulo 1 — Quimiotaxis RL
│   │   ├── agent.py            # Wrapper PPO/SAC (SB3)
│   │   ├── policy.py           # Red neuronal de política
│   │   └── reward.py           # Función de recompensa
│   │
│   ├── neural_ode/             # Módulo 2 — Señalización CheA/CheY
│   │   ├── ode_func.py         # f(z, t; θ) aprendible
│   │   ├── pi_node.py          # Physics-Informed Neural ODE
│   │   └── trainer.py          # Entrenamiento desde GEO GSE4513
│   │
│   └── lstm_division/          # Módulo 3 — División celular
│       ├── model.py            # Clasificador LSTM binario
│       ├── dataset.py          # Loader datos Taheri-Araghi 2015
│       └── trainer.py
│
├── agents/
│   └── bacterium.py            # Agente bacteria: integra los 3 módulos
│
├── simulation/
│   ├── colony.py               # Gestor de N agentes
│   ├── runner.py               # Loop principal de simulación
│   └── visualizer.py          # Render en tiempo real
│
├── validation/
│   ├── run_length_dist.py      # Validación vs. Berg & Brown 1972
│   ├── chey_rmse.py            # Validación Neural ODE vs. fluorescencia
│   └── adder_model.py          # Validación LSTM vs. Taheri-Araghi 2015
│
├── data/
│   ├── geo_gse4513/            # Datos expresión génica (NCBI GEO)
│   └── taheri_2015/            # Tiempos de división (microscopía)
│
├── configs/
│   └── default.yaml            # Hiperparámetros centralizados
│
├── tests/                      # Unit tests por módulo
├── notebooks/                  # EDA y prototipado
├── requirements.txt
└── README.md
```

---

## División de trabajo: 2 desarrolladores

### Dev A — Entorno + RL (Módulo 1)
**Responsabilidad:** Todo lo que el agente *percibe* y *decide*.

- Implementar el grid 2D con campos escalares (glucosa, O₂)
- Solver FDM para ecuación de difusión-reacción
- Wrapper de entorno compatible con Gymnasium (Gym API)
- Módulo RL: PPO como algoritmo principal, SAC como baseline
- Función de recompensa calibrada biológicamente
- Visualizador en tiempo real de trayectorias
- Tests de validación: distribución de run lengths vs. Berg & Brown 1972

### Dev B — Neural ODE + LSTM (Módulos 2 y 3)
**Responsabilidad:** Todo lo que ocurre *dentro* de la célula.

- Neural ODE para dinámica CheA/CheY (`torchdiffeq`)
- Extensión Physics-Informed (PI-NODE): conservación de masa + Michaelis-Menten
- Pipeline de datos desde NCBI GEO GSE4513
- Clasificador LSTM para predicción de división celular
- Dataset de Taheri-Araghi 2015 (adder model)
- Integración del módulo de división: spawn de agentes hijos
- Tests de validación: RMSE de CheY-P y distribución de tiempos de división

### Punto de integración 
Ambos devs se unen en `agents/bacterium.py` para acoplar los tres módulos y probar el sistema end-to-end con N=5 agentes antes de escalar.

---

## Fases de desarrollo

### Fase 1 — MVP 
**Objetivo:** Un agente solo supera a un random walker en eficiencia de búsqueda (p < 0.05)

**Dev A:**
1. Grid 2D básico con gradiente estático de glucosa
2. Agente RL con estado mínimo (gradiente local, posición)
3. Entrenamiento PPO por 500K pasos
4. Visualización de trayectoria única

**Dev B:**
1. Prototipo Neural ODE con datos sintéticos (antes de GEO real)
2. LSTM stub: devuelve división cada 1200 pasos (hardcoded para MVP)
3. Infraestructura de datos y loaders

**Hito de cierre Fase 1:** `python simulation/runner.py --agents 1 --steps 100000` muestra quimiotaxis emergente visible.

---

### Fase 2 — Biología Completa 
**Objetivo:** Run lengths dentro de 2σ de datos Berg & Brown 1972; N > 100 agentes estables

**Dev A:**
- Dinámica de glucosa y O₂ en tiempo real (consumo bidireccional)
- Escalado a N > 100 agentes (paralelización con vectorized envs de SB3)
- Validación cuantitativa de distribución de run lengths

**Dev B:**
- Neural ODE entrenada en datos reales GEO GSE4513
- PI-NODE: restricciones físicas sobre la función aprendida
- LSTM entrenado en datos Taheri-Araghi 2015
- División celular activa: spawn de agentes hijos

**Hito de cierre Fase 2:** `validation/run_length_dist.py` produce KS-test p > 0.05 contra datos biológicos.

---

### Fase 3 — Publicación 
**Objetivo:** Paper submission a NeurIPS Workshop / ICLR Workshop o conferencia IEEE

- Análisis comparativo: política RL aprendida vs. mecanismo CheA/CheY real
- Curva de crecimiento poblacional vs. datos experimentales
- Ablation study: PI-NODE vs. Neural ODE sin restricciones físicas
- Redacción de paper en LaTeX (template NeurIPS o IEEE)
- Preparación de figuras y código reproducible

---

## Configuración inicial del entorno

```bash
# 1. Clonar y crear entorno
git clone https://github.com/tu-equipo/neurocolony-ec.git
cd neurocolony-ec
conda create -n neurocolony python=3.11
conda activate neurocolony

# 2. Instalar dependencias
pip install torch torchdiffeq stable-baselines3[extra] gymnasium
pip install numpy scipy matplotlib pygame pyyaml wandb
pip install pytest black isort

# 3. Verificar instalación
python -c "import torch; import torchdiffeq; import stable_baselines3; print('OK')"

# 4. Descargar datos (Dev B)
# NCBI GEO GSE4513: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4513
# Taheri-Araghi 2015: datos en supplementary del paper (Current Biology 25(3))
```

---

## Parámetros biológicos clave (configs/default.yaml)

```yaml
environment:
  grid_size: [200, 200]        # μm × μm
  dx: 1.0                      # μm por celda
  dt: 0.01                     # s por paso
  D_glucose: 600               # μm²/s
  D_oxygen: 2000               # μm²/s

bacterium:
  run_speed: 25.0              # μm/s (promedio 20–30)
  tumble_duration: 0.1         # s
  run_duration_min: 0.1        # s
  run_duration_max: 3.0        # s
  division_length: 4.0         # μm (umbral de fisión)
  division_time_mean: 1200     # s (~20 min a 37°C)

rl:
  algorithm: PPO
  policy: MlpPolicy
  total_timesteps: 2_000_000
  learning_rate: 3.0e-4
  gamma: 0.99
  reward_alpha: 1.0            # peso nutrientes
  reward_gamma: 0.1            # peso costo energético

neural_ode:
  hidden_dim: 64
  state_dim: 4                 # [CheY-P, ppGpp, metilación, energía]
  solver: dopri5               # Dormand-Prince RK45
  rtol: 1.0e-3
  atol: 1.0e-4

lstm:
  input_dim: 3                 # [longitud, energía, tasa replicación ADN]
  hidden_dim: 128
  num_layers: 2
  sequence_length: 60          # pasos temporales de historia
  division_horizon: 30         # pasos para predecir división
```

---


## Checklist de hitos por fase

### Fase 1 — MVP ✓
- [x] Grid 2D con gradiente estático de glucosa funcional
- [x] Wrapper Gymnasium pasa `check_env()` sin errores
- [x] PPO entrena sin divergir (recompensa media creciente en primeros 200K pasos)
- [x] El agente supera al random walker en distancia media a la fuente (Mann-Whitney p < 0.05)
- [x] Visualización básica de trayectoria única operativa

### Fase 2 — Biología Completa ✓
- [x] Neural ODE entrenada: MSE < 0.05 en validación de CheY-P
- [x] LSTM: precisión > 80% en predicción de división (F1 sobre clase positiva)
- [x] División celular activa: colonia crece de N₀=10 a N>50 en 10K pasos con glucosa abundante
- [x] Run lengths: KS-test p > 0.05 contra distribución exponencial de Berg & Brown
- [x] Sistema estable con N=100 agentes durante 50K pasos sin crash de memoria

### Fase 3 — Publicación ✓
- [ ] Figura 1: comparativa trayectorias RL vs. random walker
- [ ] Figura 2: distribución run lengths (RL vs. biológico) con bandas 2σ
- [ ] Figura 3: curva de crecimiento poblacional vs. modelo logístico experimental
- [ ] Ablation: PI-NODE vs. Neural ODE plain → diferencia en RMSE CheY-P documentada
- [ ] Paper en LaTeX listo para submission

---

## Recursos y referencias rápidas

| Recurso | URL |
|---------|-----|
| NCBI GEO GSE4513 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4513 |
| torchdiffeq (Chen 2018) | https://github.com/rtqichen/torchdiffeq |
| Stable-Baselines3 | https://stable-baselines3.readthedocs.io |
| BioNumbers (parámetros físicos) | https://bionumbers.hms.harvard.edu |
| EcoCyc (vía CheA/CheY) | https://ecocyc.org/pathway?orgid=ECOLI&id=CHEMOTAXIS-PWY |
| Berg & Brown 1972 (Nature) | DOI: 10.1038/239500a0 |
| Taheri-Araghi 2015 (Curr Biol) | DOI: 10.1016/j.cub.2014.12.009 |
| NeurIPS Workshop template | https://neurips.cc/Conferences/2025/CallForWorkshops |

---

*NeuroColony-EC — Guía Vibe Coding v1.0 · Mayo 2026*
