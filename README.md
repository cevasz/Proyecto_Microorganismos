<div align="center">
  
#  NeuroColony-EC
  
**Simulación Multiagente de Colonias de *Escherichia coli***<br>
*Combinando Aprendizaje por Refuerzo, Neural ODEs y LSTM*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-8A2BE2)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

##  Descripción del Proyecto

**NeuroColony-EC** es un simulador multiagente avanzado diseñado para estudiar el comportamiento de quimiotaxis y crecimiento poblacional de la bacteria *Escherichia coli*. 

A diferencia de los simuladores tradicionales basados puramente en reglas biofísicas, este proyecto integra técnicas de **Inteligencia Artificial** de vanguardia para modelar diferentes escalas del comportamiento celular:

1. **Aprendizaje por Refuerzo (RL)**: Navegación y quimiotaxis (*Run-and-Tumble*).
2. **Neural ODEs**: Dinámica molecular y señalización interna (CheA/CheY).
3. **Redes LSTM**: Predicción del ciclo de división celular y crecimiento.

---

##  Arquitectura Modular

El proyecto está dividido en tres módulos principales de Machine Learning, gobernados por un entorno de simulación espacial 2D:

###  Módulo 1: Quimiotaxis RL (`modules/rl_agent`)
- El agente aprende a navegar gradientes de nutrientes (glucosa) evadiendo zonas de bajo oxígeno.
- Entrenado utilizando **PPO** (Proximal Policy Optimization).
- El espacio de acción define la duración del *run* y la probabilidad de *tumble*.

###  Módulo 2: Señalización Interna (`modules/neural_ode`)
- Simula la red de fosforilación de CheY utilizando ecuaciones diferenciales ordinarias neuronales (**Neural ODEs**).
- Implementa una arquitectura *Physics-Informed* (PI-NODE) para respetar la conservación de masa.
- Calibrado mediante un modelo cinético biofísico de señalización (Spiro et al., 1997) adaptado y parametrizado con los rangos dinámicos y escalas temporales inspirados por el set de datos de transcriptómica real NCBI GEO GSE4513, garantizando total transparencia metodológica y rigor científico.

###  Módulo 3: División Celular (`modules/lstm_division`)
- Un modelo de series temporales **LSTM** predice el momento exacto de la bipartición basándose en la historia metabólica y el tamaño celular.
- Valida la hipótesis del *Adder Model* (Taheri-Araghi et al., 2015).

---

##  Estructura del Repositorio

```text
neurocolony-ec/
├── agents/              # Entidad Bacterium (acopla RL + ODE + LSTM)
├── configs/             # Hiperparámetros biológicos (default.yaml)
├── data/                # Datasets externos (GEO GSE4513, Taheri 2015)
├── env/                 # Entorno de EDP y Gym Wrapper
├── modules/             # Implementaciones de ML
│   ├── lstm_division/   # Predicción de división
│   ├── neural_ode/      # Dinámica CheA/CheY
│   └── rl_agent/        # Políticas de Quimiotaxis
├── simulation/          # Bucle principal y renderizado Pygame
├── tests/               # Pruebas unitarias y de integración
└── validation/          # Scripts de validación estadística vs Biología
```

---

##  Instalación y Configuración

Se recomienda el uso de `conda` o un entorno virtual (`venv`/`uv`) para manejar las dependencias.

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/cevasz/Proyecto_Microorganismos.git
   cd Proyecto_Microorganismos
   ```

2. **Crear entorno virtual e instalar dependencias:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Verificar instalación:**
   ```bash
   python -c "import torch, torchdiffeq, stable_baselines3; print(' Entorno configurado correctamente')"
   ```

---

##  Uso Rápido (En desarrollo)

Todos los parámetros físicos y biológicos están centralizados. Puedes modificarlos en `configs/default.yaml`.

Para iniciar la simulación visual (próximamente):
```bash
python -m simulation.runner --agents 50 --steps 100000 --render True
```

---

##  Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Siéntete libre de usarlo, modificarlo y distribuirlo.

---
<div align="center">
  <i>Desarrollado para el avance de la Biología Computacional • 2026</i>
</div>
