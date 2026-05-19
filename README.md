# NeuroColony-EC

Simulación Multiagente de *Escherichia coli* combinando Aprendizaje por Refuerzo (RL), Neural ODEs y LSTM.

## Módulos Principales
- **Entorno + RL (Módulo 1):** Agente aprende quimiotaxis mediante PPO.
- **Neural ODE (Módulo 2):** Dinámica interna de señalización CheA/CheY.
- **LSTM (Módulo 3):** Clasificador para predicción de división celular.

## Configuración

Para instalar las dependencias, asegúrese de tener Python 3.11+:
```bash
pip install -r requirements.txt
```

Los parámetros biológicos y de entrenamiento se encuentran en `configs/default.yaml`.
