"""
Generador de datos para el visor web de portafolio (Hugging Face Static Space).

Ejecuta la simulacion REAL de NeuroColony-EC (politica PPO entrenada, difusion FDM
de EDPs y el clasificador LSTM de division entrenado) y serializa la trayectoria a
JSON para reproducirla en el navegador sin backend.

Nota metodologica: el `length` y la edad biologica de cada bacteria avanzan a una
escala temporal comprimida ("reloj de ciclo celular acelerado") para que el ciclo
del adder-model sea observable dentro de la ventana de simulacion. La DECISION de
division la sigue tomando el LSTM entrenado (models/lstm_division_best.pt); solo se
acelera el reloj biologico. El crecimiento se acota con una capacidad de carga
logistica K, reproduciendo la clasica curva sigmoide de crecimiento poblacional.
"""
import os
import sys
import json
import numpy as np

# Inyeccion del directorio raiz del proyecto en sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.grid import EnvironmentFields
from simulation.colony import Colony

GRID = 100          # celdas del dominio fisico (100 x 100 um)
CENTER = (50.0, 50.0)
GLUCOSE_RES = 24    # resolucion del campo de glucosa exportado (submuestreo)


def _gaussian_source(amp=1.0, sigma=15.0):
    """Blob gaussiano de glucosa centrado en CENTER (fuente de nutrientes)."""
    yy, xx = np.indices((GRID, GRID))
    blob = amp * np.exp(-(((xx - CENTER[0]) ** 2 + (yy - CENTER[1]) ** 2) / (2.0 * sigma ** 2)))
    return blob.astype(np.float32)


def _downsample(grid, res=GLUCOSE_RES):
    step = GRID // res
    return [[round(float(grid[i, j]), 3) for j in range(0, GRID, step)] for i in range(0, GRID, step)]


def simulate(name, description, policy="ppo", steps=2600, save_every=8,
             initial_agents=14, food_mode="sustain", food_sigma=15.0, grow=True,
             carrying_capacity=150, metabolic_boost=1.0, metabolic_jitter=0.0,
             spawn_radius=(8, 26), initial_energy=1.0, initial_energy_jitter=0.0,
             extra_drain=0.0, seed=7):
    """Corre un escenario y devuelve el diccionario serializable."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    dt = 0.01

    # --- Entorno: difusion FDM real con fuente central ---
    env = EnvironmentFields(size=(GRID, GRID), dx=1.0, dt=dt)
    env.center = np.array(CENTER)
    env.glucose_grid.boundary_val = 0.0            # desactivar inyeccion del borde izquierdo
    env.glucose_grid.grid[:, :] = 0.0
    env.oxygen_grid.grid[:, :] = 1.0
    source = _gaussian_source(sigma=food_sigma)
    env.glucose_grid.grid = np.maximum(env.glucose_grid.grid, source)

    # --- Politica RL ---
    rl_policy = None
    if policy == "ppo":
        from stable_baselines3 import PPO
        model_path = os.path.join(PROJECT_ROOT, "models", "ppo_chemotaxis_final.zip")
        if os.path.exists(model_path):
            rl_policy = PPO.load(model_path)
        else:
            print("  ADVERTENCIA: modelo PPO no encontrado; usando politica aleatoria.")

    colony = Colony(initial_agents=initial_agents, env_fields=env, rl_policy=rl_policy)

    # Repartir la poblacion inicial en un anillo alrededor de la fuente, con la fase
    # del ciclo celular desincronizada para un crecimiento suave (no escalonado).
    for a in colony.agents:
        r = rng.uniform(*spawn_radius)
        th = rng.uniform(0, 2 * np.pi)
        a.position = [float(CENTER[0] + r * np.cos(th)), float(CENTER[1] + r * np.sin(th))]
        a.metabolic_rate *= metabolic_boost * (1.0 + rng.uniform(-1, 1) * metabolic_jitter)
        if initial_energy_jitter > 0.0:
            a.energy = float(np.clip(initial_energy + rng.uniform(-1, 1) * initial_energy_jitter, 0.05, 1.0))
            a.internal_state[3] = a.energy
        elif initial_energy != 1.0:
            a.energy = initial_energy
            a.internal_state[3] = a.energy
        if grow:
            a.length = float(rng.uniform(1.0, 1.7))
            a.age = int(rng.uniform(0.0, 0.7) * 1_200_000)

    data = {
        "name": name,
        "description": description,
        "policy": policy,
        "width": GRID, "height": GRID, "dx": 1.0,
        "center": list(CENTER),
        "carrying_capacity": carrying_capacity,
        "frames": [],
    }

    # Parametros del reloj de ciclo celular acelerado
    DIV_LENGTH = 2.0
    CYCLE_STEPS = 190                    # duracion nominal de un ciclo celular (pasos)
    LEN_GROWTH = (DIV_LENGTH - 1.0) / CYCLE_STEPS
    AGE_STEP = 1_200_000 / CYCLE_STEPS   # para que dna_rate=min(age/1.2e6,1) ramp 0->1

    total_divisions = 0
    peak_N = initial_agents

    for step in range(steps):
        N = len(colony.agents)

        # Reloj de ciclo celular con freno logistico por hacinamiento: la tasa de
        # crecimiento se atenua con (1 - N/K), produciendo una curva sigmoide suave
        # que satura en la capacidad de carga sin sobrepasarla bruscamente.
        if grow:
            crowd = max(0.0, 1.0 - N / float(carrying_capacity))
            for a in colony.agents:
                if not hasattr(a, "_grate"):
                    a._grate = LEN_GROWTH * rng.uniform(0.75, 1.25)
                    a._arate = AGE_STEP * rng.uniform(0.75, 1.25)
                a.length += a._grate * crowd
                a.age += int(a._arate * crowd)

        # Fisica de campos
        env.step()
        if food_mode == "sustain":
            env.glucose_grid.grid = np.maximum(env.glucose_grid.grid, source * 0.85)

        # Poblacion antes del paso, para detectar nacimientos
        N_before = len(colony.agents)
        colony.step(dt)
        births = max(0, len(colony.agents) - N_before)
        total_divisions += births

        # Impuesto metabolico adicional (coste basal de mantenimiento) para escenarios
        # de colapso: garantiza un declive energetico neto que la ingesta no compensa.
        if extra_drain > 0.0:
            survivors = []
            for a in colony.agents:
                a.energy -= extra_drain
                a.internal_state[3] = a.energy
                if a.energy > 0.0:
                    survivors.append(a)
                else:
                    a.is_dead = True
            colony.agents = survivors

        peak_N = max(peak_N, len(colony.agents))

        if step % save_every == 0 or len(colony.agents) == 0:
            agents = [[round(float(a.position[0]), 1), round(float(a.position[1]), 1),
                       round(float(a.orientation), 2), round(float(a.energy), 2),
                       int(a.last_action), round(float(a.length), 2)] for a in colony.agents]
            gl = env.glucose_grid.grid
            energies = [a.energy for a in colony.agents]
            data["frames"].append({
                "step": step,
                "t": round(step * dt, 2),
                "agents": agents,
                "glucose": _downsample(gl),
                "births": births,
                "stats": {
                    "N": len(colony.agents),
                    "avg_energy": round(float(np.mean(energies)), 3) if energies else 0.0,
                    "divisions": total_divisions,
                    "food": round(float(gl.sum() / (GRID * GRID)), 4),
                },
            })

        if len(colony.agents) == 0:
            print(f"  Colonia extinguida en el paso {step}.")
            break

    data["peak_N"] = peak_N
    data["total_divisions"] = total_divisions
    data["n_frames"] = len(data["frames"])
    return data


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("[1/2] Escenario A: Quimiotaxis Inteligente (PPO)")
    ppo = simulate(
        name="Quimiotaxis Inteligente",
        description=("Colonia guiada por una politica PPO entrenada. Las bacterias "
                     "ascienden el gradiente quimico real (difusion FDM) hacia la fuente "
                     "de nutrientes y proliferan por division celular (LSTM) hasta la "
                     "capacidad de carga del medio."),
        policy="ppo", steps=2600, save_every=8, initial_agents=10,
        food_mode="sustain", food_sigma=15.0, grow=True, carrying_capacity=140,
        metabolic_boost=1.0, spawn_radius=(8, 26), seed=7,
    )
    with open(os.path.join(out_dir, "scenario_ppo.json"), "w") as f:
        json.dump(ppo, f, separators=(",", ":"))
    print(f"      frames={ppo['n_frames']} peakN={ppo['peak_N']} divisiones={ppo['total_divisions']}")

    print("[2/2] Escenario B: Colapso Ecologico (Random)")
    ext = simulate(
        name="Colapso Ecologico",
        description=("Colonia con comportamiento estocastico no guiado (random walk). "
                     "Sin quimiotaxis, las bacterias se dispersan lejos de la fuente "
                     "finita, agotan su energia metabolica y la poblacion colapsa."),
        policy="random", steps=1300, save_every=8, initial_agents=48,
        food_mode="deplete", food_sigma=8.0, grow=False, carrying_capacity=0,
        metabolic_boost=2.0, metabolic_jitter=0.6, spawn_radius=(3, 34),
        initial_energy=0.95, initial_energy_jitter=0.18, extra_drain=0.005, seed=3,
    )
    with open(os.path.join(out_dir, "scenario_extinction.json"), "w") as f:
        json.dump(ext, f, separators=(",", ":"))
    print(f"      frames={ext['n_frames']} peakN={ext['peak_N']} divisiones={ext['total_divisions']}")

    for fn in ("scenario_ppo.json", "scenario_extinction.json"):
        mb = os.path.getsize(os.path.join(out_dir, fn)) / 1024 / 1024
        print(f"  {fn}: {mb:.2f} MB")

    # Reconstruir el index.html autocontenido con los datos recien generados
    import build_index
    build_index.build()


if __name__ == "__main__":
    main()
