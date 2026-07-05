"""
Genera la telemetria de UNA celula para la vista esquematica intracelular
(estetica "microfabrica / Dyson sphere").

Combina:
  - un curso temporal de nutriente (festin -> hambruna -> recuperacion),
  - un modelo metabolico fenomenologico de la energia (ATP) de la celula,
  - un reloj de ciclo celular acelerado,
  - la probabilidad de division del LSTM REAL entrenado (models/lstm_division_best.pt).

De estas senales el visor deriva la actividad de las rutas metabolicas (circuitos),
la transcripcion del nucleoide y el ensamblaje ribosomal.
"""
import os
import sys
import json
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.lstm_division.model import DivisionLSTM


def nutrient_curve(t, T):
    """Disponibilidad de nutriente in [0,1]: festin -> hambruna -> recuperacion."""
    x = t / T
    base = 0.9
    # valle de hambruna centrado en x~0.45
    famine = 0.78 * np.exp(-((x - 0.45) ** 2) / (2 * 0.11 ** 2))
    return float(np.clip(base - famine, 0.08, 1.0))


def main(steps=1000, save_every=4, seed=5):
    rng = np.random.default_rng(seed)
    dt = 0.01

    # LSTM de division real
    model = DivisionLSTM(input_dim=3, hidden_dim=128, num_layers=2, dropout=0.2)
    lstm_path = os.path.join(PROJECT_ROOT, "models", "lstm_division_best.pt")
    if os.path.exists(lstm_path):
        model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
    model.eval()

    # Estado de la celula
    energy = 0.85
    length = 1.0
    age = 0
    from collections import deque
    history = deque(maxlen=60)

    # Constantes fenomenologicas
    K_GAIN, K_COST = 0.020, 0.012
    CYCLE_STEPS = 210
    LEN_GROWTH = (2.0 - 1.0) / CYCLE_STEPS
    AGE_STEP = 1_200_000 / CYCLE_STEPS

    frames = []
    events = []           # indices de fotograma con division
    total_divisions = 0

    for step in range(steps):
        nut = nutrient_curve(step, steps)

        # Metabolismo: energia relaja hacia el nutriente disponible
        energy = float(np.clip(energy + nut * K_GAIN - K_COST, 0.0, 1.0))

        # Reloj de ciclo celular (modula por energia: si no hay ATP no se crece)
        growth_gate = 0.25 + 0.75 * energy
        length += LEN_GROWTH * growth_gate
        age += int(AGE_STEP * growth_gate)
        dna_rate = min(age / 1_200_000.0, 1.0)
        history.append([length, energy, dna_rate])

        # Probabilidad de division del LSTM real
        pdiv = 0.0
        if len(history) == 60:
            with torch.no_grad():
                pdiv = float(model(torch.tensor([list(history)], dtype=torch.float32)).item())

        divided = False
        if pdiv > 0.85:
            divided = True
            total_divisions += 1
            length = 1.0
            age = 0
            history.clear()

        if step % save_every == 0 or divided:
            fr = {
                "t": round(step * dt, 3),
                "energy": round(energy, 4),
                "nutrient": round(nut, 4),
                "length": round(length, 3),
                "cycle": round(min((length - 1.0) / 1.0, 1.0), 4),
                "pdiv": round(pdiv, 4),
                "div": 1 if divided else 0,
            }
            if divided:
                events.append(len(frames))
            frames.append(fr)

    out = {
        "dt": dt,
        "n_frames": len(frames),
        "divisions": total_divisions,
        "events": events,
        "frames": frames,
    }
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cell_internal.json")
    with open(dst, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"cell_internal.json · {len(frames)} fotogramas · {total_divisions} divisiones · eventos={events}")
    e = [fr["energy"] for fr in frames]
    print(f"  energia min/max = {min(e):.2f}/{max(e):.2f}")


if __name__ == "__main__":
    main()
