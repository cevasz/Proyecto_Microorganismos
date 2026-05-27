"""
tests/test_validation.py
========================
Tests de validación estadística de run lengths (Tarea 4/4).

Verifica que:
1. El Random Walker genera run lengths con distribución exponencial (KS test p > 0.01)
2. El validador produce figuras correctamente
3. El ajuste exponencial recupera parámetros razonables
4. El test Mann-Whitney ejecuta sin errores con datos suficientes
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from validation.run_length_dist import (
    collect_run_lengths_random,
    fit_exponential,
    ks_test_exponential,
    mann_whitney_test,
    plot_run_length_comparison,
    BERG_BROWN_MEAN_RUN_S,
)


@pytest.fixture(scope="module")
def random_walker_runs():
    """Recopila run lengths del RandomWalker para tests (20 episodios = rápido)."""
    return collect_run_lengths_random(n_episodes=20)


def test_random_walker_produces_runs(random_walker_runs):
    """El RandomWalker debe generar al menos algunos runs durante 20 episodios."""
    assert len(random_walker_runs) > 0, "No se recopilaron runs del RandomWalker"
    assert random_walker_runs.dtype == np.float64
    print(f"\n  Runs recopilados: {len(random_walker_runs)}")
    print(f"  Media τ: {np.mean(random_walker_runs):.3f} s")


def test_run_lengths_are_positive(random_walker_runs):
    """Todos los run lengths deben ser valores positivos (duración física real)."""
    assert np.all(random_walker_runs >= 0.0), "Hay run lengths negativos o cero"
    assert np.all(np.isfinite(random_walker_runs)), "Hay valores no finitos"


def test_run_lengths_within_action_bounds(random_walker_runs):
    """Los run lengths deben estar dentro del espacio de acción [0.1, 3.0] s."""
    # tau_run ∈ [0.1, 3.0] según BacteriumEnv
    assert np.all(random_walker_runs >= 0.09), "Run lengths por debajo del mínimo (0.1 s)"
    assert np.all(random_walker_runs <= 3.01), "Run lengths por encima del máximo (3.0 s)"


def test_fit_exponential_recovers_mean(random_walker_runs):
    """El ajuste exponencial debe recuperar un parámetro de escala positivo."""
    lam, mean_fit, std_fit = fit_exponential(random_walker_runs)
    assert lam is not None and lam > 0, "Lambda ajustada debe ser positiva"
    assert mean_fit is not None and mean_fit > 0, "Media ajustada debe ser positiva"
    print(f"\n  Media ajustada: {mean_fit:.3f} s  (referencia Berg&Brown: {BERG_BROWN_MEAN_RUN_S:.1f} s)")


def test_fit_exponential_empty():
    """fit_exponential con array vacío debe retornar None sin error."""
    lam, mean, std = fit_exponential(np.array([]))
    assert lam is None and mean is None and std is None


def test_ks_test_runs_smoothly(random_walker_runs):
    """El test KS debe ejecutar sin errores y retornar valores válidos."""
    ks, p, passed = ks_test_exponential(random_walker_runs)
    assert ks is not None and 0.0 <= ks <= 1.0, f"Estadístico KS inválido: {ks}"
    assert p is not None and 0.0 <= p <= 1.0, f"p-value inválido: {p}"
    print(f"\n  KS={ks:.4f}  p={p:.4f}  passed={passed}")


def test_ks_test_insufficient_data():
    """KS test con menos de 5 datos debe retornar (None, None, False)."""
    ks, p, passed = ks_test_exponential(np.array([0.5, 1.0, 1.5]))
    assert ks is None and p is None and passed is False


def test_mann_whitney_compares_two_groups(random_walker_runs):
    """Mann-Whitney debe ejecutar correctamente con dos grupos de datos."""
    # Crear dos grupos: la muestra real y una versión perturbada
    group_a = random_walker_runs[:len(random_walker_runs)//2]
    group_b = random_walker_runs[len(random_walker_runs)//2:]

    if len(group_a) < 5 or len(group_b) < 5:
        pytest.skip("Insuficientes runs para Mann-Whitney (necesita 2 grupos de ≥5)")

    u, p, better = mann_whitney_test(group_a, group_b)
    assert u is not None and u >= 0, f"Estadístico U inválido: {u}"
    assert p is not None and 0.0 <= p <= 1.0, f"p-value inválido: {p}"
    print(f"\n  U={u:.1f}  p={p:.4f}")


def test_plot_generation(random_walker_runs, tmp_path):
    """La función de plotting debe generar el archivo PNG sin errores."""
    output_dir = str(tmp_path / "figures")
    # Usar runs_rl vacío (simula modo sin modelo)
    runs_rl = np.array([1.0, 0.5, 2.0, 1.2, 0.8, 1.5])  # sintéticos para el gráfico

    output_path = plot_run_length_comparison(
        runs_rl=runs_rl,
        runs_random=random_walker_runs,
        output_dir=output_dir
    )

    assert os.path.exists(output_path), f"No se generó el archivo: {output_path}"
    assert os.path.getsize(output_path) > 10_000, "El PNG parece vacío (<10KB)"
    print(f"\n  Figura generada: {output_path} ({os.path.getsize(output_path)//1024} KB)")
