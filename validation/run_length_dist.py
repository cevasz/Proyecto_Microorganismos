"""
validation/run_length_dist.py
=============================
Valida la distribución de duraciones de runs (tau_run) del agente PPO entrenado
contra la distribución exponencial de Berg & Brown 1972.

Berg & Brown (Nature, 1972) encontraron que los runs de E. coli siguen una
distribución aproximadamente exponencial con media ~1.0 s.

Uso:
    python -m validation.run_length_dist --model models/ppo_chemotaxis_final.zip \
                                         --episodes 200 --output validation/figures/
    python -m validation.run_length_dist --random-only --episodes 200
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Backend sin pantalla para entornos headless
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# Asegurar que el directorio raíz del proyecto esté en el path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.environment import BacteriumEnv
from modules.rl_agent.baselines import RandomWalker


# ---------------------------------------------------------------------------
# Parámetros biológicos de referencia (Berg & Brown 1972)
# ---------------------------------------------------------------------------
BERG_BROWN_MEAN_RUN_S = 1.0      # Duración media de run ~1 s
BERG_BROWN_STD_RUN_S  = 1.0      # Distribución exponencial: std = mean
ALPHA_MANN_WHITNEY    = 0.05     # Nivel de significancia estadística


def collect_run_lengths(policy, env: BacteriumEnv, n_episodes: int) -> np.ndarray:
    """
    Ejecuta n_episodes episodios con la política dada y recopila todas las
    duraciones de runs individuales registradas por el entorno.

    Args:
        policy: callable(obs) → action, o None para política aleatoria.
        env:    Instancia de BacteriumEnv ya inicializada.
        n_episodes: Número de episodios a ejecutar.

    Returns:
        np.ndarray 1D con todas las duraciones de runs en segundos.
    """
    all_runs = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=False)
            else:
                action = env.action_space.sample()

            obs, _, terminated, truncated, info = env.step(action)

        # Recopilar duraciones de runs del episodio (info del último paso)
        # BacteriumEnv acumula run_durations[] durante el episodio
        all_runs.extend(env.run_durations)

    return np.array(all_runs, dtype=np.float64)


def collect_run_lengths_random(n_episodes: int,
                                config_path: str = "configs/default.yaml") -> np.ndarray:
    """Recopila run lengths de un agente aleatorio (RandomWalker)."""
    env = BacteriumEnv(config_path=config_path)
    walker = RandomWalker(env.action_space)

    all_runs = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = walker.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
        all_runs.extend(env.run_durations)

    env.close() if hasattr(env, 'close') else None
    return np.array(all_runs, dtype=np.float64)


def fit_exponential(run_lengths: np.ndarray):
    """
    Ajusta una distribución exponencial a los run lengths usando MLE.
    La distribución exponencial tiene un solo parámetro: lambda = 1/mean.

    Returns:
        (lambda_fit, mean_fit, std_fit)
    """
    if len(run_lengths) == 0:
        return None, None, None
    loc, scale = stats.expon.fit(run_lengths, floc=0)
    mean_fit = scale
    std_fit = scale  # Para exponencial: std = mean = 1/lambda
    lambda_fit = 1.0 / scale if scale > 0 else 0.0
    return lambda_fit, mean_fit, std_fit


def ks_test_exponential(run_lengths: np.ndarray):
    """
    Realiza el test KS (Kolmogorov-Smirnov) contra distribución exponencial
    con media empírica de la muestra. 

    Returns:
        (ks_statistic, p_value, passed)
    """
    if len(run_lengths) < 5:
        return None, None, False

    mean_est = np.mean(run_lengths)
    scale_est = mean_est  # Para exponencial: scale = mean

    ks_stat, p_val = stats.kstest(
        run_lengths,
        "expon",
        args=(0, scale_est)
    )
    passed = p_val > ALPHA_MANN_WHITNEY
    return ks_stat, p_val, passed


def mann_whitney_test(runs_rl: np.ndarray, runs_random: np.ndarray):
    """
    Test Mann-Whitney U (no paramétrico) para comparar las distribuciones
    de run lengths entre el agente RL y el random walker.

    H₀: Los run lengths del agente RL provienen de la misma distribución
        que los del random walker.
    H₁: El agente RL tiene distribución significativamente diferente.

    Returns:
        (u_statistic, p_value, rl_better)
    """
    if len(runs_rl) < 5 or len(runs_random) < 5:
        return None, None, None

    u_stat, p_val = stats.mannwhitneyu(
        runs_rl,
        runs_random,
        alternative="two-sided"
    )
    # El agente RL es "mejor" si sus runs son más largos (más eficiente)
    rl_better = np.median(runs_rl) > np.median(runs_random) and p_val < ALPHA_MANN_WHITNEY
    return u_stat, p_val, rl_better


def plot_run_length_comparison(runs_rl: np.ndarray,
                                runs_random: np.ndarray,
                                output_dir: str,
                                berg_brown_mean: float = BERG_BROWN_MEAN_RUN_S):
    """
    Genera figura de comparación de distribuciones de run lengths:
    - Histograma del agente RL
    - Histograma del random walker
    - CDF exponencial teórica de Berg & Brown 1972
    - Ajuste exponencial empírico al agente RL

    Guarda como PNG en output_dir/run_length_comparison.png.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")

    # ---- Paleta cromática ----
    COLOR_RL      = "#58a6ff"   # Azul brillante (agente RL)
    COLOR_RANDOM  = "#f78166"   # Rojo-naranja (random walker)
    COLOR_THEORY  = "#3fb950"   # Verde (teórico Berg & Brown)
    COLOR_FIT     = "#e3b341"   # Amarillo (ajuste empírico)
    BG            = "#161b22"   # Fondo del gráfico

    x_max = max(
        np.percentile(runs_rl, 98) if len(runs_rl) > 0 else 5.0,
        np.percentile(runs_random, 98) if len(runs_random) > 0 else 5.0,
        4.0
    )
    x_range = np.linspace(0, x_max, 500)

    # ---- Panel izquierdo: PDF (histograma) ----
    ax1 = axes[0]
    ax1.set_facecolor(BG)

    bins = np.linspace(0, x_max, 40)

    if len(runs_rl) > 0:
        ax1.hist(runs_rl, bins=bins, density=True, alpha=0.65,
                 color=COLOR_RL, label=f"Agente RL (n={len(runs_rl)})", zorder=3)
    if len(runs_random) > 0:
        ax1.hist(runs_random, bins=bins, density=True, alpha=0.50,
                 color=COLOR_RANDOM, label=f"Random Walker (n={len(runs_random)})", zorder=2)

    # PDF exponencial teórica (Berg & Brown: λ = 1/1.0 s)
    pdf_theory = stats.expon.pdf(x_range, loc=0, scale=berg_brown_mean)
    ax1.plot(x_range, pdf_theory, "--", color=COLOR_THEORY, linewidth=2.5,
             label=f"Berg & Brown 1972\n(Exp. μ={berg_brown_mean:.1f} s)", zorder=5)

    # Ajuste exponencial empírico al agente RL
    if len(runs_rl) >= 10:
        _, mean_fit, _ = fit_exponential(runs_rl)
        if mean_fit:
            pdf_fit = stats.expon.pdf(x_range, loc=0, scale=mean_fit)
            ax1.plot(x_range, pdf_fit, "-", color=COLOR_FIT, linewidth=2.0,
                     label=f"Ajuste RL empírico\n(Exp. μ={mean_fit:.2f} s)", zorder=4)

    ax1.set_xlabel("Duración de run τ (s)", color="white", fontsize=12)
    ax1.set_ylabel("Densidad de probabilidad", color="white", fontsize=12)
    ax1.set_title("Distribución de Run Lengths", color="white", fontsize=14, fontweight="bold")
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#30363d")
    ax1.set_xlim(0, x_max)
    legend = ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=10)
    ax1.grid(True, alpha=0.2, color="#30363d")

    # ---- Panel derecho: CDF empírica ----
    ax2 = axes[1]
    ax2.set_facecolor(BG)

    if len(runs_rl) > 0:
        sorted_rl = np.sort(runs_rl)
        cdf_rl = np.arange(1, len(sorted_rl) + 1) / len(sorted_rl)
        ax2.step(sorted_rl, cdf_rl, color=COLOR_RL, linewidth=2.0,
                 label="Agente RL (ECDF)", zorder=3)

    if len(runs_random) > 0:
        sorted_rw = np.sort(runs_random)
        cdf_rw = np.arange(1, len(sorted_rw) + 1) / len(sorted_rw)
        ax2.step(sorted_rw, cdf_rw, color=COLOR_RANDOM, linewidth=2.0,
                 label="Random Walker (ECDF)", zorder=2)

    # CDF teórica (Berg & Brown)
    cdf_theory = stats.expon.cdf(x_range, loc=0, scale=berg_brown_mean)
    ax2.plot(x_range, cdf_theory, "--", color=COLOR_THEORY, linewidth=2.5,
             label="Berg & Brown 1972 (CDF teórica)", zorder=5)

    ax2.set_xlabel("Duración de run τ (s)", color="white", fontsize=12)
    ax2.set_ylabel("Probabilidad acumulada", color="white", fontsize=12)
    ax2.set_title("CDF de Run Lengths vs. Teórico", color="white", fontsize=14, fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#30363d")
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0, 1.05)
    ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=10)
    ax2.grid(True, alpha=0.2, color="#30363d")

    plt.suptitle(
        "NeuroColony-EC | Validación de Quimiotaxis RL vs. Berg & Brown 1972",
        color="white", fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, "run_length_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Figura guardada en: {output_path}")
    return output_path


def print_report(runs_rl: np.ndarray, runs_random: np.ndarray):
    """Imprime reporte estadístico completo en consola."""
    separator = "=" * 65

    print(f"\n{separator}")
    print(" REPORTE DE VALIDACIÓN — NeuroColony-EC Tarea 4/4")
    print(f"{separator}")

    # ---- Estadísticas descriptivas ----
    print("\n📊  ESTADÍSTICAS DESCRIPTIVAS")
    print(f"  {'Métrica':<35} {'Agente RL':>10}  {'Random Walker':>13}")
    print(f"  {'-'*60}")

    def fmt(arr, fn):
        return f"{fn(arr):.3f}" if len(arr) > 0 else "  N/A"

    print(f"  {'N (runs totales)':<35} {len(runs_rl):>10}  {len(runs_random):>13}")
    print(f"  {'Media τ (s)':<35} {fmt(runs_rl, np.mean):>10}  {fmt(runs_random, np.mean):>13}")
    print(f"  {'Mediana τ (s)':<35} {fmt(runs_rl, np.median):>10}  {fmt(runs_random, np.median):>13}")
    print(f"  {'Desv. estándar τ (s)':<35} {fmt(runs_rl, np.std):>10}  {fmt(runs_random, np.std):>13}")
    print(f"  {'τ P25 (s)':<35} {fmt(runs_rl, lambda x: np.percentile(x,25)):>10}  {fmt(runs_random, lambda x: np.percentile(x,25)):>13}")
    print(f"  {'τ P75 (s)':<35} {fmt(runs_rl, lambda x: np.percentile(x,75)):>10}  {fmt(runs_random, lambda x: np.percentile(x,75)):>13}")
    print(f"  {'Referencia Berg&Brown (media)':<35} {'~1.0 s':>10}  {'—':>13}")

    # ---- Test KS vs exponencial ----
    print("\n📐  TEST KS — ¿Siguen distribución exponencial?")
    for label, runs in [("Agente RL", runs_rl), ("Random Walker", runs_random)]:
        ks, p, passed = ks_test_exponential(runs)
        if ks is not None:
            status = "✅ PASA" if passed else "❌ FALLA"
            print(f"  {label:<20}  KS={ks:.4f}  p={p:.4f}  {status} (α={ALPHA_MANN_WHITNEY})")
        else:
            print(f"  {label:<20}  Datos insuficientes")

    # ---- Test Mann-Whitney RL vs RandomWalker ----
    print("\n🔬  TEST MANN-WHITNEY — ¿Difieren significativamente?")
    u, p, rl_better = mann_whitney_test(runs_rl, runs_random)
    if u is not None:
        sig = "p < 0.05 ✅" if p < ALPHA_MANN_WHITNEY else f"p = {p:.4f} (no sig.)"
        direction = "RL > RandomWalker ✅" if rl_better else "RL ≤ RandomWalker"
        print(f"  U={u:.1f}  p={p:.4f}  {sig}")
        print(f"  Dirección: {direction}")
    else:
        print("  Datos insuficientes para el test.")

    # ---- Ajuste exponencial al agente RL ----
    print("\n📈  AJUSTE EXPONENCIAL AL AGENTE RL")
    lam, mean_fit, _ = fit_exponential(runs_rl)
    if lam is not None:
        error_pct = abs(mean_fit - BERG_BROWN_MEAN_RUN_S) / BERG_BROWN_MEAN_RUN_S * 100
        within_2sigma = abs(mean_fit - BERG_BROWN_MEAN_RUN_S) <= 2.0 * BERG_BROWN_STD_RUN_S
        status = "✅ Dentro de 2σ de Berg&Brown" if within_2sigma else "⚠️  Fuera de 2σ de Berg&Brown"
        print(f"  λ ajustada = {lam:.4f} s⁻¹")
        print(f"  Media ajustada = {mean_fit:.3f} s  (referencia: {BERG_BROWN_MEAN_RUN_S:.1f} s)")
        print(f"  Error relativo = {error_pct:.1f}%")
        print(f"  {status}")

    print(f"\n{separator}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Validación de run lengths del agente RL vs. Berg & Brown 1972"
    )
    parser.add_argument(
        "--model",
        default="models/ppo_chemotaxis_final.zip",
        help="Ruta al modelo PPO entrenado (.zip de stable-baselines3)"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Ruta al archivo de configuración YAML"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Número de episodios de evaluación para recopilar run lengths"
    )
    parser.add_argument(
        "--output",
        default="validation/figures/",
        help="Directorio de salida para las figuras"
    )
    parser.add_argument(
        "--random-only",
        action="store_true",
        help="Evaluar solo el random walker (útil si aún no hay modelo entrenado)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Omitir generación de figura (solo reporte en consola)"
    )
    args = parser.parse_args()

    print("\n🔬 NeuroColony-EC | Validación de Run Lengths — Tarea 4/4")
    print(f"   Referencia: Berg & Brown, Nature 239(5374):500-504, 1972")
    print(f"   Distribución teórica: Exponencial(μ={BERG_BROWN_MEAN_RUN_S:.1f} s)\n")

    # ---- Recopilar run lengths del Random Walker ----
    print(f"▶ Recopilando run lengths del Random Walker ({args.episodes} episodios)...")
    runs_random = collect_run_lengths_random(
        n_episodes=args.episodes,
        config_path=args.config
    )
    print(f"  → {len(runs_random)} runs recopilados (Random Walker)")

    # ---- Recopilar run lengths del agente RL ----
    if args.random_only:
        print("\n⚠️  Modo --random-only activado: omitiendo carga del modelo RL.")
        runs_rl = np.array([])
    else:
        try:
            from stable_baselines3 import PPO
            print(f"▶ Cargando modelo PPO desde: {args.model}")
            model = PPO.load(args.model)
            env_rl = BacteriumEnv(config_path=args.config)
            print(f"▶ Recopilando run lengths del agente RL ({args.episodes} episodios)...")
            runs_rl = collect_run_lengths(model, env_rl, n_episodes=args.episodes)
            print(f"  → {len(runs_rl)} runs recopilados (Agente RL)")
        except FileNotFoundError:
            print(f"⚠️  Modelo no encontrado en '{args.model}'. Usando solo Random Walker.")
            runs_rl = np.array([])
        except Exception as e:
            print(f"⚠️  Error al cargar/evaluar modelo RL: {e}")
            runs_rl = np.array([])

    # ---- Reporte estadístico ----
    print_report(runs_rl, runs_random)

    # ---- Figura comparativa ----
    if not args.no_plot and (len(runs_rl) > 0 or len(runs_random) > 0):
        print("▶ Generando figura comparativa...")
        plot_run_length_comparison(
            runs_rl=runs_rl,
            runs_random=runs_random,
            output_dir=args.output
        )

    print("✅ Validación completada.\n")


if __name__ == "__main__":
    main()
