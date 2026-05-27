import numpy as np
import scipy.stats as stats
import pytest

def test_exponential_fit():
    """
    Verifica que el ajuste exponencial recupere adecuadamente los parámetros de escala
    de una muestra exponencial sintética (como las duraciones de corrida de Berg & Brown).
    """
    np.random.seed(42)
    true_scale = 0.86  # Media de Berg & Brown 1972
    sample = np.random.exponential(scale=true_scale, size=1000)
    
    # Ajuste exponencial con scipy (fijando loc=0)
    loc_fit, scale_fit = stats.expon.fit(sample, floc=0)
    
    assert loc_fit == 0.0
    # La escala ajustada debe estar muy cerca de la real con 1000 muestras
    assert np.isclose(scale_fit, true_scale, rtol=0.1)

def test_kolmogorov_smirnov_test():
    """
    Verifica el comportamiento del test Kolmogorov-Smirnov (KS-test) contra
    una distribución exponencial teórica. Debe pasar con p > 0.05 para muestras
    del mismo origen, y fallar (p < 0.01) para muestras de diferente distribución.
    """
    np.random.seed(42)
    mu_theoretical = 0.86
    
    # Caso 1: Muestras de la distribución exponencial correcta
    correct_sample = np.random.exponential(scale=mu_theoretical, size=500)
    ks_stat, p_val = stats.kstest(correct_sample, 'expon', args=(0, mu_theoretical))
    
    assert 0.0 <= ks_stat <= 1.0
    assert p_val > 0.05  # No se rechaza la hipótesis nula (mismo origen)
    
    # Caso 2: Muestras de una distribución uniforme (claramente incorrecta)
    wrong_sample = np.random.uniform(low=0.1, high=3.0, size=500)
    ks_stat_w, p_val_w = stats.kstest(wrong_sample, 'expon', args=(0, mu_theoretical))
    
    assert p_val_w < 0.01  # Se rechaza con alta confianza

def test_mann_whitney_u_test():
    """
    Verifica que el test no paramétrico de Mann-Whitney U diferencie correctamente
    dos poblaciones con medianas significativamente distintas.
    """
    np.random.seed(42)
    # Grupo A: Tiempos cortos de corrida
    group_a = np.random.exponential(scale=0.5, size=200)
    # Grupo B: Tiempos largos de corrida
    group_b = np.random.exponential(scale=1.5, size=200)
    
    u_stat, p_val = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
    
    assert u_stat >= 0
    assert p_val < 0.001  # Significancia estadística extrema

def test_chi_square_test():
    """
    Verifica que la prueba de Chi-cuadrado sobre histogramas compare
    de forma robusta la bondad de ajuste de frecuencias.
    """
    np.random.seed(42)
    
    # Frecuencias observadas y esperadas
    obs = np.random.normal(1200, 150, 1000)
    exp = np.random.normal(1200, 150, 1000)
    
    obs_freq, bins = np.histogram(obs, bins=15)
    exp_freq, _ = np.histogram(exp, bins=bins)
    
    # Evitar divisiones por cero con suavizado Laplace
    obs_freq = obs_freq + 1
    exp_freq = exp_freq + 1
    exp_freq = exp_freq * (obs_freq.sum() / exp_freq.sum())
    
    chi2, p_val = stats.chisquare(f_obs=obs_freq, f_exp=exp_freq)
    
    assert chi2 >= 0.0
    assert 0.0 <= p_val <= 1.0
