import os
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from env.environment import BacteriumEnv

class ChemotaxisLoggingCallback(BaseCallback):
    """
    Callback personalizado para Stable-Baselines3 que calcula y registra métricas
    biológicas e índices de quimiotaxis del agente bacteria cada 10K pasos globales.
    """
    def __init__(self, check_freq=10000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.last_log_step = 0
        
        # Historial de métricas de episodios completados
        self.episode_rewards = []
        self.run_durations = []
        self.run_fractions = []
        self.net_distances = []

    def _on_step(self) -> bool:
        # Extraer información de fin de episodio de los entornos vectorizados
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        
        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                if done:
                    info = infos[i]
                    # Recuperar métricas del último paso del episodio antes del auto-reset
                    self.episode_rewards.append(info.get("accumulated_reward", 0.0))
                    self.run_durations.append(info.get("mean_run_duration", 0.0))
                    self.run_fractions.append(info.get("run_fraction", 0.0))
                    self.net_distances.append(info.get("net_distance", 0.0))

        # Registrar métricas al alcanzar el intervalo establecido
        if (self.num_timesteps - self.last_log_step) >= self.check_freq:
            self.last_log_step = self.num_timesteps
            
            if len(self.episode_rewards) > 0:
                mean_reward = float(np.mean(self.episode_rewards))
                mean_run = float(np.mean(self.run_durations))
                mean_frac = float(np.mean(self.run_fractions))
                mean_dist = float(np.mean(self.net_distances))
                
                # Baseline de comparación: Random Walker con distancia neta típica de ~80 µm
                baseline_dist = 80.0
                ratio_chemotactic = mean_dist / baseline_dist
                
                print(f"\n=================================================================")
                print(f"--- MTRICAS DE QUIMIOTAXIS A LOS {self.num_timesteps} PASOS ---")
                print(f"Recompensa acumulada media: {mean_reward:.4f}")
                print(f"Duración media de runs (tau_run): {mean_run:.2f} s (Meta Berg & Brown: ~1.0s)")
                print(f"Fracción de tiempo Run vs Tumble: {mean_frac:.2%} Run / {1.0 - mean_frac:.2%} Tumble")
                print(f"Distancia neta recorrida: {mean_dist:.2f} µm (Ratio vs Random Walker: {ratio_chemotactic:.2f}x)")
                print(f"=================================================================\n")
                
                # Guardar en el logger interno de Stable-Baselines3 (Tensorboard/W&B)
                self.logger.record("chemotaxis/mean_reward", mean_reward)
                self.logger.record("chemotaxis/mean_run_duration", mean_run)
                self.logger.record("chemotaxis/run_fraction", mean_frac)
                self.logger.record("chemotaxis/net_distance", mean_dist)
                self.logger.record("chemotaxis/ratio_vs_random_walker", ratio_chemotactic)
                
                # Reiniciar acumuladores para el siguiente bloque de entrenamiento
                self.episode_rewards.clear()
                self.run_durations.clear()
                self.run_fractions.clear()
                self.net_distances.clear()
                
        return True


def train_ppo_agent(config_path="configs/default.yaml", total_timesteps=2000000):
    """
    Configura y entrena un agente PPO vectorizado en paralelo para el entorno de bacteria.
    
    Args:
        config_path (str): Ruta al archivo de configuración default.yaml.
        total_timesteps (int): Pasos de simulación globales de entrenamiento.
    """
    print(f"Cargando configuración desde: {config_path}")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
    # Extraer parámetros de entrenamiento
    rl_cfg = config.get("rl", {})
    lr = float(rl_cfg.get("learning_rate", 3.0e-4))
    gamma = float(rl_cfg.get("gamma", 0.99))
    
    # Crear directorios de guardado
    os.makedirs("models/checkpoints/", exist_ok=True)
    os.makedirs("logs/ppo_chemotaxis_tensorboard/", exist_ok=True)
    
    # 1. Crear entorno vectorizado en paralelo (N_ENVS=8)
    n_envs = 8
    print(f"Inicializando {n_envs} entornos de simulación en paralelo...")
    env = make_vec_env(
        BacteriumEnv,
        n_envs=n_envs,
        env_kwargs=dict(config_path=config_path)
    )
    
    # 2. Configurar arquitectura neuronal MLP de Actor y Crítico [256, 256]
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    # 3. Inicializar el agente PPO
    print("Inicializando agente PPO con arquitectura Actor-Crítico [256, 256]...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr,
        gamma=gamma,
        verbose=1,
        tensorboard_log="./logs/ppo_chemotaxis_tensorboard/"
    )
    
    # 4. Configurar callbacks
    # Callback personalizado de quimiotaxis cada 10K pasos
    chemotaxis_callback = ChemotaxisLoggingCallback(check_freq=10000)
    # Callback de guardado de checkpoints cada 100K pasos
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="models/checkpoints/",
        name_prefix="ppo_chemotaxis"
    )
    
    callbacks = CallbackList([chemotaxis_callback, checkpoint_callback])
    
    # 5. Iniciar entrenamiento
    print(f"Comenzando entrenamiento por {total_timesteps} pasos...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    
    # 6. Guardar política final
    final_path = "models/ppo_chemotaxis_final.zip"
    print(f"Entrenamiento completado. Guardando política final en: {final_path}")
    model.save(final_path)
    
    # Cerrar entorno
    env.close()
    print("Proceso de entrenamiento PPO completado exitosamente.")
