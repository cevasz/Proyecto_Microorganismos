import argparse
from modules.rl_agent.agent import train_ppo_agent

def main():
    """
    Punto de entrada de CLI para entrenar el agente de quimiotaxis con PPO.
    Soporta argumentos para configurar el archivo yaml y la cantidad de pasos de entrenamiento.
    """
    parser = argparse.ArgumentParser(description="Entrenador de Quimiotaxis RL (PPO) - NeuroColony-EC")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Ruta al archivo de configuración de hiperparámetros default.yaml"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2000000,
        help="Número total de pasos globales de simulación para el entrenamiento (CFL/FDM steps)"
    )
    
    args = parser.parse_args()
    
    # Iniciar el proceso de entrenamiento
    train_ppo_agent(
        config_path=args.config,
        total_timesteps=args.timesteps
    )

if __name__ == "__main__":
    main()
