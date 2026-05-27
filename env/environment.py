import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

class BacteriumEnv(gym.Env):
    """
    MVP Environment for Bacterium Chemotaxis (RL Phase 1).
    Observation: [pos_x, pos_y, grad_x, grad_y]
    Action: continuous [run_duration, tumble_probability]
    """
    def __init__(self, config_path="configs/default.yaml"):
        super(BacteriumEnv, self).__init__()
        
        # Action space: [run_duration, tumble_probability]
        # SB3 uses symmetric spaces around 0 for Box, so we use [-1, 1] and scale in step()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space: pos_x, pos_y, grad_x, grad_y
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.step_count = 0
        self.max_steps = 1000
        
        # Metrics for ChemotaxisLoggingCallback
        self.accumulated_reward = 0.0
        self.run_durations = []
        self.is_running = []
        self.start_pos = np.zeros(2)
        self.pos = np.zeros(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.accumulated_reward = 0.0
        self.run_durations = []
        self.is_running = []
        
        # Start somewhere in the grid
        self.pos = np.random.uniform(0, 100, size=2)
        self.start_pos = self.pos.copy()
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # Dummy gradient pointing towards (100, 100)
        grad = np.array([100.0 - self.pos[0], 100.0 - self.pos[1]])
        norm = np.linalg.norm(grad)
        if norm > 0:
            grad = grad / norm
        return np.array([self.pos[0], self.pos[1], grad[0], grad[1]], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        
        # Scale actions
        run_duration_scaled = (action[0] + 1.0) / 2.0 * 2.9 + 0.1 # Map [-1, 1] to [0.1, 3.0]
        tumble_prob = (action[1] + 1.0) / 2.0                     # Map [-1, 1] to [0.0, 1.0]
        
        self.run_durations.append(run_duration_scaled)
        
        # Basic run and tumble mechanics
        if np.random.rand() > tumble_prob:
            # Run: Move along gradient
            self.pos += self._get_obs()[2:4] * 25.0 * run_duration_scaled # 25 um/s
            self.is_running.append(1.0)
        else:
            # Tumble: Don't move significantly, just change direction (abstracted here)
            self.is_running.append(0.0)
            
        # Reward based on getting closer to (100, 100)
        dist_to_target = np.linalg.norm(np.array([100.0, 100.0]) - self.pos)
        reward = -dist_to_target * 0.01
        
        self.accumulated_reward += reward
        
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        # Info dictionary with required metrics for the callback
        info = {
            "accumulated_reward": float(self.accumulated_reward),
            "mean_run_duration": float(np.mean(self.run_durations)),
            "run_fraction": float(np.mean(self.is_running)),
            "net_distance": float(np.linalg.norm(self.pos - self.start_pos))
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info
