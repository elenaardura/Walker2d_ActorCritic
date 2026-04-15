import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from pathlib import Path
from typing import Callable

from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor

def preprocess(frame, size=84):
    """
    RGB uint8 (H, W, 3) -> RGB uint8 (3, size, size)
    """
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
    return np.ascontiguousarray(frame, dtype=np.uint8)

 
def make_discrete_action_set_legprototype(action_dim: int):
    Z = np.zeros(action_dim, dtype=np.float32)
    
    # Magnitudes (suaves para evitar inestabilidad al inicio)
    a = 1.0
    b = 0.5

    actions = []

    def add(vector):
        # Aseguramos que las acciones estén en el rango [-1, 1]   
        actions.append(np.clip(vector, -1.0, 1.0)) 
    
    # Acción de idle (ninguna acción)
    add(Z)
    
    # dobla tobillo pierna 1 (empuje hacia adelante)
    add(np.array([0, 0, 0, 0, 0, +a], dtype=np.float32))
    add(np.array([0, 0, 0, 0, 0, -a], dtype=np.float32))
    
    # dobla tobillo pierna 2 (empuje hacia adelante)
    add(np.array([0, 0, +a, 0, 0, 0], dtype=np.float32))
    add(np.array([0, 0, a, 0, 0, 0], dtype=np.float32))
    
    # empuja pierna 1 (extiende rodilla + empuja tobillo + hip suave)
    add(np.array([0, -a, +a, 0, 0, 0], dtype=np.float32))
    add(np.array([0, +a, -a, 0, 0, 0], dtype=np.float32))
    
    # empuja pierna 2
    add(np.array([0, 0, 0, 0, -a, +a], dtype=np.float32))
    add(np.array([0, 0, 0, 0, +a, -a], dtype=np.float32))
    
    # recupera pierna 1 (flexiona rodilla)
    add(np.array([0, +a, 0, 0, 0, 0], dtype=np.float32))
    add(np.array([0, -a, 0, 0, 0, 0], dtype=np.float32))

    # recupera pierna 2 (flexiona rodilla)
    add(np.array([0, 0, 0, 0, +a, 0], dtype=np.float32))
    add(np.array([0, 0, 0, 0, -a, 0], dtype=np.float32))

    # estabiliza (hips hacia atrás suave para no “tirarse”)
    # Hips fuertes (1.0)
    add(np.array([0, 0, 0, +a, 0, 0], dtype=np.float32))
    add(np.array([0, 0, 0, -a, 0, 0], dtype=np.float32))
    add(np.array([+a, 0, 0, 0, 0, 0], dtype=np.float32))
    add(np.array([-a, 0, 0, 0, 0, 0], dtype=np.float32))

    # Hips suaves (0.5)
    add(np.array([0, 0, 0, +b, 0, 0], dtype=np.float32))
    add(np.array([0, 0, 0, -b, 0, 0], dtype=np.float32))
    add(np.array([+b, 0, 0, 0, 0, 0], dtype=np.float32))
    add(np.array([-b, 0, 0, 0, 0, 0], dtype=np.float32))
    
    return np.stack(actions, axis=0)
    

class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Convierte acciones discretas (int) en acciones continuas (Box)
    """
    def __init__(self, env):
        super().__init__(env)
        # Comprobamos que el espacio de acciones original es continuo
        assert isinstance(env.action_space, gym.spaces.Box) 

        self._actions = make_discrete_action_set_legprototype(env.action_space.shape[0])
        self.action_space = gym.spaces.Discrete(self._actions.shape[0]) 
    def action(self, act_idx):
        # Convertimos el índice de acción discreta en la acción continua correspondiente
        return self._actions[int(act_idx)] 


class PixelStackWrapper(gym.Wrapper):
    """
    Convierte la observación en un stack de K frames RGB.
    Shape final: (3*K, size, size)
    """
    def __init__(self, env, k=4, size=84):
        super().__init__(env)
        self.k = int(k)
        self.size = int(size)
        self.frames = deque(maxlen=self.k)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3 * self.k, self.size, self.size),
            dtype=np.uint8,
        )

    def _get_frame(self):
        frame = self.env.render()
        if frame is None:
            raise RuntimeError(
                "env.render() devolvió None. Crea el entorno con render_mode='rgb_array'."
            )
        return preprocess(frame, self.size)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        frame = self._get_frame()

        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(frame)

        obs = np.concatenate(list(self.frames), axis=0)
        return obs, info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        frame = self._get_frame()
        self.frames.append(frame)

        obs = np.concatenate(list(self.frames), axis=0)
        return obs, reward, terminated, truncated, info


# Reward shaping antigua, con bonus por avance y supervivencia, y penalizaciones por altura y ángulo
class BonusBasedReward(gym.Wrapper):
    """
    Reward shaping suave para Walker2d-v5.
    """
    def __init__(
        self,
        env,
        z_ref: float = 1.10,
        angle_ref: float = 0.7,
        w_z: float = 0.7,
        w_ang: float = 0.3,
        w_smooth: float = 0.0,
        alive_bonus: float = 0.2,
        speed_bonus: float = 0.3,
    ):
        super().__init__(env)
        self.z_ref = float(z_ref)
        self.angle_ref = float(angle_ref)
        self.w_z = float(w_z)
        self.w_ang = float(w_ang)
        self.w_smooth = float(w_smooth)
        self.alive_bonus = float(alive_bonus)
        self.speed_bonus = float(speed_bonus)
        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped = float(reward)
        info["debug/base"] = shaped

        data = self.env.unwrapped.data
        z = float(data.qpos[1])
        ang = float(data.qpos[2])
        vx = float(data.qvel[0])

        shaped += self.speed_bonus * float(np.tanh(vx))
        info["debug/speed_bonus"] = self.speed_bonus * float(np.tanh(vx))

        z_pen = self.w_z * max(0.0, self.z_ref - z)
        ang_pen = self.w_ang * max(0.0, abs(ang) - self.angle_ref)

        shaped -= z_pen
        shaped -= ang_pen

        info["debug/height_pen"] = z_pen
        info["debug/angle_pen"] = ang_pen

        if self.w_smooth > 0.0:
            a = np.array(action, dtype=np.float32)
            if self.prev_action is not None:
                smooth_pen = self.w_smooth * float(np.sum((a - self.prev_action) ** 2))
                shaped -= smooth_pen
                info["debug/smooth_pen"] = smooth_pen
            else:
                info["debug/smooth_pen"] = 0.0
            self.prev_action = a
        else:
            info["debug/smooth_pen"] = 0.0

        if not (terminated or truncated):
            shaped += self.alive_bonus
            info["debug/alive_bonus"] = self.alive_bonus
        else:
            info["debug/alive_bonus"] = 0.0

        return obs, shaped, terminated, truncated, info

# Reward shaping nueva, con ponderaciones a los componentes de la reward base y penalizaciones por altura y ángulo
class ComponentBasedReward(gym.Wrapper):
    """
    Reward shaping suave para Walker2d-v5.
    """
    def __init__(
        self,
        env,
        z_ref: float = 1.10,
        angle_ref: float = 0.7,
        w_z: float = 0.7,
        w_ang: float = 0.3,
        w_smooth: float = 0.0,
        alive_weight: float = 1.25,
        speed_weight: float = 0.75,
    ):
        super().__init__(env)
        self.z_ref = float(z_ref)
        self.angle_ref = float(angle_ref)
        self.w_z = float(w_z)
        self.w_ang = float(w_ang)
        self.w_smooth = float(w_smooth)
        self.alive_weight = float(alive_weight)
        self.speed_weight = float(speed_weight)
        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        forward_reward = float(info.get("reward_forward", 0.0)) * self.speed_weight

        survive_reward = float(info.get("reward_survive", 0.0)) * self.alive_weight
        ctrl_reward = float(info.get("reward_ctrl", 0.0))

        shaped = survive_reward + forward_reward + ctrl_reward
        
        data = self.env.unwrapped.data
        z = float(data.qpos[1])
        ang = float(data.qpos[2])

        z_pen = self.w_z * max(0.0, self.z_ref - z)
        ang_pen = self.w_ang * max(0.0, abs(ang) - self.angle_ref)

        shaped -= z_pen
        shaped -= ang_pen
        
        info["debug/reward_forward"] = forward_reward
        info["debug/reward_survive"] = survive_reward
        info["debug/reward_ctrl"] = ctrl_reward
        info["debug/height_pen"] = z_pen
        info["debug/angle_pen"] = ang_pen

        if self.w_smooth > 0.0:
            a = np.array(action, dtype=np.float32)
            if self.prev_action is not None:
                smooth_pen = self.w_smooth * float(np.sum((a - self.prev_action) ** 2))
                shaped -= smooth_pen
                info["debug/smooth_pen"] = smooth_pen
            else:
                info["debug/smooth_pen"] = 0.0
            self.prev_action = a
        else:
            info["debug/smooth_pen"] = 0.0

        return obs, shaped, terminated, truncated, info

def make_single_walker_env(
    env_id="Walker2d-v5",
    seed=0,
    image_size=84,
    frame_stack=4,
    reward_shaping=False,
    terminate_when_unhealthy=True,
    healthy_z_range=(0.8, 2.0),
    record_video_folder=None,
    video_prefix="eval",
    use_discrete_actions=False,
):
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        terminate_when_unhealthy=terminate_when_unhealthy,
        healthy_z_range=healthy_z_range,
        # Aumentamos el límite de pasos por episodio para permitir episodios más largos
        # max_episode_steps=3000,  
    )

    if record_video_folder is not None:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(record_video_folder),
            episode_trigger=lambda ep: True,
            name_prefix=video_prefix,
        )
        print(f"[Video] Recording enabled. Videos will be saved to: {record_video_folder}")

    env = gym.wrappers.RecordEpisodeStatistics(env)

    if reward_shaping:
        # Si se desea reward antigua:
        # env = BonusBasedReward(env)
        
        # Si se desea reward nueva:
        env = ComponentBasedReward(env)

    env = PixelStackWrapper(env, k=frame_stack, size=image_size)

    if use_discrete_actions:
        env = DiscreteActionWrapper(env)

    env.action_space.seed(seed)
    return env


def _make_env_fn(
    rank: int,
    env_id: str,
    base_seed: int,
    image_size: int,
    frame_stack: int,
    reward_shaping: bool,
    terminate_when_unhealthy: bool,
    healthy_z_range: tuple[float, float],
    use_discrete_actions: bool = False,
) -> Callable[[], gym.Env]:                        
    
    def _thunk():
        return make_single_walker_env(
            env_id=env_id,
            seed=base_seed + rank,
            image_size=image_size,
            frame_stack=frame_stack,
            reward_shaping=reward_shaping,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            use_discrete_actions=use_discrete_actions,
        )
    return _thunk


def make_vec_walker_env(
    env_id="Walker2d-v5", 
    seed=0,
    n_envs=1,
    image_size=84,
    frame_stack=4,
    reward_shaping=False,
    terminate_when_unhealthy=True, 
    healthy_z_range=(0.8, 2.0),
    monitor_path=None,
    use_discrete_actions=False,
) -> VecEnv:
    env_fns = [
        _make_env_fn(
            rank=i,
            env_id=env_id,
            base_seed=seed,
            image_size=image_size,
            frame_stack=frame_stack,
            reward_shaping=reward_shaping,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            use_discrete_actions=use_discrete_actions,
        )
        for i in range(n_envs)
    ]

    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=str(monitor_path) if monitor_path else None)
    return vec_env