from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor

# =========================================================
# Preprocesado de píxeles
# =========================================================
def preprocess(frame, size=84):
    """
    RGB uint8 (H, W, 3) -> RGB uint8 (3, size, size)
    """
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) # Normaliza imagen a 84x84
    frame = np.transpose(frame, (2, 0, 1)) # Cambia a formato (C,H,W) para PyTorch
    # frame = frame.astype(np.float32) / 255.0 # Normaliza a [0,1]
    return np.ascontiguousarray(frame, dtype=np.uint8) 


class PixelStackWrapper(gym.Wrapper):
    """
    Reemplaza la observación original del entorno por un stack de K frames RGB.

    Salida:
        obs.shape = (3*K, size, size)
        dtype = uint8
        rango = [0, 255]
    """
    def __init__(self, env: gym.Env, k: int = 4, size: int = 84):
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

    def _get_frame(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError(
                "env.render() devolvió None. Asegúrate de crear el entorno con render_mode='rgb_array'."
            )
        return preprocess(frame, size=self.size)

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

class ProgressWithSafetyShaping(gym.Wrapper):
    """
    Wrapper de reward shaping PARA Walker2d (o similares MuJoCo) pensado para:
      - Mantener la reward por defecto como base (forward + survive - ctrl_cost)
      - Incentivar avance SIN forzar postura "bonita"
      - Penalizar solo situaciones que suelen acabar en caída (altura baja / inclinación extrema)
      - (Opcional) suavizar cambios bruscos de acción para estabilizar marcha

    Recomendación: úsalo junto a tu IgnoreAngleTerminationWrapper si quieres episodios menos binarios.
    """

    def __init__(
        self,
        env,
        z_ref: float = 1.10,          # altura "mínima de seguridad" (no obliga a ir alto, solo evita colapso)
        angle_ref: float = 0.7,      # umbral de inclinación permisivo (radianes aprox)
        w_z: float = 0.7,            # peso penalización altura
        w_ang: float = 0.3,          # peso penalización inclinación
        w_smooth: float = 0.0,        # 0.0 = desactivado (si quieres activarlo: 0.005–0.02)
        alive_bonus: float = 0.2,    # bonus pequeño por seguir vivo
        speed_bonus: float = 0.3,    # bonus suave por velocidad hacia delante (tanh)
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
        
        # Base: reward por defecto del entorno
        shaped = float(reward)
        info["debug/base"] = shaped

        # Estado interno MuJoCo
        data = self.env.unwrapped.data
        z = float(data.qpos[1])       # altura torso
        ang = float(data.qpos[2])     # ángulo torso
        vx = float(data.qvel[0])      # velocidad x

        # 1) Avance: bonus suave y saturado (no clip duro)
        shaped += self.speed_bonus * float(np.tanh(vx))
        info["debug/speed_bonus"] = self.speed_bonus * float(np.tanh(vx))

        # 2) Seguridad: penaliza solo si está "demasiado bajo" (hinge)
        shaped -= self.w_z * max(0.0, self.z_ref - z)
        info["debug/height_pen"] = self.w_z * max(0.0, self.z_ref - z)

        # 3) Seguridad: penaliza solo si está "demasiado inclinado" (hinge)
        shaped -= self.w_ang * max(0.0, abs(ang) - self.angle_ref)
        info["debug/angle_pen"] = self.w_ang * max(0.0, abs(ang) - self.angle_ref)

        # 4) Suavidad (opcional): si action es vector continuo, penaliza jerk
        #    Si action es discreta (int), w_smooth debería estar a 0.0.
        if self.w_smooth > 0.0:
            a = np.array(action, dtype=np.float32)
            if self.prev_action is not None:
                shaped -= self.w_smooth * float(np.sum((a - self.prev_action) ** 2))
            self.prev_action = a

        # 5) Alive bonus pequeño (densifica sin dominar)
        if not (terminated or truncated):
            shaped += self.alive_bonus
            info["debug/alive_bonus"] = self.alive_bonus
        else: 
            info["debug/alive_bonus"] = 0.0

        return obs, shaped, terminated, truncated, info

def make_single_walker_env(
    env_id: str = "Walker2d-v5",
    seed: int = 0,
    image_size: int = 84,
    frame_stack: int = 4,
    reward_shaping: bool = False,
    terminate_when_unhealthy: bool = True,
    healthy_z_range: tuple[float, float] = (0.8, 2.0),
    record_video_folder: str | Path | None = None,
    video_prefix: str = "eval",
) -> gym.Env:
    """
    Crea un Walker2d-v5 con observación visual RGB apilada.
    """
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        terminate_when_unhealthy=terminate_when_unhealthy,
        healthy_z_range=healthy_z_range,
    )

    if record_video_folder is not None:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(record_video_folder),
            episode_trigger=lambda ep: True,
            name_prefix=video_prefix,
        )

    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if reward_shaping:
        env = ProgressWithSafetyShaping(env)

    env = PixelStackWrapper(env, k=frame_stack, size=image_size)

    env.reset(seed=seed)
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
) -> Callable[[], gym.Env]:
    def _init():
        return make_single_walker_env(
            env_id=env_id,
            seed=base_seed + rank,
            image_size=image_size,
            frame_stack=frame_stack,
            reward_shaping=reward_shaping,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
        )
    return _init


def make_vec_walker_env(
    env_id: str = "Walker2d-v5",
    seed: int = 0,
    n_envs: int = 1,
    image_size: int = 84,
    frame_stack: int = 4,
    reward_shaping: bool = True,
    terminate_when_unhealthy: bool = True,
    healthy_z_range: tuple[float, float] = (0.8, 2.0),
    monitor_path: str | Path | None = None,
) -> VecEnv:
    """
    Vector env listo para SB3.
    """
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
        )
        for i in range(n_envs)
    ]

    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=str(monitor_path) if monitor_path else None)
    return vec_env