from pathlib import Path
from stable_baselines3 import PPO, SAC


ALGOS = {
    "ppo": PPO,
    "sac": SAC,
}


def build_model(
    algo: str,
    env,
    seed: int = 0,
    tensorboard_log: str | None = None,
    device: str = "auto",
):
    algo = algo.lower()

    common_policy_kwargs = {
        # "features_extractor_kwargs": {"features_dim": 256},
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": [512, 512],
    }

    if algo == "ppo":
        return PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=3e-4,# antes 1e-4
            n_steps=4096, # antes 1024
            batch_size=256, # antes 64, luego 128
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.015, # antes 0.0, luego 0.01 (cambiado a las 15:51)
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device,
            policy_kwargs=common_policy_kwargs,
        )

    if algo == "sac":
        return SAC(
            policy="CnnPolicy",
            env=env,
            learning_rate=1e-4, # antes 3e-4
            buffer_size=300_000, 
            learning_starts=100_000, #antes 25k (cambio a 100k el 26 a las 8:12)
            batch_size=128, # antes 256 (cambiado a las 8:14 del 26)
            tau=0.01, # antes 0.005
            gamma=0.99,
            train_freq=(4, "step"), # antes (1, "step") 
            gradient_steps=4, # antes 1
            ent_coef=0.01, # antes auto, luego 0.01 (cambiado a las 8:14 el 26)
            target_entropy="auto",
            seed=seed,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device,
            policy_kwargs={
                **common_policy_kwargs,
                # "net_arch": [256, 256],
                "net_arch": [512, 512],
                
            },
        )

    raise ValueError(f"Algoritmo no soportado: {algo}. Usa 'ppo' o 'sac'.")


def load_model(
    algo: str,
    model_path: str | Path,
    env=None,
    device: str = "auto",
):
    algo = algo.lower()

    if algo not in ALGOS:
        raise ValueError(f"Algoritmo no soportado: {algo}. Usa 'ppo' o 'sac'.")

    model_cls = ALGOS[algo]
    return model_cls.load(str(model_path), env=env, device=device)