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
        "features_extractor_kwargs": {"features_dim": 256},
    }

    if algo == "ppo":
        return PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
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
            learning_rate=3e-4,
            buffer_size=300_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            ent_coef="auto",
            seed=seed,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device,
            policy_kwargs={
                **common_policy_kwargs,
                "net_arch": [256, 256],
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