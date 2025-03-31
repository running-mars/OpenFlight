"""
Description:
    - Low-level control for Fixed-wing UAVs towards minimum-time flight

Authors:
    - Running-Mars
"""

import gym
import time

from stable_baselines3 import DDPG, PPO, SAC, TD3, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from network.p2p3d_net import P2p3dNetwork


def make_env():
    def _init():
        env = gym.make("airgym:jsbsim-uni-uav-sample-v2")
        env = Monitor(env)
        return env

    return _init


def main():
    num_envs = 16
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    policy_kwargs = dict(
        features_extractor_class=P2p3dNetwork,
        features_extractor_kwargs=dict(features_dim=64),
    )

    # Initialize RL algorithm type and parameters
    model = PPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        learning_rate=2.5e-4,
        verbose=1,
        batch_size=32,
        device="cuda",
        tensorboard_log="./tb_logs/",
    )

    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    # Create an evaluation callback with the same env
    callbacks = []
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        n_eval_episodes=5,
        best_model_save_path=".",
        log_path="./tb_logs/nav_p2p_run_" + time_stamp + "_eval",
        eval_freq=1000,
        verbose=1
    )
    callbacks.append(eval_callback)

    kwargs = {}
    kwargs["callback"] = callbacks

    model.learn(
        total_timesteps=2e6,
        tb_log_name="nav_p2p_run_" + time_stamp,
        **kwargs
    )

    model.save("nav_p2p_policy_" + time_stamp)


if __name__ == "__main__":
    main()
