# drone_hover_eval.py
import argparse
import os
import pickle
import copy
from importlib import metadata
from tqdm import trange

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs

from drone_hover_env import HoverEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("--ckpt", type=int, default=300)
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(seed=1, performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    log_dir_ego = os.path.join(log_dir, "ego")
    log_dir_adv = os.path.join(log_dir, "adv")
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))


    # Eval-only safety and visuals.
    env_cfg["episode_length_s"] = 60.0
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60
    env_cfg["eval"] = False
#    env_cfg["z_margin"] = 0.00

    # Disable reward logging during eval.
    reward_cfg["reward_scales"] = {}

    env = HoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Important: deep-copy train_cfg BEFORE constructing any runner.
    train_cfg_ego = copy.deepcopy(train_cfg)
    train_cfg_adv = copy.deepcopy(train_cfg)

    # Ego policy.
    runner_ego = OnPolicyRunner(env, train_cfg_ego, log_dir_ego, device=gs.device)
    runner_ego.load(os.path.join(log_dir_ego, f"model_{args.ckpt}.pt"))
    policy_ego = runner_ego.get_inference_policy(device=gs.device)

    # Adversary policy (same checkpoint, independent state).
    runner_adv = OnPolicyRunner(env, train_cfg_adv, log_dir_adv, device=gs.device)
    runner_adv.load(os.path.join(log_dir_adv, f"model_{args.ckpt}.pt"))
    policy_adv = runner_adv.get_inference_policy(device=gs.device)

    # Plug adversary policy into the environment.
    env.set_opponent(policy_adv)

    obs, _ = env.reset()
    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])

    with torch.no_grad():
        if args.record:
            env.cam.start_recording()
            for _ in trange(max_sim_step):
                actions = policy_ego(obs)
                obs, rews, dones, infos = env.step(actions)
                env.cam.render()
            env.cam.stop_recording(fps=env_cfg["max_visualize_FPS"])
        else:
            for _ in trange(max_sim_step):
                actions = policy_ego(obs)
                obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()
