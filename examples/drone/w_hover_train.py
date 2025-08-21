import argparse
import os
import pickle
import shutil
from importlib import metadata

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

from w_hover_env import HoverEnv


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.005,
            "entropy_coef": 0.001,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "init_noise_std": 0.1,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 100,
        "save_interval": 100,
        "empirical_normalization": True,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        "obs_stacks" : 15,
        # termination
        "termination_if_roll_greater_than": 80,  # degree
        "termination_if_pitch_greater_than": 80,
        "termination_if_close_to_ground": 0.1,
        "termination_if_x_greater_than": 3.0,
        "termination_if_y_greater_than": 3.0,
        "termination_if_z_greater_than": 3.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 1.5],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "at_target_threshold": 0.1,
        "resampling_time_s": 3.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        # adversary / platform+target movement
        "adv_max_v": 1.0,
        "adv_max_f": 1.0,
        "adv_box_h": 0.50,         # adversary box height (m)
        # curriculum
        "adv_difficulty_min": 0.0,
        "adv_difficulty_max": 1.0,
        "adv_difficulty_delta_success": 0.01,
        "adv_difficulty_delta_fail": 0.1,
        # shaping/logic params
        "approach_k": 2.0,
        "approach_v_cap": 1.0,
        "near_gate_factor": 2.0,             # ties gate decay to target threshold
        "z_margin": 0.05,                    # pad clearance [m]
        "tgo_cap": 3.0,                      # clamp for t_go [s]
        # success criteria
        "max_rel_speed_mps": 0.2,
        "max_tilt_deg":      10.0,
        "stable_time_s":     0.10,
        "retarget_frames": 10,         # 100 ms at dt=0.01
    }
    obs_cfg = {
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "approach":         500.0,
            "tan_vel_align":    1.0,
            "below_pad":        100.0,
            "smooth":           0.5,
            "ang_vel":          0.25,
            "crash":            100.0,
            "success":          5.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [1.0, 1.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=16384)
    parser.add_argument("--max_iterations", type=int, default=5001)
    parser.add_argument("-R", "--resume_ckpt", type=int, default=0)
    args = parser.parse_args()

    gs.init(seed=0, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if args.resume_ckpt == 0 and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = HoverEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if args.resume_ckpt > 0:
        resume_dir = f'logs/{args.exp_name}'
        resume_path = os.path.join(resume_dir, f'model_{args.resume_ckpt}.pt')
        print('==> resume training from', resume_path)
        runner.load(resume_path)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    os.system(f"PYOPENGL_PLATFORM=glx vk_pro python ./examples/drone/w_hover_eval.py -e {args.exp_name} --ckpt {args.resume_ckpt + args.max_iterations - 1} --record")

if __name__ == "__main__":
    main()

"""
# training
python examples/drone/hover_train.py
"""
