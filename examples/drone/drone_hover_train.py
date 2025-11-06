# selfplay_train.py
import argparse, os, pickle, shutil, copy, random
from importlib import metadata
import torch
from rsl_rl.runners import OnPolicyRunner
from collections import deque
import random, copy

import genesis as gs
from drone_hover_env import HoverEnv  # same env for both roles

def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
			"desired_kl": 0.02,
            "entropy_coef": 0.002,
            "gamma": 0.99,
			"lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
			"num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
			"value_loss_coef": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "activation": "tanh",
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "init_noise_std": 0.2,
        },
        "runner": {
            "experiment_name": exp_name,
            "checkpoint": -1,
			"load_run": -1,
            "log_interval": 1,
			"record_interval": -1,
            "resume": False,
			"resume_path": None,
            "max_iterations": max_iterations,
			"run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 100,
        "save_interval": 50,
        "empirical_normalization": True,
        "seed": 1,
    }

def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        "obs_stacks": 20,
        "base_init_pos": [0.0, 0.0, 1.5],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        "eval": False,
        # safety / terms
        "termination_if_tilt_greater_than": 80,
        "termination_if_close_to_ground": 0.1,
        "termination_if_yaw_rate_greater_than": 3.0,
        # success + shaping
        "at_target_threshold": 0.10,
        "max_rel_speed_mps": 0.2,
        "max_tilt_deg": 10.0,
        "max_angvel_radps": 1.5,
        "stable_time_s": 0.10,
        "approach_k": 2.0,
        "approach_v_cap": 1.0,
        "near_gate_factor": 2.0,
        "tgo_cap": 3.0,
        "angvel_excess_margin_radps": 0.5,
        # adversary path + geometry # TODO curriculum has gone missing?!?
        "adv_min_v": 0.0,
        "adv_max_v": 0.0,
        "adv_max_f": 1.0,
        "z_margin": 0.05,
        "adv_drone_half_thickness": 0.05,
    }
    obs_cfg = {
        "obs_scales": {
			"rel_pos": 1/3.0,
            "lin_vel": 1/3.0,
            "ang_vel": 1/3.14159
        }
    }
    reward_cfg = {  # ego rewards (env auto-dt-scales non-events and sums into episode_sums)
        "reward_scales": {
            "approach":         500.0,
            "tan_vel_align":    100.0,
            "smooth":           5.0,
            "ang_vel":          25.0,
            "crash":            20.0,
            "success":          0.5,
            "adv_success":      -1.0,
        },
        "adv_reward_scales": {
            "adv_approach":     500.0,
            "adv_escape":       500.0,
            "tan_vel_align":    50.0,
            "smooth":           5.0,
            "ang_vel":          25.0,
            "crash":            20.0,
            "success":          -0.5,
            "adv_success":      1.0,
        }
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range":[-1,1],
        "pos_y_range":[-1,1],
        "pos_z_range":[1,1]
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg

class OpponentPool:
    def __init__(self, capacity=20, p_newest=0.67, device=None):
        self.cap = capacity
        self.p_newest = p_newest
        self.device = device
        self.buf = deque()

    def _freeze_callable(self, runner):
        frozen = copy.deepcopy(runner.get_inference_policy(device=self.device))  # detaches weights
        def call(obs):
            with torch.no_grad():
                return frozen(obs)
        return call

    def save(self, runner):
        self.buf.appendleft(self._freeze_callable(runner))
        while len(self.buf) > self.cap:
            self.buf.pop()

    def sample(self):
        if not self.buf:
            return None
        if len(self.buf) == 1 or random.random() < self.p_newest:
            return self.buf[0]
        return random.choice(list(self.buf)[1:])

def check_lib():
    try:
        try:
            if metadata.version("rsl-rl"):
                raise ImportError
        except metadata.PackageNotFoundError:
            if metadata.version("rsl-rl-lib") != "2.2.4":
                raise ImportError
    except (metadata.PackageNotFoundError, ImportError) as e:
        raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e","--exp_name", default="drone-hovering-selfplay")
    p.add_argument("-B","--num_envs", type=int, default=8192)
    p.add_argument("--max_iterations", type=int, default=5001)
    p.add_argument("--alt_K", type=int, default=65)
    p.add_argument("-v","--vis", action="store_true", default=False)
    p.add_argument("--resume_ego", type=int, default=0)
    p.add_argument("--resume_adv", type=int, default=0)
    args = p.parse_args()

    check_lib()
    gs.init(seed=0, logging_level="warning", performance_mode=True)

    root = f"logs/{args.exp_name}"
    ego_dir, adv_dir = os.path.join(root,"ego"), os.path.join(root,"adv")
    if args.resume_ego==0 and args.resume_adv==0 and os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(ego_dir, exist_ok=True); os.makedirs(adv_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["visualize_target"] = bool(args.vis)

    # save cfgs for eval reuse
    train_cfg_proto = get_train_cfg(args.exp_name, args.max_iterations)
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_proto], open(os.path.join(root,"cfgs.pkl"),"wb"))

    # build envs (same class)
    envE = HoverEnv(args.num_envs, copy.deepcopy(env_cfg), copy.deepcopy(obs_cfg),
                    copy.deepcopy(reward_cfg), copy.deepcopy(command_cfg), show_viewer=args.vis)
    envA = HoverEnv(args.num_envs, copy.deepcopy(env_cfg), copy.deepcopy(obs_cfg),
                    copy.deepcopy(reward_cfg), copy.deepcopy(command_cfg), show_viewer=False)

    # role switch in env (see tiny patch below)
    envE.set_train_role("ego")
    envA.set_train_role("adv")

    # runners
    tcfgE = get_train_cfg(f"{args.exp_name}/ego", args.max_iterations)
    tcfgA = get_train_cfg(f"{args.exp_name}/adv", args.max_iterations)
    runnerE = OnPolicyRunner(envE, tcfgE, ego_dir, device=gs.device)
    runnerA = OnPolicyRunner(envA, tcfgA, adv_dir, device=gs.device)

    if args.resume_ego>0: runnerE.load(os.path.join(ego_dir, f"model_{args.resume_ego}.pt"))
    if args.resume_adv>0: runnerA.load(os.path.join(adv_dir, f"model_{args.resume_adv}.pt"))

    # bootstrap obs
    envE.reset(); envA.reset()

    # setup opponent pools
    adv_pool = OpponentPool(device=gs.device)
    ego_pool = OpponentPool(device=gs.device)
    # seed pools with the starting policies
    adv_pool.save(runnerA)
    ego_pool.save(runnerE)

    it = 0; K = max(1, args.alt_K)
    while it < args.max_iterations:
        # ego phase vs frozen adversary
        envE.set_opponent(adv_pool.sample())
        step = min(K, args.max_iterations - it); it += step
        runnerE.learn(num_learning_iterations=step, init_at_random_ep_len=True)
        ego_pool.save(runnerE)

        if it >= args.max_iterations: break

        # adversary phase vs frozen ego
        envA.set_opponent(ego_pool.sample())
        step = min(K, args.max_iterations - it); it += step
        runnerA.learn(num_learning_iterations=step, init_at_random_ep_len=True)
        adv_pool.save(runnerA)

    print("done")

if __name__ == "__main__":
    main()

#TODO use additional non-recent "anchor" policies in the pools