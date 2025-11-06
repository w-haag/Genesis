import torch
import math
import copy
import genesis as gs
from genesis.utils.geom import transform_by_quat, inv_quat
from torch.nn import functional


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device, dtype=gs.tc_float) + lower


class ObsStacker:
    def __init__(self, num_envs, obs_dim, K, device):
        self.N, self.D, self.K, self.device = num_envs, obs_dim, K, device
        self.rows = torch.arange(self.N, dtype=torch.long, device=device)
        self.offsets = torch.arange(self.K, dtype=torch.long, device=self.device)
        self.buf = torch.zeros(num_envs, K, obs_dim, dtype=gs.tc_float, device=device)
        self.ptr = torch.zeros(num_envs, dtype=torch.long, device=device)

    @torch.no_grad()
    def push(self, obs_t, done):
        inc = (~done).to(self.ptr.dtype)
        self.ptr.add_(inc).remainder_(self.K)
        self.buf[self.rows, self.ptr] = obs_t
        self.ptr[done] = 0
        obs_t_b = obs_t[done].unsqueeze(1).expand(-1, self.K, -1)
        self.buf[done] = obs_t_b
        self.buf[done, :, -1] = 1.0
        self.buf[done, 0, -1] = 0.0

    @torch.no_grad()
    def stacked(self):
        idx = (self.ptr[:, None] - self.offsets[None, :]) % self.K
        out = self.buf.gather(1, idx[..., None].expand(-1, -1, self.D))
        return out.reshape(self.N, self.K * self.D)

class MultiRateStacker:
    def __init__(self, num_envs, obs_dim, K, device, group=3, max_horizon=None):
        self.N, self.D, self.K, self.device = num_envs, obs_dim, K, device
        self.rows = torch.arange(self.N, dtype=torch.long, device=device)
        self.offsets = torch.tensor(self._build_offsets(K, group, max_horizon), dtype=torch.long, device=device)
        self.M = int(self.offsets[-1].item()) + 1
        self.buf = torch.zeros(num_envs, self.M, obs_dim, dtype=gs.tc_float, device=device)
        self.ptr = torch.zeros(num_envs, dtype=torch.long, device=device)
            
    @torch.no_grad()
    def push(self, obs_t, done):
        inc = (~done).to(self.ptr.dtype)
        self.ptr.add_(inc).remainder_(self.M)
        self.buf[self.rows, self.ptr] = obs_t

        self.ptr[done] = 0
        obs_t_broadcast = obs_t[done].unsqueeze(1).expand(-1, self.M, -1)
        self.buf[done] = obs_t_broadcast
        self.buf[done, :, -1] = 1.0
        self.buf[done, 0, -1] = 0.0

    @torch.no_grad()
    def stacked(self):
        idx = (self.ptr[:, None] - self.offsets[None, :]) % self.M
        out = self.buf.gather(1, idx[..., None].expand(-1, -1, self.D))
        return out.reshape(self.N, self.K * self.D)

    @staticmethod
    def _build_offsets(K, group, max_horizon):
        offs, stride, used = [0], 1, 0
        cap = None if max_horizon is None else int(max_horizon)
        for _ in range(1, K):
            nxt = offs[-1] + stride
            if cap is not None and nxt > cap:
                nxt = cap
            offs.append(nxt)
            used += 1
            if used == group:
                stride <<= 1
                used = 0
        return offs

class HoverEnv:
    """
    Two-drone chase:
      • Adversary is a real drone, not kinematic.
      • Adversary policy tracks the original target point `commands` (its *top* goes there).
      • Ego policy tracks the adversary’s *top* point (collision allowed).
      • Ego resets on its own terminations; adversary reaching target never triggers reset.

    API is unchanged for training/inference of the ego: step(actions) -> obs, rew, done, info.
    Provide the adversary policy via set_adversary_policy().
    """
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", False)
        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])
        self.adv_reward_scales = copy.deepcopy(reward_cfg["adv_reward_scales"])

        # sim + viewer
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(4.0, 0.0, 4.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            show_FPS=False,
        )
        self.scene.add_entity(gs.morphs.Plane())

        # visual aids
        if self.env_cfg.get("visualize_target", False):
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.05, fixed=False, collision=False),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5))),
            )
            self.target_threshold_highlight = self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.051, fixed=False, collision=False),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.5, 0.75, 0.5))),
            )
            self.target_reached_highlight = self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.052, fixed=False, collision=False),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0))),
            )
            # self.adv_target_marker = None
            self.adv_target_marker = self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.04, fixed=False, collision=False),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.2, 0.4, 1.0))),
            )
            self.highlight_hide = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float)
        else:
            self.target = None
            self.target_threshold_highlight = None
            self.target_reached_highlight = None
            self.adv_target_marker = None

        # drones
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))
        # adversary is always a drone here
        self.adversary = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # optional camera
        if self.env_cfg.get("visualize_camera", False):
            self.cam = self.scene.add_camera(
                # res=(960, 540),
                res=(1920, 1080),
                pos=(4.0, 0.0, 4.0),
                lookat=(0, 0, 1.0),
                fov=30,
                GUI=True
            )

        # constants
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        # build
        self.scene.build(n_envs=num_envs)

        # reward setup (dt-scale non-event terms)
        EVENT = {"success", "adv_success", "crash"}
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in copy.deepcopy(self.reward_cfg["reward_scales"]).keys():
            if name not in EVENT:
                self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.adv_reward_functions, self.adv_episode_sums = dict(), dict()
        for name in copy.deepcopy(self.reward_cfg["adv_reward_scales"]).keys():
            if name not in EVENT:
                self.adv_reward_scales[name] *= self.dt
            self.adv_reward_functions[name] = getattr(self, "_reward_" + name)
            self.adv_episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # buffers: ego
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        # commands: adversary's moving setpoint; commands_adv: ego's target (adversary top)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_adv = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.last_commands = torch.zeros_like(self.commands)
        self.last_commands_adv = torch.zeros_like(self.commands_adv)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.rel_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_rel_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.rel_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        # adversary state
        self.adv_actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.adv_last_actions = torch.zeros_like(self.adv_actions)
        self.adv_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.adv_last_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.adv_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.adv_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.adv_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        # geometry
        half_thickness = self.env_cfg.get("adv_drone_half_thickness", 0.05)
        self.z_margin = float(self.env_cfg["z_margin"])
        self.adv_base_offset = torch.tensor([0.0, 0.0, -(self.z_margin + half_thickness)], device=gs.device, dtype=gs.tc_float)

        self.world_z = torch.tensor([0.0, 0.0, 1.0], device=gs.device, dtype=gs.tc_float).expand(self.num_envs, 3)
        self.max_tilt_cos = math.cos(math.radians(self.env_cfg["max_tilt_deg"]))
        self.term_tilt_cos = math.cos(math.radians(self.env_cfg["termination_if_tilt_greater_than"]))
        self.base_tilt_cos = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)

        # target kinematics for ego (adversary top)
        self.tgt_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.tgt_vel_est = torch.zeros_like(self.tgt_vel)
        self.tgt_acc_est = torch.zeros_like(self.tgt_vel)

        # approach and shaping geometry
        self.app_geom = (
            torch.zeros((self.num_envs), device=gs.device),
            torch.zeros((self.num_envs, 3), device=gs.device),
            torch.zeros((self.num_envs), device=gs.device),
            torch.zeros((self.num_envs), device=gs.device),
        )
        self.prev_dist = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.prev_vel_close = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.prev_v_tan = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.prev_ang_norm = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.prev_yaw_abs = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.adv_app_geom = (
            torch.zeros((self.num_envs), device=gs.device),
            torch.zeros((self.num_envs, 3), device=gs.device),
            torch.zeros((self.num_envs), device=gs.device),
            torch.zeros((self.num_envs), device=gs.device),
        )
        self.adv_prev_dist = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.adv_prev_vel_close = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.adv_prev_v_tan = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.adv_prev_ang_norm = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.adv_prev_yaw_abs = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # obs stackers
        D = self.build_obs_template_dim()
        K = self.env_cfg["obs_stacks"]
        self.stacker = ObsStacker(self.num_envs, D, K, gs.device)
        self.adv_stacker = ObsStacker(self.num_envs, D, K, gs.device)
        self.num_obs = D * K
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.adv_obs_buf = torch.zeros_like(self.obs_buf)

        # collisions and terms
        self.adv_collision = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        self.crash_condition = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)

        # metrics
        self.m_d_sum = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_step = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_inside_sum = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_near_cnt = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_vapp_err_sum = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_vtan_sum = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_angvel_sum = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_vtgt_sum = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)

        self.stable_cnt = torch.zeros(self.num_envs, dtype=torch.int32, device=gs.device)
        self.n_stable = int(round(self.env_cfg["stable_time_s"] / self.dt))
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)

        self.extras = {"observations": {}}

        # moving path for adversary setpoint
        self.commands_anchor = torch.zeros_like(self.commands)
        self.path_a = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.path_f = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.path_phi = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        # ARL setup
        self.train_role = "ego"           # "ego" | "adv"
        self.opponent = None         # frozen ego policy when training adversary
        self.adv_rew_buf = torch.zeros_like(self.rew_buf)
        self.adv_tilt_cos = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.adv_crash_condition = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)

        self.adv_rel_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.adv_last_rel_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.adv_rel_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.adv_tgt_vel = torch.zeros_like(self.tgt_vel)
        self.adv_tgt_vel_est = torch.zeros_like(self.tgt_vel)
        self.adv_tgt_acc_est = torch.zeros_like(self.tgt_vel)

        self.adv_stable_cnt = torch.zeros(self.num_envs, dtype=torch.int32, device=gs.device)
        self.adv_n_stable = int(round(2*self.env_cfg["stable_time_s"] / self.dt))
        self.adv_success = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)

    # ---------- public API ----------
    def set_opponent(self, policy_callable):
        self.opponent = policy_callable
    def set_train_role(self, role:str):
        assert role in ("ego","adv"); self.train_role = role

    # ---------- helpers ----------
    def build_obs_template_dim(self):
        # build once to learn feature count without side-effects
        dummy = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        return self._concat_obs(
            rel_pos=dummy, rel_vel=dummy,
            tgt_vel=dummy, tgt_acc=dummy,
            dist=torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float),
            vel_close=torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float),
            t_go=torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float),
            target_vector=dummy,
            base_quat=torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float),
            base_lin=dummy, base_ang=dummy,
            last_actions=torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float),
            retarget_flag=torch.zeros((self.num_envs, 1), device=gs.device, dtype=gs.tc_float),
            extra_pos=dummy,
        ).shape[-1]

    def _concat_obs(self, **t):
        return torch.cat([
            torch.clip(t["rel_pos"] * self.obs_scales["rel_pos"], -1, 1),
            torch.clip(t["rel_vel"] * self.obs_scales["lin_vel"], -1, 1),
            torch.clip(t["tgt_vel"] * self.obs_scales.get("tgt_vel", 1/3.0), -1, 1),
            torch.clip(t["tgt_acc"] * self.obs_scales.get("tgt_acc", 1/10.0), -1, 1),
            torch.clip(t["dist"] * self.obs_scales.get("dist", 1.0), 0.0, 2.0).unsqueeze(-1),
            torch.clip(t["vel_close"] * self.obs_scales.get("vel_close", 1.0), -2.0, 2.0).unsqueeze(-1),
            (t["t_go"] * self.obs_scales.get("t_go", 1.0)).unsqueeze(-1),
            t["target_vector"],
            t["base_quat"],
            torch.clip(t["base_lin"] * self.obs_scales["lin_vel"], -1, 1),
            torch.clip(t["base_ang"] * self.obs_scales["ang_vel"], -1, 1),
            t["last_actions"],
            t["retarget_flag"],
            torch.clip(t["extra_pos"] * self.obs_scales["rel_pos"], -1, 1),
        ], dim=-1)

    def _success_mask(self):
        near = self.rel_pos.norm(dim=1) < self.env_cfg["at_target_threshold"]
        slow = self.rel_vel.norm(dim=1) < self.env_cfg["max_rel_speed_mps"]
        level = self.base_tilt_cos > self.max_tilt_cos
        angvel = self.base_ang_vel.norm(dim=1) < self.env_cfg["max_angvel_radps"]
        return near# & slow & level & angvel

    def _adv_success_mask(self):
        return self.adv_rel_pos.norm(dim=1) < self.env_cfg["at_target_threshold"]

    def _gaussian_gate(self, dist, decay=None):
        if decay is None:
            decay = max(float(self.env_cfg["near_gate_factor"] * self.env_cfg["at_target_threshold"]), 1e-6)
        return torch.exp(- (dist / decay) ** 2)

    # ---------- resets and sampling ----------
    def _resample_adv_path(self, envs_idx):
        v_max = float(self.env_cfg.get("adv_max_v", 1.0))
        v_min = float(self.env_cfg.get("adv_min_v", 0.5))
        f_max = float(self.env_cfg.get("adv_max_f", 0.6))
        f = torch.rand((len(envs_idx), 3), device=gs.device, dtype=gs.tc_float) * f_max
        f.clamp_min_(0.1)
        v = torch.rand_like(f) * v_max
        v.clamp_min_(v_min)
        a = v / (math.tau * f * math.sqrt(3))
        # keep Z excursions above ground margin
        a[:, 2] = torch.minimum(
            a[:, 2],
            (self.commands_anchor[envs_idx, 2] - self.z_margin).clamp_min(0.0),
        )
        phi = torch.rand_like(f) * math.tau
        self.path_a[envs_idx] = a
        self.path_f[envs_idx] = f
        self.path_phi[envs_idx] = phi

    def _resample_commands(self, envs_idx):
        spawn_clearance = 0.5
        todo = torch.ones(len(envs_idx), dtype=torch.bool, device=gs.device)
        MAX_TRIES = 4
        for _ in range(MAX_TRIES):
            todo_idx = envs_idx[todo]
            if todo_idx.numel() == 0:
                break
            self.commands[todo_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(todo_idx),), gs.device)
            self.commands[todo_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(todo_idx),), gs.device)
            self.commands[todo_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(todo_idx),), gs.device)

            # place ego and adversary far enough in XY
            rel = self.commands[todo_idx] - self.base_pos[todo_idx]
            ok = rel[:, :2].norm(dim=1) >= spawn_clearance
            temp = todo.clone()
            todo[temp] = ~ok

        # path anchor and new sinusoid
        self.commands_anchor[envs_idx] = self.commands[envs_idx]
        self._resample_adv_path(envs_idx)

        # reset rel terms
        self.rel_pos[envs_idx] = 0.0
        self.last_rel_pos[envs_idx] = 0.0
        self.rel_vel[envs_idx] = 0.0
        self.tgt_vel[envs_idx] = 0.0
        self.tgt_vel_est[envs_idx] = 0.0
        self.tgt_acc_est[envs_idx] = 0.0

        self.adv_rel_pos[envs_idx] = 0.0
        self.adv_last_rel_pos[envs_idx] = 0.0
        self.adv_rel_vel[envs_idx] = 0.0
        self.adv_tgt_vel[envs_idx] = 0.0
        self.adv_tgt_vel_est[envs_idx] = 0.0
        self.adv_tgt_acc_est[envs_idx] = 0.0

        self.stable_cnt[envs_idx] = 0
        self.success[envs_idx] = False

    def _log_episode_stats(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.extras["episode"] = ep = {}

        eps = 1e-6
        inv_step = 1.0 / (self.m_step[envs_idx] + eps)
        inv_near = 1.0 / (self.m_near_cnt[envs_idx] + eps)

        # metrics (mean over the just-reset subset, same names as single-drone env)
        d_mean_t = (self.m_d_sum[envs_idx]      * inv_step).mean()
        inside_t = (self.m_inside_sum[envs_idx] * inv_step).mean()
        vapp_t   = (self.m_vapp_err_sum[envs_idx] * inv_near).mean()
        vtan_t   = (self.m_vtan_sum[envs_idx]     * inv_near).mean()
        vtgt_t   = (self.m_vtgt_sum[envs_idx]     * inv_step).mean()
        angvel_t = (self.m_angvel_sum[envs_idx]   * inv_near).mean()

        if self.train_role == "ego":
            ep["metric_d_mean"]          = d_mean_t.item()
            ep["metric_inside_succ_pct"] = inside_t.item()
            ep["metric_v_app_err_near"]  = vapp_t.item()
            ep["metric_v_tan_near"]      = vtan_t.item()
            ep["metric_v_tgt_mean"]      = vtgt_t.item()
            ep["metric_angvel_near"]     = angvel_t.item()
        # TODO adv, todo fix d_mean - see https://chatgpt.com/g/g-p-689c90bcc49481919cb933c38bc0ad80-rl/c/6909c618-199c-8326-b54e-493c051da7ba

        # reward buckets
        if self.train_role == "ego":
            rew_keys = list(self.episode_sums.keys())
            if rew_keys:
                vals = torch.stack([self.episode_sums[k][envs_idx].mean() for k in rew_keys]) / self.env_cfg["episode_length_s"]
                for k, v in zip(rew_keys, vals.detach().cpu().tolist()):
                    ep[f"rew_{k}"] = v
        else:  # adversary training
            rew_keys = list(self.adv_episode_sums.keys())
            if rew_keys:
                vals = torch.stack([self.adv_episode_sums[k][envs_idx].mean() for k in rew_keys]) / self.env_cfg["episode_length_s"]
                for k, v in zip(rew_keys, vals.detach().cpu().tolist()):
                    ep[f"rew_{k}"] = v

        # termination histogram for those envs
        ri = self.extras.get("term_reset_idx_now", None)
        if ri is not None and len(ri) > 0:
            for k, v in self.extras.get("term_causes_now", {}).items():
                ep[k] = v[ri].mean().item()
            ep["term_timeout"] = self.extras["term_timeout"][ri].mean().item()

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        # ego pose
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)
        self.prev_ang_norm[envs_idx] = 0
        self.prev_yaw_abs[envs_idx] = 0
        self.last_actions[envs_idx] = 0.0

        # sample target point
        self._resample_commands(envs_idx)
        self._respawn_adv(envs_idx)


        # episode stats logging
        self.extras["episode"] = {}

        # --- log before clearing ---
        self._log_episode_stats(envs_idx)

        # --- clear accumulators (ego + adversary) ---
        for t in [self.m_d_sum, self.m_step, self.m_inside_sum, self.m_vapp_err_sum,
                self.m_near_cnt, self.m_vtan_sum, self.m_vtgt_sum, self.m_angvel_sum]:
            t[envs_idx] = 0
        for k in list(self.episode_sums.keys()):
            self.episode_sums[k][envs_idx] = 0.0
        for k in list(self.adv_episode_sums.keys()):
            self.adv_episode_sums[k][envs_idx] = 0.0

        # --- also reset adversary distance cache to avoid cross-episode spikes ---

    def _respawn_adv(self, envs_idx):
        if len(envs_idx) == 0:
            return
        jitter_xy = 0.3
        xy = torch.randn((len(envs_idx), 2), device=gs.device, dtype=gs.tc_float) * jitter_xy
        adv_base = torch.zeros((len(envs_idx), 3), device=gs.device, dtype=gs.tc_float)
        adv_base[:, :2] = self.commands[envs_idx, :2] + xy
        adv_base[:, 2]  = (self.commands[envs_idx, 2] + self.adv_base_offset[2])

        self.adv_pos[envs_idx] = adv_base
        self.adv_last_pos[envs_idx] = adv_base
        self.adv_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.adversary.set_pos(self.adv_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.adversary.set_quat(self.adv_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.adv_lin_vel[envs_idx] = 0
        self.adv_ang_vel[envs_idx] = 0
        self.adversary.zero_all_dofs_velocity(envs_idx)
        self.adv_prev_ang_norm[envs_idx] = 0
        self.adv_prev_yaw_abs[envs_idx] = 0
        self.adv_last_actions[envs_idx] = 0.0

        self.adv_rel_pos[envs_idx] = self.commands[envs_idx] - self.adv_pos[envs_idx]
        self.adv_last_rel_pos[envs_idx] = self.adv_rel_pos[envs_idx]
        self.adv_rel_vel[envs_idx] = 0.0
        self.adv_tgt_vel[envs_idx] = 0.0
        self.adv_tgt_vel_est[envs_idx] = 0.0
        self.adv_tgt_acc_est[envs_idx] = 0.0

        # follower aims slightly above leader top
        adv_top = self.adv_pos[envs_idx] - self.adv_base_offset + 0.05 * self.world_z[envs_idx]
        self.commands_adv[envs_idx] = adv_top
        self.last_commands_adv[envs_idx] = adv_top

        # keep ego; refresh relative state and caches
        self.rel_pos[envs_idx] = self.commands_adv[envs_idx] - self.base_pos[envs_idx]
        self.last_rel_pos[envs_idx] = self.rel_pos[envs_idx]
        self.rel_vel[envs_idx] = 0.0

        # approach cache
        self.update_approach_geometry(self.app_geom, self.rel_pos, self.rel_vel, envs_idx)
        d, _, vel_close, _ = self.app_geom
        self.prev_dist[envs_idx] = d[envs_idx]
        self.prev_vel_close[envs_idx] = vel_close[envs_idx]
        self.prev_v_tan[envs_idx] = 0.0

        self.update_approach_geometry(self.adv_app_geom, self.adv_rel_pos, self.adv_rel_vel, envs_idx)
        adv_d, _, adv_vel_close, _ = self.adv_app_geom
        self.adv_prev_dist[envs_idx] = adv_d[envs_idx]
        self.adv_prev_vel_close[envs_idx] = adv_vel_close[envs_idx]
        self.adv_prev_v_tan[envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))

        # push initial obs for both actors
        obs_t = self.build_obs()
        adv_obs_t = self.build_adv_obs()
        self.stacker.push(obs_t, torch.ones(self.num_envs, dtype=torch.bool, device=gs.device))
        self.adv_stacker.push(adv_obs_t, torch.ones(self.num_envs, dtype=torch.bool, device=gs.device))
        self.obs_buf = self.stacker.stacked()
        self.adv_obs_buf = self.adv_stacker.stacked()
        return self.obs_buf, None

    # ---------- geometry ----------
    def calculate_approach_geometry(self, rel_pos, rel_vel, envs_idx=None):
        rp = rel_pos if envs_idx is None else rel_pos[envs_idx]
        rv = rel_vel if envs_idx is None else rel_vel[envs_idx]
        dist = torch.norm(rp, dim=1)
        unit_vec = rp / dist.clamp_min(1e-6).unsqueeze(1)
        vel_close = - (rv * unit_vec).sum(dim=1)
        t_go = torch.clamp(dist / (vel_close.abs() + 1e-3), 0.0, self.env_cfg["tgo_cap"])
        return dist, unit_vec, vel_close, t_go

    def update_approach_geometry(self, app_geom, rel_pos, rel_vel, envs_idx=None):
        d, u, vel_close, t_go = self.calculate_approach_geometry(rel_pos, rel_vel, envs_idx)
        d_old, u_old, vel_close_old, t_go_old = app_geom
        if envs_idx is None:
            d_old.copy_(d)
            u_old.copy_(u)
            vel_close_old.copy_(vel_close)
            t_go_old.copy_(t_go)        
        else:
            d_old[envs_idx] = d
            u_old[envs_idx] = u
            vel_close_old[envs_idx] = vel_close
            t_go_old[envs_idx] = t_go

    # ---------- observations ----------
    def build_obs(self):
        dist, target_vector, vel_close, t_go = self.app_geom
        retarget_flag = torch.zeros((self.num_envs, 1), device=gs.device, dtype=gs.tc_float)
        extra_pos = self.commands - self.base_pos
        return self._concat_obs(
            rel_pos=self.rel_pos,
            rel_vel=self.rel_vel,
            tgt_vel=self.tgt_vel,
            tgt_acc=self.tgt_acc_est,
            dist=dist,
            vel_close=vel_close,
            t_go=t_go,
            target_vector=target_vector,
            base_quat=self.base_quat,
            base_lin=self.base_lin_vel,
            base_ang=self.base_ang_vel,
            last_actions=self.last_actions,
            retarget_flag=retarget_flag,
            extra_pos=extra_pos,
        )

    def build_adv_obs(self):
        dist, target_vector, vel_close, t_go = self.adv_app_geom
        retarget_flag = torch.zeros((self.num_envs, 1), device=gs.device, dtype=gs.tc_float)
        extra_pos = self.base_pos - self.adv_pos
        return self._concat_obs(
            rel_pos=self.adv_rel_pos,
            rel_vel=self.adv_rel_vel,
            tgt_vel=self.adv_tgt_vel,
            tgt_acc=self.adv_tgt_acc_est,
            dist=dist,
            vel_close=vel_close,
            t_go=t_go,
            target_vector=target_vector,
            base_quat=self.adv_quat,
            base_lin=self.adv_lin_vel,
            base_ang=self.adv_ang_vel,
            last_actions=self.adv_last_actions,
            retarget_flag=retarget_flag,
            extra_pos=extra_pos,
        )

    def get_observations(self):
        if self.train_role == "ego":
            self.extras["observations"]["critic"] = self.obs_buf
            return self.obs_buf, self.extras
        else:
            self.extras["observations"]["critic"] = self.adv_obs_buf
            return self.adv_obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    # ---------- step ----------
    def step(self, actions):
        # update adversary setpoint path first
        t = self.episode_length_buf.unsqueeze(-1).to(self.path_f.dtype) * self.dt  # [N,1]
        path = self.path_a * torch.sin(math.tau * self.path_f * t + self.path_phi)
        self.last_commands = self.commands
        self.commands = self.commands_anchor + path

        # actions
        if self.train_role == "ego":
            # PPO controls ego; adversary comes from its policy
            self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
            if self.opponent is not None:
                with torch.no_grad():
                    adv_act = self.opponent(self.adv_obs_buf)
                self.adv_actions = torch.clip(adv_act, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
            else:
                self.adv_actions[:] = 0.0
        else:
            # PPO controls adversary; ego comes from frozen opponent
            self.adv_actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
            if self.opponent is not None:
                with torch.no_grad():
                    ego_act = self.opponent(self.obs_buf)
                self.actions = torch.clip(ego_act, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
            else:
                self.actions[:] = 0.0

        # apply RPMs (14468 is hover)
        self.drone.set_propellels_rpm((1 + self.actions * 0.7) * 14468.429183500699)
        self.adversary.set_propellels_rpm((1 + self.adv_actions * 0.9) * 14468.429183500699) # make adversary slightly faster - TODO check if it works

        # advance physics
        self.scene.step()
        self.episode_length_buf += 1

        # ego state
        self.last_base_pos.copy_(self.base_pos)
        pos = self.drone.get_pos(); quat = self.drone.get_quat()
        lin = self.drone.get_vel(); ang = self.drone.get_ang()
        pos.nan_to_num_(nan=0.0); quat.nan_to_num_(nan=0.0); lin.nan_to_num_(nan=0.0); ang.nan_to_num_(nan=0.0)
        quat = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-6)
        inv_q = inv_quat(quat)
        self.base_pos.copy_(pos); self.base_quat.copy_(quat)
        self.base_lin_vel.copy_(transform_by_quat(lin, inv_q))
        self.base_ang_vel.copy_(transform_by_quat(ang, inv_q))

        # adversary state
        self.adv_last_pos.copy_(self.adv_pos)
        apos = self.adversary.get_pos(); aquat = self.adversary.get_quat()
        alin = self.adversary.get_vel(); aang = self.adversary.get_ang()
        apos.nan_to_num_(nan=0.0); aquat.nan_to_num_(nan=0.0); alin.nan_to_num_(nan=0.0); aang.nan_to_num_(nan=0.0)
        aquat = aquat / aquat.norm(dim=1, keepdim=True).clamp_min(1e-6)
        adv_inv_q = inv_quat(aquat)
        self.adv_pos.copy_(apos); self.adv_quat.copy_(aquat)
        self.adv_lin_vel.copy_(transform_by_quat(alin, adv_inv_q))
        self.adv_ang_vel.copy_(transform_by_quat(aang, adv_inv_q))

        # ego target is above adversary top
        adv_top = self.adv_pos - self.adv_base_offset + 0.05 * self.world_z
        self.last_commands_adv.copy_(self.commands_adv)
        self.commands_adv.copy_(adv_top)
        meas_tgt_vel = (self.commands_adv - self.last_commands_adv) / self.dt
        self.tgt_vel_est = 0.8 * self.tgt_vel_est + 0.2 * meas_tgt_vel
        meas_tgt_acc = (self.tgt_vel_est - self.tgt_vel) / self.dt
        self.tgt_acc_est = 0.9 * self.tgt_acc_est + 0.1 * meas_tgt_acc
        self.tgt_vel = self.tgt_vel_est

        adv_meas_tgt_vel = (self.commands - self.last_commands) / self.dt
        self.adv_tgt_vel_est = 0.8 * self.adv_tgt_vel_est + 0.2 * adv_meas_tgt_vel
        adv_meas_tgt_acc = (self.adv_tgt_vel_est - self.adv_tgt_vel) / self.dt
        self.adv_tgt_acc_est = 0.9 * self.adv_tgt_acc_est + 0.1 * adv_meas_tgt_acc
        self.adv_tgt_vel = self.adv_tgt_vel_est

        # relative kinematics
        self.last_rel_pos.copy_(self.rel_pos)
        self.rel_pos.copy_(self.commands_adv - self.base_pos)
        self.rel_vel.copy_((self.rel_pos - self.last_rel_pos) / self.dt)

        self.adv_last_rel_pos.copy_(self.adv_rel_pos)
        self.adv_rel_pos.copy_(self.commands - self.adv_pos)
        self.adv_rel_vel.copy_((self.adv_rel_pos - self.adv_last_rel_pos) / self.dt)

        # approach geometry
        d_prev, _, vc_prev, _ = self.app_geom
        self.prev_dist.copy_(d_prev)
        self.prev_vel_close.copy_(vc_prev)
        self.update_approach_geometry(self.app_geom, self.rel_pos, self.rel_vel)

        adv_d_prev, _, adv_vc_prev, _ = self.adv_app_geom
        self.adv_prev_dist.copy_(adv_d_prev)
        self.adv_prev_vel_close.copy_(adv_vc_prev)
        self.update_approach_geometry(self.adv_app_geom, self.adv_rel_pos, self.adv_rel_vel)

        # contacts
        adv_contact = self.adversary.get_contacts(with_entity=self.drone, exclude_self_contact=True)
        self.adv_collision = (adv_contact['penetration'] > 0).any(dim=1)

        # metrics
        d, u, vel_close, _ = self.app_geom
        v_des = torch.clamp(self.env_cfg["approach_k"] * d, max=self.env_cfg["approach_v_cap"])
        sigma = max(self.env_cfg["near_gate_factor"] * self.env_cfg["at_target_threshold"], self.env_cfg["at_target_threshold"])
        v_tan = self.rel_vel - (self.rel_vel * u).sum(dim=1, keepdim=True) * u
        w_norm = self.base_ang_vel.norm(dim=1)

        self.m_d_sum += d; self.m_step += 1
        self.m_inside_sum += (d < self.env_cfg["at_target_threshold"]).float()
        self.m_vtgt_sum += self.tgt_vel.norm(dim=1)
        mask = d < sigma
        self.m_near_cnt += mask.float()
        self.m_vapp_err_sum += torch.where(mask, (v_des - vel_close).abs(), 0.0)
        self.m_vtan_sum += torch.where(mask, v_tan.norm(dim=1), 0.0)
        self.m_angvel_sum += torch.where(mask, w_norm, 0.0)

        # tilt
        self.base_tilt_cos = transform_by_quat(self.world_z, inv_q)[:, 2].clamp(-1.0, 1.0)
        self.adv_tilt_cos = transform_by_quat(self.world_z, adv_inv_q)[:, 2].clamp(-1.0, 1.0)

        # success and terms (ego only)
        success_mask = self._success_mask()
        adv_success_mask = self._adv_success_mask()

        tilt_term = (self.base_tilt_cos < self.term_tilt_cos)
        yaw_term = (torch.abs(self.base_ang_vel[:, 2]) > self.env_cfg["termination_if_yaw_rate_greater_than"])
        floor_term = (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])

        adv_tilt_term = (self.adv_tilt_cos < self.term_tilt_cos)
        adv_yaw_term = (torch.abs(self.adv_ang_vel[:, 2]) > self.env_cfg["termination_if_yaw_rate_greater_than"])
        adv_floor_term = (self.adv_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])

        self.extras["term_causes_now"] = {
            "term_tilt": tilt_term.float(),
            "term_yaw": yaw_term.float(),
            "term_floor": floor_term.float(),
            "term_collision": self.adv_collision.float(),
            "term_adv_tilt": adv_tilt_term.float(),
            "term_adv_yaw": adv_yaw_term.float(),
            "term_adv_floor": adv_floor_term.float(),
        }

        if self.env_cfg["eval"]:
            self.crash_condition = (tilt_term | yaw_term | floor_term | self.adv_collision)
            self.adv_crash_condition = (adv_tilt_term | adv_yaw_term | adv_floor_term | self.adv_collision)
        else:
            self.crash_condition = (floor_term | self.adv_collision)
            self.adv_crash_condition = (adv_floor_term | self.adv_collision)

        self.stable_cnt = torch.where(success_mask, (self.stable_cnt + 1).clamp_max(self.n_stable), torch.zeros_like(self.stable_cnt))
        self.success = self.stable_cnt >= self.n_stable
        self.adv_stable_cnt = torch.where(adv_success_mask, (self.adv_stable_cnt + 1).clamp_max(self.adv_n_stable), torch.zeros_like(self.adv_stable_cnt))
        self.adv_success = self.adv_stable_cnt >= self.adv_n_stable

        # visualize targets: red = ego target (adv top), blue = adversary setpoint
        if self.target is not None:
            near = (self.rel_pos.norm(dim=1) < self.env_cfg["at_target_threshold"])
            threshold_pos = torch.where(near.unsqueeze(1), self.commands_adv, self.highlight_hide)
            reached_pos = torch.where(success_mask.unsqueeze(1), self.commands_adv, self.highlight_hide)
            self.target.set_pos(self.commands_adv, zero_velocity=True)
            self.target_threshold_highlight.set_pos(threshold_pos, zero_velocity=True)
            self.target_reached_highlight.set_pos(reached_pos, zero_velocity=True)
            if self.adv_target_marker is not None:
                self.adv_target_marker.set_pos(self.commands, zero_velocity=True)

        # rewards
        self.rewards_mapper()
        if self.train_role == "ego":
            self.rew_buf[:] = 0.0
            for name, reward_func in self.reward_functions.items():
                rew = reward_func() * self.reward_scales[name]
                rew = torch.nan_to_num(rew, 0.0, 0.0, 0.0).clamp_(-1000.0, 1000.0)
                self.rew_buf += rew
                self.episode_sums[name] += rew
        else:
            self.adv_rew_buf[:] = 0.0
            for name, reward_func in self.adv_reward_functions.items():
                rew = reward_func() * self.adv_reward_scales[name]
                rew = torch.nan_to_num(rew, 0.0, 0.0, 0.0).clamp_(-1000.0, 1000.0)
                self.adv_rew_buf += rew
                self.adv_episode_sums[name] += rew

        # resets
        if self.train_role == "ego":
            crash = self.crash_condition
        else:
            crash = self.adv_crash_condition
            
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | crash
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        reset_idx_now = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.extras["term_timeout"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["term_timeout"][(self.episode_length_buf > self.max_episode_length)] = 1.0
        self.extras["term_reset_idx_now"] = reset_idx_now

        self.reset_idx(reset_idx_now)

        # resample after success
        adv_envs_idx = torch.nonzero(self.adv_success, as_tuple=False).flatten()
        if len(adv_envs_idx) > 0:
            self._resample_commands(adv_envs_idx)
            self._respawn_adv(adv_envs_idx)
        envs_idx = torch.nonzero(self.success, as_tuple=False).flatten()
        if len(envs_idx) > 0:
            self._resample_commands(envs_idx)
            self._respawn_adv(envs_idx)

        # observations for next step
        obs_t = torch.clamp(torch.nan_to_num(self.build_obs(), nan=0.0, posinf=1e6, neginf=-1e6), -100.0, 100.0)
        adv_obs_t = torch.clamp(torch.nan_to_num(self.build_adv_obs(), nan=0.0, posinf=1e6, neginf=-1e6), -100.0, 100.0)

        done = self.reset_buf.bool(); done[envs_idx] = True # adv success does not lead to done
        self.stacker.push(obs_t, done)
        self.adv_stacker.push(adv_obs_t, done)
        self.obs_buf = self.stacker.stacked()
        self.adv_obs_buf = self.adv_stacker.stacked()

        self.last_actions[:] = self.actions[:]
        self.adv_last_actions[:] = self.adv_actions[:]

        # choose obs/reward by role before return
        if self.train_role == "ego":
            obs, rew = self.obs_buf, self.rew_buf
        else:
            obs, rew = self.adv_obs_buf, self.adv_rew_buf  # see reward below

        self.extras["observations"]["critic"] = obs
        return obs, rew, self.reset_buf, self.extras


    def rewards_mapper(self):
        if self.train_role == "ego":
            self.rew_crash_condition = self.crash_condition
            self.rew_rel_vel = self.rel_vel
            self.rew_ang_vel = self.base_ang_vel
            self.rew_actions = self.actions
            self.rew_last_actions = self.last_actions
            self.rew_app_geom = self.app_geom
            self.rew_prev_dist = self.prev_dist
            self.rew_prev_vel_close = self.prev_vel_close
            self.rew_prev_v_tan = self.prev_v_tan
            self.rew_prev_yaw_abs = self.prev_yaw_abs
            self.rew_prev_ang_norm = self.prev_ang_norm
        else:
            self.rew_crash_condition = self.adv_crash_condition
            self.rew_rel_vel = self.adv_rel_vel
            self.rew_ang_vel = self.adv_ang_vel
            self.rew_actions = self.adv_actions
            self.rew_last_actions = self.adv_last_actions
            self.rew_app_geom = self.adv_app_geom
            self.rew_prev_dist = self.adv_prev_dist
            self.rew_prev_vel_close = self.adv_prev_vel_close
            self.rew_prev_v_tan = self.adv_prev_v_tan
            self.rew_prev_yaw_abs = self.adv_prev_yaw_abs
            self.rew_prev_ang_norm = self.adv_prev_ang_norm

    # ---------- rewards ----------
    def _reward_approach(self, ego=False):
        if ego:
            app_geom = self.app_geom
            prev_dist = self.prev_dist
            prev_vel_close = self.prev_vel_close
        else:
            app_geom = self.rew_app_geom
            prev_dist = self.rew_prev_dist
            prev_vel_close = self.rew_prev_vel_close

        dist, _, vel_close, _ = app_geom
        dist_prev, vel_close_prev = prev_dist, prev_vel_close
        k, v_cap = self.env_cfg["approach_k"], self.env_cfg["approach_v_cap"]
        dist_soft = self.env_cfg["near_gate_factor"] * self.env_cfg["at_target_threshold"]
        vel_soft = 0.05

        def phi(d, v):
            dist_target = torch.zeros_like(d)
            vel_target = torch.clamp(k * d, max=v_cap)
            dist_rew = functional.smooth_l1_loss(d, dist_target, beta=dist_soft, reduction="none")
            vel_rew = functional.smooth_l1_loss(v, vel_target, beta=vel_soft, reduction="none")
            return dist_rew + vel_rew

        return phi(dist_prev, vel_close_prev) - phi(dist, vel_close)

    def _reward_adv_escape(self):
        dist = (self.base_pos - self.adv_pos).norm(dim=1)
        gate = self._gaussian_gate(dist=dist, decay=self.env_cfg["near_gate_factor"]*self.env_cfg["at_target_threshold"])
        return gate * -self._reward_approach(ego=True)

    def _reward_adv_approach(self):
        # deprioritize when ego is close
        dist = (self.base_pos - self.adv_pos).norm(dim=1)
        gate = 1.0 - self._gaussian_gate(dist=dist, decay=self.env_cfg["near_gate_factor"]*self.env_cfg["at_target_threshold"])
        return gate * self._reward_approach(ego=False)

    def _reward_tan_vel_align(self):
        dist, u, _, _ = self.rew_app_geom
        gate = self._gaussian_gate(dist)
        gate_prev = self._gaussian_gate(self.rew_prev_dist)
        v_tan = (self.rew_rel_vel - (self.rew_rel_vel * u).sum(dim=1, keepdim=True) * u).norm(dim=1)
        v_tan_rew = gate_prev * self.rew_prev_v_tan - gate * v_tan
        self.rew_prev_v_tan.copy_(v_tan)
        return v_tan_rew

    def _reward_smooth(self):
        return -torch.sum(torch.square(self.rew_actions - self.rew_last_actions), dim=1)

    def _reward_ang_vel(self):
        dist, _, _, _ = self.rew_app_geom
        gate = self._gaussian_gate(dist)
        gate_prev = self._gaussian_gate(self.rew_prev_dist)
        ang_norm = self.rew_ang_vel.norm(dim=1)
        yaw_abs = self.rew_ang_vel[:, 2].abs()
        angvel_near_rew = gate_prev * self.rew_prev_ang_norm - gate * ang_norm
        yaw_rew = (self.rew_prev_yaw_abs - yaw_abs)
        self.rew_prev_ang_norm.copy_(ang_norm)
        self.rew_prev_yaw_abs.copy_(yaw_abs)
        return angvel_near_rew + yaw_rew

    def _reward_crash(self):
        crash = self.rew_crash_condition.float()
        impact = self.rew_rel_vel.norm(dim=1) + 0.5 * self.rew_ang_vel.norm(dim=1)
        return -(crash * (1.0 + impact))

    def _reward_success(self):
        return self.success.to(self.rew_buf.dtype)

    def _reward_adv_success(self):
        return self.adv_success.to(self.rew_buf.dtype)
