import torch
import math
import copy
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from torch.nn import functional


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device, dtype=gs.tc_float) + lower

class ObsStacker:
    def __init__(self, num_envs, obs_dim, K, device):
        self.N, self.D, self.K, self.device = num_envs, obs_dim, K, device
        self.rows = torch.arange(self.N, dtype=torch.long, device=device)
        self.offsets = torch.arange(self.K, dtype=torch.long, device=self.device)
        self.buf = torch.zeros(num_envs, K, obs_dim, dtype=gs.tc_float, device=device)
        self.ptr = torch.zeros(num_envs, dtype=torch.long, device=device)  # per-env write index
            
    @torch.no_grad()
    def push(self, obs_t, done):  # done: bool [N]
        inc = (~done).to(self.ptr.dtype)
        self.ptr.add_(inc).remainder_(self.K)
        self.buf[self.rows, self.ptr] = obs_t

        self.ptr[done] = 0
        obs_t_broadcast = obs_t[done].unsqueeze(1).expand(-1, self.K, -1)
        self.buf[done] = obs_t_broadcast
        self.buf[done, :, -1] = 1.0
        self.buf[done, 0, -1] = 0.0

    @torch.no_grad()
    def stacked(self):
        idx = (self.ptr[:, None] - self.offsets[None, :]) % self.K  # [N,K]
        out = self.buf.gather(1, idx[..., None].expand(-1, -1, self.D))  # [N,K,D]
        return out.reshape(self.N, self.K * self.D)  # [N, K·D]

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
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        #self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
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
            # renderer=gs.renderers.RayTracer(  # type: ignore
            #     env_surface=gs.surfaces.Emission(
            #         emissive_texture=gs.textures.ImageTexture(
            #             image_path="textures/indoor_bright.png",
            #         ),
            #     ),
            #     env_radius=15.0,
            #     env_euler=(0, 0, 180),
            #     lights=[
            #         {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
            #     ],
            # ),
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.05, fixed=False, collision=False),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5)))
            )

            self.target_threshold_highlight = self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.051, fixed=False, collision=False),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.5, 0.75, 0.5)))
            )
            self.target_reached_highlight = self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.052, fixed=False, collision=False),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0)))
            )
            self.highlight_hide = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float)
        else:
            self.target = None
            self.target_threshold_highlight = None
            self.target_reached_highlight = None

        # add adversary
        if self.env_cfg.get("use_adversary", True):
            self.adversary_dog = self.scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", fixed=True, collision=False))
            self.adversary = self.scene.add_entity(
                morph=gs.morphs.Box(size=(0.25, 0.25, self.env_cfg["adv_box_h"]), fixed=False, collision=True),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.62, 0.64, 0.66)))
            )
            # add a thin inlaid pad for visual cue (no collision) ---
            self.adversary_pad = self.scene.add_entity(
                morph=gs.morphs.Box(size=(0.22, 0.22, 0.01), fixed=False, collision=False),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.90, 0.92, 0.94)))
            )
        else:
            self.adversary_dog = None
            self.adversary = None
            self.adversary_pad = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(960, 540),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # build scene
        self.scene.build(n_envs=num_envs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        EVENT_REWARDS = {"success", "crash"}  # no dt scaling
        for name in self.reward_scales.keys():
            if name not in EVENT_REWARDS:
                self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_adv = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.last_commands_adv = torch.zeros_like(self.commands)  # set after build()

        self.difficulty = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float) * self.env_cfg["adv_difficulty_min"]

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

        self.world_z = torch.tensor([0.0, 0.0, 1.0], device=gs.device, dtype=gs.tc_float).expand(self.num_envs, 3)
        self.max_tilt_cos  = math.cos(math.radians(self.env_cfg["max_tilt_deg"]))
        self.term_tilt_cos = math.cos(math.radians(self.env_cfg["termination_if_tilt_greater_than"]))
        self.base_tilt_cos = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)

        self.tgt_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.tgt_vel_est = torch.zeros_like(self.tgt_vel)
        self.tgt_acc_est = torch.zeros_like(self.tgt_vel)
        self.app_geom = (
            torch.zeros((self.num_envs), device=gs.device),     # dist
            torch.zeros((self.num_envs,3), device=gs.device),   # u
            torch.zeros((self.num_envs), device=gs.device),     # vel_close
            torch.zeros((self.num_envs), device=gs.device),     # t_go
        )
        self.prev_dist  = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.prev_vel_close = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)

        # reward shaping buffers
        self.prev_v_tan = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.prev_ang_norm = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.prev_yaw_abs = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        D = self.build_obs().shape[-1]
        K = self.env_cfg["obs_stacks"]
        self.stacker = MultiRateStacker(self.num_envs, D, K, gs.device, group=5)
        self.num_obs = D * K
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)

        self.adv_a = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.adv_f = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.adv_phi = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.adv_sinus = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        z_off = -(self.env_cfg["z_margin"] + 0.5*self.env_cfg["adv_box_h"])
        dog_off = -(self.env_cfg["z_margin"] + self.env_cfg["adv_box_h"] + 0.08)
        pad_off = z_off + 0.01
        self.adv_base_offset = torch.tensor([0.0, 0.0, z_off], device=gs.device, dtype=gs.tc_float)
        self.adv_dog_offset = torch.tensor([0.05, 0.0, dog_off], device=gs.device, dtype=gs.tc_float)
        self.adv_pad_offset = torch.tensor([0.0, 0.0, pad_off], device=gs.device, dtype=gs.tc_float)

        self.adv_collision = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        self.crash_condition = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)

        # metric accumulators
        self.m_d_sum = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_step = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_inside_sum = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_near_cnt = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_vapp_err_sum = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_vtan_sum = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_angvel_sum = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        self.m_vtgt_sum = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)

        self.stable_cnt = torch.zeros(self.num_envs, dtype=torch.int32, device=gs.device)   # counter of current stable frames
        self.n_stable   = int(round(self.env_cfg["stable_time_s"] / self.dt))               # amount of frames to be stable for success
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        # dog — build local DOF order and default pose in that order
        pairs = sorted(
            [(int(self.adversary_dog.get_joint(n).dof_start), n)
            for n in self.env_cfg["joint_names"]],
            key=lambda x: x[0]
        )
        self.dog_joint_names_local = [n for _, n in pairs]  # entity-local joint order
        self.dog_all_local_idx = torch.arange(len(self.dog_joint_names_local),
                                            device=gs.device, dtype=torch.long)

        self.default_dof_pos_local = torch.tensor(
            [self.env_cfg["default_joint_angles"][n] for n in self.dog_joint_names_local],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.dog_dof_pos = self.default_dof_pos_local.unsqueeze(0).repeat(self.num_envs, 1)


    def _resample_adv(self, envs_idx):
        v_max = self.env_cfg["adv_max_v"]
        f_max = self.env_cfg["adv_max_f"]
        difficulty = self.difficulty[envs_idx].unsqueeze(-1)
        adv_v = torch.rand((len(envs_idx), 3), device=gs.device, dtype=gs.tc_float) * v_max * difficulty
        adv_f = torch.rand((len(envs_idx), 3), device=gs.device, dtype=gs.tc_float) * f_max
        adv_f.clamp_min_(0.1)                                                                               # avoid huge amplitudes
        adv_a = adv_v / (math.tau*adv_f*math.sqrt(3))
        adv_a[:, 2] = torch.minimum(adv_a[:, 2], (self.commands[envs_idx, 2] - self.env_cfg["z_margin"]).clamp_min(0.0))     # avoid dipping below ground
        adv_phi = torch.rand_like(self.adv_phi[envs_idx]) * math.tau

        self.adv_a[envs_idx] = adv_a
        self.adv_f[envs_idx] = adv_f
        self.adv_phi[envs_idx] = adv_phi

        t = self.episode_length_buf[envs_idx].unsqueeze(-1).to(self.adv_f.dtype) * self.dt
        self.adv_sinus[envs_idx] = self.adv_a[envs_idx] * torch.sin(math.tau*self.adv_f[envs_idx]*t + self.adv_phi[envs_idx])

        self.commands_adv[envs_idx] = self.commands[envs_idx] + self.adv_sinus[envs_idx]
        self.last_commands_adv[envs_idx] = self.commands_adv[envs_idx]

    def _resample_commands(self, envs_idx):
        spawn_clearance = 0.2 #TODO param
        todo = torch.ones(len(envs_idx), dtype=torch.bool, device=gs.device)

        MAX_TRIES = 4
        for _ in range(MAX_TRIES):
            todo_idx = envs_idx[todo]
            if todo_idx.numel() == 0:
                break

            self.commands[todo_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(todo_idx),), gs.device)
            self.commands[todo_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(todo_idx),), gs.device)
            self.commands[todo_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(todo_idx),), gs.device)

            self._resample_adv(todo_idx)

            self.rel_pos[todo_idx] = self.commands_adv[todo_idx] - self.base_pos[todo_idx]

            clearance_mask = self.rel_pos[todo_idx, :2].norm(dim=1) >= spawn_clearance
            temp_todo = todo.clone() # for pytorch reasons...
            todo[temp_todo] = ~clearance_mask

        self.last_rel_pos[envs_idx] = self.rel_pos[envs_idx]
        self.rel_vel[envs_idx] = 0.0

        self.tgt_vel[envs_idx] = 0.0
        self.tgt_vel_est[envs_idx] = 0.0
        self.tgt_acc_est[envs_idx] = 0.0

        self.update_approach_geometry(envs_idx)
        dist, _, vel_close, _ = self.app_geom
        self.prev_dist[envs_idx] = dist[envs_idx]
        self.prev_vel_close[envs_idx] = vel_close[envs_idx]
        self.prev_v_tan[envs_idx] = 0.0

        self.stable_cnt[envs_idx] = 0
        self.success[envs_idx] = False

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions

        # 14468 is hover rpm
        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699) #yes, that's the correct API function name
        # update target pos
        self.adv_sinus = self.adv_a * torch.sin(math.tau*self.adv_f*self.episode_length_buf.unsqueeze(-1)*self.dt + self.adv_phi)
        self.last_commands_adv = self.commands_adv
        self.commands_adv = self.commands + self.adv_sinus

        meas_tgt_vel = (self.commands_adv - self.last_commands_adv) / self.dt
        self.tgt_vel_est = 0.8 * self.tgt_vel_est + 0.2 * meas_tgt_vel
        meas_tgt_acc = (self.tgt_vel_est - self.tgt_vel) / self.dt
        self.tgt_acc_est = 0.9 * self.tgt_acc_est + 0.1 * meas_tgt_acc
        self.tgt_vel = self.tgt_vel_est

        if self.adversary is not None:
            self.adversary.set_pos(self.commands_adv + self.adv_base_offset, zero_velocity=True)
            self.adversary_dog.set_pos(self.commands_adv + self.adv_dog_offset, zero_velocity=True)
            self.adversary_dog.set_dofs_position(
                position=self.default_dof_pos_local.expand(self.num_envs, -1),
                dofs_idx_local=self.dog_all_local_idx,
                zero_velocity=True,
            )
            self.adversary_pad.set_pos(self.commands_adv + self.adv_pad_offset, zero_velocity=True)
            adv_contact = self.adversary.get_contacts(with_entity=self.drone, exclude_self_contact=True)
            self.adv_collision = (adv_contact['penetration'] > 0).any(dim=1)

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos.copy_(self.base_pos)

        # sanitize raw sim outputs
        pos  = self.drone.get_pos()
        quat = self.drone.get_quat()
        lin  = self.drone.get_vel()
        ang  = self.drone.get_ang()
        pos.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)
        quat.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)
        lin.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)
        ang.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)

        quat = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-6)
        inv_base_quat = inv_quat(quat)

        self.base_pos.copy_(pos)
        self.base_quat.copy_(quat)
        self.base_lin_vel.copy_(transform_by_quat(lin, inv_base_quat))
        self.base_ang_vel.copy_(transform_by_quat(ang, inv_base_quat))

        # relatives
        self.last_rel_pos.copy_(self.rel_pos)
        self.rel_pos.copy_(self.commands_adv - self.base_pos)
        self.rel_vel.copy_((self.rel_pos - self.last_rel_pos) / self.dt)

        # approach
        d_now, _, vc_now, _ = self.app_geom
        self.prev_dist.copy_(d_now)
        self.prev_vel_close.copy_(vc_now)
        self.update_approach_geometry()

        # calculate metrics
        d,u,vel_close,_ = self.app_geom
        v_des = torch.clamp(self.env_cfg["approach_k"] * d, max=self.env_cfg["approach_v_cap"])
        sigma = max(self.env_cfg["near_gate_factor"] * self.env_cfg["at_target_threshold"], self.env_cfg["at_target_threshold"])
        v_tan = self.rel_vel - (self.rel_vel*u).sum(dim=1, keepdim=True)*u
        w_norm = self.base_ang_vel.norm(dim=1)  # rad/s

        self.m_d_sum += d
        self.m_step  += 1
        self.m_inside_sum += (d < self.env_cfg["at_target_threshold"]).float()
        self.m_vtgt_sum += self.tgt_vel.norm(dim=1)

        mask = d < sigma
        self.m_near_cnt += mask.float()
        self.m_vapp_err_sum += torch.where(mask, (v_des - vel_close).abs(), 0.0)
        self.m_vtan_sum += torch.where(mask, v_tan.norm(dim=1), 0.0)
        self.m_angvel_sum += torch.where(mask, w_norm, 0.0)

        # calculate tilt using gravity vector
        self.base_tilt_cos = transform_by_quat(self.world_z, inv_base_quat)[:, 2].clamp(-1.0, 1.0)

        success_mask = self._success_mask()

        # target visualisation
        if self.target is not None:
            near = (self.rel_pos.norm(dim=1) < self.env_cfg["at_target_threshold"])
            # per-env positions: show at target when near, park far below otherwise
            threshold_pos = torch.where(near.unsqueeze(1), self.commands_adv, self.highlight_hide)
            reached_pos = torch.where(success_mask.unsqueeze(1), self.commands_adv, self.highlight_hide)
            self.target.set_pos(self.commands_adv, zero_velocity=True)
            self.target_threshold_highlight.set_pos(threshold_pos, zero_velocity=True)
            self.target_reached_highlight.set_pos(reached_pos, zero_velocity=True)

        # check termination
        below_plane = self.base_pos[:, 2] < (self.commands_adv[:, 2] - self.env_cfg["z_margin"])
        adv_hard_hit = self.adv_collision & (~success_mask) & below_plane
        tilt_term = (self.base_tilt_cos < self.term_tilt_cos)
        x_term = (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
        y_term = (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
        z_term = (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
        angvel_term = (torch.abs(self.base_ang_vel.norm(dim=1)) > self.env_cfg["termination_if_angvel_greater_than"])
        floor_term = (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])

        self.extras["term_causes_now"] = {
            "term_tilt": tilt_term.float(),
            "term_x": x_term.float(),
            "term_y": y_term.float(),
            "term_z": z_term.float(),
            "term_yaw": angvel_term.float(),
            "term_floor": floor_term.float(),
            "term_adv_hit": adv_hard_hit.float()
        }
        self.crash_condition = (tilt_term | x_term | y_term | z_term | angvel_term | floor_term | adv_hard_hit)

        # check success
        self.stable_cnt = torch.where(success_mask, (self.stable_cnt + 1).clamp_max(self.n_stable), torch.zeros_like(self.stable_cnt))
        self.success = self.stable_cnt >= self.n_stable

        # update curriculum
        self.difficulty[self.crash_condition] -= self.env_cfg["adv_difficulty_delta_fail"]
        self.difficulty[self.success] += self.env_cfg["adv_difficulty_delta_success"]
        self.difficulty.clamp_(self.env_cfg["adv_difficulty_min"], self.env_cfg["adv_difficulty_max"])

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            rew = torch.nan_to_num(rew, 0.0, 0.0, 0.0).clamp_(-1000.0, 1000.0)
            assert rew.shape == (self.num_envs,), f"{name} returned {rew.shape}"
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # reset
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        reset_idx_now = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.extras["term_timeout"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["term_timeout"][ (self.episode_length_buf > self.max_episode_length) ] = 1.0
        self.extras["term_reset_idx_now"] = reset_idx_now  # for logging in reset_idx

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # resample successful envs
        envs_idx = torch.nonzero(self.success, as_tuple=False).flatten()
        self._resample_commands(envs_idx)

        # compute observations
        obs_t = self.build_obs()
        obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=1e6, neginf=-1e6)
        obs_t = torch.clamp(obs_t, -100.0, 100.0)

        # push obs to stacker, clear stack for reset and resampled envs
        done = self.reset_buf.bool()
        done[envs_idx] = True
        self.stacker.push(obs_t, done)        
        self.obs_buf = self.stacker.stacked()

        self.last_actions[:] = self.actions[:]
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def update_approach_geometry(self, envs_idx=None):
        if envs_idx is None:
            self.app_geom = self.calculate_approach_geometry()
            return

        d, u, vel_close, t_go = self.calculate_approach_geometry(envs_idx)
        d_old, u_old, vel_close_old, t_go_old = self.app_geom

        d_old[envs_idx] = d
        u_old[envs_idx] = u
        vel_close_old[envs_idx] = vel_close
        t_go_old[envs_idx] = t_go
        self.app_geom = d_old, u_old, vel_close_old, t_go_old

    def calculate_approach_geometry(self, envs_idx=None):
        if envs_idx is None:
            rel_pos = self.rel_pos
            rel_vel = self.rel_vel
        else:
            rel_pos = self.rel_pos[envs_idx]
            rel_vel = self.rel_vel[envs_idx]

        d = torch.norm(rel_pos, dim=1)                                                  # range to target
        u = rel_pos / d.clamp_min(1e-6).unsqueeze(1)                                    # unit vector from drone to target
        vel_close = - (rel_vel * u).sum(dim=1)                                          # closing speed along the line of sight
        t_go = torch.clamp(d / (vel_close.abs() + 1e-3), 0.0, self.env_cfg["tgo_cap"])  # time-to-go estimate
        return d, u, vel_close, t_go

    def build_obs(self):
        retarget_flag = torch.zeros((self.num_envs, 1), device=gs.device, dtype=gs.tc_float) # will be set to 1.0 by the obs stacker when necessary

        dist, target_vector, vel_close, t_go = self.app_geom

        #TODO einheitliches calling, obs scales in train ergänzen
        return torch.cat([
            torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
            torch.clip(self.rel_vel * self.obs_scales["lin_vel"], -1, 1),
            torch.clip(self.tgt_vel * self.obs_scales.get("tgt_vel", 1/3.0), -1, 1),
            torch.clip(self.tgt_acc_est * self.obs_scales.get("tgt_acc", 1/10.0), -1, 1),
            (torch.clip(dist * self.obs_scales.get("dist", 1.0), 0.0, 2.0)).unsqueeze(-1),
            (torch.clip(vel_close * self.obs_scales.get("vel_close", 1.0), -2.0, 2.0)).unsqueeze(-1),
            (t_go * self.obs_scales.get("t_go", 1.0)).unsqueeze(-1),
            target_vector,
            self.base_quat,
            torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
            torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
            self.last_actions,
            retarget_flag,
        ], dim=-1)  # shape: [N, D]

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dog dofs (correct order) and zero velocity
        self.dog_dof_pos[envs_idx] = self.default_dof_pos_local
        self.adversary_dog.set_dofs_position(
            position=self.dog_dof_pos[envs_idx],
            dofs_idx_local=self.dog_all_local_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        self.episode_length_buf[envs_idx] = 0

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos

        self._resample_commands(envs_idx)

        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)
        self.prev_ang_norm[envs_idx] = 0
        self.prev_yaw_abs[envs_idx] = 0

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        eps = 1e-6
        inv_step = 1.0 / (self.m_step[envs_idx] + eps)
        inv_near = 1.0 / (self.m_near_cnt[envs_idx] + eps)

        rew_keys = list(self.episode_sums.keys())
        if rew_keys:
            rew_vals = torch.stack(
                [self.episode_sums[k][envs_idx].mean() for k in rew_keys],
                dim=0
            ) / self.env_cfg["episode_length_s"]
            rew_cpu = rew_vals.detach().cpu().tolist()
        else:
            rew_cpu = []

        d_mean_t = (self.m_d_sum[envs_idx]          * inv_step).mean()
        inside_t = (self.m_inside_sum[envs_idx]     * inv_step).mean()
        vapp_t   = (self.m_vapp_err_sum[envs_idx]   * inv_near).mean()
        vtan_t   = (self.m_vtan_sum[envs_idx]       * inv_near).mean()
        vtgt_t   = (self.m_vtgt_sum[envs_idx]       * inv_step).mean()
        angvel_t = (self.m_angvel_sum[envs_idx]     * inv_near).mean()
        diff_t   = self.difficulty.mean()
        stats = torch.stack([d_mean_t, inside_t, vapp_t, vtan_t, vtgt_t, angvel_t, diff_t])
        stats_cpu = stats.detach().cpu().tolist()

        ep = self.extras["episode"]
        dm, ins, vapp, vtan, vtgt, angvel, diff = stats_cpu
        ep["metric_d_mean"]          = dm
        ep["metric_inside_succ_pct"] = ins
        ep["metric_v_app_err_near"]  = vapp
        ep["metric_v_tan_near"]      = vtan
        ep["metric_v_tgt_mean"]      = vtgt
        ep["metric_angvel_near"]     = angvel
        ep["metric_difficulty"]      = diff
        for k, v in zip(rew_keys, rew_cpu):
            ep[f"rew_{k}"] = v

        # Termination histogram for just-reset envs
        ri = self.extras.get("term_reset_idx_now", None)
        if ri is not None and len(ri) > 0:
            for k, v in self.extras.get("term_causes_now", {}).items():
                ep[k] = v[ri].mean().item()
            ep["term_timeout"] = self.extras["term_timeout"][ri].mean().item()


        # clear for next episodes
        for t in [self.m_d_sum, self.m_step, self.m_inside_sum, self.m_vapp_err_sum, self.m_near_cnt, self.m_vtan_sum, self.m_vtgt_sum, self.m_angvel_sum]:
            t[envs_idx] = 0
        for k in rew_keys:
            self.episode_sums[k][envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        obs_t = self.build_obs()
        self.stacker.push(obs_t, torch.ones(self.num_envs, dtype=torch.bool, device=gs.device))
        self.obs_buf = self.stacker.stacked()
        return self.obs_buf, None

    def _success_mask(self):
        near  = self.rel_pos.norm(dim=1) < self.env_cfg["at_target_threshold"]
        slow  = self.rel_vel.norm(dim=1) < self.env_cfg["max_rel_speed_mps"]
        level = self.base_tilt_cos > self.max_tilt_cos
        angvel  = self.base_ang_vel.norm(dim=1) < self.env_cfg["max_angvel_radps"]
        return near & slow & level & angvel

    def _gaussian_gate(self, dist):
        decay_distance = self.env_cfg["near_gate_factor"] * self.env_cfg["at_target_threshold"]
        decay_distance = max(float(decay_distance), 1e-6)
        return torch.exp(- (dist / decay_distance) ** 2)

    # ------------ reward functions----------------

    def _reward_approach(self):
        # unified distance + radial speed shaping
        # distance difference to target
        # vel difference to closing velocity
        # use potential to reward continuous motion toward goal instead of 'parking'
        dist, _, vel_close, _ = self.app_geom
        dist_prev, vel_close_prev = self.prev_dist, self.prev_vel_close

        k, v_cap = self.env_cfg["approach_k"], self.env_cfg["approach_v_cap"]
        dist_soft_margin = self.env_cfg["near_gate_factor"] * self.env_cfg["at_target_threshold"]
        vel_close_weight = 1.0 #TODO param
        vel_soft_margin = 0.05 #TODO param

        def phi(dist, vel):
            dist_target = torch.zeros_like(dist)
            vel_target = torch.clamp(k * dist, max=v_cap)
            dist_rew =  functional.smooth_l1_loss(dist, dist_target, beta=dist_soft_margin, reduction="none")
            vel_rew = functional.smooth_l1_loss(vel, vel_target, beta=vel_soft_margin, reduction="none")
            return dist_rew + vel_close_weight * vel_rew

        return phi(dist_prev, vel_close_prev) - phi(dist, vel_close)

    def _reward_tan_vel_align(self):
        # penalize tangential velocity near goal
        dist, u, _, _ = self.app_geom
        gate = self._gaussian_gate(dist)
        gate_prev = self._gaussian_gate(self.prev_dist)

        v_tan = (self.rel_vel - (self.rel_vel * u).sum(dim=1, keepdim=True) * u).norm(dim=1)
        v_tan_rew = gate_prev * self.prev_v_tan - gate * v_tan
        self.prev_v_tan.copy_(v_tan)

        return v_tan_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return -smooth_rew

    def _reward_ang_vel(self):
        dist, _, _, _ = self.app_geom
        gate = self._gaussian_gate(dist) # TODO calculate gate once instead of everywhere
        gate_prev = self._gaussian_gate(self.prev_dist)

        ang_norm = self.base_ang_vel.norm(dim=1)
        yaw_abs  = self.base_ang_vel[:, 2].abs()

        angvel_near_rew = gate_prev * self.prev_ang_norm - gate * ang_norm
        yaw_rew  = (self.prev_yaw_abs - yaw_abs)
        self.prev_ang_norm.copy_(ang_norm)
        self.prev_yaw_abs.copy_(yaw_abs)

        return angvel_near_rew + yaw_rew

    def _reward_crash(self):
        crash  = self.crash_condition.float()
        impact = self.rel_vel.norm(dim=1) + 0.5 * self.base_ang_vel.norm(dim=1)
        return -(crash * (1.0 + impact))

    def _reward_success(self):
        return self.success.to(self.rew_buf.dtype)