# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch

import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

class HumanoidMimic(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # self._tar_motion = cfg["env"]["tar_motion"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
                
        # if (not self.headless):
        #     self._build_marker_state_tensors()
        # breakpoint()

        self._tar_root_pos = torch.zeros_like(self._humanoid_root_states[:,0:3])
        self._tar_root_rot = torch.zeros_like(self._humanoid_root_states[:,3:7])
        self._tar_pos = torch.zeros_like(self._dof_pos)
        self._tar_vel = torch.zeros_like(self._dof_vel)
        self._tar_key_pos = torch.zeros_like(self._rigid_body_pos[:, self._key_body_ids, :])

        self.sum_reward = 0.0
        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 31+18+3
        return obs_size

    def _update_marker(self):
        self._marker_pos[..., :] = self._tar_pos
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        # if (not self.headless):
        #     self._marker_handles = []
        #     self._load_target_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    # def _load_marker_asset(self):
    #     asset_root = "ase/data/assets/mjcf/"
    #     asset_file = "location_marker.urdf"

    #     asset_options = gymapi.AssetOptions()
    #     asset_options.angular_damping = 0.01
    #     asset_options.linear_damping = 0.01
    #     asset_options.max_angular_velocity = 100.0
    #     asset_options.density = 1.0
    #     asset_options.fix_base_link = True
    #     asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

    #     self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    #     return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        # if (not self.headless):
        #     self._build_marker(env_id, env_ptr)

        return


    # def _build_marker(self, env_id, env_ptr):
    #     col_group = env_id
    #     col_filter = 2
    #     segmentation_id = 0

    #     default_pose = gymapi.Transform()
        
    #     marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
    #     self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
    #     self._marker_handles.append(marker_handle)

    #     return

    # def _build_marker_state_tensors(self):
    #     num_actors = self._root_states.shape[0] // self.num_envs
    #     self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
    #     self._marker_pos = self._marker_states[..., :3]
        
    #     self._marker_actor_ids = self._humanoid_actor_ids + 1

    #     return
    
    # def _build_reach_body_id_tensor(self, env_ptr, actor_handle, body_name):
    #     body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
    #     assert(body_id != -1)
    #     body_id = to_torch(body_id, device=self.device, dtype=torch.long)
    #     return body_id

    def _compute_reset(self):
        # max_frame = self._motion_lib.get_motion_length(0) self._reseted_ref_motion_times
        ref_motion_max_times = self._motion_lib.get_motion_length(0)-self._reseted_ref_motion_times

        root_rot = self._humanoid_root_states[:, 3:7]
        root_pos = self._humanoid_root_states[:, 0:3]

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_mimic_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights,
                                                   ref_motion_max_times, self.dt,
                                                   root_pos, root_rot, self._tar_root_pos, self._tar_root_rot)

        if self.num_envs == 2:
            self.reset_buf[1] = self.reset_buf[0]
            self._terminate_buf[1] = self._terminate_buf[0]

        if self.reset_buf[0] == 1:
            self.sum_reward = 0.0

        return

    def _update_task(self):
        # breakpoint()
        # reset_task_mask = self.progress_buf >= self._tar_change_steps
        # rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        # if len(rest_env_ids) > 0:
        #     self._reset_task(rest_env_ids)
        progress_time = (self.progress_buf+1) * self.dt + self._reseted_ref_motion_times
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state([0], progress_time)

        self._tar_root_pos = root_pos[:, 0:3].clone()
        self._tar_root_rot = root_rot.clone()
        self._tar_pos = dof_pos.clone()
        self._tar_vel = dof_vel.clone()
        self._tar_key_pos = key_pos.clone()

        return

    def _reset_task(self, env_ids):
        # n = len(env_ids)

        # rand_pos = torch.rand([n, 3], device=self.device)
        # rand_pos[..., 0:2] = self._tar_dist_max * (2.0 * rand_pos[..., 0:2] - 1.0)
        # rand_pos[..., 2] = (self._tar_height_max - self._tar_height_min) * rand_pos[..., 2] + self._tar_height_min
        
        # change_steps = torch.randint(low=self._tar_change_steps_min, high=self._tar_change_steps_max,
        #                              size=(n,), device=self.device, dtype=torch.int64)

        # self._tar_pos[env_ids, :] = rand_pos
        # self._tar_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state([0],  self.dt+self._reseted_ref_motion_times[env_ids])

        self._tar_root_pos[env_ids] = root_pos[:, 0:3].clone()
        self._tar_root_rot[env_ids] = root_rot.clone()
        self._tar_pos[env_ids] = dof_pos.clone()
        self._tar_vel[env_ids] = dof_vel.clone()
        self._tar_key_pos[env_ids] = key_pos.clone()
        # breakpoint()
        return

    def _compute_task_obs(self, env_ids=None):
        progress_time = (self.progress_buf+1) * self.dt + self._reseted_ref_motion_times
        _tar_root_pos, _tar_root_rot, _tar_pos, _tar_root_vel, _tar_root_ang_vel, _tar_dof_vel, _tar_key_pos = self._motion_lib.get_motion_state([0], progress_time)

        key_pos = self._rigid_body_pos[:, self._key_body_ids, :] - self._humanoid_root_states[:, 0:3].unsqueeze(-2)
        tar_key_pos = _tar_key_pos - _tar_root_pos.unsqueeze(-2)

        root_rot = self._humanoid_root_states[:, 3:7]
        tar_rot =_tar_root_rot

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, key_pos.shape[1], 1))
        flat_end_pos = key_pos.view(key_pos.shape[0] * key_pos.shape[1], key_pos.shape[2])
        flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                   heading_rot_expand.shape[2])
        local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
        flat_local_key_pos = local_end_pos.view(key_pos.shape[0], key_pos.shape[1] * key_pos.shape[2])


        heading_rot = torch_utils.calc_heading_quat_inv(tar_rot)

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, tar_key_pos.shape[1], 1))
        flat_end_pos = tar_key_pos.view(tar_key_pos.shape[0] * tar_key_pos.shape[1], tar_key_pos.shape[2])
        flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                   heading_rot_expand.shape[2])
        local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
        flat_local_tar_key_pos = local_end_pos.view(tar_key_pos.shape[0], tar_key_pos.shape[1] * tar_key_pos.shape[2])

        # print(flat_local_key_pos)
        # print(flat_local_tar_key_pos)
        # print("=======================")

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        tar_root_position = quat_rotate(heading_rot, _tar_root_pos - self._humanoid_root_states[:, 0:3])
        # print(self._tar_root_pos[0])
        # print(self._humanoid_root_states[:, 0:3][0])
        # print(tar_root_position[0])
        # print("------------------------")

        # print("task obs")
        # print(_tar_root_pos[0])


        if (env_ids is None):
            return torch.cat((tar_root_position, _tar_pos - self._dof_pos, flat_local_tar_key_pos - flat_local_key_pos), dim=-1)
        else:
            return torch.cat((tar_root_position[env_ids], _tar_pos[env_ids] - self._dof_pos[env_ids], flat_local_tar_key_pos[env_ids] - flat_local_key_pos[env_ids]), dim=-1)

        # # obs = compute_location_observations(root_states, tar_pos)

        # return self._tar_pos

    def _compute_reward(self, actions):
        # reach_body_pos = self._rigid_body_pos[:, self._reach_body_id, :]
        # root_rot = self._humanoid_root_states[..., 3:7]
        dof_pos = self._dof_pos.clone()
        dof_vel = self._dof_vel.clone()
        root_rot = self._humanoid_root_states[:, 3:7]
        root_pos = self._humanoid_root_states[:, 0:3]
        # key_pos = self._rigid_body_pos[:, self._key_body_ids, :].clone()


        # print("reward")
        # print(self._tar_root_pos[0])
        local_tar_key_pos = self._tar_key_pos - self._tar_root_pos.unsqueeze(-2)
        local_cur_key_pos = self._rigid_body_pos[:, self._key_body_ids, :] - self._humanoid_root_states[:, 0:3].unsqueeze(-2)

        self.rew_buf[:] = compute_mimic_reward(root_rot, dof_pos, dof_vel, local_cur_key_pos, root_pos,
            self._tar_root_rot, self._tar_pos, self._tar_vel, local_tar_key_pos, self._tar_root_pos
            )

        self.sum_reward += self.rew_buf[0]
        # print(self.sum_reward)
        # print(self.progress_buf[0])
        return

    def _draw_task(self):
        # self._update_marker()
        
        # cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        # self.gym.clear_lines(self.viewer)

        # starts = self._rigid_body_pos[:, self._reach_body_id, :]
        # ends = self._tar_pos

        # verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        # for i, env_ptr in enumerate(self.envs):
        #     curr_verts = verts[i]
        #     curr_verts = curr_verts.reshape([1, 6])
        #     self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
# def compute_location_observations(root_states, tar_pos):
#     # type: (Tensor, Tensor) -> Tensor
#     root_rot = root_states[:, 3:7]
#     heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
#     local_tar_pos = quat_rotate(heading_rot, tar_pos)

#     obs = local_tar_pos
#     return obs


    def post_physics_step(self):
        super().post_physics_step()
        self._motion_sync()
        return

    def _motion_sync(self):
        if self.num_envs != 2:
            return
        else:
            num_motions = self._motion_lib.num_motions()
            motion_times = self._reseted_ref_motion_times + self.progress_buf * self.dt

            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state([0], motion_times)
            
            root_vel = torch.zeros_like(root_vel)
            root_ang_vel = torch.zeros_like(root_ang_vel)
            dof_vel = torch.zeros_like(dof_vel)

            # env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            # if self.num_envs==2:
            env_ids = torch.zeros(1, dtype=torch.long, device=self.device)
            env_ids[0] = 1
            self._set_env_state(env_ids=env_ids, 
                                root_pos=root_pos[env_ids], 
                                root_rot=root_rot[env_ids], 
                                dof_pos=dof_pos[env_ids], 
                                root_vel=root_vel[env_ids], 
                                root_ang_vel=root_ang_vel[env_ids], 
                                dof_vel=dof_vel[env_ids])

            env_ids_int32 = self._humanoid_actor_ids[env_ids]
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self._root_states),
                                                         gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self._dof_state),
                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

        # self.rew_buf[:] = compute_mimic_reward(root_rot, dof_pos, dof_vel, local_cur_key_pos, root_pos,
        #     self._tar_root_rot, self._tar_pos, self._tar_vel, local_tar_key_pos, self._tar_root_pos
        #     )

@torch.jit.script
def compute_mimic_reward(root_rot, pos, vel, key_pos, root_pos, tar_rot, tar_pos, tar_vel, tar_key_pos, tar_root_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    # breakpoint()

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, key_pos.shape[1], 1))
    flat_end_pos = key_pos.view(key_pos.shape[0] * key_pos.shape[1], key_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(key_pos.shape[0], key_pos.shape[1] * key_pos.shape[2])


    heading_rot = torch_utils.calc_heading_quat_inv(tar_rot)

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, tar_key_pos.shape[1], 1))
    flat_end_pos = tar_key_pos.view(tar_key_pos.shape[0] * tar_key_pos.shape[1], tar_key_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_tar_key_pos = local_end_pos.view(tar_key_pos.shape[0], tar_key_pos.shape[1] * tar_key_pos.shape[2])


    # heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    # tar_root_position = quat_rotate(heading_rot, tar_root_pos - root_pos)


    pos_err_scale = 2.0
    vel_err_scale = 0.01
    
    pos_diff = pos - tar_pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)
    # print(pos_reward[0])
    
    root_rot_diff = torch_utils.quat_to_exp_map(tar_rot) - torch_utils.quat_to_exp_map(root_rot)
    # print(root_rot_diff[0])
    # print(pos[0])
    # print(tar_pos[0][:3])
    # breakpoint()
    root_rot_err = torch.sum(root_rot_diff * root_rot_diff, dim=-1)
    # print(root_rot_err[0])
    root_rot_reward = torch.exp(-10.0 * root_rot_err)

    # print(root_rot_reward[0])
    # print(pos_reward)
    # print("==------")
    
    # print(vel[0])
    # print(tar_vel[0])
    vel_diff = vel - tar_vel
    vel_err = torch.sum(vel_diff*vel_diff, dim=-1)
    vel_reward = 1.0*torch.exp(-vel_err_scale * vel_err)
    # print(vel_err[0])
    # print(vel_reward[0])
    # print("---------------")

    key_diff = flat_local_key_pos - flat_local_tar_key_pos
    key_err = torch.sum(key_diff * key_diff, dim=-1)
    key_reward = torch.exp(-20.0 * key_err)

    root_diff = root_pos - tar_root_pos
    root_err = torch.sum(root_diff * root_diff, dim=-1)
    root_reward = torch.exp(-10.0 * root_err)


    # print(pos_reward[0])
    # print(vel_reward[1])
    # print(key_reward[0])
    # print(root_reward[0])
    reward = 4.0*pos_reward*key_reward*root_reward*root_rot_reward+1.0*vel_reward
    # print(reward[0])
    # print("--------------")

    return reward

@torch.jit.script
def compute_humanoid_mimic_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights,
                           ref_motion_max_times, dt,
                           root_pos, root_rot, tar_root_pos, tar_root_rot):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)


        # root_dist = tar_root_pos - root_pos

        # heading_quat_inv = torch_utils.calc_heading_quat_inv(root_rot)
        # tar_heading_quat = torch_utils.calc_heading_quat(tar_root_rot)

        # rel_tar_quat = quat_mul(tar_heading_quat, heading_quat_inv)

        # rel_tar_ang = torch_utils.calc_heading(rel_tar_quat)

        # far_target_root = torch.logical_or(torch.sqrt(torch.sum(root_dist*root_dist, dim = -1)) > 0.5,torch.abs(rel_tar_ang) > 3.14/4)

        # has_fallen = torch.logical_or(has_fallen, far_target_root)

        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    # reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    # print("reseted?")
    # print(progress_buf*dt >= ref_motion_max_times - 0.1)
    reset = torch.where(progress_buf*dt >= ref_motion_max_times - 0.1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

