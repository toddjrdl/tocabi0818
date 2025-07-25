# dyros_reward_v2.py
"""
Reward implementation aligned with the DWL paper (Table V).
* Four blocks : velocity-tracking, periodic reward, foot-traj tracking, regularizers
* φ(e, w)    : exp(−w ‖e‖²)            (paper Eq.)
NOTE
-----
‣ Foot position / velocity are not available in current argument list.
  Place-holders (zeros) are used – replace with real link states if needed.
‣ Torch-scripted (for rl_games compatibility).
"""

from typing import List, Tuple
import torch
from isaacgymenvs.utils.torch_jit_utils import quat2euler, quat_diff_rad

# -------------------------------------------------------------------- #
# helper
# -------------------------------------------------------------------- #
@torch.jit.script
def _phi(err: torch.Tensor, w: float):
    # type: (Tensor, float) -> torch.Tensor
    return torch.exp(-w * torch.sum(err * err, dim=1))

@torch.jit.script
def _poly_height(t: torch.Tensor):  # type: ignore
    """5‑th order polynomial height profile (unnormalised)."""
    a0, a1, a2, a3, a4, a5 = 0.0, 0.1, 2.5, -4.7, 1.5, 0.6
    return a0 + a1 * t + a2 * t * t + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5

@torch.jit.script
def _poly_vel(t: torch.Tensor):  # type: ignore
    """Time derivative of _poly_height."""
    a1, a2, a3, a4, a5 = 0.1, 2.5, -4.7, 1.5, 0.6
    return a1 + 2.0 * a2 * t + 3.0 * a3 * t ** 2 + 4.0 * a4 * t ** 3 + 5.0 * a5 * t ** 4

# -------------------------------------------------------------------- #
# main reward (TorchScript)
# -------------------------------------------------------------------- #
@torch.jit.script
def compute_humanoid_walk_reward_v2(
        reset_buf: torch.Tensor,
        progress_buf: torch.Tensor,
        target_vel: torch.Tensor,
        root_states: torch.Tensor,
        joint_position_target: torch.Tensor,
        force_target: torch.Tensor,
        joint_position_states: torch.Tensor,
        joint_velocity_init: torch.Tensor,
        joint_velocity_states: torch.Tensor,
        pre_joint_velocity_states: torch.Tensor,
        actions: torch.Tensor,
        actions_pre: torch.Tensor,
        non_feet_idxs: List[int],
        contact_forces: torch.Tensor,
        contact_forces_pre: torch.Tensor,
        mocap_data_idx: torch.Tensor,
        termination_height: float,
        death_cost: float,
        policy_freq_scale: float,
        total_mass: torch.Tensor,
        contact_reward_sum: torch.Tensor,
        right_foot_idx: int,
        left_foot_idx: int,
        l_foot_pos: torch.Tensor, r_foot_pos: torch.Tensor,
        l_foot_vel: torch.Tensor, r_foot_vel: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor]:

    n_env = root_states.shape[0]
    ones  = torch.ones((n_env,), device=root_states.device)
    zeros = torch.zeros((n_env,), device=root_states.device)

    # ------------------------------------------------------------ #
    # 1) velocity-tracking (linear + angular) & orientation / height
    # ------------------------------------------------------------ #
    lin_vel = root_states[:, 7:10]           # Pẋ,ẏ,ż
    ang_vel = root_states[:, 10:13]
    cmd_lin = torch.cat((target_vel, zeros.unsqueeze(-1)), dim=1)    # (x,y,0)
    cmd_ang = torch.zeros_like(ang_vel)

    r_lin_vel = _phi(lin_vel - cmd_lin, 5.0)
    r_ang_vel = _phi(ang_vel - cmd_ang, 7.0)

    roll, pitch, _ = quat2euler(root_states[:, 3:7])
    r_orient  = _phi(torch.stack((roll, pitch), dim=1), 5.0)
    r_height  = _phi((root_states[:, 2] - 0.926).unsqueeze(-1), 10.0)

    # ------------------------------------------------------------ #
    # 2) periodic reward  (stance mask by contact force threshold)
    # ------------------------------------------------------------ #
    l_force = contact_forces[:, left_foot_idx, 2]     # Z-axis
    r_force = contact_forces[:, right_foot_idx, 2]

    l_contact = (l_force > 30.0).float()
    r_contact = (r_force > 30.0).float()

    # scaled & clipped
    scale_F = 0.5 * total_mass.squeeze(-1) * 9.81  # per env
    l_scaled = torch.clamp(l_force / scale_F, 0.0, 1.0)
    r_scaled = torch.clamp(r_force / scale_F, 0.0, 1.0)

    r_periodic_F = l_contact * l_scaled + r_contact * r_scaled

    # foot swing-phase velocity reward
    swing_L = 1.0 - l_contact                # 접촉 안 된 발이 swing
    swing_R = 1.0 - r_contact
    swing_vel_z = swing_L * l_foot_vel[:, 2] + swing_R * r_foot_vel[:, 2]

    r_periodic_V = swing_vel_z
    # ------------------------------------------------------------ #
    # 3) Quintic foot-trajectory tracking
    # ------------------------------------------------------------ #
    # Estimate time parameter t [0,1] for swing leg: use vertical ratio heuristic
    t_L = torch.clamp(l_foot_pos[:, 2] / 0.20, 0.0, 1.0) * swing_L
    t_R = torch.clamp(r_foot_pos[:, 2] / 0.20, 0.0, 1.0) * swing_R

    h_ref_L = _poly_height(t_L) * swing_L
    h_ref_R = _poly_height(t_R) * swing_R
    h_ref = h_ref_L + h_ref_R

    v_ref_L = _poly_vel(t_L) * swing_L
    v_ref_R = _poly_vel(t_R) * swing_R
    v_ref = v_ref_L + v_ref_R  # (B,)

    foot_height_z = swing_L * l_foot_pos[:, 2] + swing_R * r_foot_pos[:, 2]
    foot_vel_z    = swing_L * l_foot_vel[:, 2] + swing_R * r_foot_vel[:, 2]

    r_foot_height = _phi((foot_height_z - h_ref).unsqueeze(-1), 5.0)
    r_foot_vel    = _phi((foot_vel_z - v_ref).unsqueeze(-1), 3.0)

    # ------------------------------------------------------------ #
    # 4) regularizers
    # ------------------------------------------------------------ #
    # default joint pose
    r_joint_def = _phi(joint_position_states, 2.0)  # DWL: θ₀ = 0 rad

    # energy ≈ |a| surrogate
    tau   = actions
    omega = joint_velocity_states[:, :actions.size(1)]
    r_energy = torch.sum(torch.abs(tau * omega), dim=1)

    # action smoothness 
    a_diff = actions - actions_pre   # reuse buffer as aₜ₋₂
    r_smooth = torch.sum(a_diff * a_diff, dim=1)

    # large contact clip penalty
    r_big_clip = torch.clamp(l_scaled + r_scaled - 1.0, 0.0)

    # ------------------------------------------------------------ #
    # stack & weight (Table V)
    # ------------------------------------------------------------ #
    rewards = torch.stack((
        r_lin_vel,             # 0
        r_ang_vel,             # 1
        r_orient,              # 2
        r_height,              # 3
        r_periodic_F,          # 4
        r_periodic_V,          # 5
        r_foot_height,         # 6
        r_foot_vel,            # 7
        r_joint_def,           # 8
        r_energy,              # 9
        r_smooth,              #10
        r_big_clip             #11
    ), dim=1)

    μ = torch.tensor([
        1.0, 1.0, 1.0, 0.5,      # velocity / orient / height
        1.0, 1.0,                # periodic
        1.0, 0.5,                # foot traj
        0.2, -0.0001, -0.01, -0.01  # regularizers
    ], device=root_states.device).unsqueeze(0)

    total_reward = torch.sum(rewards * μ, dim=1)

    # ------------------------------------------------------------ #
    # update contact_reward_sum (for logging)
    # ------------------------------------------------------------ #
    contact_reward_sum = contact_reward_sum + r_periodic_F

    # names for logging – keep index aligned with rewards order
    names: List[str] = [
        "lin_vel_track", "ang_vel_track",
        "orient_track", "height_track",
        "periodic_force", "periodic_vel",
        "foot_height_track", "foot_vel_track",
        "default_joint", "energy_cost",
        "action_smooth", "contact_clip"
    ]

    return total_reward, rewards, names, contact_reward_sum
