# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

""" Run CPG """

import time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib as mpl
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False 
    })

ADD_CARTESIAN_PD = True 
TIME_STEP = 0.001
SIMULATION_TIME = 10                                 # in seconds
TEST_STEPS = int(SIMULATION_TIME / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP                 # time vector for plots
foot_y = 0.0838                                     # this is the hip length 
sideSign = np.array([-1, 1, -1, 1])                 # get correct hip sign (body right is negative)
env = QuadrupedGymEnv(render=True,                  # visualize
                    on_rack=False,                  # useful for debugging! 
                    isRLGymInterface=False,         # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,
                    record_video=False
                    )

# Initialize Hopf Network, supply gait
def omega_stance(T, duty):
    return 2*np.pi / (duty * T)

def omega_swing(T, duty):
    return 2*np.pi / ((1 - duty) * T)

duty_factor = 0.7
stride_duration = 0.50
cpg = HopfNetwork(gait="TROT", omega_stance=omega_stance(stride_duration, duty_factor), omega_swing=omega_swing(stride_duration, duty_factor), time_step=TIME_STEP)

# Initialize data structures to save CPG and robot states
r_cpg = np.zeros((4, TEST_STEPS))
theta_cpg = np.zeros((4, TEST_STEPS))
r_dot_cpg = np.zeros((4, TEST_STEPS))
theta_dot_cpg = np.zeros((4, TEST_STEPS))
foot_x_desired = np.zeros((4, TEST_STEPS))
foot_z_desired = np.zeros((4, TEST_STEPS))
foot_x_current = np.zeros((4, TEST_STEPS))
foot_z_current = np.zeros((4, TEST_STEPS))
joint_angles_desired = np.zeros((12, TEST_STEPS))
joint_angles_current = np.zeros((12, TEST_STEPS))
x_body_vel = np.zeros(TEST_STEPS)
CoT = np.zeros(TEST_STEPS)

# Joint PD gains
kp=np.array([150,150,150])
kd=np.array([2,2,2])

# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

last_base_pos = np.array([0.0, 0.0, 0.0])


# Main simulation loop
for j in range(TEST_STEPS):
    # Initialize torque array to send to motors
    action = np.zeros(12) 

    # Get desired foot positions from CPG 
    xs,zs = cpg.update()

    # Get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    q = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()
    base_pos = env.robot.GetBasePosition()
    base_pos = np.array(base_pos, dtype=float)

    # Loop through desired foot positions and calculate torques
    for i in range(4):
        # initialize torques for legi
        tau = np.zeros(3)

        # get desired foot i pos (xi, yi, zi) in leg frame
        leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])
        # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
        leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz) # [TODO] 

        # Add joint PD contribution to tau for leg i (Equation 4)
        tau += kp * (leg_q - q[3*i:3*i+3]) - kd * dq[3*i:3*i+3] # [TODO] 

        # Get desired xyz position in leg frame (use ComputeJacobianAndPosition with the joint angles you just found above)
        Jd, pd = env.robot.ComputeJacobianAndPosition(i, specific_q=leg_q) # [TODO] 

        # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
        J, p = env.robot.ComputeJacobianAndPosition(i) # [TODO] 
        
        # Get current foot velocity in leg frame (Equation 2)
        v = J @ dq[3*i:3*i+3] # [TODO] 

        # add Cartesian PD contribution
        if ADD_CARTESIAN_PD:
            # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
            tau += J.T @ (kpCartesian @ (pd - p) - kdCartesian @ v)  # [TODO]

        # Set tau for legi in action vector
        action[3*i:3*i+3] = tau

        # Save data for plotting
        foot_x_current[i, j] = p[0]
        foot_z_current[i, j] = p[2]
        foot_x_desired[i, j] = leg_xyz[0]
        foot_z_desired[i, j] = leg_xyz[2]
        joint_angles_current[3*i:3*i+3, j] = q[3*i:3*i+3]
        joint_angles_desired[3*i:3*i+3, j] = leg_q

    # Get body x velocity
    x_body_vel[j] = env.robot.GetBaseLinearVelocity()[0]  

    # Compute Cost of Transport (CoT)
    tau_all = np.array(action, dtype=float)
    vel_all = np.array(dq, dtype=float)
    power = np.abs(np.dot(tau_all, vel_all))
    energy = power * TIME_STEP
    mass = sum(env.robot.GetTotalMassFromURDF())
    distance_travelled = np.sqrt((base_pos[0] - last_base_pos[0])**2 + (base_pos[1] - last_base_pos[1])**2)
    CoT[j] = energy / (mass * 9.81 * distance_travelled)

    # Send torques to robot and simulate TIME_STEP seconds 
    env.step(action)
    last_base_pos = base_pos 

    # Save any CPG or robot states
    r_cpg[:, j] = cpg.get_r()
    theta_cpg[:, j] = cpg.get_theta()
    r_dot_cpg[:, j] = cpg.get_dr()
    theta_dot_cpg[:, j] = cpg.get_dtheta()
  
  
##################################################### 
# PLOTS
#####################################################
def plot_cpg_states():
    legs = ['FR', 'FL', 'RR', 'RL']
    leg_colors = {
        'FR': '#4169E1',  # royal blue
        'FL': '#DC143C',  # crimson
        'RR': '#FF8C00',  # dark orange
        'RL': '#228B22',  # forest green
    }
    styles = [(0, (5, 3)), (2, (5, 3)), (4, (5, 3)), (6, (5, 3))]  # phase offsets

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(6, 8))


    # r [m]
    for j, (leg, color) in enumerate(leg_colors.items()):
        i = legs.index(leg)
        axs[0].plot(
            t, r_cpg[i, :],
            color=color, linestyle=styles[j],
            label=fr'{leg}', alpha=0.9
        )
    axs[0].set_ylabel(r'$r~[\mathrm{amplitude}]$')
    axs[0].legend(loc='upper right', ncol=1)

    # \dot{r} [m/s]
    for j, (leg, color) in enumerate(leg_colors.items()):
        i = legs.index(leg)
        axs[1].plot(
            t, r_dot_cpg[i, :],
            color=color, linestyle=styles[j],
            label=fr'{leg}', alpha=0.9
        )
    axs[1].set_ylabel(r'$\dot{r}~[\mathrm{rate}]$')
    axs[1].legend(loc='upper right', ncol=1)

    # \theta [rad]
    for j, (leg, color) in enumerate(leg_colors.items()):
        i = legs.index(leg)
        axs[2].plot(
            t, theta_cpg[i, :],
            color=color, linestyle=styles[j],
            label=fr'{leg}', alpha=0.9
        )
    axs[2].set_ylabel(r'$\theta~[\mathrm{rad}]$')
    axs[2].legend(loc='upper right', ncol=1)

    # \dot{\theta} [rad/s]
    for j, (leg, color) in enumerate(leg_colors.items()):
        i = legs.index(leg)
        axs[3].plot(
            t, theta_dot_cpg[i, :],
            color=color, linestyle=styles[j],
            label=fr'{leg}', alpha=0.9
        )
    axs[3].set_ylabel(r'$\dot{\theta}~[\mathrm{rad/s}]$')
    axs[3].set_xlabel('Time [s]')
    axs[3].legend(loc='upper right', ncol=1)

    # Common formatting
    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.label_outer()  # hide inner x labels
        ax.set_xlim([0, SIMULATION_TIME])

    plt.tight_layout()
    plt.savefig("lr_mp2_group_16/cpg_states_bound_gait.png", dpi=300, bbox_inches='tight')
    plt.show()
    # r (amplitude) converges to desired value (sqrt of mu)


def plot_foot_positions():
    leg_index = 0

    fig, ax = plt.subplots(1, 1, figsize=(6,5))

    ax.plot(foot_x_desired[leg_index, :], foot_z_desired[leg_index, :], label='Desired foot position', color='#FF8C00', alpha=0.7)
    ax.plot(foot_x_current[leg_index, :], foot_z_current[leg_index, :], label='Acutal foot position', color='#4169E1', alpha=0.7)

    ax.set_ylabel('Foot z position [m]')
    ax.set_xlabel('Foot x position [m]')

    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')

    # if ADD_CARTESIAN_PD:
    #     plt.savefig("lr_mp2_group_16/foot_position_trot_gait_with_cartesian.png", dpi=300, bbox_inches='tight')
    # else:
    #     plt.savefig("lr_mp2_group_16/foot_position_trot_gait_without_cartesian.png", dpi=300, bbox_inches='tight')
    plt.savefig("lr_mp2_group_16/foot_position_trot_gait_test2.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_joint_angles():
    leg_index = 0
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6,6))
    labels = ['Hip [°]', 'Thigh [°]', 'Calf [°]']
    for k in range(3):
        axs[k].plot(t, np.rad2deg(joint_angles_desired[3*leg_index + k, :]), label='Desired joint angle', color='#228B22', alpha=0.7)
        axs[k].plot(t, np.rad2deg(joint_angles_current[3*leg_index + k, :]), label='Actual joint angle', color='#DC143C', alpha=0.7)

        axs[k].set_ylabel(labels[k])

    axs[0].legend(loc='upper right')
    axs[-1].set_xlabel('Time [s]')

    if ADD_CARTESIAN_PD:
        plt.savefig("lr_mp2_group_16/joint_angle_trot_gait_with_cartesian.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig("lr_mp2_group_16/joint_angle_trot_gait_without_cartesian.png", dpi=300, bbox_inches='tight')

    plt.show()


def plot_body_velocity():
    fig, ax = plt.subplots(1, 1, figsize=(6,5))

    ax.plot(t, x_body_vel, label='Body x velocity', color='#800080', alpha=0.7)
    ax.plot(t, CoT, label='Cost of Transport', color='#FFA500', alpha=0.7)

    ax.set_ylabel('Body x velocity [m/s]')
    ax.set_xlabel('Time [s]')

    ax.legend(loc='upper right')
    plt.savefig("lr_mp2_group_16/body_velocity_trot_gait.png", dpi=300, bbox_inches='tight')
    plt.show()

    avg_velocity = np.mean(x_body_vel[int(0.5*TEST_STEPS):])

    time_stance = (2*np.pi)/cpg._omega_stance
    time_swing = (2*np.pi)/cpg._omega_swing
    time_stride = time_stance + time_swing

    duty_factor = time_stance / time_stride

    valid = CoT[~np.isnan(CoT)]
    avg_CoT = np.mean(valid[int(0.5*len(valid)):])

    print(f"\nMetrics: Average body x velocity: {avg_velocity:.3f} m/s, Duty factor: {duty_factor:.3f}, Stride duration: {time_stride:.3f} s, Average CoT: {avg_CoT:.3f}")
    print(f"Params: ground clearance {cpg._ground_clearance:.3f} m, penetration {cpg._ground_penetration:.3f} m, desired step length {cpg._des_step_len:.3f} m \n")


def plot_duty_factor():
    # Data
    duty = np.array([0.50, 0.55, 0.60, 0.65, 0.70,
                    0.75, 0.80, 0.85, 0.90]) * 100.0
    vx = np.array([0.253, 0.543, 0.576, 0.538, 0.582,
                0.540, 0.489, 0.453, 0.254])

    cot = np.array([0.667, 0.625, 0.438, 0.393, 0.382,
                    0.443, 0.556, 0.767, 1.285])

    # Plot
    fig, ax_left = plt.subplots(figsize=(8, 5))

    # Left axis: velocity
    ax_left.plot(duty, vx, marker='o', label=r'$\bar{v_x}$')
    ax_left.set_xlabel("Duty factor [%]")
    ax_left.set_ylabel(r"Average body x velocity $\bar{v_x}$ [m/s]")
    ax_left.grid(True, linestyle='--', alpha=0.5)

    # Right axis: CoT
    ax_right = ax_left.twinx()
    ax_right.plot(duty, cot, marker='s', color='tab:red',
                label='CoT')
    ax_right.set_ylabel("Cost of Transport (CoT)")

    # Legends
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc='best')

    plt.tight_layout()
    plt.show()

# plot_cpg_states()
# plot_foot_positions()
# plot_joint_angles()
plot_body_velocity()
# plot_duty_factor()

