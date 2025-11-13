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

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

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
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(gait="TROT", time_step=TIME_STEP)

SIMULATION_TIME = 2  # seconds
TEST_STEPS = int(SIMULATION_TIME / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
joint_pos = np.zeros((12, TEST_STEPS))
joint_vel = np.zeros((12, TEST_STEPS))
foot_x = np.zeros((4, TEST_STEPS))
foot_z = np.zeros((4, TEST_STEPS))
r_cpg = np.zeros((4, TEST_STEPS))
theta_cpg = np.zeros((4, TEST_STEPS))
r_dot_cpg = np.zeros((4, TEST_STEPS))
theta_dot_cpg = np.zeros((4, TEST_STEPS))

############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])

# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 

  # get desired foot positions from CPG 
  xs,zs = cpg.update()

  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)

    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])

    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz) # [TODO] 

    # Add joint PD contribution to tau for leg i (Equation 4)
    tau += kp * (leg_q - q[3*i:3*i+3]) - kd * dq[3*i:3*i+3] # [TODO] 

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get desired xyz position in leg frame (use ComputeJacobianAndPosition with the joint angles you just found above)
      Jd, pd = env.robot.ComputeJacobianAndPosition(i, specific_q=leg_q) # [TODO] 

      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      J, p = env.robot.ComputeJacobianAndPosition(i) # [TODO] 

      # Get current foot velocity in leg frame (Equation 2)
      v = J @ dq[3*i:3*i+3] # [TODO] 

      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += J.T @ (kpCartesian @ (pd - p) - kdCartesian @ v)  # [TODO]

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO] save any CPG or robot states
  joint_pos[:, j] = q
  joint_vel[:, j] = dq
  foot_x[:, j] = xs
  foot_z[:, j] = zs
  r_cpg[:, j] = cpg.get_r()
  theta_cpg[:, j] = cpg.get_theta()
  r_dot_cpg[:, j] = cpg.get_dr()
  theta_dot_cpg[:, j] = cpg.get_dtheta()


print(r_cpg[0, :])
##################################################### 
# PLOTS
#####################################################

# === 1. Plot CPG states (r, θ, ṙ, θ̇) ===
legs = ['FR', 'FL', 'RR', 'RL']
leg_colors = {
    'FR': '#4169E1',  # royal blue
    'FL': '#DC143C',  # crimson
    'RR': '#FF8C00',  # dark orange
    'RL': '#228B22',  # forest green
}
styles = [(0, (5, 3)), (2, (5, 3)), (4, (5, 3)), (6, (5, 3))]  # phase offsets

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 8))


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
axs[3].set_xlabel(r'\textbf{Time [s]}')
axs[3].legend(loc='upper right', ncol=1)

# Common formatting
for ax in axs:
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.label_outer()  # hide inner x labels
    ax.set_xlim([0, SIMULATION_TIME])

plt.tight_layout()
plt.savefig("cpg_states_trot_gait.png", dpi=300, bbox_inches='tight')
plt.show()
# r (amplitude) converges to desired value (sqrt of mu)


# # === 2. Compare desired vs actual foot position (FR leg) ===
# leg_index = 0
# J, p_actual = env.robot.ComputeJacobianAndPosition(leg_index)
# foot_x_des = foot_x[leg_index, :]
# foot_z_des = foot_z[leg_index, :]

# # For simplicity, assume you logged p_actual or recompute for each saved q
# # We'll simulate "actual" by using the last computed J,p at the end of sim
# fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8,6))
# axs[0].plot(t, foot_x_des, label='Desired x')
# axs[0].plot(t, np.gradient(foot_x_des, TIME_STEP)*0.0 + p_actual[0], '--', label='Actual x (approx)')
# axs[0].legend(); axs[0].set_ylabel('Foot X [m]')
# axs[1].plot(t, foot_z_des, label='Desired z')
# axs[1].plot(t, np.gradient(foot_z_des, TIME_STEP)*0.0 + p_actual[2], '--', label='Actual z (approx)')
# axs[1].legend(); axs[1].set_xlabel('Time [s]'); axs[1].set_ylabel('Foot Z [m]')
# plt.suptitle("Desired vs Actual Foot Position (FR leg) — Joint PD + Cartesian PD")
# plt.show()


# # === 3. Desired vs Actual Joint Angles (FR leg) ===
# leg_index = 0
# fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8,6))
# labels = ['Hip', 'Thigh', 'Calf']
# for k in range(3):
#     axs[k].plot(t, joint_pos[3*leg_index + k, :], label='Actual')
#     axs[k].plot(t, np.gradient(joint_pos[3*leg_index + k, :], TIME_STEP)*0.0 + leg_q[k],
#                 '--', label='Desired')
#     axs[k].set_ylabel(labels[k])
#     axs[k].legend(loc='upper right')
# axs[-1].set_xlabel('Time [s]')
# plt.suptitle("Desired vs Actual Joint Angles (FR leg)")
# plt.show()