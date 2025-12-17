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

import os, sys
import gymnasium as gym
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

# utils
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results

TIME_STEP = 0.01

LEARNING_ALG = "PPO" #"SAC"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '102824115106'
log_dir = interm_dir + 'slope_cpg_bigNegx0.4yaw1.3velREWARD1.2targetvel' # change to your log dir

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
base_positions = []
base_velocities = []
base_cpg_r = []
base_cpg_theta = []

actions_list = []
rewards_list = []
CoT_list = []

# Get initial base position and robot mass for CoT calculation
robot = env.envs[0].env.robot
last_base_pos = np.array(robot.GetBasePosition(), dtype=float)
mass = sum(robot.GetTotalMassFromURDF())

for i in range(2000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test if the outputs make sense)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    
    # Extract info from step
    base_pos = np.array(info[0]['base_pos'], dtype=float)
    base_vel = info[0]['base_vel']
    
    # Get current motor velocities for energy calculation
    dq = robot.GetMotorVelocities()

    # Save data
    base_velocities.append(base_vel)
    base_cpg_r.append(info[0].get('cpg_r', np.zeros(4)))
    base_cpg_theta.append(info[0].get('cpg_theta', np.zeros(4)))

    actions_list.append(action)
    rewards_list.append(rewards)
    
    # Compute Cost of Transport (CoT)
    # Get motor torques (which were actually applied, not the raw actions)
    tau_all = np.array(robot.GetMotorTorques(), dtype=float)
    vel_all = np.array(dq, dtype=float)
    power = np.abs(np.dot(tau_all, vel_all))
    energy = power * TIME_STEP
    distance_travelled = np.sqrt((base_pos[0] - last_base_pos[0])**2 + (base_pos[1] - last_base_pos[1])**2)
    
    # Avoid division by zero
    if distance_travelled > 1e-6:
        CoT_list.append(energy / (mass * 9.81 * distance_travelled))
    else:
        CoT_list.append(0)
    
    last_base_pos = base_pos
    
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        print('Final base velocity', info[0]['base_vel'])
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 


base_velocities = np.array(base_velocities)
actions_list = np.array(actions_list)
rewards_list = np.array(rewards_list)
base_cpg_r = np.array(base_cpg_r)
base_cpg_theta = np.array(base_cpg_theta)
CoT_list = np.array(CoT_list)

# Calculate and print average metrics
avg_CoT = np.mean(CoT_list[CoT_list > 0])  # Exclude zero values
vel_norm = np.linalg.norm(base_velocities, axis=1)
avg_speed = np.mean(vel_norm)

print("\n" + "="*50)
print(f"Average Cost of Transport (CoT): {avg_CoT:.4f} J/(kg·m)")
print(f"Average Speed: {avg_speed:.4f} m/s")
print("="*50 + "\n")

# [TODO] make plots

# --- Plot velocity magnitude ---
vel_norm = np.linalg.norm(base_velocities, axis=1)
plt.figure()
plt.plot(vel_norm)
plt.title("Robot's linear velocity (x axis)")
plt.xlabel("Timestep")
plt.ylabel("|v| (m/s)")
plt.grid(True)


# --- Plot cpg_r ---
if base_cpg_r.size > 0 and base_cpg_r[0].size > 0:
    plt.figure()
    for i in range(base_cpg_r[0].size):
        plt.plot(base_cpg_r[:, i], label=f'Leg {i}')
    plt.title("cpg_r (CPG amplitudes)")
    plt.xlabel("Timestep")
    plt.ylabel("cpg_r")
    plt.legend()
    plt.grid(True)

# --- Plot cpg_theta ---
if base_cpg_theta.size > 0 and base_cpg_theta[0].size > 0:
    plt.figure()
    for i in range(base_cpg_theta[0].size):
        plt.plot(base_cpg_theta[:, i], label=f'Leg {i}')
    plt.title("cpg_theta (CPG phases)")
    plt.xlabel("Timestep")
    plt.ylabel("cpg_theta")
    plt.legend()
    plt.grid(True)

# --- Plot Cost of Transport (CoT) ---
plt.figure()
plt.plot(CoT_list)
plt.title("Cost of Transport (CoT)")
plt.xlabel("Timestep")
plt.ylabel("CoT (J/(kg·m))")
plt.grid(True)

plt.show()

'''mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus":False
})
'''

t = np.arange(2000)*TIME_STEP

def plot_cpg_states():
    legs = ['FR', 'FL', 'RR', 'RL']
    leg_colors = {
        'FR': '#4169E1',  # royal blue
        'FL': '#DC143C',  # crimson
        'RR': '#FF8C00',  # dark orange
        'RL': '#228B22',  # forest green
    }
    styles = [(0, (5, 3)), (2, (5, 3)), (4, (5, 3)), (6, (5, 3))]  # phase offsets

    # Changement ici : seulement 2 subplots au lieu de 4
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

    # r [m] - Premier subplot
    for j, (leg, color) in enumerate(leg_colors.items()):
        i = legs.index(leg)
        axs[0].plot(
            t, base_cpg_r[:, i],
            color=color, linestyle=styles[j],
            label=fr'{leg}', alpha=0.9
        )
    axs[0].set_ylabel(r'$r~[\mathrm{amplitude}]$')
    axs[0].legend(loc='upper right', ncol=1)

    # \theta [rad] - Deuxième subplot
    for j, (leg, color) in enumerate(leg_colors.items()):
        i = legs.index(leg)
        axs[1].plot(
            t, base_cpg_theta[:, i],
            color=color, linestyle=styles[j],
            label=fr'{leg}', alpha=0.9
        )
    axs[1].set_ylabel(r'$\theta~[\mathrm{rad}]$')
    axs[1].legend(loc='upper right', ncol=1)
    axs[1].set_xlabel(r'Time $[s]$')

    # Common formatting
    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.label_outer()  # hide inner x labels
        ax.set_xlim([0, 2000*TIME_STEP])

    plt.tight_layout()
    plt.savefig("lr_mp2_group_16/cpg_states_bound_gait.png", dpi=300, bbox_inches='tight')
    plt.show()
    # r (amplitude) converges to desired value (sqrt of mu)

plot_cpg_states()
