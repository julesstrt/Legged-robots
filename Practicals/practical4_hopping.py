# Hopping practical
from env.leg_gym_env import LegGymEnv
# from leg_helpers import *
import numpy as np
import matplotlib.pyplot as plt
from practical2_jacobian import jacobian_rel
from practical3_ik import ik_geometrical

env = LegGymEnv(render=True, 
                on_rack=False,    # set True to debug 
                motor_control_mode='TORQUE',
                action_repeat=1,
                # record_video=True
                )

NUM_SECONDS = 5   # simulate N seconds (sim dt is 0.001)
tau = np.zeros(2) # either torques or motor angles, depending on mode

SINGLE_JUMP = False

# sample Cartesian PD gains
kpCartesian = np.diag([500,300])
kdCartesian = np.diag([30,20])

# sample joint PD gains
kpJoint = np.array([1,1])
kdJoint = np.array([0.1,0.1])

kpCartesian = np.diag([300,200])
kdCartesian = np.diag([20,20])


t = np.linspace(0,NUM_SECONDS,NUM_SECONDS*1000 + 1)
Fx_max = 40     # max peak force in X direction
Fz_max = 90     # max peak force in Z direction
f = 1.5         # frequency

if SINGLE_JUMP:
    # kpCartesian = np.diag([200,200])
    # kdCartesian = np.diag([10,10])
    Fx_max = 0       # max peak force in X direction
    Fz_max = 200     # max peak force in Z direction
    f = 1.05

# Z force trajectory
force_traj_z = Fz_max * np.sin(2*np.pi*f * t)
force_traj_z[force_traj_z > 0] = 0
if SINGLE_JUMP:
    force_traj_z[round(1/f/0.001):] = 0 #-9.8 * 2.429 #0 # set to 0 for a single hop

# X force trajectory (change sign to go left/right)
force_traj_x = Fx_max * np.sin(2*np.pi*f * t)
# force_traj_x[force_traj_x > 0] = 0

# nominal foot position
nominal_foot_pos = np.array([0.0,-0.2]) 
max_base_z = 0

for i in range(NUM_SECONDS*1000):
    print("heyy")
    # Compute jacobian and foot_pos in leg frame (use GetMotorAngles() )
    J, ee_pos_legFrame = jacobian_rel(env.robot.GetMotorAngles())
    print("yooo")

    # foot velocity in leg frame (use GetMotorVelocities() )
    motor_vel = env.robot.GetMotorVelocities()
    foot_linvel = J @ motor_vel

    # calculate torque
    tau = J.T @ ( kpCartesian @ (nominal_foot_pos-ee_pos_legFrame) + kdCartesian @ (-foot_linvel))

    # add some small error correction based on joint positions (optional)
    tau += kpJoint * (ik_geometrical(nominal_foot_pos) - env.robot.GetMotorAngles()) + kdJoint * (-env.robot.GetMotorVelocities())

    base_pos = env.robot.GetBasePosition()
    if max_base_z < base_pos[2]:
        max_base_z = base_pos[2]
    # if base_pos[2] < 0.3:
    tau += J.T @ np.array([force_traj_x[i], force_traj_z[i]])
    action = tau

    # apply control, simulate
    env.step(tau)

print('Peak z', max_base_z)

plt.figure()
plt.plot(t,force_traj_z)
plt.xlabel('Time (s)')
plt.ylabel('Fz (N)')
plt.grid()
plt.show()

plt.figure()
plt.plot(t,force_traj_x)
plt.xlabel('Time (s)')
plt.ylabel('Fx (N)')
plt.grid()
plt.show()