# Jacobian practical
from env.leg_gym_env import LegGymEnv
import numpy as np

def jacobian_abs(q,l1=0.209,l2=0.195):
    """ Jacobian based on absolute angles (like double pendulum)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    s1 = np.sin(q[0])
    s2 = np.sin(q[1])
    c1 = np.cos(q[0])
    c2 = np.cos(q[1])

    # Jacobian
    J = np.zeros((2,2))
    J[0, 0] = l1 * c1 
    J[1, 0] = l1 * s1
    J[0, 1] = l2 * c2 
    J[1, 1] = l2 * s2

    # foot pos
    pos = np.zeros(2)
    pos[0] =  l1 * s1 + l2 * s2
    pos[1] = -l1 * c1 - l2 * c2 

    return J, pos

def jacobian_rel(q,l1=0.209,l2=0.195):
    """Jacobian based on relative angles (like URDF)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    s1 = np.sin(q[0])
    s2 = np.sin(q[1])
    c1 = np.cos(q[0])
    c2 = np.cos(q[1])

    # Jacobian 
    J = np.zeros((2,2))
    J[0,0] = -l1 * c1 - l2 * np.cos(q[0] + q[1])
    J[1,0] =  l1 * s1 + l2 * np.sin(q[0] + q[1])
    J[0,1] = -l2 * np.cos(q[0] + q[1]) 
    J[1,1] =  l2 * np.sin(q[0] + q[1])
    
    # foot pos
    pos = np.zeros(2)
    pos[0] =  -l1 * s1 - l2 * np.sin(q[0]+q[1])
    pos[1] =  -l1 * c1 - l2 * np.cos(q[0]+q[1])

    return J, pos


# env = LegGymEnv(render=True, 
#                 on_rack=False,    # set True to debug 
#                 motor_control_mode='TORQUE',
#                 action_repeat=1,
#                 )

# # env._robot_config.INIT_MOTOR_ANGLES = np.array([-np.pi/4 , np.pi/2]) # test different initial motor angles
# # obs = env.reset()
# action = np.zeros(2) # either torques or motor angles, depending on mode

# kpCartesian = np.diag([500]*2)
# kdCartesian = np.diag([30]*2)

# # for testing force 
# kpCartesian = np.diag([200,200])
# kdCartesian = np.diag([10,10])

# des_foot_pos = np.array([0.0,-0.3]) 

# while True:
#     # Compute jacobian and foot_pos in leg frame (use GetMotorAngles() )
#     J, ee_pos_legFrame = jacobian_rel(env.robot.GetMotorAngles())
#     print('test pos',ee_pos_legFrame)

#     # foot velocity in leg frame (use GetMotorVelocities() )
#     motor_vel = env.robot.GetMotorVelocities()
#     foot_linvel = J @ motor_vel

#     # calculate torque
#     tau = J.T @ ( kpCartesian @ (des_foot_pos-ee_pos_legFrame) + kdCartesian @ (-foot_linvel))

#     # if ee_pos_legFrame[1] > des_foot_pos[1]: # compensate weak Cartesian PD gains
#     tau += J.T @ (-np.array([0,9.8*env.robot.total_mass]))
#     action = tau


#     # apply control, simulate
#     env.step(action)
