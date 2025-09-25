# Jacobian practical
from env.leg_gym_env import LegGymEnv
import numpy as np
import time

def jacobian_abs(q,l1=0.209,l2=0.195):
    """ Jacobian based on absolute angles (like double pendulum)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    # Jacobian
    J = np.zeros((2,2))
    J[0, 0] = l1*np.cos(q[0])
    J[1, 0] = l1*np.sin(q[0])
    J[0, 1] = l2*np.cos(q[1])
    J[1, 1] = l2*np.sin(q[1])

    # foot pos
    pos = np.zeros(2)
    pos[0] = + l1*np.sin(q[0]) + l2*np.sin(q[1])
    pos[1] = - l1*np.cos(q[0]) - l2*np.cos(q[1])

    return J, pos

def jacobian_rel(q,l1=0.209,l2=0.195):
    """ Jacobian based on relative angles (like URDF)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    # Jacobian
    J = np.zeros((2,2))
    J[0, 0] = - l1*np.cos(q[0]) + l2*np.cos(q[0] + q[1])
    J[1, 0] = + l1*np.sin(q[0]) + l2*np.sin(q[0] + q[1])
    J[0, 1] = l2*np.cos(q[0] + q[1])
    J[1, 1] = l2*np.sin(q[0] + q[1])

    # foot pos
    pos = np.zeros(2)
    pos[0] = - l1*np.sin(q[0]) + l2*np.sin(q[0] + q[1])
    pos[1] = - l1*np.cos(q[0]) - l2*np.cos(q[0] + q[1])

    return J, pos

if __name__ == "__main__":
    env = LegGymEnv(render=True, 
                    on_rack=False,    # set True to hang up robot in air 
                    motor_control_mode='TORQUE',
                    action_repeat=1,
                    )

    NUM_STEPS = 100*1000 # simulate 100 seconds (sim dt is 0.001)

    # env._robot_config.INIT_MOTOR_ANGLES = np.array([-np.pi/4 , np.pi/2]) # test different initial motor angles
    # obs = env.reset() # reset environment if changing initial configuration 

    action = np.zeros(2) # either torques or motor angles, depending on mode

    # Test different Cartesian gains! How important are these? 
    kpCartesian = np.diag([500]*2)
    kdCartesian = np.diag([30]*2)

    # test different desired foot positions
    des_foot_pos = np.array([0.3, -0.3]) 

    start_time = time.time()
    for step in range(NUM_STEPS):

        if (step+1) % 5000 == 0:
            print(f"Step {step}: 5 seconds elapsed.")
            if des_foot_pos[1] == -0.3:
                des_foot_pos[1] = -0.2
            else:
                des_foot_pos[1] = -0.3

        # Compute jacobian and foot_pos in leg frame (use GetMotorAngles())
        motor_ang = env.robot.GetMotorAngles()
        J, foot_pos = jacobian_rel(motor_ang)
        
        # Get foot velocity in leg frame (use GetMotorVelocities())
        motor_vel = env.robot.GetMotorVelocities()
        foot_vel = J @ motor_vel

        # Calculate torque (Cartesian PD, and/or desired force)
        tau = J.T @ (kpCartesian @ (des_foot_pos - foot_pos) + kdCartesian @ (- foot_vel))
        # desired velocity is zero
        
        # add gravity compensation (Force), (get mass with env.robot.total_mass)
        tau_FF = J.T @ (np.array([0, - env.robot.total_mass * 9.81])) 

        action = tau + tau_FF
        # apply control, simulate
        env.step(action)


    # make plots of joint positions, foot positions, torques, etc.
    # [TODO]