import numpy as np
from env.simulation import QuadSimulator, SimulationOptions

from profiles import FootForceProfile

N_LEGS = 4
N_JOINTS = 3

# Frequencies for profile
FREQ0 = 0.01 
FREQ1 = 0.01

# Gains
KP_JOINT = np.array([1, 1, 1])
KD_JOINT = np.array([0.1, 0.1, 0.1])
KP_CARTESIAN = np.diag([300, 300, 300])
KD_CARTESIAN = np.diag([20, 20, 20])

DES_EE_POS = np.array([0, 0, -0.3])  # Desired foot position in leg frame
# x and z ok, but y needs to change sign

def quadruped_jump():
    # Initialize simulation
    # Feel free to change these options! (except for control_mode and timestep)
    sim_options = SimulationOptions(
        on_rack=True,  # Whether to suspend the robot in the air (helpful for debugging)
        render=True,  # Whether to use the GUI visualizer (slower than running in the background)
        record_video=False,  # Whether to record a video to file (needs render=True)
        tracking_camera=True,  # Whether the camera follows the robot (instead of free)
    )
    simulator = QuadSimulator(sim_options)

    # TODO: set parameters for the foot force profile here
    force_profile = FootForceProfile(f0=FREQ0, f1=FREQ1, Fx=20, Fy=0, Fz=200)

    # Determine number of jumps to simulate
    n_jumps = 10  # Feel free to change this number
    jump_duration =  force_profile.impulse_duration() + force_profile.idle_duration() # determine how long a jump takes
    n_steps = int(n_jumps * jump_duration / sim_options.timestep)

    for _ in range(n_steps):
        # If the simulator is closed, stop the loop
        if not simulator.is_connected():
            break

        # Step the oscillator
        force_profile.step(sim_options.timestep)

        # Compute torques as motor targets
        # The convention is as follows:
        # - A 1D array where the torques for the 3 motors follow each other for each leg
        # - The first 3 elements are the hip, thigh, calf torques for the FR leg.
        # - The order of the legs is FR, FL, RR, RL (front/rear,right/left)
        # - The resulting torque array is therefore structured as follows:
        # [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        tau = np.zeros(N_JOINTS * N_LEGS)

        # TODO: implement the functions below, and add potential controller parameters as function parameters here
        tau += nominal_position(simulator)
        tau += apply_force_profile(simulator, force_profile)
        tau += gravity_compensation(simulator)

        # If touching the ground, add virtual model
        on_ground = True  # TODO: how do we know we're on the ground?
        if on_ground:
            tau += virtual_model(simulator)

        # Set the motor commands and step the simulation
        simulator.set_motor_targets(tau)
        simulator.step()

    # Close the simulation
    simulator.close()

    # OPTIONAL: add additional functions here (e.g., plotting)

def pseudoInverse(A,lam=0.001):
    """ Pseudo inverse of matrix A. 
        Make sure to take into account dimensions of A
            i.e. if A is mxn, what dimensions should pseudoInv(A) be if m>n 
    """
    m,n = np.shape(A)
    pinvA = None

    if m >= n:
        # left pseudoinverse
        pinvA = np.linalg.inv( A.T @ A + lam**2 * np.eye(n) ) @  A.T 
    else:
        # right pseudoinverse
        pinvA = A.T @ np.linalg.inv( A @ A.T + lam**2 * np.eye(m) )

    return pinvA

def ik_numerical(init_joint_angles, des_ee_pos, J, ee_pos):
    i = 0
    max_i = 100 # max iterations
    des_joint_angles = init_joint_angles
    alpha = 0.5 # convergence factor
    lam = 0.001 # damping factor for pseudoInverse
    tol=1e-4

    # Condition to iterate: while fewer than max iterations, and while error is greater than tolerance
    while( i < max_i and abs(sum(ee_pos - des_ee_pos)) > tol ):

        # Compute pseudoinverse
        J_pinv = pseudoInverse(J, lam)

        # Find end effector error vector
        ee_error = des_ee_pos - ee_pos

        # update joint_angles
        des_joint_angles += alpha * J_pinv @ ee_error

        # update iteration counter
        i += 1

    return des_joint_angles

def nominal_position(
    simulator: QuadSimulator,
    # OPTIONAL: add potential controller parameters here (e.g., gains)
) -> np.ndarray:
    # All motor torques are in a single array
    tau = np.zeros(N_JOINTS * N_LEGS)

    for leg_id in range(N_LEGS):
        # TODO: compute nominal position torques for leg_id
        tau_i = np.zeros(3)

        J, ee_pos = simulator.get_jacobian_and_position(leg_id)
        joint_angles = simulator.get_motor_angles(leg_id)
        # init_joint_angles = env._robot_config.INIT_MOTOR_ANGLES
        joints_vel = simulator.get_motor_velocities(leg_id)
        ee_vel = J @ joints_vel

        # Cartesian component
        tau_i += J.T @ ( KP_CARTESIAN @ (DES_EE_POS - ee_pos) + KD_CARTESIAN @ (- ee_vel) )

        # Joint component
        # tau_i += KP_JOINT * (ik_numerical(joint_angles, DES_EE_POS, J, ee_pos) - joint_angles) + KD_JOINT * (- joints_vel)
        

        print("current:", joint_angles)
        print("desired:", ik_numerical(joint_angles, DES_EE_POS, J, ee_pos))

        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i
    return tau


def virtual_model(
    simulator: QuadSimulator,
    # OPTIONAL: add potential controller parameters here (e.g., gains)
) -> np.ndarray:
    # All motor torques are in a single array
    tau = np.zeros(N_JOINTS * N_LEGS)
    for leg_id in range(N_LEGS):

        # TODO: compute virtual model torques for leg_id
        tau_i = np.zeros(3)

        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i

    return tau


def gravity_compensation(
    simulator: QuadSimulator,
    # OPTIONAL: add potential controller parameters here (e.g., gains)
) -> np.ndarray:
    # All motor torques are in a single array
    tau = np.zeros(N_JOINTS * N_LEGS)
    for leg_id in range(N_LEGS):

        # TODO: compute gravity compensation torques for leg_id
        tau_i = np.zeros(3)

        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i

    return tau


def apply_force_profile(
    simulator: QuadSimulator,
    force_profile: FootForceProfile,
    # OPTIONAL: add potential controller parameters here (e.g., gains)
) -> np.ndarray:
    # All motor torques are in a single array
    tau = np.zeros(N_JOINTS * N_LEGS)
    for leg_id in range(N_LEGS):

        # TODO: compute force profile torques for leg_id
        tau_i = np.zeros(3)

        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i

    return tau


if __name__ == "__main__":
    quadruped_jump()
