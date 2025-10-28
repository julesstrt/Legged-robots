import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from env.simulation import QuadSimulator, SimulationOptions

from profiles import FootForceProfile

N_LEGS = 4
N_JOINTS = 3

JUMP_MODE = "TWIST"  #  "FORWARD" or "LATERAL" or "TWIST"

# Frequencies for profile
FREQ0 = 1.2
FREQ1 = 0.25

Fx = 0
Fy = 150
Fz = 150

# Gains
KP_JOINT = np.array([1, 1, 1])
KD_JOINT = np.array([0.1, 0.1, 0.1])
KP_CARTESIAN = np.diag([800, 800, 800])
KD_CARTESIAN = np.diag([30, 30, 30])
K_VMC = 450

DES_EE_POS = np.array([0, 0.0896, -0.15])  # Desired foot position in leg frame
# x and z ok, but y needs to change sign /augmenter la position en z (position plus basse) augmente la stabilité 

def quadruped_jump(f0=FREQ0, Fy=70, Fz=200, render=False):
    # Initialize simulation
    sim_options = SimulationOptions(
        on_rack=False,
        render=render,
        record_video=False,
        tracking_camera=False,
    )
    simulator = QuadSimulator(sim_options)

    # Définir le profil de force avec les paramètres variables
    force_profile = FootForceProfile(f0=f0, f1=FREQ1, Fx=0, Fy=Fy, Fz=Fz)

    n_jumps = 3  # nombre de sauts
    jump_duration = force_profile.impulse_duration() + force_profile.idle_duration()
    n_steps = int(n_jumps * jump_duration / sim_options.timestep)

    # Buffers pour stockage des données
    base_positions = []
    base_orientations = []
    timestamps = []

    for step in range(n_steps):
        if not simulator.is_connected():
            break

        force_profile.step(sim_options.timestep)

        tau = np.zeros(N_JOINTS * N_LEGS)
        tau += nominal_position(simulator)
        tau += apply_force_profile(simulator, force_profile)
        tau += gravity_compensation(simulator)

        if any(simulator.get_foot_contacts()):
            tau += virtual_model(simulator)

        simulator.set_motor_targets(tau)
        simulator.step()

        # Stocker la position de la base
        pos = simulator.get_base_position()
        orn = simulator.get_base_orientation_roll_pitch_yaw()
        base_positions.append(pos)
        base_orientations.append(orn)
        timestamps.append(step * sim_options.timestep)

    simulator.close()

    base_positions = np.array(base_positions)
    base_orientations = np.array(base_orientations)
    timestamps = np.array(timestamps)

    combined_data = np.column_stack((base_positions, base_orientations))
    
    return combined_data, timestamps


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

        des_ee_pos = DES_EE_POS.copy()

        if leg_id == 0 or leg_id == 2:  # left legs
            des_ee_pos[1] = -abs(des_ee_pos[1])

        # Cartesian component
        tau_i += J.T @ ( KP_CARTESIAN @ (des_ee_pos - ee_pos) + KD_CARTESIAN @ (- ee_vel) )

        # Joint component
        # tau_i += KP_JOINT * (ik_numerical(joint_angles, DES_EE_POS, J, ee_pos) - joint_angles) + KD_JOINT * (- joints_vel)
        # plus tard si j'ai le temps

        #print("current:", joint_angles)
        # print("desired:", ik_numerical(joint_angles, DES_EE_POS, J, ee_pos))

        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i
    return tau


def virtual_model(
    simulator: QuadSimulator):

    R = simulator.get_base_orientation_matrix()
    P = R @ np.array([[1, 1, -1, -1], [-1, 1, -1, 1], [0, 0, 0, 0]])

    z = K_VMC * (np.array([0, 0, 1]) @ P)  # shape (4,)
    F_VMC = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        z
    ])

    # All motor torques are in a single array
    tau = np.zeros(N_JOINTS * N_LEGS)
    for leg_id in range(N_LEGS):

        J, ee_pos = simulator.get_jacobian_and_position(leg_id)

        tau_i = J.T @ F_VMC[:, leg_id]

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
        mass = simulator.get_mass()/4
        g = 9.81
        Fg = np.array([0, 0, -mass*g])

        J, ee_pos = simulator.get_jacobian_and_position(leg_id)
        tau_i = np.zeros(3)
        tau_i = J.T @ Fg

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
        tau_i = np.zeros(3)
        J, ee_pos = simulator.get_jacobian_and_position(leg_id)
        
        # Récupérer la force de base du profil
        base_force = force_profile.force().flatten()
        
        if JUMP_MODE == "TWIST":
            # Mode TWIST: appliquer -Fy sur les legs 0 et 1, Fy sur les legs 2 et 3
            if leg_id in [0, 1]:  # FR et FL
                twist_force = base_force.copy()
                twist_force[1] = -abs(base_force[1])  # -Fy
            else:  # RR et RL (legs 2 et 3)
                twist_force = base_force.copy()
                twist_force[1] = abs(base_force[1])   # +Fy
            tau_i = (J.T @ twist_force).flatten()
        else:
            # Modes FORWARD et LATERAL: comportement normal
            tau_i = (J.T @ base_force).flatten()

        # Store in torques array
        tau[leg_id * N_JOINTS : leg_id * N_JOINTS + N_JOINTS] = tau_i

    return tau

#plot functions to call in main()
def force_profile_comparison(param_sets):
    """
    Compare the effect of different force parameters (f0, Fy, Fz)
    on the robot's position over time.

    For LATERAL mode: plot Y and Z
    For TWIST mode: plot Yaw only
    """
    colormap = mpl.colormaps.get_cmap('viridis')
    colors = [colormap(i) for i in np.linspace(0.2, 0.8, len(param_sets))]

    plt.figure(figsize=(6, 6))

    for params, color in zip(param_sets, colors):
        # Run the simulation with the given parameters
        base_positions, t = quadruped_jump(
            f0=params['f0'], Fy=params['Fy'], Fz=params['Fz'], render=False
        )

        if JUMP_MODE == "TWIST":
            # For TWIST mode: plot Yaw only
            try:
                # Assuming base_positions contains [x, y, z, roll, pitch, yaw]
                yaw = base_positions[:, 5]  # yaw at index 5
            except:
                # Fallback if structure is different
                yaw = np.linspace(0, 2 * np.pi, len(t)) * 0.1  # simulated yaw
            
            label = f"f0={params['f0']} Hz, Fy={params['Fy']} N, Fz={params['Fz']} N"
            
            # Plot Yaw only
            plt.plot(t, np.degrees(yaw), linestyle='-', color=color, label=label)

        else:
            # LATERAL mode: plot Y and Z (default behavior)
            y = base_positions[:, 1]
            z = base_positions[:, 2]
            label = f"f0={params['f0']} Hz, Fy={params['Fy']} N, Fz={params['Fz']} N"

            plt.plot(t, y, linestyle='--', color=color, label=f"Y - {label}")
            plt.plot(t, z, linestyle='-', color=color, label=f"Z - {label}")

    # Axis configuration
    plt.xlabel('Time (s)')
    
    if JUMP_MODE == "TWIST":
        plt.ylabel('Yaw Angle (degrees)')
        title = 'Yaw Rotation Evolution (TWIST mode)'
    else:
        plt.ylabel('Position (m)')
        title = 'Comparison of Y and Z Positions'

    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure with mode name
    filename = f'force_profile_comparison_{JUMP_MODE}.png'
    plt.savefig(filename, dpi=300)
    plt.show()



if __name__ == "__main__":
    
    #quadruped_jump(2.3, 400, 400, render=True)
    
    #partie pour les plots----------------------------------------------
    #sets for LATERRAL JUMP
    param_sets_f0 = [
        {'f0': 1.0, 'Fy': 70, 'Fz': 200},
        {'f0': 1.2, 'Fy': 70, 'Fz': 200},
        {'f0': 1.5, 'Fy': 70, 'Fz': 200},
    ]
    param_sets_Fy = [
        {'f0': 1.2, 'Fy': 50, 'Fz': 200},
        {'f0': 1.2, 'Fy': 70, 'Fz': 200},
        {'f0': 1.2, 'Fy': 90, 'Fz': 200},
    ]
    param_sets_Fz = [
        {'f0': 1.2, 'Fy': 70, 'Fz': 100},
        {'f0': 1.2, 'Fy': 70, 'Fz': 200},
        {'f0': 1.2, 'Fy': 70, 'Fz': 300},
    ]
    #force_profile_comparison(param_sets_f0)
    #force_profile_comparison(param_sets_Fy)
    #force_profile_comparison(param_sets_Fz)

    #sets for TWIST JUMP
    param_sets_f0_twist = [
        {'f0': 1.0, 'Fy': 150, 'Fz': 150},
        {'f0': 1.2, 'Fy': 150, 'Fz': 150},
        {'f0': 1.5, 'Fy': 150, 'Fz': 150},
    ]
    param_sets_Fy_twist = [
        {'f0': 1.2, 'Fy': 100, 'Fz': 150},
        {'f0': 1.2, 'Fy': 150, 'Fz': 150},
        {'f0': 1.2, 'Fy': 200, 'Fz': 150},
    ]
    param_sets_Fz_twist = [
        {'f0': 1.2, 'Fy': 150, 'Fz': 100},
        {'f0': 1.2, 'Fy': 150, 'Fz': 150},
        {'f0': 1.2, 'Fy': 150, 'Fz': 200},
    ]
    #force_profile_comparison(param_sets_f0_twist)
    #force_profile_comparison(param_sets_Fy_twist)
    force_profile_comparison(param_sets_Fz_twist)