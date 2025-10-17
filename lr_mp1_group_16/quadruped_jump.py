import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from env.simulation import QuadSimulator, SimulationOptions
from profiles import FootForceProfile

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False 
    })

N_LEGS = 4
N_JOINTS = 3
N_JUMPS = 3

# Parameters for force profile
FREQ0 = 1.5
FREQ1 = 0.25 # don't care, just enough time for the dog to stabilize after the last jump
FORCE_X = 75
FORCE_Y = 0
FORCE_Z = 200

# Controller gains
KP_CARTESIAN = np.diag([500, 500, 500])
KD_CARTESIAN = np.diag([20, 20, 20])
K_VMC = 500

# Desired end-effector positions
DES_EE_POS_LIST = np.array([
    [0, 0.10, -0.10],
    [0, 0.10, -0.20],
    [0, 0.10, -0.30],
])


def quadruped_jump(des_ee_pos, enable_virtual_model):
    sim_options = SimulationOptions(
        on_rack=False,
        render=True,
        record_video=False,
        tracking_camera=False,
    )
    simulator = QuadSimulator(sim_options)
    force_profile = FootForceProfile(f0=FREQ0, f1=FREQ1, Fx=FORCE_X, Fy=FORCE_Y, Fz=FORCE_Z)

    jump_duration = (force_profile.impulse_duration() + force_profile.idle_duration())
    n_steps = int((N_JUMPS * jump_duration + force_profile.idle_duration())/ sim_options.timestep)

    x_positions, y_positions, z_positions = [], [], []
    roll_angles, pitch_angles, yaw_angles = [], [], []
    contact_states = []

    for _ in range(n_steps):
        if not simulator.is_connected():
            break

        force_profile.step(sim_options.timestep)

        tau = np.zeros(N_JOINTS * N_LEGS)
        tau += nominal_position(simulator, des_ee_pos)
        tau += apply_force_profile(simulator, force_profile)
        tau += gravity_compensation(simulator)

        contacts = simulator.get_foot_contacts()
        contact_states.append(int(any(contacts)))

        if any(contacts) and enable_virtual_model:
            tau += virtual_model(simulator)

        simulator.set_motor_targets(tau)
        simulator.step()

        base_pos = simulator.get_base_position()        
        x_positions.append(base_pos[0])
        y_positions.append(base_pos[1])
        z_positions.append(base_pos[2])

        base_angles = simulator.get_base_orientation_roll_pitch_yaw()
        roll_angles.append(np.rad2deg(base_angles[0]))
        pitch_angles.append(np.rad2deg(base_angles[1]))
        yaw_angles.append(np.rad2deg(base_angles[2]))

    simulator.close()
    return (
        np.array(x_positions),
        np.array(y_positions),
        np.array(z_positions),
        np.array(roll_angles),
        np.array(pitch_angles),
        np.array(yaw_angles),
        np.array(contact_states),
        sim_options.timestep,
    )


def nominal_position(simulator, des_ee_pos):
    tau = np.zeros(N_JOINTS * N_LEGS)

    for leg_id in range(N_LEGS):
        tau_i = np.zeros(3)
        J, ee_pos = simulator.get_jacobian_and_position(leg_id)
        joints_vel = simulator.get_motor_velocities(leg_id)
        ee_vel = J @ joints_vel

        des_ee = des_ee_pos.copy()
        if leg_id in [0, 2]:  # left legs
            des_ee[1] = -abs(des_ee[1])

        tau_i += J.T @ (KP_CARTESIAN @ (des_ee - ee_pos) + KD_CARTESIAN @ (-ee_vel))
        tau[leg_id * N_JOINTS : (leg_id + 1) * N_JOINTS] = tau_i

    return tau


def apply_force_profile(simulator, force_profile):
    tau = np.zeros(N_JOINTS * N_LEGS)

    for leg_id in range(N_LEGS):
        J, _ = simulator.get_jacobian_and_position(leg_id)
        tau_i = J.T @ force_profile.force()
        tau[leg_id * N_JOINTS : (leg_id + 1) * N_JOINTS] = tau_i.flatten()

    return tau


def gravity_compensation(simulator):
    tau = np.zeros(N_JOINTS * N_LEGS)
    mass = simulator.get_mass() / 4
    g = 9.81
    Fg = np.array([0, 0, -mass * g])

    for leg_id in range(N_LEGS):
        J, _ = simulator.get_jacobian_and_position(leg_id)
        tau_i = J.T @ Fg
        tau[leg_id * N_JOINTS : (leg_id + 1) * N_JOINTS] = tau_i

    return tau


def virtual_model(simulator):
    R = simulator.get_base_orientation_matrix()
    P = R @ np.array([[1, 1, -1, -1], [-1, 1, -1, 1], [0, 0, 0, 0]])
    z = K_VMC * (np.array([0, 0, 1]) @ P)
    F_VMC = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      z])
    tau = np.zeros(N_JOINTS * N_LEGS)

    for leg_id in range(N_LEGS):
        J, _ = simulator.get_jacobian_and_position(leg_id)
        tau_i = J.T @ F_VMC[:, leg_id]
        tau[leg_id * N_JOINTS : (leg_id + 1) * N_JOINTS] = tau_i

    return tau


def nominal_foot_position_comparison():
    colormap = mpl.colormaps.get_cmap('viridis')
    colors = [colormap(0.1), colormap(0.5), colormap(0.9)]
    labels = ['for nominal foot position (0, 0.10, -0.10)', 
              'for nominal foot position (0, 0.10, -0.20)',
              'for nominal foot position (0, 0.10, -0.30)']
    robot_base = 'base'

    plt.figure(figsize=(8, 4))

    for des_pos, color, label in zip(DES_EE_POS_LIST, colors, labels):
        x, y, z, roll, pitch, yaw, dt = quadruped_jump(des_pos)
        t = np.arange(0, len(x) * dt, dt)
        plt.plot(t, x, color=color, linestyle='-.', label=rf'$x_{{\mathrm{{{robot_base}}}}}$ {label}')
        plt.plot(t, z, color=color, linestyle='-', label=rf'$z_{{\mathrm{{{robot_base}}}}}$ {label}')
        
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('nominal_foot_position_comparison.png', dpi=300)
    plt.show()


def virtual_model_comparison():
    colormap = mpl.colormaps.get_cmap('inferno')
    colors = [colormap(0.2), colormap(0.5)]
    labels = ['with virtual model', 'without virtual model']
    robot_base = 'base'
    des_pos = DES_EE_POS_LIST[1]

    plt.figure(figsize=(8, 4))

    for enable_vmc, color, label in zip([True, False], colors, labels):
        x, y, z, roll, pitch, yaw, contacts, dt = quadruped_jump(
            des_pos, enable_vmc
        )
        t = np.arange(0, len(x) * dt, dt)

        # Shade contact regions only when VMC is active
        if enable_vmc:
            contact_intervals = np.diff(np.concatenate(([0], contacts, [0])))
            start_indices = np.where(contact_intervals == 1)[0]
            end_indices = np.where(contact_intervals == -1)[0]
            for start, end in zip(start_indices, end_indices):
                plt.axvspan(
                    t[start], t[end - 1],
                    color='steelblue', alpha=0.15, zorder=0, edgecolor='none',
                    label='Contact phase' if start == start_indices[0] else None
                )

        plt.plot(t, roll, color=color, linestyle='-.', label=rf'$\phi_{{\mathrm{{{robot_base}}}}}$ {label}')
        plt.plot(t, pitch, color=color, linestyle='-', label=rf'$\theta_{{\mathrm{{{robot_base}}}}}$ {label}')
        plt.plot(t, yaw, color=color, linestyle=':', label=rf'$\psi_{{\mathrm{{{robot_base}}}}}$ {label}')

    plt.xlabel('Time (s)')
    plt.ylabel('Angles (°)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('virtual_model_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    quadruped_jump(DES_EE_POS_LIST[1], True)
    # nominal_foot_position_comparison()
    # virtual_model_comparison() 