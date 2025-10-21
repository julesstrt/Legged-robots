import optuna
import numpy as np
from functools import partial
from optuna.trial import Trial
from env.simulation import QuadSimulator, SimulationOptions
import matplotlib.pyplot as plt
import pandas as pd

from profiles import FootForceProfile
from quadruped_jump import (
    nominal_position,
    gravity_compensation,
    apply_force_profile,
    virtual_model,
)


N_LEGS = 4
N_JOINTS = 3
DES_EE_POS = np.array([0, 0.10, -0.20])
NB_TRIALS = 25
NB_JUMPS = 1
NB_SEEDS = 5


def multi_seed_optimization():
    all_values = []
    all_best_params = []
    all_param_trajectories = []

    for seed in range(NB_SEEDS):
        print(f"\n=== Running optimization with seed {seed} ===")

        sim_options = SimulationOptions(
            on_rack=False,
            render=False,
            record_video=False,
            tracking_camera=False,
        )
        simulator = QuadSimulator(sim_options)

        objective = partial(evaluate_jumping, simulator=simulator)
        sampler = optuna.samplers.TPESampler(seed=seed)

        study = optuna.create_study(
            sampler=sampler,
            direction="maximize",
        )

        study.optimize(objective, n_trials=NB_TRIALS)
        simulator.close()

        # Collect values for aggregation
        trials = [t for t in study.get_trials() if t.value is not None]
        values = np.array([t.value for t in trials])
        all_values.append(values)

        # Save parameter trajectories
        param_keys = ["force_x", "force_y", "force_z", "frequence0", "frequence1"]
        param_history = {k: [] for k in param_keys}
        for t in trials:
            for k in param_keys:
                param_history[k].append(t.params[k])
        all_param_trajectories.append(param_history)

        # Save best parameters
        all_best_params.append(study.best_params)

        print(f"Best value (seed {seed}): {study.best_value:.3f}")

    # Compute and print mean optimal params
    print_mean_optimal_params(all_best_params)

    # Plot aggregated statistics
    plot_mean_std_across_seeds(all_values)
    plot_mean_std_parameters(all_param_trajectories)


def evaluate_jumping(trial: Trial, simulator: QuadSimulator):
    force_x = trial.suggest_float("force_x", 0, 0)
    force_y = trial.suggest_float("force_y", 0, 150)
    force_z = trial.suggest_float("force_z", 150, 350)
    freq0 = trial.suggest_float("frequence0", 0.75, 1.75)
    freq1 = trial.suggest_float("frequence1", 0.25, 0.25)

    simulator.reset()

    force_profile = FootForceProfile(
        f0=freq0, f1=freq1, Fx=force_x, Fy=force_y, Fz=force_z
    )

    jump_duration = force_profile.impulse_duration() + force_profile.idle_duration()
    n_steps = int(NB_JUMPS * jump_duration / simulator.options.timestep)

    start_pos = simulator.get_base_position().copy()
    fall = False

    for _ in range(n_steps):
        force_profile.step(simulator.options.timestep)

        tau = np.zeros(N_JOINTS * N_LEGS)
        tau += nominal_position(simulator, DES_EE_POS)
        tau += apply_force_profile(simulator, force_profile)
        tau += gravity_compensation(simulator)

        if any(simulator.get_foot_contacts()):
            tau += virtual_model(simulator)

        simulator.set_motor_targets(tau)
        simulator.step()

        roll, pitch, yaw = simulator.get_base_orientation_roll_pitch_yaw()
        if abs(roll) > np.deg2rad(30) or abs(pitch) > np.deg2rad(30):
            fall = True
            break

    last_pos = simulator.get_base_position()
    delta_x = last_pos[0] - start_pos[0]
    speed = delta_x / (NB_JUMPS * jump_duration)

    last_yaw = yaw 

    # Cost functions for different jump modes
    J_forward = last_pos[0]
    J_lateral = last_pos[1]
    J_twist = last_yaw
    J_fastest = speed

    return 0.0 if fall else last_yaw


def print_mean_optimal_params(all_best_params):
    keys = all_best_params[0].keys()
    mean_params = {}
    std_params = {}

    for k in keys:
        vals = np.array([p[k] for p in all_best_params])
        mean_params[k] = np.mean(vals)
        std_params[k] = np.std(vals)

    print("\n=== Mean optimal parameters across all seeds (at 50 trials) ===")
    for k in keys:
        print(f"{k:12s}: {mean_params[k]:8.3f} ± {std_params[k]:.3f}")


def plot_mean_std_across_seeds(all_values):
    max_len = max(len(v) for v in all_values)
    padded = np.full((len(all_values), max_len), np.nan)
    for i, v in enumerate(all_values):
        padded[i, : len(v)] = v

    mean_curve = np.nanmean(padded, axis=0)
    std_curve = np.nanstd(padded, axis=0)
    trials = np.arange(max_len)

    plt.figure(figsize=(6, 4))
    plt.plot(trials, mean_curve, color="purple", linewidth=2, label=r"$\bar{v_x}$ across seeds")
    plt.fill_between(
        trials,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="purple",
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )
    plt.xlabel("Trial number")
    plt.ylabel("$v_x$ (m/s)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("twist_controller_multiseed_speed.png", dpi=300)
    plt.show()


def plot_mean_std_parameters(all_param_trajectories):
    param_labels = {
        "force_x": r"$F_x$",
        "force_z": r"$F_z$",
        "force_y": r"$F_y$",
        "frequence0": r"$f_0$",
        "frequence1": r"$f_1$",
    }

    force_keys = ["force_x", "force_z", "force_y"]
    freq_keys = ["frequence0", "frequence1"]

    force_colors = ["tab:blue", "tab:orange", "tab:gray"]
    freq_colors = ["tab:green", "tab:red"]

    max_len = max(len(p[force_keys[0]]) for p in all_param_trajectories)
    trials = np.arange(max_len)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 4), sharex=True, constrained_layout=True
    )

    for k, color in zip(force_keys, force_colors):
        padded = np.full((len(all_param_trajectories), max_len), np.nan)
        for i, params in enumerate(all_param_trajectories):
            v = np.array(params[k])
            padded[i, : len(v)] = v

        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)

        ax1.plot(trials, mean_curve, linewidth=2, color=color, label=param_labels[k])
        ax1.fill_between(
            trials,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.2,
        )

    ax1.set_ylabel("Force amplitude [N]")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    for k, color in zip(freq_keys, freq_colors):
        padded = np.full((len(all_param_trajectories), max_len), np.nan)
        for i, params in enumerate(all_param_trajectories):
            v = np.array(params[k])
            padded[i, : len(v)] = v

        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)

        ax2.plot(trials, mean_curve, linewidth=2, color=color, label=param_labels[k])
        ax2.fill_between(
            trials,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.2,
        )

    ax2.set_xlabel("Trial number")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()

    plt.savefig("twist_controller_multiseed_params.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    multi_seed_optimization()