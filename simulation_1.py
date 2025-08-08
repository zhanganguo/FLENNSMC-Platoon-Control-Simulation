import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time


# --- 1. Simulation Configuration ---
class SimulationConfig:
    N_FOLLOWERS = 4
    T_START = 0.0
    T_END = 60.0
    DT = 0.05
    D_STAR = 20.0
    DELTA_COL = 10.0
    DELTA_CON = 30.0
    C_UPPER = DELTA_CON - D_STAR
    C_LOWER = D_STAR - DELTA_COL
    k_d = 4.0
    lambda_a = 2.0
    lambda_omega = 2.5
    K_s_a = 10.0
    K_s_omega = 10.0
    K_n_a = 2.0
    K_n_omega = 2.0
    SMC_SMOOTH_EPS = 0.05
    N_RULES = 3
    N_BASIS_PER_RULE = 5
    GAMMA_a = 50.0
    GAMMA_omega = 50.0
    SIGMA_a = 0.01
    SIGMA_omega = 0.01
    delta_a = 0.1
    delta_omega = 0.1
    MARKOV_LAMBDA = np.array([[-0.5, 0.5], [0.8, -0.8]])
    ETA_A_BAR = 0.5
    ETA_OMEGA_BAR = 0.5

    @staticmethod
    def rho_a_func(t):
        if 20 <= t < 40: return -0.7
        return 1.0

    @staticmethod
    def rho_omega_func(t):
        if 30 <= t < 50: return -0.8
        return 1.0


# --- 2. Controller Implementation (FLENNSMC) ---
class FLENNSMC_Controller:
    def __init__(self, config):
        self.cfg = config
        self.n_inputs = 2
        self.rbf_centers = {}
        self.rbf_widths = {}
        for s in ['a', 'omega']:
            self.rbf_centers[s] = [np.random.uniform(-1.5, 1.5, (config.N_BASIS_PER_RULE, self.n_inputs)) for _ in
                                   range(config.N_RULES)]
            self.rbf_widths[s] = [np.ones(config.N_BASIS_PER_RULE) * 2.0 for _ in range(config.N_RULES)]

    def _gaussian_mf(self, x, center, sigma=1.0):
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)

    def _rbf_basis(self, Z, centers, widths):
        return np.exp(-np.sum((Z - centers) ** 2, axis=1) / (widths ** 2))

    def _nussbaum(self, zeta):
        return zeta ** 2 * np.cos(zeta)

    def _smooth_sgn(self, x):
        return x / (np.abs(x) + self.cfg.SMC_SMOOTH_EPS)

    def compute_control_and_derivatives(self, i, errors, states, adaptive_params, t):
        cfg = self.cfg
        e_d, e_v, e_theta, e_phi = errors
        v_prev, phi_prev, theta_i = states['predecessor']
        W_hat_a, W_hat_omega, zeta_a, zeta_omega = adaptive_params
        s_a, s_omega = states['sliding_surfaces']

        if e_d >= 0:
            q_i = e_d / (cfg.C_UPPER ** 2 - e_d ** 2 + 1e-9)
        else:
            q_i = e_d / (cfg.C_LOWER ** 2 - e_d ** 2 + 1e-9)
        v_d = v_prev * np.cos(e_theta) + cfg.k_d * q_i

        Z_a = np.array([e_v, s_a])
        mf_vals_a = [self._gaussian_mf(e_v, c) for c in [-5, 0, 5]]
        h_a = np.array(mf_vals_a) / (np.sum(mf_vals_a) + 1e-9)
        F_hat_a_total = 0.0
        for j in range(cfg.N_RULES):
            xi_j = self._rbf_basis(Z_a, self.rbf_centers['a'][j], self.rbf_widths['a'][j])
            F_hat_a_total += h_a[j] * (W_hat_a[j] @ xi_j)
        tau_a = -F_hat_a_total - cfg.lambda_a * e_v - cfg.K_s_a * s_a \
                - cfg.K_n_a * self._smooth_sgn(s_a) - q_i
        u_c_a = self._nussbaum(zeta_a) * tau_a

        Z_omega = np.array([e_phi, s_omega])
        mf_vals_omega = [self._gaussian_mf(e_phi, c) for c in [-np.pi / 4, 0, np.pi / 4]]
        h_omega = np.array(mf_vals_omega) / (np.sum(mf_vals_omega) + 1e-9)
        F_hat_omega_total = 0.0
        for j in range(cfg.N_RULES):
            xi_j = self._rbf_basis(Z_omega, self.rbf_centers['omega'][j], self.rbf_widths['omega'][j])
            F_hat_omega_total += h_omega[j] * (W_hat_omega[j] @ xi_j)
        tau_omega = -F_hat_omega_total - cfg.lambda_omega * e_phi - cfg.K_s_omega * s_omega \
                    - cfg.K_n_omega * self._smooth_sgn(s_omega)
        u_c_omega = self._nussbaum(zeta_omega) * tau_omega

        dW_hat_a_dt = []
        for j in range(cfg.N_RULES):
            xi_j = self._rbf_basis(Z_a, self.rbf_centers['a'][j], self.rbf_widths['a'][j])
            update = cfg.GAMMA_a * h_a[j] * xi_j * s_a - cfg.SIGMA_a * cfg.GAMMA_a * W_hat_a[j]
            dW_hat_a_dt.append(update)
        dW_hat_omega_dt = []
        for j in range(cfg.N_RULES):
            xi_j = self._rbf_basis(Z_omega, self.rbf_centers['omega'][j], self.rbf_widths['omega'][j])
            update = cfg.GAMMA_omega * h_omega[j] * xi_j * s_omega - cfg.SIGMA_omega * cfg.GAMMA_omega * W_hat_omega[j]
            dW_hat_omega_dt.append(update)

        dzeta_a_dt = cfg.delta_a * s_a * tau_a
        dzeta_omega_dt = cfg.delta_omega * s_omega * tau_omega

        return {"u_c_a": u_c_a, "u_c_omega": u_c_omega, "dW_hat_a_dt": np.array(dW_hat_a_dt),
                "dW_hat_omega_dt": np.array(dW_hat_omega_dt), "dzeta_a_dt": dzeta_a_dt,
                "dzeta_omega_dt": dzeta_omega_dt}


# --- 3. System Dynamics and Simulation Environment ---
class PlatoonSimulation:
    def __init__(self, config, controller):
        self.cfg = config
        self.controller = controller
        self.w_size = config.N_RULES * config.N_BASIS_PER_RULE
        self.vehicle_state_size = 6 + 2 * self.w_size + 2
        self.total_states = config.N_FOLLOWERS * self.vehicle_state_size
        self.current_mode = 0
        self.last_jump_time = 0.0
        self.mode_history = [(0.0, 0)]

    def get_leader_velocity(self, t):
        _, v, _ = self._leader_trajectory(t)
        return v

    def _leader_trajectory(self, t):
        if t < 15:
            x, y, v, phi = 15 * t, 5.0, 15.0, 0.0
        elif t < 25:
            tm = t - 15
            x = 15 * 15 + 15 * tm
            y = 5.0 - 2.5 * (1 - np.cos(np.pi * tm / 10))
            v = 15.0
            phi = -np.arctan2(2.5 * (np.pi / 10) * np.sin(np.pi * tm / 10), 15)
        else:
            x, y, v, phi = 15 * 25 + 15 * (t - 25), 0.0, 15.0, 0.0
        return np.array([x, y]), v, phi

    def _unpack_state_vector(self, y_flat):
        states = {}
        for i in range(self.cfg.N_FOLLOWERS):
            start = i * self.vehicle_state_size
            vehicle_y = y_flat[start: start + self.vehicle_state_size]
            s = {}
            s['kinematics'] = vehicle_y[0:4]
            s['s_integrals'] = vehicle_y[4:6]
            w_start = 6
            w_end_a = w_start + self.w_size
            w_end_omega = w_end_a + self.w_size
            s['W_hat_a'] = vehicle_y[w_start:w_end_a].reshape(self.cfg.N_RULES, self.cfg.N_BASIS_PER_RULE)
            s['W_hat_omega'] = vehicle_y[w_end_a:w_end_omega].reshape(self.cfg.N_RULES, self.cfg.N_BASIS_PER_RULE)
            s['zeta'] = vehicle_y[w_end_omega: w_end_omega + 2]
            states[i] = s
        return states

    def _unmodeled_dynamics(self, v, phi, t, mode):
        F_a = mode * 0.5 * np.sin(v) + 0.2 * np.cos(t) - 0.1 * v
        F_omega = mode * 0.3 * np.cos(phi) + 0.1 * np.sin(2 * t)
        return F_a, F_omega

    def dynamics(self, t, y_flat):
        cfg = self.cfg
        if t > self.last_jump_time:
            time_in_mode = t - self.last_jump_time
            rate_out = -cfg.MARKOV_LAMBDA[self.current_mode, self.current_mode]
            if rate_out > 0 and np.random.exponential(scale=1.0 / rate_out) < time_in_mode:
                rates = cfg.MARKOV_LAMBDA[self.current_mode, :].copy()
                rates[self.current_mode] = 0
                probs = rates / np.sum(rates) if np.sum(rates) > 0 else np.array(
                    [0, 1] if self.current_mode == 0 else [1, 0])
                self.current_mode = np.random.choice(len(rates), p=probs)
                self.last_jump_time = t
                self.mode_history.append((t, self.current_mode))

        all_states = self._unpack_state_vector(y_flat)
        leader_pos, leader_vel, leader_phi = self._leader_trajectory(t)
        dydt = np.zeros_like(y_flat)

        for i in range(cfg.N_FOLLOWERS):
            vehicle_i_state = all_states[i]
            x_i, y_i, v_i, phi_i = vehicle_i_state['kinematics']
            s_a_int, s_omega_int = vehicle_i_state['s_integrals']
            W_hat_a, W_hat_omega = vehicle_i_state['W_hat_a'], vehicle_i_state['W_hat_omega']
            zeta_a, zeta_omega = vehicle_i_state['zeta']
            if i == 0:
                x_prev, y_prev, v_prev, phi_prev = leader_pos[0], leader_pos[1], leader_vel, leader_phi
            else:
                x_prev, y_prev, v_prev, phi_prev = all_states[i - 1]['kinematics']

            d_i = np.sqrt((x_prev - x_i) ** 2 + (y_prev - y_i) ** 2)
            theta_i = np.arctan2(y_prev - y_i, x_prev - x_i)
            e_d = d_i - cfg.D_STAR
            e_theta = theta_i - phi_prev

            if e_d >= 0:
                q_i_temp = e_d / (cfg.C_UPPER ** 2 - e_d ** 2 + 1e-9)
            else:
                q_i_temp = e_d / (cfg.C_LOWER ** 2 - e_d ** 2 + 1e-9)
            v_d_temp = v_prev * np.cos(e_theta) + cfg.k_d * q_i_temp
            e_v = v_i - v_d_temp
            e_phi = phi_i - theta_i

            s_a = e_v + cfg.lambda_a * s_a_int
            s_omega = e_phi + cfg.lambda_omega * s_omega_int

            errors = (e_d, e_v, e_theta, e_phi)
            states = {'self': (x_i, y_i, v_i, phi_i), 'predecessor': (v_prev, phi_prev, theta_i),
                      'sliding_surfaces': (s_a, s_omega)}
            adaptive_params = (W_hat_a, W_hat_omega, zeta_a, zeta_omega)
            ctrl_out = self.controller.compute_control_and_derivatives(i, errors, states, adaptive_params, t)

            u_c_a, u_c_omega = ctrl_out['u_c_a'], ctrl_out['u_c_omega']
            F_a_true, F_omega_true = self._unmodeled_dynamics(v_i, phi_i, t, self.current_mode + 1)  # Mode is 1 or 2
            eta_a = np.random.uniform(-cfg.ETA_A_BAR, cfg.ETA_A_BAR)
            eta_omega = np.random.uniform(-cfg.ETA_OMEGA_BAR, cfg.ETA_OMEGA_BAR)
            a_i = F_a_true + cfg.rho_a_func(t) * u_c_a + eta_a
            omega_i = F_omega_true + cfg.rho_omega_func(t) * u_c_omega + eta_omega

            dkinematics = np.array([v_i * np.cos(phi_i), v_i * np.sin(phi_i), a_i, omega_i, e_v, e_phi])
            dW_a_flat = ctrl_out['dW_hat_a_dt'].flatten()
            dW_omega_flat = ctrl_out['dW_hat_omega_dt'].flatten()
            dzeta = np.array([ctrl_out['dzeta_a_dt'], ctrl_out['dzeta_omega_dt']])

            start_idx = i * self.vehicle_state_size
            dydt[start_idx:start_idx + 6] = dkinematics
            dydt[start_idx + 6: start_idx + 6 + self.w_size] = dW_a_flat
            dydt[start_idx + 6 + self.w_size: start_idx + 6 + 2 * self.w_size] = dW_omega_flat
            dydt[start_idx + 6 + 2 * self.w_size: start_idx + 6 + 2 * self.w_size + 2] = dzeta
        return dydt

    def run(self):
        y0_flat = np.zeros(self.total_states)
        leader_pos_0, leader_vel_0, _ = self._leader_trajectory(0)
        for i in range(self.cfg.N_FOLLOWERS):
            start_idx = i * self.vehicle_state_size
            y0_flat[start_idx] = leader_pos_0[0] - (i + 1) * self.cfg.D_STAR
            y0_flat[start_idx + 1] = leader_pos_0[1] + 5.0
            y0_flat[start_idx + 2] = leader_vel_0 - 2.0 * (i + 1)

        print("Starting simulation...")
        start_time = time.time()
        t_eval = np.arange(self.cfg.T_START, self.cfg.T_END, self.cfg.DT)
        sol = solve_ivp(fun=self.dynamics, t_span=[self.cfg.T_START, self.cfg.T_END], y0=y0_flat, t_eval=t_eval,
                        method='RK45')
        end_time = time.time()
        print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
        return sol, self.mode_history


# --- 4. Post-Processing and Plotting --
def process_and_plot(sol, config, controller, mode_history):
    print("Processing results for plotting...")
    cfg = config
    t = sol.t

    sim = PlatoonSimulation(cfg, controller)
    leader_traj_list = [sim._leader_trajectory(ti) for ti in t]
    leader_traj = np.array(leader_traj_list, dtype=object)
    leader_pos = np.array([p[0] for p in leader_traj])

    all_states = [sim._unpack_state_vector(sol.y[:, k]) for k in range(len(t))]

    # Expanded dictionary to store more results
    results = {i: {
        'pos': [], 'vel': [], 'e_d': [], 'e_v': [], 'e_phi': [],
        's_a': [], 's_omega': [], 'u_a': [], 'u_omega': [],
        'norm_W_a': [], 'norm_W_omega': [], 'zeta_a': [], 'zeta_omega': []
    } for i in range(cfg.N_FOLLOWERS)}

    for k, ti in enumerate(t):
        current_states = all_states[k]
        leader_p, leader_v, leader_phi = leader_traj[k]

        for i in range(cfg.N_FOLLOWERS):
            s_i = current_states[i]
            x_i, y_i, v_i, phi_i = s_i['kinematics']

            if i == 0:
                x_p, y_p, v_p, phi_p = leader_p[0], leader_p[1], leader_v, leader_phi
            else:
                x_p, y_p, v_p, phi_p = current_states[i - 1]['kinematics']

            d_i = np.sqrt((x_p - x_i) ** 2 + (y_p - y_i) ** 2)
            theta_i = np.arctan2(y_p - y_i, x_p - x_i)
            e_d = d_i - cfg.D_STAR
            e_theta = theta_i - phi_p

            if e_d >= 0:
                q_i_temp = e_d / (cfg.C_UPPER ** 2 - e_d ** 2 + 1e-9)
            else:
                q_i_temp = e_d / (cfg.C_LOWER ** 2 - e_d ** 2 + 1e-9)
            v_d = v_p * np.cos(e_theta) + cfg.k_d * q_i_temp

            e_v = v_i - v_d
            e_phi = phi_i - theta_i

            s_a_int, s_omega_int = s_i['s_integrals']
            s_a = e_v + cfg.lambda_a * s_a_int
            s_omega = e_phi + cfg.lambda_omega * s_omega_int

            # --- Store all desired values ---
            results[i]['pos'].append([x_i, y_i])
            results[i]['vel'].append(v_i)
            results[i]['e_d'].append(e_d)
            results[i]['e_v'].append(e_v)
            results[i]['e_phi'].append(e_phi)
            results[i]['s_a'].append(s_a)
            results[i]['s_omega'].append(s_omega)

            # Log control inputs by re-computing them (for simplicity)
            ctrl_out = controller.compute_control_and_derivatives(
                i, (e_d, e_v, e_theta, e_phi),
                {'self': (x_i, y_i, v_i, phi_i), 'predecessor': (v_p, phi_p, theta_i),
                 'sliding_surfaces': (s_a, s_omega)},
                (s_i['W_hat_a'], s_i['W_hat_omega'], s_i['zeta'][0], s_i['zeta'][1]), ti
            )
            results[i]['u_a'].append(ctrl_out['u_c_a'])
            results[i]['u_omega'].append(ctrl_out['u_c_omega'])

            # Log adaptive parameter norms and zeta values
            results[i]['norm_W_a'].append(np.linalg.norm(s_i['W_hat_a']))
            results[i]['norm_W_omega'].append(np.linalg.norm(s_i['W_hat_omega']))
            results[i]['zeta_a'].append(s_i['zeta'][0])
            results[i]['zeta_omega'].append(s_i['zeta'][1])

    for i in range(cfg.N_FOLLOWERS):
        for key in results[i]:
            results[i][key] = np.array(results[i][key])

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    follower_colors = plt.cm.get_cmap('tab10').colors

    def create_plot(title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True)
        return fig, ax

    # Plot 1 (Part A): X Position over Time
    fig1a, ax1a = create_plot('Platoon X Position vs. Time', 'Time (s)', 'X Position (m)')
    ax1a.plot(t, leader_pos[:, 0], 'k--', label='Leader', lw=2)
    for i in range(cfg.N_FOLLOWERS):
        ax1a.plot(t, results[i]['pos'][:, 0], color=follower_colors[i], label=f'Follower {i + 1}')
    ax1a.legend()
    fig1a.tight_layout()
    plt.show()

    # Plot 1 (Part B): Y Position over Time
    fig1b, ax1b = create_plot('Platoon Y Position vs. Time', 'Time (s)', 'Y Position (m)')
    ax1b.plot(t, leader_pos[:, 1], 'k--', label='Leader', lw=2)
    for i in range(cfg.N_FOLLOWERS):
        ax1b.plot(t, results[i]['pos'][:, 1], color=follower_colors[i], label=f'Follower {i + 1}')
    ax1b.legend()
    fig1b.tight_layout()
    plt.show()

    # Plot 2: Distance Error
    fig2, ax2 = create_plot('Inter-vehicle Distance Error $e_d(t)$', 'Time (s)', 'Distance Error (m)')
    for i in range(cfg.N_FOLLOWERS):
        ax2.plot(t, results[i]['e_d'], color=follower_colors[i], label=f'Follower {i + 1}')
    ax2.axhline(y=cfg.C_UPPER, color='k', ls=':', lw=2, label='Constraint Boundary')
    ax2.axhline(y=-cfg.C_LOWER, color='k', ls=':', lw=2)
    ax2.fill_between(t, -cfg.C_LOWER, cfg.C_UPPER, color='gray', alpha=0.2, label='Safe Zone')
    ax2.legend()
    fig2.tight_layout()
    plt.show()

    # Plot 3: Velocity Error
    fig3, ax3 = create_plot('Velocity Tracking Error $e_v(t)$', 'Time (s)', 'Velocity Error (m/s)')
    for i in range(cfg.N_FOLLOWERS):
        ax3.plot(t, results[i]['e_v'], color=follower_colors[i], label=f'Follower {i + 1}')
    ax3.legend()
    fig3.tight_layout()
    plt.show()

    # Plot 4: Velocity Consensus
    fig4, ax4 = create_plot('Velocity Consensus of the Platoon', 'Time (s)', 'Velocity (m/s)')
    leader_vels = np.array([sim.get_leader_velocity(ti) for ti in t])
    ax4.plot(t, leader_vels, 'k--', label='Leader Velocity', lw=2.5)
    for i in range(cfg.N_FOLLOWERS):
        ax4.plot(t, results[i]['vel'], color=follower_colors[i], label=f'Follower {i + 1} Velocity', lw=1.5)
    ax4.legend()
    fig4.tight_layout()
    plt.show()

    # NEW PLOT 5: Heading Angle Error
    fig5, ax5 = create_plot('Heading Angle Error $e_\\phi(t)$', 'Time (s)', 'Heading Error (rad)')
    for i in range(cfg.N_FOLLOWERS):
        ax5.plot(t, results[i]['e_phi'], color=follower_colors[i], label=f'Follower {i + 1}')
    ax5.legend()
    fig5.tight_layout()
    plt.show()

    # NEW PLOT 6: Sliding Surfaces
    fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig6.suptitle('Sliding Mode Surfaces', fontsize=16)
    for i in range(cfg.N_FOLLOWERS):
        ax6a.plot(t, results[i]['s_a'], color=follower_colors[i], label=f'Follower {i + 1}')
        ax6b.plot(t, results[i]['s_omega'], color=follower_colors[i], label=f'Follower {i + 1}')
    ax6a.set_ylabel('Sliding Surface $s_a(t)$', fontsize=14)
    ax6a.grid(True)
    ax6a.legend()
    ax6b.set_ylabel('Sliding Surface $s_\\omega(t)$', fontsize=14)
    ax6b.grid(True)
    ax6b.legend()
    ax6b.set_xlabel('Time (s)', fontsize=14)
    fig6.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # NEW PLOT 7: RBFNN Weight Norms
    fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig7.suptitle('Norm of FLENN Weights $||\\hat{W}||$', fontsize=16)
    for i in range(cfg.N_FOLLOWERS):
        ax7a.plot(t, results[i]['norm_W_a'], color=follower_colors[i], label=f'Follower {i + 1}')
        ax7b.plot(t, results[i]['norm_W_omega'], color=follower_colors[i], label=f'Follower {i + 1}')
    ax7a.set_ylabel('Norm $||\\hat{W}_a||$', fontsize=14)
    ax7a.grid(True)
    ax7a.legend()
    ax7b.set_ylabel('Norm $||\\hat{W}_\\omega||$', fontsize=14)
    ax7b.grid(True)
    ax7b.legend()
    ax7b.set_xlabel('Time (s)', fontsize=14)
    fig7.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # NEW PLOT 8: Nussbaum Gains
    fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig8.suptitle('Nussbaum Function Gains $\\zeta(t)$', fontsize=16)
    for i in range(cfg.N_FOLLOWERS):
        ax8a.plot(t, results[i]['zeta_a'], color=follower_colors[i], label=f'Follower {i + 1}')
        ax8b.plot(t, results[i]['zeta_omega'], color=follower_colors[i], label=f'Follower {i + 1}')
    ax8a.set_ylabel('Gain $\\zeta_a(t)$', fontsize=14)
    ax8a.grid(True)
    ax8a.legend()
    ax8b.set_ylabel('Gain $\\zeta_\\omega(t)$', fontsize=14)
    ax8b.grid(True)
    ax8b.legend()
    ax8b.set_xlabel('Time (s)', fontsize=14)
    fig8.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Plot 9: Control Input
    fig9, ax9 = create_plot('Control Input $u_{i,a}(t)$', 'Time (s)', 'Acceleration Control Input')
    for i in range(cfg.N_FOLLOWERS):
        ax9.plot(t, results[i]['u_a'], color=follower_colors[i], label=f'Follower {i + 1}')
    fault_patch = mpatches.Patch(color='red', alpha=0.2, label='Actuator Fault Active (t=[20,40))')
    ax9.axvspan(20, 40, color='red', alpha=0.2)
    handles, labels = ax9.get_legend_handles_labels()
    handles.append(fault_patch)
    ax9.legend(handles=handles)
    fig9.tight_layout()
    plt.show()

    # Plot 10: Markov Jumps
    fig10, ax10 = create_plot('Stochastic System Mode (Markov Jumps)', 'Time (s)', 'Mode')
    if mode_history:
        mode_times, modes = zip(*mode_history)
        plot_times = np.append(mode_times, t[-1])
        plot_modes = np.append(modes, modes[-1])
        ax10.step(plot_times, plot_modes, where='post', label='System Mode')
    ax10.set_yticks([0, 1])
    ax10.set_yticklabels(['Mode 1', 'Mode 2'], fontsize=14)
    ax10.set_ylim(-0.2, 1.2)
    fig10.tight_layout()
    plt.show()


# --- 5. Main Execution Block ---
if __name__ == '__main__':
    config = SimulationConfig()
    controller = FLENNSMC_Controller(config)
    simulation = PlatoonSimulation(config, controller)
    solution, mode_hist = simulation.run()
    process_and_plot(solution, config, controller, mode_hist)