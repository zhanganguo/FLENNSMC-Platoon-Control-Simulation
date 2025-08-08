import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import time


# --- 1. Simulation Configuration ---
class SimulationConfig:
    """Holds all simulation and controller parameters for easy tuning."""
    # --- General Simulation Parameters ---
    N_FOLLOWERS = 4
    T_START = 0.0
    T_END = 60.0
    DT = 0.05

    # --- Platoon & Constraint Parameters ---
    D_STAR = 20.0  # Desired spacing
    DELTA_COL = 10.0  # Collision distance from center
    DELTA_CON = 30.0  # Communication distance from center
    C_UPPER = DELTA_CON - D_STAR  # Upper bound for distance error
    C_LOWER = D_STAR - DELTA_COL  # Lower bound for distance error (magnitude)

    # --- System Dynamics & Uncertainties ---
    MARKOV_LAMBDA = np.array([[-0.5, 0.5], [0.8, -0.8]])  # Transition rates for Markov jumps
    ETA_A_BAR = 0.5  # Upper bound of disturbance on acceleration
    ETA_OMEGA_BAR = 0.5  # Upper bound of disturbance on angular rate

    @staticmethod
    def rho_a_func(t):
        """Time-varying unknown-direction fault for acceleration."""
        if 20 <= t < 40: return -0.7
        return 1.0

    @staticmethod
    def rho_omega_func(t):
        """Time-varying unknown-direction fault for angular rate."""
        if 30 <= t < 50: return -0.8
        return 1.0

    # --- FLENNSMC & RBFNN Controller Parameters ---
    k_d = 4.0  # Virtual control gain for distance error
    lambda_a = 2.0  # Sliding surface parameter for acceleration
    lambda_omega = 2.5  # Sliding surface parameter for angular rate
    K_s_a = 10.0  # SMC gain (sliding term) for acceleration
    K_s_omega = 10.0  # SMC gain (sliding term) for angular rate
    K_n_a = 2.0  # SMC gain (robust term) for acceleration
    K_n_omega = 2.0  # SMC gain (robust term) for angular rate
    SMC_SMOOTH_EPS = 0.05  # Smoothing factor for sgn function to avoid chattering
    N_RULES = 3  # Number of fuzzy rules
    N_BASIS_PER_RULE = 5  # RBF neurons per fuzzy rule
    GAMMA_a = 50.0  # Learning rate for acceleration weights
    GAMMA_omega = 50.0  # Learning rate for angular rate weights
    SIGMA_a = 0.01  # Leakage term for W_hat_a
    SIGMA_omega = 0.01  # Leakage term for W_hat_omega
    delta_a = 0.1  # Learning rate for Nussbaum function (acceleration)
    delta_omega = 0.1  # Learning rate for Nussbaum function (angular rate)
    # For fair comparison, total RBF neurons are the same for FLENNSMC and Traditional RBFNN
    N_BASIS_TRADITIONAL = N_RULES * N_BASIS_PER_RULE

    # --- PID Controller Gains ---
    PID_Kp_a = 5.0
    PID_Ki_a = 0.5
    PID_Kp_omega = 6.0
    PID_Ki_omega = 0.8

    # --- SMC Controller Gains ---
    SMC_K_a = 15.0  # Switching gain for acceleration SMC
    SMC_K_omega = 15.0  # Switching gain for angular rate SMC


# --- 2. Controller Implementations ---

class BaseController:
    """Base class for controllers to share common methods and structure."""

    def __init__(self, config):
        self.cfg = config
        # Total number of weights for the most complex controller, used for state vector consistency
        self.num_weights = self.cfg.N_BASIS_TRADITIONAL

    def _nussbaum(self, zeta):
        """Standard Nussbaum function for handling unknown control direction."""
        return zeta ** 2 * np.cos(zeta)

    def _smooth_sgn(self, x):
        """A smooth approximation of the signum function to reduce chattering."""
        return x / (np.abs(x) + self.cfg.SMC_SMOOTH_EPS)

    def compute_control_and_derivatives(self, i, errors, states, adaptive_params, t):
        """Placeholder for the main control computation method."""
        raise NotImplementedError


class PID_Controller(BaseController):
    """A standard Proportional-Integral-Derivative (PID) controller."""

    def compute_control_and_derivatives(self, i, errors, states, adaptive_params, t):
        cfg = self.cfg
        e_d, e_v, e_theta, e_phi = errors
        # Repurpose the sliding surface integrals as the integral terms for the PI controller
        integral_e_v, integral_e_phi = states['s_integrals']

        # Acceleration control (PI on velocity error e_v)
        # The negative sign is to ensure the control action opposes the error
        u_c_a = -(cfg.PID_Kp_a * e_v + cfg.PID_Ki_a * integral_e_v)

        # Angular rate control (PI on heading error e_phi)
        u_c_omega = -(cfg.PID_Kp_omega * e_phi + cfg.PID_Ki_omega * integral_e_phi)

        # PID has no adaptive parameters, so return zero arrays for the derivatives
        return {
            "u_c_a": u_c_a,
            "u_c_omega": u_c_omega,
            "dW_hat_a_dt": np.zeros(self.num_weights),
            "dW_hat_omega_dt": np.zeros(self.num_weights),
            "dzeta_a_dt": 0.0,
            "dzeta_omega_dt": 0.0
        }


class SMC_Controller(BaseController):
    """A standard Sliding Mode Controller (SMC)."""

    def compute_control_and_derivatives(self, i, errors, states, adaptive_params, t):
        cfg = self.cfg
        s_a, s_omega = states['sliding_surfaces']

        # The control law is a simple, robust switching term to drive the sliding surfaces to zero.
        # This form does not require knowledge of the system model (equivalent control).
        u_c_a = -cfg.SMC_K_a * self._smooth_sgn(s_a)
        u_c_omega = -cfg.SMC_K_omega * self._smooth_sgn(s_omega)

        # SMC has no adaptive parameters, so return zero arrays for the derivatives
        return {
            "u_c_a": u_c_a,
            "u_c_omega": u_c_omega,
            "dW_hat_a_dt": np.zeros(self.num_weights),
            "dW_hat_omega_dt": np.zeros(self.num_weights),
            "dzeta_a_dt": 0.0,
            "dzeta_omega_dt": 0.0
        }


class FLENNSMC_Controller(BaseController):
    """The proposed Fuzzy Logic-Enhanced Neuroadaptive Sliding Mode Controller."""

    def __init__(self, config):
        super().__init__(config)
        self.n_inputs = 2  # Inputs to RBFNN are [error, sliding_surface]
        self.rbf_centers = {}
        self.rbf_widths = {}
        # Initialize RBF centers and widths for each fuzzy rule
        for s in ['a', 'omega']:
            self.rbf_centers[s] = [np.random.uniform(-1.5, 1.5, (config.N_BASIS_PER_RULE, self.n_inputs)) for _ in
                                   range(config.N_RULES)]
            self.rbf_widths[s] = [np.ones(config.N_BASIS_PER_RULE) * 2.0 for _ in range(config.N_RULES)]

    def _gaussian_mf(self, x, center, sigma=1.0):
        """Gaussian membership function for fuzzy logic."""
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)

    def _rbf_basis(self, Z, centers, widths):
        """Radial Basis Function neural network basis functions."""
        return np.exp(-np.sum((Z[:, None] - centers.T) ** 2, axis=0) / (widths ** 2))

    def compute_control_and_derivatives(self, i, errors, states, adaptive_params, t):
        cfg, n_inputs = self.cfg, self.n_inputs
        e_d, e_v, e_theta, e_phi = errors
        W_hat_a, W_hat_omega, zeta_a, zeta_omega = adaptive_params
        s_a, s_omega = states['sliding_surfaces']

        # --- Acceleration Control ---
        Z_a = np.array([e_v, s_a])
        # Fuzzy inference: calculate firing strength of each rule
        mf_vals_a = [self._gaussian_mf(e_v, c) for c in [-5, 0, 5]]
        h_a = np.array(mf_vals_a) / (np.sum(mf_vals_a) + 1e-9)
        # Weighted sum of RBFNN outputs from each rule
        F_hat_a_total = 0.0
        for j in range(cfg.N_RULES):
            xi_j = self._rbf_basis(Z_a, self.rbf_centers['a'][j], self.rbf_widths['a'][j])
            F_hat_a_total += h_a[j] * (W_hat_a[j] @ xi_j)

        # Calculate Barrier Lyapunov Function term
        q_i = e_d / (cfg.C_UPPER ** 2 - e_d ** 2 + 1e-9) if e_d >= 0 else e_d / (cfg.C_LOWER ** 2 - e_d ** 2 + 1e-9)

        tau_a = -F_hat_a_total - cfg.lambda_a * e_v - cfg.K_s_a * s_a - cfg.K_n_a * self._smooth_sgn(s_a) - q_i
        u_c_a = self._nussbaum(zeta_a) * tau_a

        # --- Angular Rate Control ---
        Z_omega = np.array([e_phi, s_omega])
        mf_vals_omega = [self._gaussian_mf(e_phi, c) for c in [-np.pi / 4, 0, np.pi / 4]]
        h_omega = np.array(mf_vals_omega) / (np.sum(mf_vals_omega) + 1e-9)
        F_hat_omega_total = 0.0
        for j in range(cfg.N_RULES):
            xi_j = self._rbf_basis(Z_omega, self.rbf_centers['omega'][j], self.rbf_widths['omega'][j])
            F_hat_omega_total += h_omega[j] * (W_hat_omega[j] @ xi_j)
        tau_omega = -F_hat_omega_total - cfg.lambda_omega * e_phi - cfg.K_s_omega * s_omega - cfg.K_n_omega * self._smooth_sgn(
            s_omega)
        u_c_omega = self._nussbaum(zeta_omega) * tau_omega

        # --- Adaptive Law Derivatives ---
        dW_hat_a_dt, dW_hat_omega_dt = [], []
        for j in range(cfg.N_RULES):
            xi_a_j = self._rbf_basis(Z_a, self.rbf_centers['a'][j], self.rbf_widths['a'][j])
            dW_hat_a_dt.append(cfg.GAMMA_a * h_a[j] * xi_a_j * s_a - cfg.SIGMA_a * cfg.GAMMA_a * W_hat_a[j])
            xi_omega_j = self._rbf_basis(Z_omega, self.rbf_centers['omega'][j], self.rbf_widths['omega'][j])
            dW_hat_omega_dt.append(
                cfg.GAMMA_omega * h_omega[j] * xi_omega_j * s_omega - cfg.SIGMA_omega * cfg.GAMMA_omega * W_hat_omega[
                    j])

        dzeta_a_dt = cfg.delta_a * s_a * tau_a
        dzeta_omega_dt = cfg.delta_omega * s_omega * tau_omega

        return {
            "u_c_a": u_c_a, "u_c_omega": u_c_omega,
            "dW_hat_a_dt": np.array(dW_hat_a_dt), "dW_hat_omega_dt": np.array(dW_hat_omega_dt),
            "dzeta_a_dt": dzeta_a_dt, "dzeta_omega_dt": dzeta_omega_dt
        }


class Traditional_RBFNN_Controller(FLENNSMC_Controller):
    """A traditional (monolithic) RBFNN-based adaptive controller for comparison."""

    def __init__(self, config):
        super().__init__(config)
        # Override RBF centers for a single, large RBFNN
        for s in ['a', 'omega']:
            self.rbf_centers[s] = np.random.uniform(-1.5, 1.5, (config.N_BASIS_TRADITIONAL, self.n_inputs))
            self.rbf_widths[s] = np.ones(config.N_BASIS_TRADITIONAL) * 2.0

    def compute_control_and_derivatives(self, i, errors, states, adaptive_params, t):
        cfg = self.cfg
        e_d, e_v, e_theta, e_phi = errors
        W_hat_a, W_hat_omega, zeta_a, zeta_omega = adaptive_params
        s_a, s_omega = states['sliding_surfaces']

        # BLF term
        q_i = e_d / (cfg.C_UPPER ** 2 - e_d ** 2 + 1e-9) if e_d >= 0 else e_d / (cfg.C_LOWER ** 2 - e_d ** 2 + 1e-9)

        # --- Acceleration Control (Monolithic RBFNN) ---
        Z_a = np.array([e_v, s_a])
        xi_a = self._rbf_basis(Z_a, self.rbf_centers['a'], self.rbf_widths['a'])
        F_hat_a_total = W_hat_a @ xi_a
        tau_a = -F_hat_a_total - cfg.lambda_a * e_v - cfg.K_s_a * s_a - cfg.K_n_a * self._smooth_sgn(s_a) - q_i
        u_c_a = self._nussbaum(zeta_a) * tau_a

        # --- Angular Rate Control (Monolithic RBFNN) ---
        Z_omega = np.array([e_phi, s_omega])
        xi_omega = self._rbf_basis(Z_omega, self.rbf_centers['omega'], self.rbf_widths['omega'])
        F_hat_omega_total = W_hat_omega @ xi_omega
        tau_omega = -F_hat_omega_total - cfg.lambda_omega * e_phi - cfg.K_s_omega * s_omega - cfg.K_n_omega * self._smooth_sgn(
            s_omega)
        u_c_omega = self._nussbaum(zeta_omega) * tau_omega

        # --- Adaptive Law Derivatives ---
        dW_hat_a_dt = cfg.GAMMA_a * xi_a * s_a - cfg.SIGMA_a * cfg.GAMMA_a * W_hat_a
        dW_hat_omega_dt = cfg.GAMMA_omega * xi_omega * s_omega - cfg.SIGMA_omega * cfg.GAMMA_omega * W_hat_omega
        dzeta_a_dt = cfg.delta_a * s_a * tau_a
        dzeta_omega_dt = cfg.delta_omega * s_omega * tau_omega

        return {
            "u_c_a": u_c_a, "u_c_omega": u_c_omega,
            "dW_hat_a_dt": dW_hat_a_dt, "dW_hat_omega_dt": dW_hat_omega_dt,
            "dzeta_a_dt": dzeta_a_dt, "dzeta_omega_dt": dzeta_omega_dt
        }


# --- 3. System Dynamics and Simulation Environment ---
class PlatoonSimulation:
    def __init__(self, config, controller):
        self.cfg = config
        self.controller = controller
        # Use a fixed state vector size based on the most complex controller for consistency
        self.num_weights = config.N_BASIS_TRADITIONAL
        # State: [kinematics(4), s_integrals(2), W_hat_a(n), W_hat_omega(n), zeta(2)]
        self.vehicle_state_size = 6 + self.num_weights + self.num_weights + 2
        self.total_states = config.N_FOLLOWERS * self.vehicle_state_size
        self.current_mode = 0
        self.last_jump_time = 0.0
        self.mode_history = [(0.0, 0)]

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
            vehicle_y = y_flat[start:start + self.vehicle_state_size]
            s = {'kinematics': vehicle_y[0:4], 's_integrals': vehicle_y[4:6]}
            w_start, w_end_a = 6, 6 + self.num_weights
            w_end_omega = w_end_a + self.num_weights
            W_a_flat, W_omega_flat = vehicle_y[w_start:w_end_a], vehicle_y[w_end_a:w_end_omega]
            # Reshape weights only for the fuzzy controller, others use flat or ignore
            if isinstance(self.controller, FLENNSMC_Controller) and not isinstance(self.controller,
                                                                                   Traditional_RBFNN_Controller):
                s['W_hat_a'] = W_a_flat.reshape(self.cfg.N_RULES, self.cfg.N_BASIS_PER_RULE)
                s['W_hat_omega'] = W_omega_flat.reshape(self.cfg.N_RULES, self.cfg.N_BASIS_PER_RULE)
            else:
                s['W_hat_a'] = W_a_flat
                s['W_hat_omega'] = W_omega_flat
            s['zeta'] = vehicle_y[w_end_omega:w_end_omega + 2]
            states[i] = s
        return states

    def _unmodeled_dynamics(self, v, phi, t, mode):
        F_a = mode * 0.5 * np.sin(v) + 0.2 * np.cos(t) - 0.1 * v
        F_omega = mode * 0.3 * np.cos(phi) + 0.1 * np.sin(2 * t)
        return F_a, F_omega

    def dynamics(self, t, y_flat):
        cfg = self.cfg
        # Markov jump process simulation
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
            W_hat_a, W_hat_omega, zeta_a, zeta_omega = vehicle_i_state['W_hat_a'], vehicle_i_state['W_hat_omega'], \
            vehicle_i_state['zeta'][0], vehicle_i_state['zeta'][1]

            predecessor_state = all_states[i - 1]['kinematics'] if i > 0 else (
            leader_pos[0], leader_pos[1], leader_vel, leader_phi)
            x_prev, y_prev, v_prev, phi_prev = predecessor_state

            # Calculate tracking errors
            d_i = np.sqrt((x_prev - x_i) ** 2 + (y_prev - y_i) ** 2)
            theta_i = np.arctan2(y_prev - y_i, x_prev - x_i)
            e_d = d_i - cfg.D_STAR
            e_theta = theta_i - phi_prev

            # Virtual control and velocity error
            q_i_temp = e_d / (cfg.C_UPPER ** 2 - e_d ** 2 + 1e-9) if e_d >= 0 else e_d / (
                        cfg.C_LOWER ** 2 - e_d ** 2 + 1e-9)
            v_d_temp = v_prev * np.cos(e_theta) + cfg.k_d * q_i_temp
            e_v = v_i - v_d_temp
            e_phi = phi_i - theta_i

            # Sliding surfaces
            s_a = e_v + cfg.lambda_a * s_a_int
            s_omega = e_phi + cfg.lambda_omega * s_omega_int

            # Prepare inputs for the controller
            errors = (e_d, e_v, e_theta, e_phi)
            states = {'predecessor': (v_prev, phi_prev, theta_i), 'sliding_surfaces': (s_a, s_omega),
                      's_integrals': (s_a_int, s_omega_int)}
            adaptive_params = (W_hat_a, W_hat_omega, zeta_a, zeta_omega)

            # Get control action and adaptive law derivatives
            ctrl_out = self.controller.compute_control_and_derivatives(i, errors, states, adaptive_params, t)
            u_c_a, u_c_omega = ctrl_out['u_c_a'], ctrl_out['u_c_omega']

            # Apply control to system dynamics
            F_a_true, F_omega_true = self._unmodeled_dynamics(v_i, phi_i, t, self.current_mode)
            eta_a, eta_omega = np.random.uniform(-cfg.ETA_A_BAR, cfg.ETA_A_BAR), np.random.uniform(-cfg.ETA_OMEGA_BAR,
                                                                                                   cfg.ETA_OMEGA_BAR)
            a_i = F_a_true + cfg.rho_a_func(t) * u_c_a + eta_a
            omega_i = F_omega_true + cfg.rho_omega_func(t) * u_c_omega + eta_omega

            # Pack derivatives into the state vector derivative
            start_idx = i * self.vehicle_state_size
            dydt[start_idx:start_idx + 6] = [v_i * np.cos(phi_i), v_i * np.sin(phi_i), a_i, omega_i, e_v, e_phi]
            w_start, w_end_a = start_idx + 6, start_idx + 6 + self.num_weights
            w_end_omega = w_end_a + self.num_weights
            dydt[w_start:w_end_a] = ctrl_out['dW_hat_a_dt'].flatten()
            dydt[w_end_a:w_end_omega] = ctrl_out['dW_hat_omega_dt'].flatten()
            dydt[w_end_omega:w_end_omega + 2] = [ctrl_out['dzeta_a_dt'], ctrl_out['dzeta_omega_dt']]
        return dydt

    def run(self):
        y0_flat = np.zeros(self.total_states)
        leader_pos_0, leader_vel_0, _ = self._leader_trajectory(0)
        for i in range(self.cfg.N_FOLLOWERS):
            start_idx = i * self.vehicle_state_size
            y0_flat[start_idx] = leader_pos_0[0] - (i + 1) * self.cfg.D_STAR
            y0_flat[start_idx + 1] = leader_pos_0[1] + np.random.uniform(-1, 1)  # Small random initial y offset
            y0_flat[start_idx + 2] = leader_vel_0 - np.random.uniform(0.5, 2.0)  # Small random initial v offset

        controller_name = self.controller.__class__.__name__.replace("_", " ")
        print(f"--- Starting simulation for: {controller_name} ---")
        start_time = time.time()
        t_eval = np.arange(self.cfg.T_START, self.cfg.T_END, self.cfg.DT)
        sol = solve_ivp(fun=self.dynamics, t_span=[self.cfg.T_START, self.cfg.T_END], y0=y0_flat, t_eval=t_eval,
                        method='RK45')
        print(f"Simulation finished in {time.time() - start_time:.2f} seconds.")
        return sol, self.mode_history


# --- 4. Analysis and Plotting ---
def run_and_analyze(config, controller):
    simulation = PlatoonSimulation(config, controller)
    solution, mode_history = simulation.run()

    t = solution.t
    sim = PlatoonSimulation(config, controller)
    leader_traj = np.array([sim._leader_trajectory(ti) for ti in t], dtype=object)
    all_states = [sim._unpack_state_vector(solution.y[:, k]) for k in range(len(t))]

    res = {'e_d': [], 'e_v': [], 'u_a': [], 'norm_W_a': []}

    # Analyze the last follower vehicle for performance metrics
    i = config.N_FOLLOWERS - 1
    for k, ti in enumerate(t):
        current_states = all_states[k]
        leader_p, leader_v, leader_phi = leader_traj[k]
        s_i = current_states[i]
        x_i, y_i, v_i, phi_i = s_i['kinematics']
        predecessor_state = current_states[i - 1]['kinematics']
        x_p, y_p, v_p, phi_p = predecessor_state

        d_i = np.sqrt((x_p - x_i) ** 2 + (y_p - y_i) ** 2)
        theta_i = np.arctan2(y_p - y_i, x_p - x_i)
        e_d = d_i - config.D_STAR
        e_theta = theta_i - phi_p

        q_i_temp = e_d / (config.C_UPPER ** 2 - e_d ** 2 + 1e-9) if e_d >= 0 else e_d / (
                    config.C_LOWER ** 2 - e_d ** 2 + 1e-9)
        v_d = v_p * np.cos(e_theta) + config.k_d * q_i_temp
        e_v = v_i - v_d
        e_phi = phi_i - theta_i

        s_a = e_v + config.lambda_a * s_i['s_integrals'][0]
        s_omega = e_phi + config.lambda_omega * s_i['s_integrals'][1]

        ctrl_out = controller.compute_control_and_derivatives(
            i, (e_d, e_v, e_theta, e_phi),
            {'predecessor': (v_p, phi_p, theta_i), 'sliding_surfaces': (s_a, s_omega),
             's_integrals': s_i['s_integrals']},
            (s_i['W_hat_a'], s_i['W_hat_omega'], s_i['zeta'][0], s_i['zeta'][1]), ti
        )

        res['e_d'].append(e_d)
        res['e_v'].append(e_v)
        res['u_a'].append(ctrl_out['u_c_a'])
        res['norm_W_a'].append(np.linalg.norm(s_i['W_hat_a']))

    for key in res: res[key] = np.array(res[key])

    dt = t[1] - t[0] if len(t) > 1 else 1
    metrics = {
        'IAE (e_d)': np.sum(np.abs(res['e_d'])) * dt,
        'ISE (e_d)': np.sum(res['e_d'] ** 2) * dt,
        'ITAE (e_v)': np.sum(t * np.abs(res['e_v'])) * dt,
    }
    return t, res, metrics, all_states, leader_traj


def plot_all_comparisons(t, all_results, all_metrics, all_states, leader_traj, config):
    print("\n" + "=" * 50)
    print("      PERFORMANCE METRICS (Last Follower)")
    print("=" * 50)
    df = pd.DataFrame(all_metrics).T
    print(df.to_string(float_format="%.2f"))
    print("=" * 50 + "\n")

    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'Proposed FLENNSMC': 'b', 'Traditional RBFNN': 'g', 'SMC': 'r', 'PID': 'c'}
    styles = {'Proposed FLENNSMC': '-', 'Traditional RBFNN': '--', 'SMC': '-.', 'PID': ':'}
    widths = {'Proposed FLENNSMC': 2.5, 'Traditional RBFNN': 2.0, 'SMC': 2.0, 'PID': 2.0}

    def create_comp_plot(title, ylabel, data_key, ylim=None):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        for name, res in all_results.items():
            ax.plot(t, res[data_key],
                    color=colors.get(name, 'k'),
                    linestyle=styles.get(name, '-'),
                    linewidth=widths.get(name, 1.5),
                    label=name.replace("_", " "))
        if data_key == 'e_d':
            ax.axhline(y=config.C_UPPER, color='k', linestyle='--', label='Upper Constraint')
            ax.axhline(y=-config.C_LOWER, color='k', linestyle=':', label='Lower Constraint')
        ax.legend(fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if ylim: ax.set_ylim(ylim)
        fig.tight_layout()
        plt.show()

    # --- Generate Plots ---
    create_comp_plot('Comparison of Distance Error $e_d(t)$', 'Distance Error (m)', 'e_d',
                     ylim=[-config.C_LOWER - 1, config.C_UPPER + 1])
    create_comp_plot('Comparison of Velocity Error $e_v(t)$', 'Velocity Error (m/s)', 'e_v')
    create_comp_plot('Comparison of Control Effort $u_a(t)$', 'Acceleration Control Input', 'u_a')

    # Plot for adaptive weights (only for relevant controllers)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title('Comparison of RBFNN Weight Norm $||\\hat{W}_a||$', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Weight Vector Norm', fontsize=14)
    has_adaptive_plot = False
    for name, res in all_results.items():
        if 'RBFNN' in name or 'FLENNSMC' in name:
            ax.plot(t, res['norm_W_a'], color=colors.get(name, 'k'), linestyle=styles.get(name, '-'),
                    linewidth=widths.get(name, 1.5), label=name)
            has_adaptive_plot = True
#######################################################################################
    if has_adaptive_plot:
        ax.legend(fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()
        plt.show()

    # Plot 2D Platoon Trajectories for the best performing controller
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title('2D Platoon Trajectories (Proposed FLENNSMC)', fontsize=16, fontweight='bold')
    states_to_plot = all_states['Proposed FLENNSMC']
    leader_pos_x = [p[0][0] for p in leader_traj]
    leader_pos_y = [p[0][1] for p in leader_traj]
    ax.plot(leader_pos_x, leader_pos_y, 'k-', label='Leader', linewidth=2)
    for i in range(config.N_FOLLOWERS):
        follower_x = [s[i]['kinematics'][0] for s in states_to_plot]
        follower_y = [s[i]['kinematics'][1] for s in states_to_plot]
        ax.plot(follower_x, follower_y, '--', label=f'Follower {i + 1}')
    ax.set_xlabel('X Position (m)', fontsize=14)
    ax.set_ylabel('Y Position (m)', fontsize=14)
    ax.legend()
    ax.grid(True)
    # ax.axis('equal')
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    SEED = 6
    np.random.seed(SEED)

    config = SimulationConfig()

    controllers_to_run = {
        # "PID": PID_Controller(config),
        "SMC": SMC_Controller(config),
        "Traditional RBFNN": Traditional_RBFNN_Controller(config),
        "Proposed FLENNSMC": FLENNSMC_Controller(config),
    }

    all_results = {}
    all_metrics = {}
    all_states_history = {}

    # Run simulation for each controller
    for name, controller in controllers_to_run.items():
        t, res, mets, states_hist, leader_traj = run_and_analyze(config, controller)
        all_results[name] = res
        all_metrics[name] = mets
        all_states_history[name] = states_hist

    # Generate comparison plots and tables
    plot_all_comparisons(t, all_results, all_metrics, all_states_history, leader_traj, config)