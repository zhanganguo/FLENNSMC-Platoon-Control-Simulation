
---

# FLENNSMC-Platoon-Control-Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-gray.svg)
![SciPy](https://img.shields.io/badge/SciPy-orange.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-red.svg)

## Fuzzy Logic-Enhanced Neuroadaptive Fault-tolerant Control for Platoon Systems with Stochastic Disturbances and Asymmetric Spacing Constraints

This repository contains the numerical simulation code for the research paper:

**"Fuzzy Logic-Enhanced Neuroadaptive Fault-tolerant Control for Platoon Systems with Stochastic Disturbances and Asymmetric Spacing Constraints"**

By Yao Wen, Xuesong Xu, Anguo Zhang, and Yongfu Li.

---

### 📚 Project Overview

This project implements and validates a novel Fuzzy Logic-Enhanced Neuroadaptive Sliding Mode Control (FLENNSMC) framework developed for vehicular platoon systems. These systems are subject to a confluence of challenges, including highly nonlinear dynamics, stochastic disturbances (modeled by Markovian jumps), unknown-direction actuator faults, and stringent asymmetric spacing constraints. This work introduces a robust, adaptive, and fault-tolerant solution that addresses these complex and interdependent issues within a unified control architecture.

### ✨ Key Features & Contributions

*   **Integrated Control Framework:** Introduces a comprehensive FLENNSMC framework that synergistically combines fuzzy logic, RBF neural networks, SMC, asymmetric BLFs, and Nussbaum functions to provide a robust, adaptive, and fault-tolerant solution for complex stochastic nonlinear systems.
*   **Structured Neuroadaptation with Fuzzy Logic (FLERBFNN):** Leverages fuzzy logic to significantly enhance the learning capabilities and efficiency of RBFNNs. This provides a structured, localized learning process, leading to faster convergence and reduced computational burden, which is critical for real-time applications.
*   **Guaranteed Spacing Constraints & Performance:** Employs an asymmetric Barrier Lyapunov Function (BLF) to ensure strict adherence to safety and communication constraints. It also derives LMI-based conditions to guarantee a novel mixed H$\infty$ and passivity performance, ensuring strong robustness against disturbances.
*   **Fault Tolerance:** Employs Nussbaum functions to effectively compensate for actuator faults with unknown directions, ensuring system stability even under critical fault conditions.
*   **Rigorous Theoretical Analysis:** Accounts for Markovian jump parameters and time-varying delays in the system model and control design, enhancing resilience in uncertain environments.
*   **Rigorous Stability Guarantees:** Establishes, through a detailed stochastic Lyapunov-Krasovskii analysis, sufficient conditions for the Uniform Ultimate Boundedness (UUB) of all tracking errors in the mean-square sense, fully accounting for the system's stochastic nature, time delays, and adaptive fault-tolerant elements.

### ❓ Problem Statement

Controlling vehicular platoons is challenging due to inherent nonlinear dynamics, communication delays, and environmental uncertainties. When combined with stochastic operational modes (Markovian jumps), dangerous unknown-direction actuator faults, and strict asymmetric spacing requirements, these challenges demand sophisticated control strategies that can ensure safety, stability, and performance simultaneously. Traditional linear or even standard nonlinear adaptive methods often fall short in addressing this complex, multi-faceted problem in a unified and robust manner.

### 💡 Methodology (FLENNSMC)

The proposed FLENNSMC framework integrates several advanced techniques:

1. **Takagi-Sugeno (T-S) Fuzzy Model:** Partitions the state space to structure the learning process, enabling localized adaptation to different operating conditions.
2. **Fuzzy Logic-Enhanced RBFNN (FLERBFNN):** Within each fuzzy rule, a dedicated RBFNN approximates unknown nonlinear functions (e.g., unmodeled dynamics, bias faults). This "gray-box" approach makes learning more efficient and interpretable.
3. **Sliding Mode Control (SMC):** Provides inherent robustness against uncertainties and external disturbances, forcing system states onto a desired sliding manifold.
4. **Asymmetric Barrier Lyapunov Function (BLF):** Guarantees that inter-vehicle distance errors strictly remain within predefined safety bounds, which is crucial for collision avoidance and maintaining communication links.
5. **Nussbaum Function:** Adaptively compensates for the unknown gain direction of actuator faults, allowing the controller to maintain stability and performance even when fault characteristics are unknown.
6. **Lyapunov-Krasovskii Functionals & LMI:** Used for the rigorous mean-square stability proof, accounting for time-delays and Markovian jumps, and to derive computable LMI conditions for performance guarantees.

### 🖥️ Simulation Environment & Setup

*   **Language:** Python (tested with Python 3.8+).
*   **Operating System:** Platform-independent.
*   **Key Libraries:**
    *   `numpy`: For numerical operations and array manipulation.
    *   `scipy`: For scientific computing, including integration (`scipy.integrate.solve_ivp`).
    *   `matplotlib`: For plotting and visualization of results.
    *   `pandas`: For tabular data display (in comparative analysis).
*   **System:** A 2D multi-lane vehicular platoon with 1 leader and 4 follower vehicles ($N=4$).
*   **Simulation Duration:** 60 seconds.
*   **Key Conditions Simulated:**
    *   Dynamic 2D trajectory tracking (smooth lane-change).
    *   Strict asymmetric spacing constraints ($d^* = 20$\,m, $\Delta_{col} = 10$\,m, $\Delta_{con} = 30$\,m).
    *   Continuous-time Markov jump process (2 modes) for stochastic dynamics.
    *   Complex unknown nonlinear functions and external disturbances.
    *   Time-varying, unknown-direction actuator faults for both longitudinal acceleration and angular rate.

### ⬇️ Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/zhanganguo/FLENNSMC-Platoon-Control-Simulation.git
    cd FLENNSMC-Platoon-Control-Simulation
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install numpy scipy matplotlib pandas
    ```

### 🚀 Getting Started

This repository includes two main simulation scenarios, corresponding to "simulation 1" (Integrated Performance) and "simulation 2" (Comparative Performance) in the paper.

#### Scenario 1: Integrated Performance

This scenario evaluates the FLENNSMC framework's comprehensive ability to handle all challenges simultaneously. It generates individual plots for position tracking, various errors, control inputs, and adaptive parameter behaviors.

1.  **Run the Simulation:**
    Open your terminal or command prompt in the root directory of the cloned repository.
    ```bash
    python simulation_1.py
    ```
    The simulation results (plots corresponding to Figures 1-5 in the paper) will be automatically generated and displayed, then saved in the `figures/` directory.

#### Scenario 2: Comparative Performance

This scenario compares the performance of the proposed FLENNSMC with a Traditional RBFNN-based controller and a standard Sliding Mode Controller (SMC).

1.  **Run the Simulation:**
    Open your terminal or command prompt in the root directory of the cloned repository.
    ```bash
    python simulation_2.py
    ```
    The simulation will run all three controllers sequentially and then generate comparative plots (corresponding to Figures 6-7 in the paper) and print a performance metrics table in the console. Plots will also be saved in the `figure/` directory.

### 📊 Results & Analysis

The `figure/` directory contains all the plots generated from both simulation scenarios, validating the proposed FLENNSMC framework's performance.

**Key Findings Illustrated in Figures:**

*   **Integrated Performance Validation:**
    * **Distance Error with Constraints:** Shows all followers maintaining inter-vehicle distance strictly within the predefined safety bounds, crucial for collision avoidance.
      ![Distance Error with Constraints](figure/exp0_distance_error.png)

    *   **Control Input for Acceleration:** Demonstrates the FLENNSMC's ability to adapt and maintain effective control even during an unknown-direction actuator fault period (highlighted in red).
        ![Control Input for Acceleration](figure/exp0_control_input.png)

    *   **Nussbaum Function Gains:** Illustrates how the Nussbaum gains dynamically adapt over time to compensate for the unknown control direction of actuator faults, remaining bounded and ensuring system stability.
        ![Nussbaum Function Gains](figure/exp0_nussbaum_function_gains.png)

*   **Comparative Performance Validation:**
    *   **Comparison of Velocity Error:** Clearly shows the superior tracking accuracy and stability of the Proposed FLENNSMC compared to Traditional RBFNN and the unstable Standard SMC.
        ![Comparison of Velocity Error](figure/exp1_velocity_error.png)

    *   **Comparison of RBFNN Weight Norm:** Highlights FLENNSMC's higher learning efficiency by converging to a significantly smaller adaptive weight norm than Traditional RBFNN.
        ![Comparison of RBFNN Weight Norm](figure/exp1_rbfnn_weight_norm.png)

### 📞 Contact

For any questions, suggestions, or collaborations, feel free to reach out:

*   **Anguo Zhang:** [anguo.zhang@hotmail.com](mailto:anguo.zhang@hotmail.com)

### 引用 (Citation)

If you use this code or concepts from this research in your work, please cite our paper:

```bibtex
@article{Wen2024FLENNSMC,
  title={Fuzzy Logic-Enhanced Neuroadaptive Fault-tolerant Control for Platoon Systems with Stochastic Disturbances and Asymmetric Spacing Constraints},
  author={Wen, Yao and Xu, Xuesong and Zhang, Anguo and Li, Yongfu},
  journal={UNDER REVIEW},
  year={2025},
}
```

---
