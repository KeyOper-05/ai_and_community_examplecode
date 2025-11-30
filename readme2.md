

# Overall Repository Structure

The project is organized into several key modules and files, each responsible for distinct tasks. At a high level, the system is built around an **actor-critic reinforcement learning framework** to solve a heterogeneous-agent model under aggregate uncertainty. The main components are:

- **Training Modules**: Update the policy (actor) and value (critic) networks using Bellman equations.
- **Objective Functions**: Define simulation-based objectives and generate economic trajectories.
- **Core Utilities & Models**: Contain the neural network definitions, sampling routines, and visualization tools.
- **Orchestration Script**: Coordinates initialization, training loops, simulations, and checkpointing.
- **Configuration File**: Centralizes all model, economic, and training hyperparameters.

---

# Detailed File Breakdown

Below is a table summarizing each file's purpose, key components, and dependencies:

| File                         | Purpose                                                                 | Key Components                                                                                                                       | Dependencies                                             |
|------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| `module_training_bellman_v1.py` | Trains the policy and value networks using Bellman updates             | **EqumTrainer Class** with:<br> - `policy_bellman_training` (generates samples, simulates value paths, maximizes Bellman objective)<br> - `value_training` (updates value network via MSE minimization)<br> - Auxiliary methods for pretrained models, NaN handling, checkpointing | Uses functions from `module_obj_bellman_v1` and `module_basic_v1` |
| `module_obj_bellman_v1.py`   | Defines objective functions for training and simulation                  | **define_objective Class**:<br> - `obj_sim_value` for computing Bellman targets<br> **Simulation Functions**:<br> - `sim_path` to generate trajectories        | Relies on utility functions and configurations from `module_basic_v1` |
| `module_basic_v1.py`         | Provides core definitions of neural networks and utility functions         | **MyModel Class**:<br> - `policy_func` (actor network)<br> - `value_func` (critic network)<br> **DomainSampling Class** (generates state samples)<br> **Visualization**:<br> - `plot_equm_funcs`<br> **Config Class**: Loads parameters from `config_v1.json` | None (central backbone for the project)                 |
| `dashboard_v1.py`            | Main orchestration script for initialization, training, simulation, and checkpointing | **Workflow**:<br> 1. **Initialization**: Loads configuration and instantiates `MyModel`<br> 2. **Training Loop**: Alternates actor-critic updates using `EqumTrainer`<br> 3. **Simulation/Visualization**: Calls `sim_path` and `plot_equm_funcs`<br> 4. **Checkpointing**: Saves model states | Integrates functionalities from all other modules       |
| `config_v1.json`             | Centralized configuration file for all model and training parameters        | **Key Parameters**:<br> - Model architecture (input/output dimensions, layer sizes)<br> - Economic settings (β, σ, α, δ, etc.)<br> - State variables and sampling grids<br> - Training hyperparameters (epochs, batch sizes, learning rates)<br> - Miscellaneous (checkpointing, penalties) | Used by all modules for consistent configuration          |

---

# Code Relationships and Data Flow

The following table outlines the relationships between modules and the overall data flow:

| Component / Module           | Input/Source                                  | Processing / Role                                                                                                         | Output/Target                                  | Dependencies                                       |
|------------------------------|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|----------------------------------------------------|
| **Actor (Policy Network)**   | State samples, Bellman objectives             | Determines optimal asset accumulation decisions via `policy_bellman_training`                                             | Updated policy network weights              | `module_training_bellman_v1.py`, `module_basic_v1.py` |
| **Critic (Value Network)**   | Simulated returns, state samples              | Evaluates expected returns and updates using `value_training`                                                           | Updated value network weights               | `module_training_bellman_v1.py`, `module_basic_v1.py` |
| **Objective Functions Module** | State and simulation parameters from config  | Generates Bellman targets and simulation trajectories using methods like `obj_sim_value` and `sim_path`                    | Simulation trajectories, training targets   | `module_basic_v1.py`                                |
| **Core Utilities Module**    | Configuration and state variable bounds       | Defines neural network architectures (`MyModel`), sampling routines (`DomainSampling`), and visualization tools           | Normalized data, plotted functions, configurations | `config_v1.json`                                     |
| **Orchestration Script**     | All configurations and module outputs         | Coordinates the overall workflow: initializes models, alternates training phases, simulates and visualizes results         | Checkpoints, figures, trained models          | Integrates all other modules                        |

---

# Implementation and Technical Details

| Aspect                   | Description                                                                                                                                      |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| **Numerical Stability**  | Implements mechanisms to detect and reset unstable gradients; employs learning rate scheduling (using parameters like `lr_factor` and `lr_patience`) for stable convergence. |
| **Parallel Computing**   | Supports multi-GPU training using PyTorch's `torch.nn.DataParallel`, which facilitates handling large-scale state spaces efficiently.            |
| **Checkpointing**        | Periodically saves model states (e.g., `trained_policy_nn_*.pth`, `trained_value_nn_*.pth`), allowing training to resume without starting over. |
| **Normalization**        | Scales input data to a fixed range (typically [0,1]) to enhance training robustness and speed.                                                     |

---

# Customization and Extension

| Customization Area         | Description                                                                                                                                                             | How to Modify                                             |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| **Model Architecture**     | Neural network dimensions, layer sizes, and input/output settings for both policy and value networks.                                                                   | Update parameters in `config_v1.json` (e.g., `n_input`, `n1_p`, etc.) |
| **Economic Settings**      | Parameters such as discount factor (β), risk aversion (σ), production function parameters, etc.                                                                        | Edit relevant fields in `config_v1.json`                  |
| **State Variables & Sampling** | Bounds for variables like productivity (`z`) and assets (`a`), along with distribution grids and pdf values.                                                              | Adjust grids, bounds, and penalties in `config_v1.json`   |
| **Training Hyperparameters** | Epoch counts, batch sizes, learning rates, and other training-related parameters.                                                                                    | Modify values in `config_v1.json` (e.g., `num_epochs_model`, `batch_size_v`) |
| **Pretrained Models**      | Option to resume training using saved models.                                                                                                                        | Set `i_pretrainted` flag in `config_v1.json` to 1           |

---

# Execution Workflow

| Step         | Actions                                                                                                    | Command / Method                                        |
|--------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| **Setup**    | - Install dependencies (Python 3.8+, PyTorch 1.9+, NumPy, Matplotlib, Pandas).<br>- Configure parameters in `config_v1.json`. | `pip install torch numpy matplotlib pandas`           |
| **Run**      | Execute the main script that orchestrates initialization, training, simulation, and visualization.         | `python dashboard_v1.py`                                |
| **Output**   | - Trained model checkpoints saved (e.g., in the `models/` directory).<br>- Figures and plots generated (e.g., in the `figures/` folder). | Check generated files after execution.                |

