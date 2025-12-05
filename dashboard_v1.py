"""
03/02/2025,
the main code for solving Krusell-Smith.

"module_training_bellman_v1":
    the training module, it will find the NN for policy and value functions. it takes "module_obj_bellman_v1" as input.
    bellman function based method:
        will use "policy_bellman_training" to find policy function, and use "value_training" to fit a value function.
"module_obj_bellman_v1":
    define value functions (.obj_sim_value).

"module_basic_v1":
    1. the NNs for policy and value functions;
    2. sampling from the state space;
    3. plot equm functions.

"config":
    define all variables used by the code.
"""

import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt
import module_obj_bellman_v1
from module_basic_v1 import Config, MyModel, plot_equm_funcs
from module_training_bellman_v1 import EqumTrainer as BellmanTrainer
# 引入新写的模块
from module_training_euler import EulerTrainer

# ... (初始化 Model 和 Config 同原代码) ...

# when import the saved models from the expand_model_v1,
# need to use "load_pretrained_new" to load the model as the way to save is a little different

# Close all existing figure windows
plt.close('all')
# Configuration
config = Config("config_v1.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Print the selected device
print("Using device:", device)

model = MyModel(config.n_input, config.n_p_output, config.n_v_output,
                config.n1_p, config.n2_p, config.n1_v, config.n2_v)

# Modify for multiple GPUs, this can be called only once.
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

model.to(device)

def instantiate_trainer(i_train, model_instance, model_num, i_save, use_pretrained=1):
    pretrained_value_path = f'models/trained_value_nn_{model_num}.pth'
    pretrained_policy_path = f'models/trained_policy_nn_{model_num}.pth'

    # Check if pre-trained models exist and if we should use them
    pretrained_value = pretrained_value_path if use_pretrained and os.path.exists(pretrained_value_path) else None
    pretrained_policy = pretrained_policy_path if use_pretrained and os.path.exists(pretrained_policy_path) else None

    TrainerClass = BellmanTrainer
    return TrainerClass(config.num_epochs_initial, config.num_epochs_v, config.num_epochs_p,
                        config.lr_initial, config.lr_value, config.lr_policy, config.batch_size_p,
                        config.batch_size_v, config.num_worker, model_instance,
                        pretrained_value=pretrained_value,
                        pretrained_policy=pretrained_policy,
                        device=device, i_save=i_save)

# Main Execution
start_time = datetime.now()

# define the mid-point values for the distribution bins:
dist_a_mid_values = config.dist_a_mid
dist_a_mid = torch.tensor(dist_a_mid_values).unsqueeze(1).to(device)
dist_a_mesh = dist_a_mid_values[1] - dist_a_mid_values[0]


for i_iter in range(config.num_iter):
    print('iter:', i_iter)

    config.dist_a_band = config.dist_a_band + 0.3 / config.num_iter

    if i_iter == 0:
        i_save_iter = 1
        i_load_pretrainted = config.i_pretrainted
        #+++++++++++++++++++++++++++
        # simulate the model first:
        decision_trainer = instantiate_trainer(config.i_training, model, config.model_number_input, i_save_iter,
                                               use_pretrained=i_load_pretrainted)
        model = decision_trainer.get_pretrained_model()
        equm_updater = module_obj_bellman_v1.define_objective(model, device)
        domain_sampler = decision_trainer.get_domain_sampler()
        initial_data = domain_sampler.generate_samples(1, config.k_dist)
        equm_updater.sim_path(initial_data, config.n_sim_path, dist_a_mid, dist_a_mesh)

    else:
        i_save_iter = 1
        i_load_pretrainted = 0

    # training starts:
    '''bellman equation based'''
    # double check the value of "config.model_number_input"
    # decision_trainer = instantiate_trainer(config.i_training, model, config.model_number_input, i_save_iter,
    #                                        use_pretrained=i_load_pretrainted)
    # model = decision_trainer.policy_bellman_training(config.n_p_sim, dist_a_mid, dist_a_mesh)
    # model = decision_trainer.value_training(config.n_v_sim, dist_a_mid, dist_a_mesh)

    # 实例化 DL Trainer
    dl_trainer = EulerTrainer(model, device)

    # 开始训练 (替代原来的 policy_bellman_training 和 value_training)
    # 注意：DL 方法只需要训练 Policy Network，不需要 Value Network
    model = dl_trainer.train_euler(
        num_epochs=100,             # 可以根据收敛情况调整
        batch_size=config.batch_size_p,
        dist_a_mid=dist_a_mid_values,
        dist_a_mesh=dist_a_mesh
    )

    # 保存模型
    torch.save(model.state_dict(), f'models/trained_euler_nn_{config.model_number_output}.pth')

    # ... (后续的 sim_path 和画图逻辑可以完全复用，因为它们只依赖 model) ...

    num_samples = 10000
    plotter = plot_equm_funcs(num_samples, config.k_dist, dist_a_mid, model, device)
    plotter.create_plot()

# simluate and plot the path
equm_updater = module_obj_bellman_v1.define_objective(model, device)
domain_sampler = decision_trainer.get_domain_sampler()
initial_data = domain_sampler.generate_samples(1, config.k_dist)
equm_updater.sim_path(initial_data, config.n_sim_path, dist_a_mid, dist_a_mesh)

def log_memory_usage():
    """Logs GPU memory usage."""
    print(f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f}MB")
    print(f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(0) / 1024 / 1024:.2f}MB")
    print(f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(0) / 1024 / 1024:.2f}MB")


end_time = datetime.now()
log_memory_usage()
print(f'Duration: {end_time - start_time}')





