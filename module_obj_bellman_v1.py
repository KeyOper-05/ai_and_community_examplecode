"""
03/02/2025,
define the objective functions:
    1. euler equation (.obj_sim_euler);
    2. value functions (.obj_sim_value).
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import module_basic_v1
from tqdm import tqdm

# import configuration data:
config = module_basic_v1.Config("config_v1.json")


class define_objective:
    def __init__(self, model, device=None):
        self.device = device
        self.model = model.to(self.device)
        # Define ranges once
        keys = ["z", "a"]
        self.ranges = [(config.bounds[key]["min"], config.bounds[key]["max"]) for key in keys]
        # Extend the ranges for dist_a_mid
        for dist_a_pdf in config.dist_a_pdf:
            extended_min = dist_a_pdf * (1 - config.dist_a_band)
            extended_max = dist_a_pdf * (1 + config.dist_a_band)
            self.ranges.append((extended_min, extended_max))

    def get_pdf_sampler(self):
        """
        Create and return an instance of the domain_sampling class.
        """
        return module_basic_v1.DomainSampling(self.ranges, device=self.device)

    def extract_state_variables(self, x_batch):
        """
        Extract state variables from the input batch.
        """
        device = self.device
        x_z1 = x_batch[:, 0].unsqueeze(1).to(device)
        x_a1 = x_batch[:, 1].unsqueeze(1).to(device)
        x_dist1 = x_batch[:, 2:].to(device)  # Extracts all columns from the 3rd to the last
        return x_z1, x_a1, x_dist1

    def predict_model(self, input_data, function_type='policy'):
        """
        Unified model prediction function that handles DataParallel and model function selection.
        """
        if isinstance(self.model, torch.nn.DataParallel):
            model_function = self.model.module.f_policy if function_type == 'policy' else self.model.module.f_value
        else:
            model_function = self.model.f_policy if function_type == 'policy' else self.model.f_value

        return model_function(input_data)

    def obj_sim_value(self, x_batch, x_n_sim, dist_a_mid, dist_a_mesh):
        pdf_sampler = self.get_pdf_sampler()

        x_beta = config.beta
        x_z1, x_a1, x_dist1 = self.extract_state_variables(x_batch)
        x_beta_pow = [config.beta ** i for i in range(x_n_sim)]
        v0_sim_accumulator = torch.zeros_like(x_a1).to(self.device)
        v0_fit_accumulator = torch.zeros_like(x_a1).to(self.device)
        x = 1 + 1 / config.theta_l
        x_int_z1 = torch.exp(torch.tensor(1 / 2 * x * x * config.sigma_z ** 2))

        all_variables = ["z", "a", "dist_a"]
        z_index = all_variables.index("z")
        z_min, z_max = self.ranges[z_index]
        a_index = all_variables.index("a")
        a_min, a_max = self.ranges[a_index]

        tfp_grid = torch.tensor(config.tfp_grid).view(-1, 1).to(self.device)
        tfp_transition = torch.tensor(config.tfp_transition).view(config.n_tfp, config.n_tfp).to(self.device)
        x_i_tfp1 = torch.randint(config.n_tfp, (x_z1.size(0), 1), device=self.device)

        x_tfp0 = tfp_grid[x_i_tfp1.squeeze()].view(-1, 1).expand(x_z1.shape)

        for i in range(x_n_sim):
            x_z0, x_a0, x_dist0 = x_z1.clone(), x_a1.clone(), x_dist1.clone()
            x_i_tfp0 = x_i_tfp1.clone()
            x_tfp0 = tfp_grid[x_i_tfp0.squeeze()].view(-1, 1)

            x_x0_policy = torch.cat([x_z0, x_a0, x_dist0], dim=1).to(self.device)
            x_x0_policy_sd = module_basic_v1.normalize_inputs(x_x0_policy, config.bounds)
            x_x0_policy_sd = torch.cat((x_tfp0, x_x0_policy_sd), dim=1)

            x_int_z = torch.full_like(x_z0, x_int_z1).to(self.device)
            x_a0_total = (x_dist0 * dist_a_mid.T).sum(dim=1, keepdim=True).to(self.device)

            x_w0, x_l0, x_r0, x_y0 = self.calculate_aggregates(x_tfp0, x_z0, x_a0_total, x_int_z)
            x_y0_policy = self.predict_model(x_x0_policy_sd, 'policy')

            x_a1 = x_y0_policy[:, 0].unsqueeze(1) * (a_max - a_min)
            x_c0_orig = (1 + x_r0) * x_a0 + x_w0 * x_l0 * x_z0 - x_a1
            x_c0 = torch.maximum(x_c0_orig, torch.tensor(config.u_eps))
            x_c0_punish = (1 / torch.tensor(config.u_eps)) * torch.maximum(-x_c0_orig, torch.tensor(0))

            x_z1 = module_basic_v1.bounded_log_normal_samples(config.mu_z, config.sigma_z, z_min, z_max, x_z0.size(0))
            x_z1 = x_z1.unsqueeze(1).to(self.device)

            # generate x_tfp1
            transition_probs_for_x_tfp0 = tfp_transition[x_i_tfp0.squeeze()]
            x_i_tfp1 = torch.multinomial(transition_probs_for_x_tfp0, 1)
            x_tfp1 = tfp_grid[x_i_tfp1.squeeze()].view(-1, 1)

            if config.i_dist == 1:
                n_batch, n_dim = x_dist0.shape
                x_dist_g_all = self.calculate_G_batch(dist_a_mid, dist_a_mesh, x_dist0, x_tfp0)
                x_dist0_reshaped = x_dist0.view(n_batch, n_dim, 1)
                x_dist1 = torch.bmm(x_dist0_reshaped.transpose(1, 2), x_dist_g_all).transpose(1, 2).view(n_batch, n_dim)
            elif config.i_dist == 0:
                n_batch, n_dim = x_dist0.shape
                x_dist1 = pdf_sampler.generate_samples_a_pdf(n_batch, n_dim)

            x_dist1, x_dist_penalty = pdf_sampler.dist_enforce_boundaries(x_dist1, config.a_pdf_penalty)

            x_u_cl = x_c0 - config.psi_l * x_l0 ** (1 + config.theta_l) / (1 + config.theta_l)
            x_u0 = x_u_cl ** (1 - config.sigma) / (1 - config.sigma)
            x_total_punish = x_c0_punish + x_dist_penalty  # + x_u_cl_punish

            current_val = x_u0 - x_total_punish

            v0_sim_accumulator += x_beta_pow[i] * current_val
            v0_fit_accumulator += x_beta_pow[i] * x_u0

        x_v0_sim = v0_sim_accumulator
        x_v0_fit = v0_fit_accumulator
        x_x1_value = torch.cat([x_z1, x_a1, x_dist1], dim=1).to(self.device)
        x_v1_sim = self.expected_value_V(x_x1_value, x_tfp1)  # , config.eu_samples)

        x_v_sim_sum = x_v0_sim + x_beta ** x_n_sim * x_v1_sim
        x_v_fit_sum = x_v0_fit + x_beta ** x_n_sim * x_v1_sim
        x_value_data = torch.cat((x_tfp0, x_batch, x_v_fit_sum), dim=1)

        return torch.mean(-x_v_sim_sum), x_value_data.detach()

    def calculate_aggregates(self, x_tfp0, x_z0, x_a0_total, x_int_z):
        x_w0_1 = (1 - config.alpha) * (x_a0_total / x_int_z) ** config.alpha
        x_w0 = x_tfp0 * config.psi_l ** (config.alpha / config.theta_l) * x_w0_1 ** (
                    config.theta_l / (config.alpha + config.theta_l))
        x_l0 = (x_w0 * x_z0 / config.psi_l) ** (1 / config.theta_l)
        x_r0 = x_tfp0 * config.alpha * (x_w0 / (1 - config.alpha)) ** ((config.alpha - 1) / config.alpha) - config.delta
        x_y0 = x_tfp0 * x_a0_total ** config.alpha * torch.ones_like(x_l0)  # * x_l0 ** (1 - config.alpha)
        return x_w0, x_l0, x_r0, x_y0


    def find_x_z0_linear(self, x_dist_a_mid, x_a1_bin, x_dist0_batch, x_tfp0):
        '''
        find the value of z such that a+ inside the bins;
        assume that the policy function is linear in Z.
        '''
        n_batch, n = x_dist0_batch.shape
        x_z0 = torch.zeros(n_batch, n, n, device=self.device)

        # Indices and range values
        all_variables = ["z", "a", "dist_a"]
        z_index = all_variables.index("z")
        z_min, z_max = self.ranges[z_index]
        a_index = all_variables.index("a")
        a_min, a_max = self.ranges[a_index]

        # Expand x_dist_a_mid
        x_dist_a_mid_expanded = x_dist_a_mid.view(1, -1, 1).expand(n_batch, -1, n)
        x_i = x_dist_a_mid_expanded
        x_j = x_dist_a_mid_expanded.transpose(1, 2)

        # Reshape z_min and z_max for batch operation
        z_min_tensor = torch.full((n_batch, n, n), z_min, device=self.device)
        z_max_tensor = torch.full((n_batch, n, n), z_max, device=self.device)

        x_dist0_batch_expanded = x_dist0_batch.view(n_batch, 1, n).repeat(1, n * n, 1)

        # Prepare inputs for the model
        x_x0_element_min = torch.cat([z_min_tensor.flatten(start_dim=1).unsqueeze(2), x_i.flatten(start_dim=1).unsqueeze(2),
                   x_dist0_batch_expanded], 2).to(self.device)
        x_x0_element_min_flat = x_x0_element_min.view(-1, x_x0_element_min.size(-1))
        x_x0_sd_element_min = module_basic_v1.normalize_inputs(x_x0_element_min_flat, config.bounds)
        x_tmp_tfp0 = torch.repeat_interleave(x_tfp0, repeats=n*n, dim=0)
        x_x0_sd_element_min = torch.cat([x_tmp_tfp0, x_x0_sd_element_min], 1).to(self.device)
        f_min = self.predict_model(x_x0_sd_element_min, 'policy').unsqueeze(1) * (a_max - a_min)
        f_min = f_min.view(n_batch, n, n)

        x_x0_element_max = torch.cat([z_max_tensor.flatten(start_dim=1).unsqueeze(2), x_i.flatten(start_dim=1).unsqueeze(2),
                   x_dist0_batch.view(n_batch, 1, n).repeat(1, n * n, 1)], 2).to(self.device)
        x_x0_element_max_flat = x_x0_element_max.view(-1, x_x0_element_max.size(-1))
        x_x0_sd_element_max = module_basic_v1.normalize_inputs(x_x0_element_max_flat, config.bounds)
        x_x0_sd_element_max = torch.cat([x_tmp_tfp0, x_x0_sd_element_max], 1).to(self.device)
        f_max = self.predict_model(x_x0_sd_element_max, 'policy').unsqueeze(1) * (a_max - a_min)
        f_max = f_max.view(n_batch, n, n)

        # Compute x_delta and x_z0 for the whole tensor
        x_delta = (f_max - f_min) / (z_max - z_min)
        x_z0 = torch.relu((x_j + x_a1_bin - f_min) / x_delta + z_min)

        # x_j is the mid-point of the bin, x_j + x_a1_bin corresponds to the upper/lower bound of bin;
        # x_z0 is the z s.t. a+ equals to the boundary values.

        return x_z0

    def calculate_G_batch(self, x_dist_a_mid, x_dist_a_mesh, x_dist0_batch, x_tfp0):  # Add other necessary parameters

        # Call the optimized function to find x_z0
        x_a1_bin = x_dist_a_mesh / 2
        x_z0_1 = self.find_x_z0_linear(x_dist_a_mid, x_a1_bin, x_dist0_batch, x_tfp0)

        x_a1_bin = -x_dist_a_mesh / 2
        x_z0_2 = self.find_x_z0_linear(x_dist_a_mid, x_a1_bin, x_dist0_batch, x_tfp0)

        f1 = self.log_normal_cdf(x_z0_1, config.mu_z, config.sigma_z)
        f2 = self.log_normal_cdf(x_z0_2, config.mu_z, config.sigma_z)
        x_G = torch.relu(f1 - f2)

        # Normalize x_G
        row_sums = x_G.sum(dim=2, keepdim=True)

        # Mask for too small sums
        epsilon = torch.finfo(row_sums.dtype).tiny  # Small number based on dtype
        too_small_mask = row_sums <= epsilon

        # Normalize, but replace rows with small sums with a uniform distribution
        normalized_x_G_batch = torch.where(
            too_small_mask,
            torch.full_like(x_G, 1 / config.k_dist),  # Uniform distribution
            x_G / row_sums
        )

        return normalized_x_G_batch


    def expected_value_V(self, x_x0_value, x_tfp1):
        """
        Compute the expected value given the input states, using discrete values from dist_z_mid
        with their respective probabilities from dist_z_pdf.
        """
        # Extract necessary tensors
        x_z1, x_a1, x_dist1 = self.extract_state_variables(x_x0_value)

        # Load probability distribution and mid values for z distribution
        dist_z_pdf = torch.tensor(config.dist_z_pdf, device=self.device)
        dist_z_mid = torch.tensor(config.dist_z_mid, device=self.device)

        # Repeat dist_z_mid to match batch size
        z_plus_samples = dist_z_mid.unsqueeze(0).unsqueeze(-1).expand(x_z1.size(0), -1, 1)

        # Expand other state tensors to match the size
        x_a1_expanded = x_a1.unsqueeze(1).expand(-1, len(dist_z_mid), -1)
        x_dist1_expanded = x_dist1.unsqueeze(1).expand(-1, len(dist_z_mid), -1)

        # Combine tensors for model input
        x_x0_value_mc = torch.cat([z_plus_samples, x_a1_expanded, x_dist1_expanded], dim=2)

        # Normalize inputs
        x_x0_value_mc_sd = module_basic_v1.normalize_inputs(x_x0_value_mc.reshape(-1, x_x0_value_mc.size(-1)),
                                                            config.bounds)
        x_x0_value_mc_sd = torch.cat([x_tfp1.repeat(len(dist_z_mid), 1), x_x0_value_mc_sd], dim=1)
        # Apply model to predict values
        x_y0_value = self.predict_model(x_x0_value_mc_sd, 'value').view(x_z1.size(0), len(dist_z_mid))

        # Compute expected values weighted by dist_z_pdf
        expected_values = torch.sum(x_y0_value * dist_z_pdf, dim=1)

        return expected_values.unsqueeze(1)

    def log_normal_cdf(self, x, mu, sigma):
        # Convert x to the corresponding value for a normal distribution
        normal_value = torch.log(x)

        # Compute the CDF of the corresponding normal distribution
        cdf = 0.5 * (1 + torch.erf((normal_value - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))))

        return cdf

    def sim_path(self, x_batch, x_n_sim, dist_a_mid, dist_a_mesh):
        """
        Modified sim_path: Uses Monte Carlo simulation of N agents to track the real distribution.
        This is much more stable and realistic than iteratively calculating G_batch.
        """
        # 1. 设置模拟参数
        num_agents = 10000  # 模拟1万个人，样本越多图越平滑
        
        # 重新初始化状态 (不使用传入的 x_batch，而是重新生成更广泛的样本)
        z_min, z_max = self.ranges[0]
        a_min, a_max = self.ranges[1]
        
        # A. 初始化资产 a0: 均匀分布在 [a_min, a_max]
        x_a0 = torch.rand(num_agents, 1, device=self.device) * (a_max - a_min) * 0.5 + a_min
        
        # B. 初始化冲击 z0: 对数正态分布
        x_z0 = module_basic_v1.bounded_log_normal_samples(
            config.mu_z, config.sigma_z, z_min, z_max, num_agents
        ).unsqueeze(1).to(self.device)
        
        # C. 初始化宏观 TFP
        tfp_grid = torch.tensor(config.tfp_grid, device=self.device).view(-1, 1)
        tfp_transition = torch.tensor(config.tfp_transition, device=self.device).view(config.n_tfp, config.n_tfp)
        
        # 随机选一个初始 TFP
        curr_tfp_idx = torch.randint(0, config.n_tfp, (1,)).item()
        x_tfp0 = tfp_grid[curr_tfp_idx].view(1, 1).expand(num_agents, 1)
        
        # 记录数据的列表
        x_dist0_path = [] # 记录分布历史
        x_r0_path = []    # 记录利率历史
        x_tfp0_path = []  # 记录TFP历史

        print(f"Starting Monte Carlo Simulation with {num_agents} agents for {x_n_sim} steps...")

        for sim_step in range(x_n_sim):
            # 进度打印
            if sim_step % 100 == 0:
                print(f"  > Simulating step {sim_step}/{x_n_sim}...")

            # -----------------------------------------------------------
            # 1. 核心修正：使用“统计法”计算当前的真实分布
            # -----------------------------------------------------------
            # 找到每个 agent 的资产最接近分布网格(dist_a_mid)中的哪一个点
            # dist_a_mid shape: [k_dist, 1] -> 转置为 [1, k_dist]
            # x_a0 shape: [num_agents, 1]
            # distances shape: [num_agents, k_dist]
            distances = torch.abs(x_a0 - dist_a_mid.T) 
            nearest_idx = torch.argmin(distances, dim=1) # 每个agent属于哪个bin
            
            # 统计频率 (Histogram)
            current_dist = torch.zeros(1, config.k_dist, device=self.device)
            for k in range(config.k_dist):
                count = (nearest_idx == k).float().sum()
                current_dist[0, k] = count
            
            # 归一化为概率密度
            current_dist = current_dist / num_agents
            
            # 扩展到所有 agent (大家看到的宏观分布是一样的)
            x_dist0 = current_dist.expand(num_agents, -1)
            
            # 记录用于画图
            x_dist0_path.append(current_dist)

            # -----------------------------------------------------------
            # 2. 神经网络决策
            # -----------------------------------------------------------
            # 准备输入
            x_x0_policy = torch.cat([x_z0, x_a0, x_dist0], dim=1)
            x_x0_policy_sd = module_basic_v1.normalize_inputs(x_x0_policy, config.bounds)
            x_x0_policy_sd = torch.cat((x_tfp0, x_x0_policy_sd), dim=1)

            # 计算宏观变量 (Aggregates)用于记录
            # K = sum(density * mid_points)
            x_a0_total = (current_dist * dist_a_mid.T).sum()
            x_int_z_val = np.exp(0.5 * (1 + 1 / config.theta_l)**2 * config.sigma_z ** 2)
            x_int_z = torch.full_like(x_z0, x_int_z_val)
            
            x_w0, x_l0, x_r0, x_y0 = self.calculate_aggregates(x_tfp0, x_z0, x_a0_total, x_int_z)
            
            # 记录数据
            x_r0_path.append(x_r0[0,0].item())
            x_tfp0_path.append(x_tfp0[0,0].item())

            # 预测下一期资产
            x_y0_policy = self.predict_model(x_x0_policy_sd, 'policy')
            x_a1 = x_y0_policy[:, 0].unsqueeze(1) * (a_max - a_min)
            
            # -----------------------------------------------------------
            # 3. 状态更新 (Transition to t+1)
            # -----------------------------------------------------------
            x_a0 = x_a1 # 资产更新
            
            # 冲击更新 z'
            x_z0 = module_basic_v1.bounded_log_normal_samples(
                config.mu_z, config.sigma_z, z_min, z_max, num_agents
            ).unsqueeze(1).to(self.device)
            
            # TFP 更新 (Markov Chain)
            probs = tfp_transition[curr_tfp_idx]
            curr_tfp_idx = torch.multinomial(probs, 1).item()
            x_tfp0 = tfp_grid[curr_tfp_idx].view(1, 1).expand(num_agents, 1)

        # ==========================================
        # 绘图部分
        # ==========================================
        print("Simulation finished. Generating plots...")
        
        # 数据转换
        x_dist0_path = torch.cat(x_dist0_path, dim=0).cpu().detach().numpy() # [Steps, k_dist]
        dist_a_mid_np = dist_a_mid.squeeze().cpu().detach().numpy()
        n_burn = config.n_burn
        
        # ----------------------
        # Fig 1: 初始阶段分布演变
        # ----------------------
        fig21, ax21 = plt.subplots(figsize=(9, 6))
        num_plots = 5
        colors = plt.cm.viridis(np.linspace(0, 1, num_plots))
        markers = ['o', 'v', '^', '<', '>', 's']
        
        # 画前几步 (例如 0, 10, 20...)
        indices = range(0, min(x_n_sim, 50), 10) 
        for idx, i in enumerate(indices):
            if i >= x_dist0_path.shape[0]: break
            ax21.plot(dist_a_mid_np, x_dist0_path[i], 
                     color=colors[idx % len(colors)], 
                     marker=markers[idx % len(markers)], 
                     label=f't = {i}')
                     
        ax21.set_xlabel('Asset')
        ax21.set_ylabel('Density (Monte Carlo)')
        ax21.set_title('Distribution Evolution (Initial)')
        ax21.legend()
        fig21.savefig(f'figures/sim_path_initial_2d.png', dpi=300)
        plt.close(fig21)

        # ----------------------
        # Fig 2: 长期稳态分布演变
        # ----------------------
        fig22, ax22 = plt.subplots(figsize=(9, 6))
        # 画最后几步
        start_idx = max(0, x_n_sim - 50)
        indices = range(start_idx, x_n_sim, 10)
        
        for idx, i in enumerate(indices):
            if i >= x_dist0_path.shape[0]: break
            ax22.plot(dist_a_mid_np, x_dist0_path[i], 
                     color=colors[idx % len(colors)], 
                     marker=markers[idx % len(markers)], 
                     label=f't = {i}')
        
        ax22.set_xlabel('Asset')
        ax22.set_ylabel('Density (Monte Carlo)')
        ax22.set_title('Distribution Evolution (Late Stage)')
        ax22.legend()
        fig22.savefig(f'figures/sim_path_2d.png', dpi=300)
        plt.close(fig22)

        # ----------------------
        # Fig 3: 宏观变量路径 [修复了 NameError]
        # ----------------------
        fig3, ax1 = plt.subplots(figsize=(9, 6)) # <--- 这一行定义了 ax1
        ax2 = ax1.twinx()
        
        steps = np.arange(x_n_sim)[n_burn:]
        r_data = x_r0_path[n_burn:]
        tfp_data = x_tfp0_path[n_burn:]
        
        if len(steps) > 0:
            ax1.plot(steps, r_data, 'k-.', label='Interest Rate (r)')
            ax2.plot(steps, tfp_data, 'r-*', label='TFP')
            
            ax1.set_xlabel('Simulation Steps')
            ax1.set_ylabel('Interest Rate', color='k')
            ax2.set_ylabel('TFP', color='r')
            # 合并图例
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc="upper right")
            
            fig3.savefig(f'figures/sim_path_aggregates.png')
        
        plt.close(fig3)
        return