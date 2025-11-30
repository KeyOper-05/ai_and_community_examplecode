"""
module_training_euler.py
Implements the Deep Learning method (minimizing Euler Residuals) for solving the KS model.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import module_basic_v1
import module_obj_bellman_v1  # 复用其中的物理计算逻辑

# 加载配置
config = module_basic_v1.Config("config_v1.json")

class EulerTrainer:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device
        # 复用原代码的辅助类，用于计算聚合变量和分布演变
        self.obj_helper = module_obj_bellman_v1.define_objective(model, device)
        
        # 定义优化器 (只优化 Policy Network)
        # 注意：这里我们假设 MyModel 里的 policy_func 输出的是下一期资产 a'
        if isinstance(self.model, torch.nn.DataParallel):
            self.policy_net = self.model.module.policy_func
        else:
            self.policy_net = self.model.policy_func
            
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr_policy)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=config.lr_factor, patience=config.lr_patience)

    def train_euler(self, num_epochs, batch_size, dist_a_mid, dist_a_mesh):
        """
        Main training loop for the Euler method.
        """
        losses = []
        domain_sampler = module_basic_v1.DomainSampling(self.obj_helper.ranges, device=self.device)
        
        # 将 dist_a_mid 转换为 Tensor 以便计算
        dist_a_mid_tensor = torch.tensor(dist_a_mid, device=self.device).view(-1, 1)

        print("Starting Euler Equation Training...")
        
        for epoch in range(num_epochs):
            # 1. 采样状态 (State Sampling)
            # 使用现有的采样器生成 (z, a, dist)
            x_data = domain_sampler.generate_samples(num_samples=config.num_samples_policy, num_k=config.k_dist)
            dataset = TensorDataset(x_data)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            epoch_loss = 0.0
            total_batches = 0
            
            for batch_x, in data_loader:
                batch_x = batch_x.to(self.device)
                
                # 2. 计算欧拉方程损失 (Euler Loss)
                loss = self.calculate_euler_residuals(batch_x, dist_a_mid_tensor, dist_a_mesh)
                
                # 3. 反向传播 (Backpropagation)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                total_batches += 1
            
            avg_loss = epoch_loss / total_batches
            losses.append(np.log(avg_loss))
            self.scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Log Euler Loss = {np.log(avg_loss):.4f}")

        # 绘图
        plt.figure()
        plt.plot(losses)
        plt.title('Euler Equation Residuals (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Log MSE Loss')
        plt.savefig('figures/loss_euler.png')
        plt.close()
        
        return self.model

    def calculate_euler_residuals(self, x_batch, dist_a_mid, dist_a_mesh):
        """
        Calculate the MSE of the Euler equation residuals.
        Residual = u'(c_t) - beta * E_t [ (1 + r_{t+1}) * u'(c_{t+1}) ]
        """
        # ==========================
        # Period t (Current Period)
        # ==========================
        # 1. 解析当前状态
        # x_batch: [batch, z, a, dist...]
        x_z0, x_a0, x_dist0 = self.obj_helper.extract_state_variables(x_batch)
        
        # 这里的 TFP 逻辑需要参考原代码，假设 x_z0 包含了 idiosyncratic z，还需要 aggregate shock TFP?
        # 原代码逻辑里 x_batch 的第0列是 z (idiosyncratic)。
        # 但 aggregate TFP (x_tfp) 通常是独立采样的。为了简化，这里我们从 batch 里重新生成或提取。
        # 参考 obj_sim_value，我们需要生成当前 TFP 和下一期 TFP。
        
        n_batch = x_z0.size(0)
        tfp_grid = torch.tensor(config.tfp_grid, device=self.device).view(-1, 1)
        tfp_transition = torch.tensor(config.tfp_transition, device=self.device)
        
        # 随机采样当前的 Aggregate TFP (x_tfp0)
        x_i_tfp0 = torch.randint(config.n_tfp, (n_batch, 1), device=self.device)
        x_tfp0 = tfp_grid[x_i_tfp0.squeeze()].view(-1, 1)

        # 2. 宏观聚合量 (Aggregates at t)
        # 计算 total capital K
        x_a0_total = (x_dist0 * dist_a_mid.T).sum(dim=1, keepdim=True)
        # 复用原代码计算 w0, r0 (注意：原代码 calculate_aggregates 需要 helper)
        # 我们这里为了简便，直接把 calculate_aggregates 的逻辑拿过来，或者调用 helper
        x = 1 + 1 / config.theta_l
        x_int_z1 = torch.exp(torch.tensor(1 / 2 * x * x * config.sigma_z ** 2))
        x_int_z = torch.full_like(x_z0, x_int_z1).to(self.device)
        
        x_w0, x_l0, x_r0, x_y0 = self.obj_helper.calculate_aggregates(x_tfp0, x_z0, x_a0_total, x_int_z)

        # 3. 决策 (Policy Choice at t)
        # 构造输入向量
        x_x0_policy = torch.cat([x_z0, x_a0, x_dist0], dim=1)
        x_x0_policy_sd = module_basic_v1.normalize_inputs(x_x0_policy, config.bounds)
        x_x0_policy_sd = torch.cat((x_tfp0, x_x0_policy_sd), dim=1)
        
        # 预测下一期资产 a' (归一化输出 -> 实际值)
        # 注意：这里假设 model.f_policy 输出是归一化的 a'
        x_y0_policy = self.obj_helper.predict_model(x_x0_policy_sd, 'policy')
        a_min, a_max = config.bounds["a"]["min"], config.bounds["a"]["max"]
        x_a1 = x_y0_policy[:, 0].unsqueeze(1) * (a_max - a_min) # 解归一化
        
        # 4. 计算当前消费 c_t
        x_c0 = (1 + x_r0) * x_a0 + x_w0 * x_l0 * x_z0 - x_a1
        x_c0 = torch.maximum(x_c0, torch.tensor(config.u_eps)) # 保证 c > 0

        # ==========================
        # Period t+1 (Next Period)
        # ==========================
        # 5. 状态转移 (Transition)
        # A. Idiosyncratic Shock z': LogNormal transition
        z_min, z_max = config.bounds["z"]["min"], config.bounds["z"]["max"]
        x_z1 = module_basic_v1.bounded_log_normal_samples(config.mu_z, config.sigma_z, z_min, z_max, n_batch)
        x_z1 = x_z1.unsqueeze(1).to(self.device)
        
        # B. Aggregate Shock TFP': Markov transition
        transition_probs = tfp_transition[x_i_tfp0.squeeze()]
        x_i_tfp1 = torch.multinomial(transition_probs, 1)
        x_tfp1 = tfp_grid[x_i_tfp1.squeeze()].view(-1, 1)
        
        # C. Distribution Gamma': 使用 calculate_G_batch 计算分布演变
        # 这一步是 KS 模型 DL 方法的关键：利用当前 Policy 预测整个分布的移动
        if config.i_dist == 1:
            x_dist_g_all = self.obj_helper.calculate_G_batch(dist_a_mid_tensor, dist_a_mesh, x_dist0, x_tfp0)
            x_dist0_reshaped = x_dist0.view(n_batch, config.k_dist, 1)
            # 矩阵乘法得到下一期分布
            x_dist1 = torch.bmm(x_dist0_reshaped.transpose(1, 2), x_dist_g_all).transpose(1, 2).view(n_batch, config.k_dist)
            # 强边界约束
            x_dist1, _ = domain_sampler.dist_enforce_boundaries(x_dist1, config.a_pdf_penalty)
        else:
            x_dist1 = x_dist0 # 如果不更新分布（仅测试用）

        # 6. 宏观聚合量 (Aggregates at t+1)
        x_a1_total = (x_dist1 * dist_a_mid.T).sum(dim=1, keepdim=True)
        # 注意：这里用 x_z1 (个体的z) 和 x_tfp1 (宏观的tfp)
        x_w1, x_l1, x_r1, x_y1 = self.obj_helper.calculate_aggregates(x_tfp1, x_z1, x_a1_total, x_int_z) # int_z 近似不变

        # 7. 决策 (Policy Choice at t+1)
        # 我们需要预测 k_{t+2} (即 x_a2) 来算出 c_{t+1}
        x_x1_policy = torch.cat([x_z1, x_a1, x_dist1], dim=1)
        x_x1_policy_sd = module_basic_v1.normalize_inputs(x_x1_policy, config.bounds)
        x_x1_policy_sd = torch.cat((x_tfp1, x_x1_policy_sd), dim=1)
        
        # 使用同一个 Policy Net 预测
        x_y1_policy = self.obj_helper.predict_model(x_x1_policy_sd, 'policy')
        x_a2 = x_y1_policy[:, 0].unsqueeze(1) * (a_max - a_min)
        
        # 8. 计算下一期消费 c_{t+1}
        x_c1 = (1 + x_r1) * x_a1 + x_w1 * x_l1 * x_z1 - x_a2
        x_c1 = torch.maximum(x_c1, torch.tensor(config.u_eps))

        # ==========================
        # Euler Equation Loss
        # ==========================
        # u'(c) = c^(-sigma)
        lhs = x_c0 ** (-config.sigma)
        rhs = config.beta * (1 + x_r1 - config.delta) * (x_c1 ** (-config.sigma))
        
        # 定义残差：可以是 (LHS - RHS)^2，也可以是 (1 - RHS/LHS)^2 (Unit-free)
        # 通常 Unit-free 更稳定：
        residuals = 1 - rhs / lhs
        loss = torch.mean(residuals ** 2)
        
        return loss