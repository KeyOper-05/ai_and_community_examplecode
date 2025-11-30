"""
module_training_euler.py
Implements the Deep Learning method (minimizing Euler Residuals) for solving the KS model.
Corrected version with:
1. Explicit Expectation over TFP shocks.
2. Kuhn-Tucker conditions for borrowing constraints.
3. Consistent distribution updates.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import module_basic_v1
import module_obj_bellman_v1  # Reuse physical logic

# Load configuration
config = module_basic_v1.Config("config_v1.json")

class EulerTrainer:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device
        # Reuse helper class for aggregates and distribution evolution
        self.obj_helper = module_obj_bellman_v1.define_objective(model, device)
        
        # Optimizer: only optimize the policy network
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
        
        # Convert dist_a_mid to Tensor
        dist_a_mid_tensor = torch.tensor(dist_a_mid, device=self.device).view(-1, 1)

        print("Starting Euler Equation Training (with Explicit Expectation)...")
        
        for epoch in range(num_epochs):
            # 1. State Sampling
            x_data = domain_sampler.generate_samples(num_samples=config.num_samples_policy, num_k=config.k_dist)
            dataset = TensorDataset(x_data)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            epoch_loss = 0.0
            total_batches = 0
            
            for batch_x, in data_loader:
                batch_x = batch_x.to(self.device)
                
                # 2. Calculate Euler Residuals
                loss = self.calculate_euler_residuals(batch_x, dist_a_mid_tensor, dist_a_mesh)
                
                # 3. Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent explosion, common in Euler methods
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                total_batches += 1
            
            avg_loss = epoch_loss / total_batches
            losses.append(np.log(avg_loss + 1e-10)) # Avoid log(0)
            self.scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Log Euler Loss = {np.log(avg_loss + 1e-10):.4f}")

        # Plotting
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
        Calculates Euler residuals using fully vectorized operations.
        Replaces the slow for-loop over TFP branches with batch processing.
        """
        # ==========================
        # 1. Prepare Current State (t)
        # ==========================
        x_z0, x_a0, x_dist0 = self.obj_helper.extract_state_variables(x_batch)
        n_batch = x_z0.size(0)
        n_tfp = config.n_tfp

        # TFP grids and transitions
        tfp_grid = torch.tensor(config.tfp_grid, device=self.device).view(-1, 1) # [4, 1]
        tfp_transition = torch.tensor(config.tfp_transition, device=self.device).view(n_tfp, n_tfp)

        # Sample current TFP
        x_i_tfp0 = torch.randint(n_tfp, (n_batch, 1), device=self.device)
        x_tfp0 = tfp_grid[x_i_tfp0.squeeze()].view(-1, 1)

        # Aggregates at t
        x_a0_total = (x_dist0 * dist_a_mid.T).sum(dim=1, keepdim=True)
        # Precompute integral term (constant)
        x_int_z_val = np.exp(0.5 * (1 + 1 / config.theta_l)**2 * config.sigma_z ** 2)
        x_int_z = torch.full_like(x_z0, x_int_z_val)
        
        x_w0, x_l0, x_r0, x_y0 = self.obj_helper.calculate_aggregates(x_tfp0, x_z0, x_a0_total, x_int_z)

        # Policy at t
        x_x0_policy = torch.cat([x_z0, x_a0, x_dist0], dim=1)
        x_x0_policy_sd = module_basic_v1.normalize_inputs(x_x0_policy, config.bounds)
        x_x0_policy_sd = torch.cat((x_tfp0, x_x0_policy_sd), dim=1)
        
        x_y0_policy = self.obj_helper.predict_model(x_x0_policy_sd, 'policy')
        
        a_min, a_max = config.bounds["a"]["min"], config.bounds["a"]["max"]
        x_a1 = x_y0_policy[:, 0].unsqueeze(1) * (a_max - a_min) # Next Asset
        
        # Consumption at t
        x_c0 = (1 + x_r0) * x_a0 + x_w0 * x_l0 * x_z0 - x_a1
        x_c0 = torch.maximum(x_c0, torch.tensor(config.u_eps))

        # ==========================
        # 2. Prepare Next State (t+1) Common Parts
        # ==========================
        # Distribution Evolution (Run once for the whole batch)
        if config.i_dist == 1:
            x_dist_g_all = self.obj_helper.calculate_G_batch(dist_a_mid, dist_a_mesh, x_dist0, x_tfp0)
            x_dist0_reshaped = x_dist0.view(n_batch, config.k_dist, 1)
            x_dist1 = torch.bmm(x_dist0_reshaped.transpose(1, 2), x_dist_g_all).transpose(1, 2).view(n_batch, config.k_dist)
            # Boundary enforcement
            dist_sampler = self.obj_helper.get_pdf_sampler()
            x_dist1, _ = dist_sampler.dist_enforce_boundaries(x_dist1, config.a_pdf_penalty)
        else:
            x_dist1 = x_dist0

        # Idiosyncratic Shock z' (Sampled once)
        z_min, z_max = config.bounds["z"]["min"], config.bounds["z"]["max"]
        x_z1 = module_basic_v1.bounded_log_normal_samples(config.mu_z, config.sigma_z, z_min, z_max, n_batch)
        x_z1 = x_z1.unsqueeze(1).to(self.device)
        
        # Aggregates common inputs
        x_a1_total = (x_dist1 * dist_a_mid.T).sum(dim=1, keepdim=True)

        # ==========================
        # 3. Vectorized Expectation Calculation (The Speedup)
        # ==========================
        # Goal: Calculate E[RHS] by processing all 4 TFP branches in parallel
        
        # Expand inputs to shape [N * n_tfp, ...]
        # repeat_interleave: [s1, s1, s1, s1, s2, s2, s2, s2 ...]
        x_z1_rep = x_z1.repeat_interleave(n_tfp, dim=0) 
        x_a1_rep = x_a1.repeat_interleave(n_tfp, dim=0)
        x_dist1_rep = x_dist1.repeat_interleave(n_tfp, dim=0)
        x_a1_total_rep = x_a1_total.repeat_interleave(n_tfp, dim=0)
        x_int_z_rep = x_int_z.repeat_interleave(n_tfp, dim=0)
        
        # TFP grid repeated: [t0, t1, t2, t3, t0, t1, t2, t3 ...]
        x_tfp1_rep = tfp_grid.repeat(n_batch, 1)
        
        # Calculate Aggregates for all branches at once
        x_w1_all, x_l1_all, x_r1_all, _ = self.obj_helper.calculate_aggregates(x_tfp1_rep, x_z1_rep, x_a1_total_rep, x_int_z_rep)
        
        # Policy Prediction for all branches
        x_x1_policy_rep = torch.cat([x_z1_rep, x_a1_rep, x_dist1_rep], dim=1)
        x_x1_policy_sd_rep = module_basic_v1.normalize_inputs(x_x1_policy_rep, config.bounds)
        x_x1_policy_sd_rep = torch.cat((x_tfp1_rep, x_x1_policy_sd_rep), dim=1)
        
        # Forward pass (Batch size is now N * 4)
        x_y1_policy_rep = self.obj_helper.predict_model(x_x1_policy_sd_rep, 'policy')
        x_a2_rep = x_y1_policy_rep[:, 0].unsqueeze(1) * (a_max - a_min)
        
        # Consumption at t+1
        x_c1_rep = (1 + x_r1_all) * x_a1_rep + x_w1_all * x_l1_all * x_z1_rep - x_a2_rep
        x_c1_rep = torch.maximum(x_c1_rep, torch.tensor(config.u_eps))
        
        # Marginal Utility
        mu_next_rep = config.beta * (1 + x_r1_all - config.delta) * (x_c1_rep ** (-config.sigma))
        
        # Reshape back to [N, n_tfp] to compute expectation
        mu_next_reshaped = mu_next_rep.view(n_batch, n_tfp)
        
        # Get transition probabilities [N, n_tfp]
        probs = tfp_transition[x_i_tfp0.squeeze()]
        
        # Compute Expectation: dot product of probabilities and MU
        # sum(probs * mu, dim=1)
        rhs_expected = (probs * mu_next_reshaped).sum(dim=1, keepdim=True)

        # ==========================
        # 4. Loss Calculation
        # ==========================
        lhs = x_c0 ** (-config.sigma)
        residuals = 1 - rhs_expected / lhs
        
        # Borrowing Constraint Handling
        is_constrained = x_y0_policy[:, 0] < 1e-3
        want_to_borrow = residuals > 0 # LHS > RHS means current C is too low (MU too high), want to borrow more
        mask_ignore = is_constrained.unsqueeze(1) & want_to_borrow
        
        weights = torch.ones_like(residuals)
        weights[mask_ignore] = 0.0
        
        loss = torch.mean((weights * residuals) ** 2)
        return loss