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
        pdf_sampler = self.get_pdf_sampler()
        x_beta = config.beta
        all_variables = ["z", "a", "dist_a"]
        z_index = all_variables.index("z")
        z_min, z_max = self.ranges[z_index]
        a_index = all_variables.index("a")
        a_min, a_max = self.ranges[a_index]
        tfp_grid = torch.tensor(config.tfp_grid).view(-1, 1)
        tfp_transition = torch.tensor(config.tfp_transition).view(config.n_tfp, config.n_tfp)

        x_z1, x_a1, x_dist1 = self.extract_state_variables(x_batch)
        x_i_tfp1 = torch.randint(low=0, high=config.n_tfp, size=x_z1.size())
        x_tfp1 = tfp_grid[x_i_tfp1.squeeze()].view(-1, 1).to(self.device)

        # Initialize path tensors
        x_i_tfp0_path, x_tfp0_path, x_z0_path, x_a0_path, x_dist0_path = [], [], [], [], []
        x_w0_path, x_l0_path, x_r0_path, x_y0_path = [], [], [], []

        for sim_step in range(x_n_sim):
            x_i_tfp0, x_tfp0, x_z0, x_a0, x_dist0 = x_i_tfp1.clone(), x_tfp1.clone(), x_z1.clone(), x_a1.clone(), x_dist1.clone()
            # Record the values at each step
            x_i_tfp0_path.append(x_i_tfp0)
            x_tfp0_path.append(x_tfp0)
            x_z0_path.append(x_z0)
            x_a0_path.append(x_a0)
            x_dist0_path.append(x_dist0)

            x_x0_policy = torch.cat([x_z0, x_a0, x_dist0], dim=1).to(self.device)
            x_x0_policy_sd = module_basic_v1.normalize_inputs(x_x0_policy, config.bounds)
            x_x0_policy_sd = torch.cat((x_tfp0, x_x0_policy_sd), dim=1)

            x = 1 + 1 / config.theta_l
            x_int_z1 = torch.exp(torch.tensor(1 / 2 * x * x * config.sigma_z ** 2))
            x_int_z = torch.full_like(x_z0, x_int_z1).to(self.device)
            x_a0_total = (x_dist0 * dist_a_mid.T).sum(dim=1, keepdim=True)
            x_w0, x_l0, x_r0, x_y0 = self.calculate_aggregates(x_tfp0, x_z0, x_a0_total, x_int_z)

            # Add aggregate values to the lists
            x_w0_path.append(x_w0)
            x_l0_path.append(x_l0)
            x_r0_path.append(x_r0)
            x_y0_path.append(x_y0)

            if isinstance(self.model, torch.nn.DataParallel):
                x_y0_policy = self.model.module.f_policy(x_x0_policy_sd)
            else:
                x_y0_policy = self.model.f_policy(x_x0_policy_sd)

            x_a1 = x_y0_policy[:, 0].unsqueeze(1) * (a_max - a_min)
            x_c0 = (1 + x_r0) * x_a0 + x_w0 * x_l0 * x_z0 - x_a1

            z_lower_bound, z_upper_bound = config.get_z_bounds()
            x_z1 = module_basic_v1.bounded_log_normal_samples(config.mu_z, config.sigma_z, z_lower_bound, z_upper_bound,
                                                              x_z0.size(0))
            x_z1 = x_z1.unsqueeze(1).to(self.device)
            # First, gather the transition probabilities for each current state in x_tfp0
            transition_probs_for_x_tfp0 = tfp_transition[x_i_tfp0.squeeze()]
            x_i_tfp1 = torch.multinomial(transition_probs_for_x_tfp0, 1).unsqueeze(1)
            random_tfp = torch.normal(mean=0, std=config.eps_tfp, size=(1,)).to(self.device)
            x_tfp1 = torch.exp(config.rho_tfp * torch.log(x_tfp0) + random_tfp)

            if config.i_dist == 1:
                n_batch, n_dim = x_dist0.shape
                x_dist_g_all = self.calculate_G_batch(dist_a_mid, dist_a_mesh, x_dist0, x_tfp0)
                x_dist0_reshaped = x_dist0.view(n_batch, n_dim, 1)
                x_dist1 = torch.bmm(x_dist0_reshaped.transpose(1, 2), x_dist_g_all).transpose(1, 2).view(n_batch, n_dim)

            elif config.i_dist == 0:
                x_dist1 = x_dist0
                n_batch, n_dim = x_dist0.shape
                x_dist1 = pdf_sampler.generate_samples_a_pdf(n_batch, n_dim)

        # Convert lists to tensors
        x_z0_path = torch.cat(x_z0_path, dim=0)
        x_a0_path = torch.cat(x_a0_path, dim=0)
        x_dist0_path = torch.cat(x_dist0_path, dim=0)

        # Ensure both tensors have the same dtype
        dist_a_mid_tensor = dist_a_mid.squeeze().to(dtype=torch.float32)
        sim_steps_tensor = torch.arange(x_n_sim, dtype=torch.float32)

        # Plotting
        n_burn = config.n_burn  # Specify the number of elements to truncate

        ####################################################################
        # Fig. 1: path for distribution -- 2D, initial periods
        ####################################################################
        fig21, ax21 = plt.subplots(figsize=(9, 6))

        # Determine the number of plots you will create
        num_plots = 5  # Adjust this to change the number of steps you plot

        # Generate a color for each plot
        colors = plt.cm.viridis(np.linspace(0, 1, num_plots))

        # Define a list of markers you want to use
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']

        # Calculate the indices based on the desired number of plots and steps between each plot
        # We are now starting from the beginning of the dataset
        indices = range(1, min(x_dist0_path.shape[0], num_plots * config.n_sim_step), config.n_sim_step)

        # Loop through selected elements of x_dist0_path
        for idx, (i, marker) in enumerate(zip(indices, markers)):
            X = dist_a_mid_tensor.detach().cpu().numpy()
            Y = x_dist0_path[i, :].detach().cpu().numpy()
            # Use the index to cycle through markers and colors, avoiding an out-of-index error
            ax21.plot(X, Y, color=colors[idx % len(colors)], marker=marker, label=f't = {i}')

        ax21.set_xlabel('Asset')
        ax21.set_ylabel('Density')
        ax21.legend(loc='best', ncol=2)  # Adjust number of columns in legend if needed

        #plt.show()
        fig21.savefig(f'figures/sim_path_initial_2d.png', dpi=600)
        plt.close(fig21)

        ####################################################################
        # Fig. 2: path for distribution -- 2D
        ####################################################################
        fig22, ax22 = plt.subplots(figsize=(9, 6))

        # Determine the number of plots you will create
        num_plots = 5  # Adjust this to change the number of steps you plot

        # Generate a color for each plot
        colors = plt.cm.viridis(np.linspace(0, 1, num_plots))

        # Define a list of markers you want to use
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']

        # Calculate the starting index based on the desired number of plots and steps between each plot
        start_index = max(0, x_dist0_path.shape[0] - num_plots * config.n_sim_step)
        indices = range(start_index, x_dist0_path.shape[0], config.n_sim_step)

        # Loop through selected elements of x_dist0_path
        for idx, (i, marker) in enumerate(zip(indices, markers)):
            X = dist_a_mid_tensor.detach().cpu().numpy()
            Y = x_dist0_path[i, :].detach().cpu().numpy()
            # Use the index to cycle through markers and colors, avoiding an out-of-index error
            ax22.plot(X, Y, color=colors[idx % len(colors)], marker=marker, label=f't = {i}')

        ax22.set_xlabel('Asset')
        ax22.set_ylabel('Density')
        ax22.legend(loc='best', ncol=2)  # Adjust number of columns in legend if needed

        #plt.show()
        fig22.savefig(f'figures/sim_path_2d.png', dpi=600)
        plt.close(fig22)


        # New code for third plot
        fig3, ax1 = plt.subplots(figsize=(9, 6))
        ax2 = ax1.twinx()

        # Convert lists to tensors or arrays for plotting

        x_w0_array = torch.cat(x_w0_path, dim=0).detach().cpu().numpy()[n_burn:]
        x_l0_array = torch.cat(x_l0_path, dim=0).detach().cpu().numpy()[n_burn:]
        x_r0_array = torch.cat(x_r0_path, dim=0).detach().cpu().numpy()[n_burn:]
        x_y0_array = torch.cat(x_y0_path, dim=0).detach().cpu().numpy()[n_burn:]
        x_tfp0_array = torch.cat(x_tfp0_path, dim=0).detach().cpu().numpy()[n_burn:]
        sim_steps_array = np.arange(x_n_sim)[n_burn:]

        # Plot each aggregate
        #ax3.plot(sim_steps_array, x_w0_array, label='x_w0')
        #ax3.plot(sim_steps_array, x_l0_array, label='x_l0')
        #ax3.plot(sim_steps_array, x_y0_array, label='x_y0')

        ax1.plot(sim_steps_array, x_r0_array, 'k-.', label='x_r0')  # Green line with circle markers
        ax2.plot(sim_steps_array, x_tfp0_array, 'r-*', label='x_tfp0')  # Magenta line with triangle markers

        # Labels and Legend
        ax1.set_xlabel('Simulation Steps')
        ax1.set_ylabel('x_r0', color='b')
        ax2.set_ylabel('x_tfp0', color='r')
        fig3.legend(loc="upper right")

        #plt.show()
        fig3.savefig(f'figures/sim_path_aggregates.png')
        plt.close(fig3)

        return