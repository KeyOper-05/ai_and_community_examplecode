"""
03/02/2025,
basic module to setup the model.
   1. the NNs for policy and value functions;
   2. sampling from the state space;
   3. plot equm functions.
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
from torch.distributions.log_normal import LogNormal
from scipy.interpolate import griddata

# Load configuration from the JSON file
class Config:
    def __init__(self, config_file):
        with open(config_file, "r") as json_file:
            self.config = json.load(json_file)

        # extract all parameters
        for key, value in self.config.items():
            setattr(self, key, value)

    def get_z_bounds(self):
        z_bounds = self.bounds.get("z", {})
        return z_bounds.get("min"), z_bounds.get("max")


class MyModel(nn.Module):
    """
    define the neural networks:
    """
    class MyReLU(nn.Module):
        def forward(self, x):
            return nn.functional.relu(x) + 1

    def __init__(self, n_input, n_p_output, n_v_output, n1_p, n2_p, n1_v, n2_v):
        super(MyModel, self).__init__()

        # Define MyReLU as a Module
        self.my_relu = self.MyReLU()

        # Define policy function
        self.policy_func = nn.Sequential(
            nn.Linear(n_input, n1_p),
            nn.ReLU(),
            nn.Linear(n1_p, n2_p),
            nn.ReLU(),
            nn.Linear(n2_p, n_p_output),
            nn.Sigmoid()
        )

        # Define value function
        self.value_func = nn.Sequential(
            nn.Linear(n_input, n1_v),
            nn.ReLU(),
            nn.Linear(n1_v, n2_v),
            nn.ReLU(),
            nn.Linear(n2_v, n_v_output)
        )

    def f_policy(self, x):
        return self.policy_func(x)

    def f_value(self, x):
        return self.value_func(x)


class DomainSampling:
    def __init__(self, ranges, device=None):
        self.ranges = ranges
        self.device = device
        self.config = Config("config_v1.json")

    def generate_samples(self, num_samples, num_k):
        '''num_k is the dim of the last columns needs to be normalized as they represent the distribution'''
        keys = ["z", "a"]
        ranges = [(self.config.bounds[key]["min"], self.config.bounds[key]["max"]) for key in keys]
        # Extend the ranges for dist_a_pdf
        for dist_a_pdf in self.config.dist_a_pdf:
            extended_min = dist_a_pdf * (1 - self.config.dist_a_band)
            extended_max = dist_a_pdf * (1 + self.config.dist_a_band)
            ranges.append((extended_min, extended_max))

        n_states = len(ranges)
        samples_tensor = torch.empty(num_samples, n_states, dtype=torch.float32, device=self.device)

        # Parameters for the lognormal distribution
        mu, sigma = self.config.mu_z, self.config.sigma_z
        all_variables = ["z", "a", "dist_a"]
        z_index = all_variables.index("z")
        z_min, z_max = self.ranges[z_index]
        log_normal = LogNormal(mu, sigma)

        # Efficient generation of the first column (z)
        z_samples = torch.zeros(num_samples, dtype=torch.float32, device=self.device)
        for i in range(num_samples):
            sample_accepted = False
            attempts = 0
            while not sample_accepted and attempts < 1000:  # Limit the number of attempts
                sample = log_normal.sample()
                if z_min <= sample <= z_max:
                    z_samples[i] = sample
                    sample_accepted = True
                attempts += 1
            if not sample_accepted:
                raise ValueError(f"Unable to generate a valid sample in {attempts} attempts.")

        samples_tensor[:, 0] = z_samples

        # Generate other columns
        for i, (lower, upper) in enumerate(ranges[1:], start=1):
            samples_tensor[:, i] = torch.rand(num_samples, dtype=torch.float32, device=self.device) * (
                        upper - lower) + lower

        # Normalize the last num_k columns
        sum_last_k = samples_tensor[:, -num_k:].sum(dim=1, keepdim=True)
        samples_tensor[:, -num_k:] /= sum_last_k

        return samples_tensor

    def generate_samples_a_pdf(self, n_batch, num_k):
        # Extract the range for the last num_k columns from the configuration
        ranges = []
        for dist_a_pdf in self.config.dist_a_pdf[-num_k:]:  # Get the last num_k distributions
            extended_min = dist_a_pdf * (1 - self.config.dist_a_band_path)
            extended_max = dist_a_pdf * (1 + self.config.dist_a_band_path)
            ranges.append((extended_min, extended_max))

        # Generate samples for the last num_k columns
        samples_tensor = torch.empty(n_batch, num_k, dtype=torch.float32, device=self.device)
        for i, (lower, upper) in enumerate(ranges):
            samples_tensor[:, i] = torch.rand(n_batch, dtype=torch.float32, device=self.device) * (
                        upper - lower) + lower

        # Normalize the samples
        sum_last_k = samples_tensor.sum(dim=1, keepdim=True)
        normalized_samples = samples_tensor / sum_last_k

        return normalized_samples

    def dist_enforce_boundaries(self, x_dist1, a_pdf_penalty):
        n_batch, n_dim = x_dist1.shape

        # Check if dist_a_pdf has the same number of elements as the columns of x_dist1
        if len(self.config.dist_a_pdf) != n_dim:
            raise ValueError("Length of dist_a_pdf must be equal to the number of columns in x_dist1")

        # Convert dist_a_pdf to a tensor and prepare for broadcasting
        dist_a_pdf_tensor = torch.tensor(self.config.dist_a_pdf, device=self.device)

        # Calculate extended_min and extended_max
        extended_min = dist_a_pdf_tensor * (1 - self.config.dist_a_band)
        extended_max = dist_a_pdf_tensor * (1 + self.config.dist_a_band)

        # Enforce boundaries for each element
        penalty_below = a_pdf_penalty * (extended_min[None, :] - x_dist1).clamp(min=0)
        penalty_above = a_pdf_penalty * (x_dist1 - extended_max[None, :]).clamp(min=0)
        total_penalty = a_pdf_penalty * (penalty_below + penalty_above).sum(dim=1).unsqueeze(-1)
        x_dist1_clamped = x_dist1.clamp(min=extended_min[None, :], max=extended_max[None, :])

        return x_dist1_clamped, total_penalty


class plot_equm_funcs:
    def __init__(self, num_samples, num_k, dist_a_mid, model, device=None):
        self.num_samples = num_samples
        self.num_k = num_k
        self.dist_a_mid = dist_a_mid
        self.model = model
        self.device = device
        self.config = Config("config_v1.json")  # Assuming Config is defined elsewhere
        self.initialize_ranges()

    def initialize_ranges(self):
        # Initialize the ranges based on the config
        keys = ["z", "a"]
        self.ranges = [(self.config.bounds[key]["min"], self.config.bounds[key]["max"]) for key in keys]

    def generate_samples_fixed_k(self):
        config = Config("config_v1.json")
        keys = ["z", "a"]
        ranges = [(config.bounds[key]["min"], config.bounds[key]["max"]) for key in keys]
        for dist_a_pdf in config.dist_a_pdf:
            extended_min = dist_a_pdf * (1 - config.dist_a_band)
            extended_max = dist_a_pdf * (1 + config.dist_a_band)
            ranges.append((extended_min, extended_max))

        n_states = len(ranges)
        samples_tensor = torch.empty(self.num_samples, n_states, dtype=torch.float32, device=self.device)

        mu, sigma = config.mu_z, config.sigma_z
        z_lower_bound, z_upper_bound = config.get_z_bounds()
        log_normal = LogNormal(mu, sigma)

        # Generate samples for each row
        for i in range(self.num_samples):
            # Generate z within bounds
            z_sample = log_normal.sample()
            while not (z_lower_bound <= z_sample <= z_upper_bound):
                z_sample = log_normal.sample()
            samples_tensor[i, 0] = z_sample

            # Generate a and first row of the remaining columns
            for j, (lower, upper) in enumerate(ranges[1:], start=1):
                samples_tensor[i, j] = torch.rand(1, dtype=torch.float32, device=self.device) * (upper - lower) + lower

        # Normalize and replicate last num_k columns
        sum_last_k = samples_tensor[0, -self.num_k:].sum()
        samples_tensor[:, -self.num_k:] = samples_tensor[0, -self.num_k:] / sum_last_k

        return samples_tensor

    def create_plot(self):
        config = Config("config_v1.json")
        all_variables = ["z", "a", "dist_a"]
        z_index = all_variables.index("z")
        z_min, z_max = self.ranges[z_index]
        a_index = all_variables.index("a")
        a_min, a_max = self.ranges[a_index]

        # Generate samples
        samples_tensor = self.generate_samples_fixed_k()
        x_z0, x_a0, x_dist0 = self.extract_state_variables(samples_tensor)

        # Compute x_c0 using your model
        x_x0_policy = torch.cat([x_z0, x_a0, x_dist0], 1).to(self.device)

        x_i_tfp0 = torch.zeros_like(x_z0).long().to(self.device)
        tfp_grid = torch.tensor(config.tfp_grid).view(-1, 1).to(self.device)
        x_tfp0 = tfp_grid[x_i_tfp0.squeeze()].view(-1, 1)

        x_x0_policy_sd = normalize_inputs(x_x0_policy, config.bounds)
        x_x0_policy_sd = torch.cat((x_tfp0, x_x0_policy_sd), dim=1)

        # compute the aggregate variables:
        # Compute the value of int(eps_z**(1/theta))
        sum_z = torch.mean(x_z0 ** (1 + 1 / config.theta_l))
        x_int_z = torch.full_like(x_z0, sum_z)
        x_a0_total = (x_dist0 * self.dist_a_mid.T).sum(dim=1, keepdim=True)

        x_w0_1 = (1 - config.alpha) * (x_a0_total / x_int_z) ** config.alpha
        x_w0 = config.psi_l ** (config.alpha / config.theta_l) * x_w0_1 ** (
                    config.theta_l / (config.alpha + config.theta_l))
        x_l0 = (x_w0 * x_z0 / config.psi_l) ** (1 / config.theta_l)
        x_r0 = config.alpha * (x_w0 / (1 - config.alpha)) ** ((config.alpha - 1) / config.alpha) - config.delta
        x_y0 = x_a0_total ** config.alpha

        # compute the endogenous state variables:
        if isinstance(self.model, torch.nn.DataParallel):
            x_y0_policy = self.model.module.f_policy(x_x0_policy_sd)
        else:
            x_y0_policy = self.model.f_policy(x_x0_policy_sd)

        # x_y0_policy = self.model.f_policy(x_x0_policy_sd)
        x_a1 = x_y0_policy[:, 0].unsqueeze(1) * (a_max - a_min)
        x_c0 = (1 + x_r0) * x_a0 + x_w0 * x_l0 * x_z0 - x_a1

        if isinstance(self.model, torch.nn.DataParallel):
            x_v = self.model.module.f_value(x_x0_policy_sd)[:, 0].unsqueeze(1)
        else:
            x_v = self.model.f_value(x_x0_policy_sd)[:, 0].unsqueeze(1)

        # x_v = self.model.f_value(x_x0_policy_sd)[:, 0].unsqueeze(1)

        # Extract the first two columns for the scatter plot
        x1 = x_z0.cpu().numpy()
        x2 = x_a0.cpu().numpy()
        z_c = x_c0.detach().cpu().numpy()
        z_a = x_a1.detach().cpu().numpy()
        z_v = x_v.detach().cpu().numpy()

        # Create scatter plots
        # Scatter plot for z_c
        fig1 = plt.figure(figsize=(9, 9))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.scatter(x1, x2, z_c, c='blue', marker='o')
        ax1.set_xlabel('z')
        ax1.set_ylabel('a')
        ax1.set_zlabel('c')
        plt.savefig(f'figures/scatter_policy_c.png')
        #plt.show()
        plt.close()

        # Scatter plot for z_a
        fig2 = plt.figure(figsize=(9, 9))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(x1, x2, z_a, c='red', marker='o')
        ax2.set_xlabel('z')
        ax2.set_ylabel('a')
        ax2.set_zlabel('a+')
        plt.savefig(f'figures/scatter_policy_a1.png')
        #plt.show()
        plt.close()

        # Scatter plot for z_v
        fig3 = plt.figure(figsize=(9, 9))
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.scatter(x1, x2, z_v, c='green', marker='o')
        ax3.set_xlabel('z')
        ax3.set_ylabel('a')
        ax3.set_zlabel('V')
        plt.savefig(f'figures/scatter_value.png')
        #plt.show()
        plt.close()

    def extract_state_variables(self, x_sample):
        """
        Extract state variables from the input batch.
        """
        device = self.device
        x_z = x_sample[:, 0].unsqueeze(1).to(device)
        x_a = x_sample[:, 1].unsqueeze(1).to(device)
        x_dist = x_sample[:, 2:].to(device)  # Extracts all columns from the 3rd to the last
        return x_z, x_a, x_dist

def normalize_inputs(inputs, normalization_bounds):
    '''
    Normalize the first two columns of the input by their min and max values.
    Normalize the last k columns so that their sum equals 1.
    '''
    n, total_columns = inputs.shape
    outputs = torch.zeros_like(inputs)

    # Normalize first two columns
    for i in range(2):
        key = list(normalization_bounds.keys())[i]
        min_val = normalization_bounds[key]["min"]
        max_val = normalization_bounds[key]["max"]
        outputs[:, i] = (inputs[:, i] - min_val) / (max_val - min_val)

    # Normalize last k columns by their row sum
    k_columns = inputs[:, 2:]
    row_sums = k_columns.sum(dim=1, keepdim=True)
    normalized_k_columns = k_columns / row_sums
    outputs[:, 2:] = normalized_k_columns

    return outputs


def bounded_log_normal_samples(mu, sigma, lower_bound, upper_bound, num_samples):
    """
    Draw n_samples from a log-normal distribution bounded by lower_bound and upper_bound.
    """
    log_normal = LogNormal(mu, sigma)
    samples = torch.empty(num_samples)

    for i in range(num_samples):
        sample_accepted = False
        while not sample_accepted:
            sample = log_normal.sample().item()
            if lower_bound <= sample <= upper_bound:
                samples[i] = sample
                sample_accepted = True
    return samples

def enforce_boundary(x, min_val, max_val, lambda_penalty):
    '''
    reset the variable if it exceeds its boundary
    '''
    penalty = 0
    # Calculate penalties for values below the minimum and above the maximum
    penalty_below = lambda_penalty * (min_val - x).clamp(min=0)
    penalty_above = lambda_penalty * (x - max_val).clamp(min=0)

    # Total penalty for the tensor
    total_penalty = lambda_penalty * penalty_below + (1-lambda_penalty) * penalty_above

    # Clamp the tensor values within the boundary
    x_clamped = x.clamp(min=min_val, max=max_val)

    return x_clamped, total_penalty


def contains_nan(tensor):
    return torch.isnan(tensor).any().item()


