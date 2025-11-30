"""
03/02/2025,
the training module, it will find the NN for policy and value functions. it takes "module_obj_bellman_v1" as input.
bellman function based method:
    will use "policy_bellman_training" to find policy function, and use "value_training" to fit a value function.
"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import module_basic_v1, module_obj_bellman_v1
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import pandas as pd

# import configuration data:
config = module_basic_v1.Config("config_v1.json")

class EqumTrainer:
    def __init__(self, num_epochs_initial, num_epochs_v, num_epochs_p, lr_initial, lr_v, lr_p, batch_size_p,
                 batch_size_v,
                 n_worker, model, pretrained_value=None, pretrained_policy=None, device=None, i_save=0):
        self.num_epochs_initial = num_epochs_initial
        self.num_epochs_v = num_epochs_v
        self.num_epochs_p = num_epochs_p
        self.lr_initial = lr_initial
        self.lr_v = lr_v
        self.lr_p = lr_p
        self.batch_size_p = batch_size_p
        self.batch_size_v = batch_size_v
        self.n_worker = n_worker
        self.device = device
        self.i_save = i_save
        self.pretrained_value = pretrained_value
        self.pretrained_policy = pretrained_policy
        self.model = model#.to(self.device)
        #if torch.cuda.device_count() > 1:
        #    print("Using", torch.cuda.device_count(), "GPUs!")
        #    self.model = nn.DataParallel(self.model)
        self.x_total_sample_p = int(config.num_samples_policy * config.num_samples_expand_p)
        self.x_total_sample_v = int(config.num_samples_value * config.num_samples_expand_v * config.num_epochs_draw)

        # Define ranges once
        keys = ["a", "z"]
        self.ranges = [(config.bounds[key]["min"], config.bounds[key]["max"]) for key in keys]
        # Extend the ranges for dist_a_mid
        for dist_a_pdf in config.dist_a_pdf:
            extended_min = dist_a_pdf * (1 - config.dist_a_band)
            extended_max = dist_a_pdf * (1 + config.dist_a_band)
            self.ranges.append((extended_min, extended_max))

    def get_domain_sampler(self):
        """
        Create and return an instance of the domain_sampling class.
        """
        return module_basic_v1.DomainSampling(self.ranges, device=self.device)

    def load_pretrained(self, target_func, pretrained_path):
        if pretrained_path is not None:
            # Load the pretrained state dictionary
            pretrained_state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))

            # Adjust the keys in the state dictionary
            adjusted_state_dict = {}
            prefix = f"{target_func}."  # Target function's prefix, e.g., 'policy_func.'
            for k, v in pretrained_state_dict.items():
                if k.startswith(prefix):
                    new_key = k[len(prefix):]  # Remove the prefix
                    adjusted_state_dict[new_key] = v

            # Load the adjusted state dictionary into the target function of the model
            getattr(self.model, target_func).load_state_dict(adjusted_state_dict)

    def load_pretrained_new(self, target_func, pretrained_path):
        if pretrained_path is not None:
            # Load the pretrained state dictionary
            pretrained_state_dict = torch.load(pretrained_path, map_location=self.device)

            # Obtain the target model component
            model_component = getattr(self.model, target_func, None)
            if model_component is None:
                raise AttributeError(f"Model does not have a component named '{target_func}'")

            # Load the state dictionary directly into the target component
            model_component.load_state_dict(pretrained_state_dict)

    def get_pretrained_model(self):
        # Define paths to the pretrained models
        pretrained_value_path = self.pretrained_value
        pretrained_policy_path = self.pretrained_policy

        if pretrained_policy_path is not None and os.path.exists(pretrained_policy_path):
            self.load_pretrained("policy_func", pretrained_policy_path)

        if pretrained_value_path is not None and os.path.exists(pretrained_value_path):
            self.load_pretrained("value_func", pretrained_value_path)

        return self.model

    def policy_bellman_training(self, x_n_sim, dist_a_mid, dist_a_mesh):
        """
        Find the optimal policy function via maximizing the Bellman equation.
        """
        self.load_pretrained("policy_func", self.pretrained_policy)

        domain_sampler = self.get_domain_sampler()
        equm_updater = module_obj_bellman_v1.define_objective(self.model, self.device)

        #policy_func = self.model.module.policy_func
        policy_func = self.model.module.policy_func if isinstance(self.model, torch.nn.DataParallel) else self.model.policy_func

        optimizer_policy = torch.optim.Adam(policy_func.parameters(), lr=self.lr_p, weight_decay=config.l2_penalty)
        scheduler = ReduceLROnPlateau(optimizer_policy, mode='min', factor=config.lr_factor,
                                      patience=config.lr_patience, verbose=False)

        # Initialize a list to store losses for the dynamic graph
        losses = []

        # Create a figure to display the losses
        plt.figure()

        # +++++++++++++++++++
        # training begins:
        # +++++++++++++++++++
        '''
        "obj_sim_value" takes the samples of state variables (X), value function as given, 
            uses the current policy functions to compute the value function (Y),
            the code below find a NN (for policy func.) maximizing Y. 
        '''

        #with tqdm(total=self.num_epochs_p, desc='P. Training-Bellman', position=0) as pbar_epoch:
        for epoch in range(self.num_epochs_p):
            # Generate samples
            x_data = domain_sampler.generate_samples(num_samples=self.x_total_sample_p, num_k=config.k_dist)
            dataset = TensorDataset(x_data)
            data_loader = DataLoader(dataset, batch_size=self.batch_size_p, shuffle=True, pin_memory=False,
                                     num_workers=self.n_worker)
            epoch_loss_sum = 0.0  # Variable to accumulate the sum of v_sim values
            total_samples = 0  # Counter for the total number of samples processed

            epoch_loss = 0.0
            for batch_x, in data_loader:
                batch_x = batch_x.to(self.device)
                v_sim, _ = equm_updater.obj_sim_value(batch_x, x_n_sim, dist_a_mid, dist_a_mesh)#, x_dist_g_all)

                # Accumulate the sum of v_sim values and count the samples
                epoch_loss_sum += v_sim.sum().item()
                total_samples += batch_x.size(0)

                # Backward pass and optimization
                optimizer_policy.zero_grad()
                v_sim.mean().backward()
                optimizer_policy.step()

                # Check nan values in the parameters and reset
                #nan_detected = any(torch.isnan(param).any() for param in self.model.policy_func.parameters())
                nan_detected = any(torch.isnan(param).any() for param in policy_func.parameters())

                if nan_detected:
                    # Option 1: Reinitialize the weights and biases
                    for layer in policy_func:
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()

            # Update the progress bar for the epochs
            #pbar_epoch.update(1)

            # Compute the mean loss for the epoch
            mean_epoch_loss = epoch_loss_sum / total_samples

            # Store the mean loss value
            losses.append(mean_epoch_loss)

            # Update learning rate based on the validation loss
            scheduler.step(epoch_loss_sum / len(data_loader))

        # training complete.

        # Update the graph with the final losses
        self.update_graph(losses, title='losses for policy--maximizing Bellman')
        #plt.show()
        plt.savefig(f'figures/loss_policy_bellman.png')
        plt.close()
        policy_func.state_dict()
        if self.i_save == 1:
            torch.save(EqumTrainer.remove_module_from_checkpoint(self.model.state_dict()),
                       f'models/trained_policy_nn_{config.model_number_output}.pth')

        return self.model

    def value_training(self, x_n_sim, dist_a_mid, dist_a_mesh):
        """
        Find the value function by evaluating the policy function via the Bellman function.
        """
        self.load_pretrained("value_func", self.pretrained_value)

        domain_sampler = self.get_domain_sampler()
        equm_updater = module_obj_bellman_v1.define_objective(self.model, self.device)

        # Access the policy function, compatible with DataParallel and without it
        value_func = self.model.module.value_func if isinstance(self.model, torch.nn.DataParallel) else self.model.value_func

        optimizer_value = torch.optim.Adam(value_func.parameters(), lr=self.lr_v)        
        scheduler = ReduceLROnPlateau(optimizer_value, mode='min', factor=config.lr_factor, patience=config.lr_patience,
                                      verbose=False)
        loss_function = nn.MSELoss()

        # Initialize a list to store losses
        losses = []

        for epoch in range(self.num_epochs_v):
            # Generate samples and calculate x_value, domain_data, value_data for every num_epochs_draw epochs
            if epoch % config.num_epochs_draw == 0:
                x_data_all_epochs = domain_sampler.generate_samples(num_samples=self.x_total_sample_v,
                                                                    num_k=config.k_dist)
                with torch.no_grad():
                    _, value_data_all_epochs = equm_updater.obj_sim_value(x_data_all_epochs, x_n_sim, dist_a_mid,
                                                                          dist_a_mesh)

            # Use the computed values for each epoch within the range of num_epochs_draw
            start_idx = epoch * config.num_samples_value % len(value_data_all_epochs)
            end_idx = min(start_idx + config.num_samples_value, len(value_data_all_epochs))
            value_data = value_data_all_epochs[start_idx:end_idx]

            # Load the data
            dataset = TensorDataset(value_data[:, 0:config.n_input], value_data[:, config.n_input])
            data_loader = DataLoader(dataset, batch_size=self.batch_size_v, shuffle=True, pin_memory=False,
                                     num_workers=self.n_worker)

            epoch_loss = 0.0
            self.model.train()  # Ensure the model is in training mode
            for batch_x, batch_y in data_loader:
                data_x = batch_x.to(self.device)
                data_y = batch_y.unsqueeze(1).to(self.device)
                if isinstance(self.model, torch.nn.DataParallel):
                    data_fit = self.model.module.f_value(data_x)
                else:
                    data_fit = self.model.f_value(data_x)

                loss_value = loss_function(data_y, data_fit)

                optimizer_value.zero_grad()
                loss_value.backward()
                optimizer_value.step()

                epoch_loss += loss_value.item()

            # Update losses
            losses.append(np.log(epoch_loss / len(data_loader.dataset)))

            # Update learning rate
            scheduler.step(epoch_loss)

        # Plot the losses after all epochs are completed
        self.update_graph(losses, title='Losses for Value Fitting Bellman')
        #plt.show()
        plt.savefig(f'figures/loss_value_fitting.png')
        plt.close()
        value_func.state_dict()
        if self.i_save:
            torch.save(self.remove_module_from_checkpoint(self.model.state_dict()),
                       f'models/trained_value_nn_{config.model_number_output}.pth')

        return self.model


    def update_graph(self, losses, title='x_title'):
        plt.clf()
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(title)
        #plt.pause(0.01)

    def check_and_reset_nan_params(self, network):
        # Check for NaN values in the parameters
        nan_detected = False
        for param in network.parameters():
            if torch.isnan(param).any():
                nan_detected = True
                break

        if nan_detected:
            # Reset the neural network
            # Option 1: Reinitialize the weights and biases
            for layer in network:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            # Option 2: Load the last known good state of the network (uncomment the line below if needed)
            # network.load_state_dict(network.state_dict())

            # Optionally, print a message to indicate that NaNs were detected and the network was reset
            print("NaN detected in parameters. Resetting the neural network.")
        return network

    @staticmethod
    def remove_module_from_checkpoint(state_dict):
        """Remove the 'module.' prefix from the keys in the state dictionary."""
        return {k.replace('module.', ''): v for k, v in state_dict.items()}



