"""
VAE for Tabular Dataset
"""

# imports
# torch and friends
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# standard
import os
import time
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib

"""
Helper Functions
"""

def enforce_one_hot(x_cvae, len_numerical, len_ohes):
    """
    Enforces one-hot encoding for categorical variables in the resampled data.
    
    Parameters:
    -----------
    x_cvae : np.ndarray
        Resampled data array with shape (num_samples, num_features).
    len_numerical : int
        Number of numerical features in the data.
    len_ohes : list of int
        List of integers where each element represents the length of each one-hot encoded categorical feature.
    
    Returns:
    --------
    np.ndarray
        Processed data with one-hot categorical features.
    """
    x_cvae_processed = x_cvae.copy()
    
    start_idx = len_numerical
    for cat_len in len_ohes:
        end_idx = start_idx + cat_len

        cat_block = x_cvae_processed[:, start_idx:end_idx]  # Slice the categorical section
        argmax_indices = np.argmax(cat_block, axis=1)       # Get argmax index
        one_hot_fixed = np.zeros_like(cat_block)
        one_hot_fixed[np.arange(cat_block.shape[0]), argmax_indices] = 1  # Set 1 at argmax index

        x_cvae_processed[:, start_idx:end_idx] = one_hot_fixed  # Replace with fixed one-hot
        start_idx = end_idx  # Move to next categorical block (if more than one)

    return x_cvae_processed

def calc_reconstruction_loss(x, rec, loss_type="mse"):
        """Calculate reconstruction loss."""
        if loss_type == "mse":
            return F.mse_loss(rec, x, reduction="none").sum(dim=1).mean()
        elif loss_type == "bce":
            return F.binary_cross_entropy(rec, x, reduction="none").sum(dim=1).mean()
        else:
            raise ValueError(f"Unsupported reconstruction loss type: {loss_type}")

def calc_kl(logvar, mu):
    """Calculate KL divergence."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()   

def calc_constraint_loss(rec_std, rec_raw, rec_cat, std_bounds):
    """
    Calculate the constraint loss based on the raw reconstruction.

    rec_std : Tensor
        The standardized reconstruction. (N, FN)
    rec_raw : Tensor
        The raw reconstruction output from the decoder. (N, FN)
    rec_cat : Tensor
        The categorical reconstruction output from the decoder. (N, FC)
    std_bounds : tuple
        A tuple containing the lower and upper bounds for the standard deviation. (lb: (C, FN), ub: (C, FN)) 
    """

    upper_violation = torch.clamp(rec_std - std_bounds[1], min=0)  # positive if above UB
    lower_violation = torch.clamp(std_bounds[0] - rec_std, min=0)  # positive if below LB
    loss = (upper_violation.pow(2) + lower_violation.pow(2)).sum(dim=1).mean()

    # Ab/Ag
    a = rec_raw[:, -1]  # Assuming the last column is the Ab/Ag feature

    pred_class = rec_cat.argmax(dim=1)
    rectangular_mask = (pred_class == 2).float()  # 1 for rectangular
    section_loss = (a * rectangular_mask).pow(2).mean()

    return loss + section_loss


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

"""
Models
"""
 
class ConditionalEncoder(nn.Module):
    def __init__(self, input_dim, label_dim, layers_dim, batch_norm, activation, dropout_rate):
        super().__init__()
        self.latent_dim = layers_dim[-1]
        self.input_units = input_dim + label_dim

        encoder_layers = []
        in_features = self.input_units
        for dim in layers_dim[:-1]:
            encoder_layers.append(nn.Linear(in_features, dim))
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(activation)
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            in_features = dim

        # Bottleneck layer
        encoder_layers.append(nn.Linear(in_features, 2 * self.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x, labels):
        x = torch.cat((x, labels), dim=-1)
        y = self.encoder(x)
        mu, logvar = y.chunk(2, dim=1)  # Split into mean and log-variance
        return mu, logvar


class ConditionalDecoder(nn.Module):
    def __init__(self, num_numerical, num_categorical, label_dim, layers_dim, 
                 batch_norm, activation, dropout_rate, feature_layer, scaler_info):
        super().__init__()

        self.latent_dim = layers_dim[-1]
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.scaler_mean, self.scaler_scale = scaler_info

        # Map each numerical feature index to its activation function
        self.feature_activations = []
        for i in range(num_numerical):
            if i in feature_layer.get('softplus', []):
                self.feature_activations.append(nn.Softplus())
            elif i in feature_layer.get('relu', []):
                self.feature_activations.append(nn.Softplus())#nn.ReLU())
            else:
                self.feature_activations.append(nn.Identity())  # unconstrained

        decoder_layers = []
        in_features = self.latent_dim + label_dim
        for dim in reversed(layers_dim[:-1]):
            decoder_layers.append(nn.Linear(in_features, dim))
            if batch_norm:
                decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(activation)
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            in_features = dim

        self.decoder = nn.Sequential(*decoder_layers)

        # Output layers
        self.decoder_num = nn.Linear(in_features, num_numerical)
        self.decoder_cat = nn.ModuleList([nn.Linear(in_features, len_ohe) for len_ohe in num_categorical])

    def forward(self, z, labels, return_raw=False):
        # Concatenate latent + labels
        z = torch.cat((z, labels), dim=-1)
        decoded = self.decoder(z)

        # Numerical reconstruction
        x_num_raw_out = self.decoder_num(decoded)  # unconstrained outputs

        # Apply feature-wise constraints (ReLU, Softplus, etc.)
        if self.feature_activations:
            x_num_constrained = torch.empty_like(x_num_raw_out)
            for i, act in enumerate(self.feature_activations):
                x_num_constrained[:, i] = act(x_num_raw_out[:, i])
        else:
            x_num_constrained = x_num_raw_out

        # Apply standardization for training loss
        x_num_scaled = (x_num_constrained - self.scaler_mean) / self.scaler_scale

        # Categorical outputs
        x_cat = [F.softmax(decoder(decoded), dim=-1) for decoder in self.decoder_cat]

        if return_raw:
            # Return both scaled & raw for different losses
            return torch.cat([x_num_scaled] + x_cat, dim=1), x_num_constrained
        else:
            return torch.cat([x_num_scaled] + x_cat, dim=1)


class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, num_numerical, num_categorical, label_dim, layers_dim=None,
                 batch_norm=True, activation="relu", dropout_rate=0.0, feature_layer={}, scaler_info=(0, 1)):
        super().__init__()

        if layers_dim is None:
            layers_dim = [64, 32, 16, 3]
        
        self.is_trained = False
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.input_units = num_numerical + sum(num_categorical)
        self.latent_dim = layers_dim[-1]
        self.label_dim = label_dim
        self.scaler_mean, self.scaler_scale = scaler_info

        # Activation function selection
        activations = {"relu": nn.ReLU(), "elu": nn.ELU()}
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]

        self.encoder = ConditionalEncoder(self.input_units, label_dim, layers_dim,
                                          batch_norm, self.activation, dropout_rate)

        self.decoder = ConditionalDecoder(num_numerical, num_categorical, label_dim, layers_dim,
                                          batch_norm, self.activation, dropout_rate, feature_layer, scaler_info)

    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = reparameterize(mu, logvar)
        y, y_raw = self.decoder(z, labels, return_raw=True)
        return mu, logvar, z, y, y_raw

    def sample(self, z: Tensor, labels) -> Tensor:
        return self.decoder(z, labels)

    def set_random_seed(self, seed):
        """Set the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {seed}")

    def train_model(self, x_train: Tensor, labels: Tensor, std_bounds: tuple,
                    lr: float = 5e-3, batch_size: int = 32, n_iter: int = 1000, 
                    anneal_start_kl: float = 0.1, anneal_start_phys: float = 0.3, anneal_warmup: float = 0.4, 
                    beta_kl: float = 1.0, beta_rec: float = 1.0, beta_phys: float = 1.0,
                    recon_loss_type: str = "mse", seed: int = -1, verbose = False) -> None:
        # Set random seed for reproducibility
        if seed != -1:
            self.set_random_seed(seed)

        x_train = x_train.to(dtype=torch.float32)
        labels = labels.to(dtype=torch.float32)       
        lb_bounds, ub_bounds = std_bounds         
        lb_bounds = lb_bounds.to(dtype=torch.float32)
        ub_bounds = ub_bounds.to(dtype=torch.float32) 

        train_dataset = TensorDataset(x_train, labels, lb_bounds, ub_bounds)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr)  # Single learning rate for both encoder and decoder
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=n_iter // 5, gamma=0.1)

        self.train()  # Set the model to training mode
        for it in tqdm(range(n_iter)):
            epoch_loss = 0  # To track total loss per epoch
            progress = it / n_iter
            # Adjust beta_kl and beta_phys based on annealing schedule
            if progress > anneal_start_kl:
                ratio_kl = (progress - anneal_start_kl) / anneal_warmup
                beta_kl_iter = min(ratio_kl, 1.0) * beta_kl
            else:
                beta_kl_iter = 0

            if progress > anneal_start_phys:
                ratio_phys = (progress - anneal_start_phys) / anneal_warmup
                beta_phys_iter = min(ratio_phys, 1.0) * beta_phys
            else:
                beta_phys_iter = 0

            for batch_idx, (x, labels, lb_batch, ub_batch) in enumerate(data_loader):
                x, labels = x.to(device), labels.to(device)
                
                # Forward pass
                real_mu, real_logvar, z, rec, rec_raw = self(x, labels)

                # Compute losses
                loss_rec = calc_reconstruction_loss(x, rec, loss_type=recon_loss_type)
                loss_kl = calc_kl(real_logvar, real_mu)
                rec_num = rec[:, :self.num_numerical]  # Numerical part of the reconstruction
                rec_cat = rec[:, self.num_numerical:]  # Categorical part of the reconstruction
                loss_phys = calc_constraint_loss(rec_num, rec_raw, rec_cat, (lb_batch, ub_batch))
                loss = beta_rec * loss_rec + beta_kl_iter * loss_kl + beta_phys_iter * loss_phys

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()  # Accumulate loss for the epoch

            # Scheduler step after every epoch (optional)
            scheduler.step()

            # Print loss every few iterations or epochs
            if (it + 1) % 100 == 0 and verbose:
                print(f"Iteration {it+1}/{n_iter}, Loss: {epoch_loss/len(data_loader)}")
        
        # toggle indicator 
        self.is_trained = True

    
    def resample(self, X, y, label_scaler, additional_sample=0):
        """
        Balances the dataset using class count
        
        Parameters:
        -----------
        X : torch.Tensor
            A 2D array where each row represents a sample and each column represents a feature.
        y : torch.Tensor
            A 2D array where each row represents a sample of OHE label.
        label_scaler : OneHotEncoder
            The label encoder used to convert the label to class.    

        Returns:
        --------
        np.ndarray
            Resampled 2D array containing features.
        np.ndarray
            Resampled 1D array containing class label.
        """
        if not self.is_trained:
            raise ValueError("The model has not been trained")
        
        generated_data = []
        generated_label = []
        
        # compute class counts and maximum counts
        class_counts = y.sum(axis=0).int()  # Ensure class counts are integers
        max_counts = class_counts.max().item()  # Get max count as an integer

        for idx, count in enumerate(class_counts):
            if count < max_counts:
                num_samples = max_counts - count + additional_sample # Ensure integer
                z = torch.randn(size=(num_samples, self.latent_dim))  # Latent space samples

                # One-hot encode the label
                class_label = torch.zeros((num_samples, len(class_counts)), dtype=torch.float32)
                class_label[:, idx] = 1

                # Generate samples
                generated_samples = self.sample(z, class_label)
                generated_data.append(generated_samples)
                generated_label.append(class_label)

        generated_data = torch.cat(generated_data, dim=0)
        generated_label = torch.cat(generated_label, dim=0)
        
        X_cvae = torch.cat((X, generated_data), dim=0).detach().numpy()
        y_cvae_label = torch.cat((y, generated_label), dim=0)
        y_cvae = label_scaler.inverse_transform(y_cvae_label).ravel()
        
        return X_cvae, y_cvae