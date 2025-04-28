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
    def __init__(self, num_numerical, num_categorical, label_dim, layers_dim, batch_norm, activation, dropout_rate):
        super().__init__()

        self.latent_dim = layers_dim[-1]
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical

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

    def forward(self, z, labels):
        z = torch.cat((z, labels), dim=-1)
        decoded = self.decoder(z)

        # Numerical reconstruction
        x_num = self.decoder_num(decoded)

        # Categorical reconstruction with softmax
        x_cat = [F.softmax(decoder(decoded), dim=-1) for decoder in self.decoder_cat]

        return torch.cat([x_num] + x_cat, dim=1)


class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, num_numerical, num_categorical, label_dim, layers_dim=None,
                 batch_norm=True, activation="relu", dropout_rate=0.0):
        super().__init__()

        if layers_dim is None:
            layers_dim = [64, 32, 16, 3]
        
        self.is_trained = False
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.input_units = num_numerical + sum(num_categorical)
        self.latent_dim = layers_dim[-1]
        self.label_dim = label_dim

        # Activation function selection
        activations = {"relu": nn.ReLU(), "elu": nn.ELU()}
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]

        self.encoder = ConditionalEncoder(self.input_units, label_dim, layers_dim,
                                          batch_norm, self.activation, dropout_rate)

        self.decoder = ConditionalDecoder(num_numerical, num_categorical, label_dim, layers_dim,
                                          batch_norm, self.activation, dropout_rate)

    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        y = self.decoder(z, labels)
        return mu, logvar, z, y
    
    def sample(self, z: Tensor, labels) -> Tensor:
        return self.decoder(z, labels)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def set_random_seed(self, seed):
        """Set the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {seed}")

    def train_model(self, x_train: Tensor, labels: Tensor, lr: float = 5e-3, 
                batch_size: int = 32, n_iter: int = 1000,
                beta_kl: float = 1.0, beta_rec: float = 1.0, 
                recon_loss_type: str = "mse", seed: int = -1, verbose = False) -> None:
        # Set random seed for reproducibility
        if seed != -1:
            self.set_random_seed(seed)

        x_train = x_train.to(dtype=torch.float32)
        labels = labels.to(dtype=torch.float32)                 

        train_dataset = TensorDataset(x_train, labels)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr)  # Single learning rate for both encoder and decoder
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=n_iter // 5, gamma=0.1)

        start_time = time.time()

        self.train()  # Set the model to training mode
        for it in tqdm(range(n_iter)):
            epoch_loss = 0  # To track total loss per epoch

            for batch_idx, (x, labels) in enumerate(data_loader):
                x, labels = x.to(device), labels.to(device)

                # Forward pass
                real_mu, real_logvar, z, rec = self(x, labels) 

                # Compute losses
                loss_rec = calc_reconstruction_loss(x, rec, loss_type=recon_loss_type, reduction="mean")
                loss_kl = calc_kl(real_logvar, real_mu, reduce="mean")
                loss = beta_rec * loss_rec + beta_kl * loss_kl

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

    @staticmethod
    def calc_reconstruction_loss(x, rec, loss_type="mse"):
        """Calculate reconstruction loss."""
        if loss_type == "mse":
            return F.mse_loss(rec, x, reduction="mean")
        elif loss_type == "bce":
            return F.binary_cross_entropy(rec, x, reduction="mean")
        else:
            raise ValueError(f"Unsupported reconstruction loss type: {loss_type}")

    @staticmethod
    def calc_kl(logvar, mu):
        """Calculate KL divergence."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
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