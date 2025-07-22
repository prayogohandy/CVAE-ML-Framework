import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
import pickle

# Save results to a pickle file
def save_results(results_dict, filename="results.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(results_dict, f)

# Load results from a pickle file
def load_results(filename="results.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

def combine_results(file1, file2, filename="results.pkl"):
    # Load the dictionaries from the pickle files
    with open(file1, 'rb') as f1:
        dict1 = pickle.load(f1)
    
    with open(file2, 'rb') as f2:
        dict2 = pickle.load(f2)
    
    # Combine the dictionaries (dict2 will overwrite dict1 in case of key conflicts)
    combined_dict = {**dict1, **dict2}

    # Save the combined dictionary to a new pickle file
    with open(filename, 'wb') as out_file:
        pickle.dump(combined_dict, out_file)
    
    return combined_dict

def remove_method_from_result(key, save_path="temp_multi_pipeline.pkl"):
    if os.path.exists(save_path):
        results_dict = load_results(filename=save_path)  # Load saved results
    else:
        print("Save file not found.")
        return
    
    # Remove all keys that start with key
    results_dict = {
        k: v for k, v in results_dict.items()
        if not (k.startswith(key) )
    }

    # Save the updated results back to the file
    save_results(results_dict, filename=save_path)

def set_random_seed(seed=0):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)  # Set seed for CPU
    torch.cuda.manual_seed(seed)  # Set seed for GPU (if using CUDA)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
    np.random.seed(seed)  # Set seed for NumPy
    random.seed(seed)  # Set seed for Python random
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark for reproducibility

def print_ranges(df):
    ranges = pd.DataFrame({
        'Min': df.min(),
        'Max': df.max()
    })
    print(ranges)


def process_dataframe(df, label=None, scaler='standard', categorical_encoder='onehot', existing_info=None):
    """
    Process a dataframe for ML models with numerical scaler and categorical encoder.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input features.
    label : pd.Series or pd.DataFrame, optional
        Target labels.
    scaler : str
        One of ['none', 'standard', 'minmax'].
    categorical_encoder : str
        One of ['onehot', 'ordinal'].
    existing_info : dict, optional
        Previously fitted scalers and encoders.

    """
    fit = existing_info is None
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- Scale numerical ---
    scaler_dict = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'none': None
    }
    if scaler not in scaler_dict:
        raise ValueError(f"Unknown scaler: {scaler}")
    selected_scaler = scaler_dict[scaler] if fit else existing_info['selected_scaler']

    df_numerical = df[numerical_cols].copy()
    if selected_scaler is not None:
        if fit:
            df_numerical = pd.DataFrame(
                selected_scaler.fit_transform(df_numerical),
                columns=numerical_cols, index=df.index
            )
        else:
            df_numerical = pd.DataFrame(
                selected_scaler.transform(df_numerical),
                columns=numerical_cols, index=df.index
            )

    # --- Encode categorical ---
    if categorical_encoder not in ['onehot', 'ordinal']:
        raise ValueError("categorical_encoder must be 'onehot' or 'ordinal'.")

    encoded_df = None
    encoders = [] if fit else existing_info['encoders']
    len_ohes = []

    if categorical_cols:
        if categorical_encoder == 'onehot':
            ohe_dfs = []
            for i, col in enumerate(categorical_cols):
                if fit:
                    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = enc.fit_transform(df[[col]])
                    encoders.append(enc)
                else:
                    enc = encoders[i]
                    encoded = enc.transform(df[[col]])

                ohe_cols = enc.get_feature_names_out([col])
                ohe_df = pd.DataFrame(encoded, columns=ohe_cols, index=df.index)
                ohe_dfs.append(ohe_df)
                len_ohes.append(ohe_df.shape[1])

            encoded_df = pd.concat(ohe_dfs, axis=1)

        elif categorical_encoder == 'ordinal':
            if fit:
                enc = OrdinalEncoder()
                encoded = enc.fit_transform(df[categorical_cols])
                encoders = enc
            else:
                enc = encoders
                encoded = enc.transform(df[categorical_cols])

            encoded_df = pd.DataFrame(encoded, columns=categorical_cols, index=df.index)

    # --- Combine numerical and categorical ---
    if encoded_df is not None:
        final_df = pd.concat([df_numerical, encoded_df], axis=1)
    else:
        final_df = df_numerical

    # --- Label processing ---
    label_array = None
    label_scaler = None if fit else existing_info.get('label_scaler')
    if label is not None:
        if isinstance(label, pd.Series):
            label = label.to_frame()

        if label.shape[1] != 1:
            raise ValueError("label must have exactly one column.")

        if fit:
            label_scaler = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            label_ohe = label_scaler.fit_transform(label)
        else:
            label_ohe = label_scaler.transform(label)

        label_array = label_ohe.astype(np.float32)

    # --- Final output ---
    feature_array = final_df.to_numpy(dtype=np.float32)

    additional_info = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'selected_scaler': selected_scaler,
        'encoders': encoders,
        'encoder_type': categorical_encoder,
        'label_scaler': label_scaler,
        'label_dim': label_array.shape[1] if label_array is not None else 0,
        'len_numerical': len(numerical_cols),
        'len_ohes': len_ohes if categorical_encoder == 'onehot' else None
    }

    return feature_array, label_array, additional_info

def reverse_process_dataframe(processed_data, additional_info):
    """
    Reverse processed dataframe or tensor array to its original form.

    Parameters:
    -----------
    processed_data : pd.DataFrame, np.ndarray, or torch.Tensor
        Processed data (numerical + encoded categorical).
    additional_info : dict
        Info dict containing scalers and encoders used.

    Returns:
    --------
    pd.DataFrame
        Reconstructed dataframe with original numerical and categorical columns.
    """
    scaler = additional_info['selected_scaler']
    encoders = additional_info['encoders']
    encoder_type = additional_info['encoder_type']
    numerical_cols = additional_info['numerical_cols']
    categorical_cols = additional_info['categorical_cols']
    len_numerical = additional_info['len_numerical']
    len_ohes = additional_info.get('len_ohes')

    # Convert tensor to numpy if needed
    if isinstance(processed_data, torch.Tensor):
        processed_data = processed_data.detach().cpu().numpy()

    # Step 1: Reverse scaling for numerical columns
    numerical_array = processed_data[:, :len_numerical]
    if scaler is not None:
        numerical_array = scaler.inverse_transform(numerical_array)

    reversed_df = pd.DataFrame(numerical_array, columns=numerical_cols)

    # Step 2: Reverse categorical encoding
    if encoder_type == 'onehot':
        start_idx = len_numerical
        for i, (cat_col, ohe_len, ohe_encoder) in enumerate(zip(categorical_cols, len_ohes, encoders)):
            ohe_data = processed_data[:, start_idx:start_idx + ohe_len]

            # Enforce proper one-hot (argmax-based)
            argmax_indices = np.argmax(ohe_data, axis=1)
            one_hot_clean = np.zeros_like(ohe_data)
            one_hot_clean[np.arange(ohe_data.shape[0]), argmax_indices] = 1

            # Decode to category labels
            cat_values = ohe_encoder.inverse_transform(one_hot_clean)
            reversed_df[cat_col] = cat_values.flatten()

            start_idx += ohe_len

    elif encoder_type == 'ordinal':
        ordinal_data = processed_data[:, len_numerical:]
        ordinal_data_rounded = np.rint(ordinal_data).astype(int)
        cat_values = encoders.inverse_transform(ordinal_data_rounded)
        for i, col in enumerate(categorical_cols):
            reversed_df[col] = cat_values[:, i]

    return reversed_df


def plot_embed(embedding, color_labels=None, x_label='Latent Variable 1', y_label='Latent Variable 2', 
               z_label='Latent Variable 3', point_size=4, axis_limits=None, filename=None):
    """
    Visualize high-dimensional embeddings in either 2D or 3D space using Plotly.

    If the embedding has 3 dimensions, a 3D scatter plot will be created. If it has 2 dimensions,
    a 2D scatter plot will be created. Optionally, labels for coloring the points can be provided.

    Parameters:
    -----------
    embedding : np.ndarray
        A 2D array where each row represents a sample's embedding. The number of columns must be 2 or 3.

    color_labels : array-like, optional
        A 1D array of labels corresponding to the samples in `embedding`. The length should match
        the number of rows in `embedding`. If provided, these labels will be used for coloring the points.

    x_label : str, optional
        Label for the X-axis. Default is 'Latent Variable 1'.

    y_label : str, optional
        Label for the Y-axis. Default is 'Latent Variable 2'.

    z_label : str, optional
        Label for the Z-axis. Default is 'Latent Variable 3'. Only used for 3D plots.

    point_size : int, optional
        Size of the scatter points. Default is 2.

    axis_limits : dict, optional
        Dictionary with axis limits. Keys should be 'x', 'y', and 'z' for the respective axis limits as tuples (min, max).

    Returns:
    --------
    None
        The function directly displays the plot using Plotly.
    """
    if embedding.shape[1] not in [2, 3]:
        raise ValueError("Embedding must have exactly 2 or 3 dimensions.")

    if color_labels is not None and len(color_labels) != embedding.shape[0]:
        raise ValueError("Length of color_labels must match the number of rows in embedding.")

    if embedding.shape[1] == 3:
        # 3D Scatter Plot
        fig = px.scatter_3d(
            embedding, 
            x=0, y=1, z=2, 
            color=color_labels,
            size_max=point_size
        )
        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(
            autosize=False,
            width=600,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins as needed
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label,
                camera=dict(eye=dict(x=1.75, y=1.5, z=1.5))
            )
        )
        # Set axis limits if provided
        if axis_limits is not None:
            if 'x' in axis_limits:
                fig.update_layout(scene=dict(xaxis=dict(range=axis_limits['x'])))
            if 'y' in axis_limits:
                fig.update_layout(scene=dict(yaxis=dict(range=axis_limits['y'])))
            if 'z' in axis_limits:
                fig.update_layout(scene=dict(zaxis=dict(range=axis_limits['z'])))
    else:
        # 2D Scatter Plot
        fig = px.scatter(
            embedding, 
            x=0, y=1, 
            color=color_labels,
            size_max=point_size
        )
        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=600,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins as needed
        )
        # Set x-axis and y-axis limits if provided
        if axis_limits is not None:
            if 'x' in axis_limits:
                fig.update_layout(xaxis=dict(range=axis_limits['x']))
            if 'y' in axis_limits:
                fig.update_layout(yaxis=dict(range=axis_limits['y']))
    fig.show()

    if filename is not None:
        pio.write_image(fig, filename, scale = 3)
    
def distribution_comparison(real_data, synthetic_data, n_bins=10, columns=None, stacked=False, 
                            rename_dict=None, filename=None):
    """
    Compare distributions of real and synthetic data with customizable column names and an option to save the plot.

    Parameters:
    -----------
    real_data : pd.DataFrame
        DataFrame with real data.

    synthetic_data : pd.DataFrame
        DataFrame with synthetic data.

    n_bins : int, optional
        Number of bins for histograms. Default is 10.

    columns : list of str, optional
        Specific columns to plot. If None, all columns are plotted.

    stacked : bool, optional
        If True, plots each column in a single subplot. If False, uses side-by-side subplots for real and synthetic.

    rename_dict : dict, optional
        Dictionary to rename columns for display.

    filename : str, optional
        Path to save the plot. If None, the plot will only be displayed.

    Returns:
    --------
    None
        The function displays and optionally saves the plot.
    """
    if columns is None:
        columns = real_data.columns  # Use all columns if not specified

    num_columns = len(columns)
    if stacked:
        fig, axes = plt.subplots(num_columns, 1, figsize=(5, 5 * num_columns))
    else:
        fig, axes = plt.subplots(num_columns, 2, figsize=(10, 5 * num_columns))

    for i, col in enumerate(columns):
        real_col = real_data[col]
        synthetic_col = synthetic_data[col]

        # Determine x-axis label from rename_dict or default to column name
        x_label = rename_dict[col] if rename_dict and col in rename_dict else col

        # Check if the column is categorical
        if real_col.dtype == 'object' or synthetic_col.dtype == 'object':

            # Map real_col and synthetic_col if categorical and rename_dict is provided
            if rename_dict:
                real_col = real_col.map(rename_dict)
                synthetic_col = synthetic_col.map(rename_dict)
            
            ax = axes[i] if stacked else axes[i, 0]
            sns.histplot(real_col, ax=ax, color='blue', label='Real', discrete=True)
            ax.set_xlabel(x_label)
            
            ax = axes[i] if stacked else axes[i, 1]
            sns.histplot(synthetic_col, ax=ax, color='orange', label='Synthetic', discrete=True)
            ax.set_xlabel(x_label)
            ax.legend()
        else:
            # Determine the common bin range
            min_val = min(real_col.min(), synthetic_col.min())
            max_val = max(real_col.max(), synthetic_col.max())
            bin_range = np.linspace(min_val, max_val, num=n_bins)

            if stacked:
                ax = axes[i]
                sns.histplot(real_col, kde=True, ax=ax, color='blue', label='Real', alpha=0.5, bins=bin_range)
                sns.histplot(synthetic_col, kde=True, ax=ax, color='orange', label='Synthetic', alpha=0.5, bins=bin_range)
                ax.set_xlabel(x_label)
                ax.legend()
            else:
                # Plot real data distribution
                sns.histplot(real_col, kde=True, ax=axes[i, 0], color='blue', label='Real', bins=bin_range)
                axes[i, 0].set_title(f"Real Data Distribution: {x_label}")
                axes[i, 0].set_xlabel(x_label)

                # Plot synthetic data distribution
                sns.histplot(synthetic_col, kde=True, ax=axes[i, 1], color='orange', label='Synthetic', bins=bin_range)
                axes[i, 1].set_title(f"Synthetic Data Distribution: {x_label}")
                axes[i, 1].set_xlabel(x_label)

                # Set y-axis limits to be the same for side-by-side comparison
                max_y = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
                axes[i, 0].set_ylim(0, max_y)
                axes[i, 1].set_ylim(0, max_y)

    plt.tight_layout()

    # Save the plot if filename is provided
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', dpi=600)
    
    plt.show()
    
def corrmap_comparison(real_data, synthetic_data, rename_dict=None, filename_prefix=None):
    """
    Compare correlation matrices of real and synthetic data with optional renaming and saving options.

    Parameters:
    -----------
    real_data : pd.DataFrame
        DataFrame with real data.

    synthetic_data : pd.DataFrame
        DataFrame with synthetic data.

    rename_dict : dict, optional
        Dictionary to rename columns for display.

    filename_prefix : str, optional
        Prefix to use when saving the correlation matrix plots. If None, plots are only displayed.

    Returns:
    --------
    None
        Displays and optionally saves the correlation matrix plots.
    """
    # Function to perform One-Hot Encoding on categorical columns
    def one_hot_encode(df):
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        return pd.get_dummies(df, columns=categorical_cols)

    # Apply One-Hot Encoding
    real_data_encoded = one_hot_encode(real_data)
    synthetic_data_encoded = one_hot_encode(synthetic_data)

    # Calculate correlation matrices
    real_corr = real_data_encoded.corr()
    synthetic_corr = synthetic_data_encoded.corr()

    # Define a common order for the heatmap (using real_corr as reference)
    common_order = real_corr.columns

    # Reindex both correlation matrices to match this common order
    real_corr = real_corr.reindex(index=common_order, columns=common_order)
    synthetic_corr = synthetic_corr.reindex(index=common_order, columns=common_order)
    
    # Rename columns and index if rename_dict is provided
    if rename_dict:
        real_corr = real_corr.rename(index=rename_dict, columns=rename_dict)
        synthetic_corr = synthetic_corr.rename(index=rename_dict, columns=rename_dict)

    # Set up the figure and axes for correlation maps
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # Plot the correlation matrix for real data
    sns.heatmap(real_corr, ax=axes[0], cmap='coolwarm', annot=True, fmt=".2f", cbar=True)
    axes[0].set_title('Real Data Correlation Matrix')

    # Plot the correlation matrix for synthetic data
    sns.heatmap(synthetic_corr, ax=axes[1], cmap='coolwarm', annot=True, fmt=".2f", cbar=True)
    axes[1].set_title('Synthetic Data Correlation Matrix')

    # Save the correlation matrices if filename_prefix is provided
    if filename_prefix:
        fig.savefig(f'{filename_prefix}_correlation_matrices.png', bbox_inches='tight', dpi=600)

    # Plot the difference in correlation matrices
    fig_diff, ax_diff = plt.subplots(figsize=(10, 10))
    diff_corr = abs(real_corr - synthetic_corr)
    sns.heatmap(diff_corr, ax=ax_diff, cmap='coolwarm', annot=True, fmt=".2f", cbar=True)
    ax_diff.set_title('Difference in Correlation (Real - Synthetic)')
    ax_diff.set_xlabel('Features')
    ax_diff.set_ylabel('Features')

    # Save the difference matrix if filename_prefix is provided
    if filename_prefix:
        fig_diff.savefig(f'{filename_prefix}_correlation_difference.png', bbox_inches='tight', dpi = 600)

    plt.tight_layout()
    plt.show()