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

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
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


def process_dataframe(df, label=None, scaler='standard', existing_info=None):
    """
    Process a dataframe for ML models with optional reuse of previously fitted scalers/encoders.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature dataframe.
    label : pd.DataFrame, pd.Series, or None
        Label to be one-hot encoded.
    scaler : str
        One of ['none', 'standard', 'minmax'].
    existing_info : dict or None
        Contains the pre-fitted scaler, OHE encoders, and label encoder.

    Returns
    -------
    torch.Tensor : feature tensor
    torch.Tensor or None : label tensor
    int : number of numerical columns
    list : list of one-hot encoded dimensions for categorical columns
    dict : processing info (scalers, encoders, column names)
    """
    fit = existing_info is None
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Handle scaler
    scaler_dict = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'none': None
    }
    if scaler not in scaler_dict:
        raise ValueError(f"Scaler '{scaler}' not recognized.")

    if fit:
        selected_scaler = scaler_dict[scaler]
    else:
        selected_scaler = existing_info['selected_scaler']

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

    # Categorical encoding
    ohe_dfs = []
    ohe_scalers = [] if fit else existing_info['ohe_scalers']
    len_ohes = []

    for i, col in enumerate(categorical_cols):
        if fit:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe_transformed = ohe.fit_transform(df[[col]])
            ohe_scalers.append(ohe)
        else:
            ohe = ohe_scalers[i]
            ohe_transformed = ohe.transform(df[[col]])

        ohe_columns = ohe.get_feature_names_out([col])
        ohe_df = pd.DataFrame(ohe_transformed, columns=ohe_columns, index=df.index)

        ohe_dfs.append(ohe_df)
        len_ohes.append(ohe_df.shape[1])

    final_df = pd.concat([df_numerical] + ohe_dfs, axis=1) if ohe_dfs else df_numerical

    # Label processing
    label_ohe_df = None
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

        label_columns = label_scaler.get_feature_names_out(label.columns)
        label_ohe_df = pd.DataFrame(label_ohe, columns=label_columns, index=label.index)

    # Convert to tensors
    feature_tensor = torch.tensor(final_df.to_numpy(dtype=np.float32), dtype=torch.float32)
    label_tensor = (
        torch.tensor(label_ohe_df.to_numpy(dtype=np.float32), dtype=torch.float32)
        if label_ohe_df is not None else None
    )

    additional_info = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'selected_scaler': selected_scaler,
        'ohe_scalers': ohe_scalers,
        'label_scaler': label_scaler,
        'label_dim': label_ohe_df.shape[1] if label_ohe_df is not None else 0
    }

    return feature_tensor, label_tensor, len(numerical_cols), len_ohes, additional_info


def reverse_process_dataframe(processed_data, len_numerical, len_ohes, additional_info):
    """
    Reverse the processed dataframe or tensor array to its original form by reversing scaling on numerical columns
    and converting one-hot encoded columns back to categorical values using the fitted OHE scalers.

    Parameters:
    -----------
    processed_data : pd.DataFrame or torch.Tensor
        The processed data with scaled numerical columns and one-hot encoded categorical columns.
    len_numerical : int
        The number of numerical columns in the original dataframe.
    len_ohes : list
        List containing the number of one-hot encoded columns for each categorical feature.
    additional_info : dict
        A dictionary containing column names, the scaler that was used during processing, 
        and the list of fitted OneHotEncoder objects.

    Returns:
    --------
    pd.DataFrame
        The reversed dataframe with original numerical and categorical columns.
    """

    scaler = additional_info['selected_scaler']
    ohe_scalers = additional_info['ohe_scalers']
    numerical_cols = additional_info['numerical_cols']
    categorical_cols = additional_info['categorical_cols']

    # Convert to NumPy array if processed_data is a PyTorch tensor
    if isinstance(processed_data, torch.Tensor):
        processed_data = processed_data.detach().cpu().numpy()

    # Step 1: Reverse scaling for numerical columns
    if scaler is not None:
        numerical_data = scaler.inverse_transform(processed_data[:, :len_numerical])
    else:
        numerical_data = processed_data[:, :len_numerical]

    # Step 2: Reverse one-hot encoding for categorical columns
    start_idx = len_numerical
    categorical_data = {}

    for i, (ohe_len, ohe_scaler, cat_col) in enumerate(zip(len_ohes, ohe_scalers, categorical_cols)):
        # Extract the one-hot encoded columns for this categorical feature
        ohe_columns = processed_data[:, start_idx:start_idx + ohe_len]

        # Reverse the one-hot encoding using the fitted OHE scaler
        categorical_data[cat_col] = ohe_scaler.inverse_transform(ohe_columns)

        # Move to the next set of OHE columns
        start_idx += ohe_len

    # Step 3: Combine numerical and categorical columns
    reversed_df = pd.DataFrame(numerical_data, columns=numerical_cols)

    for cat_col, cat_data in categorical_data.items():
        reversed_df[cat_col] = cat_data.flatten()  # Flatten to get rid of extra dimension

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