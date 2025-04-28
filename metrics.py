import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, classification_report

FONT_ANNOT = 12
FONT_TITLE = 16
FONT_AXIS = 14
FONT_TICKS = 12

def extended_confusion_matrix(y_true, y_pred, y_mask=None, plot=True, class_names=None, prefix=None):
    # Constants for font sizes
    FONT_ANNOT = 14
    FONT_TITLE = 16
    FONT_AXIS = 14
    FONT_TICKS = 14

    if class_names is None:
        unique_classes = np.unique(y_true)
        class_names = [f'{i}' for i in unique_classes]
    all_classes = np.arange(len(class_names))
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if y_mask is not None:
        cm_mask = confusion_matrix(y_mask, y_pred)

    # Calculate precision and recall for each class
    precision = precision_score(y_true, y_pred, average=None, labels=all_classes)
    recall = recall_score(y_true, y_pred, average=None, labels=all_classes)
    accuracy = accuracy_score(y_true, y_pred)

    cm = cm.astype(np.float32)
    temp = np.zeros((len(all_classes), len(all_classes)))
    temp[:cm.shape[0], :cm.shape[1]] = cm
    cm = temp
    
    if plot:
        total = np.sum(cm)
        labels = [[f"{val:0.0f}\n{val / total:.2%}" for val in row] for row in cm]
        if y_mask is not None:
            labels[0] = [f"{cm_mask[1][i+1]:0.0f}+{cm_mask[0][i+1]:0.0f} \n{val / total:.2%}" for i, val in enumerate(cm[0])]

        ax = sns.heatmap(cm, annot=labels, cmap='Reds', fmt='',
                         xticklabels=class_names, yticklabels=class_names, cbar=False,
                         annot_kws={"fontsize": FONT_ANNOT})
        # ax.set_title('Confusion Matrix', fontweight='bold', fontsize=FONT_TITLE)
        ax.tick_params(labeltop=True, labelbottom=False, length=0, labelsize=FONT_TICKS)

        # matrix for the extra column and row
        f_mat = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1))
        f_mat[:-1, -1] = recall
        f_mat[-1, :-1] = precision
        f_mat[-1, -1] = accuracy

        f_mask = np.ones_like(f_mat)
        f_mask[:, -1] = 0
        f_mask[-1, :] = 0

        f_color = np.ones_like(f_mat)
        f_color[-1, -1] = 0

        f_annot = [[f"{val:0.2%}" for val in row] for row in f_mat]
        f_annot[-1][-1] = "Accuracy:\n" + f_annot[-1][-1]

        sns.heatmap(f_color, mask=f_mask, annot=f_annot, fmt='',
                    xticklabels=class_names + ["Recall"],
                    yticklabels=class_names + ["Precision"],
                    cmap=ListedColormap(['skyblue', 'lightgrey']), cbar=False, ax=ax,
                    annot_kws={"fontsize": FONT_ANNOT})

        ax.xaxis.set_label_position('top')
        ax.set_xlabel('Predicted Class', fontweight='bold', fontsize=FONT_AXIS)
        ax.set_ylabel('Actual Class', fontweight='bold', fontsize=FONT_AXIS)

        if prefix is not None:
            os.makedirs("figures", exist_ok=True)
            plt.savefig(f"figures/{prefix}_confusion_matrix.jpeg", dpi=600, bbox_inches='tight')
        plt.show()

    return cm, recall, precision, accuracy


def replot_extended_confusion_matrix(output, n_run=20, class_names=None, prefix=None):
    # Constants for font sizes
    FONT_ANNOT = 14
    FONT_TITLE = 16
    FONT_AXIS = 14
    FONT_TICKS = 14

    cm, recall, precision, accuracy = output

    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]

    total = np.sum(cm)
    labels = [[f"{val/n_run:0.1f}\n{val / total:.2%}" for val in row] for row in cm]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax = sns.heatmap(cm, annot=labels, cmap='Reds', fmt='',
                     xticklabels=class_names, yticklabels=class_names, cbar=False,
                     annot_kws={"fontsize": FONT_ANNOT})
    # ax.set_title('Confusion Matrix', fontweight='bold', fontsize=FONT_TITLE)
    ax.tick_params(labeltop=True, labelbottom=False, length=0, labelsize=FONT_TICKS)

    f_mat = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1))
    f_mat[:-1, -1] = recall
    f_mat[-1, :-1] = precision
    f_mat[-1, -1] = accuracy

    f_mask = np.ones_like(f_mat)
    f_mask[:, -1] = 0
    f_mask[-1, :] = 0

    f_color = np.ones_like(f_mat)
    f_color[-1, -1] = 0

    f_annot = [[f"{val:0.2%}" for val in row] for row in f_mat]
    f_annot[-1][-1] = "Accuracy:\n" + f_annot[-1][-1]

    sns.heatmap(f_color, mask=f_mask, annot=f_annot, fmt='',
                xticklabels=class_names + ["Recall"],
                yticklabels=class_names + ["Precision"],
                cmap=ListedColormap(['skyblue', 'lightgrey']), cbar=False, ax=ax,
                annot_kws={"fontsize": FONT_ANNOT})

    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Predicted Class', fontweight='bold', fontsize=FONT_AXIS)
    ax.set_ylabel('Actual Class', fontweight='bold', fontsize=FONT_AXIS)

    if prefix is not None:
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/{prefix}_confusion_matrix.jpeg", dpi=600, bbox_inches='tight')
    plt.show()

def classification_result(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    return accuracy, precision, recall, f1

def get_unique_filename(save_dir="plots", base_name="boxplot", extension="png"):
    """
    Generate a unique filename by appending an incrementing counter to the base name.
    
    Parameters:
    -----------
    save_dir : str
        The directory where the file will be saved.
    base_name : str
        The base name for the file.
    extension : str
        The file extension (e.g., "png", "pdf").
        
    Returns:
    --------
    str
        A unique filename.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    counter = 1
    filename = f"{base_name}_{counter}.{extension}"
    
    # Check if file already exists and increment counter until a unique name is found
    while os.path.exists(os.path.join(save_dir, filename)):
        counter += 1
        filename = f"{base_name}_{counter}.{extension}"
    
    return os.path.join(save_dir, filename)

def plot_metric_boxplot(results_dict, models, folds, metric="Accuracy", alt_names=None):
    """
    Plot a boxplot for a specific metric across different methods and fold types.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of results from `run_multiple_pipelines()`.
    models : list
        List of model names (e.g., ["XGB", "LGB", "CB", "RF", "SVM", "Stacking"]).
    folds : list
        List of fold types (e.g., ["regular", "smote", "cvae"]).
    metric : str
        The metric to plot (e.g., "Accuracy").
    alt_names : list
        List of names to rename the algorithm in models
    """
    data = []
    # Count occurrences of model names
    name_counter = Counter()
    alt_count = 0
    
    # Gather data for boxplot
    for model_name in models:
        is_paper_model = False
        if isinstance(model_name, str):
            name = model_name
        elif isinstance(model_name, list):
            name = "+".join(model_name)
        elif hasattr(model_name, "fit"):
            if alt_names is None:
                name = model_name.__class__.__name__  
            else:
                name = alt_names[alt_count]
                alt_count += 1
                is_paper_model = True

        # Track duplicates
        name_counter[name] += 1
        if name_counter[name] > 1:
            name = f"{name}_{name_counter[name]}"
        
        for fold_type in folds:
            if is_paper_model and fold_type != 'regular':
                continue
            
            key = f"{name}_{fold_type}"
            if key in results_dict:
                df_results = results_dict[key]["results_df"]
                metric_values = df_results[metric]
                for value in metric_values:
                    data.append({"Model": name, "Fold": fold_type, "Metric": value})
    
    # Convert data to DataFrame
    df_plot = pd.DataFrame(data)
    
    # Plot boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="Metric", hue="Fold", data=df_plot)
    plt.ylabel(metric)  # Rename the y-axis label
    plt.title(f"Boxplot of {metric} Across Models and Folds")

    plt.xticks(rotation=90)
    
    # Generate unique filename with timestamp
    filename = get_unique_filename()

    plt.savefig(filename, bbox_inches="tight", dpi=600)
    plt.show()

def summarize_results_table(results_dict, metrics=['Accuracy', 'Recall', 'Precision', 'F1-Score'], mode='median', 
                            model_order=["NB", "DT", "RF", "ET", "XGB", "LGB", "CB"], method_order=["regular", "random", "smote", "adasyn", "cvae"]):
    summary = []

    for key, data in results_dict.items():
        df = data['results_df']
        avg_time_taken = data['time'] / 20
    
        # Extract model and method from key
        model, method = key.split('_', 1) if '_' in key else (key, '')
    
        # Apply filtering
        if model_order and model not in model_order:
            continue
        if method_order and method not in method_order:
            continue
    
        entry = {'Model': model, 'Method': method}
    
        for metric in metrics:
            if metric not in df.columns:
                raise ValueError(f"Metric '{metric}' not found in results_df for '{key}'.")
    
            if mode == 'average':
                value = df[metric].mean()
            elif mode == 'best':
                value = df[metric].max()
            elif mode == 'median':
                value = df[metric].median()
            else:
                raise ValueError("mode should be 'average', 'median', or 'best'")
    
            entry[metric] = round(value, 4)
    
        entry['Avg Time (s)'] = round(avg_time_taken, 2)
        summary.append(entry)
    
    result_df = pd.DataFrame(summary)
    
    # Apply custom ordering if provided
    if model_order:
        result_df['Model'] = pd.Categorical(result_df['Model'], categories=model_order, ordered=True)
    if method_order:
        result_df['Method'] = pd.Categorical(result_df['Method'], categories=method_order, ordered=True)
    
    return result_df.sort_values(by=['Model', 'Method']).reset_index(drop=True)

def summarize_model_params(results_dict, model_name='XGB'):
    summary = []

    for key, data in results_dict.items():
        if not key.startswith(model_name):
            continue
        
        df = data['results_df']
        params_list = data['params']
        
        # Get index of the best accuracy run
        best_idx = df['Accuracy'].idxmax()
        best_params = params_list[best_idx]
        
        summary.append({
            'Model_Method': key,
            'Best Params': best_params
        })

    return pd.DataFrame(summary)
