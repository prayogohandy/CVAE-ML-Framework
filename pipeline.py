import os
import time
import numpy as np
import pandas as pd
from collections import Counter
import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from functions import *
from metrics import *

class_names = ["F", "FS", "S", "SL"]


def get_model(model_name, trial):
    """Return model and hyperparameter space for Optuna."""
    if model_name == "RF":
        params = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500),
            "max_depth": trial.suggest_int("rf_max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0)
        }
        return RandomForestClassifier(**params)

    elif model_name == "XGB":
        params = {
            "learning_rate": trial.suggest_float("xgb_learning_rate", 1e-4, 0.3, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("xgb_min_child_weight", 1e-2, 10, log=True),
            "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-4, 1e2, log=True),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 1e2, log=True),
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 500)
        }
        return XGBClassifier(**params)

    elif model_name == "LGB":
        params = {
            "learning_rate": trial.suggest_float("lgb_learning_rate", 1e-4, 0.3, log=True),
            "num_leaves": trial.suggest_int("lgb_num_leaves", 10, 100),
            "max_depth": trial.suggest_int("lgb_max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("lgb_min_child_samples", 5, 50),
            "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("lgb_colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-4, 1e2, log=True),
            "reg_alpha": trial.suggest_float("lgb_reg_alpha", 1e-4, 1e2, log=True),
            "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 500)
        }
        return LGBMClassifier(**params, verbose=-1)

    elif model_name == "CB":
        params = {
            'iterations': trial.suggest_int('cb_iterations', 100, 500),
            'depth': trial.suggest_int('cb_depth', 3, 12),
            'learning_rate': trial.suggest_float('cb_learning_rate', 1e-4, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('cb_l2_leaf_reg', 1e-4, 1e2, log=True),
            'border_count': trial.suggest_int('cb_border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('cb_bagging_temperature', 0, 2)
        }
        return CatBoostClassifier(**params, verbose=0)

    elif model_name == "AdaBoost":
        params = {
            "n_estimators": trial.suggest_int("adaboost_n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("adaboost_learning_rate", 1e-4, 0.3, log=True)
        }
        return AdaBoostClassifier(**params)

    elif model_name == "ET":
        params = {
            "n_estimators": trial.suggest_int("et_n_estimators", 100, 500),
            "max_depth": trial.suggest_int("et_max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("et_min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("et_min_samples_leaf", 1, 10),
        }
        return ExtraTreesClassifier(**params)

    elif model_name == "DT":
        params = {
            "max_depth": trial.suggest_int("dt_max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("dt_min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("dt_min_samples_leaf", 1, 10),
        }
        return DecisionTreeClassifier(**params)

    elif model_name == "MLP":
        num_layers = trial.suggest_int("mlp_num_layers", 3, 12)  # Number of layers between 2 and 5
        hidden_layer_sizes = tuple(
            trial.suggest_int(f"mlp_layer_{i}", 50, 200, step=25) for i in range(num_layers)  # Nodes as multiples of 25
        )
        params = {
            "hidden_layer_sizes": hidden_layer_sizes,  # Suggested hidden layer sizes
            "activation": trial.suggest_categorical("mlp_activation", ["relu", "tanh", "logistic"]),  # Activation function
            "alpha": trial.suggest_float("mlp_alpha", 1e-4, 1e-2, log=True)  # Regularization term alpha
        }
        return MLPClassifier(**params)

    elif model_name == "NB":
        return GaussianNB()

    else:
        raise ValueError(f"Model {model_name} is not supported!")

# Hyperparameter optimization
def objective(trial, model_name, X, y):
    """Optuna objective function for tuning hyperparameters using CV."""
    model = get_model(model_name, trial)
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # Correct use of skf
    scores = []
    
    for step, (train_idx, val_idx) in enumerate(skf.split(X, y)):  # Use skf here
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred)
        scores.append(score)
        
        trial.report(np.mean(scores), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(scores)  # Return average accuracy over folds

def tune_model(model_name, X, y):
    """Run Optuna optimization for a given model."""
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())  
    study.optimize(lambda trial: objective(trial, model_name, X, y), n_trials=200, show_progress_bar=True)
    return study.best_params

def run_pipeline(model_name, folds, name_prefix, temp_save='temp_single_pipeline.pkl'):
    # Check if temp_save exists to resume
    if os.path.exists(temp_save):
        # Load previous results
        saved_data = load_results(temp_save)
        completed_iterations = {entry['iteration'] for entry in saved_data}
        y_pred_folds = [entry['y_pred'] for entry in saved_data]
        y_test_folds = [entry['y_test'] for entry in saved_data]
        result_folds = [entry['metrics'] for entry in saved_data]
        params_folds = [entry['params'] for entry in saved_data]
        print(f"Resuming from iteration {max(completed_iterations) + 1}")
    else:
        # Initialize if no previous results
        saved_data = []  # Use this to store all results for saving at each step
        completed_iterations = set()
        y_pred_folds = []
        y_test_folds = []
        result_folds = []
        params_folds = []

    for iteration, (X_train, y_train, X_test, y_test) in enumerate(folds):
        if iteration in completed_iterations:
            # Skip completed iterations
            continue
        
        if isinstance(model_name, str) or isinstance(model_name, list):            
            best_params = tune_model(model_name, X_train, y_train)
            model = get_model(model_name, optuna.trial.FixedTrial(best_params))
            params_folds.append(best_params)
        elif hasattr(model_name, "fit"):
            model = model_name
            best_params = None  # No hyperparameters if model is passed directly
        else:
            raise ValueError("Model is not supported!")
        
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_folds.append(y_pred)
        y_test_folds.append(y_test)
        
        # Get classification results
        accuracy, precision, recall, f1_score = classification_result(y_test, y_pred)
        fold_metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score
        }
        result_folds.append(fold_metrics)

        # Prepare result for this iteration
        result_entry = {
            'iteration': iteration,
            'params': best_params,
            'y_pred': y_pred,
            'y_test': y_test,
            'metrics': fold_metrics
        }
        saved_data.append(result_entry)

        # Save the entire list of results to temp_save after each iteration
        save_results(saved_data, temp_save)

    # Concatenate predictions and ground truth
    y_preds = np.concatenate(y_pred_folds)
    y_tests = np.concatenate(y_test_folds)
    
    out = extended_confusion_matrix(y_tests, y_preds, class_names=class_names, prefix=name_prefix)

    # Convert list of tuples into a DataFrame
    df_results = pd.DataFrame(result_folds, columns=["Accuracy", "Precision", "Recall", "F1-Score"])
    df_results['Mean'] = df_results[['Accuracy', 'Precision', "Recall", "F1-Score"]].mean(axis=1)
    df_results['Std'] = df_results[['Accuracy', "Precision", "Recall", "F1-Score"]].std(axis=1)
    
    # Remove temp_save if complete
    if os.path.exists(temp_save):
        os.remove(temp_save)
    
    return out, df_results, params_folds

def run_multiple_pipelines(models, folds_dict, save_path="temp_multi_pipeline.pkl", alt_names=None):
    """
    Run pipeline for multiple models and fold types and store the results.
    
    Parameters:
    -----------
    models : list
        List of model names (e.g., ["XGB", "LGB", "CB", "RF", "SVM", "Stacking"]).
    folds_dict : dict
        Dictionary of fold types (e.g., {"regular": regular_folds, "smote": smote_folds, "cvae": cvae_folds}).
    save_path : str, optional
        Path to save the results dictionary incrementally.

    Returns:
    --------
    results_dict : dict
        Dictionary storing the results for each model and fold type.
    """
    # Load existing results if the file exists
    if os.path.exists(save_path):
        results_dict = load_results(filename=save_path)
    else:
        results_dict = {}

    # Count occurrences of model names
    name_counter = Counter()
    alt_count = 0
    for model_name in models:
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

        # Track duplicates
        name_counter[name] += 1
        if name_counter[name] > 1:
            name = f"{name}_{name_counter[name]}"

        for fold_type, folds in folds_dict.items():
            key = f"{name}_{fold_type}"
            print(f"Running {name} on {fold_type} folds")
            
            if key in results_dict:
                # skip the computation
                output = results_dict[key]["final_cm"]
                replot_extended_confusion_matrix(output, n_run=20, class_names=class_names, prefix=key+"_avg")
                continue
            
            start_time = time.perf_counter()
            out, df_results, params_folds = run_pipeline(model_name, folds, key)
            elapsed_time = time.perf_counter() - start_time
            
            # Store the results
            results_dict[key] = {
                "final_cm": out,
                "results_df": df_results,
                "params": params_folds,
                "time": elapsed_time
            }

            # Save progress after every model-fold execution
            save_results(results_dict, filename=save_path)
    return results_dict