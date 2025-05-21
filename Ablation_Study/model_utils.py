"""
Utility functions for model loading, training, and evaluation.

This module provides helper functions for dynamically loading models from configuration,
training models, and evaluating their performance.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import importlib
import json
from tqdm.auto import tqdm
import pandas as pd
import hashlib
import time
import glob

def load_model_from_config(model_config, **kwargs):
    """
    Dynamically load a model from its configuration.
    
    Args:
        model_config (dict): Model configuration dictionary
        **kwargs: Additional parameters to override the default ones
    
    Returns:
        nn.Module: Instantiated model
    """
    # Import the module containing the model class
    module = importlib.import_module(model_config['module'])
    
    # Get the model class
    model_class = getattr(module, model_config['class_name'])
    
    # Prepare parameters by combining defaults with overrides
    params = model_config['params'].copy()
    params.update(kwargs)
    
    # Instantiate the model
    model = model_class(**params)
    
    return model

def load_hybrid_model(cnn_config, seq_config, hybrid_config, **kwargs):
    """
    Load a hybrid model combining a CNN and a sequence model.
    
    Args:
        cnn_config (dict): CNN model configuration
        seq_config (dict): Sequence model configuration
        hybrid_config (dict): Hybrid model configuration
        **kwargs: Additional parameters to override the defaults
    
    Returns:
        nn.Module: Instantiated hybrid model
    """
    # Load the CNN model
    cnn_model = load_model_from_config(cnn_config)
    
    # Load the sequence model
    seq_model = load_model_from_config(seq_config)
    
    # Import the hybrid model module
    module = importlib.import_module(hybrid_config['module'])
    
    # Get the hybrid model class
    hybrid_class = getattr(module, hybrid_config['class_name'])
    
    # Prepare parameters
    params = hybrid_config['params'].copy()
    params.update(kwargs)
    
    # Instantiate the hybrid model
    hybrid_model = hybrid_class(cnn_model, seq_model, **params)
    
    return hybrid_model

def create_dataloaders(X_patches, y_labels, X_traces=None, batch_size=32, 
                       train_split=0.7, val_split=0.15, test_split=0.15, 
                       random_seed=42, stratify=True):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        X_patches (np.ndarray): 3D patches data (N, C, D, H, W)
        y_labels (np.ndarray): Labels (N,)
        X_traces (np.ndarray, optional): Trace data (N, D)
        batch_size (int): Batch size
        train_split (float): Proportion of data for training
        val_split (float): Proportion of data for validation
        test_split (float): Proportion of data for testing
        random_seed (int): Random seed for reproducibility
        stratify (bool): Whether to stratify the splits by class
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Convert numpy arrays to PyTorch tensors
    X_patches_tensor = torch.tensor(X_patches, dtype=torch.float32)
    y_labels_tensor = torch.tensor(y_labels, dtype=torch.long)
    
    if X_traces is not None:
        X_traces_tensor = torch.tensor(X_traces, dtype=torch.float32)
        dataset = TensorDataset(X_patches_tensor, X_traces_tensor, y_labels_tensor)
    else:
        dataset = TensorDataset(X_patches_tensor, y_labels_tensor)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Create splits
    if stratify:
        # This is a simplified approach; for true stratification, consider using StratifiedKFold
        # from sklearn and implementing a custom splitting logic
        torch.manual_seed(random_seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
    else:
        torch.manual_seed(random_seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                num_epochs=50, device='cuda', early_stopping_patience=10, 
                save_path=None, verbose=True):
    """
    Train a model with early stopping.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs (int): Maximum number of epochs
        device (str): Device to use ('cuda' or 'cpu')
        early_stopping_patience (int): Patience for early stopping
        save_path (str, optional): Path to save the best model
        verbose (bool): Whether to print progress
    
    Returns:
        tuple: (trained_model, history)
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") if verbose else train_loader
        
        for batch in train_iterator:
            # Handle different batch structures
            if len(batch) == 3:  # X_patches, X_traces, y
                X_patches, X_traces, y = batch
                X_patches, X_traces, y = X_patches.to(device), X_traces.to(device), y.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(X_patches, X_traces)
            else:  # X_patches, y
                X_patches, y = batch
                X_patches, y = X_patches.to(device), y.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(X_patches)
            
            loss = criterion(outputs, y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * X_patches.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") if verbose else val_loader
        
        with torch.no_grad():
            for batch in val_iterator:
                # Handle different batch structures
                if len(batch) == 3:  # X_patches, X_traces, y
                    X_patches, X_traces, y = batch
                    X_patches, X_traces, y = X_patches.to(device), X_traces.to(device), y.to(device)
                    
                    # Forward pass
                    outputs = model(X_patches, X_traces)
                else:  # X_patches, y
                    X_patches, y = batch
                    X_patches, y = X_patches.to(device), y.to(device)
                    
                    # Forward pass
                    outputs = model(X_patches)
                
                loss = criterion(outputs, y)
                
                # Statistics
                val_loss += loss.item() * X_patches.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model if saved
    if save_path is not None and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    
    return model, history

def evaluate_model(model, test_loader, metrics_config, device='cuda', class_names=None):
    """
    Evaluate a model on test data.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        metrics_config (list): List of metric configurations
        device (str): Device to use ('cuda' or 'cpu')
        class_names (list, optional): List of class names
    
    Returns:
        dict: Evaluation results
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Initialize lists to store predictions and ground truth
    all_preds = []
    all_labels = []
    
    # Evaluate
    with torch.no_grad():
        for batch in test_loader:
            # Handle different batch structures
            if len(batch) == 3:  # X_patches, X_traces, y
                X_patches, X_traces, y = batch
                X_patches, X_traces, y = X_patches.to(device), X_traces.to(device), y.to(device)
                
                # Forward pass
                outputs = model(X_patches, X_traces)
            else:  # X_patches, y
                X_patches, y = batch
                X_patches, y = X_patches.to(device), y.to(device)
                
                # Forward pass
                outputs = model(X_patches)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    results = {}
    for metric_config in metrics_config:
        metric_name = metric_config['name']
        metric_function_name = metric_config['function']
        metric_module = importlib.import_module(metric_config['module'])
        metric_function = getattr(metric_module, metric_function_name)
        
        # Apply metric function with parameters
        metric_params = metric_config.get('params', {})
        if metric_name == 'confusion_matrix':
            if class_names is not None:
                cm = metric_function(all_labels, all_preds, **metric_params)
                results[metric_name] = cm
            else:
                cm = metric_function(all_labels, all_preds, **metric_params)
                results[metric_name] = cm
        elif metric_name == 'classification_report':
            report = metric_function(all_labels, all_preds, **metric_params)
            results[metric_name] = report
        else:
            results[metric_name] = metric_function(all_labels, all_preds, **metric_params)
    
    return results

def plot_confusion_matrix(cm, class_names=None, figsize=(10, 8), cmap='Blues', 
                          normalize=True, save_path=None):
    """
    Plot a confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list, optional): List of class names
        figsize (tuple): Figure size
        cmap (str): Colormap
        normalize (bool): Whether to normalize the confusion matrix
        save_path (str, optional): Path to save the figure
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return plt.gcf()

def plot_training_history(history, figsize=(12, 5), save_path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Training history
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return plt.gcf()

def plot_model_comparison(results_df, metric='accuracy', figsize=(12, 8), 
                          save_path=None, sort=True):
    """
    Plot model comparison.
    
    Args:
        results_df (pd.DataFrame): DataFrame with model results
        metric (str): Metric to compare
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        sort (bool): Whether to sort by metric value
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=figsize)
    
    if sort:
        results_df = results_df.sort_values(by=metric, ascending=False)
    
    sns.barplot(x=metric, y='model_name', data=results_df)
    plt.title(f'Model Comparison - {metric}')
    plt.xlabel(metric.capitalize())
    plt.ylabel('Model')
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return plt.gcf()

def generate_model_combination_id(cnn_key, seq_key, config):
    """
    Generate a unique identifier for a model combination.
    
    Args:
        cnn_key (str): CNN model key
        seq_key (str): Sequence model key
        config: Configuration object
    
    Returns:
        str: Unique identifier for the model combination
    """
    # Create a string representation of the model combination
    cnn_config = config.CNN_MODELS[cnn_key]
    seq_config = config.SEQ_MODELS[seq_key]
    
    # Include model names and key parameters in the hash
    cnn_params = str(cnn_config['params'])
    seq_params = str(seq_config['params'])
    training_params = str(config.TRAINING_PARAMS)
    
    # Create a hash of the configuration to ensure uniqueness
    combination_str = f"{cnn_key}_{seq_key}_{cnn_params}_{seq_params}_{training_params}"
    combination_hash = hashlib.md5(combination_str.encode()).hexdigest()[:10]
    
    return f"{cnn_key}_{seq_key}_{combination_hash}"

def is_model_combination_completed(model_id, results_dir):
    """
    Check if a model combination has already been trained and evaluated.
    
    Args:
        model_id (str): Model combination identifier
        results_dir (str): Directory where results are stored
    
    Returns:
        bool: True if the model combination has been completed, False otherwise
    """
    # Check for the existence of the completion marker file
    completion_marker = os.path.join(results_dir, f"{model_id}_completed.json")
    
    if os.path.exists(completion_marker):
        try:
            # Verify the completion marker file is valid
            with open(completion_marker, 'r') as f:
                completion_data = json.load(f)
            
            # Check if all required files exist
            required_files = [
                os.path.join(results_dir, f"{model_id}_best.pth"),
                os.path.join(results_dir, f"{model_id}_history.json"),
                os.path.join(results_dir, f"{model_id}_results.json")
            ]
            
            all_files_exist = all(os.path.exists(file) for file in required_files)
            
            # Verify the completion status and timestamp
            is_complete = completion_data.get('completed', False)
            
            return is_complete and all_files_exist
        except (json.JSONDecodeError, KeyError, TypeError):
            # If the file is corrupted or missing required fields, consider it incomplete
            return False
    
    return False

def mark_model_combination_completed(model_id, results_dir, result_entry):
    """
    Mark a model combination as completed.
    
    Args:
        model_id (str): Model combination identifier
        results_dir (str): Directory where results are stored
        result_entry (dict): Result data for the model combination
    """
    # Create a completion marker file
    completion_marker = os.path.join(results_dir, f"{model_id}_completed.json")
    
    # Prepare completion data
    completion_data = {
        'model_id': model_id,
        'completed': True,
        'timestamp': time.time(),
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics_summary': {
            key: value for key, value in result_entry.items() 
            if key not in ['model_name', 'cnn_model', 'seq_model', 'cnn_description', 'seq_description']
            and isinstance(value, (int, float, np.int32, np.int64, np.float32, np.float64))
        }
    }
    
    # Write the completion marker file
    with open(completion_marker, 'w') as f:
        json.dump(completion_data, f, indent=2)

def load_existing_results(results_dir):
    """
    Load existing results from completed model combinations.
    
    Args:
        results_dir (str): Directory where results are stored
    
    Returns:
        list: List of result entries for completed model combinations
    """
    results_list = []
    
    # Find all completed model combinations
    completion_markers = glob.glob(os.path.join(results_dir, "*_completed.json"))
    
    for marker in completion_markers:
        model_id = os.path.basename(marker).replace("_completed.json", "")
        result_file = os.path.join(results_dir, f"{model_id}_results.json")
        
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result_entry = json.load(f)
                results_list.append(result_entry)
            except (json.JSONDecodeError, KeyError, TypeError):
                # Skip corrupted result files
                print(f"Warning: Corrupted result file for {model_id}")
    
    return results_list

def run_ablation_study(config, X_patches, y_labels, X_traces=None, device='cuda', 
                       class_names=None, results_dir='ablation_results'):
    """
    Run an ablation study with different model combinations.
    
    Args:
        config: Configuration object
        X_patches (np.ndarray): 3D patches data
        y_labels (np.ndarray): Labels
        X_traces (np.ndarray, optional): Trace data
        device (str): Device to use ('cuda' or 'cpu')
        class_names (list, optional): List of class names
        results_dir (str): Directory to save results
    
    Returns:
        pd.DataFrame: DataFrame with results
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_patches, y_labels, X_traces,
        batch_size=config.TRAINING_PARAMS['batch_size'],
        train_split=config.DATA_CONFIG['train_split'],
        val_split=config.DATA_CONFIG['val_split'],
        test_split=config.DATA_CONFIG['test_split'],
        random_seed=config.DATA_CONFIG['random_seed'],
        stratify=config.DATA_CONFIG['stratify']
    )
    
    # Determine model combinations to run
    if config.ABLATION_CONFIG['run_all_combinations']:
        combinations = []
        for cnn_key in config.ABLATION_CONFIG['cnn_models']:
            for seq_key in config.ABLATION_CONFIG['seq_models']:
                combinations.append((cnn_key, seq_key))
    else:
        combinations = config.ABLATION_CONFIG['specific_combinations']
    
    # Load existing results
    results_list = load_existing_results(results_dir)
    
    # Create a progress file to track overall progress
    progress_file = os.path.join(results_dir, 'ablation_progress.json')
    total_combinations = len(combinations)
    completed_combinations = 0
    
    # Run each combination
    for cnn_key, seq_key in combinations:
        # Generate a unique model ID for this combination
        model_id = generate_model_combination_id(cnn_key, seq_key, config)
        model_name = f"{cnn_key}_{seq_key}"
        
        # Check if this combination has already been completed
        if is_model_combination_completed(model_id, results_dir):
            print(f"\n=== Skipping {model_name} (already completed) ===\n")
            completed_combinations += 1
            
            # Update progress
            progress_data = {
                'total_combinations': total_combinations,
                'completed_combinations': completed_combinations,
                'progress_percentage': (completed_combinations / total_combinations) * 100,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
            continue
        
        print(f"\n=== Training {model_name} ===\n")
        
        # Get model configurations
        cnn_config = config.CNN_MODELS[cnn_key]
        seq_config = config.SEQ_MODELS[seq_key]
        
        # Load hybrid model
        model = load_hybrid_model(cnn_config, seq_config, config.HYBRID_MODEL)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.TRAINING_PARAMS['learning_rate'],
            weight_decay=config.TRAINING_PARAMS['weight_decay']
        )
        
        # Define scheduler
        if config.TRAINING_PARAMS['scheduler'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **config.TRAINING_PARAMS['scheduler_params']
            )
        elif config.TRAINING_PARAMS['scheduler'] == 'step_lr':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                **config.TRAINING_PARAMS['scheduler_params']
            )
        else:
            scheduler = None
        
        # Train model
        save_path = os.path.join(results_dir, f"{model_id}_best.pth") if config.ABLATION_CONFIG['save_checkpoints'] else None
        
        try:
            model, history = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                scheduler=scheduler,
                num_epochs=config.TRAINING_PARAMS['num_epochs'],
                device=device,
                early_stopping_patience=config.TRAINING_PARAMS['early_stopping_patience'],
                save_path=save_path,
                verbose=True
            )
            
            # Save training history
            history_path = os.path.join(results_dir, f"{model_id}_history.json")
            with open(history_path, 'w') as f:
                json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
            
            # Plot training history
            if config.ABLATION_CONFIG['generate_visualizations'] and config.VISUALIZATION_CONFIG['plot_training_curves']:
                history_plot_path = os.path.join(results_dir, f"{model_id}_training_curves.{config.VISUALIZATION_CONFIG['save_format']}")
                plot_training_history(history, figsize=config.VISUALIZATION_CONFIG['figsize'], save_path=history_plot_path)
            
            # Evaluate model
            results = evaluate_model(model, test_loader, config.EVALUATION_METRICS, device=device, class_names=class_names)
            
            # Plot confusion matrix
            if config.ABLATION_CONFIG['generate_visualizations'] and config.VISUALIZATION_CONFIG['plot_confusion_matrix']:
                cm_plot_path = os.path.join(results_dir, f"{model_id}_confusion_matrix.{config.VISUALIZATION_CONFIG['save_format']}")
                plot_confusion_matrix(
                    results['confusion_matrix'],
                    class_names=class_names,
                    figsize=config.VISUALIZATION_CONFIG['figsize'],
                    cmap=config.VISUALIZATION_CONFIG['cmap'],
                    normalize=True,
                    save_path=cm_plot_path
                )
            
            # Save results
            result_entry = {
                'model_id': model_id,
                'model_name': model_name,
                'cnn_model': cnn_key,
                'seq_model': seq_key,
                'cnn_description': cnn_config['description'],
                'seq_description': seq_config['description']
            }
            
            # Add metrics to result entry
            for metric_name, metric_value in results.items():
                if metric_name not in ['confusion_matrix', 'classification_report']:
                    result_entry[metric_name] = metric_value
            
            # Add per-class metrics from classification report
            if 'classification_report' in results:
                report = results['classification_report']
                for class_idx, class_metrics in report.items():
                    if isinstance(class_metrics, dict):  # Skip 'accuracy', 'macro avg', etc.
                        class_name = class_names[int(class_idx)] if class_names is not None else f"class_{class_idx}"
                        for metric_name, metric_value in class_metrics.items():
                            result_entry[f"{class_name}_{metric_name}"] = metric_value
            
            # Add to results list
            results_list.append(result_entry)
            
            # Save individual result
            result_path = os.path.join(results_dir, f"{model_id}_results.json")
            with open(result_path, 'w') as f:
                json.dump({k: float(v) if isinstance(v, (int, float, np.int32, np.int64, np.float32, np.float64)) else v 
                          for k, v in result_entry.items()}, f, indent=2)
            
            # Mark this combination as completed
            mark_model_combination_completed(model_id, results_dir, result_entry)
            
            # Update combined results after each model
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)
            
            # Update progress
            completed_combinations += 1
            progress_data = {
                'total_combinations': total_combinations,
                'completed_combinations': completed_combinations,
                'progress_percentage': (completed_combinations / total_combinations) * 100,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
        except Exception as e:
            # Log the error
            error_log_path = os.path.join(results_dir, f"{model_id}_error.log")
            with open(error_log_path, 'w') as f:
                f.write(f"Error training {model_name}: {str(e)}\n")
            print(f"Error training {model_name}: {str(e)}")
            
            # Continue with next combination
            continue
    
    # Create DataFrame with all results
    results_df = pd.DataFrame(results_list)
    
    # Save combined results
    results_df.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)
    
    # Plot model comparisons
    if config.ABLATION_CONFIG['generate_visualizations']:
        if config.VISUALIZATION_CONFIG['plot_accuracy_comparison']:
            acc_plot_path = os.path.join(results_dir, f"accuracy_comparison.{config.VISUALIZATION_CONFIG['save_format']}")
            plot_model_comparison(
                results_df, metric='accuracy',
                figsize=config.VISUALIZATION_CONFIG['figsize'],
                save_path=acc_plot_path
            )
        
        if config.VISUALIZATION_CONFIG['plot_f1_comparison']:
            f1_plot_path = os.path.join(results_dir, f"f1_comparison.{config.VISUALIZATION_CONFIG['save_format']}")
            plot_model_comparison(
                results_df, metric='f1',
                figsize=config.VISUALIZATION_CONFIG['figsize'],
                save_path=f1_plot_path
            )
    
    return results_df
