"""
Configuration module for seismic interpretation ablation study.

This module provides configuration options for model selection, training parameters,
and evaluation metrics for the seismic interpretation ablation study.
"""

# Available CNN models
CNN_MODELS = {
    'cnn3d': {
        'class_name': 'SeismicCNN3D',
        'module': 'models.cnn_models',
        'description': 'Original 3D CNN with two convolutional layers',
        'params': {
            'num_classes': 8,
            'input_channels': 1,
            'patch_depth': 32,
            'patch_height': 32,
            'patch_width': 32
        }
    },
    'resnet3d': {
        'class_name': 'SeismicResNet3D',
        'module': 'models.cnn_models',
        'description': '3D ResNet with residual connections for deeper feature extraction',
        'params': {
            'num_classes': 8,
            'input_channels': 1,
            'patch_depth': 32,
            'patch_height': 32,
            'patch_width': 32,
            'blocks_per_layer': [2, 2, 2, 2]
        }
    },
    'attention_unet3d': {
        'class_name': 'SeismicAttentionUNet3D',
        'module': 'models.cnn_models',
        'description': '3D U-Net with attention gates for focused feature extraction',
        'params': {
            'num_classes': 8,
            'input_channels': 1,
            'patch_depth': 32,
            'patch_height': 32,
            'patch_width': 32,
            'features': [32, 64, 128, 256]
        }
    },
    'patchnet3d': {
        'class_name': 'SeismicPatchNet3D',
        'module': 'models.cnn_models',
        'description': 'Multi-scale 3D CNN with dilated convolutions for capturing features at different scales',
        'params': {
            'num_classes': 8,
            'input_channels': 1,
            'patch_depth': 32,
            'patch_height': 32,
            'patch_width': 32,
            'base_features': 32
        }
    }
}

# Available sequence models
SEQ_MODELS = {
    'bilstm': {
        'class_name': 'SeismicBiLSTM',
        'module': 'models.rnn_models',
        'description': 'Bidirectional LSTM for processing seismic traces',
        'params': {
            'num_classes': 8,
            'input_size': 1,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout_rate': 0.3
        }
    },
    'lstm': {
        'class_name': 'SeismicLSTM',
        'module': 'models.rnn_models',
        'description': 'Unidirectional LSTM for processing seismic traces',
        'params': {
            'num_classes': 8,
            'input_size': 1,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout_rate': 0.3
        }
    },
    'transformer': {
        'class_name': 'SeismicTransformer',
        'module': 'models.rnn_models',
        'description': 'Transformer model with self-attention for processing seismic traces',
        'params': {
            'num_classes': 8,
            'input_size': 1,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout_rate': 0.1
        }
    },
    'wide_deep_transformer': {
        'class_name': 'WideDeepTransformer',
        'module': 'models.rnn_models',
        'description': 'Transformer with parallel wide (shallow) and deep paths for capturing different patterns',
        'params': {
            'num_classes': 8,
            'input_size': 1,
            'd_model': 64,
            'nhead_wide': 8,
            'nhead_deep': 4,
            'num_layers_wide': 1,
            'num_layers_deep': 4,
            'dim_feedforward': 256,
            'dropout_rate': 0.1
        }
    },
    'custom_transformer': {
        'class_name': 'SeismicCustomTransformer',
        'module': 'models.rnn_models',
        'description': 'Custom transformer implementation with flexible attention mechanisms',
        'params': {
            'num_classes': 8,
            'input_size': 1,
            'embed_dim': 64,
            'num_heads': 4,
            'num_layers': 3,
            'hidden_dim': 256,
            'dropout_rate': 0.1
        }
    }
}

# Hybrid model configuration
HYBRID_MODEL = {
    'class_name': 'HybridModel',
    'module': 'models.hybrid_models',
    'description': 'Hybrid model combining CNN and sequence models',
    'params': {
        'num_classes': 8,
        'fusion_hidden_size': 128
    }
}

# Training parameters
TRAINING_PARAMS = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'optimizer': 'adam',
    'scheduler': 'reduce_on_plateau',
    'scheduler_params': {
        'factor': 0.5,
        'patience': 5,
        'min_lr': 1e-6
    }
}

# Evaluation metrics
EVALUATION_METRICS = [
    {
        'name': 'accuracy',
        'function': 'accuracy_score',
        'module': 'sklearn.metrics',
        'params': {}
    },
    {
        'name': 'precision',
        'function': 'precision_score',
        'module': 'sklearn.metrics',
        'params': {'average': 'weighted'}
    },
    {
        'name': 'recall',
        'function': 'recall_score',
        'module': 'sklearn.metrics',
        'params': {'average': 'weighted'}
    },
    {
        'name': 'f1',
        'function': 'f1_score',
        'module': 'sklearn.metrics',
        'params': {'average': 'weighted'}
    },
    {
        'name': 'confusion_matrix',
        'function': 'confusion_matrix',
        'module': 'sklearn.metrics',
        'params': {}
    },
    {
        'name': 'classification_report',
        'function': 'classification_report',
        'module': 'sklearn.metrics',
        'params': {'output_dict': True}
    },
    {
        'name': 'cohen_kappa',
        'function': 'cohen_kappa_score',
        'module': 'sklearn.metrics',
        'params': {}
    },
    {
        'name': 'balanced_accuracy',
        'function': 'balanced_accuracy_score',
        'module': 'sklearn.metrics',
        'params': {}
    }
]

# Ablation study configuration
ABLATION_CONFIG = {
    # List of CNN models to include in the ablation study
    'cnn_models': ['cnn3d', 'resnet3d', 'attention_unet3d', 'patchnet3d'],
    
    # List of sequence models to include in the ablation study
    'seq_models': ['bilstm', 'lstm', 'transformer', 'wide_deep_transformer', 'custom_transformer'],
    
    # Whether to run all combinations or a subset
    'run_all_combinations': True,
    
    # Specific combinations to run if not running all
    'specific_combinations': [
        ('cnn3d', 'bilstm'),
        ('resnet3d', 'transformer'),
        ('attention_unet3d', 'lstm'),
        ('patchnet3d', 'custom_transformer')
    ],
    
    # Results directory
    'results_dir': 'ablation_results',
    
    # Whether to save model checkpoints
    'save_checkpoints': True,
    
    # Whether to generate visualizations
    'generate_visualizations': True
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'figsize': (12, 8),
    'cmap': 'viridis',
    'dpi': 300,
    'save_format': 'png',
    'save_dir': 'visualizations',
    'plot_confusion_matrix': True,
    'plot_accuracy_comparison': True,
    'plot_f1_comparison': True,
    'plot_training_curves': True,
    'plot_feature_importance': False  # Requires additional implementation
}

# Data configuration
DATA_CONFIG = {
    'preprocessed_data_path': 'preprocessed_data/preprocessed_seismic_data.npz',
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'random_seed': 42,
    'stratify': True
}
