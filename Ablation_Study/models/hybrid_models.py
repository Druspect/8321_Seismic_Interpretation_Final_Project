# models/hybrid_models.py

import torch
import torch.nn as nn
import inspect

class HybridModel(nn.Module):
    """Generalized hybrid model combining any CNN and sequence model.
    
    This model allows flexible combination of any CNN model with any sequence model
    for seismic facies classification. It extracts features from both models and
    combines them through a joint classifier.
    
    Args:
        cnn_model (nn.Module): Any CNN model from cnn_models.py
        seq_model (nn.Module): Any sequence model from rnn_models.py
        num_classes (int): Number of output classes
        fusion_hidden_size (int): Size of the hidden layer in the fusion classifier
    """
    def __init__(self, cnn_model, seq_model, num_classes=8, fusion_hidden_size=128):
        super(HybridModel, self).__init__()
        
        # Store the models
        self.cnn_model = cnn_model
        self.seq_model = seq_model
        
        # Extract CNN feature extractor
        self._extract_cnn_features()
        
        # Extract sequence model feature extractor
        self._extract_seq_features()
        
        # Determine feature sizes
        cnn_feature_size = self._get_cnn_feature_size()
        seq_feature_size = self._get_seq_feature_size()
        
        print(f"Initializing HybridModel...")
        print(f"  CNN model: {cnn_model.__class__.__name__}")
        print(f"  Sequence model: {seq_model.__class__.__name__}")
        print(f"  CNN feature size: {cnn_feature_size}")
        print(f"  Sequence feature size: {seq_feature_size}")
        
        # Create combined classifier
        combined_feature_size = cnn_feature_size + seq_feature_size
        self.combined_classifier = nn.Sequential(
            nn.Linear(combined_feature_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_hidden_size, num_classes)
        )
        print(f"  Combined feature size: {combined_feature_size}")
        print(f"  Fusion hidden size: {fusion_hidden_size}")
        print(f"  Number of classes: {num_classes}")
        
    def _extract_cnn_features(self):
        """Extract the feature extractor part from the CNN model."""
        # Try different attribute names that might contain the feature extractor
        if hasattr(self.cnn_model, 'features'):
            self.cnn_features = self.cnn_model.features
        elif hasattr(self.cnn_model, 'encoder'):
            self.cnn_features = self.cnn_model.encoder
        elif hasattr(self.cnn_model, 'feature_extractor'):
            self.cnn_features = self.cnn_model.feature_extractor
        else:
            # If no standard feature extractor is found, use the model up to the classifier
            # This is a fallback and might not work for all models
            print("Warning: Could not identify CNN feature extractor. Using model without classifier.")
            # Create a new Sequential module with all layers except the classifier
            if hasattr(self.cnn_model, 'classifier'):
                modules = list(self.cnn_model.children())[:-1]  # All except the last module
                self.cnn_features = nn.Sequential(*modules)
            else:
                raise ValueError("Cannot extract features from CNN model. Please ensure the model has a 'features', 'encoder', 'feature_extractor', or 'classifier' attribute.")
    
    def _extract_seq_features(self):
        """Extract the feature extractor part from the sequence model."""
        # For LSTM/BiLSTM models
        if hasattr(self.seq_model, 'lstm'):
            self.seq_features = self.seq_model.lstm
            self.seq_model_type = 'lstm'
        # For transformer models
        elif hasattr(self.seq_model, 'transformer_encoder'):
            self.seq_features = self.seq_model.transformer_encoder
            self.seq_model_type = 'transformer'
            # Extract embedding and positional encoding if available
            if hasattr(self.seq_model, 'embedding'):
                self.seq_embedding = self.seq_model.embedding
            if hasattr(self.seq_model, 'pos_encoder'):
                self.seq_pos_encoder = self.seq_model.pos_encoder
        elif hasattr(self.seq_model, 'transformer_blocks'):
            self.seq_features = self.seq_model.transformer_blocks
            self.seq_model_type = 'custom_transformer'
            # Extract embedding and positional encoding if available
            if hasattr(self.seq_model, 'embedding'):
                self.seq_embedding = self.seq_model.embedding
            if hasattr(self.seq_model, 'pos_encoder'):
                self.seq_pos_encoder = self.seq_model.pos_encoder
        elif hasattr(self.seq_model, 'wide_transformer') and hasattr(self.seq_model, 'deep_transformer'):
            # Special case for WideDeepTransformer
            self.wide_transformer = self.seq_model.wide_transformer
            self.deep_transformer = self.seq_model.deep_transformer
            self.seq_model_type = 'wide_deep_transformer'
            # Need embedding and positional encoding too
            if hasattr(self.seq_model, 'embedding'):
                self.seq_embedding = self.seq_model.embedding
            if hasattr(self.seq_model, 'pos_encoder'):
                self.seq_pos_encoder = self.seq_model.pos_encoder
        else:
            raise ValueError("Cannot extract features from sequence model. Please ensure the model has a 'lstm', 'transformer_encoder', 'transformer_blocks', or 'wide_transformer'/'deep_transformer' attributes.")
    
    def _get_cnn_feature_size(self):
        """Determine the feature size of the CNN model."""
        # Try to get feature size from model attributes
        if hasattr(self.cnn_model, 'flattened_size'):
            return self.cnn_model.flattened_size
        
        # Try to infer from the first layer of the classifier
        try:
            if hasattr(self.cnn_model, 'classifier') and isinstance(self.cnn_model.classifier, nn.Sequential):
                for layer in self.cnn_model.classifier:
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
        except (AttributeError, IndexError):
            pass
        
        # If all else fails, do a forward pass with a dummy input
        print("Warning: Could not determine CNN feature size from model attributes. Using dummy forward pass.")
        with torch.no_grad():
            # Try to infer input shape from model's __init__ parameters
            init_params = inspect.signature(self.cnn_model.__class__.__init__).parameters
            input_channels = 1  # Default
            patch_depth = patch_height = patch_width = 32  # Default
            
            if 'input_channels' in init_params:
                input_channels = getattr(self.cnn_model, 'input_channels', 1)
            if 'patch_depth' in init_params:
                patch_depth = getattr(self.cnn_model, 'patch_depth', 32)
            if 'patch_height' in init_params:
                patch_height = getattr(self.cnn_model, 'patch_height', 32)
            if 'patch_width' in init_params:
                patch_width = getattr(self.cnn_model, 'patch_width', 32)
            
            dummy_input = torch.zeros(1, input_channels, patch_depth, patch_height, patch_width)
            features = self.cnn_features(dummy_input)
            
            # Handle different output shapes
            if isinstance(features, torch.Tensor):
                if features.dim() > 2:
                    features = features.view(1, -1)
                return features.size(1)
            else:
                # If features is not a tensor (e.g., a list of tensors), this is a more complex model
                # We would need custom handling here
                raise ValueError("CNN feature extractor output is not a tensor. Custom handling required.")
    
    def _get_seq_feature_size(self):
        """Determine the feature size of the sequence model."""
        # For LSTM/BiLSTM models
        if self.seq_model_type == 'lstm':
            hidden_size = self.seq_features.hidden_size
            num_directions = 2 if self.seq_features.bidirectional else 1
            return hidden_size * num_directions
        
        # For transformer models
        elif self.seq_model_type in ['transformer', 'custom_transformer']:
            # Try to get from model attributes
            if hasattr(self.seq_model, 'd_model'):
                return self.seq_model.d_model
            elif hasattr(self.seq_model, 'embed_dim'):
                return self.seq_model.embed_dim
            
            # Try to infer from the first layer of the classifier
            try:
                if hasattr(self.seq_model, 'classifier') and isinstance(self.seq_model.classifier, nn.Sequential):
                    for layer in self.seq_model.classifier:
                        if isinstance(layer, nn.Linear):
                            return layer.in_features
            except (AttributeError, IndexError):
                pass
            
            # Default for most transformer models
            return 64  # Common default size
        
        # For WideDeepTransformer
        elif self.seq_model_type == 'wide_deep_transformer':
            # This model has a fusion layer that combines wide and deep paths
            if hasattr(self.seq_model, 'fusion'):
                return self.seq_model.fusion.out_features
            else:
                # Try to get d_model
                return getattr(self.seq_model, 'd_model', 64)
        
        else:
            raise ValueError(f"Unknown sequence model type: {self.seq_model_type}")
    
    def _ensure_embedding_layer(self, input_size, d_model):
        """Ensure embedding layer exists with correct dimensions."""
        if not hasattr(self, 'seq_embedding'):
            self.seq_embedding = nn.Linear(input_size, d_model)
            print(f"Created embedding layer: input_size={input_size}, d_model={d_model}")
    
    def _ensure_positional_encoding(self, d_model):
        """Ensure positional encoding layer exists with correct dimensions."""
        if not hasattr(self, 'seq_pos_encoder'):
            # Import here to avoid circular imports
            from models.rnn_models import PositionalEncoding
            self.seq_pos_encoder = PositionalEncoding(d_model, dropout=0.1)
            print(f"Created positional encoding layer: d_model={d_model}")
    
    def forward(self, cnn_input, seq_input):
        """Forward pass through the hybrid model.
        
        Args:
            cnn_input (torch.Tensor): Input for the CNN model (batch_size, channels, depth, height, width)
            seq_input (torch.Tensor): Input for the sequence model (batch_size, seq_len) or (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: Classification output (batch_size, num_classes)
        """
        # Process CNN input
        cnn_features = self.cnn_features(cnn_input)
        if cnn_features.dim() > 2:
            cnn_features = cnn_features.view(cnn_features.size(0), -1)  # Flatten
        
        # Process sequence input
        # Ensure seq_input has the right shape
        if seq_input.dim() == 2:  # (batch_size, seq_len)
            seq_input = seq_input.unsqueeze(-1)  # Add feature dimension
        
        # Different handling based on sequence model type
        if self.seq_model_type == 'lstm':
            seq_output, _ = self.seq_features(seq_input)
            seq_features = seq_output[:, -1, :]  # Use last time step
            
        elif self.seq_model_type == 'transformer':
            # For standard transformer models
            # Get the expected embedding dimension
            if hasattr(self.seq_model, 'd_model'):
                d_model = self.seq_model.d_model
            elif hasattr(self.seq_model, 'embed_dim'):
                d_model = self.seq_model.embed_dim
            else:
                d_model = 64  # Default from config
            
            # Ensure embedding and positional encoding layers exist
            self._ensure_embedding_layer(seq_input.size(-1), d_model)
            self._ensure_positional_encoding(d_model)
            
            # Apply embedding and positional encoding
            seq_input = self.seq_embedding(seq_input)
            seq_input = self.seq_pos_encoder(seq_input)
            
            # Pass through transformer encoder
            seq_features = self.seq_features(seq_input)[:, -1, :]  # Use last time step
            
        elif self.seq_model_type == 'custom_transformer':
            # For custom transformer with blocks
            # Get the expected embedding dimension
            if hasattr(self.seq_model, 'embed_dim'):
                d_model = self.seq_model.embed_dim
            elif hasattr(self.seq_model, 'd_model'):
                d_model = self.seq_model.d_model
            else:
                d_model = 64  # Default from config
            
            # Ensure embedding and positional encoding layers exist
            self._ensure_embedding_layer(seq_input.size(-1), d_model)
            self._ensure_positional_encoding(d_model)
            
            # Apply embedding and positional encoding
            seq_input = self.seq_embedding(seq_input)
            seq_input = self.seq_pos_encoder(seq_input)
            
            # Apply transformer blocks sequentially
            seq_features = seq_input
            for block in self.seq_features:
                seq_features = block(seq_features)
            
            seq_features = seq_features[:, -1, :]  # Use last time step
            
        elif self.seq_model_type == 'wide_deep_transformer':
            # For WideDeepTransformer
            # Get the expected embedding dimension
            if hasattr(self.seq_model, 'd_model'):
                d_model = self.seq_model.d_model
            else:
                d_model = 64  # Default from config
            
            # Ensure embedding and positional encoding layers exist
            self._ensure_embedding_layer(seq_input.size(-1), d_model)
            self._ensure_positional_encoding(d_model)
            
            # Apply embedding and positional encoding
            seq_input = self.seq_embedding(seq_input)
            seq_input = self.seq_pos_encoder(seq_input)
            
            # Process through wide and deep paths
            wide_output = self.wide_transformer(seq_input)
            deep_output = self.deep_transformer(seq_input)
            
            # Extract last time step from both paths
            wide_last = wide_output[:, -1, :]
            deep_last = deep_output[:, -1, :]
            
            # Concatenate and use the fusion layer from the original model
            seq_features = self.seq_model.fusion(torch.cat([wide_last, deep_last], dim=1))
        
        # Combine features
        combined_features = torch.cat([cnn_features, seq_features], dim=1)
        
        # Classification
        output = self.combined_classifier(combined_features)
        
        return output


# Legacy class for backward compatibility
class HybridCNNBiLSTM(HybridModel):
    """Legacy hybrid model combining CNN and BiLSTM for backward compatibility.
    
    This class is maintained for backward compatibility with existing code.
    It is a special case of the more general HybridModel.
    
    Args:
        cnn_model (nn.Module): CNN model from cnn_models.py
        bilstm_model (nn.Module): BiLSTM model from rnn_models.py
        num_classes (int): Number of output classes
    """
    def __init__(self, cnn_model, bilstm_model, num_classes=8):
        super(HybridCNNBiLSTM, self).__init__(cnn_model, bilstm_model, num_classes, fusion_hidden_size=128)
        print("Note: Using legacy HybridCNNBiLSTM class. Consider using the more general HybridModel class.")


if __name__ == '__main__':
    # To run this test, ensure cnn_models.py and rnn_models.py are in the same directory
    # or the 'models' directory is in PYTHONPATH.
    try:
        from cnn_models import SeismicCNN3D, SeismicResNet3D, SeismicAttentionUNet3D, SeismicPatchNet3D
        from rnn_models import SeismicBiLSTM, SeismicLSTM, SeismicTransformer, WideDeepTransformer, SeismicCustomTransformer
    except ImportError:
        print("Could not import model classes. Make sure they are in the same directory or models package is in PYTHONPATH.")
        # Define minimal dummy classes for testing
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                
        class SeismicCNN3D(DummyModel):
            def __init__(self, num_classes=8):
                super().__init__()
                self.features = nn.Sequential(nn.Conv3d(1,16,3,padding=1), nn.MaxPool3d(2))
                self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16*16*16*16, 64), nn.Linear(64, num_classes))
                self.flattened_size = 16*16*16*16
                
        class SeismicResNet3D(DummyModel):
            def __init__(self, num_classes=8):
                super().__init__()
                self.features = nn.Sequential(nn.Conv3d(1,16,3,padding=1), nn.MaxPool3d(2))
                self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16*16*16*16, 64), nn.Linear(64, num_classes))
                self.flattened_size = 16*16*16*16
                
        class SeismicAttentionUNet3D(DummyModel):
            def __init__(self, num_classes=8):
                super().__init__()
                self.features = nn.Sequential(nn.Conv3d(1,16,3,padding=1), nn.MaxPool3d(2))
                self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16*16*16*16, 64), nn.Linear(64, num_classes))
                self.flattened_size = 16*16*16*16
                
        class SeismicPatchNet3D(DummyModel):
            def __init__(self, num_classes=8):
                super().__init__()
                self.features = nn.Sequential(nn.Conv3d(1,16,3,padding=1), nn.MaxPool3d(2))
                self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16*16*16*16, 64), nn.Linear(64, num_classes))
                self.flattened_size = 16*16*16*16
                
        class SeismicBiLSTM(DummyModel):
            def __init__(self, num_classes=8):
                super().__init__()
                self.lstm = nn.LSTM(1, 64, 2, batch_first=True, bidirectional=True)
                self.classifier = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, num_classes))
                
        class SeismicLSTM(DummyModel):
            def __init__(self, num_classes=8):
                super().__init__()
                self.lstm = nn.LSTM(1, 128, 2, batch_first=True, bidirectional=False)
                self.classifier = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, num_classes))
                
        class SeismicTransformer(DummyModel):
            def __init__(self, num_classes=8):
                super().__init__()
                self.d_model = 64
                self.embedding = nn.Linear(1, self.d_model)
                self.pos_encoder = nn.Identity()
                encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.classifier = nn.Sequential(nn.Linear(self.d_model, 64), nn.Linear(64, num_classes))
                
        class WideDeepTransformer(DummyModel):
            def __init__(self, num_classes=8):
                super().__init__()
                self.d_model = 64
                self.embedding = nn.Linear(1, self.d_model)
                self.pos_encoder = nn.Identity()
                wide_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
                deep_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
                self.wide_transformer = nn.TransformerEncoder(wide_layer, num_layers=1)
                self.deep_transformer = nn.TransformerEncoder(deep_layer, num_layers=4)
                self.fusion = nn.Linear(self.d_model * 2, self.d_model)
                self.classifier = nn.Sequential(nn.Linear(self.d_model, 64), nn.Linear(64, num_classes))
                
        class SeismicCustomTransformer(DummyModel):
            def __init__(self, num_classes=8):
                super().__init__()
                self.embed_dim = 64
                self.embedding = nn.Linear(1, self.embed_dim)
                self.pos_encoder = nn.Identity()
                self.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(3)])
                self.classifier = nn.Sequential(nn.Linear(self.embed_dim, 64), nn.Linear(64, num_classes))
    
    # Test with different model combinations
    batch_size = 2
    patch_size = 32
    seq_len = 64
    
    # Create dummy inputs
    dummy_patches = torch.randn(batch_size, 1, patch_size, patch_size, patch_size)
    dummy_traces = torch.randn(batch_size, seq_len)
    
    # Test CNN + BiLSTM
    cnn_model = SeismicCNN3D()
    bilstm_model = SeismicBiLSTM()
    hybrid_model = HybridModel(cnn_model, bilstm_model)
    output = hybrid_model(dummy_patches, dummy_traces)
    print(f"CNN + BiLSTM output shape: {output.shape}")
    
    # Test ResNet + Transformer
    resnet_model = SeismicResNet3D()
    transformer_model = SeismicTransformer()
    hybrid_model = HybridModel(resnet_model, transformer_model)
    output = hybrid_model(dummy_patches, dummy_traces)
    print(f"ResNet + Transformer output shape: {output.shape}")
    
    # Test UNet + WideDeepTransformer
    unet_model = SeismicAttentionUNet3D()
    widedeep_model = WideDeepTransformer()
    hybrid_model = HybridModel(unet_model, widedeep_model)
    output = hybrid_model(dummy_patches, dummy_traces)
    print(f"UNet + WideDeepTransformer output shape: {output.shape}")
    
    # Test PatchNet + CustomTransformer
    patchnet_model = SeismicPatchNet3D()
    custom_transformer_model = SeismicCustomTransformer()
    hybrid_model = HybridModel(patchnet_model, custom_transformer_model)
    output = hybrid_model(dummy_patches, dummy_traces)
    print(f"PatchNet + CustomTransformer output shape: {output.shape}")
