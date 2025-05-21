# models/rnn_models.py

import torch
import torch.nn as nn
import math

class SeismicBiLSTM(nn.Module):
    """Bidirectional LSTM for seismic trace classification.
    
    A bidirectional LSTM model that processes seismic traces as sequences
    and classifies them into different facies.
    
    Args:
        num_classes (int): Number of output classes
        input_size (int): Number of input features per time step (typically 1 for seismic amplitude)
        hidden_size (int): Number of features in the hidden state
        num_layers (int): Number of recurrent layers
        dropout_rate (float): Dropout probability for regularization
    """
    def __init__(self, num_classes=8, input_size=1, hidden_size=64, num_layers=2, dropout_rate=0.3):
        super(SeismicBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout_rate, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), # Multiply by 2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # If input is (batch_size, seq_len), unsqueeze to add input_size dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        lstm_out, _ = self.lstm(x)
        # Use the output of the last time step for classification
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        last_hidden_state = lstm_out[:, -1, :] 
        output = self.classifier(last_hidden_state)
        return output


class SeismicLSTM(nn.Module):
    """Unidirectional LSTM for seismic trace classification.
    
    A standard unidirectional LSTM model that processes seismic traces as sequences
    and classifies them into different facies.
    
    Args:
        num_classes (int): Number of output classes
        input_size (int): Number of input features per time step (typically 1 for seismic amplitude)
        hidden_size (int): Number of features in the hidden state
        num_layers (int): Number of recurrent layers
        dropout_rate (float): Dropout probability for regularization
    """
    def __init__(self, num_classes=8, input_size=1, hidden_size=128, num_layers=2, dropout_rate=0.3):
        super(SeismicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate, bidirectional=False)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # If input is (batch_size, seq_len), unsqueeze to add input_size dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        lstm_out, _ = self.lstm(x)
        # Use the output of the last time step for classification
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        last_hidden_state = lstm_out[:, -1, :] 
        output = self.classifier(last_hidden_state)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models.
    
    Adds positional information to input embeddings.
    
    Args:
        d_model (int): Embedding dimension
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SeismicTransformer(nn.Module):
    """Transformer model for seismic trace classification.
    
    A custom transformer model that processes seismic traces as sequences
    and classifies them into different facies.
    
    Args:
        num_classes (int): Number of output classes
        input_size (int): Number of input features per time step (typically 1 for seismic amplitude)
        d_model (int): Embedding dimension
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer encoder layers
        dim_feedforward (int): Dimension of feedforward network
        dropout_rate (float): Dropout probability for regularization
    """
    def __init__(self, num_classes=8, input_size=1, d_model=64, nhead=4, 
                 num_layers=2, dim_feedforward=256, dropout_rate=0.1):
        super(SeismicTransformer, self).__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout_rate)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # If input is (batch_size, seq_len), unsqueeze to add input_size dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        # Embed input to d_model dimension
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Use the output of the last time step for classification
        last_hidden_state = x[:, -1, :]
        output = self.classifier(last_hidden_state)
        
        return output


class WideDeepTransformer(nn.Module):
    """Wide and Deep Transformer model for seismic trace classification.
    
    A transformer model with parallel wide (shallow) and deep paths
    for capturing both local and global patterns in seismic traces.
    
    Args:
        num_classes (int): Number of output classes
        input_size (int): Number of input features per time step (typically 1 for seismic amplitude)
        d_model (int): Embedding dimension
        nhead_wide (int): Number of attention heads in wide path
        nhead_deep (int): Number of attention heads in deep path
        num_layers_wide (int): Number of transformer encoder layers in wide path
        num_layers_deep (int): Number of transformer encoder layers in deep path
        dim_feedforward (int): Dimension of feedforward network
        dropout_rate (float): Dropout probability for regularization
    """
    def __init__(self, num_classes=8, input_size=1, d_model=64, 
                 nhead_wide=8, nhead_deep=4, num_layers_wide=1, num_layers_deep=4, 
                 dim_feedforward=256, dropout_rate=0.1):
        super(WideDeepTransformer, self).__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout_rate)
        
        # Wide path (shallow, more heads)
        wide_encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead_wide, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.wide_transformer = nn.TransformerEncoder(wide_encoder_layers, num_layers=num_layers_wide)
        
        # Deep path (deeper, fewer heads)
        deep_encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead_deep, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.deep_transformer = nn.TransformerEncoder(deep_encoder_layers, num_layers=num_layers_deep)
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # If input is (batch_size, seq_len), unsqueeze to add input_size dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        # Embed input to d_model dimension
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Wide path
        wide_output = self.wide_transformer(x)
        
        # Deep path
        deep_output = self.deep_transformer(x)
        
        # Extract last time step from both paths
        wide_last = wide_output[:, -1, :]
        deep_last = deep_output[:, -1, :]
        
        # Concatenate and fuse
        combined = torch.cat([wide_last, deep_last], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        output = self.classifier(fused)
        
        return output


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project to query, key, value
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network for transformer blocks.
    
    Args:
        embed_dim (int): Embedding dimension
        hidden_dim (int): Hidden dimension
        dropout (float): Dropout probability
    """
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer encoder block.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        hidden_dim (int): Hidden dimension in feed-forward network
        dropout (float): Dropout probability
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x


class SeismicCustomTransformer(nn.Module):
    """Custom transformer implementation for seismic trace classification.
    
    A transformer model with custom implementation of attention mechanisms
    for processing seismic traces.
    
    Args:
        num_classes (int): Number of output classes
        input_size (int): Number of input features per time step (typically 1 for seismic amplitude)
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer blocks
        hidden_dim (int): Hidden dimension in feed-forward network
        dropout_rate (float): Dropout probability for regularization
    """
    def __init__(self, num_classes=8, input_size=1, embed_dim=64, num_heads=4, 
                 num_layers=3, hidden_dim=256, dropout_rate=0.1):
        super(SeismicCustomTransformer, self).__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_size, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout_rate)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # If input is (batch_size, seq_len), unsqueeze to add input_size dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        # Embed input to embed_dim dimension
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Use the output of the last time step for classification
        last_hidden_state = x[:, -1, :]
        output = self.classifier(last_hidden_state)
        
        return output


if __name__ == '__main__':
    # Example usage and testing
    num_classes = 8
    batch_size = 2
    seq_len = 32
    input_size = 1
    
    # Test BiLSTM model
    bilstm_model = SeismicBiLSTM(num_classes=num_classes)
    print("\n--- BiLSTM Model ---")
    print(bilstm_model)
    dummy_trace = torch.randn(batch_size, seq_len, input_size)
    output = bilstm_model(dummy_trace)
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([batch_size, num_classes])
    
    # Test LSTM model
    lstm_model = SeismicLSTM(num_classes=num_classes)
    print("\n--- LSTM Model ---")
    print(lstm_model)
    output = lstm_model(dummy_trace)
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([batch_size, num_classes])
    
    # Test Transformer model
    transformer_model = SeismicTransformer(num_classes=num_classes)
    print("\n--- Transformer Model ---")
    print(transformer_model)
    output = transformer_model(dummy_trace)
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([batch_size, num_classes])
    
    # Test Wide-Deep Transformer model
    wide_deep_model = WideDeepTransformer(num_classes=num_classes)
    print("\n--- Wide-Deep Transformer Model ---")
    print(wide_deep_model)
    output = wide_deep_model(dummy_trace)
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([batch_size, num_classes])
    
    # Test Custom Transformer model
    custom_transformer_model = SeismicCustomTransformer(num_classes=num_classes)
    print("\n--- Custom Transformer Model ---")
    print(custom_transformer_model)
    output = custom_transformer_model(dummy_trace)
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([batch_size, num_classes])
