# models/cnn_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SeismicCNN3D(nn.Module):
    """Original 3D CNN model for seismic facies classification.
    
    A simple 3D CNN with two convolutional layers followed by max pooling
    and a fully connected classifier head.
    
    Args:
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (typically 1 for seismic amplitude)
        patch_depth (int): Depth dimension of input patch
        patch_height (int): Height dimension of input patch
        patch_width (int): Width dimension of input patch
    """
    def __init__(self, num_classes=8, input_channels=1, patch_depth=32, patch_height=32, patch_width=32):
        super(SeismicCNN3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, patch_depth, patch_height, patch_width)
            flattened_size = self.features(dummy_input).view(1, -1).size(1)
            
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock3D(nn.Module):
    """3D Residual block with bottleneck design.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for the first convolution
        downsample (nn.Module, optional): Downsampling layer for skip connection
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class SeismicResNet3D(nn.Module):
    """3D ResNet architecture for seismic facies classification.
    
    Implements a 3D ResNet with configurable depth, specifically designed for
    volumetric seismic data. Uses residual connections to enable deeper networks
    and better gradient flow.
    
    Args:
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (typically 1 for seismic amplitude)
        patch_depth (int): Depth dimension of input patch
        patch_height (int): Height dimension of input patch
        patch_width (int): Width dimension of input patch
        blocks_per_layer (list): Number of residual blocks in each layer
    """
    def __init__(self, num_classes=8, input_channels=1, patch_depth=32, patch_height=32, 
                 patch_width=32, blocks_per_layer=[2, 2, 2, 2]):
        super(SeismicResNet3D, self).__init__()
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, blocks_per_layer[0], stride=1)
        self.layer2 = self._make_layer(128, blocks_per_layer[1], stride=2)
        self.layer3 = self._make_layer(256, blocks_per_layer[2], stride=2)
        self.layer4 = self._make_layer(512, blocks_per_layer[3], stride=2)
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, patch_depth, patch_height, patch_width)
            x = self.conv1(dummy_input)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            self.flattened_size = x.view(1, -1).size(1)
        
        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
            
        layers = []
        layers.append(ResidualBlock3D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class DoubleConv3D(nn.Module):
    """Double 3D convolution block with batch normalization.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)


class AttentionGate3D(nn.Module):
    """3D Attention Gate for U-Net.
    
    Implements a spatial attention mechanism to focus on relevant features.
    
    Args:
        F_g (int): Number of feature channels from the encoder path
        F_l (int): Number of feature channels from the skip connection
        F_int (int): Number of intermediate channels
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class SeismicAttentionUNet3D(nn.Module):
    """3D U-Net with attention gates for seismic facies classification.
    
    Implements a 3D U-Net architecture with attention gates at skip connections,
    specifically designed for volumetric seismic data. The attention mechanism
    helps focus on relevant features during upsampling.
    
    Args:
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (typically 1 for seismic amplitude)
        patch_depth (int): Depth dimension of input patch
        patch_height (int): Height dimension of input patch
        patch_width (int): Width dimension of input patch
        features (list): Number of features at each level of the U-Net
    """
    def __init__(self, num_classes=8, input_channels=1, patch_depth=32, patch_height=32, 
                 patch_width=32, features=[32, 64, 128, 256]):
        super(SeismicAttentionUNet3D, self).__init__()
        self.encoder1 = DoubleConv3D(input_channels, features[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = DoubleConv3D(features[0], features[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = DoubleConv3D(features[1], features[2])
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = DoubleConv3D(features[2], features[3])
        
        self.upconv3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.attention3 = AttentionGate3D(F_g=features[2], F_l=features[2], F_int=features[2]//2)
        self.decoder3 = DoubleConv3D(features[2]*2, features[2])
        
        self.upconv2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.attention2 = AttentionGate3D(F_g=features[1], F_l=features[1], F_int=features[1]//2)
        self.decoder2 = DoubleConv3D(features[1]*2, features[1])
        
        self.upconv1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.attention1 = AttentionGate3D(F_g=features[0], F_l=features[0], F_int=features[0]//2)
        self.decoder1 = DoubleConv3D(features[0]*2, features[0])
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, patch_depth, patch_height, patch_width)
            e1 = self.encoder1(dummy_input)
            p1 = self.pool1(e1)
            
            e2 = self.encoder2(p1)
            p2 = self.pool2(e2)
            
            e3 = self.encoder3(p2)
            p3 = self.pool3(e3)
            
            b = self.bottleneck(p3)
            
            d3 = self.upconv3(b)
            a3 = self.attention3(d3, e3)
            d3 = torch.cat([d3, a3], dim=1)
            d3 = self.decoder3(d3)
            
            d2 = self.upconv2(d3)
            a2 = self.attention2(d2, e2)
            d2 = torch.cat([d2, a2], dim=1)
            d2 = self.decoder2(d2)
            
            d1 = self.upconv1(d2)
            a1 = self.attention1(d1, e1)
            d1 = torch.cat([d1, a1], dim=1)
            d1 = self.decoder1(d1)
            
            self.flattened_size = d1.view(1, -1).size(1)
        
        # Global average pooling and classifier
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features[0], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder with attention
        d3 = self.upconv3(b)
        a3 = self.attention3(d3, e3)
        d3 = torch.cat([d3, a3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        a2 = self.attention2(d2, e2)
        d2 = torch.cat([d2, a2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        a1 = self.attention1(d1, e1)
        d1 = torch.cat([d1, a1], dim=1)
        d1 = self.decoder1(d1)
        
        # Classification head
        x = self.gap(d1)
        x = self.classifier(x)
        
        return x


class DilatedConv3D(nn.Module):
    """3D Dilated convolution block.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size
        dilation (int): Dilation rate
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(DilatedConv3D, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                             padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MultiScaleFeatureFusion(nn.Module):
    """Multi-scale feature fusion module.
    
    Fuses features from different scales using 1x1x1 convolutions.
    
    Args:
        in_channels (list): List of input channels for each scale
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv3d(in_ch, out_channels, kernel_size=1, bias=False) 
            for in_ch in in_channels
        ])
        self.fusion = nn.Conv3d(out_channels * len(in_channels), out_channels, 
                               kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, xs):
        # Process each input scale
        outs = []
        for i, x in enumerate(xs):
            outs.append(self.convs[i](x))
        
        # Ensure all feature maps have the same spatial dimensions
        target_size = outs[0].size()[2:]
        for i in range(1, len(outs)):
            outs[i] = F.interpolate(outs[i], size=target_size, mode='trilinear', align_corners=False)
        
        # Concatenate and fuse
        x = torch.cat(outs, dim=1)
        x = self.fusion(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x


class SeismicPatchNet3D(nn.Module):
    """Multi-scale 3D CNN with dilated convolutions for seismic facies classification.
    
    Implements a multi-scale feature fusion network with dilated convolutions,
    inspired by SeismicPatchNet (Nature Communications, 2020). This architecture
    is specifically designed for capturing multi-scale features in volumetric
    seismic data.
    
    Args:
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (typically 1 for seismic amplitude)
        patch_depth (int): Depth dimension of input patch
        patch_height (int): Height dimension of input patch
        patch_width (int): Width dimension of input patch
        base_features (int): Number of base features
    """
    def __init__(self, num_classes=8, input_channels=1, patch_depth=32, patch_height=32, 
                 patch_width=32, base_features=32):
        super(SeismicPatchNet3D, self).__init__()
        
        # Initial convolution
        self.conv_in = nn.Sequential(
            nn.Conv3d(input_channels, base_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_features),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale dilated convolutions
        self.dilation1 = DilatedConv3D(base_features, base_features, dilation=1)
        self.dilation2 = DilatedConv3D(base_features, base_features, dilation=2)
        self.dilation4 = DilatedConv3D(base_features, base_features, dilation=4)
        
        # Feature fusion
        self.fusion1 = MultiScaleFeatureFusion(
            [base_features, base_features, base_features], 
            base_features * 2
        )
        
        # Downsampling path
        self.down1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(base_features * 2, base_features * 4)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(base_features * 4, base_features * 8)
        )
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, patch_depth, patch_height, patch_width)
            x = self.conv_in(dummy_input)
            
            # Multi-scale dilated convolutions
            d1 = self.dilation1(x)
            d2 = self.dilation2(x)
            d4 = self.dilation4(x)
            
            # Feature fusion
            x = self.fusion1([d1, d2, d4])
            
            # Downsampling
            x = self.down1(x)
            x = self.down2(x)
            
            # Global average pooling
            x = F.adaptive_avg_pool3d(x, (1, 1, 1))
            self.flattened_size = x.view(1, -1).size(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Initial convolution
        x = self.conv_in(x)
        
        # Multi-scale dilated convolutions
        d1 = self.dilation1(x)
        d2 = self.dilation2(x)
        d4 = self.dilation4(x)
        
        # Feature fusion
        x = self.fusion1([d1, d2, d4])
        
        # Downsampling
        x = self.down1(x)
        x = self.down2(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        
        # Classification
        x = self.classifier(x)
        
        return x


if __name__ == '__main__':
    # Example usage and testing
    num_classes = 8
    batch_size = 2
    patch_d, patch_h, patch_w = 32, 32, 32
    
    # Test original CNN model
    original_model = SeismicCNN3D(num_classes=num_classes)
    print("\n--- Original CNN Model ---")
    print(original_model)
    dummy_patch = torch.randn(batch_size, 1, patch_d, patch_h, patch_w)
    output = original_model(dummy_patch)
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([batch_size, num_classes])
    
    # Test ResNet model
    resnet_model = SeismicResNet3D(num_classes=num_classes)
    print("\n--- 3D ResNet Model ---")
    print(resnet_model)
    output = resnet_model(dummy_patch)
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([batch_size, num_classes])
    
    # Test Attention U-Net model
    attention_unet_model = SeismicAttentionUNet3D(num_classes=num_classes)
    print("\n--- 3D Attention U-Net Model ---")
    print(attention_unet_model)
    output = attention_unet_model(dummy_patch)
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([batch_size, num_classes])
    
    # Test SeismicPatchNet model
    patchnet_model = SeismicPatchNet3D(num_classes=num_classes)
    print("\n--- 3D SeismicPatchNet Model ---")
    print(patchnet_model)
    output = patchnet_model(dummy_patch)
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([batch_size, num_classes])
