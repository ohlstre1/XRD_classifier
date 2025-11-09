#!/usr/bin/env python3
"""
1D ResNet Architecture for XRD Signal Classification
===================================================

ResNet-18 adapted for 1D XRD patterns with prototypical learning.
Takes input shape (batch, 1, 4500) and outputs normalized embeddings (batch, embedding_dim).

Key adaptations:
- 1D convolutions instead of 2D
- Appropriate kernel sizes and strides for 1D signals
- Global average pooling for sequence-to-vector transformation
- L2 normalization for embeddings on unit hypersphere
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class BasicBlock1D(nn.Module):
    """
    Basic ResNet block adapted for 1D signals.

    Architecture:
    - Conv1D -> BatchNorm1D -> ReLU -> Conv1D -> BatchNorm1D
    - Skip connection with optional downsampling
    - Final ReLU activation
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        """
        Initialize BasicBlock1D.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution
            downsample: Optional downsampling layer for skip connection
        """
        super(BasicBlock1D, self).__init__()

        # First convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through basic block.

        Args:
            x: Input tensor [batch, channels, length]

        Returns:
            Output tensor [batch, channels, length]
        """
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    ResNet-18 adapted for 1D XRD signals.

    Architecture progression:
    Input: [batch, 1, 4500]
    Conv1: [batch, 64, 2250] (stride=2)
    MaxPool: [batch, 64, 1125] (stride=2)
    Layer1: [batch, 64, 1125] (2 basic blocks)
    Layer2: [batch, 128, 563] (2 basic blocks, stride=2)
    Layer3: [batch, 256, 282] (2 basic blocks, stride=2)
    Layer4: [batch, 512, 141] (2 basic blocks, stride=2)
    AvgPool: [batch, 512, 1] (global average pooling)
    FC: [batch, embedding_dim]
    L2 Norm: [batch, embedding_dim] (unit sphere)
    """

    def __init__(self, block: nn.Module = BasicBlock1D, layers: List[int] = [2, 2, 2, 2],
                 in_channels: int = 1, embedding_dim: int = 256):
        """
        Initialize ResNet1D.

        Args:
            block: Basic block type (BasicBlock1D)
            layers: Number of blocks per layer [layer1, layer2, layer3, layer4]
            in_channels: Number of input channels (1 for XRD)
            embedding_dim: Output embedding dimension
        """
        super(ResNet1D, self).__init__()

        self.in_channels = 64
        self.embedding_dim = embedding_dim

        # Initial convolution: reduce length by half
        # Input: [batch, 1, 4500] -> Output: [batch, 64, 2250]
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        # Max pooling: reduce length by half again
        # Input: [batch, 64, 2250] -> Output: [batch, 64, 1125]
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])        # [batch, 64, 1125]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # [batch, 128, 563]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # [batch, 256, 282]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # [batch, 512, 141]

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # [batch, 512, 1]

        # Embedding head
        self.fc = nn.Linear(512 * block.expansion, embedding_dim)
        self.bn_fc = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(0.1)  # Light regularization

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block: nn.Module, out_channels: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """
        Create a ResNet layer with multiple blocks.

        Args:
            block: Block type (BasicBlock1D)
            out_channels: Number of output channels
            blocks: Number of blocks in this layer
            stride: Stride for first block

        Returns:
            Sequential layer
        """
        downsample = None

        # Create downsampling layer if needed
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        # First block (may have stride > 1)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # Remaining blocks (stride = 1)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet1D.

        Args:
            x: Input XRD patterns [batch, 1, 4500]

        Returns:
            Normalized embeddings [batch, embedding_dim]
        """
        # Input validation
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"Expected input shape [batch, 1, length], got {x.shape}")

        # Initial convolution and pooling
        x = self.conv1(x)       # [batch, 64, 2250]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [batch, 64, 1125]

        # ResNet layers
        x = self.layer1(x)      # [batch, 64, 1125]
        x = self.layer2(x)      # [batch, 128, 563]
        x = self.layer3(x)      # [batch, 256, 282]
        x = self.layer4(x)      # [batch, 512, 141]

        # Global average pooling
        x = self.avgpool(x)     # [batch, 512, 1]
        x = torch.flatten(x, 1) # [batch, 512]

        # Embedding head
        x = self.dropout(x)
        x = self.fc(x)          # [batch, embedding_dim]
        x = self.bn_fc(x)

        # L2 normalization - critical for prototypical learning
        x = F.normalize(x, p=2, dim=1)  # [batch, embedding_dim] on unit sphere

        return x

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Extract feature maps from different layers for analysis.

        Args:
            x: Input XRD patterns [batch, 1, 4500]

        Returns:
            Dictionary of feature maps from different layers
        """
        features = {}

        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['conv1'] = x.clone()

        x = self.maxpool(x)
        features['maxpool'] = x.clone()

        # ResNet layers
        x = self.layer1(x)
        features['layer1'] = x.clone()

        x = self.layer2(x)
        features['layer2'] = x.clone()

        x = self.layer3(x)
        features['layer3'] = x.clone()

        x = self.layer4(x)
        features['layer4'] = x.clone()

        # Final embedding
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_fc(x)
        x = F.normalize(x, p=2, dim=1)
        features['embedding'] = x.clone()

        return features


def create_resnet1d_18(embedding_dim: int = 256, **kwargs) -> ResNet1D:
    """
    Create ResNet1D-18 model.

    Args:
        embedding_dim: Output embedding dimension
        **kwargs: Additional arguments

    Returns:
        ResNet1D model
    """
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], embedding_dim=embedding_dim, **kwargs)


def create_resnet1d_34(embedding_dim: int = 256, **kwargs) -> ResNet1D:
    """
    Create ResNet1D-34 model.

    Args:
        embedding_dim: Output embedding dimension
        **kwargs: Additional arguments

    Returns:
        ResNet1D model
    """
    return ResNet1D(BasicBlock1D, [3, 4, 6, 3], embedding_dim=embedding_dim, **kwargs)


def test_resnet1d():
    """Test function for ResNet1D architecture."""
    print("Testing ResNet1D architecture...")

    # Create model
    model = create_resnet1d_18(embedding_dim=256)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test with random input
    batch_size = 4
    input_length = 4500
    x = torch.randn(batch_size, 1, input_length)

    print(f"Input shape: {x.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        embeddings = model(x)
        print(f"Output embeddings shape: {embeddings.shape}")
        print(f"Embedding norms: {torch.norm(embeddings, p=2, dim=1)}")

        # Test feature extraction
        features = model.get_feature_maps(x)
        print("\nFeature map shapes:")
        for name, feature in features.items():
            print(f"  {name}: {feature.shape}")

    print("✅ ResNet1D test passed!")


if __name__ == "__main__":
    test_resnet1d()