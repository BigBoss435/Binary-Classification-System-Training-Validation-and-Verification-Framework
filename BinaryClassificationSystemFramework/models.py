"""
Neural network Models for Melanoma Classification

This module contains the model architecture and custom loss funcion
for melanoma classification using deep learning techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b3, EfficientNet_B3_Weights

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance in binary classification.

    Focal Loss is particularly effective for medical datasets where positive cases
    (melanoma) are much rarer than negative cases (benign lesions). It reduces
    the relative loss for well-classified examples and focuses learning on hard
    examples that are difficult to classify.

    Formula: FL(p_t) = -α(1-p_t)^γ * log(p_t)

    Args:
        alpha (float): Weighting factor for rare class (default=0.75)
                    Higher values give more weight to the positive class.
        gamma (float): Focusing parameter (default=2.0)
                    Higher values reduce the loss contribution from easy examples.
        
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        """
        Initialize Focal Loss with specified parameters.

        Args:
            alpha (float): Balancing factor between positive/negative examples
            gamma (float): Modulating factor to focus on hard examples
        """
        super().__init__()
        self.alpha = alpha  # Weight for positive class (melanoma class)
        self.gamma = gamma  # Focus parameter for hard examples

    def forward(self, inputs, targets):
        """
        Compute focal loss for the given inputs and targets.

        Args:
            inputs (torch.Tensor): Raw logits from model (before sigmoid)
                                Shape: (batch_size, 1)
            targets (torch.Tensor): Ground truth binary labels (0 or 1)
                                Shape: (batch_size, 1)

        Returns:
            torch.Tensor: Computed focal loss value (scalar)
        """
        # Compute binary cross entropy loss without reduction
        # This gives us the standard BCE loss for each sample
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute p_t (probability of the true class)
        # pt represents how confident the model is about the correct prediction
        pt = torch.exp(-ce_loss)

        # Apply focal loss formula: α(1-p_t)^γ * BCE_loss
        # (1-pt)^gamma down-weights easy examples (high pt)
        # alpha balances positive/negative examples
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        # Return mean loss across the batch
        return focal_loss.mean()


class Swish(torch.autograd.Function):
    """
    Swish activation function implementation.
    Swish: f(x) = x * sigmoid(x)
    Often outperforms ReLU in deep networks, especially for medical imaging.
    """
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    """Wrapper module for Swish activation."""
    def forward(self, x):
        return Swish.apply(x)


def create_melanoma_model(num_classes=1, pretrained=True):
    """
    Create ResNet50 model for melanoma binary classification.

    Uses transfer learning with a pre-trained ResNet-50 backbone, which is
    particularly effective for medical image analysis due to:
    1. Rich feature representations learned from ImageNet.
    2. Ability to detect low-level features (edges, textures) relevant to skin lesions.
    3. Efficient training with limited medical data.

    Architecture:
    - Backbone: ResNet-50 (pre-trained on ImageNet)
    - Input: 224x224x3 RGB images
    - Output: Single logit for binary classification (melanoma vs benign)

    Args:
        num_classes (int): Number of output classes (default: 1 for binary classification)
        pretrained (bool): Whether to use ImageNet pre-trained weights (default: True)

    Returns:
        torch.nn.Module: Configured ResNet-50 model ready for melanoma classification
    """
    # Load pre-trained ResNet-50 model with ImageNet weights
    # Pre-trained weights provide feature extractors that understand:
    # - Basic visual patterns (edges, corners, textures)
    # - Complex patterns (shapes, objects)
    # - Hierarchical feature representations
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Modify the final classification layer for binary classification
    # Original ResNet-50 FC layer: 2048 -> 1000 (ImageNet classes)
    # Modified FC layer: 2048 -> 1 (melanoma probability)
    num_ftrs = model.fc.in_features  # Get input features to FC layer (2048 for ResNet-50)

    # Replace final layer with binary classification head
    # Single output will be passed through sigmoid for probability
    model.fc = nn.Linear(num_ftrs, 2)
    
    return model


class AdvancedMelanomaModel(nn.Module):
    """
    Advanced EfficientNet-B3 model for melanoma classification with multiple enhancements.
    
    This architecture is designed for the ENHANCED pipeline and includes:
    1. EfficientNet-B3 backbone (superior efficiency and accuracy)
    2. Multi-dropout test-time augmentation for robustness
    3. Optional meta-features integration (clinical data)
    4. Swish activation for better gradient flow
    5. Batch normalization for stable training
    
    Architecture advantages over basic model:
    - ~40% fewer parameters than ResNet-50 (12M vs 25.6M)
    - Better accuracy through compound scaling
    - Improved generalization via test-time dropout
    - Can incorporate clinical metadata (age, sex, lesion location)
    
    Args:
        out_dim (int): Number of output classes (default=2 for binary classification)
        n_meta_features (int): Number of metadata features to incorporate (default=0)
        n_meta_dim (list): Hidden dimensions for metadata processing (default=[512, 128])
        pretrained (bool): Whether to use ImageNet pre-trained weights (default=True)
        dropout_rate (float): Dropout rate for regularization (default=0.5)
        num_dropouts (int): Number of dropout layers for test-time augmentation (default=5)
    """
    
    def __init__(self, out_dim=2, n_meta_features=0, n_meta_dim=[512, 128], 
                 pretrained=True, dropout_rate=0.5, num_dropouts=5):
        super(AdvancedMelanomaModel, self).__init__()
        self.n_meta_features = n_meta_features
        
        # Load pre-trained EfficientNet-B3 backbone
        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
            self.backbone = efficientnet_b3(weights=weights)
        else:
            self.backbone = efficientnet_b3(weights=None)
        
        # Get the number of features from the backbone
        # EfficientNet-B3 classifier is a Sequential with [Dropout, Linear]
        in_ch = self.backbone.classifier[1].in_features  # 1536 for B3
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Multi-dropout layers for test-time augmentation
        # During inference, predictions from all dropout layers are averaged
        # This improves robustness and uncertainty estimation
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(num_dropouts)
        ])
        
        # Optional metadata processing network
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            # Concatenate image features with metadata features
            in_ch += n_meta_dim[1]
        
        # Final classification layer
        self.classifier = nn.Linear(in_ch, out_dim)
    
    def extract_features(self, x):
        """
        Extract features from the backbone network.
        
        Args:
            x (torch.Tensor): Input images (batch_size, 3, 300, 300)
            
        Returns:
            torch.Tensor: Extracted features (batch_size, 1536)
        """
        return self.backbone(x)
    
    def forward(self, x, x_meta=None):
        """
        Forward pass with optional metadata integration and multi-dropout.
        
        Args:
            x (torch.Tensor): Input images (batch_size, 3, 300, 300)
            x_meta (torch.Tensor, optional): Metadata features (batch_size, n_meta_features)
            
        Returns:
            torch.Tensor: Class logits (batch_size, out_dim)
        """
        # Extract image features
        features = self.extract_features(x)
        
        # Integrate metadata if provided
        if self.n_meta_features > 0 and x_meta is not None:
            x_meta = self.meta(x_meta)
            features = torch.cat((features, x_meta), dim=1)
        
        # Multi-dropout test-time augmentation
        # Average predictions across multiple dropout masks
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.classifier(dropout(features))
            else:
                out += self.classifier(dropout(features))
        
        # Average the outputs
        out /= len(self.dropouts)
        
        return out


def create_advanced_melanoma_model(out_dim=2, n_meta_features=0, n_meta_dim=[512, 128],
                                   pretrained=True, dropout_rate=0.5, num_dropouts=5):
    """
    Factory function to create the advanced melanoma classification model.
    
    This function is designed for the ENHANCED pipeline in comparative studies.
    Use create_melanoma_model() for the CONTROL pipeline.
    
    Args:
        out_dim (int): Number of output classes (default=2)
        n_meta_features (int): Number of metadata features (default=0)
        n_meta_dim (list): Hidden layer sizes for metadata network
        pretrained (bool): Use ImageNet pre-trained weights
        dropout_rate (float): Dropout probability
        num_dropouts (int): Number of dropout layers for TTA
        
    Returns:
        AdvancedMelanomaModel: Configured model ready for training
        
    Example:
        # Basic usage (no metadata)
        model = create_advanced_melanoma_model(pretrained=True)
        
        # With clinical metadata (age, sex, lesion_location_x, lesion_location_y)
        model = create_advanced_melanoma_model(
            n_meta_features=4,
            n_meta_dim=[256, 64]
        )
    """
    return AdvancedMelanomaModel(
        out_dim=out_dim,
        n_meta_features=n_meta_features,
        n_meta_dim=n_meta_dim,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        num_dropouts=num_dropouts
    )