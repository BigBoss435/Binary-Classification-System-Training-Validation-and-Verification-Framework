"""
Dataset creation and preprocessing module for melanoma classification.

This module handles the complete data pipeline for the ISIC 2020 melanoma detection dataset,
including data loading, validation, augmentation, and creation of balanced data loaders.
The pipeline is designed specifically for medical image classification with severe class imbalance.
"""

import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from PIL import Image
import logging
import json
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data_validation import DataValidationPipeline, generate_validation_summary
from config import USE_ALBUMENTATIONS

class MelanomaDataset(Dataset):
    """
    Custom PyTorch Dataset for melanoma classification with support for both 
    PyTorch and Albumentations transforms.
    
    Handles loading images from file paths and applying transformations.

    This dataset class is responsible for:
    - Loading dermoscopic images from disk using PIL/OpenCV
    - Converting images to RGB format (handles grayscale/RGBA edge cases)
    - Applying data transformations (Albumentations or PyTorch transforms)
    - Converting labels to proper tensor format for BCEWithLogitsLoss

    The dataset follows PyTorch's Dataset interface, making it compatible with
    DataLoader for efficient loading and multiprocessing.
    """
    def __init__(self, dataframe, transform=None, use_albumentations=True):
        """
        Initialize dataset with dataframe containing file paths and labels.

        Args:
            dataframe: (pd.DataFrame): DataFrame with 'filepath' and 'target' columns
                                    - 'filepath': Full path to image file
                                    - 'target': Binary label (0=benign, 1=melanoma)
            transform: Optional transforms to apply (Albumentations or PyTorch)
            use_albumentations (bool): Whether to use Albumentations (True) or PyTorch transforms (False)
        """
        self.df = dataframe
        self.transform = transform
        self.use_albumentations = use_albumentations

    def __len__(self):
        """
        Return total number of samples in dataset.

        Required by PyTorch Dataset interface. Used by DataLoader to determine
        batch count and for iteration control.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Load and return a single sample (image, label) pair.
        
        This method is called by PyTorch DataLoader for each sample during training/inference.
        It performs the core data loading pipeline:
        1. Load image from disk
        2. Convert to RGB format (ensure 3-channel consistency)
        3. Apply transformations (Albumentations or PyTorch)
        4. Convert label to float tensor for BCE loss compatibility

        Args:
            idx (int): Index of the sample to retrieve (0 to len(dataset)-1)

        Returns:
            tuple: (transformed_image, label)
                    - transformed_image: torch.Tensor of shape (3, 224, 224)
                    - label: torch.Tensor scalar (0.0 or 1.0)

        Raises:
            FileNotFoundError: If image file doesn't exist
            PIL.UnidentifiedImageError: If image file is corrupted/unreadable
        """
        row = self.df.iloc[idx]

        if self.use_albumentations:
            # Load image with OpenCV for Albumentations (expects numpy array)
            image = cv2.imread(row['filepath'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Apply Albumentations transforms
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
        else:
            # Load image with PIL for PyTorch transforms
            image = Image.open(row['filepath']).convert("RGB")
            
            # Apply PyTorch transforms
            if self.transform:
                image = self.transform(image)

        # Convert label to float tensor with BCEWithLogitsLoss compatibility
        # BCEWithLogitsLoss expects float targets, not long/int
        label = torch.tensor(row['target']).float()

        return image, label

def create_albumentations_transforms(image_size=224):
    """
    Create comprehensive Albumentations-based data augmentation and normalization transforms.
    
    Albumentations provides more advanced augmentation techniques specifically beneficial
    for medical image classification:
    
    Key advantages over PyTorch transforms:
    - More diverse and realistic augmentations
    - Better handling of medical image characteristics
    - More efficient implementation
    - Advanced augmentations like CLAHE, optical distortion, grid distortion
    - Proper coordinate transformations for bounding boxes (if needed later)
    
    Training Transforms:
    - Medical-specific augmentations that preserve anatomical structure
    - Advanced lighting and contrast adjustments
    - Realistic geometric distortions
    - Noise and blur variations
    
    Validation/Test Transforms:
    - Minimal processing for consistent evaluation
    - Only normalization to match training distribution
    
    Args:
        image_size (int): Target image size (default=224 for ResNet, use 300 for EfficientNet-B3)
    
    Returns:
        tuple: (train_transform, val_transform, tta_transform)
    """
    
    # Training transformations with advanced medical-focused augmentation
    train_transform = A.Compose([
        # Resize and geometric augmentations
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), interpolation=1, p=1.0),
        
        # Basic geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
        
        # Advanced geometric distortions - important for medical images
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
        A.ElasticTransform(alpha=1, sigma=20, alpha_affine=20, p=0.2),
        
        # Affine transformations
        A.ShiftScaleRotate(
            shift_limit=0.2, 
            scale_limit=0.3, 
            rotate_limit=15, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0, 
            p=0.5
        ),
        
        # Advanced lighting and contrast - crucial for dermoscopic images
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),  # Contrast Limited Adaptive Histogram Equalization
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        ], p=0.8),
        
        # Color augmentations - simulate different camera/lighting conditions
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
            A.ChannelShuffle(p=1.0),
        ], p=0.5),
        
        # Noise and blur - simulate acquisition variations
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=1.0),
        ], p=0.3),
        
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # Advanced augmentations for robustness
        A.CoarseDropout(
            max_holes=8, 
            max_height=32, 
            max_width=32, 
            min_holes=1, 
            min_height=8, 
            min_width=8, 
            fill_value=0, 
            p=0.3
        ),
        
        # Medical image specific: simulate skin texture variations
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1), 
            num_shadows_lower=1, 
            num_shadows_upper=2, 
            shadow_dimension=5, 
            p=0.2
        ),
        
        # Normalize to ImageNet statistics (required for pretrained models)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        
        # Convert to PyTorch tensor
        ToTensorV2()
    ])
    
    # Validation/test transformations - deterministic and minimal
    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()
    ])
    
    # Test Time Augmentation (TTA) transforms
    tta_transform = A.Compose([
        A.Resize(height=image_size, width=image_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()
    ])
    
    return train_transform, val_transform, tta_transform

def create_transforms():
    """
    Create comprehensive data augmentation and normalization transforms.

    The transforms are carefully designed for dermoscopic image analysis:

    Training Transforms:
    - Aggressive augmentation to improve generalization and reduce overfitting
    - Simulates real-world variations in dermoscopic image acquisition
    - Maintains anatomical plausibility of skin lesions

    Validation Transforms:
    - Minimal processing for consistent evaluation
    - Only normalization to match training distribution

    TTA Transforms:
    - Test Time Augmentation for improved inference robustness
    - Light augmentation to create ensemble of predictions
    
    Returns:
        tuple: (train_transform, val_transform, tta_transform)
    """
    # Training transformations with aggressive augmentation
    # Medical images benefit from extensive augmentation due to:
    # 1. Limited dataset size compared to natural images
    # 2. High intra-class variability (lesions vary greatly)
    # 3. Need for robustness to acquisition conditions
    train_transform = transforms.Compose([
        # Resize with random cropping - simulates different zoom levels
        # Scale (0.7, 1.0) ensures we don't crop out important lesion features
        transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),

        # Geometric augmentation - skin lesions have no preferred orientation
        transforms.RandomHorizontalFlip(p=0.5),  # Mirror simetry is valid for skin
        transforms.RandomVerticalFlip(p=0.5),  # Vertical flip is also valid

        # Affine transformations simulate camera angle variations
        # degrees=45: Rotation up to ±45° (lesions can appear at any angle)
        # translate=(0.2, 0.2): Up to 20% translation (off-center imaging)
        # scale=(0.7, 1.3): ±30% scaling (different distances from skin)
        # shear=10: Up to 10° shearing (non-perpendicular camera angle)
        transforms.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.7, 1.3), shear=10),

        # Additional rotation for full 360° coverage
        transforms.RandomRotation(90),

        # Color augmentation to simulate different lighting/camera conditions
        # Medical images often have lighting variations between acquisitions
        # brightness=0.4: ±40% brightness change (lighting conditions)
        # contrast=0.4: ±40% contrast change (camera settings)
        # saturation=0.4: ±40% saturation change (white balance variations)
        # hue=0.2: ±20% hue shift (color temperature differences)
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),

        # Gaussian blur with 30% probability
        # Simulates slight focus issues or compression artifacts
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),

        # Elastic deformation with 20% probability
        # Simulates skin stretching/compression during image acquisition
        # alpha=50.0: Displacement magnitude
        # sigma=5.0: Smoothness of deformation
        transforms.RandomApply([
            transforms.ElasticTransform(alpha=50.0, sigma=5.0)
        ], p=0.2),

        # Convert PIL Image to tensor with values in [0,1] range
        transforms.ToTensor(),

        # ImageNet normalization - CRITICAL for pretrained models
        # These specific values are required for models pretrained on ImageNet
        # Mean: [0.485, 0.456, 0.406] for R,G,B channels
        # Std:  [0.229, 0.224, 0.225] for R,G,B channels
        # Formula: normalized = (pixel - mean) / std
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),

        # Add small amount of Gaussian noise for additional robustness
        # Helps model handle noisy inputs and improves generalization
        # Scale=0.01 keeps noise subtle to avoid corrupting features
        transforms.Lambda(add_gaussian_noise),
    ])

    # Validation/test transformations - deterministic and minimal
    # No augmentation to ensure fair, reproducible evaluation
    # Only necessary preprocessing: resize + normalize
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406],  # Same normalization as training
                             [0.229, 0.224, 0.225])
    ])

    # Test Time Augmentation (TTA) transforms for improved inference accuracy
    # Similar to training augmentation but applied during inference
    # Multiple augmented versions are averaged for more robust predictions
    tta_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% probability augmentations
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),  # Smaller rotation for TTA stability
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform, tta_transform

def add_gaussian_noise(x):
    """Add small Gaussian noise to the image tensor."""
    return x + torch.randn_like(x) * 0.01

def validate_image_file(filepath):
    """
    Enhanced image validation that checks file integrity and properties.

    Performs comprehensive validation beyond just file existence:
    1. File existance check
    2. Image format verification
    3. Image integrity verification (can be opened/decoded)
    4. Minimum size requirements
    5. Format compatibility check

    Args:
        filepath (str): Path to image file to validate

    Returns:
        bool: True if image passes all validation checks, False otherwise

    Note:
        Uses PIL's verify() method which checks image integrity without
        fully loading the image into memory, making it efficient for
        large-scale validation.
    """
    try:
        # Basic file existence check
        if not os.path.exists(filepath):
            return False
        # First pass: verify image integrity
        # verify() checks if the image file is valid without loading full image
        with Image.open(filepath) as img:
            img.verify()

        # Second pass: check image properties
        # Need to re-open after verify() as it closes the file
        with Image.open(filepath) as img:
            width, height = img.size

            # Reject images that are too small (likely corrupted or thumbnails)
            # 50x50 is minimum reasonable size for medical image analysis
            if width < 50 or height < 50:  # Too small
                return False
            
            # Ensure image format in JPEG (ISIC standard)
            # This also helps filter out non-image files
            if img.format not in ['JPEG', 'JPG']:  # Wrong format
                return False
            
        return True
    except Exception:
        # Any exception during validation indicates corrupted/invalid file
        # This catches PIL.UnidentifiedImageError, OSError, etc.
        return False

def load_and_validate_data(csv_path, images_dir, logger, run_validation=True, run_eda=True):
    """
    Load ISIC 2020 dataset and perform comprehensive validation and cleaning.
    
    This function is the entry point for data loading and handles the complete
    data validation pipeline including:
    1. Loading CSV metadata with image names and labels
    2. Constructing full file paths to images
    3. Running comprehensive validation checks
    4. Basic data cleaning and deduplication
    5. Image file integrity validation
    6. Generating validation reports
    7. Comprehensive Exploratory Data Analysis (optional)
    
    The validation process is critical for medical datasets where data quality
    directly impacts model reliability and patient safety.
    
    Args:
        csv_path (str): Path to ISIC 2020 ground truth CSV file
        images_dir (str): Directory containing JPEG image files
        logger (logging.Logger): Logger for progress monitoring
        run_validation (bool): Whether to run validation pipeline
        run_eda (bool): Whether to run comprehensive EDA analysis
        
    Returns:
        pd.DataFrame: Cleaned and validated dataset with 'filepath' and 'target' columns
        
    Raises:
        ValueError: If required CSV columns are missing
        FileNotFoundError: If CSV file or data directory doesn't exist
    """
    logger.info("Loading and validating dataset...")
    
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the ISIC 2020 ground truth CSV containing image names with directory structure
    # target=1 indicates melanoma, target=0 indicates benign lesion
    df = pd.read_csv(csv_path)
    
    # Create full file paths by combining image names with directory structure
    # Assumes images are stored as JPEG files in the images directory
    df['filepath'] = df['image_name'].apply(lambda x: f"{images_dir}/{x}.jpg")
    
    # Run comprehensive validation pipeline if requested
    if run_validation:
        logger.info("Starting enhanced data validation pipeline...")
        
        # Initialize validation pipeline
        validator = DataValidationPipeline(logger)
        
        # Execute comprehensive validation checks
        # Returns: (validation_passed: bool, validation_results: dict)
        # validation_passed indicates if critical issues were found
        # validation_results contains detailed analysis and recommendations
        validation_passed, validation_results = validator.run_comprehensive_validation(df)
        
        # Save validation report
        validation_report_path = os.path.join(results_dir, "data_validation_report.json")
        with open(validation_report_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {validation_report_path}")
        
        # Generate human-readable summary
        generate_validation_summary(validation_results, results_dir, logger)
        
        if not validation_passed:
            logger.warning("Data validation failed, but continuing with preprocessing...")
            logger.warning("Check validation report for details.")
        else:
            logger.info("All data validation checks passed!")
    
    # Run comprehensive EDA analysis if requested and validation passed or was skipped
    if run_eda:
        logger.info("Starting comprehensive EDA analysis...")
        try:
            # Create EDA subdirectory
            eda_output_dir = os.path.join(results_dir, "eda_analysis")
            os.makedirs(eda_output_dir, exist_ok=True)
            
            # Initialize and run EDA
            from exploratory_data_analysis import MelanomaEDA
            eda_analyzer = MelanomaEDA(logger)
            eda_results = eda_analyzer.run_comprehensive_eda(df, eda_output_dir)
            
            logger.info("EDA analysis completed successfully")
            logger.info(f"EDA reports saved to: {eda_output_dir}")
            
        except Exception as e:
            logger.error(f"EDA analysis failed: {e}")
            logger.info("Continuing with data loading without EDA...")
    
    # Continue with existing basic validation and cleanup
    logger.info("Performing data cleanup based on validation results...")
    
    # Ensure CSV has required columns for training
    required_cols = {"image_name", "target"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    # Remove duplicate image entries (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset="image_name", keep="first")
    logger.info(f"Dropped {before - len(df)} duplicate rows based on image_name.")
    
    # Validate target labels are binary (0 or 1) and not NaN
    bad_targets = ~df["target"].isin([0,1]) | df["target"].isna()
    if bad_targets.any():
        logger.warning(f"Removing {bad_targets.sum()} rows with invalid/NaN targets.")
        df = df[~bad_targets]
    
    # Enhanced image file validation
    logger.info("Validating image file integrity...")
    df["valid_image"] = df["filepath"].apply(validate_image_file)
    invalid_count = (~df["valid_image"]).sum()
    if invalid_count > 0:
        logger.warning(f"Removing {invalid_count} corrupted/invalid images")
        df = df[df["valid_image"]].drop(columns=["valid_image"])
    
    # Handle outliers in numerical features
    logger.info("Detecting and handling outliers in numerical features...")
    df = handle_outliers_in_dataset(df, method='cap', logger=logger)
    
    # Display final class distribution
    counts = df["target"].value_counts().sort_index()
    total = len(df)
    melanoma_ratio = counts.get(1, 0) / total
    logger.info(f"Final dataset - Total: {total}, Benign: {counts.get(0,0)}, "
               f"Melanoma: {counts.get(1,0)} ({melanoma_ratio:.2%})")
    
    # Save cleaned dataset
    cleaned_data_path = os.path.join(results_dir, "cleaned_dataset.csv")
    df.to_csv(cleaned_data_path, index=False)
    logger.info(f"Cleaned dataset saved to: {cleaned_data_path}")
    
    return df

def handle_outliers_in_dataset(df: pd.DataFrame, method: str = 'cap', logger=None):
    """
    Handle outliers in numerical features using specified method.
    
    This function applies outlier treatment to numerical columns in the dataset
    to improve model robustness and training stability. Different methods are
    available depending on the use case and data characteristics.
    
    Args:
        df (pd.DataFrame): Input dataset
        method (str): Outlier handling method. Options:
            - 'cap': Cap outliers at percentile boundaries (default)
            - 'remove': Remove outlier samples entirely
            - 'transform': Apply transformation to reduce outlier impact
            - 'flag': Add outlier flags as features
        logger (logging.Logger): Logger for monitoring
        
    Returns:
        pd.DataFrame: Dataset with outliers handled
        
    Outlier Handling Methods:
        1. 'cap': Winsorization - cap extreme values at 5th/95th percentiles
        2. 'remove': Remove samples with outliers (reduces dataset size)
        3. 'transform': Log transformation to reduce skewness
        4. 'flag': Keep outliers but add binary flags for model awareness
    """
    if logger:
        logger.info(f"Handling outliers using method: {method}")
    
    df_processed = df.copy()
    numerical_columns = ['age_approx']
    
    # Only process columns that exist in the dataset
    numerical_columns = [col for col in numerical_columns if col in df_processed.columns]
    
    if not numerical_columns:
        if logger:
            logger.info("No numerical columns found for outlier handling")
        return df_processed
    
    outlier_info = {
        'method': method,
        'columns_processed': numerical_columns,
        'original_shape': df_processed.shape,
        'changes_made': {}
    }
    
    for col in numerical_columns:
        if df_processed[col].isnull().all():
            continue
            
        original_data = df_processed[col].copy()
        clean_data = original_data.dropna()
        
        if len(clean_data) == 0:
            continue
        
        # Detect outliers using IQR method
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
        n_outliers = outlier_mask.sum()
        
        outlier_info['changes_made'][col] = {
            'outliers_detected': n_outliers,
            'outlier_percentage': (n_outliers / len(df_processed)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        if n_outliers == 0:
            if logger:
                logger.info(f"No outliers found in {col}")
            continue
        
        if method == 'cap':
            # Winsorization: cap at 5th and 95th percentiles
            lower_cap = clean_data.quantile(0.05)
            upper_cap = clean_data.quantile(0.95)
            
            df_processed[col] = df_processed[col].clip(lower=lower_cap, upper=upper_cap)
            
            outlier_info['changes_made'][col].update({
                'lower_cap': lower_cap,
                'upper_cap': upper_cap,
                'values_capped': (original_data < lower_cap).sum() + (original_data > upper_cap).sum()
            })
            
            if logger:
                logger.info(f"Capped {outlier_info['changes_made'][col]['values_capped']} outlier values in {col}")
                
        elif method == 'remove':
            # Remove samples with outliers
            df_processed = df_processed[~outlier_mask]
            
            outlier_info['changes_made'][col]['samples_removed'] = n_outliers
            
            if logger:
                logger.info(f"Removed {n_outliers} samples with outliers in {col}")
                
        elif method == 'transform':
            # Apply log transformation if all values are positive
            if (df_processed[col] > 0).all():
                # Add small constant to handle zeros
                df_processed[f'{col}_log'] = np.log1p(df_processed[col])
                outlier_info['changes_made'][col]['transformation'] = 'log1p'
                
                if logger:
                    logger.info(f"Applied log transformation to {col}")
            else:
                # Use square root transformation for mixed sign data
                df_processed[f'{col}_sqrt'] = np.sign(df_processed[col]) * np.sqrt(np.abs(df_processed[col]))
                outlier_info['changes_made'][col]['transformation'] = 'signed_sqrt'
                
                if logger:
                    logger.info(f"Applied signed square root transformation to {col}")
                    
        elif method == 'flag':
            # Add binary outlier flags
            df_processed[f'{col}_outlier'] = outlier_mask.astype(int)
            
            outlier_info['changes_made'][col]['flag_column'] = f'{col}_outlier'
            outlier_info['changes_made'][col]['outliers_flagged'] = outlier_mask.sum()
            
            if logger:
                logger.info(f"Added outlier flag column for {col}: {outlier_mask.sum()} outliers flagged")
    
    outlier_info['final_shape'] = df_processed.shape
    
    if logger:
        logger.info(f"Outlier handling completed. Shape: {outlier_info['original_shape']} -> {outlier_info['final_shape']}")
    
    # Store outlier processing info for later use
    df_processed._outlier_info = outlier_info
    
    return df_processed

def create_train_val_test_split(df, test_size=0.15, val_size=0.1765, random_state=42, logger=None):
    """
    Create stratified train/validation/test splits with proper class balance.

    Medical ML best practices require three distinct datasets:

    1. Training Set (70%): Used for model weight updates and learning
        - Sees this data during backpropagation
        - Used to compute gradients and update parameters

    2. Validation Set (15%): Used for hyperparameter tuning and model selection
        - Never used for weight updates
        - Guides decisions about learning rate, architecture, early stopping
        - Helps prevent overfitting to training data

    3. Test Set (15%): Final evaluation on completely unseen data
        - Never used during training or tuning process
        - Provides unbiased estimate of real-world performance
        - Critical for medical applications where overoptimistic estimates are dangerous

    Stratfication ensures proportional class representation across splits,
    which is crucial for imbalanced datasets like ISIC (2% melanoma, 98% benign).

    Args:
        df (pd.DataFrame): Full dataset with 'target' column
        test_size (float): Proportion for test set (default 0.15 = 15%)
        val_size (float): Proportion of remaining data for validation (default 0.1765 ≈ 15% of total)
        random_state (int): Random seed for reproducible splits
        logger (logging.Logger): Logger for monitoring

    Returns:
        tuple: (train_df, val_df, test_df, train_val_df)
               - train_df: Training data (70% of original)
               - val_df: Validation data (15% of original) 
               - test_df: Test data (15% of original)
               - train_val_df: Combined train+val for some use cases
    """
    if logger:
        logger.info(f"Creating stratified train/val/test split...")
    
    # First split: separate test set (15%) from train+val (85%)
    # stratify=df['target'] ensures proportional melanoma/benign ratio in both splits
    # This is critical because melanoma is only ~2% of the dataset
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['target'], random_state=random_state
    )
    
    # Second split: divide remaining 85% into train (70% total) and val (15% total)
    # val_size=0.1765 means 17.65% of the 85% remaining = 15% of original dataset
    # This gives final proportions: 70% train, 15% val, 15% test
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, stratify=train_val_df["target"], random_state=random_state
    )
    
    if logger:
        logger.info(f"Dataset splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Log class distributions to verify stratification worked correctly
        # All splits should have approximately the same melanoma ratio
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            counts = split_df["target"].value_counts().sort_index()
            melanoma_ratio = counts.get(1, 0) / len(split_df)
            logger.info(f"{name} split - Benign: {counts.get(0,0)}, Melanoma: {counts.get(1,0)} ({melanoma_ratio:.2%})")
    
    return train_df, val_df, test_df, train_val_df

def create_weighted_sampler(train_df, logger=None):
    """
    Create weighted sampler for handling severe class imbalance during training.
    
    The ISIC dataset has extreme class imbalance (~2% melanoma, ~98% benign).
    Without proper handling, the model would learn to predict "benign" for everything
    and achieve ~98% accuracy while missing all melanoma cases - a catastrophic
    failure for medical diagnosis.
    
    WeightedRandomSampler addresses this by:
    1. Calculating inverse frequency weights (rare classes get higher weights)
    2. Sampling with replacement using these weights
    3. Ensuring roughly balanced batches during training
    
    Mathematical approach:
    - Class weight = Total samples / (Number of classes × Class count)
    - Sample weight = Class weight for that sample's class
    - Sampling probability ∝ Sample weight
    
    This ensures the model sees roughly equal numbers of melanoma and benign
    examples during training, forcing it to learn discriminative features for both.
    
    Args:
        train_df (pd.DataFrame): Training dataframe with 'target' column
        logger (logging.Logger): Logger for monitoring
        
    Returns:
        torch.utils.data.WeightedRandomSampler: Configured sampler for training
        
    Example:
        If dataset has 98,000 benign and 2,000 melanoma samples:
        - Benign weight = 100,000 / (2 × 98,000) = 0.51
        - Melanoma weight = 100,000 / (2 × 2,000) = 25.0
        - Melanoma samples are 49× more likely to be selected
    """
    # Count occurrences of each class (0: benign, 1: melanoma)
    class_counts = train_df['target'].value_counts().sort_index().to_numpy()
    
    # Calculate inverse frequency weights
    # Formula: weight = total_samples / (num_classes × class_count)
    # This gives higher weights to minority classes
    total_samples = len(train_df)
    class_weights = total_samples / (2 * class_counts)

    # Apply square root to moderate extreme weights
    # Without this, minority class weight can be 50-100× higher
    # Square root reduces this to 7-10× while maintaining imbalance correction
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = torch.sqrt(class_weights)
    
    # Assign weight to each sample based on its class
    # Creates weight array where sample_weights[i] = class_weights[target[i]]
    sample_weights = class_weights[train_df['target'].values]
    
    # Create sampler that samples with replacement using calculated weights
    # num_samples=len(sample_weights) ensures each epoch has same length
    # replacement=True allows oversampling minority class
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    if logger:
        logger.info(f"Created weighted sampler - Class weights: {class_weights.tolist()}")
        logger.info(f"Class distribution - Benign: {class_counts[0]}, Melanoma: {class_counts[1]}")
        logger.info(f"Weight ratio (Melanoma/Benign): {class_weights[1]/class_weights[0]:.2f}x")
    
    return sampler

def create_data_loaders(train_df, val_df, test_df, train_transform, val_transform, 
                       batch_size=32, num_workers=0, pin_memory=True, use_albumentations=True, logger=None):
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        train_df, val_df, test_df: DataFrames for each split
        train_transform, val_transform: Transforms for training and validation
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        use_albumentations: Whether to use Albumentations transforms
        logger: Logger instance for monitoring
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
    """
    if logger:
        transform_type = "Albumentations" if use_albumentations else "PyTorch"
        logger.info(f"Creating data loaders with {transform_type} transforms and batch_size={batch_size}")
    
    # Create dataset instances
    train_dataset = MelanomaDataset(train_df, transform=train_transform, use_albumentations=use_albumentations)
    val_dataset = MelanomaDataset(val_df, transform=val_transform, use_albumentations=use_albumentations)
    test_dataset = MelanomaDataset(test_df, transform=val_transform, use_albumentations=use_albumentations)
    
    # Create weighted sampler for training to handle class imbalance
    train_sampler = create_weighted_sampler(train_df, logger)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    if logger:
        logger.info(f"Data loaders created successfully")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")  
        logger.info(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def setup_dataset_pipeline(csv_path="data/ISIC_2020_Training_GroundTruth_v2.csv",
                          data_dir="data/jpeg/train",
                          results_dir="results",
                          batch_size=32,
                          test_size=0.15,
                          val_size=0.1765,
                          random_state=42,
                          num_workers=0,
                          pin_memory=True,
                          use_albumentations=None,  # If None, will use config.USE_ALBUMENTATIONS
                          image_size=224,  # Image size: 224 for ResNet, 300 for EfficientNet-B3
                          run_eda=True,  # Whether to run comprehensive EDA
                          logger=None):
    """
    Complete dataset setup pipeline with support for both PyTorch and Albumentations transforms.
    
    This function handles the entire dataset preparation process:
    1. Load and validate data
    2. Create train/val/test splits
    3. Create transforms (Albumentations or PyTorch)
    4. Create data loaders
    
    Args:
        csv_path: Path to CSV file with image labels
        data_dir: Directory containing image files
        results_dir: Directory for saving reports and cleaned data
        batch_size: Batch size for data loaders
        test_size: Proportion for test set
        val_size: Proportion of remaining data for validation
        random_state: Random seed for reproducibility
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        use_albumentations: Whether to use Albumentations (True) or PyTorch transforms (False).
                          If None, uses config.USE_ALBUMENTATIONS
        image_size: Target image size (224 for ResNet-50, 300 for EfficientNet-B3)
        run_eda: Whether to run comprehensive exploratory data analysis
        logger: Logger instance for monitoring
        
    Returns:
        dict: Dictionary containing all dataset components
    """
    # Use config setting if not explicitly provided
    if use_albumentations is None:
        use_albumentations = USE_ALBUMENTATIONS
    
    if logger:
        transform_type = "Albumentations" if use_albumentations else "PyTorch"
        logger.info(f"Starting complete dataset pipeline with {transform_type} transforms...")
        logger.info(f"Transform selection: use_albumentations={use_albumentations} (from config: {USE_ALBUMENTATIONS})")
    
    # Step 1: Load and validate data (with EDA controlled by parameter)
    df = load_and_validate_data(csv_path, data_dir, logger, run_validation=True, run_eda=run_eda)
    
    # Step 2: Create train/val/test splits
    train_df, val_df, test_df, train_val_df = create_train_val_test_split(
        df, test_size, val_size, random_state, logger
    )
    
    # Step 3: Create transforms
    if use_albumentations:
        train_transform, val_transform, tta_transform = create_albumentations_transforms(image_size=image_size)
    else:
        train_transform, val_transform, tta_transform = create_transforms()
    
    # Step 4: Create data loaders
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_data_loaders(
        train_df, val_df, test_df, train_transform, val_transform,
        batch_size, num_workers, pin_memory, use_albumentations, logger
    )
    
    # Package everything for easy access
    dataset_components = {
        'dataframes': {
            'full': df,
            'train': train_df,
            'val': val_df, 
            'test': test_df,
            'train_val': train_val_df
        },
        'datasets': {
            'train': train_ds,
            'val': val_ds,
            'test': test_ds
        },
        'loaders': {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        },
        'transforms': {
            'train': train_transform,
            'val': val_transform,
            'tta': tta_transform
        },
        'info': {
            'batch_size': batch_size,
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'melanoma_ratio': df['target'].sum() / len(df),
            'use_albumentations': use_albumentations
        }
    }
    
    if logger:
        logger.info("Dataset pipeline completed successfully")
        logger.info(f"Total samples: {dataset_components['info']['total_samples']}")
        logger.info(f"Melanoma ratio: {dataset_components['info']['melanoma_ratio']:.2%}")
        logger.info(f"Using {transform_type} transforms")
    
    return dataset_components