# =========================================
# Melanoma Classification Training Script
# =========================================
# This script implements a comprehensive deep learning pipeline for binary melanoma classification
# using transfer learning with EfficientNet-B3 on the ISIC 2020 dataset. The system is designed
# specifically for medical image analysis with robust validation, ensemble methods, and
# clinical-grade evaluation metrics.
#
# Key Features:
# - Advanced EfficientNet-B3 architecture with multi-dropout test-time augmentation
# - Swish activation and compound scaling for superior performance
# - Class imbalance handling through Focal Loss and weighted sampling
# - Cross-validation for robust performance estimation
# - Test-time augmentation for improved inference accuracy
# - Mixed precision training for efficiency
# - Comprehensive evaluation with medical-relevant metrics
#
# Author: Deividas Kalvelis

import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from config import *
from utils import setup_logging, log_system_stats, log_model_info, get_device
from models import create_melanoma_model, create_advanced_melanoma_model, FocalLoss
from training import train_one_epoch, evaluate_with_threshold_search
from evaluation import (
    predict_proba, compute_classification_metrics, find_optimal_threshold,
    plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
)
from ensemble_training import (
    create_ensemble_models, train_ensemble, 
    evaluate_ensemble, ensemble_predict
)
from dataset import setup_dataset_pipeline, MelanomaDataset
from cross_validation import run_cross_validation, evaluate_ensemble_on_test
from fairness_assessment import evaluate_fairness
from adversarial_testing import evaluate_adversarial_robustness
import pandas as pd

def main():
    """
    Main training pipeline for melanoma classification.
    
    This function orchestrates the complete machine learning pipeline for melanoma detection:
    
    1. **Environment Setup**: Device detection, logging configuration, system monitoring
    2. **Data Pipeline**: Dataset loading, validation, preprocessing, and augmentation
    3. **Model Configuration**: Architecture setup, loss function tuning, optimizer selection
    4. **Training Strategy**: Cross-validation vs single split, early stopping, checkpointing
    5. **Evaluation**: Comprehensive metrics, visualization, and clinical interpretation
    
    The pipeline is designed for medical AI applications where:
    - Reproducibility is critical for regulatory approval
    - Performance estimation must be unbiased and robust
    - False negatives (missed melanomas) are more costly than false positives
    - Model interpretability and confidence measures are essential
    """

    # ============================================================================
    # PHASE 1: ENVIRONMENT AND SYSTEM SETUP
    # ============================================================================
    
    # Device configuration with automatic GPU detection and fallback
    # Mixed precision (AMP) significantly speeds up training on modern GPUs
    # while maintaining numerical stability for medical image analysis
    device = get_device()
    print(f"Device: {device}, Use AMP: {USE_AMP}")

    # Initialize comprehensive logging system
    # All training decisions, hyperparameters, and results are logged
    logger = setup_logging(RESULTS_DIR)
    logger.info(f"Starting melanoma classification training on device: {device}")
    logger.info(f"Config - TTA_N: {TTA_N}, USE_AMP: {USE_AMP}")
    
    # Log system specifications for reproducibility
    # Hardware differences can affect training dynamics and final performance
    # This information is crucial for result interpretation and reproduction
    log_system_stats(logger)

    # ============================================================================
    # PHASE 2: DATASET PIPELINE SETUP
    # ============================================================================
    logger.info("Setting up dataset pipeline...")
    
    # Execute comprehensive data loading and validation pipeline
    # This function handles the complete data processing workflow:
    # - CSV loading and validation
    # - Image file integrity checking
    # - Data cleaning and preprocessing
    # - Stratified train/val/test splitting
    # - Augmentation pipeline configuration (300x300 for EfficientNet-B3)
    # - Weighted sampler creation for class imbalance
    # - DataLoader setup with optimal parameters
    dataset_components = setup_dataset_pipeline(
        csv_path=CSV_PATH,                    # ISIC 2020 ground truth CSV
        data_dir=DATA_DIR,                    # Directory containing JPEG images
        results_dir=RESULTS_DIR,              # Output directory for reports
        batch_size=BATCH_SIZE,                # Training batch size
        test_size=TEST_SIZE,                  # Proportion for test set (15%)
        val_size=VAL_SIZE,                    # Proportion for validation (15%)
        random_state=RANDOM_STATE,            # Seed for reproducible splits
        num_workers=4,                        # Single-threaded loading (stable)
        pin_memory=True,                      # Faster GPU transfer
        image_size=300,                       # 300x300 for EfficientNet-B3 (vs 224 for ResNet)
        run_eda=RUN_EDA,                      # Run comprehensive EDA analysis
        logger=logger                         # Logging instance
    )

    # Extract dataset components for clarity and easier access
    # The setup function returns a comprehensive dictionary with all components
    # needed for training, validation, and testing
    train_df = dataset_components['dataframes']['train']           # Training metadata
    val_df = dataset_components['dataframes']['val']               # Validation metadata
    test_df = dataset_components['dataframes']['test']             # Test metadata
    train_val_df = dataset_components['dataframes']['train_val']   # Combined for CV
    
    train_loader = dataset_components['loaders']['train']          # Balanced training loader
    val_loader = dataset_components['loaders']['val']              # Validation loader
    test_loader = dataset_components['loaders']['test']            # Test loader
    
    train_transform = dataset_components['transforms']['train']    # Augmentation pipeline
    val_transform = dataset_components['transforms']['val']        # Normalization only
    tta_transform = dataset_components['transforms']['tta']        # Test-time augmentation

    # Log comprehensive dataset statistics for analysis and debugging
    # This information helps verify data splits and understand class distribution
    logger.info(f"Dataset setup complete:")
    logger.info(f"  Total samples: {dataset_components['info']['total_samples']}")
    logger.info(f"  Train: {dataset_components['info']['train_samples']}")
    logger.info(f"  Val: {dataset_components['info']['val_samples']}")
    logger.info(f"  Test: {dataset_components['info']['test_samples']}")
    logger.info(f"  Melanoma ratio: {dataset_components['info']['melanoma_ratio']:.2%}")
    
    # Console output for immediate feedback during interactive sessions
    print(f"Sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    
    # ============================================================================
    # PHASE 3: MODEL AND OPTIMIZATION SETUP
    # ============================================================================
    
    # Initialize EfficientNet-B3 model with advanced architecture
    # Features: Multi-dropout TTA, Swish activation, compound scaling
    # Superior efficiency: 12M parameters vs 25.6M for ResNet-50
    # Better accuracy through optimal depth/width/resolution scaling
    model = create_advanced_melanoma_model(
        out_dim=2,
        pretrained=True,
        dropout_rate=0.5,
        num_dropouts=5
    ).to(device)
    
    logger.info("Model: Advanced EfficientNet-B3 with multi-dropout TTA")
    logger.info("  - Parameters: ~12M (vs 25.6M for ResNet-50)")
    logger.info("  - Input size: 300x300 (vs 224x224 for ResNet-50)")
    logger.info("  - Dropout layers: 5 (test-time augmentation)")
    logger.info("  - Activation: Swish")
    
    # Configure Focal Loss for severe class imbalance
    # Alpha parameter dynamically calculated from training data distribution
    # Higher alpha gives more weight to the minority class (melanoma)
    # Gamma=3.0 provides strong focus on hard-to-classify examples
    alpha = 1 - train_df['target'].sum() / len(train_df)  # Complement of melanoma ratio
    criterion = FocalLoss(alpha=alpha, gamma=3.0)
    logger.info(f"Focal Loss configured - Alpha: {alpha:.4f}, Gamma: 3.0")

    # Adam optimizer with conservative learning rate for medical data
    # Lower learning rates prevent overfitting on limited medical datasets
    # Weight decay provides L2 regularization for better generalization
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,  # Conservative learning rate (typically 1e-4 to 3e-4 for medical)
        weight_decay=WEIGHT_DECAY  # L2 regularization strength
    )

    # ReduceLROnPlateau scheduler for adaptive learning rate adjustment
    # Dynamically reduces learning rate when validation performance plateaus
    # This helps fine-tune the model when it gets stuck in suboptimal regions
    # and prevents overshooting optimal solutions during late training stages
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # 'max' for AUC (maximize), 'min' for loss
        factor=0.5,           # Reduce LR by half
        patience=5,           # Wait 5 epochs before reducing
        verbose=True,         # Print LR changes
        min_lr=1e-6          # Don't go below this
    )
    
    # Mixed precision training setup for efficiency
    # GradScaler handles gradient scaling to prevent underflow in fp16
    # Significantly speeds up training on modern GPUs while maintaining accuracy
    scaler = torch.amp.GradScaler(device="cuda") if USE_AMP else None
    if USE_AMP:
        logger.info("Mixed precision training enabled with GradScaler")

    # ============================================================================
    # PHASE 4: TRAINING STRATEGY EXECUTION
    # ============================================================================
    
    # Choose training strategy based on configuration
    # Cross-validation provides more robust performance estimates
    # Single split is faster for initial experiments and development
    if USE_CROSS_VALIDATION:
        logger.info(f"Using {N_FOLDS}-fold cross-validation for robust evaluation")
        final_results = run_cross_validation_training(
            train_val_df, train_transform, val_transform, test_loader,
            device, scaler, logger
        )
    else:
        logger.info("Using single train/validation/test split")
        final_results = run_single_split_training(
            model, train_loader, val_loader, test_loader,
            optimizer, scheduler, criterion, scaler,
            device, logger, tta_transform
        )
    
    # ============================================================================
    # PHASE 5: FAIRNESS AND SECURITY ASSESSMENT
    # ============================================================================
    
    logger.info("=" * 60)
    logger.info("PHASE 5: FAIRNESS AND SECURITY ASSESSMENT")
    logger.info("=" * 60)
    
    # Prepare demographic data for fairness analysis if enabled
    if RUN_FAIRNESS_ASSESSMENT:
        logger.info("\nPreparing demographic features for fairness assessment...")
        sensitive_features_df = prepare_sensitive_features(test_df)
        
        if not sensitive_features_df.empty:
            logger.info("Performing fairness assessment...")
            try:
                # Get test predictions from training results
                if 'test_predictions' in final_results:
                    test_probs = np.array(final_results['test_predictions']['test_probs'])
                    test_pred = np.array(final_results['test_predictions']['test_pred'])
                    test_true = np.array(final_results['test_predictions']['test_true'])
                    
                    fairness_results = evaluate_fairness(
                        y_true=test_true,
                        y_pred=test_pred,
                        y_probs=test_probs,
                        sensitive_features_df=sensitive_features_df,
                        results_dir=RESULTS_DIR,
                        logger=logger
                    )
                else:
                    logger.warning("Test predictions not available in training results")
                    fairness_results = {"warning": "Test predictions not available"}
                
            except Exception as e:
                logger.error(f"Fairness assessment failed: {str(e)}")
                fairness_results = {"error": str(e)}
        else:
            logger.warning("No sensitive features available for fairness assessment")
            fairness_results = {"warning": "No sensitive features available"}
        
        final_results['fairness_assessment'] = fairness_results
    
    # Adversarial Robustness Testing if enabled
    if RUN_ADVERSARIAL_TESTING:
        logger.info("\nPerforming adversarial robustness testing...")
        try:
            # Load the best model for adversarial testing
            if USE_CROSS_VALIDATION:
                logger.info("Loading best CV model for adversarial testing...")
                # For CV, use one of the fold models or create a fresh model
                test_model = create_advanced_melanoma_model(
                    out_dim=2,
                    pretrained=True,
                    dropout_rate=0.5,
                    num_dropouts=5
                ).to(device)
                # Load checkpoint if available
            else:
                # Use the trained model from single split
                test_model = model
            
            adversarial_results = evaluate_adversarial_robustness(
                model=test_model,
                test_loader=test_loader,
                device=device,
                results_dir=RESULTS_DIR,
                logger=logger,
                num_samples=ADVERSARIAL_NUM_SAMPLES
            )
            
            final_results['adversarial_robustness'] = adversarial_results
            
        except Exception as e:
            logger.error(f"Adversarial testing failed: {str(e)}")
            final_results['adversarial_robustness'] = {"error": str(e)}
    
    # Calibration and Reliability Assessment if enabled
    if RUN_CALIBRATION:
        logger.info("\n" + "="*60)
        logger.info("CALIBRATION AND RELIABILITY ASSESSMENT")
        logger.info("="*60)
        try:
            from calibration import run_full_calibration_pipeline
            
            # Determine which model to calibrate based on training strategy
            if USE_CROSS_VALIDATION:
                logger.info("Loading best CV model for calibration...")
                calibration_model = create_advanced_melanoma_model(
                    out_dim=2,
                    pretrained=True,
                    dropout_rate=0.5,
                    num_dropouts=5
                ).to(device)
                # Try to load the best checkpoint from CV
                best_checkpoint = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
                if os.path.exists(best_checkpoint):
                    calibration_model.load_state_dict(torch.load(best_checkpoint))
                    logger.info(f"Loaded best model from {best_checkpoint}")
                else:
                    logger.warning("No checkpoint found, using freshly initialized model for calibration")
            else:
                # Use the trained model from single split
                calibration_model = model
                logger.info("Using trained model from single split training")
            
            # Run full calibration pipeline
            calibration_results, calibrated_model = run_full_calibration_pipeline(
                model=calibration_model,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                output_dir=CALIBRATION_DIR,
                logger=logger
            )
            
            # Log summary
            logger.info("\n" + "-"*60)
            logger.info("CALIBRATION SUMMARY")
            logger.info("-"*60)
            pre_ece = calibration_results['pre_calibration']['metrics']['ECE']
            post_ece = calibration_results['post_calibration']['metrics']['ECE']
            improvement = calibration_results['improvement']['ECE_reduction']
            temperature = calibration_results['post_calibration']['temperature']
            
            logger.info(f"Pre-calibration ECE:  {pre_ece:.4f}")
            logger.info(f"Post-calibration ECE: {post_ece:.4f}")
            logger.info(f"ECE improvement:      {improvement:.4f}")
            logger.info(f"Optimal temperature:  {temperature:.4f}")
            
            # Log reliability assessment
            logger.info("\nRELIABILITY METRICS:")
            for threshold, metrics in calibration_results['reliability']['confidence_thresholds'].items():
                logger.info(f"  Confidence >{threshold}: "
                          f"Accuracy={metrics['accuracy']:.4f}, "
                          f"Coverage={metrics['coverage']:.2%}")
            
            # Save calibrated model
            calibrated_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model_calibrated.pth')
            torch.save(calibrated_model.state_dict(), calibrated_checkpoint_path)
            logger.info(f"\nCalibrated model saved to {calibrated_checkpoint_path}")
            
            # Add results to final output
            final_results['calibration'] = {
                'pre_calibration_ECE': pre_ece,
                'post_calibration_ECE': post_ece,
                'ECE_improvement': improvement,
                'temperature': temperature,
                'brier_score': calibration_results['post_calibration']['metrics']['Brier_score'],
                'reliability_metrics': calibration_results['reliability']
            }
            
            logger.info("Calibration completed successfully")
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            final_results['calibration'] = {"error": str(e)}
    
    # ============================================================================
    # PHASE 6: RESULTS CONSOLIDATION AND STORAGE
    # ============================================================================
    
    # Save comprehensive results for analysis and reporting
    # JSON format enables easy parsing for automated analysis and reporting
    # Includes all metrics, configuration, and metadata for full reproducibility
    results_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)  # default=str handles datetime objects
    
    logger.info(f"Final results saved to {results_path}")
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    # Final console output for immediate feedback
    print("All done. Results saved to", RESULTS_DIR)

def run_cross_validation_training(train_val_df, train_transform, val_transform, 
                                 test_loader, device, scaler, logger):
    """
    Execute k-fold cross-validation training pipeline for robust model evaluation.
    
    Cross-validation is the gold standard for medical ML evaluation because:
    
    1. **Unbiased Performance Estimation**: Provides realistic estimates that
       generalize to new patient populations by testing on multiple data splits
       
    2. **Model Stability Assessment**: Reveals how sensitive the model is to
       specific training examples and helps identify overfitting
       
    3. **Ensemble Benefits**: Multiple trained models can be combined for
       improved accuracy and robustness in clinical deployment
       
    4. **Statistical Rigor**: Enables calculation of confidence intervals
       and significance tests for performance metrics
    
    Process Overview:
    - Split data into k stratified folds maintaining class balance
    - Train independent models on k-1 folds, validate on remaining fold
    - Collect performance statistics across all folds
    - Create ensemble from all models for final test evaluation
    - Generate comprehensive reports and visualizations
    
    Args:
        train_val_df (pd.DataFrame): Combined training and validation data
        train_transform (torchvision.transforms): Training augmentation pipeline
        val_transform (torchvision.transforms): Validation normalization pipeline
        test_loader (DataLoader): Independent test set for final evaluation
        device (torch.device): Training device (CPU/GPU)
        scaler: Mixed precision scaler for efficient training
        logger (logging.Logger): Logging instance for detailed tracking
        
    Returns:
        dict: Comprehensive results including CV statistics and ensemble metrics
    """
    
    # Define plotting functions for visualization
    # These functions create standardized plots for consistent evaluation
    # across all folds and final ensemble evaluation
    plot_functions = {
        'roc': plot_roc_curve,                    # ROC curves for AUC visualization
        'pr': plot_precision_recall_curve,        # PR curves for imbalanced data
        'cm': plot_confusion_matrix               # Confusion matrices for error analysis
    }
    
    # Execute comprehensive cross-validation
    # This function handles all aspects of k-fold training:
    # - Stratified fold creation
    # - Independent model training per fold
    # - Performance tracking and early stopping
    # - Detailed logging and checkpointing
    # - Statistical analysis across folds
    cv_stats, cv_models = run_cross_validation(
        train_val_df=train_val_df,                    # Data for CV splits
        train_transform=train_transform,              # Augmentation pipeline
        val_transform=val_transform,                  # Validation preprocessing
        dataset_class=MelanomaDataset,                # PyTorch Dataset class
        train_one_epoch_func=train_one_epoch,         # Single epoch training function
        evaluate_func=evaluate_with_threshold_search, # Validation evaluation function
        predict_proba_func=predict_proba,             # Probability prediction function
        compute_metrics_func=compute_classification_metrics,  # Metrics computation
        find_threshold_func=find_optimal_threshold,   # Threshold optimization
        plot_functions=plot_functions,                # Visualization functions
        device=device,                                # Training device
        scaler=scaler,                                # Mixed precision scaler
        n_folds=N_FOLDS,                             # Number of CV folds
        max_epochs=MAX_EPOCHS                         # Maximum epochs per fold
    )
    
    logger.info("Cross-validation training completed. Evaluating ensemble on test set...")
    
    # Evaluate ensemble of CV models on independent test set
    # Ensemble methods typically outperform individual models by:
    # - Reducing prediction variance through averaging
    # - Capturing different aspects of the data through diverse training
    # - Providing uncertainty estimates through prediction variance
    ensemble_test_metrics, ensemble_test_probs, test_true = evaluate_ensemble_on_test(
        cv_models=cv_models,                          # List of trained CV models
        test_loader=test_loader,                      # Independent test data
        predict_proba_func=predict_proba,             # Prediction function
        compute_metrics_func=compute_classification_metrics,  # Metrics computation
        find_threshold_func=find_optimal_threshold,   # Threshold optimization
        plot_functions=plot_functions,                # Visualization functions
        device=device,                                # Inference device
        tta_n=TTA_N                                  # Test-time augmentation iterations
    )
    
    logger.info("Cross-validation pipeline completed successfully")

    # Return comprehensive results for analysis and reporting
    # Includes both CV statistics and final ensemble performance
    return {
        "training_method": "cross_validation",
        "cross_validation_stats": cv_stats,          # Detailed CV performance statistics
        "ensemble_test_metrics": ensemble_test_metrics,  # Final ensemble test results
        "configuration": {                            # Complete configuration for reproducibility
            "n_folds": N_FOLDS,
            "tta_n": TTA_N,
            "use_amp": USE_AMP,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "device": str(device)
        }
    }

def run_single_split_training(model, train_loader, val_loader, test_loader,
                             optimizer, scheduler, criterion, scaler,
                             device, logger, tta_transform):
    """
    Execute traditional single train/validation/test split training pipeline.
    
    This training approach uses a fixed data split for:
    - Faster experimentation and development
    - Baseline model development
    - Hyperparameter tuning
    - Initial proof-of-concept validation
    
    While less robust than cross-validation, single split training is useful for:
    - Large datasets where CV is computationally prohibitive
    - Initial model development and debugging
    - Hyperparameter optimization studies
    - Quick baseline establishment
    
    The pipeline includes:
    - Complete training loop with early stopping
    - Learning rate scheduling
    - Model checkpointing and recovery
    - Comprehensive evaluation with TTA
    - Detailed logging and monitoring
    
    Args:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): Training data with balanced sampling
        val_loader (DataLoader): Validation data for model selection
        test_loader (DataLoader): Independent test data for final evaluation
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        scheduler: Learning rate scheduler
        criterion: Loss function (typically Focal Loss for imbalanced data)
        scaler: Mixed precision scaler for efficiency
        device (torch.device): Training device
        logger (logging.Logger): Logging instance
        tta_transform: Test-time augmentation transforms
        
    Returns:
        dict: Training summary, validation metrics, test metrics, and configuration
    """
    
    # ========================================================================
    # ENSEMBLE TRAINING BRANCH
    # ========================================================================
    if USE_ENSEMBLE_TRAINING:
        logger.info("=" * 60)
        logger.info("ENSEMBLE TRAINING MODE ENABLED")
        logger.info(f"Training {ENSEMBLE_N_MODELS} diverse models")
        logger.info(f"Diversity strategy: {ENSEMBLE_DIVERSITY_STRATEGY}")
        logger.info(f"Aggregation method: {ENSEMBLE_AGGREGATION}")
        logger.info("=" * 60)
        
        # Create ensemble models with diversity
        ensemble_models = create_ensemble_models(
            n_models=ENSEMBLE_N_MODELS,
            device=device,
            diversity_strategy=ENSEMBLE_DIVERSITY_STRATEGY
        )
        
        # Create independent optimizers and schedulers for each model
        ensemble_optimizers = [
            optim.Adam(m.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            for m in ensemble_models
        ]
        
        ensemble_schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
            )
            for opt in ensemble_optimizers
        ]
        
        # Train ensemble
        logger.info("\nStarting ensemble training...")
        trained_models, training_histories = train_ensemble(
            models=ensemble_models,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizers=ensemble_optimizers,
            schedulers=ensemble_schedulers,
            criterion=criterion,
            scaler=scaler,
            device=device,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            checkpoint_dir=CHECKPOINT_DIR,
            logger=logger
        )
        
        # Evaluate ensemble on validation set
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING ENSEMBLE ON VALIDATION SET")
        logger.info("=" * 60)
        
        val_probs, val_true = ensemble_predict(
            models=trained_models,
            loader=val_loader,
            device=device,
            aggregation=ENSEMBLE_AGGREGATION,
            use_tta=(TTA_N > 0),
            tta_n=TTA_N if TTA_N > 0 else 0,
            tta_transform=tta_transform if TTA_N > 0 else None
        )
        
        # Find optimal threshold using ensemble predictions
        optimal_threshold = find_optimal_threshold(val_true, val_probs, logger)
        logger.info(f"Optimal ensemble threshold: {optimal_threshold:.4f}")
        
        val_metrics, val_pred = compute_classification_metrics(
            val_true, val_probs, threshold=optimal_threshold
        )
        
        logger.info("Ensemble Validation Metrics:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Generate validation visualizations
        plot_roc_curve(val_true, val_probs, 
                      os.path.join(METRICS_DIR, "roc_ensemble_val.png"))
        plot_precision_recall_curve(val_true, val_probs, 
                                   os.path.join(METRICS_DIR, "pr_curve_ensemble_val.png"))
        plot_confusion_matrix(val_true, val_pred, 
                            os.path.join(METRICS_DIR, "confusion_matrix_ensemble_val.png"))
        
        # Evaluate ensemble on test set
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING ENSEMBLE ON TEST SET")
        logger.info("=" * 60)
        
        ensemble_test_results = evaluate_ensemble(
            models=trained_models,
            test_loader=test_loader,
            device=device,
            optimal_threshold=optimal_threshold,
            aggregation=ENSEMBLE_AGGREGATION,
            use_tta=(TTA_N > 0),
            tta_n=TTA_N if TTA_N > 0 else 0,
            tta_transform=tta_transform if TTA_N > 0 else None,
            logger=logger
        )
        
        # Generate test visualizations
        plot_roc_curve(ensemble_test_results['true_labels'], 
                      ensemble_test_results['probabilities'],
                      os.path.join(METRICS_DIR, "roc_ensemble_test.png"))
        plot_precision_recall_curve(ensemble_test_results['true_labels'],
                                   ensemble_test_results['probabilities'],
                                   os.path.join(METRICS_DIR, "pr_curve_ensemble_test.png"))
        plot_confusion_matrix(ensemble_test_results['true_labels'],
                            ensemble_test_results['predictions'],
                            os.path.join(METRICS_DIR, "confusion_matrix_ensemble_test.png"))
        
        # Calculate ensemble statistics
        best_val_aucs = [h['best_auc'] for h in training_histories]
        mean_val_auc = np.mean(best_val_aucs)
        std_val_auc = np.std(best_val_aucs)
        
        logger.info("\n" + "=" * 60)
        logger.info("ENSEMBLE TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Individual model validation AUCs:")
        for idx, auc in enumerate(best_val_aucs):
            logger.info(f"  Model {idx+1}: {auc:.6f}")
        logger.info(f"Mean validation AUC: {mean_val_auc:.6f} ± {std_val_auc:.6f}")
        logger.info(f"Ensemble test AUC: {ensemble_test_results['metrics']['roc_auc']:.6f}")
        
        improvement = ensemble_test_results['metrics']['roc_auc'] - mean_val_auc
        logger.info(f"Ensemble improvement over mean: {improvement:+.6f}")
        
        # Return ensemble results
        return {
            "training_method": "ensemble_training",
            "ensemble_config": {
                "n_models": ENSEMBLE_N_MODELS,
                "diversity_strategy": ENSEMBLE_DIVERSITY_STRATEGY,
                "aggregation": ENSEMBLE_AGGREGATION
            },
            "training_histories": training_histories,
            "individual_model_aucs": {
                "validation_aucs": best_val_aucs,
                "mean": mean_val_auc,
                "std": std_val_auc
            },
            "validation_metrics": val_metrics,
            "test_metrics": ensemble_test_results['metrics'],
            "test_predictions": {
                "test_probs": ensemble_test_results['probabilities'].tolist(),
                "test_pred": ensemble_test_results['predictions'].tolist(),
                "test_true": ensemble_test_results['true_labels'].tolist()
            },
            "configuration": {
                "tta_n": TTA_N,
                "use_amp": USE_AMP,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "device": str(device)
            }
        }
    
    # ========================================================================
    # SINGLE MODEL TRAINING (ORIGINAL IMPLEMENTATION)
    # ========================================================================
    
    logger.info("=" * 60)
    logger.info("SINGLE MODEL TRAINING MODE")
    logger.info("=" * 60)
    
    # Log detailed model architecture information
    log_model_info(model, logger)
    
    # Initialize TensorBoard for real-time training monitoring
    writer = SummaryWriter(log_dir=os.path.join(RESULTS_DIR, "logs"))
    print(f"TensorBoard logs saved to {writer.log_dir}")
    logger.info(f"TensorBoard logging initialized at: {writer.log_dir}")
    
    # Log comprehensive training configuration
    logger.info(f"Training Configuration:")
    logger.info(f"  Max epochs: {MAX_EPOCHS}")
    logger.info(f"  Batch size: {train_loader.batch_size}")
    logger.info(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    logger.info(f"  Weight decay: {optimizer.param_groups[0]['weight_decay']:.2e}")
    logger.info(f"  Mixed precision: {USE_AMP}")
    logger.info(f"  Early stopping patience: {PATIENCE}")
    logger.info(f"  Device: {device}")
    
    # Initialize training state variables
    best_auc = 0.0
    patience_counter = 0
    training_start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    # Checkpoint management
    start_epoch = 0
    epochs_ran = start_epoch
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    # Load existing checkpoint if available
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        epochs_ran = start_epoch
        best_auc = checkpoint['best_auc']
        logger.info(f"Resuming from epoch {start_epoch} with best AUC: {best_auc:.4f}")
    
    # Main training loop
    for epoch in range(start_epoch, MAX_EPOCHS):
        epoch_start_time = datetime.now()

        logger.info(f"\n{'='*20} EPOCH {epoch + 1}/{MAX_EPOCHS} {'='*20}")
        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
        
        if epoch % 5 == 0:
            log_system_stats(logger)
        
        # Training phase
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch, logger)

        # Validation phase
        val_auc, val_detailed_metrics = evaluate_with_threshold_search(model, val_loader, epoch, logger)

        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        
        # Epoch summary
        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_loss:.6f}")
        logger.info(f"  Val AUC: {val_auc:.6f}")
        logger.info(f"  Epoch Time: {epoch_time:.2f}s")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        epochs_ran += 1

        print(f"Epoch {epoch + 1}/{MAX_EPOCHS} - Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("ROC-AUC/val", val_auc, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Epoch_Time", epoch_time, epoch)
        
        if val_detailed_metrics:
            for metric_name, metric_value in val_detailed_metrics.items():
                if isinstance(metric_value, (int, float)):
                    writer.add_scalar(f"Validation/{metric_name}", metric_value, epoch)

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_auc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            logger.info(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
        
        # Checkpointing
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_auc": best_auc,
            "train_loss": train_loss,
            "val_auc": val_auc,
            "val_detailed_metrics": val_detailed_metrics
        }
        
        epoch_ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(ckpt, epoch_ckpt_path)
        
        # Best model tracking and early stopping
        if val_auc > best_auc:
            improvement = val_auc - best_auc
            best_auc = val_auc
            best_path = os.path.join(CHECKPOINT_DIR, f"best_model.pth")
            patience_counter = 0

            torch.save(ckpt, best_path)

            logger.info(f"NEW BEST MODEL! AUC improved by {improvement:.6f} to {best_auc:.6f}")
            print(f"✓ Saved new best model with AUC={best_auc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            print(f"No improvement. Patience {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            logger.info(f"EARLY STOPPING triggered after {epoch + 1} epochs")
            print("Early stopping triggered!")
            break
    
    # Training completion
    training_time = (datetime.now() - training_start_time).total_seconds()

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Total training time: {training_time:.2f}s ({training_time/60:.2f} minutes)")
    logger.info(f"Best validation AUC: {best_auc:.6f}")
    logger.info("=" * 60)
    
    print(f"Training completed in {training_time/60:.2f} minutes")
    writer.close()
    
    # Load best model for evaluation
    best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best model from epoch {ckpt.get('epoch', 'unknown')}")
        print(f"Loaded best model from {best_path} (epoch {ckpt.get('epoch')})")
    else:
        logger.warning("Best model checkpoint not found, using current model state")
    
    # Validation set evaluation
    logger.info("Performing final validation set evaluation...")

    val_probs, val_true = predict_proba(
        model=model,
        loader=val_loader,
        device=device,
        use_tta=(TTA_N > 0),
        tta_n=TTA_N if TTA_N > 0 else 0,
        tta_transform=tta_transform if TTA_N > 0 else None
    )

    optimal_threshold = find_optimal_threshold(val_true, val_probs, logger)
    logger.info(f"Optimal validation threshold: {optimal_threshold:.4f}")

    val_metrics, val_pred = compute_classification_metrics(
        val_true, val_probs, threshold=optimal_threshold
    )
    
    logger.info("Validation metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    plot_roc_curve(val_true, val_probs, os.path.join(METRICS_DIR, "roc_val.png"))
    plot_precision_recall_curve(val_true, val_probs, os.path.join(METRICS_DIR, "pr_curve_val.png"))
    plot_confusion_matrix(val_true, val_pred, os.path.join(METRICS_DIR, "confusion_matrix.png"))
    
    # Test set evaluation
    logger.info("Performing final test set evaluation...")

    test_probs, test_true = predict_proba(
        model=model,
        loader=test_loader,
        device=device,
        use_tta=(TTA_N > 0),
        tta_n=TTA_N if TTA_N > 0 else 0,
        tta_transform=tta_transform if TTA_N > 0 else None
    )

    test_metrics, test_pred = compute_classification_metrics(
        test_true, test_probs, threshold=optimal_threshold
    )

    logger.info("Test metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    plot_roc_curve(test_true, test_probs, os.path.join(METRICS_DIR, "roc_test.png"))
    plot_precision_recall_curve(test_true, test_probs, os.path.join(METRICS_DIR, "pr_curve_test.png"))
    plot_confusion_matrix(test_true, test_pred, os.path.join(METRICS_DIR, "confusion_matrix_test.png"))
    
    # Return results
    return {
        "training_method": "single_split",
        "training_summary": {
            "total_epochs": epochs_ran,
            "best_validation_auc": best_auc,
            "training_time_minutes": training_time / 60,
            "final_learning_rate": optimizer.param_groups[0]['lr']
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_predictions": {
            "test_probs": test_probs.tolist() if hasattr(test_probs, 'tolist') else test_probs,
            "test_pred": test_pred.tolist() if hasattr(test_pred, 'tolist') else test_pred,
            "test_true": test_true.tolist() if hasattr(test_true, 'tolist') else test_true
        },
        "configuration": {
            "tta_n": TTA_N,
            "use_amp": USE_AMP,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "device": str(device)
        }
    }

def prepare_sensitive_features(test_df):
    """
    Extract sensitive demographic features for fairness analysis.
    
    Args:
        test_df: Test dataframe with metadata
        
    Returns:
        pd.DataFrame: Sensitive features (age_group, sex, etc.)
    """
    sensitive_features = pd.DataFrame()
    
    # Age groups (if available in your data)
    if 'age_approx' in test_df.columns:
        sensitive_features['age_group'] = pd.cut(
            test_df['age_approx'],
            bins=[0, 30, 50, 70, 100],
            labels=['<30', '30-50', '50-70', '70+']
        )
    
    # Sex (if available)
    if 'sex' in test_df.columns:
        sensitive_features['sex'] = test_df['sex']
    
    # Anatomical site (if available)
    if 'anatom_site_general_challenge' in test_df.columns:
        sensitive_features['anatom_site'] = test_df['anatom_site_general_challenge']
    
    return sensitive_features

if __name__ == "__main__":
    main()