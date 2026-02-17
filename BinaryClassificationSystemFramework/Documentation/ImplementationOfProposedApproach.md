# Implementation of the Proposed Approach

## Introduction

This thesis proposes a comprehensive software framework for developing and validating deep learning models in medical imaging, demonstrated through melanoma classification. The framework implements transfer learning, focal loss for class imbalance, test-time augmentation, and extensive evaluation metrics.

The modular pipeline covers data preprocessing, model development, training, validation, and performance analysis, demonstrated on the ISIC 2020 dermoscopic image dataset. The implementation follows the BPMN workflow illustrated in Figure X.

---

## 1. Data Collection and Preprocessing

This phase encompasses data acquisition and preparation using Pandas, NumPy, and scikit-learn. The framework implements a two-stage approach: metadata processing for tabular patient information, and image transformation pipelines for neural network consumption.

### 1.1 Data Collection

The data collection process loads dataset metadata and links each patient record to its corresponding dermoscopic image.

```
ALGORITHM 1: Data Collection
─────────────────────────────────────────────────────────────
Input: CSV_PATH (path to metadata file), DATA_DIR (image directory)
Output: DataFrame with image paths and labels

BEGIN
    1. Load metadata from CSV_PATH into DataFrame
    2. FOR each record in DataFrame DO
        2.1 Construct filepath ← DATA_DIR + image_name + ".jpg"
        2.2 Append filepath to record
    END FOR
    3. Log dataset statistics:
        - Total samples count
        - Class distribution (benign vs melanoma)
        - Prevalence rate calculation
    4. RETURN DataFrame with complete paths
END
─────────────────────────────────────────────────────────────
```

### 1.2 Data Preprocessing

The preprocessing stage transforms raw data for deep learning consumption, handling missing values and preparing image transformations. Categorical missing values are filled with "Unknown", whilst numerical values use median imputation. Image augmentation during training improves generalisation, whilst validation uses consistent transformations.

```
ALGORITHM 2: Data Preprocessing Pipeline
─────────────────────────────────────────────────────────────
Input: Raw DataFrame D
Output: Preprocessed DataFrame D'

BEGIN
    1. HANDLE MISSING VALUES:
        FOR each column c in D DO
            IF c is categorical THEN
                Fill missing with "Unknown"
            ELSE IF c is numerical THEN
                Fill missing with median(c)
            END IF
        END FOR
    
    2. ENCODE CATEGORICAL FEATURES:
        FOR each categorical column c DO
            Apply label encoding or one-hot encoding
        END FOR
    
    3. DERIVE ADDITIONAL FEATURES:
        - age_group ← Bin age into categories [<30, 30-50, 50-70, 70+]
        - anatomical_risk ← Map anatomical sites to risk levels
    
    4. DEFINE IMAGE TRANSFORMATIONS:
        training_transforms ← {
            Resize(IMAGE_SIZE),
            RandomRotation(±20°),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            ColorJitter(brightness, contrast, saturation),
            Normalize(ImageNet_mean, ImageNet_std)
        }
        
        validation_transforms ← {
            Resize(IMAGE_SIZE),
            Normalize(ImageNet_mean, ImageNet_std)
        }
    
    5. RETURN D', training_transforms, validation_transforms
END
─────────────────────────────────────────────────────────────
```

---

## 2. Data Validation

The data validation phase implements the decision gateway: "Is data quality sufficient?" Using Great Expectations and PyDeequ, the pipeline applies schema validation, image integrity checks, statistical quality control, and class distribution analysis to ensure data integrity before model training.

### 2.1 Comprehensive Validation Pipeline

The validation pipeline integrates schema validation, image integrity checks, and statistical quality control. Critical failures halt the pipeline early to prevent training on flawed data.

```
ALGORITHM 3: Data Validation Pipeline
─────────────────────────────────────────────────────────────
Input: DataFrame D, DATA_DIR
Output: validation_passed (Boolean), validation_report

BEGIN
    validation_results ← {}
    critical_failures ← []
    
    1. SCHEMA VALIDATION (Great Expectations):
        expected_schema ← {
            'image_name': String, not_null,
            'target': Integer ∈ {0, 1},
            'age_approx': Float, range [0, 120],
            'sex': String ∈ {'male', 'female', 'unknown'},
            'anatom_site': String, categorical
        }
        
        FOR each column c in expected_schema DO
            IF NOT validate_column(D[c], expected_schema[c]) THEN
                critical_failures.append(c)
            END IF
        END FOR
        
        validation_results['schema'] ← schema_report
    
    2. IMAGE INTEGRITY VERIFICATION:
        corrupted_images ← []
        missing_images ← []
        
        FOR each record r in D DO
            filepath ← r.filepath
            IF NOT file_exists(filepath) THEN
                missing_images.append(filepath)
            ELSE IF NOT is_valid_image(filepath) THEN
                corrupted_images.append(filepath)
            END IF
        END FOR
        
        validation_results['image_integrity'] ← {
            'missing': missing_images,
            'corrupted': corrupted_images,
            'valid_ratio': (total - len(missing) - len(corrupted)) / total
        }
    
    3. STATISTICAL QUALITY CONTROL:
        // Outlier detection using multiple methods
        FOR each numerical column c DO
            outliers_iqr ← detect_outliers_IQR(D[c], threshold=1.5)
            outliers_zscore ← detect_outliers_modified_zscore(D[c], threshold=3.5)
            outliers_isolation ← detect_outliers_isolation_forest(D[c])
            
            consensus_outliers ← intersection(outliers_iqr, outliers_zscore)
            validation_results['outliers'][c] ← consensus_outliers
        END FOR
    
    4. CLASS DISTRIBUTION ANALYSIS:
        class_counts ← count_by_class(D['target'])
        imbalance_ratio ← max(class_counts) / min(class_counts)
        
        IF imbalance_ratio > CRITICAL_IMBALANCE_THRESHOLD THEN
            Log warning: "Severe class imbalance detected"
            validation_results['class_imbalance'] ← 'severe'
        END IF
    
    5. DUPLICATE DETECTION:
        duplicate_images ← find_duplicates(D['image_name'])
        validation_results['duplicates'] ← duplicate_images
    
    6. DECISION GATEWAY - Data Quality Sufficient?
        IF len(critical_failures) > 0 OR 
           len(missing_images) > MAX_MISSING_THRESHOLD OR
           len(corrupted_images) > MAX_CORRUPTED_THRESHOLD THEN
            validation_passed ← FALSE
            // Return to data collection (as per BPMN)
        ELSE
            validation_passed ← TRUE
            // Proceed to feature engineering
        END IF
    
    7. GENERATE REPORTS:
        Export validation_report.json
        Export validation_summary.txt
    
    8. RETURN validation_passed, validation_results
END
─────────────────────────────────────────────────────────────
```

### 2.2 Outlier Detection Methods

The framework employs IQR, Modified Z-Score, and Isolation Forest methods for outlier detection. A consensus-based approach flags outliers only when identified by multiple methods, reducing false positives whilst maintaining sensitivity.

```
ALGORITHM 3.1: Interquartile Range (IQR) Method
─────────────────────────────────────────────────────────────
Input: Array X, threshold k (default=1.5)
Output: Set of outlier indices

BEGIN
    Q1 ← percentile(X, 25)
    Q3 ← percentile(X, 75)
    IQR ← Q3 - Q1
    
    lower_bound ← Q1 - k × IQR
    upper_bound ← Q3 + k × IQR
    
    outliers ← {i : X[i] < lower_bound OR X[i] > upper_bound}
    RETURN outliers
END
─────────────────────────────────────────────────────────────
```

```
ALGORITHM 3.2: Modified Z-Score Method
─────────────────────────────────────────────────────────────
Input: Array X, threshold t (default=3.5)
Output: Set of outlier indices

BEGIN
    median_X ← median(X)
    MAD ← median(|X - median_X|)  // Median Absolute Deviation
    
    IF MAD = 0 THEN
        MAD ← 1.0  // Prevent division by zero
    END IF
    
    modified_z_scores ← 0.6745 × (X - median_X) / MAD
    
    outliers ← {i : |modified_z_scores[i]| > t}
    RETURN outliers
END
─────────────────────────────────────────────────────────────
```

---

## 3. Feature Engineering and Data Splitting

Upon successful validation, the pipeline proceeds to feature engineering and stratified data splitting using scikit-learn. Stratified splitting ensures each partition maintains the original class distribution, critical for the severe class imbalance (~2% melanoma prevalence) typical of medical datasets.

### 3.1 Feature Engineering

Feature engineering transforms age into clinically meaningful categories, maps anatomical sites to sun exposure risk levels, and computes class weights to address imbalanced data during training.

```
ALGORITHM 4: Feature Engineering for Medical Imaging
─────────────────────────────────────────────────────────────
Input: Validated DataFrame D
Output: Feature-enhanced DataFrame D_enhanced

BEGIN
    1. DEMOGRAPHIC FEATURE ENGINEERING:
        // Age group stratification for clinical relevance
        FOR each record r in D DO
            IF r.age < 30 THEN
                r.age_group ← '<30'
            ELSE IF r.age < 50 THEN
                r.age_group ← '30-50'
            ELSE IF r.age < 70 THEN
                r.age_group ← '50-70'
            ELSE
                r.age_group ← '70+'
            END IF
        END FOR
    
    2. ANATOMICAL RISK MAPPING:
        risk_mapping ← {
            'torso': 'high_exposure',
            'upper extremity': 'high_exposure',
            'lower extremity': 'moderate_exposure',
            'head/neck': 'high_exposure',
            'palms/soles': 'low_exposure',
            'oral/genital': 'low_exposure'
        }
        
        FOR each record r in D DO
            r.anatomical_risk ← risk_mapping[r.anatom_site]
        END FOR
    
    3. COMPUTE CLASS WEIGHTS FOR IMBALANCE:
        class_counts ← count_by_class(D['target'])
        total_samples ← sum(class_counts)
        
        FOR each class c DO
            class_weights[c] ← total_samples / (n_classes × class_counts[c])
        END FOR
        
        // Calculate per-sample weights for weighted sampling
        FOR each record r in D DO
            r.sample_weight ← class_weights[r.target]
        END FOR
    
    4. RETURN D_enhanced, class_weights
END
─────────────────────────────────────────────────────────────
```

### 3.2 Stratified Data Splitting

The framework implements two-stage stratified splitting: first separating the test set, then partitioning remaining data into training and validation subsets. Fixed random seeds ensure reproducibility.

```
ALGORITHM 5: Stratified Train-Validation-Test Split
─────────────────────────────────────────────────────────────
Input: DataFrame D, test_ratio, val_ratio, random_seed
Output: D_train, D_val, D_test

BEGIN
    1. SET random seed for reproducibility
    
    2. FIRST SPLIT - Separate test set:
        D_train_val, D_test ← stratified_split(
            D,
            test_size = test_ratio,
            stratify = D['target'],
            random_state = random_seed
        )
    
    3. SECOND SPLIT - Separate validation from training:
        adjusted_val_ratio ← val_ratio / (1 - test_ratio)
        
        D_train, D_val ← stratified_split(
            D_train_val,
            test_size = adjusted_val_ratio,
            stratify = D_train_val['target'],
            random_state = random_seed
        )
    
    4. VERIFY CLASS DISTRIBUTIONS:
        FOR each split S in {D_train, D_val, D_test} DO
            melanoma_ratio ← count(S['target'] = 1) / len(S)
            Log: "Split contains {melanoma_ratio}% melanoma cases"
            
            ASSERT |melanoma_ratio - original_ratio| < TOLERANCE
        END FOR
    
    5. LOG SPLIT STATISTICS:
        Log: "Training set: {len(D_train)} samples"
        Log: "Validation set: {len(D_val)} samples"
        Log: "Test set: {len(D_test)} samples"
    
    6. RETURN D_train, D_val, D_test
END
─────────────────────────────────────────────────────────────
```

---

## 4. Model Training and Tuning

This phase implements transfer learning with pre-trained CNNs, using specialised loss functions for class imbalance. The training incorporates mixed precision training, learning rate scheduling, and early stopping for optimal performance.

### 4.1 Model Architecture

The framework uses ResNet-50 with ImageNet pre-trained weights. The classification head is replaced with a custom head for binary melanoma classification.

```
ALGORITHM 6: Model Architecture Construction
─────────────────────────────────────────────────────────────
Input: num_classes, dropout_rate, pretrained_weights
Output: Configured neural network model

BEGIN
    1. LOAD BASE ARCHITECTURE:
        base_model ← ResNet50(weights = 'ImageNet')
        
        // Freeze early layers (feature extraction)
        FOR layer in base_model.layers[0:freeze_layers] DO
            layer.trainable ← FALSE
        END FOR
    
    2. MODIFY CLASSIFICATION HEAD:
        num_features ← base_model.fc.in_features
        
        classifier ← Sequential(
            Dropout(dropout_rate),
            Linear(num_features → 512),
            ReLU(),
            BatchNorm1d(512),
            Dropout(dropout_rate / 2),
            Linear(512 → num_classes),
            Sigmoid()  // For binary classification
        )
        
        base_model.fc ← classifier
    
    3. INITIALISE WEIGHTS:
        FOR layer in classifier DO
            IF layer is Linear THEN
                Xavier_uniform_initialisation(layer.weights)
                Zero_initialisation(layer.bias)
            END IF
        END FOR
    
    4. RETURN base_model
END
─────────────────────────────────────────────────────────────
```

### 4.2 Focal Loss for Class Imbalance

To address severe class imbalance (~2% positive cases), the framework implements Focal Loss, which down-weights well-classified examples using a modulating factor $(1 - p_t)^\gamma$. This focuses learning on hard examples rather than abundant easy negatives.

```
ALGORITHM 7: Focal Loss Computation
─────────────────────────────────────────────────────────────
Input: predictions p, ground_truth y, α (class weight), γ (focusing parameter)
Output: Focal loss value

BEGIN
    // Standard cross-entropy component
    BCE ← -[y × log(p) + (1-y) × log(1-p)]
    
    // Focal modulating factor
    p_t ← y × p + (1-y) × (1-p)  // Probability of true class
    focal_weight ← (1 - p_t)^γ
    
    // Class balancing factor
    α_t ← y × α + (1-y) × (1-α)
    
    // Combined focal loss
    focal_loss ← α_t × focal_weight × BCE
    
    RETURN mean(focal_loss)
END
─────────────────────────────────────────────────────────────

// Typical configuration for melanoma detection:
// α = 0.25 (higher weight for minority class)
// γ = 2.0  (focusing parameter)
```

### 4.3 Training Loop with Mixed Precision

The training uses mixed precision (16-bit forward/backward passes, 32-bit updates) with Adam optimiser. Checkpointing saves the best model, and early stopping halts training when validation performance plateaus.

```
ALGORITHM 8: Model Training Loop
─────────────────────────────────────────────────────────────
Input: model, train_loader, val_loader, optimiser, criterion, 
       max_epochs, patience, device
Output: trained_model, training_history

BEGIN
    best_val_auc ← 0.0
    patience_counter ← 0
    training_history ← []
    scaler ← GradientScaler()  // For mixed precision
    
    FOR epoch = 1 TO max_epochs DO
        // ═══════════════════════════════════════════════════
        // TRAINING PHASE
        // ═══════════════════════════════════════════════════
        model.train()
        epoch_loss ← 0.0
        
        FOR batch in train_loader DO
            images, labels ← batch
            images ← images.to(device)
            labels ← labels.to(device)
            
            optimiser.zero_grad()
            
            // Mixed precision forward pass
            WITH autocast(device_type='cuda') DO
                predictions ← model(images)
                loss ← criterion(predictions, labels)
            END WITH
            
            // Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            
            epoch_loss ← epoch_loss + loss.item()
        END FOR
        
        avg_train_loss ← epoch_loss / len(train_loader)
        
        // ═══════════════════════════════════════════════════
        // VALIDATION PHASE
        // ═══════════════════════════════════════════════════
        model.eval()
        val_predictions ← []
        val_labels ← []
        
        WITH no_gradient() DO
            FOR batch in val_loader DO
                images, labels ← batch
                predictions ← model(images.to(device))
                val_predictions.extend(predictions)
                val_labels.extend(labels)
            END FOR
        END WITH
        
        val_auc ← compute_ROC_AUC(val_labels, val_predictions)
        
        // ═══════════════════════════════════════════════════
        // LEARNING RATE SCHEDULING
        // ═══════════════════════════════════════════════════
        scheduler.step(val_auc)
        
        // ═══════════════════════════════════════════════════
        // EARLY STOPPING CHECK
        // ═══════════════════════════════════════════════════
        IF val_auc > best_val_auc THEN
            best_val_auc ← val_auc
            patience_counter ← 0
            save_checkpoint(model, 'best_model.pth')
            Log: "New best model saved (AUC: {val_auc})"
        ELSE
            patience_counter ← patience_counter + 1
        END IF
        
        IF patience_counter ≥ patience THEN
            Log: "Early stopping triggered at epoch {epoch}"
            BREAK
        END IF
        
        training_history.append({epoch, avg_train_loss, val_auc})
    END FOR
    
    // Load best model
    model ← load_checkpoint('best_model.pth')
    
    RETURN model, training_history
END
─────────────────────────────────────────────────────────────
```

### 4.4 Ensemble Training (Optional Enhancement)

For enhanced robustness, the framework supports ensemble training with multiple diverse models. Diversity is achieved through varied initialisation or dropout variation. Predictions are aggregated via averaging, weighted averaging, or voting.

```
ALGORITHM 9: Ensemble Training
─────────────────────────────────────────────────────────────
Input: n_models, train_loader, val_loader, diversity_strategy
Output: ensemble_models[], training_histories[]

BEGIN
    ensemble_models ← []
    training_histories ← []
    
    FOR i = 1 TO n_models DO
        Log: "Training ensemble model {i}/{n_models}"
        
        // Create diverse model based on strategy
        IF diversity_strategy = "varied_init" THEN
            set_random_seed(BASE_SEED + i × 100)
            model_i ← create_model()
        ELSE IF diversity_strategy = "dropout_variation" THEN
            dropout_rate ← 0.3 + (i × 0.1)
            model_i ← create_model(dropout = dropout_rate)
        END IF
        
        // Train individual model
        trained_model, history ← train_model(
            model_i, train_loader, val_loader
        )
        
        ensemble_models.append(trained_model)
        training_histories.append(history)
    END FOR
    
    RETURN ensemble_models, training_histories
END
─────────────────────────────────────────────────────────────
```

```
ALGORITHM 10: Ensemble Prediction Aggregation
─────────────────────────────────────────────────────────────
Input: ensemble_models[], test_loader, aggregation_method
Output: aggregated_predictions

BEGIN
    all_predictions ← []
    
    // Collect predictions from each model
    FOR model in ensemble_models DO
        model_predictions ← predict(model, test_loader)
        all_predictions.append(model_predictions)
    END FOR
    
    // Stack predictions: shape (n_models, n_samples)
    predictions_matrix ← stack(all_predictions)
    
    // Aggregate based on method
    IF aggregation_method = "average" THEN
        aggregated ← mean(predictions_matrix, axis=0)
    ELSE IF aggregation_method = "weighted" THEN
        // Weight by validation AUC
        weights ← softmax(validation_aucs)
        aggregated ← weighted_mean(predictions_matrix, weights)
    ELSE IF aggregation_method = "voting" THEN
        binary_predictions ← (predictions_matrix > 0.5)
        aggregated ← mean(binary_predictions, axis=0)
    END IF
    
    RETURN aggregated
END
─────────────────────────────────────────────────────────────
```

---

## 5. Model Evaluation on Holdout Set

The evaluation phase computes comprehensive metrics on the held-out test set, implementing the decision gateway: "Are performance metrics satisfactory?" The framework assesses sensitivity, specificity, and calibrated probability estimates essential for medical AI.

### 5.1 Test-Time Augmentation (TTA)

Test-time augmentation improves prediction robustness by averaging predictions across multiple augmented versions of each test image, reducing view-specific noise.

```
ALGORITHM 11: Test-Time Augmentation
─────────────────────────────────────────────────────────────
Input: model, image, tta_transforms, n_augmentations
Output: robust_prediction

BEGIN
    predictions ← []
    
    // Original image prediction
    pred_original ← model(normalise(image))
    predictions.append(pred_original)
    
    // Augmented predictions
    FOR i = 1 TO n_augmentations DO
        augmented_image ← apply_random_augmentation(image, tta_transforms)
        pred_aug ← model(normalise(augmented_image))
        predictions.append(pred_aug)
    END FOR
    
    // Aggregate predictions
    robust_prediction ← mean(predictions)
    
    RETURN robust_prediction
END
─────────────────────────────────────────────────────────────
```

### 5.2 Optimal Threshold Selection

The framework optimises classification threshold using Youden's J statistic to balance sensitivity and specificity. For melanoma screening, sensitivity is typically prioritised to minimise missed diagnoses.

```
ALGORITHM 12: Optimal Threshold Determination
─────────────────────────────────────────────────────────────
Input: true_labels, predicted_probabilities
Output: optimal_threshold, threshold_metrics

BEGIN
    thresholds ← linspace(0.0, 1.0, 1000)
    best_threshold ← 0.5
    best_metric ← 0.0
    
    FOR threshold in thresholds DO
        predictions ← (predicted_probabilities > threshold)
        
        // Calculate Youden's J statistic
        sensitivity ← TP / (TP + FN)
        specificity ← TN / (TN + FP)
        youden_j ← sensitivity + specificity - 1
        
        // Alternative: F1 Score optimisation
        precision ← TP / (TP + FP)
        recall ← sensitivity
        f1 ← 2 × (precision × recall) / (precision + recall)
        
        // Select best threshold (prioritise recall for medical)
        IF youden_j > best_metric THEN
            best_metric ← youden_j
            best_threshold ← threshold
        END IF
    END FOR
    
    Log: "Optimal threshold: {best_threshold}"
    Log: "Youden's J: {best_metric}"
    
    RETURN best_threshold, threshold_metrics
END
─────────────────────────────────────────────────────────────
```

### 5.3 Comprehensive Metrics Computation

The framework computes accuracy, precision, recall, F1, ROC-AUC, Average Precision, sensitivity, specificity, PPV, NPV, MCC, and Cohen's Kappa to characterise model performance comprehensively.

```
ALGORITHM 13: Classification Metrics Computation
─────────────────────────────────────────────────────────────
Input: true_labels, predictions, probabilities, threshold
Output: metrics_dictionary

BEGIN
    binary_predictions ← (probabilities > threshold)
    
    // Confusion Matrix Elements
    TP ← count(predictions = 1 AND true_labels = 1)
    TN ← count(predictions = 0 AND true_labels = 0)
    FP ← count(predictions = 1 AND true_labels = 0)
    FN ← count(predictions = 0 AND true_labels = 1)
    
    // Primary Metrics
    metrics ← {
        'accuracy': (TP + TN) / (TP + TN + FP + FN),
        'precision': TP / (TP + FP),
        'recall': TP / (TP + FN),           // Sensitivity
        'specificity': TN / (TN + FP),
        'f1_score': 2 × precision × recall / (precision + recall),
        'roc_auc': compute_AUC(true_labels, probabilities),
        'average_precision': compute_AP(true_labels, probabilities)
    }
    
    // Medical-Specific Metrics
    metrics['negative_predictive_value'] ← TN / (TN + FN)
    metrics['positive_predictive_value'] ← TP / (TP + FP)
    
    RETURN metrics
END
─────────────────────────────────────────────────────────────
```

### 5.4 Performance Evaluation Decision Gateway

The decision gateway determines whether the model meets minimum acceptable performance thresholds (AUC, sensitivity, specificity) based on clinical requirements before proceeding to fairness assessment.

```
ALGORITHM 14: Performance Satisfactory Decision
─────────────────────────────────────────────────────────────
Input: computed_metrics, performance_thresholds
Output: is_satisfactory (Boolean), recommendations

BEGIN
    // Define minimum acceptable thresholds for medical AI
    min_thresholds ← {
        'roc_auc': 0.85,
        'recall': 0.80,      // High sensitivity required
        'specificity': 0.70,
        'f1_score': 0.50
    }
    
    failed_criteria ← []
    
    FOR metric, threshold in min_thresholds DO
        IF computed_metrics[metric] < threshold THEN
            failed_criteria.append(metric)
        END IF
    END FOR
    
    IF len(failed_criteria) = 0 THEN
        is_satisfactory ← TRUE
        // Proceed to Fairness Assessment (per BPMN)
    ELSE
        is_satisfactory ← FALSE
        // Return to Model Training and Tuning (per BPMN)
        recommendations ← generate_improvement_suggestions(failed_criteria)
    END IF
    
    RETURN is_satisfactory, recommendations
END
─────────────────────────────────────────────────────────────
```

---

## 6. Assessing Fairness and Bias

Following satisfactory performance, the framework assesses fairness across demographic subgroups (age, sex, anatomical site). Multiple fairness metrics are computed to identify potential algorithmic bias that could result in unequal diagnostic accuracy.

### 6.1 Fairness Metrics Computation

The framework computes demographic parity, equalised odds, and predictive parity across sensitive attributes to identify performance disparities between demographic groups.

```
ALGORITHM 15: Fairness Assessment Across Sensitive Attributes
─────────────────────────────────────────────────────────────
Input: predictions, true_labels, sensitive_features, attribute_name
Output: fairness_report

BEGIN
    fairness_report ← {}
    unique_groups ← unique(sensitive_features[attribute_name])
    
    // Compute metrics per subgroup
    group_metrics ← {}
    FOR group in unique_groups DO
        mask ← (sensitive_features[attribute_name] = group)
        group_preds ← predictions[mask]
        group_labels ← true_labels[mask]
        
        group_metrics[group] ← {
            'size': count(mask),
            'prevalence': mean(group_labels),
            'accuracy': compute_accuracy(group_labels, group_preds),
            'recall': compute_recall(group_labels, group_preds),
            'precision': compute_precision(group_labels, group_preds),
            'fpr': compute_false_positive_rate(group_labels, group_preds),
            'fnr': compute_false_negative_rate(group_labels, group_preds),
            'auc': compute_AUC(group_labels, group_preds)
        }
    END FOR
    
    // Compute Fairness Disparities
    reference_group ← group_with_largest_size(group_metrics)
    
    FOR group in unique_groups DO
        IF group ≠ reference_group THEN
            // Demographic Parity Difference
            fairness_report['demographic_parity'][group] ← 
                |positive_rate(group) - positive_rate(reference_group)|
            
            // Equalised Odds Difference
            fairness_report['equalised_odds_tpr'][group] ← 
                |recall(group) - recall(reference_group)|
            fairness_report['equalised_odds_fpr'][group] ← 
                |fpr(group) - fpr(reference_group)|
            
            // Predictive Parity
            fairness_report['predictive_parity'][group] ← 
                |precision(group) - precision(reference_group)|
        END IF
    END FOR
    
    RETURN fairness_report, group_metrics
END
─────────────────────────────────────────────────────────────
```

### 6.2 Fairness Criteria Evaluation

The decision gateway "Are fairness criteria met?" determines whether disparities fall within configurable acceptable thresholds before deployment.

```
ALGORITHM 16: Fairness Criteria Decision Gateway
─────────────────────────────────────────────────────────────
Input: fairness_report, fairness_thresholds
Output: fairness_met (Boolean), bias_areas

BEGIN
    // Define acceptable disparity thresholds
    max_disparity ← {
        'demographic_parity': 0.10,    // 10% difference allowed
        'equalised_odds_tpr': 0.10,
        'equalised_odds_fpr': 0.10,
        'predictive_parity': 0.10
    }
    
    bias_areas ← []
    
    // Check each fairness criterion
    FOR attribute in sensitive_attributes DO
        FOR criterion, threshold in max_disparity DO
            max_observed_disparity ← max(fairness_report[criterion][attribute])
            
            IF max_observed_disparity > threshold THEN
                bias_areas.append({
                    'attribute': attribute,
                    'criterion': criterion,
                    'disparity': max_observed_disparity
                })
            END IF
        END FOR
    END FOR
    
    // Decision Gateway: Are fairness criteria met?
    IF len(bias_areas) = 0 THEN
        fairness_met ← TRUE
        // Proceed to Robustness Testing (per BPMN)
    ELSE
        fairness_met ← FALSE
        // Proceed to Bias Mitigation (per BPMN)
        Log: "Fairness criteria not met. Initiating bias mitigation."
    END IF
    
    RETURN fairness_met, bias_areas
END
─────────────────────────────────────────────────────────────
```

### 6.3 Bias Mitigation Strategies

When fairness criteria are not met, the framework applies mitigation: rebalancing training data, threshold adjustment per group, or calibration adjustment. The strategy choice depends on the specific fairness criterion and acceptable accuracy-fairness tradeoff.

```
ALGORITHM 17: Bias Mitigation
─────────────────────────────────────────────────────────────
Input: model, training_data, bias_areas
Output: mitigated_model

BEGIN
    FOR bias in bias_areas DO
        attribute ← bias.attribute
        
        IF bias.criterion = "demographic_parity" THEN
            // Rebalancing strategy
            training_data ← resample_to_balance(
                training_data, 
                stratify_by = [attribute, 'target']
            )
        
        ELSE IF bias.criterion = "equalised_odds" THEN
            // Threshold adjustment per group
            FOR group in unique(training_data[attribute]) DO
                group_threshold[group] ← optimise_threshold_for_group(
                    group_data, 
                    target_metric = 'equalised_odds'
                )
            END FOR
        
        ELSE IF bias.criterion = "predictive_parity" THEN
            // Calibration adjustment
            calibrator ← train_calibrator(
                model_outputs, 
                true_labels,
                group_aware = TRUE
            )
        END IF
    END FOR
    
    // Retrain or adjust model
    IF requires_retraining THEN
        mitigated_model ← retrain_model(model, training_data)
    ELSE
        mitigated_model ← apply_post_processing(model, calibrator)
    END IF
    
    // Return to Fairness Assessment (per BPMN loop)
    RETURN mitigated_model
END
─────────────────────────────────────────────────────────────
```

---

## 7. Testing Robustness and Security

This phase evaluates model resilience to adversarial perturbations using multiple attack methods (FGSM, PGD) with varying strengths. The decision gateway "Is model robustness acceptable?" determines whether enhancement is needed.

### 7.1 Adversarial Attack Generation

The framework implements FGSM (single-step) and PGD (iterative) attacks. These gradient-based methods compute perturbations that maximally increase the loss function whilst remaining imperceptible.

```
ALGORITHM 18: Fast Gradient Sign Method (FGSM) Attack
─────────────────────────────────────────────────────────────
Input: model, image, true_label, epsilon
Output: adversarial_image, perturbation

BEGIN
    // Enable gradient computation for input
    image.requires_gradient ← TRUE
    
    // Forward pass
    prediction ← model(image)
    loss ← criterion(prediction, true_label)
    
    // Backward pass to get gradients w.r.t. input
    loss.backward()
    gradient ← image.gradient
    
    // Create perturbation using sign of gradient
    perturbation ← epsilon × sign(gradient)
    
    // Generate adversarial example
    adversarial_image ← image + perturbation
    
    // Clip to valid image range [0, 1]
    adversarial_image ← clip(adversarial_image, 0, 1)
    
    RETURN adversarial_image, perturbation
END
─────────────────────────────────────────────────────────────
```

PGD extends FGSM with multiple gradient steps and projection, finding stronger adversarial examples. It is a standard benchmark for evaluating adversarial robustness.

```
ALGORITHM 19: Projected Gradient Descent (PGD) Attack
─────────────────────────────────────────────────────────────
Input: model, image, true_label, epsilon, alpha, num_iterations
Output: adversarial_image

BEGIN
    // Initialise with random perturbation within epsilon ball
    adversarial_image ← image + uniform_random(-epsilon, epsilon)
    adversarial_image ← clip(adversarial_image, 0, 1)
    
    FOR iteration = 1 TO num_iterations DO
        adversarial_image.requires_gradient ← TRUE
        
        // Forward and backward pass
        prediction ← model(adversarial_image)
        loss ← criterion(prediction, true_label)
        loss.backward()
        
        // Take gradient step
        gradient ← adversarial_image.gradient
        adversarial_image ← adversarial_image + alpha × sign(gradient)
        
        // Project back onto epsilon ball
        perturbation ← adversarial_image - image
        perturbation ← clip(perturbation, -epsilon, epsilon)
        adversarial_image ← image + perturbation
        
        // Clip to valid range
        adversarial_image ← clip(adversarial_image, 0, 1)
    END FOR
    
    RETURN adversarial_image
END
─────────────────────────────────────────────────────────────
```

### 7.2 Robustness Evaluation

The evaluation computes attack success rate and confidence degradation across multiple epsilon values, characterising the robustness-accuracy tradeoff at different perturbation magnitudes.

```
ALGORITHM 20: Comprehensive Robustness Assessment
─────────────────────────────────────────────────────────────
Input: model, test_loader, attack_methods, epsilon_values
Output: robustness_report

BEGIN
    robustness_report ← {}
    
    FOR attack_method in attack_methods DO
        FOR epsilon in epsilon_values DO
            successful_attacks ← 0
            total_samples ← 0
            confidence_drops ← []
            
            FOR batch in test_loader DO
                images, labels ← batch
                
                // Original predictions
                original_preds ← model(images)
                original_conf ← max(softmax(original_preds))
                
                // Generate adversarial examples
                IF attack_method = "FGSM" THEN
                    adv_images ← FGSM_attack(model, images, labels, epsilon)
                ELSE IF attack_method = "PGD" THEN
                    adv_images ← PGD_attack(model, images, labels, epsilon)
                END IF
                
                // Adversarial predictions
                adv_preds ← model(adv_images)
                adv_conf ← max(softmax(adv_preds))
                
                // Count successful attacks (prediction changed)
                FOR i = 1 TO len(images) DO
                    IF argmax(original_preds[i]) ≠ argmax(adv_preds[i]) THEN
                        successful_attacks ← successful_attacks + 1
                    END IF
                    confidence_drops.append(original_conf[i] - adv_conf[i])
                END FOR
                
                total_samples ← total_samples + len(images)
            END FOR
            
            // Compute robustness metrics
            attack_success_rate ← successful_attacks / total_samples
            robustness_score ← 1 - attack_success_rate
            avg_confidence_drop ← mean(confidence_drops)
            
            robustness_report[attack_method][epsilon] ← {
                'robustness_score': robustness_score,
                'attack_success_rate': attack_success_rate,
                'avg_confidence_drop': avg_confidence_drop,
                'samples_tested': total_samples
            }
        END FOR
    END FOR
    
    RETURN robustness_report
END
─────────────────────────────────────────────────────────────
```

### 7.3 Robustness Decision Gateway and Enhancement

The decision gateway compares robustness scores against minimum thresholds. Models failing requirements proceed to adversarial training for enhancement.

```
ALGORITHM 21: Robustness Acceptability Decision
─────────────────────────────────────────────────────────────
Input: robustness_report, acceptability_thresholds
Output: is_acceptable (Boolean), enhancement_needed

BEGIN
    // Define minimum robustness requirements
    min_robustness ← {
        'FGSM': {0.01: 0.85, 0.03: 0.70, 0.05: 0.50},
        'PGD': {0.01: 0.80, 0.03: 0.60, 0.05: 0.40}
    }
    
    failures ← []
    
    FOR attack_method in robustness_report DO
        FOR epsilon, metrics in robustness_report[attack_method] DO
            required_score ← min_robustness[attack_method][epsilon]
            actual_score ← metrics['robustness_score']
            
            IF actual_score < required_score THEN
                failures.append({
                    'attack': attack_method,
                    'epsilon': epsilon,
                    'required': required_score,
                    'actual': actual_score
                })
            END IF
        END FOR
    END FOR
    
    // Decision Gateway: Is model robustness acceptable?
    IF len(failures) = 0 THEN
        is_acceptable ← TRUE
        // Proceed to Calibration (per BPMN)
    ELSE
        is_acceptable ← FALSE
        // Proceed to Enhancing Robustness (per BPMN)
    END IF
    
    RETURN is_acceptable, failures
END
─────────────────────────────────────────────────────────────
```

### 7.4 Robustness Enhancement via Adversarial Training

Adversarial training augments the training process with adversarial examples generated via PGD, forcing the model to learn robust representations. The adversarial ratio parameter controls the tradeoff between clean and robust accuracy.

```
ALGORITHM 22: Adversarial Training for Robustness Enhancement
─────────────────────────────────────────────────────────────
Input: model, train_loader, epsilon, adversarial_ratio
Output: robust_model

BEGIN
    FOR epoch = 1 TO max_epochs DO
        FOR batch in train_loader DO
            images, labels ← batch
            
            // Decide whether to use adversarial examples
            IF random() < adversarial_ratio THEN
                // Generate adversarial examples for training
                adv_images ← PGD_attack(
                    model, images, labels, 
                    epsilon = epsilon,
                    num_iterations = 7
                )
                training_images ← adv_images
            ELSE
                training_images ← images
            END IF
            
            // Standard training step
            optimiser.zero_grad()
            predictions ← model(training_images)
            loss ← criterion(predictions, labels)
            loss.backward()
            optimiser.step()
        END FOR
        
        // Evaluate both clean and robust accuracy
        clean_accuracy ← evaluate(model, val_loader_clean)
        robust_accuracy ← evaluate(model, val_loader_adversarial)
        
        Log: "Epoch {epoch}: Clean={clean_accuracy}, Robust={robust_accuracy}"
    END FOR
    
    // Return to Robustness Testing (per BPMN loop)
    RETURN model
END
─────────────────────────────────────────────────────────────
```

---

## 8. Calibration and Reliability Checking

Calibration ensures predicted probabilities accurately reflect true outcome likelihoods—critical for clinical deployment. Post-hoc calibration methods address neural networks' tendency to produce overconfident predictions.

### 8.1 Calibration Assessment

Calibration is assessed via reliability diagrams and Expected Calibration Error (ECE), which measures the alignment between predicted probabilities and observed outcome frequencies.

```
ALGORITHM 23: Calibration Evaluation
─────────────────────────────────────────────────────────────
Input: predicted_probabilities, true_labels, n_bins
Output: calibration_metrics, reliability_diagram_data

BEGIN
    // Bin predictions into probability ranges
    bin_edges ← linspace(0, 1, n_bins + 1)
    bin_indices ← digitize(predicted_probabilities, bin_edges)
    
    calibration_data ← []
    
    FOR bin_id = 1 TO n_bins DO
        bin_mask ← (bin_indices = bin_id)
        
        IF count(bin_mask) > 0 THEN
            // Mean predicted probability in bin
            mean_predicted ← mean(predicted_probabilities[bin_mask])
            
            // Actual positive rate in bin
            actual_positive_rate ← mean(true_labels[bin_mask])
            
            // Bin size for weighting
            bin_size ← count(bin_mask)
            
            calibration_data.append({
                'bin': bin_id,
                'mean_predicted': mean_predicted,
                'actual_rate': actual_positive_rate,
                'size': bin_size
            })
        END IF
    END FOR
    
    // Expected Calibration Error (ECE)
    ECE ← 0
    total_samples ← len(predicted_probabilities)
    
    FOR bin in calibration_data DO
        ECE ← ECE + (bin.size / total_samples) × 
              |bin.actual_rate - bin.mean_predicted|
    END FOR
    
    // Maximum Calibration Error (MCE)
    MCE ← max(|bin.actual_rate - bin.mean_predicted| FOR bin in calibration_data)
    
    // Brier Score (proper scoring rule)
    Brier_score ← mean((predicted_probabilities - true_labels)²)
    
    calibration_metrics ← {
        'ECE': ECE,
        'MCE': MCE,
        'Brier_score': Brier_score
    }
    
    RETURN calibration_metrics, calibration_data
END
─────────────────────────────────────────────────────────────
```

### 8.2 Probability Calibration

Temperature scaling learns a single parameter to scale logits, optimised on validation data to minimise negative log-likelihood. This preserves ranking ability (AUC) whilst improving calibration.

```
ALGORITHM 24: Platt Scaling Calibration
─────────────────────────────────────────────────────────────
Input: model, validation_loader
Output: calibrated_model

BEGIN
    // Collect model outputs on validation set
    logits ← []
    labels ← []
    
    FOR batch in validation_loader DO
        images, batch_labels ← batch
        batch_logits ← model.get_logits(images)  // Before sigmoid
        logits.extend(batch_logits)
        labels.extend(batch_labels)
    END FOR
    
    // Fit temperature scaling parameter
    temperature ← Parameter(initial_value = 1.0)
    
    // Optimise temperature using NLL loss
    FOR iteration = 1 TO max_iterations DO
        scaled_logits ← logits / temperature
        calibrated_probs ← sigmoid(scaled_logits)
        
        loss ← negative_log_likelihood(labels, calibrated_probs)
        
        // Update temperature
        gradient ← compute_gradient(loss, temperature)
        temperature ← temperature - learning_rate × gradient
    END FOR
    
    Log: "Optimal temperature: {temperature}"
    
    // Wrap model with calibration layer
    calibrated_model ← CalibratedModel(model, temperature)
    
    RETURN calibrated_model
END
─────────────────────────────────────────────────────────────
```

### 8.3 Reliability Assessment

Reliability assessment evaluates the confidence-accuracy relationship, enabling selective prediction where low-confidence cases are flagged for human review.

```
ALGORITHM 25: Clinical Reliability Assessment
─────────────────────────────────────────────────────────────
Input: calibrated_model, test_loader, confidence_thresholds
Output: reliability_report

BEGIN
    reliability_report ← {}
    
    // Evaluate at different confidence levels
    FOR threshold in confidence_thresholds DO
        high_confidence_mask ← (max_probability > threshold)
        
        // Accuracy on high-confidence predictions
        confident_accuracy ← accuracy(
            predictions[high_confidence_mask],
            labels[high_confidence_mask]
        )
        
        // Coverage (proportion of samples above threshold)
        coverage ← count(high_confidence_mask) / total_samples
        
        reliability_report[threshold] ← {
            'accuracy': confident_accuracy,
            'coverage': coverage,
            'samples': count(high_confidence_mask)
        }
    END FOR
    
    // Selective prediction analysis
    // (reject low-confidence predictions for human review)
    FOR rejection_rate in [0.1, 0.2, 0.3] DO
        confidence_threshold ← percentile(max_probabilities, rejection_rate × 100)
        accepted_mask ← (max_probability > confidence_threshold)
        
        accepted_accuracy ← accuracy(predictions[accepted_mask], labels[accepted_mask])
        
        reliability_report['selective'][rejection_rate] ← {
            'threshold': confidence_threshold,
            'accuracy_on_accepted': accepted_accuracy,
            'rejection_rate': rejection_rate
        }
    END FOR
    
    RETURN reliability_report
END
─────────────────────────────────────────────────────────────
```

---

## 9. Documenting and Reporting

The final phase generates documentation for regulatory review, clinical deployment, and reproducibility. The framework produces machine-readable JSON, human-readable summaries, and visualisations.

### 9.1 V&V Report Generation

The V&V report consolidates all validation results covering data quality, model development, performance, fairness, robustness, and calibration.

```
ALGORITHM 26: Comprehensive V&V Report Generation
─────────────────────────────────────────────────────────────
Input: all_pipeline_results
Output: V&V_Report (structured document)

BEGIN
    report ← initialise_report_structure()
    
    // Section 1: Executive Summary
    report.executive_summary ← {
        'model_purpose': "Melanoma classification from dermoscopic images",
        'dataset': dataset_summary,
        'final_performance': {
            'AUC': final_metrics.roc_auc,
            'Sensitivity': final_metrics.recall,
            'Specificity': final_metrics.specificity
        },
        'v&v_outcome': overall_pass_fail_status
    }
    
    // Section 2: Data Quality Report
    report.data_quality ← {
        'validation_results': data_validation_results,
        'eda_findings': eda_summary,
        'data_issues_addressed': resolved_issues_list
    }
    
    // Section 3: Model Development
    report.model_development ← {
        'architecture': model_architecture_description,
        'training_configuration': hyperparameters,
        'training_history': training_curves,
        'cross_validation_results': cv_results (if applicable)
    }
    
    // Section 4: Performance Evaluation
    report.performance ← {
        'test_set_metrics': comprehensive_metrics,
        'confusion_matrix': confusion_matrix_data,
        'roc_curve': roc_curve_data,
        'precision_recall_curve': pr_curve_data,
        'threshold_analysis': threshold_optimisation_results
    }
    
    // Section 5: Fairness Assessment
    report.fairness ← {
        'demographic_analysis': fairness_report,
        'subgroup_performance': group_metrics,
        'bias_mitigation_applied': mitigation_actions,
        'fairness_criteria_status': criteria_pass_fail
    }
    
    // Section 6: Robustness and Security
    report.robustness ← {
        'adversarial_testing_results': robustness_report,
        'attack_methods_tested': attack_methods_list,
        'robustness_enhancement': enhancement_actions,
        'security_assessment': security_status
    }
    
    // Section 7: Calibration and Reliability
    report.calibration ← {
        'calibration_metrics': calibration_metrics,
        'reliability_diagram': reliability_data,
        'confidence_analysis': confidence_reliability
    }
    
    // Section 8: Limitations and Recommendations
    report.limitations ← {
        'known_limitations': identified_limitations,
        'deployment_recommendations': deployment_guidelines,
        'monitoring_requirements': post_deployment_monitoring
    }
    
    // Generate outputs
    Export report as JSON (machine-readable)
    Export report as PDF (human-readable)
    Export visualisations as PNG files
    
    RETURN report
END
─────────────────────────────────────────────────────────────
```

### 9.2 Metrics Summary Export

The metrics summary provides a condensed view of all quantitative validation results with pass/fail status for each criterion, facilitating integration with MLOps infrastructure.

```
ALGORITHM 27: Metrics Summary Generation
─────────────────────────────────────────────────────────────
Input: all_computed_metrics
Output: metrics_summary.json, metrics_summary.txt

BEGIN
    summary ← {
        'timestamp': current_datetime(),
        'model_version': MODEL_VERSION,
        'dataset_version': DATASET_VERSION,
        
        'performance_metrics': {
            'primary': {
                'roc_auc': metrics.roc_auc,
                'accuracy': metrics.accuracy,
                'f1_score': metrics.f1_score
            },
            'clinical': {
                'sensitivity': metrics.recall,
                'specificity': metrics.specificity,
                'ppv': metrics.positive_predictive_value,
                'npv': metrics.negative_predictive_value
            }
        },
        
        'fairness_metrics': {
            'demographic_parity_ratio': fairness.dp_ratio,
            'equalised_odds_difference': fairness.eo_diff,
            'subgroups_evaluated': fairness.subgroups
        },
        
        'robustness_metrics': {
            'fgsm_robustness': robustness.fgsm_score,
            'pgd_robustness': robustness.pgd_score,
            'average_robustness': robustness.average
        },
        
        'calibration_metrics': {
            'expected_calibration_error': calibration.ECE,
            'brier_score': calibration.brier
        },
        
        'v&v_status': {
            'performance_criteria': PASS/FAIL,
            'fairness_criteria': PASS/FAIL,
            'robustness_criteria': PASS/FAIL,
            'calibration_criteria': PASS/FAIL,
            'overall_status': PASS/FAIL
        }
    }
    
    // Export in multiple formats
    write_json(summary, 'metrics_summary.json')
    write_formatted_text(summary, 'metrics_summary.txt')
    
    RETURN summary
END
─────────────────────────────────────────────────────────────
```

---

## 10. Conclusion

This implementation realises the comprehensive V&V framework depicted in the BPMN workflow, providing a systematic approach to developing and validating deep learning models for medical image analysis.

Key contributions include:

1. **Data Quality Assurance**: Rigorous validation using Great Expectations with multi-method outlier detection.

2. **Robust Model Development**: Transfer learning with ResNet-50, Focal Loss for class imbalance, and optional ensemble training.

3. **Comprehensive Evaluation**: Multi-faceted assessment including ROC-AUC, sensitivity, specificity, and predictive values.

4. **Fairness and Equity**: Systematic evaluation across demographic subgroups with configurable mitigation strategies.

5. **Security and Robustness**: Adversarial testing (FGSM, PGD) with adversarial training for enhancement.

6. **Calibration and Reliability**: Temperature scaling calibration with selective prediction for uncertain cases.

7. **Regulatory-Ready Documentation**: Comprehensive reports for regulatory submission and scientific reproducibility.

The modular architecture ensures adaptability to diverse medical imaging applications. This framework provides a robust foundation for developing trustworthy medical AI systems.

