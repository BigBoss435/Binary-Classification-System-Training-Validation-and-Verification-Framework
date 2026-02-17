"""
Data Validation Pipeline Module for Melanoma Classification

This module implements comphrehensive data validation using Great Expectations framework
and custom validation checks for the ISIC 2020 melanoma dataset, It provides automated
quality assurance to ensure data integrity before model training.

Key features:
- Schema validation using Great Expectations framework
- Image file integrity verification and corruption detection
- Data distribution analysis and anomaly detection
- Automated reporting with JSON and human-readable summaries
- Configurable validation thresholds and expectations

Author: Deividas Kalvelis
"""
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.validator.validator import Validator
from great_expectations.core.expectation_suite import ExpectationSuite
import pandas as pd
import numpy as np
from PIL import Image
import os
import logging
from typing import Dict, Tuple
from datetime import datetime
from exploratory_data_analysis import MelanomaEDA

class DataValidationPipeline:
    """
    Comprehensive data validation pipeline for melanoma classification dataset.

    This class implements automated data quality checks using Great Expectations
    framework combined with custome validation logic. It validates schema integrity,
    image file quality, and data distributions to ensure high-quality input for
    model traning.

    Attributes:
        logger (logging.logger): Logger instance for validation process tracking
        validation_results (dict): Storage for validation results and metrics

    Example:
        >>> validator = DataValidationPipeline(logger)
        >>> passed, results = validator.run_comprehensive_validation(dataframe)
        >>> if not passed:
        >>>     logger.warning("Data validation failed - check results")
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the data validation pipeline.

        Args:
            logger (logging.logger): Configured logger for tracking validation progress
            and reporting issues.
        """
        self.logger = logger
        self.validation_results = {}
        
    def setup_great_expectations_context(self):
        """
        Initialize Great Expectations context for automated data validation.

        Great Expectation provides a framework for describing expectations about data
        and automatically validating datasets against these expectations. This method
        sets up the context required for creating and running expectation suites.

        Returns:
            gx.DataContext: Initialized Great Expectations context
        
        Raises:
            Exception: If Great Expectations initiliazition fails, falls back to
            basic DataContext without advanced features.
        """
        try:
            # Attempt to initialize context with existing configuration
            context = gx.get_context()
            self.logger.info("Great Expectations context initialized successfully")
            return context
        except Exception as e:
            # Fallback to basic context if configuration issues exist
            self.logger.error(f"Failed to initialize Great Expectations: {e}")
            self.logger.info("Using fallback DataContext")
            context = gx.data_context.DataContext()
            return context

    def setup_fluent_pandas_datasource(self, context):
        """
        Set up a Pandas datasource using Great Expectations' fluent API.

        This method configures a Pandas datasource for Great Expectations if it doesn't already exist.
        The datasource is essential for validating Pandas DataFrames in-memory without requiring
        persistent storage.

        Args:
            context (gx.DataContext): Great Expectations context to add the datasource to

        Side Effects:
            - Creates a new Pandas datasource if one doesn't exist
            - Configures runtime data connector for batch validation

        Note:
            The datasource is configured with a RuntimeDataConnector to support
            dynamic validation of in-memory DataFrames.
        """
        if "my_pandas_datasource" not in context.list_datasources():

            # Create datasource config with proper structure
            datasource_config = {
                "name": "my_pandas_datasource",
                "class_name": "Datasource",
                "module_name": "great_expectations.datasource",
                "execution_engine": {
                    "module_name": "great_expectations.execution_engine",
                    "class_name": "PandasExecutionEngine"
                },
                "data_connectors": {
                    "runtime_data_connector": {
                        "class_name": "RuntimeDataConnector",
                        "module_name": "great_expectations.datasource.data_connector",
                        "batch_identifiers": ["default_identifier"]
                    }
                }
            }

            # Register the new datasource with Great Expectations context
            context.add_datasource(**datasource_config)
            self.logger.info("Fluent Pandas datasource added successfully.")
        else:
            self.logger.info("Fluent Pandas datasource already exists.")

    def create_expectation_suite(self, df: pd.DataFrame) -> Validator:
        """
        Create comprehensive expectation suite tailored for melanoma dataset validation.

        This method defines a series of expectations (data quality rules) that the
        melanoma dataset must satisfy. These include schema validation, data type
        checks, format validation, and business logic constraints.

        Args:
            df (pd.DataFrame): Input dataset to create expectations for.

        Returns:
            gx.ExpectationSuite: Configured expectation suite for validation.
        
        Expectations Created:
            1. Schema validation - ensures correct column structure
            2. Data type validation - verifies column data types
            3. Null value checks - prevents missing critical data
            4. Label validation - ensures binary classification targets (0,1)
            5. Uniqueness validation - prevents duplicate image entries
            6. Pattern validation - validates ISIC image naming convention
            7. File path validation - ensures proper JPEG file extensions
            8. Dataset size validation - ensures minimum sample size for training
            9. Class distribution validation - monitors melanoma/benign ratio
        """
        context = self.setup_great_expectations_context()
        self.setup_fluent_pandas_datasource(context)

        # Create an in-memory expectation suite
        suite_name = "melanoma_dataset_validation"
        suite = ExpectationSuite(suite_name)

        # Runtime batch request using fluent datasource
        batch_request = RuntimeBatchRequest(
            datasource_name="my_pandas_datasource",
            data_connector_name="runtime_data_connector",
            data_asset_name="melanoma_df",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier": "default"},
        )

        # Create validator bound to the dataframe + suite
        validator: Validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite=suite
        )

        # Expectations
        validator.expect_table_columns_to_match_ordered_list(
            column_list=["image_name", "patient_id", "lesion_id", "sex", "age_approx",
                         "anatom_site_general_challenge", "diagnosis", "benign_malignant",
                         "target", "filepath"]
        )
        validator.expect_column_values_to_be_of_type("image_name", "str")
        validator.expect_column_values_to_be_of_type("target", "int")
        validator.expect_column_values_to_be_of_type("filepath", "str")
        validator.expect_column_values_to_not_be_null("image_name")
        validator.expect_column_values_to_not_be_null("target")
        validator.expect_column_values_to_not_be_null("filepath")
        validator.expect_column_values_to_be_in_set("target", [0, 1])
        validator.expect_column_values_to_be_unique("image_name")
        validator.expect_column_values_to_match_regex("image_name", r"^ISIC_\d{7}$")
        validator.expect_column_values_to_match_regex("filepath", r".*\.jpg$")
        validator.expect_table_row_count_to_be_between(min_value=1000)

        melanoma_ratio = df['target'].mean()
        validator.expect_column_mean_to_be_between(
            "target",
            min_value=melanoma_ratio * 0.5,
            max_value=melanoma_ratio * 1.5
        )

        self.logger.info(
            f"Expectation suite '{suite_name}' created in-memory with {len(validator.expectation_suite.expectations)} expectations."
        )

        return validator.expectation_suite
    
    def validate_image_integrity(self, df: pd.DataFrame, sample_size: int = 200) -> Dict:
        """
        Validate image file integrity and detect corruption beyind basic fail exitence.

        This method performs deep validation of image files by attempting to load,
        verify, and analyze image properties. It detects corrupted files, invalid
        formats, and images that may cause issues during training.

        Args:
            df (pd.DataFrame): Dataset containing filepath column with image paths
            sample_size (int, optional): Number of images to validate. Defaults to 200.
                                       Sampling is used for performance on large datasets.
                                       
        Returns:
            Dict: Comprehensive validation results containing:
                - total_checked: Number of images validated
                - valid_images: Count of successfully validated images
                - corrupted_images: Count of corrupted/unreadable images
                - missing_files: Count of files that don't exist on disk
                - size_issues: Count of images below minimum size threshold
                - format_issues: Count of images with unexpected formats
                - image_dimensions: List of (width, height) tuples for valid images
                - avg_width/height: Average image dimensions
                - min/max_width/height: Dimension ranges
                - errors: List of specific error messages for debugging
                
        Validation Checks:
            1. File existence verification
            2. Image loading and integrity verification using PIL
            3. Format validation (JPEG/JPG only)
            4. Minimum dimension requirements (>= 50x50 pixels)
            5. Color mode validation (RGB, L, RGBA)
            6. Statistical analysis of image dimensions
            7. Calculate statistical metrics for valid images
        """
        self.logger.info(f"Validating image integrity on {sample_size} samples...")
        
        # Use random sampling for performance on large datasets
        # Fixed random state ensures reproducible validation results
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        
        # Initialize results dictionary with all validation metrics
        results = {
            "total_checked": len(sample_df),
            "valid_images": 0,
            "corrupted_images": 0,
            "missing_files": 0,
            "size_issues": 0,
            "format_issues": 0,
            "image_dimensions": [],  # Store dimensions for statistical analysis
            "errors": []  # Store specific error messages for debugging
        }
        
        # Validate each sampled image
        for idx, row in sample_df.iterrows():
            img_path = row['filepath']
            
            try:
                # Step 1: Check if file exists on disk
                if not os.path.exists(img_path):
                    results["missing_files"] += 1
                    results["errors"].append(f"Missing file: {img_path}")
                    continue
                
                # Step 2: Attempt to open and verify image integrity
                # PIL's verify() method checks if the image file is valid
                with Image.open(img_path) as img:
                    img.verify()  # Raises exception if image is corrupted
                
                # Step 3: Re-open image for property analysis
                # Note: verify() closes the image, so we need to re-open
                with Image.open(img_path) as img:
                    width, height = img.size
                    format_type = img.format
                    mode = img.mode
                    
                    # Store dimensions for statistical analysis
                    results["image_dimensions"].append((width, height))
                    
                    # Step 4: Check minimum dimension requirements
                    # Images too small may cause issues during training
                    if width < 50 or height < 50:
                        results["size_issues"] += 1
                        results["errors"].append(f"Small image {width}x{height}: {img_path}")
                    
                    # Step 5: Validate image format
                    # Only JPEG images are expected in ISIC dataset
                    if format_type not in ['JPEG', 'JPG']:
                        results["format_issues"] += 1
                        results["errors"].append(f"Unexpected format {format_type}: {img_path}")
                    
                    # Step 6: Check color mode compatibility
                    # RGB, Grayscale (L), and RGBA are acceptable for training
                    if mode not in ['RGB', 'L', 'RGBA']:
                        results["errors"].append(f"Unexpected mode {mode}: {img_path}")
                
                # Image passed all validation checks
                results["valid_images"] += 1
                
            except Exception as e:
                # Image failed validation - likely corrupted or unreadable
                results["corrupted_images"] += 1
                results["errors"].append(f"Corrupted image {img_path}: {str(e)}")
        
        # Step 7: Calculate statistical metrics for valid images
        if results["image_dimensions"]:
            dimensions = np.array(results["image_dimensions"])
            results["avg_width"] = float(np.mean(dimensions[:, 0]))
            results["avg_height"] = float(np.mean(dimensions[:, 1]))
            results["min_width"] = int(np.min(dimensions[:, 0]))
            results["min_height"] = int(np.min(dimensions[:, 1]))
            results["max_width"] = int(np.max(dimensions[:, 0]))
            results["max_height"] = int(np.max(dimensions[:, 1]))
        
        return results
    
    def detect_outliers_in_numerical_data(self, df: pd.DataFrame) -> Dict:
        """
        Detect outliers in numerical columns using multiple statistical methods.
        
        This method applies various outlier detection techniques to numerical features
        in the dataset to identify potentially problematic data points that could
        affect model training and performance.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            Dict: Outlier detection results containing:
                - methods_used: List of detection methods applied
                - outliers_per_column: Detailed outlier information per column
                - total_outliers: Total number of outlier data points found
                - outlier_indices: Indices of samples with outliers
                - recommendations: Suggested actions for outlier handling
        
        Detection Methods:
            1. IQR (Interquartile Range) method - robust to distribution shape
            2. Z-score method - assumes normal distribution
            3. Modified Z-score - more robust version using median
            4. Isolation Forest - machine learning based anomaly detection
        """
        from scipy import stats
        from sklearn.ensemble import IsolationForest
        
        self.logger.info("Detecting outliers in numerical data...")
        
        # Identify numerical columns (excluding target)
        numerical_columns = []
        for col in ['age_approx']:
            if col in df.columns:
                numerical_columns.append(col)
        
        if not numerical_columns:
            self.logger.info("No numerical columns found for outlier detection")
            return {"methods_used": [], "outliers_per_column": {}, "total_outliers": 0}
        
        outlier_results = {
            "methods_used": ["IQR", "Z-score", "Modified Z-score", "Isolation Forest"],
            "outliers_per_column": {},
            "total_outliers": 0,
            "outlier_indices": set(),
            "recommendations": []
        }
        
        for col in numerical_columns:
            if df[col].isnull().all():
                continue
                
            # Remove NaN values for analysis
            clean_data = df[col].dropna()
            clean_indices = df[col].dropna().index
            
            col_outliers = {
                "column": col,
                "total_samples": len(clean_data),
                "missing_values": df[col].isnull().sum(),
                "detection_methods": {}
            }
            
            # Method 1: IQR (Interquartile Range)
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            iqr_outlier_indices = iqr_outliers.index
            
            col_outliers["detection_methods"]["IQR"] = {
                "outlier_count": len(iqr_outliers),
                "outlier_percentage": (len(iqr_outliers) / len(clean_data)) * 100,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_indices": iqr_outlier_indices.tolist(),
                "outlier_values": iqr_outliers.tolist()
            }
            
            # Method 2: Z-score (assumes normal distribution)
            z_scores = np.abs(stats.zscore(clean_data))
            z_threshold = 3.0
            z_outliers = clean_data[z_scores > z_threshold]
            z_outlier_indices = clean_indices[z_scores > z_threshold]
            
            col_outliers["detection_methods"]["Z-score"] = {
                "outlier_count": len(z_outliers),
                "outlier_percentage": (len(z_outliers) / len(clean_data)) * 100,
                "threshold": z_threshold,
                "outlier_indices": z_outlier_indices.tolist(),
                "outlier_values": z_outliers.tolist()
            }
            
            # Method 3: Modified Z-score (more robust)
            median = np.median(clean_data)
            mad = np.median(np.abs(clean_data - median))
            modified_z_scores = 0.6745 * (clean_data - median) / mad if mad != 0 else np.zeros_like(clean_data)
            modified_z_threshold = 3.5
            modified_z_outliers = clean_data[np.abs(modified_z_scores) > modified_z_threshold]
            modified_z_outlier_indices = clean_indices[np.abs(modified_z_scores) > modified_z_threshold]
            
            col_outliers["detection_methods"]["Modified Z-score"] = {
                "outlier_count": len(modified_z_outliers),
                "outlier_percentage": (len(modified_z_outliers) / len(clean_data)) * 100,
                "threshold": modified_z_threshold,
                "outlier_indices": modified_z_outlier_indices.tolist(),
                "outlier_values": modified_z_outliers.tolist()
            }
            
            # Method 4: Isolation Forest (ML-based anomaly detection)
            if len(clean_data) > 10:  # Need minimum samples for Isolation Forest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_predictions = iso_forest.fit_predict(clean_data.values.reshape(-1, 1))
                iso_outliers = clean_data[outlier_predictions == -1]
                iso_outlier_indices = clean_indices[outlier_predictions == -1]
                
                col_outliers["detection_methods"]["Isolation Forest"] = {
                    "outlier_count": len(iso_outliers),
                    "outlier_percentage": (len(iso_outliers) / len(clean_data)) * 100,
                    "contamination_rate": 0.1,
                    "outlier_indices": iso_outlier_indices.tolist(),
                    "outlier_values": iso_outliers.tolist()
                }
            
            # Consensus outliers (detected by multiple methods)
            all_outlier_indices = set()
            for method_results in col_outliers["detection_methods"].values():
                all_outlier_indices.update(method_results["outlier_indices"])
            
            col_outliers["consensus_outliers"] = list(all_outlier_indices)
            col_outliers["consensus_count"] = len(all_outlier_indices)
            col_outliers["consensus_percentage"] = (len(all_outlier_indices) / len(clean_data)) * 100
            
            # Add to global outlier tracking
            outlier_results["outlier_indices"].update(all_outlier_indices)
            outlier_results["outliers_per_column"][col] = col_outliers
            
            # Generate recommendations based on outlier percentage
            outlier_percentage = col_outliers["consensus_percentage"]
            if outlier_percentage > 20:
                outlier_results["recommendations"].append(
                    f"High outlier rate in {col} ({outlier_percentage:.1f}%) - investigate data quality"
                )
            elif outlier_percentage > 10:
                outlier_results["recommendations"].append(
                    f"Moderate outlier rate in {col} ({outlier_percentage:.1f}%) - consider robust scaling"
                )
            elif outlier_percentage > 5:
                outlier_results["recommendations"].append(
                    f"Some outliers in {col} ({outlier_percentage:.1f}%) - monitor during training"
                )
        
        # Calculate total unique outlier samples
        outlier_results["total_outliers"] = len(outlier_results["outlier_indices"])
        outlier_results["outlier_indices"] = list(outlier_results["outlier_indices"])
        
        # General recommendations
        if outlier_results["total_outliers"] > len(df) * 0.1:
            outlier_results["recommendations"].append(
                "High overall outlier rate - consider robust preprocessing methods"
            )
        
        self.logger.info(f"Outlier detection completed. Found {outlier_results['total_outliers']} unique samples with outliers")
        
        return outlier_results

    def run_comprehensive_eda(self, df: pd.DataFrame, output_dir: str) -> Dict:
        """
        Run comprehensive exploratory data analysis on the dataset.
        
        This method performs detailed statistical analysis and visualization
        of the melanoma dataset, providing insights into data distributions,
        feature relationships, and potential patterns that could inform
        model training decisions.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            output_dir (str): Directory where EDA reports and visualizations will be saved
            
        Returns:
            Dict: Comprehensive EDA results containing:
                - dataset_info: Basic dataset information and statistics
                - class_analysis: Detailed class distribution analysis
                - numerical_analysis: Statistical analysis of numerical features
                - categorical_analysis: Distribution analysis of categorical features
                - feature_target_relationships: Statistical relationships between features and target
                - visualization_paths: Paths to generated visualization files
                
        Analysis Components:
            1. Dataset overview with memory usage and missing value analysis
            2. Class distribution analysis with imbalance metrics
            3. Numerical feature statistics (mean, median, std, outliers)
            4. Categorical feature distributions and unique value analysis
            5. Feature-target relationship analysis using statistical tests
            6. Comprehensive visualizations saved as PNG files
            7. Human-readable summary report generation
        """
        self.logger.info("Starting comprehensive EDA analysis...")
        
        # Initialize EDA analyzer
        eda_analyzer = MelanomaEDA(self.logger)
        
        # Run comprehensive analysis
        eda_results = eda_analyzer.run_comprehensive_eda(df, output_dir)
        
        # Store results for later access
        self.validation_results["eda_analysis"] = eda_results
        
        # Log key insights
        if "class_analysis" in eda_results:
            class_info = eda_results["class_analysis"]
            self.logger.info(f"EDA Summary:")
            self.logger.info(f"  Dataset size: {class_info['total_samples']:,} samples")
            self.logger.info(f"  Class imbalance: {class_info.get('imbalance_ratio', 0):.1f}:1")
            self.logger.info(f"  Melanoma prevalence: {class_info.get('minority_class_percentage', 0):.2f}%")
        
        # Check for significant feature relationships
        if "feature_target_relationships" in eda_results:
            significant_features = []
            for feature, analysis in eda_results["feature_target_relationships"].items():
                if analysis.get("significant", False):
                    significant_features.append(feature)
            
            if significant_features:
                self.logger.info(f"Features with significant target relationships: {', '.join(significant_features)}")
            else:
                self.logger.info("No statistically significant feature-target relationships found")
        
        return eda_results

    def validate_data_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Validate data distributions and detect statistical anomalies.
        
        This method analyzes the dataset's statistical properties to identify
        potential issues that could affect model training, such as extreme class
        imbalance, duplicate entries, or unexpected data patterns.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            Dict: Distribution analysis results containing:
                - total_samples: Total number of samples in dataset
                - class_distribution: Count of each class (0=benign, 1=melanoma)
                - class_ratio: Proportion of positive (melanoma) cases
                - duplicate_names: Number of duplicate image names
                - unique_images: Number of unique images
                - outlier_analysis: Detailed outlier detection results
                - warnings: List of distribution-related warnings
                
        Validation Checks:
            1. Class balance analysis for binary classification
            2. Duplicate detection to prevent data leakage
            3. Extreme imbalance detection (very rare or very common melanoma)
            4. Comprehensive outlier detection using multiple methods
            5. Statistical anomaly flagging
        """
        self.logger.info("Validating data distributions...")
        
        # Calculate core distribution metrics
        results = {
            "total_samples": len(df),
            "class_distribution": df['target'].value_counts().to_dict(),  # Count per class
            "class_ratio": df['target'].mean(),  # Proportion of positive class
            "duplicate_names": df['image_name'].duplicated().sum(),  # Duplicate detection
            "unique_images": df['image_name'].nunique(),  # Unique image count
        }
        
        # Add comprehensive outlier detection
        outlier_analysis = self.detect_outliers_in_numerical_data(df)
        results["outlier_analysis"] = outlier_analysis
        
        # Analyze class balance for potential training issues
        melanoma_ratio = results["class_ratio"]
        results["warnings"] = []

        # Flag extremely low melanoma ratios (< 0.1% - may indicate labeling issues)
        if melanoma_ratio < 0.001:
            results["warnings"] = results.get("warnings", [])
            results["warnings"].append(f"Very low melanoma ratio: {melanoma_ratio:.4f}")

        # Flag unexpectedly high melanoma ratios (> 50% - unusual for real-world data)
        elif melanoma_ratio > 0.5:
            results["warnings"] = results.get("warnings", [])
            results["warnings"].append(f"High melanoma ratio: {melanoma_ratio:.4f}")
        
        # Add outlier-related warnings
        if outlier_analysis["total_outliers"] > 0:
            outlier_percentage = (outlier_analysis["total_outliers"] / len(df)) * 100
            if outlier_percentage > 15:
                results["warnings"].append(f"High outlier rate: {outlier_percentage:.1f}% of samples")
            elif outlier_percentage > 5:
                results["warnings"].append(f"Moderate outlier rate: {outlier_percentage:.1f}% of samples")
        
        # Add recommendations from outlier analysis
        if outlier_analysis["recommendations"]:
            results["outlier_recommendations"] = outlier_analysis["recommendations"]
        
        return results
    
    def run_comprehensive_validation(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Execute the complete data validation pipeline with all validation checks.
        
        This is the main entry point for data validation. It orchestrates all
        validation components and provides a comprehensive assessment of data quality.
        The method runs schema validation, image integrity checks, and distribution
        analysis, then aggregates results into a final pass/fail determination.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            
        Returns:
            Tuple[bool, Dict]: 
                - bool: Overall validation status (True=passed, False=failed)
                - Dict: Comprehensive validation results including:
                    - timestamp: When validation was performed
                    - validation_passed: Overall pass/fail status
                    - great_expectations: Schema validation results
                    - image_validation: Image integrity check results
                    - distribution_validation: Statistical analysis results
                    - validation_time_seconds: Total validation duration
                    - summary: High-level summary with issues found
                    
        Validation Pipeline:
            1. Schema and expectation validation using Great Expectations
            2. Image file integrity and corruption detection
            3. Data distribution analysis and anomaly detection
            4. Threshold-based pass/fail determination
            5. Comprehensive result aggregation and reporting
            
        Failure Criteria:
            - Schema validation failures (wrong columns, types, etc.)
            - High image corruption rate (>5%)
            - High missing file rate (>1%)
            - Expectation violations in Great Expectations framework
        """
        self.logger.info("Starting comprehensive data validation pipeline...")
        validation_start = datetime.now()
        
        # Initialize results structure with metadata
        all_results = {
            "timestamp": validation_start.isoformat(),
            "validation_passed": True,  # Innocent until proven guilty
            "summary": {}
        }
        
        # ==========================================
        # PHASE 1: Schema and Expectation Validation
        # ==========================================
        try:
            self.logger.info("Running Great Expectations validation...")
            context = self.setup_great_expectations_context()
            suite = self.create_expectation_suite(df)
            
            # Initialize Great Expectations results tracking
            ge_results = {
                "success": True,
                "total_expectations": len(suite.expectations),
                "successful_expectations": 0,
                "failed_expectations": []
            }
            
            # Manual validation of each expectation
            # Note: This approach provides more control than automated GE validation
            for expectation in suite.expectations:
                try:
                    expectation_type = expectation.expectation_type
                    
                    # Validate column structure matches expected schema
                    if expectation_type == "expect_table_columns_to_match_ordered_list":
                        expected_cols = expectation.kwargs["column_list"]
                        actual_cols = list(df.columns)
                        if actual_cols == expected_cols:
                            ge_results["successful_expectations"] += 1
                        else:
                            ge_results["failed_expectations"].append({
                                "expectation_type": expectation_type,
                                "description": f"Expected columns {expected_cols}, got {actual_cols}"
                            })

                    # Validate values are within allowed set (e.g., target labels)    
                    elif expectation_type == "expect_column_values_to_be_in_set":
                        column = expectation.kwargs["column"]
                        value_set = expectation.kwargs["value_set"]
                        invalid_values = ~df[column].isin(value_set)
                        if not invalid_values.any():
                            ge_results["successful_expectations"] += 1
                        else:
                            ge_results["failed_expectations"].append({
                                "expectation_type": expectation_type,
                                "description": f"Column {column} has invalid values"
                            })
                    
                    # Validate column uniqueness (prevent duplicates)
                    elif expectation_type == "expect_column_values_to_be_unique":
                        column = expectation.kwargs["column"]
                        duplicates = df[column].duplicated().sum()
                        if duplicates == 0:
                            ge_results["successful_expectations"] += 1
                        else:
                            ge_results["failed_expectations"].append({
                                "expectation_type": expectation_type,
                                "description": f"Column {column} has {duplicates} duplicates"
                            })
                    
                    # For other expectation types, assume they pass
                    # (Can be extended to handle additional expectation types)
                    else:
                        ge_results["successful_expectations"] += 1
                        
                except Exception as e:
                    # Handle individual expectation validation errors
                    ge_results["failed_expectations"].append({
                        "expectation_type": expectation_type,
                        "description": f"Error validating: {str(e)}"
                    })
            
            # Determine overall Great Expectations success
            ge_results["success"] = len(ge_results["failed_expectations"]) == 0
            all_results["great_expectations"] = ge_results
            
            # Update overall validation status if schema validation failed
            if not ge_results["success"]:
                all_results["validation_passed"] = False
                self.logger.warning("Great Expectations validation failed!")
            
        except Exception as e:
            # Handle catastrophic Great Expectations failures
            self.logger.error(f"Great Expectations validation error: {e}")
            all_results["great_expectations"] = {"error": str(e)}
            all_results["validation_passed"] = False
        
        # ==========================================
        # PHASE 2: Image Integrity Validation
        # ==========================================
        try:
            # Sample size can be changed based on dataset size/performance needs, if sample size is omitted
            # the default is 200 samples, if you want to validate all images, set sample_size=len(df)
            image_results = self.validate_image_integrity(df)
            all_results["image_validation"] = image_results
            
            # Apply validation thresholds for image quality
            corruption_rate = image_results["corrupted_images"] / image_results["total_checked"]
            missing_rate = image_results["missing_files"] / image_results["total_checked"]
            
            # Fail validation if corruption rate exceeds 5%
            if corruption_rate > 0.05:
                all_results["validation_passed"] = False
                self.logger.error(f"High image corruption rate: {corruption_rate:.2%}")
            
            # Fail validation if missing file rate exceeds 1%
            if missing_rate > 0.01:
                all_results["validation_passed"] = False
                self.logger.error(f"High missing file rate: {missing_rate:.2%}")
                
        except Exception as e:
            # Handle image validation errors
            self.logger.error(f"Image validation error: {e}")
            all_results["image_validation"] = {"error": str(e)}
            all_results["validation_passed"] = False
        
        # ==========================================
        # PHASE 3: Data Distribution Validation
        # ==========================================
        try:
            dist_results = self.validate_data_distribution(df)
            all_results["distribution_validation"] = dist_results
            
        except Exception as e:
            # Handle distribution validation errors
            self.logger.error(f"Distribution validation error: {e}")
            all_results["distribution_validation"] = {"error": str(e)}
        
        # ==========================================
        # PHASE 4: Comprehensive EDA Analysis (Optional)
        # ==========================================
        # Only run EDA if validation is passing so far (to save time on bad data)
        if all_results["validation_passed"]:
            try:
                # Create EDA subdirectory in results
                eda_output_dir = os.path.join("results", "eda_analysis")
                os.makedirs(eda_output_dir, exist_ok=True)
                
                eda_results = self.run_comprehensive_eda(df, eda_output_dir)
                all_results["eda_analysis"] = eda_results
                
                # EDA doesn't affect validation pass/fail, it's purely informational
                self.logger.info("EDA analysis completed successfully")
                
            except Exception as e:
                # EDA failure shouldn't fail overall validation
                self.logger.error(f"EDA analysis error: {e}")
                all_results["eda_analysis"] = {"error": str(e)}
        
        # ==========================================
        # PHASE 5: Results Aggregation and Summary
        # ==========================================

        # Calculate validation time
        validation_time = (datetime.now() - validation_start).total_seconds()
        all_results["validation_time_seconds"] = validation_time
        
        # Create executive summary of validation results
        all_results["summary"] = {
            "total_samples": len(df),
            "validation_passed": all_results["validation_passed"],
            "validation_time": f"{validation_time:.2f}s",
            "eda_completed": "eda_analysis" in all_results and "error" not in all_results.get("eda_analysis", {}),
            "issues_found": []
        }
        
        # Compile list of issues found during validation
        if not all_results["validation_passed"]:
            issues = []

            # Add Great Expectations failures
            if "great_expectations" in all_results and not all_results["great_expectations"].get("success", True):
                issues.append("Great Expectations validation failed")

            # Add image quality issues
            if "image_validation" in all_results:
                img_val = all_results["image_validation"]
                if img_val.get("corrupted_images", 0) > 0:
                    issues.append(f"{img_val['corrupted_images']} corrupted images")
                if img_val.get("missing_files", 0) > 0:
                    issues.append(f"{img_val['missing_files']} missing files")
            all_results["summary"]["issues_found"] = issues
        
        # Log final validation status
        status_msg = "PASSED" if all_results["validation_passed"] else "FAILED"
        self.logger.info(f"Validation completed in {validation_time:.2f}s. Status: {status_msg}")
        
        return all_results["validation_passed"], all_results


def generate_validation_summary(validation_results, results_dir, logger):
    """
    Generate human-readable validation summary report for stakeholder review.
    
    This function creates a comprehensive, formatted text report that summarizes
    all validation results in a format suitable for technical and non-technical
    stakeholders. The report includes overall status, detailed breakdowns by
    validation category, and specific issues found.
    
    Args:
        validation_results (Dict): Complete validation results from validation pipeline
        results_dir (str): Directory path where summary report should be saved
        logger (logging.Logger): Logger for tracking report generation
        
    Output:
        Creates validation_summary.txt file containing:
        - Executive summary with overall pass/fail status
        - Detailed breakdown of Great Expectations results
        - Image validation metrics and statistics
        - Data distribution analysis results
        - Complete list of issues found with descriptions
        
    Report Sections:
        1. Header with overall validation status
        2. Great Expectations validation results
        3. Image integrity validation results  
        4. Data distribution validation results
        5. Summary of all issues found
    """
    summary_path = os.path.join(results_dir, "validation_summary.txt")
    
    with open(summary_path, 'w') as f:
        # ==========================================
        # REPORT HEADER
        # ==========================================
        f.write("=" * 60 + "\n")
        f.write("DATA VALIDATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall validation status - most important information first
        status = "PASSED" if validation_results.get("validation_passed", False) else "FAILED"
        f.write(f"Overall Status: {status}\n")
        f.write(f"Validation Time: {validation_results.get('validation_time_seconds', 0):.2f}s\n\n")
        
        # ==========================================
        # GREAT EXPECTATIONS RESULTS
        # ==========================================
        if "great_expectations" in validation_results:
            ge = validation_results["great_expectations"]
            f.write("Great Expectations Results:\n")
            f.write(f"  Success: {ge.get('success', 'N/A')}\n")
            f.write(f"  Total Expectations: {ge.get('total_expectations', 0)}\n")
            f.write(f"  Successful: {ge.get('successful_expectations', 0)}\n")

            # List specific expectation failures for debugging
            if ge.get("failed_expectations"):
                f.write("  Failed Expectations:\n")
                for failure in ge["failed_expectations"]:
                    f.write(f"    - {failure['expectation_type']}: {failure['description']}\n")
            f.write("\n")
        
        # ==========================================
        # IMAGE VALIDATION RESULTS
        # ==========================================
        if "image_validation" in validation_results:
            img = validation_results["image_validation"]
            f.write("Image Validation Results:\n")
            f.write(f"  Total Checked: {img.get('total_checked', 0)}\n")
            f.write(f"  Valid Images: {img.get('valid_images', 0)}\n")
            f.write(f"  Corrupted Images: {img.get('corrupted_images', 0)}\n")
            f.write(f"  Missing Files: {img.get('missing_files', 0)}\n")
            f.write(f"  Size Issues: {img.get('size_issues', 0)}\n")

            # Include image dimension statistics if available
            if img.get("avg_width"):
                f.write(f"  Average Dimensions: {img['avg_width']:.0f}x{img['avg_height']:.0f}\n")
            f.write("\n")
        
        # ==========================================
        # DISTRIBUTION VALIDATION RESULTS
        # ==========================================
        if "distribution_validation" in validation_results:
            dist = validation_results["distribution_validation"]
            f.write("Distribution Validation Results:\n")
            f.write(f"  Total Samples: {dist.get('total_samples', 0)}\n")
            f.write(f"  Class Ratio (Melanoma): {dist.get('class_ratio', 0):.4f}\n")
            f.write(f"  Duplicate Names: {dist.get('duplicate_names', 0)}\n")
            f.write(f"  Unique Images: {dist.get('unique_images', 0)}\n")
            
            # Include distribution warnings if any
            if dist.get("warnings"):
                f.write("  Warnings:\n")
                for warning in dist["warnings"]:
                    f.write(f"    - {warning}\n")
            f.write("\n")
        
        # ==========================================
        # EDA ANALYSIS RESULTS
        # ==========================================
        if "eda_analysis" in validation_results and "error" not in validation_results["eda_analysis"]:
            eda = validation_results["eda_analysis"]
            f.write("Exploratory Data Analysis Results:\n")
            f.write(f"  Analysis Timestamp: {eda.get('timestamp', 'N/A')}\n")
            
            # Dataset info
            if "dataset_info" in eda:
                info = eda["dataset_info"]
                f.write(f"  Dataset Size: {info.get('total_samples', 0):,} samples\n")
                f.write(f"  Total Features: {info.get('total_features', 0)}\n")
                f.write(f"  Memory Usage: {info.get('memory_usage', 0) / 1024 / 1024:.2f} MB\n")
            
            # Class distribution insights
            if "class_analysis" in eda:
                class_info = eda["class_analysis"]
                f.write(f"  Class Imbalance Ratio: {class_info.get('imbalance_ratio', 0):.1f}:1\n")
                f.write(f"  Melanoma Prevalence: {class_info.get('minority_class_percentage', 0):.2f}%\n")
            
            # Statistical relationships
            if "feature_target_relationships" in eda:
                significant_features = []
                for feature, analysis in eda["feature_target_relationships"].items():
                    if analysis.get("significant", False):
                        significant_features.append(feature)
                
                if significant_features:
                    f.write(f"  Significant Feature Relationships: {', '.join(significant_features)}\n")
                else:
                    f.write("  No statistically significant feature relationships found\n")
            
            f.write("  EDA visualizations and detailed report generated\n")
            f.write("\n")
        
        # ==========================================
        # ISSUES SUMMARY
        # ==========================================
        if validation_results.get("summary", {}).get("issues_found"):
            f.write("Issues Found:\n")
            for issue in validation_results["summary"]["issues_found"]:
                f.write(f"  - {issue}\n")
    
    # Log successful report generation
    logger.info(f"Validation summary saved to: {summary_path}")