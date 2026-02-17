"""
Standalone EDA Script for Melanoma Dataset

This script provides a command-line interface for performing comprehensive
exploratory data analysis on the melanoma dataset without running the full
training pipeline. It's designed for quick data exploration, quality assessment,
and generating reports for stakeholders

The script handles:
- Loading ISIC 2020 melanoma dataset from CSV
- Validating data file existence and accessibility
- Running comprehensive statistical analysis
- Generating visualizations and reports
- Providing detailed logging and progress tracking

Usage:
    python run_eda.py [csv_path] [images_dir] [output_dir]
    
Example:
    python run_eda.py data/ISIC_2020_Training_GroundTruth_v2.csv data/jpeg/train results/eda_only

Author: Deividas Kalvelis
"""

import argparse
import os
import pandas as pd
from exploratory_data_analysis import MelanomaEDA
from utils import setup_logging

def run_standalone_eda(csv_path, images_dir, output_dir):
    """
    Execute comprehensive EDA analysis on melanoma dataset.

    This function orchestrates the complete EDA workflow:
    - Sets up logging infrastructure
    - Validated input data availability
    - Loads and preprocesses the dataset
    - Executes statistical analysis and visualization generation
    
    Args:
        csv_path: Path to ISIC dataset CSV file containing metadata and labels
        images_dir: Directory containing image files
        output_dir: Output directory for EDA results

    Returns:
        Dictionary containing complete EDA results including:
        - Dataset statistics and metadata
        - Class distribution analysis
        - Feature analysis results
        - Statistical test results
    """
    # Setup logging using utils function
    logger = setup_logging(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting comprehensive EDA analysis...")
    logger.info(f"CSV Path: {csv_path}")
    logger.info(f"Images Directory: {images_dir}")
    logger.info(f"Output Directory: {output_dir}")
    
    # Load dataset
    logger.info("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Add filepath column
    df['filepath'] = df['image_name'].apply(lambda x: f"{images_dir}/{x}.jpg")
    
    logger.info(f"Dataset loaded: {len(df)} samples, {len(df.columns)} features")
    
    # Initialize EDA analyzer
    eda_analyzer = MelanomaEDA(logger)
    
    # Run comprehensive analysis
    results = eda_analyzer.run_comprehensive_eda(df, output_dir)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EDA ANALYSIS COMPLETED")
    logger.info("="*60)
    
    if "dataset_info" in results:
        info = results["dataset_info"]
        logger.info(f"Total Samples: {info.get('total_samples', 0):,}")
        logger.info(f"Features: {info.get('total_features', 0)}")
        logger.info(f"Memory Usage: {info.get('memory_usage', 0) / 1024 / 1024:.2f} MB")
    
    if "class_analysis" in results:
        class_info = results["class_analysis"]
        logger.info(f"Melanoma Cases: {class_info.get('class_counts', {}).get(1, 0):,}")
        logger.info(f"Benign Cases: {class_info.get('class_counts', {}).get(0, 0):,}")
        logger.info(f"Imbalance Ratio: {class_info.get('imbalance_ratio', 0):.1f}:1")
    
    logger.info(f"\nDetailed results saved to: {output_dir}")
    logger.info("Files generated:")
    logger.info("  - eda_report.json (detailed analysis)")
    logger.info("  - eda_summary.txt (human-readable summary)")
    logger.info("  - eda_visualizations/ (charts and plots)")
    
    return results

def main():
    """Main function for standalone EDA execution."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive EDA on melanoma dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'csv_path',
        nargs='?',
        default='data/ISIC_2020_Training_GroundTruth_v2.csv',
        help='Path to ISIC dataset CSV file'
    )
    
    parser.add_argument(
        'images_dir',
        nargs='?',
        default='data/jpeg/train',
        help='Directory containing image files'
    )
    
    parser.add_argument(
        'output_dir',
        nargs='?',
        default='results/eda_analysis',
        help='Output directory for EDA results'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return 1
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return 1
    
    # Run EDA
    try:
        run_standalone_eda(args.csv_path, args.images_dir, args.output_dir)
        return 0
    except Exception as e:
        print(f"Error during EDA analysis: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
