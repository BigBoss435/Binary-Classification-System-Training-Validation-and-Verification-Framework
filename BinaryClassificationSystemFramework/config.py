import os
import torch

# Training Configuration
TTA_N = 4
USE_AMP = torch.cuda.is_available()
USE_CROSS_VALIDATION = False
N_FOLDS = 5

# Data Augmentation Configuration
USE_ALBUMENTATIONS = True  # Set to False to use PyTorch transforms instead

# Data Analysis Configuration
RUN_EDA = True  # Set to False to skip exploratory data analysis

# Fairness and Security Assessment Configuration
RUN_FAIRNESS_ASSESSMENT = True  # Set to False to skip fairness evaluation
RUN_ADVERSARIAL_TESTING = False  # Set to False to skip adversarial robustness testing
ADVERSARIAL_NUM_SAMPLES = 500  # Number of samples for adversarial testing (adjust based on computational resources)
ADV_TRAINING = False # Set to False to skip adversarial training
ADV_EPS = 0.03  # Perturbation magnitude for adversarial training

# Calibration and Reliability Configuration
RUN_CALIBRATION = True  # Set to False to skip calibration and reliability assessment
CALIBRATION_N_BINS = 10  # Number of bins for reliability diagram
CALIBRATION_CONFIDENCE_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]  # Confidence thresholds to evaluate
CALIBRATION_REJECTION_RATES = [0.1, 0.2, 0.3]  # Rejection rates for selective prediction

# Directory Configuration
RESULTS_DIR = "results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
CV_RESULTS_DIR = os.path.join(RESULTS_DIR, "cross_validation")
CALIBRATION_DIR = os.path.join(RESULTS_DIR, "calibration")

# Ensemble Training Configuration
USE_ENSEMBLE_TRAINING = False  # Toggle for ensemble training
ENSEMBLE_N_MODELS = 3  # Number of models in ensemble
ENSEMBLE_DIVERSITY_STRATEGY = "varied_init"  # Options: "varied_init", "different_augmentation", "dropout_variation"
ENSEMBLE_AGGREGATION = "average"  # Options: "average", "weighted", "voting"

# Training Hyperparameters
MAX_EPOCHS = 5
PATIENCE = 15
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# Data Configuration
CSV_PATH = "data/ISIC_2020_Training_GroundTruth_v2.csv"
DATA_DIR = "data/jpeg/train"
TEST_SIZE = 0.15
VAL_SIZE = 0.1765
RANDOM_STATE = 42

def create_directories():
    """Create necessary directories."""
    dirs = [RESULTS_DIR, CHECKPOINT_DIR, METRICS_DIR, CV_RESULTS_DIR, CALIBRATION_DIR]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)