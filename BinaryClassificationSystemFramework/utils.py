import os
import logging
import psutil
import torch
from datetime import datetime

def setup_logging(results_dir: str) -> logging.Logger:
    """Setup comprehensive logging with file and console output.
    
    Creates a timestamped log file and configures both file and console logging
    to track training progress, errors, and system metrics.

    Args:
        results_dir: Directory to save log files

    Returns:
        logging.logger: Configured logger instance
    """
    # Create a timestamped log file
    log_file = os.path.join(results_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Clear any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,  # Log INFO level and above (INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Timestamp, logger name, level, message
        handlers=[
            logging.FileHandler(log_file),  # Write to file
            logging.StreamHandler()  # Write to console
        ]
    )
    logger = logging.getLogger(__name__)  # Get logger for this module
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def log_system_stats(logger):
    """
    Log comprehensive system resource usage for monitoring.
    
    Monitors CPU, RAM, disk usage, and GPU memory to identify potential
    bottlenecks or resource constraints during training.
    
    Args:
        logger: Logger instance for output
    """
    try:
        # CPU and RAM usage monitoring
        cpu_percent = psutil.cpu_percent(interval=1)  # 1-second interval for accuracy
        memory = psutil.virtual_memory()  # Get memory information
        ram_percent = memory.percent  # Percentage of RAM used
        ram_used_gb = memory.used / (1024**3)  # Convert bytes to GB
        ram_total_gb = memory.total / (1024**3)  # Total RAM in GB
        
        # Disk usage monitoring for storage space
        disk = psutil.disk_usage('.')  # Current directory disk usage
        disk_percent = (disk.used / disk.total) * 100  # Percentage of disk used
        disk_free_gb = disk.free / (1024**3)  # Free disk space in GB
        
        # GPU memory usage monitoring (if CUDA is available)
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # Currently allocated GPU memory (GB)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # Reserved GPU memory (GB)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Total GPU memory (GB)
            gpu_utilization = (gpu_memory_allocated / gpu_memory_total) * 100  # GPU memory usage percentage
            
            gpu_info = (f"GPU Memory: {gpu_memory_allocated:.2f}GB/{gpu_memory_total:.2f}GB "
                       f"({gpu_utilization:.1f}%) | GPU Reserved: {gpu_memory_reserved:.2f}GB")
        
        # Log comprehensive system stats in one line
        logger.info(f"System Stats - CPU: {cpu_percent:.1f}% | "
                   f"RAM: {ram_used_gb:.2f}GB/{ram_total_gb:.2f}GB ({ram_percent:.1f}%) | "
                   f"Disk Free: {disk_free_gb:.1f}GB ({100-disk_percent:.1f}%) | {gpu_info}")
        
        # Warning thresholds for resource monitoring
        if ram_percent > 90:
            logger.warning(f"High RAM usage: {ram_percent:.1f}%")
        if gpu_info and gpu_utilization > 90:
            logger.warning(f"High GPU memory usage: {gpu_utilization:.1f}%")
        if cpu_percent > 90:
            logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")

def log_model_info(model, logger):
    """
    Log detailed model architecture information.
    
    Provides insights into model complexity, memory usage, and parameter counts
    for understanding computational requirements.
    
    Args:
        model: PyTorch model to analyze
        logger: Logger instance for output
    """
    # Count total parameters (all weights and biases)
    total_params = sum(p.numel() for p in model.parameters())
    # Count trainable parameters (parameters that will be updated during training)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Count frozen parameters (pretrained weights that won't be updated)
    frozen_params = total_params - trainable_params
    
    # Calculate model size in memory (MB)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    
    # Log comprehensive model information
    logger.info(f"Model Info - Total params: {total_params:,} | "
               f"Trainable: {trainable_params:,} | "
               f"Frozen: {frozen_params:,} | "
               f"Model size: {model_size_mb:.2f}MB")

def get_device():
    """Get the appropriate device (CUDA or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print(torch.version.cuda)
        print(torch.cuda.get_device_name(0))
    return device