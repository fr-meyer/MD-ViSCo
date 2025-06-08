# Standard library imports
import logging
from typing import Any, Optional, Union

# Third-party imports
import wandb

def format_dataset_name(dataset: str) -> str:
    """Format dataset name to consistent format.
    
    Args:
        dataset: Name of the dataset (e.g., 'pulsedb', 'uci')
        
    Returns:
        Formatted dataset name (e.g., 'PulseDB', 'UCI')
    """
    dataset = dataset.lower()
    if dataset == 'pulsedb':
        return 'PulseDB'
    elif dataset == 'uci':
        return 'UCI'
    else:
        return dataset.capitalize()

def format_method_name(method: str) -> str:
    """Format method name to consistent format.
    
    Args:
        method: Name of the method (e.g., 'nabnet', 'ppg2abp')
        
    Returns:
        Formatted method name (e.g., 'NABNet', 'PPG2ABP')
    """
    if method is None:
        return 'Ours'
        
    method = method.lower()
    if method == 'nabnet':
        return 'NABNet'
    elif method == 'patchtst':
        return 'PatchTST'
    elif method == 'p2ewgan':
        return 'P2E-WGAN'
    elif method == 'ours':
        return 'Ours'
    elif '2' in method:  # Handle direction-based methods like 'ppg2abp'
        parts = method.split('2')
        return f"{parts[0].upper()}2{parts[1].upper()}"
    else:
        return method.capitalize()

def format_direction(direction: str) -> str:
    """Format direction string to consistent format.
    
    Args:
        direction: Direction string (e.g., 'PPG2ABP', 'ABP2PPG')
        
    Returns:
        Formatted direction string (e.g., 'PPG→ABP', 'ABP→PPG')
    """
    if '2' in direction:
        source, target = direction.split('2')
        return f"{source.upper()}→{target.upper()}"
    return direction

def log_bhs_style_result(
    dataset: str,
    threshold: str,
    direction: str,
    measure: str,
    method: str,
    value: float,
    seed: int,
    args: dict
) -> None:
    """Log a single BHS-style result.
    
    Args:
        dataset: Name of the dataset (e.g., 'UCI', 'PulseDB')
        threshold: Threshold value (e.g., '<=5mmHg')
        direction: Direction of translation (e.g., 'PPG→ABP')
        measure: Type of measurement (e.g., 'SBP', 'DBP', 'MAP')
        method: Name of the method used
        value: Result value
        seed: Random seed used
        args: Arguments dictionary containing wandb configuration
    """
    if wandb.run is None and args.project_name is not None:
        init_wandb(project_name=args.project_name, config=vars(args))
        
    if wandb.run is not None:
        wandb.log({
            "Dataset": format_dataset_name(dataset),
            "Threshold": threshold,
            "Direction": format_direction(direction),
            "Measure": measure,
            "Method": format_method_name(method),
            "Value": value,
            "seed": seed,
            "Task": "BHS"
        })
    else:
        logging.warning("wandb run not found - logs will not be sent to wandb")
        
    finish_wandb()

def log_aami_style_result(
    dataset: str,
    method: str,
    direction: str,
    measure: str,
    value: float,
    std: float,
    seed: int,
    args: dict
) -> None:
    """Log a single AAMI-style result.
    
    Args:
        dataset: Name of the dataset (e.g., 'UCI', 'PulseDB')
        method: Name of the method used
        direction: Direction of translation (e.g., 'PPG→ABP')
        measure: Type of measurement (e.g., 'SBP', 'DBP', 'MAP')
        value: Result value
        std: Standard deviation
        seed: Random seed used
        args: Arguments dictionary containing wandb configuration
    """
    if wandb.run is None and args.project_name is not None:
        init_wandb(project_name=args.project_name, config=vars(args))
        
    if wandb.run is not None:
        wandb.log({
            "Dataset": format_dataset_name(dataset),
            "Method": format_method_name(method),
            "Direction": format_direction(direction),
            "Measure": measure,
            "Value": value,
            "Std": std,
            "seed": seed,
            "Task": "AAMI"
        })
    else:
        logging.warning("wandb run not found - logs will not be sent to wandb")
        
    finish_wandb()

def log_clinical_tasks_result(
    dataset: str,
    method: str,
    direction: str,
    metric: str,
    value: float,
    std: float,
    seed: int,
    args: dict,
    filtered: bool = False
) -> None:
    """Log a single Clinical tasks result.
    
    Args:
        dataset: Name of the dataset (e.g., 'UCI', 'PulseDB')
        method: Name of the method used
        direction: Direction of translation (e.g., 'PPG→ABP')
        metric: Type of metric (e.g., 'MAE')
        value: Result value
        std: Standard deviation
        seed: Random seed used
        args: Arguments dictionary containing wandb configuration
        filtered: Whether this is a filtered metric (default: False)
    """
    if wandb.run is None and args.project_name is not None:
        init_wandb(project_name=args.project_name, config=vars(args))
        
    if wandb.run is not None:
        wandb.log({
            "Dataset": format_dataset_name(dataset),
            "Method": format_method_name(method),
            "Direction": format_direction(direction),
            "Metric": metric,
            "Value": value,
            "Std": std,
            "seed": seed,
            "Task": "Clinical_Tasks",
            "Filtered": filtered
        })
    else:
        logging.warning("wandb run not found - logs will not be sent to wandb")
        
    finish_wandb()

def log_approx_result(
    dataset: str,
    method: str,
    direction: str,
    metric: str,
    value: float,
    std: float,
    seed: int,
    args: dict
) -> None:
    """Log a single Approximation result.
    
    Args:
        dataset: Name of the dataset (e.g., 'UCI', 'PulseDB')
        method: Name of the method used
        direction: Direction of translation (e.g., 'PPG→ABP')
        metric: Type of metric (e.g., 'MAE', 'PC')
        value: Result value
        std: Standard deviation
        seed: Random seed used
        args: Arguments dictionary containing wandb configuration
    """
    if wandb.run is None and args.project_name is not None:
        init_wandb(project_name=args.project_name, config=vars(args))
        
    if wandb.run is not None:
        wandb.log({
            "Dataset": format_dataset_name(dataset),
            "Method": format_method_name(method),
            "Direction": format_direction(direction),
            "Metric": metric,
            "Value": value,
            "Std": std,
            "seed": seed,
            "Task": "Approximation"
        })
    else:
        logging.warning("wandb run not found - logs will not be sent to wandb")
        
    finish_wandb()

def log_waveform_features_result(
    dataset: str,
    method: str,
    direction: str,
    measure: str,
    metric: str,
    unit: str,
    value: float,
    std: float,
    seed: int,
    args: dict
) -> None:
    """Log a single waveform features result.
    
    Args:
        dataset: Name of the dataset (e.g., 'UCI', 'PulseDB')
        method: Name of the method used
        direction: Direction of translation (e.g., 'PPG→ECG')
        measure: Type of measurement (e.g., 'SQI', 'P-Wave Morphology')
        metric: Type of metric (e.g., 'MAE')
        unit: Unit of measurement (e.g., 's', '-')
        value: Result value
        std: Standard deviation
        seed: Random seed used
        args: Arguments dictionary containing wandb configuration
    """
    if wandb.run is None and args.project_name is not None:
        init_wandb(project_name=args.project_name, config=vars(args))
        
    if wandb.run is not None:
        wandb.log({
            "Dataset": format_dataset_name(dataset),
            "Method": format_method_name(method),
            "Direction": format_direction(direction),
            "Measure": measure,
            "Metric": metric,
            "Unit": unit,
            "Value": value,
            "Std": std,
            "seed": seed,
            "Task": "Waveform_Features"
        })
    else:
        logging.warning("wandb run not found - logs will not be sent to wandb")
        
    finish_wandb()

def log_data_distribution(
    waveform_type: str,
    value: float,
    unit: str,
    dataset: str,
    split: str,
    args: Union[dict, Any]
) -> None:
    """Log data distribution metrics to wandb.
    
    Args:
        waveform_type: Type of waveform (e.g., 'ECG', 'PPG', 'ABP')
        value: Measurement value
        unit: Unit of measurement (e.g., 'BPM', 'mmHg')
        dataset: Name of the dataset (e.g., 'PulseDB', 'UCI')
        split: Data split (e.g., 'train', 'val', 'test')
        args: Arguments dictionary or object containing wandb configuration
    """
    project_name = args.project_name if hasattr(args, 'project_name') else args.get('project_name')
    
    if wandb.run is None and project_name is not None:
        init_wandb(project_name=project_name, config=vars(args) if hasattr(args, '__dict__') else args)
        
    if wandb.run is not None:
        wandb.log({
            "Waveform_Type": waveform_type,
            "Value": value,
            "Unit": unit,
            "Dataset": format_dataset_name(dataset),
            "Split": split,
            "Task": "Data_Distribution"
        })
    else:
        logging.warning("wandb run not found - logs will not be sent to wandb")
        
    finish_wandb()

def init_wandb(
    project_name: str,
    entity: str = "single_waveform",
    run_name: Optional[str] = None,
    config: Optional[dict] = None
) -> None:
    """Initialize a new wandb run.
    
    Args:
        project_name: Name of the wandb project
        entity: Name of the wandb entity (default: "single_waveform")
        run_name: Optional name for this specific run
        config: Optional configuration dictionary to log
    """
    try:
        wandb.init(
            entity=entity,
            project=project_name,
            name=run_name,
            config=config
        )
        if wandb.run is None:
            logging.warning("wandb run not found - logs will not be sent to wandb")
        else:
            logging.info(f"Using wandb run: {wandb.run.name}")
    except Exception as e:
        logging.error(f"Failed to start wandb run: {str(e)}")

def log_config(config: dict) -> None:
    """Log configuration parameters to wandb.
    
    Args:
        config: Dictionary of configuration parameters to log
    """
    if wandb.run is not None:
        wandb.config.update(config)
    else:
        logging.warning("wandb run not found - logs will not be sent to wandb")

def log_metrics(metrics: dict) -> None:
    """Log a dictionary of metrics to wandb.
    
    Args:
        metrics: Dictionary of metric names and values to log
    """
    if wandb.run is not None:
        wandb.log(metrics)
    else:
        logging.warning("wandb run not found - logs will not be sent to wandb")

def log_model_summary(model) -> None:
    """Log model architecture summary to wandb.
    
    Args:
        model: PyTorch model to summarize
    """
    if wandb.run is not None:
        wandb.watch(model)
    else:
        logging.warning("wandb run not found - logs will not be sent to wandb")

def finish_wandb() -> None:
    """Finish the current wandb run."""
    if wandb.run is not None:
        wandb.finish()
    else:
        logging.warning("wandb run not found - no run to finish") 