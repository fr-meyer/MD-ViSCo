from dataclasses import dataclass
from typing import Optional, Dict, Any
from hydra.core.config_store import ConfigStore

@dataclass
class EarlyStoppingConfig:
    """
    Configuration for EarlyStopping, which halts training if the validation loss does not improve after a specified number of epochs (patience).
    """
    _target_: str = "src.utils.train_utils.EarlyStopping"

    # Early stopping parameters
    patience: int = 5
    threshold: float = 0.0
    threshold_mode: str = 'rel'
    verbose: bool = False
    delta: float = 0.0

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    
    Attributes:
        patience (int): How long to wait after last time validation loss improved
        verbose (bool): If True, prints a message for each validation loss improvement
        counter (int): The number of epochs since last improvement
        best_loss (float): Best observed validation loss
        early_stop (bool): True if training should stop
        delta (float): Minimum change in monitored quantity to qualify as an improvement
        threshold (float): Threshold for measuring the new optimum
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode, dynamic_threshold = best * (1 + threshold)
                            in 'max' mode or best * (1 - threshold) in `min` mode.
                            In `abs` mode, dynamic_threshold = best + threshold
    """
    def __init__(self, patience: int = 5, threshold: float = 0.0, threshold_mode: str = 'rel', verbose: bool = False, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False
        self.delta = delta
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def __call__(self, val_loss: float) -> None:
        """
        Args:
            val_loss (float): Validation loss from current epoch
            
        Returns:
            None, but updates early_stop flag if needed
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss * (1 + self.threshold) if self.threshold_mode == 'rel' else val_loss > self.best_loss + self.threshold:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the early stopping object as a dictionary."""
        return {
            'patience': self.patience,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_loss': self.best_loss,
            'early_stop': self.early_stop,
            'delta': self.delta,
            'threshold': self.threshold,
            'threshold_mode': self.threshold_mode
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the early stopping state from a dictionary.
        
        Args:
            state_dict (dict): Dictionary containing early stopping state
        """
        self.patience = state_dict['patience']
        self.verbose = state_dict['verbose']
        self.counter = state_dict['counter']
        self.best_loss = state_dict['best_loss']
        self.early_stop = state_dict['early_stop']
        self.delta = state_dict['delta']
        self.threshold = state_dict['threshold']
        self.threshold_mode = state_dict['threshold_mode']

cs = ConfigStore.instance()
cs.store(name="base_early_stopping", node=EarlyStoppingConfig, group="early_stopping")