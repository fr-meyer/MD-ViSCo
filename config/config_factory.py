from typing import Union

# Base dataset configs
from config.datasets.dataset_configs import UCIBaseConfig, PulseDBBaseConfig

# Model configs
from config.datasets.PulseDB.model.app_model_config import PulseDBApproximationConfig
from config.datasets.PulseDB.model.ref_model_config import PulseDBRefinementConfig
from config.datasets.UCI.model.app_model_config import UCIApproximationConfig
from config.datasets.UCI.model.ref_model_config import UCIRefinementConfig

# Baseline model configs
# NABNet
from config.datasets.PulseDB.baseline.NABNet.app_model_config import PulseDBApproximationNabNetConfig
from config.datasets.PulseDB.baseline.NABNet.ref_model_config import PulseDBRefinementNabNetConfig
from config.datasets.UCI.baseline.NABNet.app_model_config import UCIApproximationNabNetConfig
from config.datasets.UCI.baseline.NABNet.ref_model_config import UCIRefinementNabNetConfig

# PPG2ABP
from config.datasets.PulseDB.baseline.PPG2ABP.app_model_config import PulseDBApproximationPPG2ABPConfig
from config.datasets.PulseDB.baseline.PPG2ABP.ref_model_config import PulseDBRefinementPPG2ABPConfig
from config.datasets.UCI.baseline.PPG2ABP.app_model_config import UCIApproximationPPG2ABPConfig
from config.datasets.UCI.baseline.PPG2ABP.ref_model_config import UCIRefinementPPG2ABPConfig

# P2E-WGAN
from config.datasets.PulseDB.baseline.P2E_WGAN.app_model_config import PulseDBApproximationP2EWGANConfig
from config.datasets.PulseDB.baseline.P2E_WGAN.ref_model_config import PulseDBRefinementP2EWGANConfig
from config.datasets.UCI.baseline.P2E_WGAN.app_model_config import UCIApproximationP2EWGANConfig
from config.datasets.UCI.baseline.P2E_WGAN.ref_model_config import UCIRefinementP2EWGANConfig

# PatchTST
from config.datasets.PulseDB.baseline.PatchTST.app_model_config import PulseDBApproximationPatchTSTConfig
from config.datasets.PulseDB.baseline.PatchTST.ref_model_config import PulseDBRefinementPatchTSTConfig
from config.datasets.UCI.baseline.PatchTST.app_model_config import UCIApproximationPatchTSTConfig
from config.datasets.UCI.baseline.PatchTST.ref_model_config import UCIRefinementPatchTSTConfig

class ConfigFactory:
    @staticmethod
    def create_config(args) -> Union[
        UCIBaseConfig, PulseDBBaseConfig,
        PulseDBApproximationConfig, PulseDBRefinementConfig,
        PulseDBApproximationNabNetConfig, PulseDBRefinementNabNetConfig,
        PulseDBApproximationPPG2ABPConfig, PulseDBRefinementPPG2ABPConfig,
        PulseDBApproximationP2EWGANConfig, PulseDBRefinementP2EWGANConfig,
        PulseDBApproximationPatchTSTConfig, PulseDBRefinementPatchTSTConfig,
        UCIApproximationConfig, UCIRefinementConfig,
        UCIApproximationNabNetConfig, UCIRefinementNabNetConfig,
        UCIApproximationPPG2ABPConfig, UCIRefinementPPG2ABPConfig,
        UCIApproximationPatchTSTConfig, UCIRefinementPatchTSTConfig,
        UCIApproximationP2EWGANConfig, UCIRefinementP2EWGANConfig
    ]:
        """Create configuration based on command line arguments
        
        Args:
            args: Parsed command line arguments containing:
                - dataset: Dataset name ('uci', 'pulsedb')
                - model_type: Model type ('approximation' or 'refinement')
                - model_name: Optional model name (e.g. 'nabnet')
                - direction: Signal conversion direction
                - is_finetuning: Whether to perform finetuning
                - bp_norm: Whether to use BP normalization
                - use_patient_split: Whether to use patient-wise splitting
        """
        dataset = args.dataset.lower()
        
        # If model_type is None, return base dataset config
        if args.model_type is None:
            if dataset == 'uci':
                return UCIBaseConfig()
            elif dataset == 'pulsedb':
                return PulseDBBaseConfig()
            raise ValueError(f"Unknown dataset: {dataset}")
            
        # Handle model-specific configs
        model_type = args.model_type.lower()
        model_name = args.model_name.lower() if args.model_name else None

        if dataset == 'uci':
            if model_type == 'approximation':
                if model_name == 'nabnet':
                    return UCIApproximationNabNetConfig(args=args)
                elif model_name == 'ppg2abp':
                    return UCIApproximationPPG2ABPConfig(args=args)
                elif model_name == 'patchtst':
                    return UCIApproximationPatchTSTConfig(args=args)
                elif model_name == 'p2ewgan':
                    return UCIApproximationP2EWGANConfig(args=args)
                elif model_name == 'mdvisco':
                    return UCIApproximationConfig(args=args)
            elif model_type == 'refinement':
                if model_name == 'nabnet':
                    return UCIRefinementNabNetConfig(args=args)
                elif model_name == 'ppg2abp':
                    return UCIRefinementPPG2ABPConfig(args=args)
                elif model_name == 'patchtst':
                    return UCIRefinementPatchTSTConfig(args=args)
                elif model_name == 'p2ewgan':
                    return UCIRefinementP2EWGANConfig(args=args)
                elif model_name == 'mdvisco':
                    return UCIRefinementConfig(args=args)
        
        elif dataset == 'pulsedb':
            if model_type == 'approximation':
                if model_name == 'nabnet':
                    return PulseDBApproximationNabNetConfig(args=args)
                elif model_name == 'ppg2abp':
                    return PulseDBApproximationPPG2ABPConfig(args=args)
                elif model_name == 'p2ewgan':
                    return PulseDBApproximationP2EWGANConfig(args=args)
                elif model_name == 'patchtst':
                    return PulseDBApproximationPatchTSTConfig(args=args)
                elif model_name == 'mdvisco':
                    return PulseDBApproximationConfig(args=args)
            elif model_type == 'refinement':
                if model_name == 'nabnet':
                    return PulseDBRefinementNabNetConfig(args=args)
                elif model_name == 'ppg2abp':
                    return PulseDBRefinementPPG2ABPConfig(args=args)
                elif model_name == 'p2ewgan':
                    return PulseDBRefinementP2EWGANConfig(args=args)
                elif model_name == 'patchtst':
                    return PulseDBRefinementPatchTSTConfig(args=args)
                elif model_name == 'mdvisco':
                    return PulseDBRefinementConfig(args=args)
                
        raise ValueError(f"Unknown combination of dataset: {dataset}, model_type: {model_type}, and model_name: {model_name}") 