"""
Blood pressure utility functions.

This module provides utility functions for blood pressure calculations
that were previously in the deprecated test_utils module.
"""

import torch


def extract_bp_values(waveform, dim=2):
    """
    Extract SBP and DBP values from waveform.
    
    This function replaces the deprecated extract_bp_values from test_utils.
    
    Args:
        waveform (torch.Tensor): Input waveform tensor
        dim (int): Dimension along which to find min/max values
        
    Returns:
        tuple: (sbp_values, dbp_values) tensors
    """
    sbp_values = torch.max(waveform, dim=dim)[0]
    dbp_values = torch.min(waveform, dim=dim)[0]
    
    if len(sbp_values.shape) == 1:
        sbp_values = sbp_values.unsqueeze(1)
        dbp_values = dbp_values.unsqueeze(1)
    
    return sbp_values, dbp_values 