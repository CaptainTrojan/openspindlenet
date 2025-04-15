import os
import pkg_resources

from .inference import detect_spindles, SpindleInference
from .evaluator import Evaluator
from .visualization import visualize_spindles

# Define function to get paths to included models
def get_model_path(model_type="eeg"):
    """Get the path to a bundled model file.
    
    Args:
        model_type (str): Either 'eeg' or 'ieeg'
        
    Returns:
        str: Path to the model file
    """
    if model_type.lower() not in ["eeg", "ieeg"]:
        raise ValueError("model_type must be either 'eeg' or 'ieeg'")
        
    return pkg_resources.resource_filename(
        "openspindlenet", 
        f"models/spindle-detector-{model_type.lower()}.onnx"
    )

# Define function to get paths to example data
def get_example_data_path(data_type="eeg"):
    """Get the path to example data.
    
    Args:
        data_type (str): Either 'eeg' or 'ieeg'
        
    Returns:
        str: Path to the example data file
    """
    if data_type.lower() not in ["eeg", "ieeg"]:
        raise ValueError("data_type must be either 'eeg' or 'ieeg'")
        
    return pkg_resources.resource_filename(
        "openspindlenet", 
        f"data/example_{data_type.lower()}.txt"
    )

__all__ = [
    'detect_spindles', 
    'SpindleInference', 
    'Evaluator', 
    'visualize_spindles',
    'get_model_path',
    'get_example_data_path'
]
