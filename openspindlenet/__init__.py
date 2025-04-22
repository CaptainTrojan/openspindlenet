import os
import pkg_resources

from .inference import detect_spindles as detect

# Internal functions - not exposed directly
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
        f"data/{data_type.lower()}_sample.txt"
    )

__all__ = ['detect']
