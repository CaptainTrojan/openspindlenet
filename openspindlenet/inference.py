import numpy as np
import pywt
import onnxruntime as ort
import os
import sys
from typing import Union, Dict, List, Tuple, Any

from .evaluator import Evaluator

class SpindleInference:
    def __init__(self, model_path):
        """Initialize the inference engine with the ONNX model path."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.session = ort.InferenceSession(model_path)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
    def __normalize(self, data: np.ndarray):
        """Normalize the data."""
        normed = (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)
        return np.nan_to_num(normed, nan=0.0)

    def __convert_to_scalogram(self, data: np.ndarray):
        """Convert the raw data to a scalogram using CWT."""
        # Matches the preprocessing used to generate HDF5 datasets in mayo_spindles.
        coeffs, frequencies = pywt.cwt(data, np.geomspace(135, 270, num=15), 'shan6-13', sampling_period=1/250)
        return np.abs(coeffs)
    
    def __sigmoid(self, x):
        """Apply sigmoid function to input array."""
        return 1 / (1 + np.exp(-x))
    
    def load_data(self, file_path_or_array: Union[str, np.ndarray]) -> np.ndarray:
        """Load data from a text file or numpy array."""
        if isinstance(file_path_or_array, str):
            # Load from file
            try:
                with open(file_path_or_array, 'r') as f:
                    data = f.read().strip().split('\n')
                    data = np.array([float(x) for x in data])
            except Exception as e:
                raise IOError(f"Error loading data from file: {e}")
        elif isinstance(file_path_or_array, np.ndarray):
            # Use provided array
            data = file_path_or_array
        else:
            raise TypeError(f"Expected file path (str) or numpy array, got {type(file_path_or_array)}")
                
        if len(data) != 7500:
            raise ValueError(f"Expected 7500 values, but got {len(data)}")
                
        return data

    def load_labels(self, file_path_or_array: Union[str, np.ndarray]) -> np.ndarray:
        """Load binary segmentation labels from a text file or numpy array."""
        if isinstance(file_path_or_array, str):
            try:
                with open(file_path_or_array, 'r') as f:
                    labels = np.array([float(x) for x in f.read().strip().split('\n')], dtype=np.float32)
            except Exception as e:
                raise IOError(f"Error loading labels from file: {e}")
        elif isinstance(file_path_or_array, np.ndarray):
            labels = file_path_or_array.astype(np.float32)
        else:
            raise TypeError(f"Expected labels path (str) or numpy array, got {type(file_path_or_array)}")

        if labels.ndim > 2:
            raise ValueError(f"Expected labels to be 1D or 2D, got shape {labels.shape}")
        if labels.ndim == 2 and labels.shape[1] != 1:
            raise ValueError(f"Expected labels shape [N] or [N,1], got {labels.shape}")
        if labels.ndim == 2:
            labels = labels[:, 0]
        if len(labels) != 7500:
            raise ValueError(f"Expected 7500 label values, but got {len(labels)}")

        return labels
    
    def preprocess(self, data):
        """Preprocess the data by converting to scalogram and normalizing."""
        # Reshape for model input - adding batch and channel dimensions
        raw_data = data.reshape(1, 1, -1)
        raw_normalized = self.__normalize(raw_data)
        
        # Generate scalogram
        spectrogram = self.__convert_to_scalogram(data)
        # Add batch dimension to spectrogram
        spectrogram = spectrogram.reshape(1, *spectrogram.shape)
        spectrogram_normalized = self.__normalize(spectrogram)
        
        return raw_normalized, spectrogram_normalized
    
    def predict(self, raw_signal, spectrogram):
        """Run inference with the ONNX model."""
        # Create input dictionary
        inputs = {
            'raw_signal': raw_signal.astype(np.float32),
            'spectrogram': spectrogram.astype(np.float32)
        }
        
        # Check that required inputs are present
        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"Model requires input '{name}' which was not provided")
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        return outputs
    
    def postprocess(self, outputs, confidence_threshold=0.5, nms_iou_threshold=0.3):
        """Apply post-processing to model outputs."""
        # Apply sigmoid to all outputs for post-processing
        processed_outputs = [self.__sigmoid(output) for output in outputs]
        
        # Drop batch dimension for outputs
        for i in range(len(processed_outputs)):
            processed_outputs[i] = processed_outputs[i][0]
        
        # Create output dictionary
        output_dict = {}
        for i, name in enumerate(self.output_names):
            output_dict[name] = processed_outputs[i]
            
        # Apply NMS to detections if present
        if 'detection' in output_dict:
            detections = output_dict['detection']
            # Process single sample
            seq_len = 7500  # Standard sequence length
            intervals = Evaluator.detections_to_intervals(detections, seq_len, confidence_threshold=confidence_threshold)
            nms_intervals = Evaluator.intervals_nms(intervals, iou_threshold=nms_iou_threshold)
            output_dict['detection_intervals'] = nms_intervals
        
        return processed_outputs, output_dict
    
    def run(self, data: np.ndarray, confidence_threshold=0.5, nms_iou_threshold=0.3) -> Dict[str, Any]:
        """Run the full inference pipeline."""
        # Preprocess data
        raw_signal, spectrogram = self.preprocess(data)
        
        # Run inference
        outputs = self.predict(raw_signal, spectrogram)
        
        # Post-process outputs
        _, output_dict = self.postprocess(outputs, confidence_threshold=confidence_threshold, nms_iou_threshold=nms_iou_threshold)
        
        return output_dict


def detect_spindles(
    data_source: Union[str, np.ndarray],
    model_type: str = "eeg",
    custom_model_path: str = None,
    labels: Union[str, np.ndarray, None] = None,
    confidence_threshold: float = 0.5,
    nms_iou_threshold: float = 0.3,
    match_iou_threshold: float = 0.3,
    segmentation_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Detect spindles in EEG or iEEG data.
    
    Args:
        data_source: Path to a text file or numpy array containing the signal
        model_type: Type of model to use ('eeg' or 'ieeg')
        custom_model_path: Path to a custom ONNX model file (overrides model_type)
        labels: Optional segmentation labels ([7500] or [7500,1]) or path to labels txt
        confidence_threshold: Detection confidence threshold
        nms_iou_threshold: IoU threshold for detection NMS
        match_iou_threshold: IoU threshold for one-to-one TP matching
        segmentation_threshold: Threshold for segmentation metrics
        
    Returns:
        Dictionary containing detection results
    """
    # Determine model path
    if custom_model_path:
        model_path = custom_model_path
    else:
        # Import here to avoid circular imports
        from . import get_model_path
        model_path = get_model_path(model_type)
    
    # Initialize inference
    inference = SpindleInference(model_path)
    
    # Load data
    data = inference.load_data(data_source)
    
    # Run inference
    results = inference.run(data, confidence_threshold=confidence_threshold, nms_iou_threshold=nms_iou_threshold)
    
    # Add the raw signal for visualization
    results['raw_signal'] = data

    if labels is not None:
        y_true = inference.load_labels(labels).reshape(-1, 1)
        metrics = {
            'detection': Evaluator.detection_metrics_from_arrays(
                y_true_segmentation=y_true,
                y_pred_detection=results['detection'],
                confidence_threshold=confidence_threshold,
                nms_iou_threshold=nms_iou_threshold,
                match_iou_threshold=match_iou_threshold,
            ),
            'segmentation': Evaluator.segmentation_metrics_from_arrays(
                y_true_segmentation=y_true,
                y_pred_segmentation=results.get('segmentation', np.zeros_like(y_true, dtype=np.float32)),
                threshold=segmentation_threshold,
            ),
        }
        results['metrics'] = metrics
    
    return results
