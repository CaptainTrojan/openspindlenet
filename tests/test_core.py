"""
Tests for core OpenSpindleNet functionality
"""
import pytest
import numpy as np
from openspindlenet import detect

def test_detect_import():
    """Test that the detect function is properly exposed."""
    # Simply verify that we can import detect from the package root
    assert callable(detect)

def test_detect_with_array(sample_eeg_data):
    """Test detection with numpy array input."""
    results = detect(sample_eeg_data, model_type="eeg")
    
    # Check that the result has expected structure
    assert isinstance(results, dict)
    assert "detection_intervals" in results
    assert "raw_signal" in results
    
    # Check that the raw signal is included correctly
    assert isinstance(results["raw_signal"], np.ndarray)
    assert results["raw_signal"].shape == (7500,)
    
    # Check that intervals are in correct format
    intervals = results["detection_intervals"]
    # We don't know how many will be detected, but they should be pairs of start/end times
    if len(intervals) > 0:
        assert intervals.shape[1] == 2  # Each interval has start and end time
        assert all(intervals[:, 1] > intervals[:, 0])  # End times should be greater than start times

def test_detect_with_file_path(sample_eeg_data_file):
    """Test detection with file path input."""
    results = detect(str(sample_eeg_data_file), model_type="eeg")
    
    # Check that the result has expected structure
    assert isinstance(results, dict)
    assert "detection_intervals" in results
    
    # Verify the raw signal was loaded correctly
    assert "raw_signal" in results
    assert isinstance(results["raw_signal"], np.ndarray)
    assert results["raw_signal"].shape == (7500,)

def test_accessing_detection_results(sample_eeg_data):
    """Test accessing detection results as shown in the README examples."""
    # This test specifically mirrors the README example for accessing results
    results = detect(sample_eeg_data, model_type="eeg")
    
    # Extract the detection intervals
    intervals = results["detection_intervals"]
    
    # We should be able to print the number of spindles
    num_spindles = len(intervals)
    assert isinstance(num_spindles, int)
    
    # We should be able to iterate through intervals and access start/end times
    if num_spindles > 0:
        # Test that we can iterate and unpack as shown in README
        for i, (start, end) in enumerate(intervals):
            # Verify start and end are numerical values
            assert isinstance(start, (int, float))
            assert isinstance(end, (int, float))
            # Verify end time is after start time
            assert end > start
            # Format the output as shown in README
            spindle_str = f"Spindle {i+1}: start={start:.1f}, end={end:.1f}"
            assert isinstance(spindle_str, str)