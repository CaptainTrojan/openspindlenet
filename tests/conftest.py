"""
Test fixtures for OpenSpindleNet tests
"""
import os
import pytest
import numpy as np

@pytest.fixture
def sample_eeg_data():
    """Return a sample EEG signal for testing."""
    # Generate a simple sine wave with noise as test data
    # This creates a 7500-point signal (the required length)
    x = np.linspace(0, 15, 7500)  # 15 seconds at 500Hz
    # Base signal: sum of sine waves at different frequencies
    signal = np.sin(2 * np.pi * 0.5 * x) + 0.5 * np.sin(2 * np.pi * 11 * x)
    # Add some random noise
    noise = 0.1 * np.random.normal(0, 1, 7500)
    return signal + noise

@pytest.fixture
def sample_eeg_data_file(tmp_path, sample_eeg_data):
    """Create a temporary file with sample EEG data."""
    file_path = tmp_path / "test_eeg.txt"
    np.savetxt(file_path, sample_eeg_data)
    return file_path