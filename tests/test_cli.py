"""
Tests for OpenSpindleNet CLI functionality
"""
from __future__ import annotations

import os
import sys
import pytest
import importlib
from pathlib import Path
import subprocess
import re
import shutil

# Skip all tests in this file if CLI dependencies are not installed
has_cli_deps = all(
    importlib.util.find_spec(pkg) is not None
    for pkg in ["typer", "rich", "matplotlib", "tqdm"]
)

skip_if_no_cli_deps = pytest.mark.skipif(
    not has_cli_deps,
    reason="CLI dependencies not installed"
)


def _cli_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "openspindlenet.cli", *args]

def test_cli_missing_dependencies_message():
    """Test that appropriate message is shown when CLI dependencies are missing."""
    # We'll temporarily modify PYTHONPATH to simulate missing dependencies
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)
    
    # Run a subprocess with modified modules
    cmd = [sys.executable, "-c", 
           "import sys; sys.modules['typer'] = None; " +
           "from openspindlenet.cli import main; main()"]
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    # Should exit with error code
    assert result.returncode != 0
    
    # Should show the error message
    assert "CLI dependencies are not installed" in result.stdout or "CLI dependencies are not installed" in result.stderr
    assert "pip install openspindlenet[cli]" in result.stdout or "pip install openspindlenet[cli]" in result.stderr

@skip_if_no_cli_deps
def test_cli_detect_command(sample_eeg_data_file, tmp_path):
    """Test the detect command when CLI dependencies are available."""
    output_file = tmp_path / "output.txt"
    
    # Run the CLI command
    cmd = _cli_cmd("detect", str(sample_eeg_data_file), "--output-file", str(output_file))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check that command executed successfully
    assert result.returncode == 0
    
    # Check that output file was created
    assert output_file.exists()

@skip_if_no_cli_deps
def test_cli_detect_with_model_type(sample_eeg_data_file):
    """Test the detect command with model-type parameter as shown in the README."""
    # Test with EEG model type (explicitly)
    cmd = _cli_cmd("detect", str(sample_eeg_data_file), "--model-type", "eeg", "--no-visualize")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Command should execute successfully
    assert result.returncode == 0
    assert "Detected" in result.stdout
    
    # Test with iEEG model type if available
    try:
        # Check if the iEEG model exists
        from openspindlenet import get_model_path
        get_model_path("ieeg")
        
        # Run command with iEEG model
        cmd = _cli_cmd("detect", str(sample_eeg_data_file), "--model-type", "ieeg", "--no-visualize")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Detected" in result.stdout
    except:
        # Skip this part of the test if iEEG model is not available
        pass

@skip_if_no_cli_deps
def test_cli_detect_with_visualize(sample_eeg_data_file):
    """Test the detect command with the visualize flag as shown in the README."""
    # We can test that the flag is recognized, but we can't easily test
    # that a visualization appears, so we'll check that the command mentions
    # generating a visualization
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to prevent display
    
    cmd = _cli_cmd("detect", str(sample_eeg_data_file), "--visualize")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Command should execute successfully
    assert result.returncode == 0
    assert "Saved visualization to" in result.stdout

@skip_if_no_cli_deps
def test_cli_info_command():
    """Test the info command."""
    # Run the CLI command
    cmd = _cli_cmd("info")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check that command executed successfully
    assert result.returncode == 0
    
    # Check that it contains expected output
    assert "OpenSpindleNet" in result.stdout
    assert "EEG" in result.stdout
    assert "iEEG" in result.stdout

@skip_if_no_cli_deps
def test_cli_example_command():
    """Test the example command."""
    # This test will only work if example data is available
    # Skip if example data files don't exist
    if not (Path(__file__).parent.parent / "openspindlenet" / "data" / "eeg_sample.txt").exists():
        pytest.skip("Example data files not available")
    
    # Run the CLI command with --no-visualize to avoid opening plot window
    cmd = _cli_cmd("example", "--no-visualize")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check that command executed successfully
    assert result.returncode == 0
    
    # Check that it contains expected output
    assert "Detected" in result.stdout
    assert "example" in result.stdout.lower()

@skip_if_no_cli_deps
def test_cli_example_with_data_type():
    """Test the example command with the data-type parameter as shown in the README."""
    # Test with EEG data type
    cmd = _cli_cmd("example", "--data-type", "eeg", "--no-visualize")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Command should execute successfully
    assert result.returncode == 0
    assert "Detected" in result.stdout
    
    # Test with iEEG data type if available
    if (Path(__file__).parent.parent / "openspindlenet" / "data" / "ieeg_sample.txt").exists():
        cmd = _cli_cmd("example", "--data-type", "ieeg", "--no-visualize")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Detected" in result.stdout

@skip_if_no_cli_deps
def test_visualization_saving(sample_eeg_data_file, tmp_path):
    """Test that visualizations are correctly saved to a file."""
    # Use a specific output path in the temp directory
    vis_path = tmp_path / "test_visualization.pdf"
    
    # Enable visualization and specify the output path
    cmd = _cli_cmd("detect", str(sample_eeg_data_file),
                   "--visualize", "--visualization-file", str(vis_path))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Command should execute successfully
    assert result.returncode == 0
    
    # Check if the visualization messages appear in the output
    assert "Saved visualization to" in result.stdout
    
    # Check if the visualization file exists
    assert vis_path.exists()
    assert vis_path.stat().st_size > 0  # File should not be empty
    
    # Check if the output message correctly shows the path
    assert "Saved visualization to" in result.stdout
    # Using Path.name to just match the filename part since paths might differ
    normalized_stdout = re.sub(r"\s+", "", result.stdout)
    assert vis_path.name in normalized_stdout

@skip_if_no_cli_deps
def test_example_visualization_saving(tmp_path):
    """Test that example command saves visualizations correctly."""
    # Use a specific output path in the temp directory
    vis_path = tmp_path / "example_visualization.pdf"
    
    # Enable visualization and specify the output path
    cmd = _cli_cmd("example", "--visualization-file", str(vis_path))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Command should execute successfully
    assert result.returncode == 0
    
    # Check if the visualization messages appear in the output
    assert "Saved visualization to" in result.stdout
    
    # Check if the visualization file exists
    assert vis_path.exists()
    assert vis_path.stat().st_size > 0  # File should not be empty
    
    # Check if the output message correctly shows the path
    assert "Saved visualization to" in result.stdout
    # Using Path.name to just match the filename part since paths might differ
    normalized_stdout = re.sub(r"\s+", "", result.stdout)
    assert vis_path.name in normalized_stdout