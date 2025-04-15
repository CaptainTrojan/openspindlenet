# Installation Guide for OpenSpindleNet

## Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

## Installation Options

### Option 1: Install from Source (Development Mode)

1. Clone the repository:
   ```bash
   git clone https://github.com/CaptainTrojan/openspindlenet.git
   cd openspindlenet
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

3. Test the installation:
   ```bash
   spindle-detect --help
   ```

### Option 2: Regular Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/CaptainTrojan/openspindlenet.git
   cd openspindlenet
   ```

2. Install the package:
   ```bash
   pip install .
   ```

3. Test the installation:
   ```bash
   spindle-detect --help
   ```

## Verifying Model Files

The model files should be automatically included when you install the package. To verify they are correctly installed, you can run:

```python
from openspindlenet import get_model_path
print(get_model_path("eeg"))  # Should print the path to the EEG model
print(get_model_path("ieeg"))  # Should print the path to the iEEG model
```

## Troubleshooting

If you get an error like "Model file not found", try:

1. Reinstalling the package: `pip install -e .` (for development mode) or `pip install .` (for regular installation)
2. Verify the models exist in the root `models/` directory with the correct names:
   - EEG model: `spindle-detector-eeg.onnx`
   - iEEG model: `spindle-detector-ieeg.onnx`
