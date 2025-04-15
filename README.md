# OpenSpindleNet

An open-source tool for sleep spindle detection using neural networks.

## Installation

```bash
pip install openspindlenet
```

## Usage

### Command Line Interface

```bash
# Detect spindles in an EEG file
spindle-detect path/to/eeg.txt --model eeg --vis

# Detect spindles in an iEEG file
spindle-detect path/to/ieeg.txt --model ieeg --vis
```

### Python API

```python
import numpy as np
from openspindlenet import detect_spindles

# From a file
results = detect_spindles("path/to/eeg.txt", model_type="eeg")

# From a numpy array
eeg_data = np.random.randn(7500)
results = detect_spindles(eeg_data, model_type="eeg")

# Access results
print(f"Number of spindles detected: {len(results['detection_intervals'])}")
for i, (start, end, conf) in enumerate(results['detection_intervals']):
    print(f"Spindle {i+1}: start={start:.1f}, end={end:.1f}, confidence={conf:.4f}")
```
