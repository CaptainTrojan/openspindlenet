# OpenSpindleNet

An open-source tool for sleep spindle detection using neural networks.

## Installation

Basic installation (core detection only):
```bash
pip install .
```

Full installation with CLI capabilities:
```bash
pip install .[cli]
```

## Usage

### Python API

```python
import numpy as np
from openspindlenet import detect

# From a file
results = detect("path/to/eeg.txt", model_type="eeg")

# From a numpy array
eeg_data = np.random.randn(7500)
results = detect(eeg_data, model_type="eeg")

# Access results
print(f"Number of spindles detected: {len(results['detection_intervals'])}")
for i, (start, end) in enumerate(results['detection_intervals']):
    print(f"Spindle {i+1}: start={start:.1f}, end={end:.1f}")
```

### Command Line Interface

The CLI requires additional dependencies. Install with `pip install openspindlenet[cli]`.

```bash
# Detect spindles in an EEG file
spindle-detect detect path/to/eeg.txt --model-type eeg --visualize

# Detect spindles in an iEEG file
spindle-detect detect path/to/ieeg.txt --model-type ieeg --visualize

# Show information about available models
spindle-detect info

# Run detection on included example data
spindle-detect example --data-type eeg
```

## Development

### Testing

Run the test suite:
```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/
```

## Cite

If you find our work useful, please cite our publication
```
@article{sejak2025openspindlenet,
  title={OpenSpindleNet: An Open-Source Deep Learning Network for Reliable Sleep Spindle Detection},
  author={Sejak, Michal and Mivalt, Filip and Sladky, Vladimir and Vsiansky, Vit and Carvalho, Diego Zaquera and St. Louis, Erik and Worrell, Gregory and Kremen, Vaclav},
  journal={medRxiv},
  pages={2025--04},
  year={2025},
  publisher={Cold Spring Harbor Laboratory Press}
}
```

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
