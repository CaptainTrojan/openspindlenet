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

# Optional: evaluate against segmentation labels (same length as signal)
labels = np.zeros(7500, dtype=np.float32)
scored = detect(eeg_data, model_type="eeg", labels=labels)
print(scored["metrics"]["detection"])  # precision/recall/f1 + TP/FP/FN
print(scored["metrics"]["segmentation"])  # precision/recall/f1 + TP/FP/FN

# Access results
print(f"Number of spindles detected: {len(results['detection_intervals'])}")
for i, (start, end, confidence) in enumerate(results['detection_intervals']):
  print(f"Spindle {i+1}: start={start:.1f}, end={end:.1f}, conf={confidence:.2f}")
```

### Command Line Interface

The CLI requires additional dependencies. Install with `pip install openspindlenet[cli]`.

```bash
# Detect spindles in an EEG file
openspindlenet detect path/to/eeg.txt --model-type eeg --visualize

# Detect spindles in an iEEG file
openspindlenet detect path/to/ieeg.txt --model-type ieeg --visualize

# Evaluate one signal against labels and write JSON/CSV outputs
openspindlenet eval path/to/eeg.txt path/to/labels.txt --visualize

# Show information about available models
openspindlenet info

# Run detection on included example data
openspindlenet example --data-type eeg
```

`eval` writes timestamped JSON and CSV metric outputs by default and, with `--visualize`, generates a PDF that includes predictions, labels, and metric summary.

## DREAMS Repro Bundle

The repository includes a reproducibility bundle under [dreams-bundle](dreams-bundle) with:

- DREAMS test signals dumped into TXT files
- matching TXT label files
- README instructions for one-file CLI evaluation

Main workflow:

```bash
cd dreams-bundle
openspindlenet eval data/signals/test_000.txt data/labels/test_000.txt --visualize
```

Use `mayo_spindles/mayo_spindles/create_dreams_repro_bundle.py` to regenerate the TXT bundle from DREAMS_HDF5.

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
