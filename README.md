# preprocessing pipeline

This library provides pre-built methods for processing ragged dataset which is
particularly useful for time series data of channels with different sampling
rates
## Getting Started

### Prerequisites

- Anaconda or Miniconda
- Git version control software
- Required Python packages (list in `requirements.txt`)
- Access to the shared box folder with the data
- Access to the Zotero reference library with Better BibTex installed and
  automated reference bib export

### Installation

```base
pip install git+https://github.com/umnil/preprocessing-pipeline.git
```

**NOTE** This package use to be imported simply using `import pipeline`. In
order to conform to PEP8 standards that the module name match the package, the
module has been renamed to `preprocessingpipeline`. In case you don't want to
go back and edit all your code, you cans simply run the following to make
`import pipeline` work:

```bash
DIR=$(pip show preprocessing-pipeline | grep Location | sed -Ee 's/Location: //g')
ln -s "${DIR}/preprocessingpipeline" "${DIR}/pipeline"
```

# Example

```python
# imports
import mne
import numpy as np

from bids import BIDSLayout, BIDSLayoutIndexer
from pathlib import Path
from preprocessingpipeline import TransformPipeline, Windower
from preprocessingpipeline.mne import Labeler

# paths
data_dir = Path("to", "bids", "data", "directory")

# silence MNE
mne.set_log_level("critical")

# load bids
indexer = BIDSLayoutIndexer(validate=False)
bids_layout = BIDSLayout(data_dir, indexer=indexer)

# select all edf files
edf_files = bids_layout.get(extension="edf")

# read all edf files to mne raw objects
raws = [mne.io.read_raw_edf(file.path) for file in edf_files]

# define our preprocessing transformer
preprocessor = TransformPipeline([
    ("lab", Labeler()),
    ("wnd", Windower(samples_per_window=80*8, label_scheme=3, window_step=80, trial_size=83600)),
    ("con", TFunctionTransformer(funcs.concat, kw_args={"active": True})),
    ("flt", mne.decoding.TemporalFilter(sfreq=200, l_freq=5)),
    ("psd", mne.decoding.PSDEstimator(sfreq=200))
])

# Preprocess all runs
data = [preprocessor.fit_transform(x.pick([0, 2])) for x in raws]

# data is now a list of matrices, each with the shape (windows, channels, frequencies)
```

## Contributing

Contributions are welcome! Please create a new issue to discuss any major
changes or improvements.

### Creating Issues

To create a new issue:
1. Go to the `Issues` tab.
2. Click on `New Issue`.
3. Provide a descriptive title and detailed description.
4. Assign appropriate labels and, if possible, link to related issues or pull
   requests.

### Pull Requests

To submit a pull request:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to your fork.
4. Create a pull request from your branch to the main repository.
