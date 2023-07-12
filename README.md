# preprocessing pipeline

This library provides pre-built methods for processing ragged dataset which is
particularly useful for time series data of channels with different sampling
rates

# Example

```python
# imports
import mne
import numpy as np

from bids import BIDSLayout, BIDSLayoutIndexer
from pathlib import Path
from pipeline import TransformPipeline, Windower
from pipeline.mne import Labeler

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
    ("flt", mne.decoding.TemporalFilter(sfreq=200, l_freq=5)),
    ("psd", mne.decoding.PSDEstimator(sfreq=200))
])

# Preprocess all runs
data = [preprocessor.fit_transform(x.pick([0, 2])) for x in raws]

# data is now a list of matrices, each with the shape (windows, channels, frequencies)
```
