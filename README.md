# preprocessing pipeline

This library provides pre-built methods for:

1. Extracting packeted neural data from pandas `DataFrame`s into a MxN matrix
   where $`M`$ = number of packets and $`N`$ = number of data points per
   packet
   - if multiple channels exist their data points are concatenated and the
     break points for delineating where data from one channel stops and the
     other begins is stored in the object under the `packet_channel_sizes`
     property)
1. Windowing the data so that $`M`$ becomes the number of windows
1. Filtering the windows of data. By default this is a 1Hz High-Pass filter,
   but options are available for flexible processing
1. Featurizing the data. The default is to convert the time-series data into
   the frequency domain such that the resulting matrix is MxN where M = the
   number of windows and N = a concatenated set of frequency features from
   each channel
