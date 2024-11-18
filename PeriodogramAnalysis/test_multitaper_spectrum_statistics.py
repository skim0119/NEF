import numpy as np
from mne.time_frequency import psd_array_multitaper
from miv.datasets.openephys_sample import load_data
from miv.core.operator import Operator, DataLoader
from miv.io.openephys import Data, DataManager
from multitaper_spectrum_statistics import multitaper_psd

path: str = load_data(progbar_disable=True).data_collection_path
dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]


def test_multitaper_psd():
    for signal in data.load():

        for channel_index in range(5):
            psd, freqs = multitaper_psd(signal.data[:, channel_index], signal.rate)
            psd_mne, freqs_mne = psd_array_multitaper(
                signal.data[:, channel_index],
                signal.rate,
                adaptive=True,
                normalization="full",
                verbose=0,
            )

            difference_psd = psd_mne - psd
            sum_diff_psd = np.sum(np.abs(difference_psd))
            sum_pds = np.sum(np.abs(psd_mne))

            # Test the difference is smaller than 1%
            assert sum_diff_psd / sum_pds < 1e-2
