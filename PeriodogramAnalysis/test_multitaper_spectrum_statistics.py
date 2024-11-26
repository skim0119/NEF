import numpy as np
from mne.time_frequency import psd_array_multitaper
from miv.datasets.openephys_sample import load_data
from miv.core.operator import Operator, DataLoader
from miv.io.openephys import Data, DataManager
from multitaper_spectrum_statistics import multitaper_psd


def test_multitaper_psd():
    path: str = load_data(progbar_disable=True).data_collection_path
    dataset: DataManager = DataManager(data_collection_path=path)
    data: DataLoader = dataset[0]

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

            np.testing.assert_allclose(psd, psd_mne)
