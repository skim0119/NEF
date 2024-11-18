import numpy as np

from miv.core.datatype import Signal
from spectrogram_analysis import SpectrogramAnalysis
from power_density_statistics import SpectrumAnalysisWelch


def mock_signal_generator():
    data1 = np.array(
        [
            np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100)),  # 5 Hz
            np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100)),  # 10 Hz
        ]
    ).T
    timestamps = np.linspace(0, 1, 100)
    signal1 = Signal(data=data1, timestamps=timestamps, rate=100.0)
    yield signal1

    data2 = np.array(
        [
            np.sin(2 * np.pi * 15 * np.linspace(0, 1, 200)),  # 15 Hz
            np.sin(2 * np.pi * 20 * np.linspace(0, 1, 200)),  # 20 Hz
        ]
    ).T
    timestamps = np.linspace(0, 1, 100)
    signal2 = Signal(data=data2, timestamps=timestamps, rate=100.0)
    yield signal2


def test_SpectrumAnalysis_call_and_computing_spectrum():
    """
    Test call func and computing_spectrum, check if spec_dict is as expected.
    """
    spectrum_analysis = SpectrogramAnalysis()
    spec_dict = spectrum_analysis(mock_signal_generator())

    # test structure of spec_dict
    assert isinstance(spec_dict, dict)
    assert len(spec_dict) == 2

    # Test the content of power_dict
    for channel, channel_data in spec_dict.items():
        assert "frequencies" in channel_data
        assert "times" in channel_data
        assert "Sxx" in channel_data

        assert len(spec_dict[channel]["frequencies"]) == 2
        assert len(spec_dict[channel]["times"]) == 2
        assert len(spec_dict[channel]["Sxx"]) == 2
