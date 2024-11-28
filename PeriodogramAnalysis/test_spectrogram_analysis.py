from typing import Generator, Any

import numpy as np
from scipy.signal import spectrogram

from miv.core.datatype import Signal
from spectrogram_analysis import SpectrogramAnalysis


def signal_input() -> Generator:
    timestamps = np.linspace(0, 10, 1000, endpoint=False)
    signal = (
        np.sin(2 * np.pi * 5 * timestamps)
        + np.sin(2 * np.pi * 10 * timestamps)
        + np.sin(2 * np.pi * 15 * timestamps)
        + np.sin(2 * np.pi * 20 * timestamps)
    )

    data1 = np.array([signal, signal]).T
    signal1 = Signal(data=data1, timestamps=timestamps, rate=100)
    yield signal1

    data2 = np.array([signal, signal]).T
    signal2 = Signal(data=data2, timestamps=timestamps, rate=100)
    yield signal2


def test_SpectrumAnalysis_call() -> None:
    """
    Test call func and computing_spectrum, check if spec_dict is as expected.
    """
    spectrum_analysis = SpectrogramAnalysis()
    spec_dict = spectrum_analysis(signal_input())

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


def test_computing_spectrum() -> None:
    """
    Test the computing_spectrum method.
    """
    power_analysis = SpectrogramAnalysis()
    spec_dict: dict[str, Any] = {}

    signal_gen = signal_input()
    signal = next(signal_gen)

    result_spec_dict = power_analysis.computing_spectrum(signal, spec_dict)

    for channel_index in result_spec_dict:
        channel_data = result_spec_dict[channel_index]

        for i in range(len(channel_data["frequencies"])):
            signal_no_bias = signal.data[:, channel_index] - np.mean(
                signal.data[:, channel_index]
            )
            freqs_expected, times_expected, sxx_expected = spectrogram(
                signal_no_bias,
                fs=signal.rate,
                nperseg=int(signal.rate * power_analysis.nperseg_ratio),
                noverlap=int((int(signal.rate * power_analysis.nperseg_ratio)) / 2),
            )

            np.testing.assert_allclose(channel_data["frequencies"][i], freqs_expected)
            np.testing.assert_allclose(channel_data["times"][i], times_expected)
            np.testing.assert_allclose(channel_data["Sxx"][i], sxx_expected)

    print("All tests passed successfully.")
