from typing import Generator, Any

import pytest
import numpy as np
from scipy.signal import spectrogram
from miv.core.datatype import Signal
from spectrogram_analysis import SpectrogramAnalysis


def signal_input():
    timestamps = np.linspace(0, 10, 1000, endpoint=False)
    signal = (
        np.sin(2 * np.pi * 5 * timestamps)
        + np.sin(2 * np.pi * 10 * timestamps)
        + np.sin(2 * np.pi * 15 * timestamps)
        + np.sin(2 * np.pi * 20 * timestamps)
    )

    data1 = np.array([signal, signal]).T
    signal = Signal(data=data1, timestamps=timestamps, rate=100)
    return signal


def test_call_invalid_input_parameters():
    with pytest.raises(ValueError):
        SpectrogramAnalysis(frequency_limit=(2, 1))
    with pytest.raises(ValueError):
        SpectrogramAnalysis(frequency_limit=(5, 10, 20, 30))

    with pytest.raises(ValueError):
        SpectrogramAnalysis(plotting_interval=(5, 65))
    with pytest.raises(ValueError):
        SpectrogramAnalysis(frequency_limit=(30, 10))
    with pytest.raises(ValueError):
        SpectrogramAnalysis(frequency_limit=(20, 10, 40, 30))

    with pytest.raises(ValueError):
        SpectrogramAnalysis(nperseg_ratio=-0.1)
    with pytest.raises(ValueError):
        SpectrogramAnalysis(nperseg_ratio=0)


def test_call_return_shape():
    analyzer = SpectrogramAnalysis(plotting_interval=(5, 60))
    result = analyzer(signal_input())
    frequencies, times, sxx_list = result

    assert isinstance(frequencies, np.ndarray)
    assert isinstance(times, np.ndarray)
    assert frequencies.shape[0] == 13
    assert times.shape[0] == 76
    assert sxx_list.shape == (2, frequencies.shape[0], times.shape[0])


def test_computing_spectrum() -> None:
    power_analysis = SpectrogramAnalysis()
    signal_gen = signal_input()
    frequencies, times, sxx_list = power_analysis(signal_gen)

    for ch in range(signal_gen.data.shape[1]):
        channel_data = signal_gen.data[:, ch]

        signal_no_bias = channel_data - np.mean(channel_data)
        freqs_expected, times_expected, sxx_expected = spectrogram(
            signal_no_bias,
            fs=signal_gen.rate,
            nperseg=int(signal_gen.rate * power_analysis.nperseg_ratio),
            noverlap=int((int(signal_gen.rate * power_analysis.nperseg_ratio)) / 2),
        )

        np.testing.assert_allclose(frequencies, freqs_expected)
        np.testing.assert_allclose(times, times_expected)
        np.testing.assert_allclose(sxx_list[ch, :, :], sxx_expected)
