from typing import Generator, Any

import pytest
import numpy as np
from scipy.signal import spectrogram
from miv.core.datatype import Signal
from spectrogram_analysis import SpectrogramAnalysis
from test_power_density_statistics import mock_signal_generator


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


def test_call_return_shape(mock_signal_generator):
    analyzer = SpectrogramAnalysis(plotting_interval=(5, 60))
    signal, _ = mock_signal_generator
    result = analyzer(signal)
    frequencies, times, sxx_list = result

    assert isinstance(frequencies, np.ndarray)
    assert isinstance(times, np.ndarray)
    assert frequencies.shape[0] == 13
    assert times.shape[0] == 76
    assert sxx_list.shape == (2, frequencies.shape[0], times.shape[0])


def test_computing_spectrum(mock_signal_generator, mocker) -> None:
    power_analysis = SpectrogramAnalysis()
    signal, _ = mock_signal_generator

    mock_spectrogram = mocker.patch("scipy.signal.spectrogram")

    mock_spectrogram.return_value = (
        np.random.rand(13),
        np.random.rand(76),
        np.random.rand(13, 76),
    )

    frequencies, times, sxx_list = power_analysis(signal)

    assert mock_spectrogram.call_count == 2

    for ch, call in enumerate(mock_spectrogram.call_args_list):
        args, kwargs = call

        expected_signal = signal.data[:, ch] - np.mean(signal.data[:, ch])
        np.testing.assert_array_equal(args[0], expected_signal)

        # Test if the function is called with proper parameters
        assert kwargs["fs"] == signal.rate
        assert kwargs["nperseg"] == int(signal.rate * power_analysis.nperseg_ratio)
        assert kwargs["noverlap"] == int(
            (int(signal.rate * power_analysis.nperseg_ratio)) / 2
        )

    # Test the return
    assert frequencies.shape == (13,)
    assert times.shape == (76,)
    assert sxx_list.shape == (2, 13, 76)
