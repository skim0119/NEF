from typing import Generator, Any

import pytest
import numpy as np
from spectrogram_analysis import SpectrogramAnalysis


@pytest.fixture
def initialize_spectrogram(generate_signal):
    def module_init(num_channel):
        analyzer = SpectrogramAnalysis()
        signal, expected_freqs = generate_signal(num_channel=num_channel)
        frequencies, times, sxx_list = analyzer(signal)

        return frequencies, times, sxx_list, expected_freqs, signal

    return module_init


def test_call_invalid_input_parameters():
    # Test improper input parameter
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


def test_none_input():
    # Test empty input signal
    with pytest.raises(Exception):
        analyzer = SpectrogramAnalysis()
        analyzer(None)


def test_call_first_channel(initialize_spectrogram, num_channel=1):
    # Test shape of output for a single channel signal
    frequencies, times, sxx_list, *rest = initialize_spectrogram(
        num_channel=num_channel
    )
    """
    nperseg_ratio = 0.25.
    nperseg = int(100 * 0.25) = 25 points per segment.
    noverlap = int(nperseg / 2):
    noverlap = int(25 / 2) = 12.
    frequencies.shape[0] = nperseg / 2 + 1 = 13
    M= (signal_length−noverlap) / (nperseg−noverlap) = 988 / 13 = 76
    """

    assert isinstance(frequencies, np.ndarray)
    assert isinstance(times, np.ndarray)
    assert frequencies.shape[0] == 13
    assert times.shape[0] == 76
    assert sxx_list.shape == (1, frequencies.shape[0], times.shape[0])


@pytest.mark.parametrize("num_channel", [2, 4, 6, 8])
def test_call_multiple_channel(initialize_spectrogram, num_channel):
    # Test shape of output for a multiple channel signal
    frequencies, times, sxx_list, *rest = initialize_spectrogram(
        num_channel=num_channel
    )
    assert sxx_list.shape == (num_channel, frequencies.shape[0], times.shape[0])


@pytest.fixture
def mock_spectrogram(mocker):
    # Fixture to mock `scipy.signal.spectrogram`
    mock = mocker.patch("scipy.signal.spectrogram")
    mock.return_value = (
        np.ones(13),
        np.ones(76),
        np.ones([13, 76]),
    )
    return mock


@pytest.mark.parametrize("num_channel", [1, 2, 4, 6, 8])
def test_computing_spectrum_computed_signal(
    mock_spectrogram, num_channel, generate_signal
) -> None:
    analyzer = SpectrogramAnalysis()
    signal, expected_freqs = generate_signal(num_channel=num_channel)

    analyzer(signal)
    assert mock_spectrogram.call_count == num_channel

    # Test input signal
    for ch, call in enumerate(mock_spectrogram.call_args_list):
        args, kwargs = call

        expected_signal = signal.data[:, ch] - np.mean(signal.data[:, ch])
        np.testing.assert_array_equal(args[0], expected_signal)


@pytest.mark.parametrize("num_channel", [1, 2, 4, 6, 8])
def test_computing_spectrum(mock_spectrogram, num_channel, generate_signal) -> None:
    analyzer = SpectrogramAnalysis()
    signal, expected_freqs = generate_signal(num_channel=num_channel)

    frequencies, times, sxx_list = analyzer(signal)
    assert mock_spectrogram.call_count == num_channel

    for ch, call in enumerate(mock_spectrogram.call_args_list):
        args, kwargs = call

        # Test if the function is called with proper parameters
        assert kwargs["fs"] == signal.rate
        assert kwargs["nperseg"] == int(signal.rate * analyzer.nperseg_ratio)
        assert kwargs["noverlap"] == int(
            (int(signal.rate * analyzer.nperseg_ratio)) / 2
        )

    # Test return values
    assert frequencies.shape == (13,)
    assert times.shape == (76,)
    assert sxx_list.shape == (num_channel, 13, 76)
