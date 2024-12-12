from typing import Any, Optional, Tuple

import scipy
import pytest
import numpy as np
from spectrum_analysis import PowerSpectrumAnalysis
from scipy.integrate import simpson


@pytest.fixture
def initialize_module():
    def psd_input(num_channel) -> tuple:
        timestamps = np.linspace(0, 10, 1000, endpoint=False)
        base_frequencies = np.linspace(1, 50, num_channel)
        psd_list: list = []

        for i, freq in enumerate(base_frequencies):
            signal = np.sin(2 * np.pi * freq * timestamps)
            signal_no_bias = signal - np.mean(signal)
            freqs, psd_channel = scipy.signal.welch(
                signal_no_bias, fs=100, nperseg=400, nfft=1600
            )
            psd_list.append(psd_channel)

        return freqs, np.array(psd_list).T

    def module_init(num_channel, band=((0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100))):
        analyzer = PowerSpectrumAnalysis(band_list=band)
        result = analyzer(psd_input(num_channel))
        freqs, psd, psd_idx = result

        return freqs, psd, psd_idx

    return module_init


def test_call_invalid_input_parameters():
    with pytest.raises(ValueError):
        PowerSpectrumAnalysis(band_display=(5, 10, 20, 30))
    with pytest.raises(ValueError):
        PowerSpectrumAnalysis(band_display=(-2, 5))

    with pytest.raises(ValueError):
        PowerSpectrumAnalysis(band_display=((8, 5)))
    with pytest.raises(ValueError):
        PowerSpectrumAnalysis(band_display=((-3, 5)))
    with pytest.raises(ValueError):
        PowerSpectrumAnalysis(band_display=((8), (12, 30), (30, 100)))


def test_call_empty_input():
    input = (np.array([]), np.array([]))

    with pytest.raises(ValueError):
        analysis = PowerSpectrumAnalysis()
        analysis(input)


"""
Welch
Δf = sampling_rate / nfft = 100 / 1600 = 0.0625 Hz
Frequency Range = [0, fs/2] = [0, 50] Hz
Number of Frequency Points = (50 - 0) / Δf + 1 = 801
"""


def test_call_single_channel(initialize_module, num_channel=1):
    freqs, psd, psd_idx = initialize_module(num_channel=num_channel)

    assert isinstance(freqs, np.ndarray)
    assert isinstance(psd, np.ndarray)
    assert freqs.shape[0] == 801
    assert psd.shape[0] == 801
    assert psd.shape[1] == num_channel
    # There are 5 bands
    assert psd_idx.shape == (801, 5 * num_channel)


@pytest.mark.parametrize("num_channel", [2, 4, 7, 9])
def test_call_multiple_channels(initialize_module, num_channel):
    freqs, psd, psd_idx = initialize_module(num_channel=num_channel)

    assert psd.shape[1] == num_channel
    # There are 5 bands
    assert psd_idx.shape == (801, 5 * num_channel)


def test_call_empty_band_list(initialize_module, num_channel=1) -> None:
    freqs, psd, psd_idx = initialize_module(num_channel=num_channel, band=())
    assert freqs.shape[0] == 801
    assert psd.shape[0] == 801
    assert psd.shape[1] == num_channel
    assert psd_idx.shape == (0,)


def test_computing_absolute_and_relative_power_first_channel(
    initialize_module, mocker, num_channel=1
) -> None:
    analysis = PowerSpectrumAnalysis()
    freqs, psd, psd_idx = initialize_module(num_channel=num_channel)

    mocker.patch.object(analysis, "num_channel", num_channel)
    psd_idx, power, rel_power = analysis.computing_absolute_and_relative_power(
        freqs, psd
    )

    freq_res = freqs[1] - freqs[0]
    total_power = simpson(psd, dx=freq_res, axis=0)

    for i, band in enumerate(analysis.band_list):
        psd_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        manual_power = simpson(psd[psd_idx], dx=freq_res, axis=0)
        computed_power = power[i]
        computed_rel_power = rel_power[i]

        assert np.isclose(manual_power, computed_power)

        manual_rel_power = manual_power / total_power
        assert np.isclose(manual_rel_power, computed_rel_power)


@pytest.mark.parametrize("num_channel", [2, 4, 7, 9])
def test_computing_absolute_and_relative_power_multi_channel(
    initialize_module, mocker, num_channel
) -> None:
    analysis = PowerSpectrumAnalysis()
    freqs, psd, psd_idx = initialize_module(num_channel=num_channel)

    mocker.patch.object(analysis, "num_channel", num_channel)
    psd_idx, power, rel_power = analysis.computing_absolute_and_relative_power(
        freqs, psd
    )

    # There are 5 bands
    assert power.shape == (num_channel * 5,)
    assert rel_power.shape == (num_channel * 5,)


def test_band_out_of_frequency_range() -> None:
    # Input frequency has max range 100 Hz, while band to be analysed is 200-250Hz
    analysis = PowerSpectrumAnalysis(band_list=((200, 250),))

    freqs = np.linspace(0, 100, 1000)
    psd = np.random.rand(1000, 1)

    psd_idx, power, rel_power = analysis.computing_absolute_and_relative_power(
        freqs, psd
    )

    assert np.all(power == 0)
    assert np.all(rel_power == 0)


@pytest.fixture
def initiate_computing_ratio_and_bandpower(initialize_module):
    def initiate_func(num_channel, mocker):
        analysis = PowerSpectrumAnalysis()
        freqs, psd, psd_idx = initialize_module(num_channel=num_channel)

        mock_logger = mocker.patch.object(analysis, "logger")
        mocker.patch.object(analysis, "num_channel", num_channel)
        mocker.patch.object(analysis, "chunk", 0)

        psd_idx, power, rel_power = analysis.computing_absolute_and_relative_power(
            freqs, psd
        )

        absolute_powers, relative_powers = analysis.computing_ratio_and_bandpower(
            power, rel_power
        )
        return power, rel_power, absolute_powers, relative_powers, mock_logger

    return initiate_func


def test_computing_ratio_and_bandpower_first_channel(
    initiate_computing_ratio_and_bandpower, mocker, num_channel=1
) -> None:
    power, rel_power, absolute_powers, relative_powers, _ = (
        initiate_computing_ratio_and_bandpower(num_channel=num_channel, mocker=mocker)
    )

    np.testing.assert_allclose(absolute_powers, power)
    np.testing.assert_allclose(relative_powers, rel_power)


@pytest.mark.parametrize("num_channel", [2, 4, 7, 9])
def test_computing_ratio_and_bandpower_multiple_channel(
    initiate_computing_ratio_and_bandpower, mocker, num_channel
) -> None:
    power, rel_power, absolute_powers, relative_powers, _ = (
        initiate_computing_ratio_and_bandpower(num_channel=num_channel, mocker=mocker)
    )

    np.testing.assert_allclose(absolute_powers, power)
    np.testing.assert_allclose(relative_powers, rel_power)


@pytest.mark.parametrize("num_channel", [1, 2, 4, 7, 9])
def test_computing_ratio_and_bandpower_logger_call(
    initiate_computing_ratio_and_bandpower, mocker, num_channel
) -> None:
    _, _, _, _, mock_logger = initiate_computing_ratio_and_bandpower(
        num_channel=num_channel, mocker=mocker
    )
    mock_logger.info.assert_called()
    assert mock_logger.info.call_count == num_channel * 11
