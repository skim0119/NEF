from typing import Generator, Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from miv.core.datatype import Signal
from power_density_statistics import (
    SpectrumAnalysisBase,
    SpectrumAnalysisPeriodogram,
    SpectrumAnalysisWelch,
)


@pytest.fixture
def mock_signal_generator():
    timestamps = np.linspace(0, 10, 1000, endpoint=False)
    signal = (
        np.sin(2 * np.pi * 5 * timestamps)
        + np.sin(2 * np.pi * 10 * timestamps)
        + np.sin(2 * np.pi * 15 * timestamps)
        + np.sin(2 * np.pi * 20 * timestamps)
    )
    noise = np.random.normal(loc=0, scale=0.5, size=signal.shape)
    signal += noise
    data1 = np.array([signal, 2 * signal]).T
    signal = Signal(data=data1, timestamps=timestamps, rate=100)
    expected_freqs = [5, 10, 15, 20]
    return signal, expected_freqs


def test_call_invalid_input_parameters(mock_signal_generator):
    signal, _ = mock_signal_generator
    with pytest.raises(NotImplementedError):
        analyzer = SpectrumAnalysisBase(window_length_for_welch=4)
        analyzer(signal)

    with pytest.raises(ValueError):
        SpectrumAnalysisWelch(window_length_for_welch=-1)

    with pytest.raises(ValueError):
        SpectrumAnalysisWelch(window_length_for_welch=0.75)


def test_call_return_shape(mock_signal_generator):
    analyzer = SpectrumAnalysisWelch(window_length_for_welch=4)
    signal, _ = mock_signal_generator
    result = analyzer(signal)
    freqs, psd = result

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(freqs, np.ndarray)
    assert isinstance(psd, np.ndarray)
    assert freqs.shape[0] == 801
    assert psd.shape[1] == 2


@pytest.mark.parametrize(
    "analysis_class",
    [SpectrumAnalysisWelch, SpectrumAnalysisPeriodogram],
)
def test_spectrum_analysis_computation(analysis_class, mock_signal_generator) -> None:
    analysis = analysis_class()
    signal, expected_freqs = mock_signal_generator
    freqs, psd = analysis(signal)

    for channel in range(2):
        psd_channel = psd[:, channel]
        total_psd_sum = np.sum(psd_channel)

        range_psd_sum = 0
        for exp_freq in expected_freqs:
            range_psd_sum += np.sum(
                psd[(freqs >= exp_freq - 0.5) & (freqs <= exp_freq + 0.5)]
            )
            idx = np.argmin(np.abs(freqs - exp_freq))
            spike_psd = psd[idx, channel]
            # Test the peaks of psd is as expected
            assert np.isclose(freqs[idx], exp_freq)
        # Test the psd around peaks comprises more than 90% of total psd
        assert range_psd_sum / total_psd_sum >= 0.9

        for i, freq in enumerate(freqs):
            if all(abs(freq - exp_freq) > 0.5 for exp_freq in expected_freqs):
                assert psd[i, channel] < 0.05 * spike_psd


@pytest.mark.parametrize(
    "analysis_class, rate",
    [
        (SpectrumAnalysisWelch, 50),
        (SpectrumAnalysisWelch, 200),
        (SpectrumAnalysisWelch, 500),
        (SpectrumAnalysisPeriodogram, 50),
        (SpectrumAnalysisPeriodogram, 200),
        (SpectrumAnalysisPeriodogram, 500),
    ],
)
def test_different_sampling_rates(analysis_class, rate):
    timestamps = np.linspace(0, 10, rate * 10, endpoint=False)
    signal_data = np.sin(2 * np.pi * 5 * timestamps)
    data = np.array([signal_data, signal_data]).T
    signal = Signal(data=data, timestamps=timestamps, rate=rate)

    analyzer = analysis_class(window_length_for_welch=4)
    freqs, psd = analyzer(signal)

    assert freqs.max() == rate / 2
