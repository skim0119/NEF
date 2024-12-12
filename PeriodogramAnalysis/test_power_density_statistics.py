import numpy as np
import pytest
from miv.core.datatype import Signal
from power_density_statistics import (
    SpectrumAnalysisBase,
    SpectrumAnalysisPeriodogram,
    SpectrumAnalysisWelch,
)


@pytest.fixture
def initialize_module():
    expected_freqs = [5, 10, 15, 20]

    def generate_signal(num_channel):
        timestamps = np.linspace(0, 10, 1000, endpoint=False)
        signal = (
            np.sin(2 * np.pi * 5 * timestamps)
            + np.sin(2 * np.pi * 10 * timestamps)
            + np.sin(2 * np.pi * 15 * timestamps)
            + np.sin(2 * np.pi * 20 * timestamps)
        )
        np.random.seed(42)
        noise = np.random.normal(loc=0, scale=0.5, size=signal.shape)
        signal += noise
        data = np.empty((0, len(signal)))
        for ch in range(num_channel):
            data = np.vstack((data, (ch + 2) * signal))
        signal = Signal(data=data.T, timestamps=timestamps, rate=100)
        return signal, expected_freqs

    def module_init(module_name, num_channel):
        analyzer = module_name(window_size_for_welch=4)
        signal, expected_freqs = generate_signal(num_channel=num_channel)
        freqs, psd = analyzer(signal)

        return freqs, psd, expected_freqs

    return module_init


def test_call_invalid_input_parameters(initialize_module, num_channel=1):
    # Test wrong input parameters
    with pytest.raises(NotImplementedError):
        initialize_module(SpectrumAnalysisBase, num_channel=num_channel)

    with pytest.raises(ValueError):
        SpectrumAnalysisWelch(window_size_for_welch=-1)

    with pytest.raises(ValueError):
        SpectrumAnalysisWelch(window_size_for_welch=0.75)

    with pytest.raises(ValueError):
        SpectrumAnalysisWelch(window_size_for_welch=0)


@pytest.mark.parametrize(
    "analysis_class",
    [SpectrumAnalysisWelch, SpectrumAnalysisPeriodogram],
)
def test_none_input(analysis_class):
    # Test empty input signal
    with pytest.raises(Exception):
        analyzer = analysis_class()
        analyzer(None)


"""
Welch
Δf = sampling_rate / nfft = 100 / 1600 = 0.0625 Hz
Frequency Range = [0, fs/2] = [0, 50] Hz
Number of Frequency Points = (50 - 0) / Δf + 1 = 801

periodogram
Signal length = nfft = 1000
Length of freqs = nfft / 2 + 1 = 501
More details is here: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html>
"""


@pytest.mark.parametrize(
    "analysis_class_and_shape",
    [(SpectrumAnalysisWelch, 801), (SpectrumAnalysisPeriodogram, 501)],
)
def test_signal_first_channel(
    analysis_class_and_shape, initialize_module, num_channel=1
):
    analysis_class, expected_shape = analysis_class_and_shape
    freqs, psd, _ = initialize_module(analysis_class, num_channel=num_channel)

    assert isinstance(freqs, np.ndarray)
    assert isinstance(psd, np.ndarray)

    assert freqs.shape[0] == expected_shape
    assert psd.shape[0] == expected_shape


@pytest.mark.parametrize(
    "analysis_class", [SpectrumAnalysisWelch, SpectrumAnalysisPeriodogram]
)
@pytest.mark.parametrize("num_channel", [1, 4, 7, 9])
def test_signal_multiple_channel(initialize_module, analysis_class, num_channel):
    freqs, psd, _ = initialize_module(analysis_class, num_channel=num_channel)
    assert psd.shape[1] == num_channel


@pytest.mark.parametrize(
    "analysis_class",
    [SpectrumAnalysisWelch, SpectrumAnalysisPeriodogram],
)
def test_spectrum_analysis_computation(analysis_class, initialize_module) -> None:
    freqs, psd, expected_freqs = initialize_module(analysis_class, num_channel=1)

    total_psd_sum = np.sum(psd)

    range_psd_sum = 0
    for exp_freq in expected_freqs:

        # Sum up psd around peaks (within +-0.5Hz)
        range_psd_sum += np.sum(
            psd[(freqs >= exp_freq - 0.5) & (freqs <= exp_freq + 0.5)]
        )
        idx = np.argmin(np.abs(freqs - exp_freq))
        spike_psd = psd[idx]

        # Test the peaks of psd is as expected
        assert np.isclose(freqs[idx], exp_freq)

    # Test the psd around peaks comprises more than 80% of total psd, if test
    # doesn't pass, need to decrease threshold
    assert range_psd_sum / total_psd_sum >= 0.8

    # Test psd value out of the range (+-0.5Hz) above is at most 5% of the peak value
    for i, freq in enumerate(freqs):
        if all(abs(freq - exp_freq) > 0.5 for exp_freq in expected_freqs):
            assert psd[i] < 0.05 * spike_psd


@pytest.mark.parametrize(
    "analysis_class", [SpectrumAnalysisWelch, SpectrumAnalysisPeriodogram]
)
@pytest.mark.parametrize("rate", [50, 200, 500])
def test_different_sampling_rates(analysis_class, rate):
    timestamps = np.linspace(0, 10, rate * 10, endpoint=False)
    signal_data = np.sin(2 * np.pi * 5 * timestamps)
    data = np.array([signal_data, signal_data]).T
    signal = Signal(data=data, timestamps=timestamps, rate=rate)

    analyzer = analysis_class()
    freqs, psd = analyzer(signal)

    assert freqs.max() == rate / 2


@pytest.mark.parametrize(
    "analysis_class", [SpectrumAnalysisWelch, SpectrumAnalysisPeriodogram]
)
def test_signal_shorter_than_window(analysis_class):
    timestamps = np.linspace(0, 0.1, 10, endpoint=False)
    signal_data = np.sin(2 * np.pi * 5 * timestamps)
    data = np.array([signal_data, signal_data]).T
    signal = Signal(data=data, timestamps=timestamps, rate=100)

    analyzer = analysis_class(window_size_for_welch=100)
    with pytest.raises(Exception):
        analyzer(signal)

    analyzer = analysis_class(window_size_for_welch=10)
    with pytest.raises(Exception):
        analyzer(signal)
