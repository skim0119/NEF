import numpy as np

from miv.core.datatype import Signal
from power_density_statistics import (
    SpectrumAnalysisPeriodogram,
    SpectrumAnalysisMultitaper,
    SpectrumAnalysisWelch,
)


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


def test_SpectrumAnalysisWelch(tmp_path):
    """
    Test if class SpectrumAnalysisWelch can return expected psd_dict and plots.
    """
    analysis = SpectrumAnalysisWelch()
    psd_dict = analysis(mock_signal_generator())

    assert isinstance(psd_dict, dict)
    assert len(psd_dict) == 2

    # check the content of spec_dict
    for channel, channel_data in psd_dict.items():
        assert "freqs" in channel_data
        assert "psd" in channel_data

    analysis.plot_spectrum(output=psd_dict, input=None, show=False, save_path=tmp_path)


def test_SpectrumAnalysisPeriodogram(tmp_path):
    """
    Test if class SpectrumAnalysisPeriodogram can return expected psd_dict and plots.
    """
    analysis = SpectrumAnalysisPeriodogram()
    psd_dict = analysis(mock_signal_generator())

    assert isinstance(psd_dict, dict)
    assert len(psd_dict) == 2

    # check the content of spec_dict
    for channel, channel_data in psd_dict.items():
        assert "freqs" in channel_data
        assert "psd" in channel_data


def test_SpectrumAnalysisMultitaper(tmp_path):
    """
    Test if class SpectrumAnalysisMultitaper can return expected psd_dict and plots.
    """
    analysis = SpectrumAnalysisMultitaper()
    psd_dict = analysis(mock_signal_generator())

    assert isinstance(psd_dict, dict)
    assert len(psd_dict) == 2

    # check the content of spec_dict
    for channel, channel_data in psd_dict.items():
        assert "freqs" in channel_data
        assert "psd" in channel_data
