import numpy as np

from miv.core.datatype import Signal
from power_density_statistics import (
    SpectrumAnalysisPeriodogram,
    SpectrumAnalysisMultitaper,
    SpectrumAnalysisWelch,
)


def mock_signal_generator():
    timestamps = np.linspace(0, 10, 1000, endpoint=False)
    signal = (
        np.sin(2 * np.pi * 5 * timestamps)
        + np.sin(2 * np.pi * 10 * timestamps)
        + np.sin(2 * np.pi * 15 * timestamps)
        + np.sin(2 * np.pi * 20 * timestamps)
    )

    data1 = np.array([signal, signal]).T
    signal1 = Signal(data=data1, timestamps=timestamps, rate=100.0)
    yield signal1

    data2 = np.array([signal, signal]).T
    signal2 = Signal(data=data2, timestamps=timestamps, rate=100.0)
    yield signal2


def test_SpectrumAnalysisBase_call():
    analysis_wel = SpectrumAnalysisWelch()
    psd_dict_wel = analysis_wel(mock_signal_generator())
    analysis_mul = SpectrumAnalysisMultitaper()
    psd_dict_mul = analysis_mul(mock_signal_generator())
    analysis_per = SpectrumAnalysisPeriodogram()
    psd_dict_per = analysis_per(mock_signal_generator())

    psd_dict_list = [psd_dict_wel, psd_dict_mul, psd_dict_per]

    for psd_dict in psd_dict_list:
        assert isinstance(psd_dict, dict)
        assert len(psd_dict) == 2

        # check the content of spec_dict
        for channel, channel_data in psd_dict.items():
            assert "freqs" in channel_data
            assert "psd" in channel_data


def test_SpectrumAnalysisWelch(tmp_path):
    """
    Test if class SpectrumAnalysisWelch can return expected psd_dict and plots.
    """
    analysis = SpectrumAnalysisWelch()
    psd_dict = analysis(mock_signal_generator())

    for channel, channel_data in psd_dict.items():
        for chunk in range(len(psd_dict[channel]["freqs"])):
            freqs = channel_data["freqs"][chunk]
            freqs = np.array(freqs)
            psd = channel_data["psd"][chunk]
            total_psd_sum = np.sum(psd)
            expected_freqs = [5, 10, 15, 20]

            range_psd_sum = 0
            for exp_freq in expected_freqs:
                range_psd_sum += np.sum(
                    psd[(freqs >= exp_freq - 1) & (freqs <= exp_freq + 1)]
                )

                idx = np.argmin(np.abs(freqs - exp_freq))
                # Test the peaks of psd is as expected
                assert freqs[idx] == exp_freq or np.isclose(
                    freqs[idx], exp_freq, atol=0.1
                )
            # Test the psd around peaks comprises more than 90% of total psd
            assert range_psd_sum / total_psd_sum >= 0.9


def test_SpectrumAnalysisPeriodogram(tmp_path):
    """
    Test if class SpectrumAnalysisPeriodogram can return expected psd_dict and plots.
    """
    analysis = SpectrumAnalysisPeriodogram()
    psd_dict = analysis(mock_signal_generator())

    for channel, channel_data in psd_dict.items():
        for chunk in range(len(psd_dict[channel]["freqs"])):
            freqs = channel_data["freqs"][chunk]
            freqs = np.array(freqs)
            psd = channel_data["psd"][chunk]
            total_psd_sum = np.sum(psd)
            expected_freqs = [5, 10, 15, 20]

            range_psd_sum = 0
            for exp_freq in expected_freqs:
                range_psd_sum += np.sum(
                    psd[(freqs >= exp_freq - 1) & (freqs <= exp_freq + 1)]
                )

                idx = np.argmin(np.abs(freqs - exp_freq))
                assert freqs[idx] == exp_freq or np.isclose(
                    freqs[idx], exp_freq, atol=0.1
                )

            assert range_psd_sum / total_psd_sum >= 0.9


def test_SpectrumAnalysisMultitaper(tmp_path):
    """
    Test if class SpectrumAnalysisMultitaper can return expected psd_dict and plots.
    """
    analysis = SpectrumAnalysisMultitaper()
    psd_dict = analysis(mock_signal_generator())

    for channel, channel_data in psd_dict.items():
        for chunk in range(len(psd_dict[channel]["freqs"])):
            freqs = channel_data["freqs"][chunk]
            freqs = np.array(freqs)
            psd = channel_data["psd"][chunk]
            total_psd_sum = np.sum(psd)
            expected_freqs = [5, 10, 15, 20]

            range_psd_sum = 0
            for exp_freq in expected_freqs:
                range_psd_sum += np.sum(
                    psd[(freqs >= exp_freq - 1) & (freqs <= exp_freq + 1)]
                )

                idx = np.argmin(np.abs(freqs - exp_freq))
                assert freqs[idx] == exp_freq or np.isclose(
                    freqs[idx], exp_freq, atol=0.1
                )

            assert range_psd_sum / total_psd_sum >= 0.9
