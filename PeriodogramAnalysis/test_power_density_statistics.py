import numpy as np
import pytest
from collections import defaultdict
from miv.core.datatype import Signal
from power_density_statistics import (
    SpectrumAnalysisBase,
    SpectrumAnalysisPeriodogram,
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


def test_SpectrumAnalysisBase_compute_psd_not_implemented():
    analyzer = SpectrumAnalysisBase()
    signal = next(mock_signal_generator())
    psd_dict = {}
    with pytest.raises(NotImplementedError) as exc_info:
        analyzer.compute_psd(signal, psd_dict)
    assert str(exc_info.value) == (
        "The compute_psd method is not implemented in the base class. "
        "This base class is not intended for standalone use. Please use a subclass "
        "such as SpectrumAnalysisWelch, SpectrumAnalysisPeriodogram, or SpectrumAnalysisMultitaper."
    )


def test_SpectrumAnalysisBase_call(mocker):
    # Test SpectrumAnalysisBase default
    analyzer = SpectrumAnalysisBase(
        window_length_for_welch=8, band_display=(10, 200), tag="Custom Analysis"
    )
    assert analyzer.window_length_for_welch == 8
    assert analyzer.band_display == (10, 200)
    assert analyzer.tag == "Custom Analysis"

    # Test SpectrumAnalysisBase __call__
    mock_compute_psd = mocker.patch.object(
        SpectrumAnalysisBase, "compute_psd", return_value=defaultdict(dict)
    )
    analyzer = SpectrumAnalysisBase()
    result = analyzer(mock_signal_generator())

    assert mock_compute_psd.call_count == 2
    assert result == defaultdict(dict)

    # Test __call__ of children class
    analysis_wel = SpectrumAnalysisWelch()
    psd_dict_wel = analysis_wel(mock_signal_generator())
    analysis_per = SpectrumAnalysisPeriodogram()
    psd_dict_per = analysis_per(mock_signal_generator())

    psd_dict_list = [psd_dict_wel, psd_dict_per]

    for psd_dict in psd_dict_list:
        assert isinstance(psd_dict, dict)
        assert len(psd_dict) == 2

        # check the content of spec_dict
        for channel, channel_data in psd_dict.items():
            assert "freqs" in channel_data
            assert "psd" in channel_data


def test_SpectrumAnalysisWelch(tmp_path):
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
                assert np.isclose(freqs[idx], exp_freq)
            # Test the psd around peaks comprises more than 90% of total psd
            assert range_psd_sum / total_psd_sum >= 0.9


def test_SpectrumAnalysisPeriodogram(tmp_path):
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
                assert np.isclose(freqs[idx], exp_freq)

            assert range_psd_sum / total_psd_sum >= 0.9


def test_SpectrumAnalysisWelch_compute_psd():
    analyzer = SpectrumAnalysisWelch()
    psd_dict_mock = {
        0: {  # Channel index 0
            "freqs": [0, 4, 9, 14, 19, 24],
            "psd": [0, 0.4, 0.5, 0.4, 0.2, 0.1],
        },
        1: {  # Channel index 1
            "freqs": [0, 5, 10, 15, 20, 25],
            "psd": [0, 0.5, 0.3, 0.2, 0.1, 0],
        },
    }
    signal = next(mock_signal_generator())
    result = analyzer.compute_psd(signal, psd_dict_mock)

    assert isinstance(result, dict)
    assert len(result) == 2  # Two channels
    for channel_index in range(2):  # Two channels
        assert channel_index in result
        assert "freqs" in result[channel_index]
        assert "psd" in result[channel_index]
        assert len(result[channel_index]["freqs"]) == 7
        assert len(result[channel_index]["psd"]) == 7

        # Validate numerical values
        np.testing.assert_array_equal(
            result[channel_index]["freqs"][:6],
            psd_dict_mock[channel_index]["freqs"][:6],
        )
        np.testing.assert_array_equal(
            result[channel_index]["psd"][:6], psd_dict_mock[channel_index]["psd"][:6]
        )

    psd_dict = {}
    result = analyzer.compute_psd(signal, psd_dict)

    assert isinstance(result, dict)
    assert len(result) == 2


def test_SpectrumAnalysisPeriodogram_compute_psd():
    analyzer = SpectrumAnalysisPeriodogram()
    psd_dict_mock = {
        0: {  # Channel index 0
            "freqs": [0, 4, 9, 14, 19, 24],
            "psd": [0, 0.4, 0.5, 0.4, 0.2, 0.1],
        },
        1: {  # Channel index 1
            "freqs": [0, 5, 10, 15, 20, 25],
            "psd": [0, 0.5, 0.3, 0.2, 0.1, 0],
        },
    }
    signal = next(mock_signal_generator())
    result = analyzer.compute_psd(signal, psd_dict_mock)

    assert isinstance(result, dict)
    assert len(result) == 2  # Two channels
    for channel_index in range(2):  # Two channels
        assert channel_index in result
        assert "freqs" in result[channel_index]
        assert "psd" in result[channel_index]
        assert len(result[channel_index]["freqs"]) == 7
        assert len(result[channel_index]["psd"]) == 7

        # Validate numerical values
        np.testing.assert_array_equal(
            result[channel_index]["freqs"][:6],
            psd_dict_mock[channel_index]["freqs"][:6],
        )
        np.testing.assert_array_equal(
            result[channel_index]["psd"][:6], psd_dict_mock[channel_index]["psd"][:6]
        )

    psd_dict = {}
    result = analyzer.compute_psd(signal, psd_dict)

    assert isinstance(result, dict)
    assert len(result) == 2
