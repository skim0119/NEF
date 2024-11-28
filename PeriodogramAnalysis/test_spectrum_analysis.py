from typing import Any, Optional, Tuple

from pytest_mock import MockerFixture
import numpy as np
from spectrum_analysis import PowerSpectrumAnalysis
from scipy.integrate import simpson


def psd_input() -> dict[int, dict[str, Any]]:
    psd_dict = {
        0: {  # channel index
            "freqs": [
                np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50]),  # chunk 0
                np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50]),  # chunk 1
            ],
            "psd": [
                np.array(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
                ),  # chunk 0
                np.array(
                    [
                        0.2,
                        0.25,
                        0.3,
                        0.35,
                        0.4,
                        0.425,
                        0.45,
                        0.475,
                        0.5,
                        0.525,
                        0.55,
                        0.575,
                    ]
                ),  # chunk 1
            ],
        },
        1: {  # channel index
            "freqs": [
                np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50]),  # chunk 0
                np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50]),  # chunk 1
            ],
            "psd": [
                np.array(
                    [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
                ),  # chunk 0
                np.array(
                    [
                        0.575,
                        0.55,
                        0.525,
                        0.5,
                        0.475,
                        0.45,
                        0.425,
                        0.4,
                        0.35,
                        0.3,
                        0.25,
                        0.2,
                    ]
                ),  # chunk 1
            ],
        },
    }

    return psd_dict


def test_power_spectrum_analysis_call() -> None:
    """
    Test PowerSpectrumAnalysis class to ensure it computes the power_dict correctly
    based on the provided psd_dict, and returns the expected values.
    """
    psd_dict = psd_input()
    analysis = PowerSpectrumAnalysis()

    psd_dict, power_dict = analysis(psd_dict)
    assert psd_dict == psd_dict

    # Test the content of power_dict
    for channel, channel_data in power_dict.items():
        assert "psd_idx" in channel_data
        assert "power_list" in channel_data
        assert "rel_power_list" in channel_data

        assert len(power_dict[channel]["psd_idx"]) == 10
        assert len(power_dict[channel]["power_list"]) == 10
        assert len(power_dict[channel]["rel_power_list"]) == 10

        for power in power_dict[channel]["power_list"]:
            assert power >= 0
        for rel_power in power_dict[channel]["rel_power_list"]:
            assert rel_power >= 0
            assert rel_power <= 1


def test_computing_absolute_and_relative_power() -> None:
    """
    Test the computing_absolute_and_relative_power method.
    """

    power_analysis = PowerSpectrumAnalysis(
        band_list=((1, 5), (5, 15), (15, 35), (35, 50))
    )

    psd_dict = psd_input()
    power_dict: dict[str, Any] = {
        "psd_idx": [],
        "power_list": [],
        "rel_power_list": [],
    }

    result_power_dict = power_analysis.computing_absolute_and_relative_power(
        psd_dict[0], power_dict
    )

    freqs = psd_dict[0]["freqs"][0]
    psd = psd_dict[0]["psd"][0]
    freq_res = freqs[1] - freqs[0]
    total_power = simpson(psd, dx=freq_res)

    for i, band in enumerate(power_analysis.band_list):
        psd_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        manual_power = simpson(psd[psd_idx], dx=freq_res)
        computed_power = result_power_dict["power_list"][i]
        computed_rel_power = result_power_dict["rel_power_list"][i]

        assert np.isclose(manual_power, computed_power)

        manual_rel_power = manual_power / total_power
        assert np.isclose(manual_rel_power, computed_rel_power)


def test_computing_ratio_and_bandpower(mocker: MockerFixture) -> None:
    """
    Test computing_ratio_and_bandpower, test how many logger are called.
    """
    channel_idx = 0
    analysis = PowerSpectrumAnalysis()
    psd_dict = psd_input()
    psd_dict, power_dict = analysis(psd_dict)

    logger_info_spy = mocker.spy(analysis.logger, "info")
    analysis.computing_ratio_and_bandpower(
        psd_dict[channel_idx], power_dict[channel_idx], channel_idx
    )

    # Test how many times logger is called
    assert logger_info_spy.call_count == 11 * 2
