import numpy as np
from spectrum_analysis import PowerSpectrumAnalysis


def mock_psd_list():
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


def test_power_spectrum_analysis_call_and_power_computation():
    """
    Test PowerSpectrumAnalysis class to ensure it computes the power_dict correctly
    based on the provided psd_dict, and returns the expected values.
    """
    psd_dict = mock_psd_list()
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


def test_computing_ratio_and_bandpower(mocker):
    """
    Test computing_ratio_and_bandpower, test how many logger are called.
    """
    channel_idx = 0
    analysis = PowerSpectrumAnalysis()
    psd_dict = mock_psd_list()
    psd_dict, power_dict = analysis(psd_dict)

    logger_info_spy = mocker.spy(analysis.logger, "info")
    analysis.computing_ratio_and_bandpower(
        psd_dict[channel_idx], power_dict[channel_idx], channel_idx
    )

    # Test how many times logger is called
    assert logger_info_spy.call_count == 21 * 2


def test_computing_bandpower():
    """
    Test test_computing_bandpower, test how many logger are called.
    """
    analysis = PowerSpectrumAnalysis()
    psd_dict = mock_psd_list()
    psd_dict, power_dict = analysis(psd_dict)
    band_list: tuple = ((0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100))

    for channel, channel_data in psd_dict.items():
        for chunk in range(len(psd_dict[channel]["freqs"])):
            for band in band_list:
                # Check common band power computation
                bandpower = analysis.computing_bandpower(
                    psd_dict[channel]["freqs"][chunk],
                    psd_dict[channel]["psd"][chunk],
                    band,
                )
                assert isinstance(bandpower, float)
                assert bandpower >= 0
                # Test relative power computation
                relative_bandpower = analysis.computing_bandpower(
                    psd_dict[channel]["freqs"][chunk],
                    psd_dict[channel]["psd"][chunk],
                    band,
                    relative=True,
                )
                assert isinstance(relative_bandpower, float)
                assert relative_bandpower >= 0
                assert relative_bandpower <= 1
                assert bandpower != relative_bandpower
