import numpy as np
import pytest
from scipy.integrate import simpson

from miv.core.datatype import Signal
from spectrum_analysis import PowerSpectrumAnalysis
from spectrogram_analysis import SpectrogramAnalysis
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
    ).T  # 形状为 (100, 2)
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


def mock_psd_list():
    psd_dict = {
        0: {  # chunk index
            0: {  # channel index
                "freqs": np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50]),
                "psd": np.array(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
                ),
            },
            1: {  # channel index
                "freqs": np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50]),
                "psd": np.array(
                    [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
                ),
            },
        },
        1: {  # chunk index (新添加的 chunk 1)
            0: {  # channel index
                "freqs": np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50]),
                "psd": np.array(
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
                ),
            },
            1: {  # channel index
                "freqs": np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50]),
                "psd": np.array(
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
                ),
            },
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
    for chunk_idx, chunk_data in power_dict.items():
        for channel_idx, channel_data in chunk_data.items():
            assert "psd_idx" in channel_data
            assert "power_list" in channel_data
            assert "rel_power_list" in channel_data

            for power in channel_data["power_list"]:
                assert power >= 0
            for rel_power in channel_data["rel_power_list"]:
                assert rel_power >= 0
                assert rel_power <= 1


def test_plot_periodogram(tmp_path):
    """
    Test plot_periodogram, see if it will generate expected plots.
    """
    analysis = PowerSpectrumAnalysis()
    psd_dict = mock_psd_list()
    psd_dict, power_dict = analysis(psd_dict)
    output = (psd_dict, power_dict)

    analysis.plot_periodogram(output=output, input=None, show=False, save_path=tmp_path)

    # Check if plots are saved correctly for each chunk and channel
    for chunk in psd_dict.keys():
        for channel in psd_dict[chunk].keys():
            channel_folder = tmp_path / f"channel{channel:03d}"
            plot_file = channel_folder / f"periodogram_{chunk:03d}.png"
            assert plot_file.exists()


def test_computing_ratio_and_bandpower(mocker):
    """
    Test computing_ratio_and_bandpower, test how many logger are called.
    """
    chunk_idx = 0
    analysis = PowerSpectrumAnalysis()
    psd_dict = mock_psd_list()
    psd_dict, power_dict = analysis(psd_dict)

    logger_info_spy = mocker.spy(analysis.logger, "info")
    analysis.computing_ratio_and_bandpower(
        psd_dict[chunk_idx], power_dict[chunk_idx], chunk_idx
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

    for chunk in psd_dict.keys():
        for channel in psd_dict[chunk].keys():
            for band in band_list:
                # Check common band power computation
                bandpower = analysis.computing_bandpower(psd_dict[chunk][channel], band)
                assert isinstance(bandpower, float)
                assert bandpower >= 0
                # Test relative power computation
                relative_bandpower = analysis.computing_bandpower(
                    psd_dict[chunk][channel], band, relative=True
                )
                assert isinstance(relative_bandpower, float)
                assert relative_bandpower >= 0
                assert relative_bandpower <= 1
                assert bandpower != relative_bandpower


def test_SpectrumAnalysis_call_and_computing_spectrum():
    """
    Test call func and computing_spectrum, check if spec_dict is as expected.
    """
    spectrum_analysis = SpectrogramAnalysis()
    spec_dict = spectrum_analysis(mock_signal_generator())

    # test structure of spec_dict
    assert isinstance(spec_dict, dict)
    assert len(spec_dict) == 2

    # check the content of spec_dict
    for chunk_index in spec_dict.keys():
        for channel_index in spec_dict[chunk_index].keys():
            channel_data = spec_dict[chunk_index][channel_index]
            assert "frequencies" in channel_data
            assert "times" in channel_data
            assert "Sxx" in channel_data


def test_plot_spectrogram(tmp_path):
    """
    Test plot_spectrogram, see if it will generate expected plots.
    """
    spectrum_analysis = SpectrogramAnalysis()
    spec_dict = spectrum_analysis(mock_signal_generator())

    spectrum_analysis.plot_spectrogram(
        output=spec_dict, input=None, show=False, save_path=tmp_path
    )

    # Check if plots are saved correctly for each chunk and channel
    for chunk in spec_dict.keys():
        for channel in spec_dict[chunk].keys():
            channel_folder = tmp_path / f"channel{channel:03d}"
            plot_file = channel_folder / f"spectrogram_{chunk:03d}.png"
            assert plot_file.exists()


def test_SpectrumAnalysisWelch(tmp_path):
    """
    Test if class SpectrumAnalysisWelch can return expected psd_dict and plots.
    """
    analysis = SpectrumAnalysisWelch()
    psd_dict = analysis(mock_signal_generator())

    assert isinstance(psd_dict, dict)
    assert len(psd_dict) == 2

    # check the content of spec_dict
    for chunk_index in psd_dict.keys():
        for channel_index in psd_dict[chunk_index].keys():
            channel_data = psd_dict[chunk_index][channel_index]
            assert "freqs" in channel_data
            assert "psd" in channel_data

    analysis.plot_spectrum(output=psd_dict, input=None, show=False, save_path=tmp_path)

    # Check if plots are saved correctly for each chunk and channel
    for chunk in psd_dict.keys():
        for channel in psd_dict[chunk].keys():
            channel_folder = tmp_path / f"channel{channel:03d}"
            plot_file = channel_folder / f"power_density_{chunk:03d}.png"
            assert plot_file.exists()


def test_SpectrumAnalysisPeriodogram(tmp_path):
    """
    Test if class SpectrumAnalysisPeriodogram can return expected psd_dict and plots.
    """
    analysis = SpectrumAnalysisPeriodogram()
    psd_dict = analysis(mock_signal_generator())

    assert isinstance(psd_dict, dict)
    assert len(psd_dict) == 2

    # check the content of spec_dict
    for chunk_index in psd_dict.keys():
        for channel_index in psd_dict[chunk_index].keys():
            channel_data = psd_dict[chunk_index][channel_index]
            assert "freqs" in channel_data
            assert "psd" in channel_data

    analysis.plot_spectrum(output=psd_dict, input=None, show=False, save_path=tmp_path)

    # Check if plots are saved correctly for each chunk and channel
    for chunk in psd_dict.keys():
        for channel in psd_dict[chunk].keys():
            channel_folder = tmp_path / f"channel{channel:03d}"
            plot_file = channel_folder / f"power_density_{chunk:03d}.png"
            assert plot_file.exists()


def test_SpectrumAnalysisMultitaper(tmp_path):
    """
    Test if class SpectrumAnalysisMultitaper can return expected psd_dict and plots.
    """
    analysis = SpectrumAnalysisMultitaper()
    psd_dict = analysis(mock_signal_generator())

    assert isinstance(psd_dict, dict)
    assert len(psd_dict) == 2

    # check the content of spec_dict
    for chunk_index in psd_dict.keys():
        for channel_index in psd_dict[chunk_index].keys():
            channel_data = psd_dict[chunk_index][channel_index]
            assert "freqs" in channel_data
            assert "psd" in channel_data

    analysis.plot_spectrum(output=psd_dict, input=None, show=False, save_path=tmp_path)

    # Check if plots are saved correctly for each chunk and channel
    for chunk in psd_dict.keys():
        for channel in psd_dict[chunk].keys():
            channel_folder = tmp_path / f"channel{channel:03d}"
            plot_file = channel_folder / f"power_density_{chunk:03d}.png"
            assert plot_file.exists()
