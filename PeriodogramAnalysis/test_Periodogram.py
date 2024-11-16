import numpy as np
import pytest
from miv.core.datatype import Signal
from PowerSpectrumAnalysis import PowerSpectrumAnalysis
from SpectrogramAnalysis import SpectrogramAnalysis
from Power_Spectral_Density import SpectrumAnalysisPeriodogram, SpectrumAnalysisMultitaper, SpectrumAnalysisWelch
from scipy.integrate import simpson

def mock_signal_generator():
    data1 = np.array([
        np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100)),  # 5 Hz
        np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))  # 10 Hz
    ]).T  # 形状为 (100, 2)
    timestamps = np.linspace(0, 1, 100)
    signal1 = Signal(data=data1, timestamps=timestamps, rate=100.0)
    yield signal1

    data2 = np.array([
        np.sin(2 * np.pi * 15 * np.linspace(0, 1, 200)),  # 15 Hz
        np.sin(2 * np.pi * 20 * np.linspace(0, 1, 200))  # 20 Hz
    ]).T
    timestamps = np.linspace(0, 1, 100)
    signal2 = Signal(data=data2, timestamps=timestamps, rate=100.0)
    yield signal2


def mock_psd_list():
    psd_dict = {
        0: {  # chunk index
            0: {  # channel index
                "freqs": np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50]),
                "psd": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
            },
            1: {  # channel index
                "freqs": np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50]),
                "psd": np.array([0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3])
            }
        },
        1: {  # chunk index (新添加的 chunk 1)
            0: {  # channel index
                "freqs": np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50]),
                "psd": np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575])
            },
            1: {  # channel index
                "freqs": np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50]),
                "psd": np.array([0.575, 0.55, 0.525, 0.5, 0.475, 0.45, 0.425, 0.4, 0.35, 0.3, 0.25, 0.2])
            }
        }
    }

    return psd_dict

def test_power_spectrum_analysis_call_and_computing_absolute_and_relative_power():
    """
    Test the __call__ func and computing_absolute_and_relative_power to ensure it correctly computes
    the power_dict based on the provided psd_dict, and test the return is as expected.
    """
    # Sample PSD data for 1 chunk and 2 channels
    psd_dict = mock_psd_list()

    expected_power_dict = {
        0: {  # chunk index
            0: {  # channel index
                "idx_delta": np.array([True, True, True, True, False, False, False, False, False, False, False, False]),
                "idx_theta": np.array(
                    [False, False, False, True, True, True, False, False, False, False, False, False]),
                "idx_alpha": np.array(
                    [False, False, False, False, False, False, True, True, False, False, False, False]),
                "idx_beta": np.array(
                    [False, False, False, False, False, False, False, False, True, True, False, False]),
                "idx_gamma": np.array(
                    [False, False, False, False, False, False, False, False, False, False, True, True]),
                "delta_power": simpson([0.1, 0.2, 0.3, 0.4], dx=1),  # ≈0.516666...
                "theta_power": simpson([0.4, 0.5, 0.55], dx=1),  # ≈0.516666...
                "alpha_power": simpson([0.6, 0.65], dx=1),  # =1.25
                "beta_power": simpson([0.7, 0.75], dx=1),  # =1.45
                "gamma_power": simpson([0.8, 0.85], dx=1),  # =1.725
                "delta_rel_power": simpson([0.1, 0.2, 0.3, 0.4], dx=1) / simpson(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], dx=1),  # ~0.1022
                "theta_rel_power": simpson([0.4, 0.5, 0.55], dx=1) / simpson(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], dx=1),  # ~0.1022
                "alpha_rel_power": simpson([0.6, 0.65], dx=1) / simpson(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], dx=1),  # ~0.2478
                "beta_rel_power": simpson([0.7, 0.75], dx=1) / simpson(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], dx=1),  # ~0.2871
                "gamma_rel_power": simpson([0.8, 0.85], dx=1) / simpson(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], dx=1),  # ~0.3420
            },
            1: {  # channel index
                "idx_delta": np.array([True, True, True, True, False, False, False, False, False, False, False, False]),
                "idx_theta": np.array(
                    [False, False, False, True, True, True, False, False, False, False, False, False]),
                "idx_alpha": np.array(
                    [False, False, False, False, False, False, True, True, False, False, False, False]),
                "idx_beta": np.array(
                    [False, False, False, False, False, False, False, False, True, True, False, False]),
                "idx_gamma": np.array(
                    [False, False, False, False, False, False, False, False, False, False, True, True]),
                "delta_power": simpson([0.85, 0.8, 0.75, 0.7], dx=1),  # 3.05
                "theta_power": simpson([0.7, 0.65, 0.6], dx=1),  # 0.675
                "alpha_power": simpson([0.55, 0.5], dx=1),  # 0.525
                "beta_power": simpson([0.45, 0.4], dx=1),  # 0.425
                "gamma_power": simpson([0.35, 0.3], dx=1),  # 0.325
                "delta_rel_power": simpson([0.85, 0.8, 0.75, 0.7], dx=1) / simpson(
                    [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3], dx=1),  # ~0.5304
                "theta_rel_power": simpson([0.7, 0.65, 0.6], dx=1) / simpson(
                    [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3], dx=1),  # ~0.1174
                "alpha_rel_power": simpson([0.55, 0.5], dx=1) / simpson(
                    [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3], dx=1),  # ~0.0913
                "beta_rel_power": simpson([0.45, 0.4], dx=1) / simpson(
                    [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3], dx=1),  # ~0.0739
                "gamma_rel_power": simpson([0.35, 0.3], dx=1) / simpson(
                    [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3], dx=1),  # ~0.0565
            }
        },
        1: {  # chunk index (新添加的 chunk 1)
            0: {  # channel index
                "idx_delta": np.array([True, True, True, True, False, False, False, False, False, False, False, False]),
                "idx_theta": np.array(
                    [False, False, False, True, True, True, False, False, False, False, False, False]),
                "idx_alpha": np.array(
                    [False, False, False, False, False, False, True, True, False, False, False, False]),
                "idx_beta": np.array(
                    [False, False, False, False, False, False, False, False, True, True, False, False]),
                "idx_gamma": np.array(
                    [False, False, False, False, False, False, False, False, False, False, True, True]),
                "delta_power": simpson([0.2, 0.25, 0.3, 0.35], dx=1),  # ≈0.875
                "theta_power": simpson([0.35, 0.4, 0.425], dx=1),  # ≈0.45
                "alpha_power": simpson([0.45, 0.475], dx=1),  # ≈0.925
                "beta_power": simpson([0.5, 0.525], dx=1),  # ≈1.05
                "gamma_power": simpson([0.55, 0.575], dx=1),  # ≈1.15
                "delta_rel_power": simpson([0.2, 0.25, 0.3, 0.35], dx=1) / simpson(
                    [0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575], dx=1),
                # ~0.875 / 5.05 ≈ 0.1733
                "theta_rel_power": simpson([0.35, 0.4, 0.425], dx=1) / simpson(
                    [0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575], dx=1),
                # ~0.45 / 5.05 ≈ 0.0891
                "alpha_rel_power": simpson([0.45, 0.475], dx=1) / simpson(
                    [0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575], dx=1),
                # ~0.925 / 5.05 ≈ 0.1832
                "beta_rel_power": simpson([0.5, 0.525], dx=1) / simpson(
                    [0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575], dx=1),
                # ~1.05 / 5.05 ≈ 0.2079
                "gamma_rel_power": simpson([0.55, 0.575], dx=1) / simpson(
                    [0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575], dx=1),
                # ~1.15 / 5.05 ≈ 0.2277
            },
            1: {  # channel index
                "idx_delta": np.array([True, True, True, True, False, False, False, False, False, False, False, False]),
                "idx_theta": np.array(
                    [False, False, False, True, True, True, False, False, False, False, False, False]),
                "idx_alpha": np.array(
                    [False, False, False, False, False, False, True, True, False, False, False, False]),
                "idx_beta": np.array(
                    [False, False, False, False, False, False, False, False, True, True, False, False]),
                "idx_gamma": np.array(
                    [False, False, False, False, False, False, False, False, False, False, True, True]),
                "delta_power": simpson([0.575, 0.55, 0.525, 0.5], dx=1),  # ≈2.15
                "theta_power": simpson([0.5, 0.475, 0.45], dx=1),  # ≈1.325
                "alpha_power": simpson([0.425, 0.4], dx=1),  # ≈0.825
                "beta_power": simpson([0.35, 0.3], dx=1),  # ≈0.65
                "gamma_power": simpson([0.25, 0.2], dx=1),  # ≈0.225
                "delta_rel_power": simpson([0.575, 0.55, 0.525, 0.5], dx=1) / simpson(
                    [0.575, 0.55, 0.525, 0.5, 0.475, 0.45, 0.425, 0.4, 0.35, 0.3, 0.25, 0.2], dx=1),
                # ~2.15 / 5.85 ≈ 0.3675
                "theta_rel_power": simpson([0.5, 0.475, 0.45], dx=1) / simpson(
                    [0.575, 0.55, 0.525, 0.5, 0.475, 0.45, 0.425, 0.4, 0.35, 0.3, 0.25, 0.2], dx=1),
                # ~1.325 / 5.85 ≈ 0.2265
                "alpha_rel_power": simpson([0.425, 0.4], dx=1) / simpson(
                    [0.575, 0.55, 0.525, 0.5, 0.475, 0.45, 0.425, 0.4, 0.35, 0.3, 0.25, 0.2], dx=1),
                # ~0.825 / 5.85 ≈ 0.1410
                "beta_rel_power": simpson([0.35, 0.3], dx=1) / simpson(
                    [0.575, 0.55, 0.525, 0.5, 0.475, 0.45, 0.425, 0.4, 0.35, 0.3, 0.25, 0.2], dx=1),
                # ~0.65 / 5.85 ≈ 0.1111
                "gamma_rel_power": simpson([0.25, 0.2], dx=1) / simpson(
                    [0.575, 0.55, 0.525, 0.5, 0.475, 0.45, 0.425, 0.4, 0.35, 0.3, 0.25, 0.2], dx=1),
                # ~0.225 / 5.85 ≈ 0.0385
            }
        }
    }

    analysis = PowerSpectrumAnalysis()
    psd_output, power_output = analysis(psd_dict)

    # Test psd_dict returned to plot func is correct
    assert psd_output == psd_dict

    # Test power_dict returned to plot func is correct
    for chunk_idx in expected_power_dict.keys():
        for channel_idx in expected_power_dict[chunk_idx].keys():
            expected_channel_power = expected_power_dict[chunk_idx][channel_idx]
            actual_channel_power = power_output[chunk_idx][channel_idx]
            for key in expected_channel_power.keys():
                expected_value = expected_channel_power[key]
                actual_value = actual_channel_power[key]
                if isinstance(expected_value, np.ndarray):
                    np.testing.assert_array_equal(actual_value, expected_value)
                else:
                    assert actual_value == pytest.approx(expected_value, rel=1e-4)

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
            plot_file = tmp_path / f"Chunk{chunk}_periodogram_of_channel_{channel}.png"
            assert plot_file.exists()


def test_computing_ratio_and_bandpower(mocker):
    """
    Test computing_ratio_and_bandpower, test how many logger are called.
    """
    chunk_idx = 0
    analysis = PowerSpectrumAnalysis()
    psd_dict = mock_psd_list()
    psd_dict, power_dict = analysis(psd_dict)

    logger_info_spy = mocker.spy(analysis.logger, 'info')
    analysis.computing_ratio_and_bandpower(psd_dict[chunk_idx], power_dict[chunk_idx], chunk_idx)

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
                relative_bandpower = analysis.computing_bandpower(psd_dict[chunk][channel], band, relative=True)
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

    spectrum_analysis.plot_spectrogram(output=spec_dict, input=None, show=False, save_path=tmp_path)

    # Check if plots are saved correctly for each chunk and channel
    for chunk in spec_dict.keys():
        for channel in spec_dict[chunk].keys():
            plot_file = tmp_path / f'Chunk{chunk}_Spectrogram_Channel_{channel}.png'
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

    analysis.plot_spectrum_methods_welch(output=psd_dict, input=None, show=False, save_path=tmp_path)

    # Check if plots are saved correctly for each chunk and channel
    for chunk in psd_dict.keys():
        for channel in psd_dict[chunk].keys():
            plot_file = tmp_path / f"Chunk{chunk}_welch_channel_{channel}.png"
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

    analysis.plot_spectrum_methods_periodogram(output=psd_dict, input=None, show=False, save_path=tmp_path)

    # Check if plots are saved correctly for each chunk and channel
    for chunk in psd_dict.keys():
        for channel in psd_dict[chunk].keys():
            plot_file = tmp_path / f"Chunk{chunk}_periodogram_channel_{channel}.png"
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

    analysis.plot_spectrum_methods_multitaper(output=psd_dict, input=None, show=False, save_path=tmp_path)

    # Check if plots are saved correctly for each chunk and channel
    for chunk in psd_dict.keys():
        for channel in psd_dict[chunk].keys():
            plot_file = tmp_path / f"Chunk{chunk}_multitaper_channel_{channel}.png"
            assert plot_file.exists()

