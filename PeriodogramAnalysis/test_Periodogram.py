import numpy as np

from miv.core.datatype import Signal
from PowerSpectrumAnalysis import PowerSpectrumAnalysis
from SpectrogramAnalysis import SpectrogramAnalysis

def mock_power():
    data = np.random.rand(30000, 10)  # Mock data with 30,000 samples and 5 channels
    timestamps = np.linspace(1, 30, 30000)  # Mock timestamps from 1 to 30 seconds
    rate = 1000  # Sampling rate of 1000 Hz
    return Signal(data=data, timestamps=timestamps, rate=rate)

def mock_power_generator():
    chunk_size = 3
    data = np.random.rand(30000, 10)
    timestamps = np.linspace(1, 30, 30000)
    rate = 1000

    for i in range(chunk_size):
        start_idx = i * chunk_size
        end_idx = start_idx + 10000
        chunk_data = data[start_idx:end_idx, :]
        chunk_timestamps = timestamps[start_idx:end_idx]

        yield Signal(data=chunk_data, timestamps=chunk_timestamps, rate=rate)


def test_periodogram_analysis_call():
    analysis = PowerSpectrumAnalysis(window_length_for_welch=2)

    signal = list(mock_power_generator())
    psd_dict, power_dict = analysis(signal)

    # Check the output are both dicts
    assert isinstance(psd_dict, dict)
    assert isinstance(power_dict, dict)

    # Check that all chunks are in the dicts
    for chunk_idx in range(len(signal)):
        assert chunk_idx in psd_dict
        assert chunk_idx in power_dict

        # Check PSD dict and power dict contains needed channels only, and has needed data
        for channel in range(10):
            assert channel in psd_dict[chunk_idx]
            assert 'freqs' in psd_dict[chunk_idx][channel]
            assert 'psd' in psd_dict[chunk_idx][channel]

            assert channel in power_dict[chunk_idx]
            required_keys = [
                'idx_delta', 'idx_theta', 'idx_alpha', 'idx_beta', 'idx_gamma',
                'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
                'delta_rel_power', 'theta_rel_power', 'alpha_rel_power', 'beta_rel_power', 'gamma_rel_power'
            ]
            for key in required_keys:
                assert key in power_dict[chunk_idx][channel]

def test_periodogram_analysis_call_default():
    analysis = PowerSpectrumAnalysis()

    signal = list(mock_power_generator())
    analysis(signal)
    # Test default settings
    assert analysis.band_list == ((0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100))
    assert analysis.window_length_for_welch == 4


def test_computing_ratio_and_bandpower(mocker):
    signal = list(mock_power_generator())
    analysis = PowerSpectrumAnalysis(window_length_for_welch=2)
    analysis2 = PowerSpectrumAnalysis(window_length_for_welch=2, band_list=((0.5, 4), (4, 8)))

    psd_dict, power_dict = analysis(signal)
    logger_info_spy = mocker.spy(analysis.logger, 'info')
    analysis.computing_ratio_and_bandpower(signal[0], power_dict, 0)

    # Test how many times logger is called, 31 = 1 + 10 * 4*5, where 5 is the number of bands
    assert logger_info_spy.call_count == signal[0].number_of_channels * 31

    psd_dict, power_dict = analysis2(signal)
    logger_info_spy2 = mocker.spy(analysis2.logger, 'info')
    analysis2.computing_ratio_and_bandpower(signal[0], power_dict, 0)

    # Test how many times logger is called, 19 = 1 + 10 + 4*2, where 2 is the number of bands
    assert logger_info_spy2.call_count == signal[0].number_of_channels * 19


def test_computing_multitaper_bandpower():
    analysis = PowerSpectrumAnalysis(window_length_for_welch=2)
    signal = mock_power()
    band = [8, 12]  # Alpha band
    channel = 0

    bandpower = analysis.computing_multitaper_bandpower(signal, band, channel)

    # Check common band power computation
    assert isinstance(bandpower, float)
    assert bandpower >= 0

    # Test relative power computation
    relative_bandpower = analysis.computing_multitaper_bandpower(signal, band, channel, relative=True)
    assert isinstance(relative_bandpower, float)
    assert relative_bandpower >= 0
    assert relative_bandpower <= 1
    assert bandpower != relative_bandpower


def test_computing_welch_bandpower():
    analysis = PowerSpectrumAnalysis(window_length_for_welch=2)
    signal = mock_power()
    band = [8, 12]  # Alpha band
    channel = 0

    bandpower = analysis.computing_welch_bandpower(signal, band, channel)

    # Check common band power computation
    assert isinstance(bandpower, float)
    assert bandpower >= 0

    # Test relative power computation
    relative_bandpower = analysis.computing_welch_bandpower(signal, band, channel, relative=True)
    assert isinstance(relative_bandpower, float)
    assert relative_bandpower >= 0
    assert relative_bandpower <= 1
    assert bandpower != relative_bandpower


def test_plot_welch_periodogram(tmp_path):
    analysis = PowerSpectrumAnalysis(window_length_for_welch=2)
    signal = list(mock_power_generator())
    psd_dict, power_dict = analysis(signal)
    output = (psd_dict, power_dict)

    analysis.plot_welch_periodogram(output, signal, show=False, save_path=tmp_path)

    # Check if plots are saved correctly for each chunk and channel
    for chunk in psd_dict.keys():
        for channel in psd_dict[chunk].keys():
            plot_file = tmp_path / f"Chunk{chunk}_Welch_periodogram_of_channel_{channel}.png"
            assert plot_file.exists()


def test_SpectrumAnalysis_call():
    Spectrum_Analysis = SpectrogramAnalysis(
        frequency_limit=[0, 10],
        window_length_for_welch=4,
        band_display=[0, 5]
    )

    signal = list(mock_power_generator())
    psd_welch_dict, psd_periodogram_dict, psd_multitaper_dict, spec_dict = Spectrum_Analysis(signal)

    # Check the output are both dicts
    assert isinstance(psd_welch_dict, dict)
    assert isinstance(psd_periodogram_dict, dict)
    assert isinstance(psd_multitaper_dict, dict)
    assert isinstance(spec_dict, dict)

    # Check that all chunks are in the dicts
    for chunk_idx in range(len(signal)):
        assert chunk_idx in psd_welch_dict.keys()
        assert chunk_idx in psd_periodogram_dict.keys()
        assert chunk_idx in psd_multitaper_dict.keys()
        assert chunk_idx in spec_dict.keys()

        # Check PSD dict and spec dict contains needed channels only, and has needed data
        for channel in range(10):
            dict_list = [psd_welch_dict, psd_periodogram_dict, psd_multitaper_dict]
            required_keys_psd = ['freqs', 'psd']
            for psd_dict in dict_list:
                for key in required_keys_psd:
                    assert key in psd_dict[chunk_idx][channel]

            required_keys_spec = ['frequencies', 'times', 'Sxx']
            for key in required_keys_spec:
                assert key in spec_dict[chunk_idx][channel]

def test_SpectrumAnalysis_call_default():
    Spectrum_Analysis = SpectrogramAnalysis()

    signal = list(mock_power_generator())
    Spectrum_Analysis(signal)
    # Test if default setting is correct
    assert Spectrum_Analysis.band_display == [0, 100]
    assert Spectrum_Analysis.window_length_for_welch == 4
    assert Spectrum_Analysis.frequency_limit == [0.5, 100]
    assert Spectrum_Analysis.plotting_interval == [0, 60]
    assert Spectrum_Analysis.nperseg_ratio == 0.25

def test_plot_spectrum_methods(tmp_path):
    Spectrum_Analysis = SpectrogramAnalysis()
    signal = list(mock_power_generator())
    psd_welch_dict, psd_periodogram_dict, psd_multitaper_dict, spec_dict = Spectrum_Analysis(signal)
    output = (psd_welch_dict, psd_periodogram_dict, psd_multitaper_dict, spec_dict)

    Spectrum_Analysis.plot_spectrum_methods(output, signal, show=False, save_path=tmp_path)

    # Check if plots are saved correctly for each chunk and channel
    for chunk in spec_dict.keys():
        for channel in spec_dict[chunk].keys():
            plot_file = tmp_path / f"Chunk{chunk}_Comparison_figure_channel_{channel}.png"
            assert plot_file.exists()

def test_plot_spectrogram(tmp_path):
    Spectrum_Analysis = SpectrogramAnalysis()
    signal = list(mock_power_generator())
    psd_welch_dict, psd_periodogram_dict, psd_multitaper_dict, spec_dict = Spectrum_Analysis(signal)
    output = (psd_welch_dict, psd_periodogram_dict, psd_multitaper_dict, spec_dict)

    Spectrum_Analysis.plot_spectrogram(output, signal, show=False, save_path=tmp_path)

    # Check if plots are saved correctly for each chunk and channel
    for chunk in spec_dict.keys():
        for channel in spec_dict[chunk].keys():
            plot_file = tmp_path / f'Chunk{chunk}_Spectrogram_Channel_{channel}.png'
            assert plot_file.exists()

