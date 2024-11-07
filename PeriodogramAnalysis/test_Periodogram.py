import os
import pytest
import numpy as np

from miv.core.datatype import Signal
from PeriodogramAnalysis import PeriodogramAnalysis

def mock_power():
    data = np.random.rand(30000, 5)  # Mock data with 30,000 samples and 5 channels
    timestamps = np.linspace(1, 30, 30000)  # Mock timestamps from 1 to 30 seconds
    rate = 1000  # Sampling rate of 1000 Hz
    return Signal(data=data, timestamps=timestamps, rate=rate)

def mock_power_generator():
    chunk_size = 3
    data = np.random.rand(30000, 5)
    timestamps = np.linspace(1, 30, 30000)
    rate = 1000

    for i in range(chunk_size):
        start_idx = i * chunk_size
        end_idx = start_idx + 10000
        chunk_data = data[start_idx:end_idx, :]
        chunk_timestamps = timestamps[start_idx:end_idx]

        yield Signal(data=chunk_data, timestamps=chunk_timestamps, rate=rate)


def test_periodogram_analysis_call():
    analysis = PeriodogramAnalysis(window_length_for_welch=2, exclude_channel_list=[1, 3])

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
        for channel in range(analysis.num_channels):
            if channel in analysis.exclude_channel_list:
                assert channel not in psd_dict[chunk_idx]
                assert channel not in power_dict[chunk_idx]

            else:
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

