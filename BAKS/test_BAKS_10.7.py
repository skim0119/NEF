import numpy as np
import pytest
import os

from BAKS import BAKS_firing_rate, spike_detection


def test_BAKS_with_multiple_channels(mocker):
    BAKs_firing_rate = BAKS_firing_rate()

    spike_detection = mocker.Mock()
    spike_detection.output.return_value = spike_detection
    spike_detection.get_view.return_value = [
        np.array([0.1, 0.5, 1.0, 1.5]),
        np.array([0.2, 0.6, 1.1, 1.6]),
    ]
    BAKs_firing_rate(spike_detection)

    assert len(BAKs_firing_rate.firing_rate_list) > 0

    for ch, FiringRate, rate in BAKs_firing_rate.firing_rate_list:
        assert FiringRate > 0, "FiringRate should be positive"


def test_BAKS_empty_input(mocker, tmp_path):
    BAKs_firing_rate = BAKS_firing_rate()

    spike_detection = mocker.Mock()
    spike_detection.output.return_value = spike_detection
    spike_detection.get_view.return_value = []
    BAKs_firing_rate(spike_detection)

    assert len(BAKs_firing_rate.firing_rate_list) == 0

    for ch, FiringRate, rate in BAKs_firing_rate.firing_rate_list:
        assert FiringRate == 0, "FiringRate should be positive"


def test_BAKS_with_different_data_length(mocker):
    BAKs_firing_rate = BAKS_firing_rate()

    spike_detection = mocker.Mock()
    spike_detection.output.return_value = spike_detection
    spike_detection.get_view.return_value = [
        np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        np.array([0.2, 0.6, 1.1, 1.6]),
    ]
    BAKs_firing_rate(spike_detection)

    assert len(BAKs_firing_rate.firing_rate_list) > 0

    for ch, FiringRate, rate in BAKs_firing_rate.firing_rate_list:
        assert FiringRate > 0, "FiringRate should be positive"


if __name__ == "__main__":
    pytest.main()
