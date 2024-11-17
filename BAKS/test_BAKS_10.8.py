import numpy as np
import pytest
import os

from BAKS import BAKSFiringRate, spike_detection


def test_BAKS_with_multiple_channels(mocker):
    BAKs_firing_rate = BAKSFiringRate()

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


def test_BAKS_empty_input(mocker):
    BAKs_firing_rate = BAKSFiringRate()

    spike_detection = mocker.Mock()
    spike_detection.output.return_value = spike_detection
    spike_detection.get_view.return_value = []
    BAKs_firing_rate(spike_detection)

    assert len(BAKs_firing_rate.firing_rate_list) == 0

    for ch, FiringRate, rate in BAKs_firing_rate.firing_rate_list:
        assert FiringRate == 0, "FiringRate should be positive"


def test_bandwidth_with_multiple_channels(mocker):
    BAKs_firing_rate = BAKSFiringRate()

    spike_detection = mocker.Mock()
    spike_detection.output.return_value = spike_detection
    spike_detection.get_view.return_value = [
        np.array([0.2, 0.5, 1.2, 1.3]),
        np.array([0.1, 0.8, 1.2, 1.5]),
    ]
    BAKs_firing_rate(spike_detection)

    assert len(BAKs_firing_rate.bandwidth_list) > 0
    assert (
        BAKs_firing_rate.bandwidth_list[0] != BAKs_firing_rate.bandwidth_list[1]
    ), "bandwidths should be different"

    for ch, bandwidth in BAKs_firing_rate.bandwidth_list:
        assert bandwidth > 0, "bandwidth should be positive"

    for i in range(2):
        assert (
            BAKs_firing_rate.bandwidth_list[i][0]
            == BAKs_firing_rate.firing_rate_list[i][0]
        ), "channels should be sorted based on firing rate"


def test_BAKS_with_different_data_length(mocker):
    BAKs_firing_rate = BAKSFiringRate()

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
