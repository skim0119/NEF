import os
import numpy as np
import pytest
from BAKS import BAKS_firing_rate


def test_BAKS():
    BAKs_firing_rate = BAKS_firing_rate()

    SpikeTimes = np.array([0.1, 0.5, 1.0, 1.5])
    Time = 2.0
    a = 0.32
    b = len(SpikeTimes) ** (4. / 5.)

    FiringRate, h = BAKs_firing_rate.BAKS(SpikeTimes, Time, a, b)

    assert FiringRate > 0, "FiringRate should be positive"
    assert h > 0, "Adaptive bandwidth h should be positive"

def test_call_method():
    BAKs_firing_rate = BAKS_firing_rate()
    class Mockspikedetection:
        def output(self):
            return self

        def get_view(self, start, end):
            # Mocking the behavior of get_view() to return spike data
            return [np.array([0.1, 0.5, 1.0, 1.5]), np.array([0.2, 0.6, 1.1, 1.6])]

    spike_detection = Mockspikedetection()
    BAKs_firing_rate(spike_detection)

    with open('spike/firing_rate_summary_sorted.txt', 'r') as f:
        lines = f.readlines()
        assert len(lines) > 1

def test_file_generation():
    BAKs_firing_rate = BAKS_firing_rate()

    class Mockspikedetection:
        def output(self):
            return self

        def get_view(self, start, end):
            # Mocking the behavior of get_view() to return spike data
            return [np.array([0.2, 0.6, 1.1, 1.6])]

    spike_detection = Mockspikedetection()
    BAKs_firing_rate(spike_detection)

    assert os.path.exists('spike/firing_rate_summary_sorted.txt')

    with open('spike/firing_rate_summary_sorted.txt', 'r') as f:
        lines = f.readlines()
        assert "channel, firing_rate_hz, ref_firing_rate_spikes/time" in lines[0]


if __name__ == '__main__':
    pytest.main()