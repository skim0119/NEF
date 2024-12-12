import numpy as np
import pytest
from miv.core.datatype import Signal


@pytest.fixture(scope="module")
def generate_signal():
    def generate(num_channel):
        expected_freqs = [5, 10, 15, 20]
        timestamps = np.linspace(0, 10, 1000, endpoint=False)
        signal = (
            np.sin(2 * np.pi * 5 * timestamps)
            + np.sin(2 * np.pi * 10 * timestamps)
            + np.sin(2 * np.pi * 15 * timestamps)
            + np.sin(2 * np.pi * 20 * timestamps)
        )
        np.random.seed(42)
        noise = np.random.normal(loc=0, scale=0.5, size=signal.shape)
        signal += noise
        data = np.empty((0, len(signal)))
        for ch in range(num_channel):
            data = np.vstack((data, (ch + 2) * signal))
        signal = Signal(data=data.T, timestamps=timestamps, rate=100)
        return signal, expected_freqs

    return generate
