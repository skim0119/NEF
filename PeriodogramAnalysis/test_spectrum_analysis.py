from typing import Any, Optional, Tuple

import pytest
import numpy as np
from spectrum_analysis import PowerSpectrumAnalysis
from scipy.integrate import simpson


def psd_input() -> dict[int, dict[str, Any]]:
    freqs = np.array([1, 2, 3, 4, 5, 7, 9, 10, 15, 25, 35, 50])
    psd = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
            [0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575],
        ]
    ).T

    output = (freqs, psd)

    return output


def test_call_invalid_input_parameters():
    with pytest.raises(ValueError):
        PowerSpectrumAnalysis(band_display=(5, 10, 20, 30))
    with pytest.raises(ValueError):
        PowerSpectrumAnalysis(band_display=(-2, 5))

    with pytest.raises(ValueError):
        PowerSpectrumAnalysis(band_display=((8, 5)))
    with pytest.raises(ValueError):
        PowerSpectrumAnalysis(band_display=((-3, 5)))
    with pytest.raises(ValueError):
        PowerSpectrumAnalysis(band_display=((8), (12, 30), (30, 100)))


def test_call_return_shape():
    analyzer = PowerSpectrumAnalysis()
    result = analyzer(psd_input())
    freqs, psd, psd_idx = result

    assert isinstance(freqs, np.ndarray)
    assert isinstance(psd, np.ndarray)
    assert freqs.shape[0] == 12
    assert psd.shape[0] == 12
    assert psd_idx.shape == (12, 10)


def test_computing_absolute_and_relative_power(mocker) -> None:
    analysis = PowerSpectrumAnalysis()
    result = analysis(psd_input())
    freqs, psd, psd_idx = result

    power_analysis = PowerSpectrumAnalysis()
    mocker.patch.object(power_analysis, "num_channel", 2)

    psd_idx, power, rel_power = power_analysis.computing_absolute_and_relative_power(
        freqs, psd
    )

    for ch in range(2):
        psd_channel = psd[:, ch]
        freq_res = freqs[1] - freqs[0]
        total_power = simpson(psd_channel, dx=freq_res)

        for i, band in enumerate(power_analysis.band_list):
            psd_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            manual_power = simpson(psd_channel[psd_idx], dx=freq_res)
            computed_power = power[ch * 5 + i]
            computed_rel_power = rel_power[ch * 5 + i]

            assert np.isclose(manual_power, computed_power)

            manual_rel_power = manual_power / total_power
            assert np.isclose(manual_rel_power, computed_rel_power)


def test_computing_ratio_and_bandpower(mocker) -> None:
    power = np.array([1, 2, 3, 4, 5])
    rel_power = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    power_analysis = PowerSpectrumAnalysis()
    mocker.patch.object(power_analysis, "num_channel", 2)
    mocker.patch.object(power_analysis, "chunk", 0)
    mocker.patch.object(
        power_analysis, "band_list", ((0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100))
    )

    result = power_analysis.computing_ratio_and_bandpower(power, rel_power)
    absolute_powers, relative_powers = result

    np.testing.assert_allclose(absolute_powers, power)
    np.testing.assert_allclose(relative_powers, rel_power)
