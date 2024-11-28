from typing import Any, Optional

import os
from dataclasses import dataclass

import pathlib
import scipy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.core.datatype import Signal
from miv.typing import SignalType


@dataclass
class SpectrumAnalysisBase(OperatorMixin):
    """
    A base class for performing spectral analysis on signal data.

    Attributes
    ----------
    window_length_for_welch : int
        The length of the window for Welch's method, defined as a multiple of the signal's sampling rate. Default is 4.
    band_display : Tuple[float, float]
        The frequency band range to display on the plot. Default is (0, 100) Hz.
    tag : str
        A string representing the tag used as the title of the plot. Default is "Base PSD Analysis".
    """

    window_length_for_welch: int = 4
    band_display: tuple[float, float] = (0, 100)
    tag: str = "Base PSD spectrum analysis"

    def __post_init__(self) -> None:
        super().__init__()

    @cache_call
    def __call__(self, signal: SignalType) -> dict[int, dict[str, Any]]:
        """
        Compute the Power Spectral Density (PSD) for each chunk of the given signal.

        Parameters
        ----------
        signal : SignalType
            The input signal to be analyzed.

        Returns
        -------
        dict[int, dict[str, Any]]
            A dictionary where the keys are channel indices, and each value is another dictionary containing
            the PSD data for the specified channels. The inner dictionary includes keys such as:
            - "freqs": A NumPy array of frequency values in Hz.
            - "psd": A NumPy array of PSD values.
        """

        psd_dict: dict[int, dict[str, Any]] = defaultdict(dict)
        for chunk_idx, signal_piece in enumerate(signal):
            psd_dict = self.compute_psd(signal_piece, psd_dict)
        return psd_dict

    def compute_psd(
        self, signal: Signal, psd_dict: dict[int, dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:
        """
        compute_psd(signal: Signal) -> Dict[int, Dict[str, Any]]:
        Abstract method to be overridden in subclasses to compute the PSD for a given signal.
        """
        raise NotImplementedError(
            "The compute_psd method is not implemented in the base class. "
            "This base class is not intended for standalone use. Please use a subclass "
            "such as SpectrumAnalysisWelch, SpectrumAnalysisPeriodogram, or SpectrumAnalysisMultitaper."
        )

    def plot_spectrum(
        self,
        output: dict,
        input: None,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> None:
        """
        Plots the PSD of the given signal output and optionally saves the plot to a specified path.

        Parameters
        ----------
        output : dict
            The output from the `__call__` method containing PSD data.
        show : bool, optional
            If True, displays the plot. Default is False.
        save_path : Optional[pathlib.Path], optional
            The path to save the plot. If None, the plot is not saved. Default is None.
        """
        psd_dict = output
        for channel in psd_dict.keys():
            for chunk in range(len(psd_dict[channel]["freqs"])):
                if save_path is not None:
                    channel_folder = os.path.join(save_path, f"channel{channel:03d}")
                    os.makedirs(channel_folder, exist_ok=True)

                plt.plot(
                    psd_dict[channel]["freqs"][chunk],
                    psd_dict[channel]["psd"][chunk],
                    lw=1,
                    color="k",
                )
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Power/Frequency (dB/Hz)")
                plt.title(self.tag)
                plt.xlim(self.band_display)
                plt.ylim(ymin=0)

                if show:
                    plt.show()
                if save_path is not None:
                    plot_path = os.path.join(
                        channel_folder, f"power_density_{chunk:03d}.png"
                    )
                    plt.savefig(plot_path, dpi=300)
                plt.close("all")


@dataclass
class SpectrumAnalysisWelch(SpectrumAnalysisBase):
    """
    A class that performs spectral analysis using the Welch method.
    """

    tag: str = "Welch PSD spectrum analysis"

    def compute_psd(
        self, signal: Signal, psd_welch_dict: dict[int, dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:

        win = self.window_length_for_welch * signal.rate
        channel_signal: np.ndarray

        for channel_index in range(signal.number_of_channels):
            signal_no_bias = signal[channel_index] - np.mean(signal[channel_index])
            freqs, psd = scipy.signal.welch(
                signal_no_bias, fs=signal.rate, nperseg=win, nfft=4 * win
            )
            if channel_index not in psd_welch_dict:
                psd_welch_dict[channel_index] = {"freqs": [], "psd": []}
            psd_welch_dict[channel_index]["freqs"].append(freqs)
            psd_welch_dict[channel_index]["psd"].append(psd)
        return psd_welch_dict


@dataclass
class SpectrumAnalysisPeriodogram(SpectrumAnalysisBase):
    """
    A class that performs spectral analysis using the Periodogram method.
    """

    tag: str = "Periodogram PSD spectrum analysis"

    def compute_psd(
        self, signal: Signal, psd_periodogram_dict: dict[int, dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:

        channel_signal: np.ndarray

        for channel_index in range(signal.number_of_channels):
            signal_no_bias = signal[channel_index] - np.mean(signal[channel_index])
            freqs, psd = scipy.signal.periodogram(signal_no_bias, signal.rate)

            if channel_index not in psd_periodogram_dict:
                psd_periodogram_dict[channel_index] = {"freqs": [], "psd": []}
            psd_periodogram_dict[channel_index]["freqs"].append(freqs)
            psd_periodogram_dict[channel_index]["psd"].append(psd)

        return psd_periodogram_dict
