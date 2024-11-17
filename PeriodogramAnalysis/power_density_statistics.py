import os

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from dataclasses import dataclass
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional

from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.core.datatype import Signal
from miv.typing import SignalType
from multitaper_spectrum_statistics import multitaper_psd


@dataclass
class SpectrumAnalysisBase(OperatorMixin):
    """
    A base class for performing spectral analysis on signal data.

    Attributes
    ----------
    band_display : Tuple[float, float]
        The frequency band range to display on the plot. Default is (0, 100) Hz.
    tag : str
        A string representing the tag used as the title of the plot. Default is "Base PSD Analysis".
    """

    band_display: Tuple[float, float] = (0, 100)
    tag: str = "Base PSD Analysis"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, signal: SignalType) -> Dict[int, Dict[int, Dict[str, Any]]]:

        psd_dict: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for chunk_idx, signal_piece in enumerate(signal):
            psd_dict[chunk_idx] = self.compute_psd(signal_piece)
        return psd_dict

    def compute_psd(self, signal: Signal) -> Dict[int, Dict[str, Any]]:
        """
        compute_psd(signal: Signal) -> Dict[int, Dict[str, Any]]:
        Abstract method to be overridden in subclasses to compute the PSD for a given signal.
        """
        raise NotImplementedError("compute_psd not finished")

    def plot_spectrum(
        self,
        output,
        input,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ):
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
        for chunk in psd_dict.keys():
            for channel in psd_dict[chunk].keys():
                plt.plot(
                    psd_dict[chunk][channel]["freqs"],
                    psd_dict[chunk][channel]["psd"],
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
                        save_path,
                        f"Chunk{chunk}_{self.tag.replace(' ', '_').lower()}_channel_{channel}.png",
                    )
                    plt.savefig(plot_path, dpi=300)
                plt.close('all')


@dataclass
class SpectrumAnalysisWelch(SpectrumAnalysisBase):
    """
    A class that performs spectral analysis using the Welch method.

    Attributes
    ----------
    window_length_for_welch : int
        The length of the window for Welch's method, defined as a multiple of the signal's sampling rate. Default is 4.
    band_display : Tuple[float, float]
        The frequency band range to display on the plot. Default is (0, 100) Hz.
    tag : str
        A string representing the tag used as the title of the plot. Default is "Welch PSD".
    """
    window_length_for_welch: int = 4
    band_display: Tuple[float, float] = (0, 100)
    tag: str = "Welch PSD"

    def compute_psd(self, signal: Signal) -> Dict[int, Dict[str, Any]]:
        """
        Compute the power spectral density (PSD) for a given signal using Welch's method.

        Parameters
        ----------
        signal : Signal
            A 2-dimensional signal object, with the shape (length, number_of_channels).

        Returns
        -------
        Dict[int, Dict[str, Any]]
            A dictionary containing the PSD data for each channel, with keys representing the channel number.
            Each entry contains a dictionary with:
            - 'freqs': The frequency bins.
            - 'psd': The PSD values at each frequency.
        """
        win = self.window_length_for_welch * signal.rate
        psd_welch_dict: Dict[int, Dict[str, Any]] = defaultdict(dict)

        for channel, channel_signal in enumerate(signal):
            signal_no_bias = channel_signal - np.mean(channel_signal)
            freqs, psd = sig.welch(
                signal_no_bias, fs=signal.rate, nperseg=win, nfft=4 * win
            )
            psd_welch_dict[channel] = {"freqs": freqs, "psd": psd}
        return psd_welch_dict


@dataclass
class SpectrumAnalysisPeriodogram(SpectrumAnalysisBase):
    band_display: Tuple[float, float] = (0, 100)
    tag: str = "Periodogram PSD"

    def compute_psd(self, signal: Signal) -> Dict[int, Dict[str, Any]]:
        psd_periodogram_dict: Dict[int, Dict[str, Any]] = defaultdict(dict)

        for channel, channel_signal in enumerate(signal):
            signal_no_bias = channel_signal - np.mean(channel_signal)
            freqs, psd = sig.periodogram(signal_no_bias, signal.rate)
            psd_periodogram_dict[channel] = {"freqs": freqs, "psd": psd}

        return psd_periodogram_dict


@dataclass
class SpectrumAnalysisMultitaper(SpectrumAnalysisBase):
    band_display: Tuple[float, float] = (0, 100)
    tag: str = "Multitaper PSD"

    def compute_psd(self, signal: Signal) -> Dict[int, Dict[str, Any]]:
        psd_multitaper_dict: Dict[int, Dict[str, Any]] = defaultdict(dict)

        for channel, channel_signal in enumerate(signal):
            signal_no_bias = channel_signal - np.mean(channel_signal)
            psd, freqs = multitaper_psd(signal_no_bias, signal.rate)
            psd_multitaper_dict[channel] = {"freqs": freqs, "psd": psd}

        return psd_multitaper_dict
