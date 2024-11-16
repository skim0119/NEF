import os

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.integrate import simpson
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Tuple, Dict, Any, Generator, Optional

from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.core.datatype import Signal
from miv.typing import SignalType
from MultitaperPowerSpectrum import multitaper_psd

@dataclass
class SpectrumAnalysisWelch(OperatorMixin):
    window_length_for_welch: int = 4
    band_display: Tuple[float, float] = field(default_factory=lambda: [0, 100])
    tag: str = "Welch PSD"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(
        self, signal: SignalType
    ) -> Dict[int, Dict[int, Dict[str, Any]]]:

        psd_dict: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for chunk_idx, signal_piece in enumerate(signal):
            if chunk_idx >= 1:
                break
            signal_piece.data = signal_piece.data[:, [2]]
            psd_dict[chunk_idx] = self.spectrum_analysis_welch(signal_piece)

        return psd_dict

    def spectrum_analysis_welch(self, signal: Signal) -> Dict[int, Dict[str, Any]]:
        """
        Compute PSD using Welch's method for the signal.

        Parameters:
        -----------
        signal : SignalType
            Input signal to be analyzed.

        Returns:
        --------
        psd_welch_dict
            Dictionary containing PSD values for all channels in this chunk.
        """
        win = self.window_length_for_welch * signal.rate
        psd_welch_dict: Dict[int, Dict[str, Any]] = defaultdict(dict)

        for channel in range(signal.number_of_channels):
            signal_no_bias = signal.data[:, channel] - np.mean(signal.data[:, channel])
            freqs, psd = sig.welch(
                signal_no_bias, fs=signal.rate, nperseg=win, nfft=4 * win
            )
            psd_welch_dict[channel] = {"freqs": freqs, "psd": psd}
        return psd_welch_dict


    def plot_spectrum_methods_welch(
            self,
            output,
            input,
            show: bool = False,
            save_path: Optional[pathlib.Path] = None,
    ):

        psd_dict = output
        for chunk in psd_dict.keys():
            for channel in psd_dict[chunk].keys():
                plt.stem(
                    psd_dict[chunk][channel]["freqs"],
                    psd_dict[chunk][channel]["psd"],
                    linefmt="slategrey",
                    basefmt=" ",
                    markerfmt=" ",
                )
                plt.plot(
                    psd_dict[chunk][channel]["freqs"],
                    psd_dict[chunk][channel]["psd"],
                    lw=0.5,
                    color="k",
                )
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Decibels (dB / Hz)")
                plt.title("Welch")
                plt.xlim(self.band_display)
                plt.ylim(ymin=0)

                if show:
                    plt.show()
                if save_path is not None:
                    plot_path = os.path.join(
                        save_path,
                        f"Chunk{chunk}_welch_channel_{channel}.png",
                    )
                    plt.savefig(plot_path, dpi=300)
                plt.close()

@dataclass
class SpectrumAnalysisPeriodogram(OperatorMixin):
    band_display: Tuple[float, float] = field(default_factory=lambda: [0, 100])
    tag: str = "Periodogram PSD"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(
        self, signal: SignalType
    ) -> Dict[int, Dict[int, Dict[str, Any]]]:

        psd_dict: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for chunk_idx, signal_piece in enumerate(signal):
            if chunk_idx >= 1:
                break
            signal_piece.data = signal_piece.data[:, [2]]
            psd_dict[chunk_idx] = self.spectrum_analysis_periodogram(signal_piece)

        return psd_dict

    def spectrum_analysis_periodogram(self, signal: Signal) -> Dict[int, Dict[str, Any]]:
        """
        Compute PSD using periodogram for the signal.

        Parameters:
        -----------
        signal : SignalType
            Input signal to be analyzed.

        Returns:
        --------
        psd_periodogram_dict
            Dictionary containing PSD values for all channels in this chunk.
        """
        psd_periodogram_dict: Dict[int, Dict[str, Any]] = defaultdict(dict)

        for channel in range(signal.number_of_channels):
            signal_no_bias = signal.data[:, channel] - np.mean(signal.data[:, channel])
            freqs, psd = sig.periodogram(signal_no_bias, signal.rate)
            psd_periodogram_dict[channel] = {"freqs": freqs, "psd": psd}
        return psd_periodogram_dict

    def plot_spectrum_methods_periodogram(
            self,
            output,
            input,
            show: bool = False,
            save_path: Optional[pathlib.Path] = None,
    ):

        psd_dict = output
        for chunk in psd_dict.keys():
            for channel in psd_dict[chunk].keys():
                plt.stem(
                    psd_dict[chunk][channel]["freqs"],
                    psd_dict[chunk][channel]["psd"],
                    linefmt="slategrey",
                    basefmt=" ",
                    markerfmt=" ",
                )
                plt.plot(
                    psd_dict[chunk][channel]["freqs"],
                    psd_dict[chunk][channel]["psd"],
                    lw=0.5,
                    color="k",
                )
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Decibels (dB / Hz)")
                plt.title("Periodogram")
                plt.xlim(self.band_display)
                plt.ylim(ymin=0)

                if show:
                    plt.show()
                if save_path is not None:
                    plot_path = os.path.join(
                        save_path,
                        f"Chunk{chunk}_periodogram_channel_{channel}.png",
                    )
                    plt.savefig(plot_path, dpi=300)
                plt.close()

@dataclass
class SpectrumAnalysisMultitaper(OperatorMixin):
    band_display: Tuple[float, float] = field(default_factory=lambda: [0, 100])
    tag: str = "Multitaper PSD"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(
        self, signal: SignalType
    ) -> Dict[int, Dict[int, Dict[str, Any]]]:

        psd_dict: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for chunk_idx, signal_piece in enumerate(signal):
            if chunk_idx >= 1:
                break
            signal_piece.data = signal_piece.data[:, [2]]
            psd_dict[chunk_idx] = self.spectrum_analysis_multitaper(signal_piece)

        return psd_dict

    def spectrum_analysis_multitaper(self, signal: Signal) -> Dict[int, Dict[str, Any]]:
        """
        Compute Welch's periodogram for the signal.

        Parameters:
        -----------
        signal : SignalType
            Input signal to be analyzed.

        Returns:
        --------
        psd_multitaper_dict
            Dictionary containing PSD values for all channels in this chunk.
        """
        psd_multitaper_dict: Dict[int, Dict[str, Any]] = defaultdict(dict)

        for channel in range(signal.number_of_channels):
            signal_no_bias = signal.data[:, channel] - np.mean(signal.data[:, channel])
            psd, freqs = multitaper_psd(signal_no_bias, signal.rate)
            psd_multitaper_dict[channel] = {"freqs": freqs, "psd": psd}
        return psd_multitaper_dict

    def plot_spectrum_methods_multitaper(
        self,
        output,
        input,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ):

        psd_dict = output
        for chunk in psd_dict.keys():
            for channel in psd_dict[chunk].keys():
                plt.stem(
                    psd_dict[chunk][channel]["freqs"],
                    psd_dict[chunk][channel]["psd"],
                    linefmt="slategrey",
                    basefmt=" ",
                    markerfmt=" ",
                )
                plt.plot(
                    psd_dict[chunk][channel]["freqs"],
                    psd_dict[chunk][channel]["psd"],
                    lw=0.5,
                    color="k",
                )
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Decibels (dB / Hz)")
                plt.title("Multitaper")
                plt.xlim(self.band_display)
                plt.ylim(ymin=0)

                if show:
                    plt.show()
                if save_path is not None:
                    plot_path = os.path.join(
                        save_path,
                        f"Chunk{chunk}_multitaper_channel_{channel}.png",
                    )
                    plt.savefig(plot_path, dpi=300)
                plt.close()