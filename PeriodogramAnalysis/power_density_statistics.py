import os

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Tuple, Dict, Any, Generator, Optional

from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.core.datatype import Signal
from miv.typing import SignalType
from multitaper_spectrum_statistics import multitaper_psd


@dataclass
class SpectrumAnalysisBase(OperatorMixin):
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
        raise NotImplementedError("compute_psd not finished")

    def plot_spectrum(
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
                plt.close()


@dataclass
class SpectrumAnalysisWelch(SpectrumAnalysisBase):
    window_length_for_welch: int = 4
    band_display: Tuple[float, float] = (0, 100)
    tag: str = "Welch PSD"

    def compute_psd(self, signal: Signal) -> Dict[int, Dict[str, Any]]:
        win = self.window_length_for_welch * signal.rate
        psd_welch_dict: Dict[int, Dict[str, Any]] = defaultdict(dict)

        for channel in range(signal.number_of_channels):
            signal_no_bias = signal.data[:, channel] - np.mean(signal.data[:, channel])
            freqs, psd = sig.welch(
                signal_no_bias, fs=signal.rate, nperseg=win, nfft=4 * win
            )
            psd_welch_dict[channel] = {"freqs": freqs, "psd": psd}
        return psd_welch_dict

@dataclass
class SpectrumAnalysisPeriodogram(SpectrumAnalysisBase):
    window_length_for_welch: int = 4
    band_display: Tuple[float, float] = (0, 100)
    tag: str = "Periodogram PSD"

    def compute_psd(self, signal: Signal) -> Dict[int, Dict[str, Any]]:
        psd_periodogram_dict: Dict[int, Dict[str, Any]] = defaultdict(dict)

        for channel in range(signal.number_of_channels):
            signal_no_bias = signal.data[:, channel] - np.mean(signal.data[:, channel])
            freqs, psd = sig.periodogram(signal_no_bias, signal.rate)
            psd_periodogram_dict[channel] = {"freqs": freqs, "psd": psd}

        return psd_periodogram_dict

@dataclass
class SpectrumAnalysisMultitaper(SpectrumAnalysisBase):
    window_length_for_welch: int = 4
    band_display: Tuple[float, float] = (0, 100)
    tag: str = "Multitaper PSD"

    def compute_psd(self, signal: Signal) -> Dict[int, Dict[str, Any]]:
        psd_multitaper_dict: Dict[int, Dict[str, Any]] = defaultdict(dict)

        for channel in range(signal.number_of_channels):
            signal_no_bias = signal.data[:, channel] - np.mean(signal.data[:, channel])
            psd, freqs = multitaper_psd(signal_no_bias, signal.rate)
            psd_multitaper_dict[channel] = {"freqs": freqs, "psd": psd}

        return psd_multitaper_dict
