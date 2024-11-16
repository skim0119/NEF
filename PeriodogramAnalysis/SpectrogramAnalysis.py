import os

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Generator, Optional

from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.core.datatype import Signal
from miv.typing import SignalType
from MultitaperPowerSpectrum import multitaper_psd
from PowerSpectrumAnalysis import PowerSpectrumAnalysis


@dataclass
class SpectrogramAnalysis(OperatorMixin):
    """
    A class to perform Spectrum Analysis using multiple methods including Welch's, Periodogram, and multitaper PSD.

    Attributes:
    -----------
    band_display : list
        Frequency band range to display in spectrum comparison plots.
    window_length_for_welch : float
        Window length in seconds for Welch's method.
    frequency_limit : list
        Frequency range limit for analysis.
    nperseg : int
        Number of points per segment for spectrogram computation.
    noverlap : int
        Number of points to overlap between segments for spectrogram.
    """

    band_display: Tuple[float, float] = field(default_factory=lambda: [0, 100])
    window_length_for_welch: int = 4
    frequency_limit: Tuple[float, float] = field(default_factory=lambda: [0.5, 100])
    plotting_interval: Tuple[float, float] = field(default_factory=lambda: [0, 60])
    nperseg_ratio: float = 0.25

    tag = "Spectrogram Analysis"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, signal: SignalType) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Perform spectrum analysis on the given signal using multiple methods.

        Parameters:
        -----------
        signal : Generator
            Input signal to be analyzed.

        Returns:
        --------
        psd_dict:
            PSD dictionary
        spec_dict
            spectrum dictionary
        """
        spec_dict: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)

        for chunk_index, signal_piece in enumerate(signal):
            spec_dict[chunk_index] = self.computing_spectrum(signal_piece)

        return spec_dict

    def computing_spectrum(self, signal: Signal) -> Dict[int, Dict[str, Any]]:
        spec_dict: Dict[int, Dict[str, Any]] = defaultdict(dict)

        nperseg = int(signal.rate * self.nperseg_ratio)
        noverlap = int(nperseg / 2)
        for channel in range(signal.number_of_channels):
            signal_no_bias = signal.data[:, channel] - np.mean(signal.data[:, channel])
            frequencies, times, Sxx = sig.spectrogram(
                signal_no_bias, fs=signal.rate, nperseg=nperseg, noverlap=noverlap
            )
            spec_dict[channel] = {
                "frequencies": frequencies,
                "times": times,
                "Sxx": Sxx,
            }
        return spec_dict

    def plot_spectrogram(self, output, input, show=False, save_path=None):
        """
        Plot spectrogram of the signal for given chunks and channels.

        Parameters:
        -----------
        output : tuple
            Output from the __call__ method, containing spectrogram data dictionary.
        show : bool, optional (default=False)
            If set to True, the spectrogram will be displayed.
        save_path : str, optional (default=None)
            If provided, the spectrogram plot will be saved to the given path with filenames indicating the chunk and channel.
        """
        spec_dict = output

        for chunk in spec_dict.keys():
            for channel in spec_dict[chunk].keys():

                spectrogram_data = spec_dict[chunk][channel]
                frequencies = spectrogram_data["frequencies"]
                times = spectrogram_data["times"]
                Sxx = spectrogram_data["Sxx"]
                Sxx = np.maximum(Sxx, 1e-2)
                Sxx_log = 10 * np.log10(Sxx)

                fig, ax = plt.subplots(2, 1, figsize=(14, 12))

                cax1 = ax[0].pcolormesh(
                    times, frequencies, Sxx_log, shading="gouraud", cmap="inferno"
                )
                ax[0].set_title("Spectrogram")
                ax[0].set_xlabel("Time (s)")
                ax[0].set_ylabel("Frequency (Hz)")
                ax[0].set_ylim(self.frequency_limit)
                ax[0].set_xlim(self.plotting_interval)
                for freq in [4, 8, 12, 30]:
                    ax[0].axhline(
                        y=freq,
                        color="black",
                        linestyle="--",
                        linewidth=1,
                        label=f"{freq} Hz",
                    )

                cax2 = ax[1].pcolormesh(
                    times, frequencies, Sxx_log, shading="gouraud", cmap="inferno"
                )
                ax[1].set_xlabel("Time (s)")
                ax[1].set_ylabel("Frequency (Hz)")
                ax[1].set_ylim([0, 12])
                ax[1].set_xlim(self.plotting_interval)
                for freq in [4, 8, 12, 30]:
                    ax[1].axhline(
                        y=freq,
                        color="black",
                        linestyle="--",
                        linewidth=1,
                        label=f"{freq} Hz",
                    )

                fig.colorbar(
                    cax1,
                    ax=ax[:],
                    location="right",
                    label="Power spectral density (dB/Hz)",
                    fraction=0.02,
                    pad=0.04,
                )

                # If Histogram is needed
                # psd_values = Sxx_log.flatten()
                # ax[1].hist(psd_values[psd_values > -40], bins=100, color='blue', alpha=0.7)
                # ax[1].set_title('Histogram of Power Spectral Density')
                # ax[1].set_xlabel('Power spectral density (dB/Hz)')
                # ax[1].set_ylabel('Count')

                if show:
                    plt.show()
                if save_path is not None:
                    plot_path = os.path.join(
                        save_path, f"Chunk{chunk}_Spectrogram_Channel_{channel}.png"
                    )
                    plt.savefig(plot_path, dpi=300)
                plt.close()
