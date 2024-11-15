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
from PeriodogramAnalysis import PeriodogramAnalysis


@dataclass
class SpectrumAnalysis(OperatorMixin):
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

    band_display: list[float] = field(default_factory=lambda: [0, 100])
    window_length_for_welch: int = 4
    frequency_limit: list[float] = field(default_factory=lambda: [0.5, 100])
    plotting_interval: list[float] = field(default_factory=lambda: [0, 60])
    nperseg_ratio: float = 0.25

    tag = "Spectrum_Analysis"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(
        self, signal: SignalType
    ) -> Tuple[
        Dict[int, Dict[int, Dict[str, Any]]], Dict[int, Dict[int, Dict[str, Any]]],
        Dict[int, Dict[int, Dict[str, Any]]], Dict[int, Dict[int, Dict[str, Any]]]
    ]:
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
        psd_welch_dict: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        psd_periodogram_dict: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        psd_multitaper_dict: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        spec_dict: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)

        periodogramanalysis = PeriodogramAnalysis(window_length_for_welch = self.window_length_for_welch)
        for chunk_index, signal_piece in enumerate(signal):
            # Plot spectrum using different methods
            psd_welch_dict[chunk_index] = periodogramanalysis.SpectrumAnalysisWelch(signal_piece)
            psd_periodogram_dict[chunk_index] = periodogramanalysis.SpectrumAnalysisPeriodogram(signal_piece)
            psd_multitaper_dict[chunk_index] = periodogramanalysis.SpectrumAnalysisMultitaper(signal_piece)
            # Plot spectrogram
            spec_dict[chunk_index] = self.computing_spectrum(signal_piece)

        return psd_welch_dict, psd_periodogram_dict, psd_multitaper_dict, spec_dict

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

    def plot_spectrum_methods(
        self,
        output,
        input,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ):
        """
        Plot spectrum estimates using different methods (Periodogram, Welch, Multitaper).

        Parameters:
        -----------
        output : tuple
            Output from the __call__ method, which includes the PSD dictionary.
        show : bool, optional (default=False)
            If set to True, the plot will be displayed.
        save_path : str, optional (default=None)
            If provided, the plot will be saved to the given path with filenames indicating the chunk and channel.
        """
        psd_welch_dict = output[0]
        psd_periodogram_dict = output[1]
        psd_multitaper_dict = output[2]

        for chunk in psd_welch_dict.keys():
            for channel in psd_welch_dict[chunk].keys():

                fig, (ax1, ax2, ax3) = plt.subplots(
                    1, 3, figsize=(12, 4), sharex=True, sharey=True
                )

                psd = [
                    (ax1, psd_periodogram_dict, "freqs", "psd"),
                    (ax2, psd_welch_dict, "freqs", "psd"),
                    (ax3, psd_multitaper_dict, "freqs", "psd"),
                ]
                for ax, psd_dict, fregs_type, psd_type in psd:
                    ax.stem(
                        psd_dict[chunk][channel][fregs_type],
                        psd_dict[chunk][channel][psd_type],
                        linefmt="slategrey",
                        basefmt=" ",
                        markerfmt=" ",
                    )
                    ax.plot(
                        psd_dict[chunk][channel][fregs_type],
                        psd_dict[chunk][channel][psd_type],
                        lw=2,
                        color="k",
                    )

                ax1.set_xlabel("Frequency (Hz)")
                ax1.set_ylabel("Decibels (dB / Hz)")
                ax1.set_title("Periodogram")
                ax2.set_title("Welch")
                ax3.set_title("Multitaper")
                ax1.set_xlim(self.band_display)

                ax1.set_ylim(ymin=0)
                ax2.set_ylim(ymin=0)
                ax3.set_ylim(ymin=0)
                if show:
                    plt.show()
                if save_path is not None:
                    plot_path = os.path.join(
                        save_path,
                        f"Chunk{chunk}_Comparison_figure_channel_{channel}.png",
                    )
                    plt.savefig(plot_path, dpi=300)
                plt.close()

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
        spec_dict = output[3]

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
