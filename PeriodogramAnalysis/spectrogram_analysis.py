from typing import Any

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import scipy
from collections import defaultdict

from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.core.datatype import Signal
from miv.typing import SignalType


@dataclass
class SpectrogramAnalysis(OperatorMixin):
    """
    A class to perform Spectrum Analysis using multiple methods including Welch's, Periodogram, and multitaper PSD.

    Attributes:
    -----------
    frequency_limit : list
        Frequency range limit for analysis.
    nperseg : int
        Number of points per segment for spectrogram computation.
    noverlap : int
        Number of points to overlap between segments for spectrogram.
    """

    frequency_limit: tuple[float, float] = (0.5, 100)
    plotting_interval: tuple[float, float] = (0, 60)
    nperseg_ratio: float = 0.25
    tag = "Spectrogram Analysis"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, signal: SignalType) -> dict[int, dict[str, Any]]:
        """
        Perform spectrum analysis on the given signal.

        Parameters:
        -----------
        signal : Generator
            Input signal to be analyzed.

        Returns:
        --------
        spec_dict
            spectrum dictionary
        """
        spec_dict: dict[int, dict[str, Any]] = defaultdict(dict)

        for chunk_index, signal_piece in enumerate(signal):
            spec_dict = self.computing_spectrum(signal_piece, spec_dict)

        return spec_dict

    def computing_spectrum(
        self, signal: Signal, spec_dict: dict[int, dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:
        nperseg = int(signal.rate * self.nperseg_ratio)
        noverlap = int(nperseg / 2)
        for channel_index, channel_signal in enumerate(signal):
            signal_no_bias = channel_signal - np.mean(channel_signal)
            frequencies, times, sxx = scipy.signal.spectrogram(
                signal_no_bias, fs=signal.rate, nperseg=nperseg, noverlap=noverlap
            )
            if channel_index not in spec_dict:
                spec_dict[channel_index] = {"frequencies": [], "times": [], "Sxx": []}
            spec_dict[channel_index]["frequencies"].append(frequencies)
            spec_dict[channel_index]["times"].append(times)
            spec_dict[channel_index]["Sxx"].append(sxx)

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

        for channel in spec_dict.keys():
            for chunk in range(len(spec_dict[channel]["frequencies"])):
                channel_folder = os.path.join(save_path, f"channel{channel:03d}")
                os.makedirs(channel_folder, exist_ok=True)

                frequencies = spec_dict[channel]["frequencies"][chunk]
                times = spec_dict[channel]["times"][chunk]
                sxx = spec_dict[channel]["Sxx"][chunk]
                sxx = np.maximum(sxx, 1e-2)
                sxx_log = 10 * np.log10(sxx)

                fig, ax = plt.subplots(2, 1, figsize=(14, 12))

                cax1 = ax[0].pcolormesh(
                    times, frequencies, sxx_log, shading="gouraud", cmap="inferno"
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
                    times, frequencies, sxx_log, shading="gouraud", cmap="inferno"
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
                        channel_folder, f"spectrogram_{chunk:03d}.png"
                    )
                    plt.savefig(plot_path, dpi=300)
                plt.close("all")
