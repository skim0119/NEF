from typing import Any, Optional

import pathlib
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import scipy
from collections import defaultdict

from miv.core.operator_generator.operator import GeneratorOperatorMixin
from miv.core.operator_generator.wrapper import cache_generator_call
from miv.core.datatype import Signal
from miv.typing import SignalType


@dataclass
class SpectrogramAnalysis(GeneratorOperatorMixin):
    """
    A class to perform Spectrum Analysis using multiple methods including Welch's, Periodogram, and multitaper PSD.

    Attributes:
    -----------
    frequency_limit : list
        Frequency range limit for analysis.
    nperseg_ratio : float
        Specifies the ratio of the number of points per segment (`nperseg`) to the sampling rate.
    """

    frequency_limit: tuple[float, float] = (0.5, 100)
    plotting_interval: tuple[float, float] = (0, 60)
    nperseg_ratio: float = 0.25
    tag = "Spectrogram Analysis"

    def __post_init__(self) -> None:
        super().__init__()
        if not len(self.frequency_limit) == 2:
            raise ValueError("frequency_limit must be a tuple with two values.")
        if self.frequency_limit[0] >= self.frequency_limit[1]:
            raise ValueError("frequency_limit[0] must be less than frequency_limit[1].")

        if not len(self.plotting_interval) == 2:
            raise ValueError("plotting_interval must be a tuple with two values.")
        if self.plotting_interval[0] < 0 or self.plotting_interval[1] > 60:
            raise ValueError(
                "plotting_interval values must be within the range [0, 60]."
            )
        if self.plotting_interval[0] >= self.plotting_interval[1]:
            raise ValueError(
                "plotting_interval[0] must be less than plotting_interval[1]."
            )

        if not self.nperseg_ratio > 0:
            raise ValueError("nperseg_ratio must be a positive number.")

        self.num_channel: int = 0
        self.chunk: int = -1

    @cache_generator_call
    def __call__(self, signal: Signal) -> tuple:
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
        self.rate = signal.rate
        self.num_channel = signal.number_of_channels
        self.chunk += 1

        return self.computing_spectrum(signal.data)

    def computing_spectrum(self, data: np.ndarray) -> tuple:

        nperseg = int(self.rate * self.nperseg_ratio)
        noverlap = int(nperseg / 2)

        sxx_list: list = []

        for ch in range(self.num_channel):
            signal_no_bias = data[:, ch] - np.mean(data[:, ch])
            frequencies, times, sxx = scipy.signal.spectrogram(
                signal_no_bias, fs=self.rate, nperseg=nperseg, noverlap=noverlap
            )

            sxx_list.append(sxx)

        return frequencies, times, np.array(sxx_list)

    def plot_spectrogram(
        self,
        output: tuple,
        input: None,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> None:
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
        for frequencies, times, sxx in output:
            for channel in range(self.num_channel):
                if save_path is not None:
                    channel_folder = os.path.join(save_path, f"channel{channel:03d}")
                    os.makedirs(channel_folder, exist_ok=True)

                sxx_channel = np.maximum(sxx[channel, :, :], 1e-2)
                sxx_log = 10 * np.log10(sxx_channel)

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
                        channel_folder, f"spectrogram_{self.chunk:03d}.png"
                    )
                    plt.savefig(plot_path, dpi=300)
                plt.close("all")
