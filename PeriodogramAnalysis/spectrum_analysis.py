from typing import Optional

import os
from dataclasses import dataclass

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

from miv.core.operator_generator.operator import GeneratorOperatorMixin
from miv.core.operator_generator.wrapper import cache_generator_call


@dataclass
class PowerSpectrumAnalysis(GeneratorOperatorMixin):
    """
    A class to perform PSD Analysis using Welch, periodogram and multitaper PSD.
    The algorithm implementation is inspired from: <https://raphaelvallat.com/bandpower.html>

    Attributes:
    -----------
    band_list : list
        Frequency bands to analyze, default is [[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 100]], i.e. the five common frequency bands.
    """

    band_display: tuple[float, float] = (0, 100)
    band_list: tuple[tuple[float, float], ...] = (
        (0.5, 4),
        (4, 8),
        (8, 12),
        (12, 30),
        (30, 100),
    )
    tag: str = "PowerSpectrum Analysis"

    def __post_init__(self) -> None:
        super().__init__()
        if not len(self.band_display) == 2:
            raise ValueError("band_display must be a tuple with two values.")
        if not (
            self.band_display[0] >= 0 and self.band_display[1] > self.band_display[0]
        ):
            raise ValueError(
                "band_display must contain two non-negative values, where band_display[0] < band_display[1]."
            )
        for band in self.band_list:
            if not (isinstance(band, tuple) and len(band) == 2):
                raise ValueError(
                    "Each band in band_list must be a tuple with two values."
                )
            low, high = band
            if not (low >= 0 and low < high):
                raise ValueError(f"Band {band} must satisfy 0 <= low < high.")

        self.num_channel: Optional[int] = None
        self.chunk: int = -1

    @cache_generator_call
    def __call__(self, input: tuple) -> tuple:
        """
        Perform the periodogram analysis on the given signal.

        Parameters:
        -----------
        signal : Generator
            Input signal to be analyzed.

        Returns:
        --------
        psd_dict:
            PSD dictionary
        power_dict
            power dictionary
        """
        freqs = input[0]
        psd = input[1]
        self.chunk += 1
        self.num_channel = psd.shape[1]

        psd_idx, power, rel_power = self.computing_absolute_and_relative_power(
            freqs, psd
        )
        self.computing_ratio_and_bandpower(power, rel_power)

        return freqs, psd, psd_idx

    def computing_absolute_and_relative_power(
        self, freqs: np.ndarray, psd: np.ndarray
    ) -> tuple:
        """
        Compute absolute and relative power for different frequency bands.

        Parameters:
        -----------
        psd_dict : dict
            Dictionary containing PSD values for each chunk and channel.

        Returns:
        --------
        power_dict
            Dictionary containing power values.
        """
        psd_idx_list: list = []
        power_list: list = []
        rel_power_list: list = []

        freq_res = freqs[1] - freqs[0]

        for ch in range(self.num_channel):
            total_power = simpson(psd[:, ch], dx=freq_res)

            for band in self.band_list:
                psd_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
                power = simpson(psd[psd_idx, ch], dx=freq_res)
                rel_power = power / total_power

                psd_idx_list.append(psd_idx)
                power_list.append(power)
                rel_power_list.append(rel_power)

        return (
            np.array(psd_idx_list).T,
            np.array(power_list).T,
            np.array(rel_power_list).T,
        )

    def computing_ratio_and_bandpower(
        self, power: np.ndarray, rel_power: np.ndarray
    ) -> tuple:
        """
        Compute power ratios and band powers for specific bands.

        Parameters:
        -----------
        psd_dict : dict
            Dictionary containing PSD values of each channel in this chunk.
        power_dict : dict
            Dictionary containing power values of each channel in this chunk.
        """
        absolute_powers = power[
            (self.chunk * len(self.band_list)) : (
                (self.chunk + 1) * len(self.band_list)
            )
        ]
        relative_powers = rel_power[
            (self.chunk * len(self.band_list)) : (
                (self.chunk + 1) * len(self.band_list)
            )
        ]

        for channel in range(self.num_channel):
            self.logger.info(f"Analysis of chunk {self.chunk}, channel {channel}\n")

            for band, abs_power, rel_power in zip(
                self.band_list, absolute_powers, relative_powers
            ):
                self.logger.info(
                    f"{band}: Absolute power of channel {channel} is: {abs_power:.3f} uV^2\n"
                )
                self.logger.info(
                    f"{band}: Relative power of channel {channel} is: {rel_power:.3f} uV^2\n\n"
                )
        return absolute_powers, relative_powers

    def plot_periodogram(
        self,
        output: tuple,
        input: None,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> None:

        color_template = plt.cm.get_cmap("tab20", len(self.band_list))

        for freqs, psd, psd_idx in output:

            for channel in range(psd.shape[1]):
                if save_path is not None:
                    channel_folder = os.path.join(save_path, f"channel{channel:03d}")
                    os.makedirs(channel_folder, exist_ok=True)

                fig1, ax = plt.subplots(figsize=(8, 6))
                ax.plot(
                    freqs,
                    psd[:, channel],
                    lw=1,
                    color="k",
                )
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Power/Frequency (dB/Hz)")
                ax.set_title(self.tag)
                ax.set_xlim(self.band_display)
                ax.set_ylim(ymin=0)

                if show:
                    plt.show()
                if save_path is not None:
                    plot_path = os.path.join(
                        channel_folder, f"power_density_{self.chunk:03d}.png"
                    )
                    plt.savefig(plot_path, dpi=300)

                fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                ax1.semilogx(freqs, psd[:, channel], lw=1.5, color="k")
                ax1.set_xlabel("Frequency (Hz)")
                ax1.set_ylabel("Power spectral density (V^2 / Hz)")
                ax1.set_title(f"Welch's periodogram of channel {channel}")

                for i in range(len(self.band_list)):
                    ax1.fill_between(
                        freqs,
                        psd[:, channel],
                        where=psd_idx[:, i],
                        color=color_template(i),
                        label=self.band_list[i],
                    )
                ax1.set_ylim([0, np.max(psd) * 1.1])
                ax1.set_xlim([1e-1, 100])
                ax1.legend()

                ax2.plot(freqs, psd[:, channel], lw=1.5, color="k")
                ax2.set_xlabel("Frequency (Hz)")
                ax2.set_ylabel("Power spectral density (V^2 / Hz)")
                for i in range(len(self.band_list)):
                    ax2.fill_between(
                        freqs,
                        psd[:, channel],
                        where=psd_idx[:, i],
                        color=color_template(i),
                        label=self.band_list[i],
                    )
                ax2.set_ylim([0, np.max(psd) * 1.1])
                ax2.set_xlim([0, 100])
                ax2.legend()

                if show:
                    plt.show()
                if save_path is not None:
                    plot_path = os.path.join(
                        channel_folder,
                        f"periodogram_{self.chunk:03d}.png",
                    )
                    plt.savefig(plot_path, dpi=300)
                plt.close("all")
