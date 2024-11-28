from typing import Any, Optional, Tuple

import os
from dataclasses import dataclass

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from collections import defaultdict

from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call


@dataclass
class PowerSpectrumAnalysis(OperatorMixin):
    """
    A class to perform PSD Analysis using Welch, periodogram and multitaper PSD.
    The algorithm implementation is inspired from: <https://raphaelvallat.com/bandpower.html>

    Attributes:
    -----------
    band_list : list
        Frequency bands to analyze, default is [[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 100]], i.e. the five common frequency bands.
    """

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

    @cache_call
    def __call__(
        self, psd_dict: dict[int, dict[str, Any]]
    ) -> tuple:
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
        power_dict: dict[int, dict[str, Any]] = defaultdict(dict)
        for channel in psd_dict.keys():
            if channel not in power_dict:
                power_dict[channel] = {
                    "psd_idx": [],
                    "power_list": [],
                    "rel_power_list": [],
                }
            power_dict[channel] = self.computing_absolute_and_relative_power(
                psd_dict[channel], power_dict[channel]
            )
            self.computing_ratio_and_bandpower(
                psd_dict[channel], power_dict[channel], channel
            )

        return psd_dict, power_dict

    def computing_absolute_and_relative_power(
        self, psd_dict: dict[str, Any], power_dict: dict[str, Any]
    ) -> dict[str, Any]:
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

        for chunk in range(len(psd_dict["freqs"])):
            freqs = psd_dict["freqs"][chunk]
            psd = psd_dict["psd"][chunk]
            freq_res = freqs[1] - freqs[0]
            total_power = simpson(psd, dx=freq_res)

            for band in self.band_list:
                psd_idx = np.logical_and(freqs >= band[0], freqs <= band[1])

                power = simpson(psd[psd_idx], dx=freq_res)
                rel_power = power / total_power

                power_dict["psd_idx"].append(psd_idx)
                power_dict["power_list"].append(power)
                power_dict["rel_power_list"].append(rel_power)

        return power_dict

    def plot_periodogram(
        self,
        output: tuple,
        input: None,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ) -> None:
        """
        Plot periodogram for the given signal, w.r.t. all channels in all chunks.

        Parameters:
        -----------
        output : tuple
            Output from the __call__ method, including PSD dictionary and other information.
        show : bool
            Flag to indicate whether to show the plot.
        save_path : str
            Path to save the plot.
        """
        color_template = plt.cm.get_cmap("tab20", len(self.band_list))
        psd_dict = output[0]
        power_dict = output[1]

        for channel in psd_dict.keys():
            for chunk in range(len(psd_dict[channel]["freqs"])):
                if save_path is not None:
                    channel_folder = os.path.join(save_path, f"channel{channel:03d}")
                    os.makedirs(channel_folder, exist_ok=True)

                freqs = psd_dict[channel]["freqs"][chunk]
                psd = psd_dict[channel]["psd"][chunk]
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                ax1.semilogx(freqs, psd, lw=1.5, color="k")
                ax1.set_xlabel("Frequency (Hz)")
                ax1.set_ylabel("Power spectral density (V^2 / Hz)")
                ax1.set_title(f"Welch's periodogram of channel {channel}")
                for i, (band_idx_list, label) in enumerate(
                    zip(
                        power_dict[channel]["psd_idx"][
                            chunk
                            * len(self.band_list) : (chunk + 1)
                            * len(self.band_list)
                        ],
                        self.band_list,
                    )
                ):
                    ax1.fill_between(
                        freqs,
                        psd,
                        where=band_idx_list,
                        color=color_template(i),
                        label=label,
                    )
                ax1.set_ylim([0, np.max(psd) * 1.1])
                ax1.set_xlim([1e-1, 100])
                ax1.legend()

                ax2.plot(freqs, psd, lw=1.5, color="k")
                ax2.set_xlabel("Frequency (Hz)")
                ax2.set_ylabel("Power spectral density (V^2 / Hz)")
                for i, (band_idx_list, label) in enumerate(
                    zip(
                        power_dict[channel]["psd_idx"][
                            chunk
                            * len(self.band_list) : (chunk + 1)
                            * len(self.band_list)
                        ],
                        self.band_list,
                    )
                ):
                    ax2.fill_between(
                        freqs,
                        psd,
                        where=band_idx_list,
                        color=color_template(i),
                        label=label,
                    )
                ax2.set_ylim([0, np.max(psd) * 1.1])
                ax2.set_xlim([0, 100])
                ax2.legend()

                if show:
                    plt.show()
                if save_path is not None:
                    plot_path = os.path.join(
                        channel_folder,
                        f"periodogram_{chunk:03d}.png",
                    )
                    plt.savefig(plot_path, dpi=300)
                plt.close()

    def computing_ratio_and_bandpower(
        self, psd_dict: dict[str, Any], power_dict: dict[str, Any], channel: int
    ) -> None:
        """
        Compute power ratios and band powers for specific bands.

        Parameters:
        -----------
        psd_dict : dict
            Dictionary containing PSD values of each channel in this chunk.
        power_dict : dict
            Dictionary containing power values of each channel in this chunk.
        """
        for chunk in range(len(psd_dict["freqs"])):
            self.logger.info(f"Analysis of chunk {chunk}, channel {channel}\n")

            absolute_powers = power_dict["power_list"][
                (chunk * len(self.band_list)) : ((chunk + 1) * len(self.band_list))
            ]
            relative_powers = power_dict["rel_power_list"][
                (chunk * len(self.band_list)) : ((chunk + 1) * len(self.band_list))
            ]

            for band, abs_power, rel_power in zip(
                self.band_list, absolute_powers, relative_powers
            ):
                self.logger.info(
                    f"{band}: Absolute power of channel {channel} is: {abs_power:.3f} uV^2\n"
                )
                self.logger.info(
                    f"{band}: Relative power of channel {channel} is: {rel_power:.3f} uV^2\n\n"
                )
