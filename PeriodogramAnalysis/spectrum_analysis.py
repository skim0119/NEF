from typing import Any, Optional

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

    band_list: tuple[tuple[int, int], ...] = (
        (0.5, 4),
        (4, 8),
        (8, 12),
        (12, 30),
        (30, 100),
    )
    tag: str = "PowerSpectrum Analysis"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(
        self, psd_dict: dict[int, dict[int, dict[str, Any]]]
    ) -> tuple[
        dict[int, dict[int, dict[str, Any]]], dict[int, dict[int, dict[str, Any]]]
    ]:
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
        power_dict: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
        for chunk_idx in psd_dict.keys():
            power_dict[chunk_idx] = self.computing_absolute_and_relative_power(
                psd_dict[chunk_idx]
            )
            self.computing_ratio_and_bandpower(
                psd_dict[chunk_idx], power_dict[chunk_idx], chunk_idx
            )

        return psd_dict, power_dict

    def computing_absolute_and_relative_power(
        self, psd_dict: dict[int, dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:
        """
        Compute absolute and relative power for different frequency bands.

        Parameters:
        -----------
        psd_dict : dict
            Dictionary containing PSD values for each chunk and channel.

        Returns:
        --------
        power_dict
            Dictionary containing power values for all channels in this chunk.
        """
        power_dict: dict[int, dict[str, Any]] = defaultdict(list)

        for channel in psd_dict.keys():
            freqs = psd_dict[channel]["freqs"]
            psd = psd_dict[channel]["psd"]
            freq_res = freqs[1] - freqs[0]
            total_power = simpson(psd, dx=freq_res)

            if channel not in power_dict:
                power_dict[channel] = {
                    "psd_idx": [],
                    "power_list": [],
                    "rel_power_list": [],
                }

            for band in self.band_list:
                psd_idx = np.logical_and(freqs >= band[0], freqs <= band[1])

                power = simpson(psd[psd_idx], dx=freq_res)
                rel_power = power / total_power

                power_dict[channel]["psd_idx"].append(psd_idx)
                power_dict[channel]["power_list"].append(power)
                power_dict[channel]["rel_power_list"].append(rel_power)

        return power_dict

    def plot_periodogram(
        self,
        output,
        input,
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
    ):
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
        color_template = plt.cm.get_cmap(
            "tab20", len(self.band_list)
        )  # 使用 'tab20' 配色方案
        psd_dict = output[0]
        power_dict = output[1]

        for chunk in psd_dict.keys():
            for channel in psd_dict[chunk].keys():
                channel_folder = os.path.join(save_path, f"channel{channel:03d}")
                os.makedirs(channel_folder, exist_ok=True)

                freqs = psd_dict[chunk][channel]["freqs"]
                psd = psd_dict[chunk][channel]["psd"]
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                ax1.semilogx(freqs, psd, lw=1.5, color="k")
                ax1.set_xlabel("Frequency (Hz)")
                ax1.set_ylabel("Power spectral density (V^2 / Hz)")
                ax1.set_title(f"Welch's periodogram of channel {channel}")
                for i, (band_idx, label) in enumerate(
                    zip(power_dict[chunk][channel]["psd_idx"], self.band_list)
                ):
                    ax1.fill_between(
                        freqs,
                        psd,
                        where=band_idx,
                        color=color_template(i),
                        label=label,
                    )
                ax1.set_ylim([0, np.max(psd) * 1.1])
                ax1.set_xlim([1e-1, 100])
                ax1.legend()

                ax2.plot(freqs, psd, lw=1.5, color="k")
                ax2.set_xlabel("Frequency (Hz)")
                ax2.set_ylabel("Power spectral density (V^2 / Hz)")
                for i, (band_idx, label) in enumerate(
                    zip(power_dict[chunk][channel]["psd_idx"], self.band_list)
                ):
                    ax1.fill_between(
                        freqs,
                        psd,
                        where=band_idx,
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
        self,
        psd_dict: dict[int, dict[str, Any]],
        power_dict: dict[int, dict[str, Any]],
        chunk_index: int,
    ):
        """
        Compute power ratios and band powers for specific bands.

        Parameters:
        -----------
        psd_dict : dict
            Dictionary containing PSD values of each channel in this chunk.
        power_dict : dict
            Dictionary containing power values of each channel in this chunk.
        """
        for channel in psd_dict.keys():
            self.logger.info(f"Analysis of chunk {chunk_index}, channel {channel}\n")

            absolute_powers = power_dict[channel]["power_list"]
            relative_powers = power_dict[channel]["rel_power_list"]

            for band in self.band_list:
                multitaper_power = self.computing_bandpower(psd_dict[channel], band)
                multitaper_power_rel = self.computing_bandpower(
                    psd_dict[channel], band, relative=True
                )
                self.logger.info(
                    f"{band[0]}Hz to {band[1]}Hz: Power (absolute): {multitaper_power:.3f}\n"
                )
                self.logger.info(
                    f"{band[0]}Hz to {band[1]}Hz: Power (relative): {multitaper_power_rel:.3f}\n"
                )

            for band, abs_power, rel_power in zip(
                self.band_list, absolute_powers, relative_powers
            ):
                self.logger.info(
                    f"Absolute {band} power of channel {channel} is: {abs_power:.3f} uV^2\n"
                )
                self.logger.info(
                    f"Relative {band} power of channel {channel} is: {rel_power:.3f} uV^2\n\n"
                )

    def computing_bandpower(
        self,
        psd_dict_single_channel,
        band: tuple[float, float],
        relative: bool = False,
    ) -> float:
        """
        Calculate the power within a specified frequency band from PSD.

        Parameters
        ----------
        psd_dict_single_channel : dict
            A dictionary containing the PSD data for a single channel. It must include the following keys:
            - "psd": A NumPy array of PSD values.
            - "freqs": A NumPy array of corresponding frequency values in Hz.
        band : tuple of float
            The frequency band for which to calculate the power.
        relative : bool, optional
            If set to True, the method returns the relative power within the specified band by dividing the
            band power by the total power across all frequencies.

        Returns
        -------
        float
            The calculated power within the specified frequency band.
        """
        low, high = band
        psd = psd_dict_single_channel["psd"]
        freqs = psd_dict_single_channel["freqs"]
        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bp = simpson(psd[idx_band], dx=freq_res)

        if relative:
            bp /= simpson(psd, dx=freq_res)
        return bp
