import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.integrate import simpson
from dataclasses import dataclass, field
from collections import defaultdict

from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.typing import SignalType
from PowerSpectrumAnalysis import multitaper_psd

@dataclass
class PeriodogramAnalysis(OperatorMixin):
    """
       A class to perform Periodogram Analysis using Welch's method and multitaper PSD.

       Attributes:
       -----------
       exclude_channels : list
           List of channels to be excluded from analysis.
       band : list
           List of frequency bands to analyze, default is [[0.5, 4], [12, 30]].
       window_length_for_welch : float
           Window length in seconds for Welch's method.
    """

    exclude_channel_list: list = field(default_factory=list)
    band_list: list = field(default_factory=lambda: [[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 100]])
    window_length_for_welch: float = 4

    tag: str = "Periodogram Analysis"

    def __post_init__(self):
        super().__init__()

    # @cache_call
    def __call__(self, signal: SignalType):
        """
        Perform the periodogram analysis on the given signal.

        Parameters:
        -----------
        signal : SignalType
            Input signal to be analyzed.

        Returns:
        --------
        psd_dict:
            PSD dictionary
        num_of_chunks
            number of chunks
        power_dict
            power dictionary
        """
        psd_dict = defaultdict(dict)
        power_dict = defaultdict(dict)
        for chunk_idx, signal_piece in enumerate(signal):
            if chunk_idx > 1:
                break
            self.chunk_idx = chunk_idx
            num_of_chunks = self.chunk_idx + 1
            self.num_channels = signal_piece.number_of_channels

            # Compute psd_dict and power_dict for welch_periodogram plotting
            psd_dict[self.chunk_idx] = self.computing_welch_periodogram(signal_piece)
            power_dict[self.chunk_idx] = self.computing_absolute_and_relative_power(psd_dict[self.chunk_idx])

            # Compute band power and their ratio
            # self.computing_ratio_and_bandpower(signal_piece, power_dict, num_of_chunks)
        return psd_dict, num_of_chunks, power_dict

    def computing_welch_periodogram(self, signal):
        """
        Compute Welch's periodogram for the signal.

        Parameters:
        -----------
        signal : SignalType
            Input signal to be analyzed.

        Returns:
        --------
        dict
            Dictionary containing PSD values for each chunk and channel.
        """
        win = self.window_length_for_welch * signal.rate
        psd_dict = defaultdict(dict)
        for channel in range(self.num_channels):
            if channel in self.exclude_channel_list:
                continue
            freqs, psd = sig.welch(signal.data[:, channel], fs=signal.rate, nperseg=win)
            psd_dict[channel] = {
                'freqs': freqs,
                'psd': psd
            }
        return psd_dict

    def computing_absolute_and_relative_power(self, psd_dict):
        """
        Compute absolute and relative power for different frequency bands.

        Parameters:
        -----------
        psd_dict : dict
            Dictionary containing PSD values for each chunk and channel.
        num_of_chunks : int
            Number of chunks to divide the signal into.

        Returns:
        --------
        dict
            Dictionary containing power values for each chunk and channel.
        """
        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30),
            "gamma": (30, 100)
        }

        power_dict = defaultdict(dict)
        for channel in range(self.num_channels):
            if channel in self.exclude_channel_list:
                continue

            freqs = psd_dict[channel]['freqs']
            psd = psd_dict[channel]['psd']
            freq_res = freqs[1] - freqs[0]

            idx_delta = np.logical_and(freqs >= bands['delta'][0], freqs <= bands['delta'][1])
            idx_theta = np.logical_and(freqs >= bands['theta'][0], freqs <= bands['theta'][1])
            idx_alpha = np.logical_and(freqs >= bands['alpha'][0], freqs <= bands['alpha'][1])
            idx_beta = np.logical_and(freqs >= bands['beta'][0], freqs <= bands['beta'][1])
            idx_gamma = np.logical_and(freqs >= bands['gamma'][0], freqs <= bands['gamma'][1])

            delta_power = simpson(psd[idx_delta], dx=freq_res)
            theta_power = simpson(psd[idx_theta], dx=freq_res)
            alpha_power = simpson(psd[idx_alpha], dx=freq_res)
            beta_power = simpson(psd[idx_beta], dx=freq_res)
            gamma_power = simpson(psd[idx_gamma], dx=freq_res)
            total_power = simpson(psd, dx=freq_res)

            delta_rel_power = delta_power / total_power
            theta_rel_power = theta_power / total_power
            alpha_rel_power = alpha_power / total_power
            beta_rel_power = beta_power / total_power
            gamma_rel_power = gamma_power / total_power

            power_dict[channel] = {
                'idx_delta': idx_delta,
                'idx_theta': idx_theta,
                'idx_alpha': idx_alpha,
                'idx_beta': idx_beta,
                'idx_gamma': idx_gamma,
                'delta_power': delta_power,
                'theta_power': theta_power,
                'alpha_power': alpha_power,
                'beta_power': beta_power,
                'gamma_power': gamma_power,
                'delta_rel_power': delta_rel_power,
                'theta_rel_power': theta_rel_power,
                'alpha_rel_power': alpha_rel_power,
                'beta_rel_power': beta_rel_power,
                'gamma_rel_power': gamma_rel_power
            }

        return power_dict

    def plot_welch_periodogram(self, output, input, show=False, save_path=None):
        """
        Plot Welch's periodogram for the given signal.

        Parameters:
        -----------
        output : tuple
            Output from the __call__ method, including PSD dictionary and other information.
        input : SignalType
            Input signal for plotting.
        show : bool
            Flag to indicate whether to show the plot.
        save_path : str
            Path to save the plot.
        """
        psd_dict = output[0]
        num_of_segments = output[1]
        power_dict = output[2]

        for chunk in range(num_of_segments):
            for channel in range(self.num_channels):
                if channel in self.exclude_channel_list:
                    continue
                print(num_of_segments)
                print(chunk)
                print(f"{psd_dict.keys()}")
                freqs = psd_dict[chunk][channel]['freqs']
                psd = psd_dict[chunk][channel]['psd']
                plt.figure(figsize=(8, 4))
                plt.plot(freqs, psd, lw=1.5, color='k')

                plt.fill_between(freqs, psd, where=power_dict[chunk][channel]['idx_delta'], color='skyblue')
                plt.fill_between(freqs, psd, where=power_dict[chunk][channel]['idx_theta'], color='lightseagreen')
                plt.fill_between(freqs, psd, where=power_dict[chunk][channel]['idx_alpha'], color='goldenrod')
                plt.fill_between(freqs, psd, where=power_dict[chunk][channel]['idx_beta'], color='deeppink')
                plt.fill_between(freqs, psd, where=power_dict[chunk][channel]['idx_gamma'], color='khaki')

                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power spectral density (V^2 / Hz)')
                plt.ylim([0, np.max(psd_dict[chunk][channel]['psd']) * 1.1])
                plt.title(f"Welch's periodogram of channel {channel}")
                plt.xlim([0, 100])

                if show:
                    plt.show()
                if save_path is not None:
                    fig_save_path = os.path.join(save_path, "analysis_figures", f"welch_periodogram_of_chunk_{chunk}")
                    os.makedirs(fig_save_path, exist_ok=True)
                    plot_path = os.path.join(fig_save_path, f"welch_periodogram_of_channel_{channel}.png")
                    plt.savefig(plot_path, dpi=300)
                plt.close()

    def computing_ratio_and_bandpower(self, signal, power_dict, num_of_segments, ):
        """
        Compute power ratios and bandpowers for specific bands using Welch's and multitaper methods.

        Parameters:
        -----------
        signal : SignalType
            Input signal to be analyzed.
        power_dict : dict
            Dictionary containing power values for each chunk and channel.
        num_of_segments : int
            Number of segments to divide the signal into.
        """
        band = self.band
        band_a = band[0]
        band_b = band[1]

        for chunk in range(num_of_segments):
            for channel in range(self.num_channels):
                if channel in self.exclude_channel_list:
                    continue

                self.logger.info(f"Analysis of chunk {chunk}, channel {channel}\n")

                band_names = ['delta', 'theta', 'alpha', 'beta']
                absolute_powers = [power_dict[chunk][channel]['delta_power'],
                                    power_dict[chunk][channel]['theta_power'],
                                    power_dict[chunk][channel]['alpha_power'],
                                    power_dict[chunk][channel]['beta_power']]
                relative_powers = [power_dict[chunk][channel]['delta_rel_power'],
                                    power_dict[chunk][channel]['theta_rel_power'],
                                    power_dict[chunk][channel]['alpha_rel_power'],
                                    power_dict[chunk][channel]['beta_rel_power']]

                multitaper_power = self.computing_multitaper_bandpower(signal, band_a, channel)
                multitaper_power_rel = self.computing_multitaper_bandpower(signal, band_a, channel, relative=True)
                self.logger.info(f"{band_a[0]}Hz to {band_a[1]}Hz: Power (absolute) (Multitaper): {multitaper_power:.3f}\n")
                self.logger.info(f"{band_a[0]}Hz to {band_a[1]}Hz: Power (relative) (Multitaper): {multitaper_power_rel:.3f}\n")

                welch_power = self.computing_welch_bandpower(signal, band_a, channel) / self.computing_welch_bandpower(signal, band_b, channel)
                welch_power_rel = (self.computing_welch_bandpower(signal, band_a, channel, relative=True)
                            / self.computing_welch_bandpower(signal, band_b, channel, relative=True))

                self.logger.info(f"{band_a[0]}Hz to {band_a[1]}Hz / {band_b[0]}Hz to {band_b[1]}: Power ratio (absolute) (Welch): {welch_power:.3f}\n")
                self.logger.info(f"{band_a[0]}Hz to {band_a[1]}Hz / {band_b[0]}Hz to {band_b[1]}: Power ratio (relative) (Welch): {welch_power_rel:.3f}\n\n")

                for band, abs_power, rel_power in zip(band_names, absolute_powers, relative_powers):
                    self.logger.info(
                        f"Absolute {band} power (Welch) of channel {channel} is: {abs_power:.3f} uV^2\n")
                    self.logger.info(
                        f"Relative {band} power (Welch) of channel {channel} is: {rel_power:.3f} uV^2\n\n")

    def computing_multitaper_bandpower(self, signal, band, channel, relative=False):
        """
        Compute bandpower using multitaper method.

        Parameters:
        -----------
        signal : SignalType
            Input signal to be analyzed.
        band : list
            Frequency band of interest.
        channel : int
            Channel number to compute bandpower for.
        relative : bool
            If True, compute relative bandpower.

        Returns:
        --------
        float
            Computed bandpower.
        """
        band = np.asarray(band)
        low, high = band
        local_signal = signal.data
        frequency = signal.rate

        psd, freqs = multitaper_psd(local_signal[:, channel], frequency)
        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bp = simpson(psd[idx_band], dx=freq_res)

        if relative:
            bp /= simpson(psd, dx=freq_res)
        return bp

    def computing_welch_bandpower(self, signal, band, channel, relative=False):
        """
        Compute bandpower using Welch's method.

        Parameters:
        -----------
        signal : SignalType
            Input signal to be analyzed.
        band : list
            Frequency band of interest.
        channel : int
            Channel number to compute bandpower for.
        relative : bool
            If True, compute relative bandpower.

        Returns:
        --------
        float
            Computed bandpower.
        """
        low, high = band
        local_signal = signal.data
        frequency = signal.rate

        if self.window_length_for_welchis is not None:
            nperseg = self.window_length_for_welch * frequency 
        else:
            nperseg = (2 / low) * frequency
        freqs, psd = sig.welch(local_signal[:, channel], frequency, nperseg=nperseg)

        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bp = simpson(psd[idx_band], dx=freq_res)

        if relative:
            bp /= simpson(psd, dx=freq_res)
        return bp

