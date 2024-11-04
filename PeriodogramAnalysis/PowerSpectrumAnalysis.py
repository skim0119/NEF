import os

import inspect
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.integrate import simpson
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import dpss
from dataclasses import dataclass, field
from typing import Optional, Union

from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.core.datatype import Signal
from miv.typing import SignalType

@ dataclass()
class PowerSpectrumAnalysis(OperatorMixin):
    channel: Optional[list] = field(default_factory=list)
    band: Optional[Union[list, tuple]] = None
    band_display: Optional[list] = None
    time: Optional[list] = None
    win_sec: Optional[float] = None
    frequency_limit: Optional[list] = None
    nperseg: int = 2048
    noverlap: int = 1024
    db: bool = True

    tag = "Power Spectrum Analysis"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, signal: SignalType):

        self.accumulated_signal = None
        if not inspect.isgenerator(signal):
            self.accumulated_signal = signal
        else:
            for data_chunk in signal:
                if self.accumulated_signal == None:
                    self.accumulated_signal = Signal(data=data_chunk.data, timestamps=data_chunk.timestamps, rate=data_chunk.rate)
                    self.accumulated_signal.rate = data_chunk.rate
                else:
                    self.accumulated_signal.data = np.append(self.accumulated_signal.data, data_chunk.data, axis=0)
                    self.accumulated_signal.timestamps = np.append(self.accumulated_signal.timestamps, data_chunk.timestamps)
            print(f"Signal shape: {self.accumulated_signal.shape}")

        return (self.accumulated_signal, self.channel, self.band, self.band_display, self.time, self.win_sec, self.frequency_limit,
                self.nperseg, self.noverlap, self.db)

    def plot_view_raw_data(self, output, input, show=False, save_path=None):
        accumulated_signal = output[0]
        channel_list = output[1]
        time = output[4]

        sns.set_theme(font_scale=1.2)

        local_signal = accumulated_signal.data
        frequency = accumulated_signal.rate
        if not channel_list:
            channel_list = list(range(local_signal.shape[1]))

        for channel in channel_list:
            # Define sampling frequency and time vector for the current chunk
            start_idx, end_idx = int(time[0] * frequency), int(time[1] * frequency)

            data = local_signal[start_idx:end_idx, channel]
            time_vector = np.linspace(time[0], time[1], len(data))

            # Plot the signal for the current chunk
            plt.figure(figsize=(12, 4))
            plt.plot(time_vector, data, lw=0.5, color='k')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Voltage')
            plt.xlim([time_vector.min(), time_vector.max()])
            plt.title(f'N3 sleep EEG data (Channel {channel})')
            sns.despine()
            if show:
                plt.show()
            if save_path is not None:
                fig_save_path = os.path.join(save_path, "Raw data plot")
                os.makedirs(fig_save_path, exist_ok=True)
                plot_path = os.path.join(fig_save_path, f'Wave_display_channel_{channel}.png')
                plt.savefig(plot_path, dpi=300)
            plt.close()

    def plot_welch_periodogram(self, output, input, show=False, save_path=None):
        accumulated_signal = output[0]
        channel_list = output[1]
        band = output[2]
        win_sec = output[5]

        local_signal = accumulated_signal.data
        frequency = accumulated_signal.rate

        win = 4 * frequency
        if not channel_list:
            channel_list = list(range(local_signal.shape[1]))

        for channel in channel_list:
            freqs, psd = signal.welch(local_signal[:, channel], fs=frequency, nperseg=win)

            # Plot the power spectrum for the current channel
            sns.set(font_scale=1.2, style='white')
            plt.figure(figsize=(8, 4))
            plt.plot(freqs, psd, lw=1.5, color='k')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power spectral density (V^2 / Hz)')
            plt.ylim([0, np.max(psd) * 1.1])
            plt.title(f"Welch's periodogram of channel {channel}")
            plt.xlim([0, 100])
            sns.despine()
            if show:
                plt.show()
            if save_path is not None:
                fig_save_path = os.path.join(save_path, "Analysis figures")
                os.makedirs(fig_save_path, exist_ok=True)
                plot_path = os.path.join(fig_save_path, f"Welch's periodogram of channel {channel}.png")
                plt.savefig(plot_path, dpi=300)
            plt.close()

            self.define_delta_band(freqs, psd, channel, show, save_path)
            self.computing_average_band_power(freqs, psd, channel, save_path)

        if save_path is not None:
            self.computing_ratio_between_bands(accumulated_signal, win_sec, channel_list, band[0], band[1], save_path)

    def define_delta_band(self, freqs, psd, channel, show, save_path):
        # Define delta lower and upper limits
        low, high = 0.5, 4

        # Find intersecting values in frequency vector
        idx_delta = np.logical_and(freqs >= low, freqs <= high)

        # Plot the power spectral density and fill the delta area
        plt.figure(figsize=(7, 4))
        plt.plot(freqs, psd, lw=2, color='k')
        plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (uV^2 / Hz)')
        plt.xlim([0, 100])
        plt.ylim([0, np.max(psd) * 1.1])
        plt.title("Welch's periodogram")
        sns.despine()
        if show:
            plt.show()
        if save_path is not None:
            fig_save_path = os.path.join(save_path, "Analysis figures")
            os.makedirs(fig_save_path, exist_ok=True)
            plot_path = os.path.join(fig_save_path, f"Welch's periodogram (delta band) {channel}.png")
            plt.savefig(plot_path, dpi=300)
        plt.close()

    def computing_average_band_power(self, freqs, psd, channel, save_path):

        low, high = 0.5, 4
        idx_delta = np.logical_and(freqs >= low, freqs <= high)
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

        # Compute the absolute power by approximating the area under the curve
        delta_power = simpson(psd[idx_delta], dx=freq_res)
        if save_path is not None:
            summary_file_path = os.path.join(save_path, "Summary file.txt")

            # Relative delta power (expressed as a percentage of total power)
            total_power = simpson(psd, dx=freq_res)
            delta_rel_power = delta_power / total_power

            with open(summary_file_path, 'a') as summary_file:
                summary_file.write(f'Absolute delta power (Welch) of channel {channel} is: {delta_power:.3f} uV^2\n')
                summary_file.write(f'Relative delta power (Welch) of channel {channel} is: {delta_rel_power:.3f} uV^2\n')

    def computing_ratio_between_bands(self, accumulated_signal, win_sec, channel_list, band_a, band_b, save_path):
        """Compute and log the power ratio between specified frequency bands for each channel.

        This function calculates the absolute and relative power for a specified band (e.g., delta)
        and the power ratio (e.g., delta/beta) for each channel in the `channel_list`. The results are
        computed using both Multitaper and Welch methods and are saved in a summary file.

        Parameters
        ----------
        win_sec : float
            Window length in seconds for computing the Welch power spectral density (PSD).
        channel_list : list of int
            A list of channel indices for which to calculate band power and power ratios.
        band_a : list of float
            A two-element list specifying the lower and upper frequencies of the first band of interest (e.g., delta).
        band_b : list of float
            A two-element list specifying the lower and upper frequencies of the second band of interest (e.g., beta).
        """
        if not channel_list:
            channel_list = list(range(accumulated_signal.data.shape[1]))
        summary_file_path = os.path.join(save_path, "Band Power comparison.txt")

        for channel in channel_list:
            with open(summary_file_path, 'a') as summary_file:
                summary_file.write(f"Power of channel {channel}\n")
            bp = self.computing_bandpower(band_a, channel,'multitaper')
            bp_rel = self.computing_bandpower(band_a, channel,'multitaper', relative=True)
            with open(summary_file_path, 'a') as summary_file:
                summary_file.write(f"{band_a[0]}Hz to {band_a[1]}Hz: Power (absolute) (Multitaper): {bp:.3f}\n")
                summary_file.write(f"{band_a[0]}Hz to {band_a[1]}Hz: Power (relative) (Multitaper): {bp_rel:.3f}\n\n")

            # Delta/beta ratio based on the absolute power
            with open(summary_file_path, 'a') as summary_file:
                summary_file.write(f"Power ratio of channel {channel}\n")
            db = self.computing_bandpower(band_a, channel, window_sec=win_sec) / self.computing_bandpower(band_b, channel, window_sec=win_sec)
            # Delta/beta ratio based on the relative power
            db_rel = (self.computing_bandpower(band_a, channel, window_sec=win_sec, relative=True)
                      / self.computing_bandpower(band_b, channel, window_sec=win_sec, relative=True))
            with open(summary_file_path, 'a') as summary_file:
                summary_file.write(f"{band_a[0]}Hz to {band_a[1]}Hz / {band_b[0]}Hz to {band_b[1]}: Power ratio (absolute) (Welch): {db:.3f}\n")
                summary_file.write(f"{band_a[0]}Hz to {band_a[1]}Hz / {band_b[0]}Hz to {band_b[1]}: Power ratio (relative) (Welch): {db_rel:.3f}\n\n")

    def computing_bandpower(self, band, channel, method='welch', window_sec=None, relative=False):
        """Compute the average power of the signal in a specific frequency band.

        This function calculates the power of a specific frequency band for the given EEG data channel using different methods (e.g., Welch or multitaper). The power can be computed either as an absolute value or relative to the total power of the signal.

        Parameters
        ----------
        band : list
            A list specifying the lower and upper frequencies of the band of interest (e.g., [0.5, 4] for the delta band).
        channel : int
            The index of the EEG data channel to analyze.
        method : str, optional
            The method to use for power spectral density estimation. Options are 'welch' (default) or 'multitaper'.
        window_sec : float, optional
            The length of each window in seconds for Welch's method. If None, the default window length is set to (1 / min(band)) * 2.
        relative : bool, optional
            If True, compute the relative power (as a fraction of total power). If False (default), compute the absolute power.

        Returns
        -------
        bp : float
            The computed band power, either as an absolute value or as a relative percentage of total power.
        """

        band = np.asarray(band)
        low, high = band
        local_signal = self.accumulated_signal.data
        frequency = self.accumulated_signal.rate

        if method == 'welch':
            if window_sec is not None:
                nperseg = window_sec * frequency
            else:
                nperseg = (2 / low) * frequency
            freqs, psd = signal.welch(local_signal[:, channel], frequency, nperseg=nperseg)

        elif method == 'multitaper':
            psd, freqs = self.multitaper_psd(local_signal[:, channel], frequency)

        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        bp = simpson(psd[idx_band], dx=freq_res)

        summary_file_path = os.path.join(self.analysis_path, "Band Power comparison.txt")
        if relative:
            bp /= simpson(psd, dx=freq_res)
            with open(summary_file_path, 'a') as summary_file:
                summary_file.write(f"{low}Hz to {high}Hz: Relative band power: {bp:.3f} ({method})\n")
        else:
            with open(summary_file_path, 'a') as summary_file:
                summary_file.write(f"{low}Hz to {high}Hz: Band power: {bp:.3f} uV^2 ({method})\n")

        return bp

    def plot_spectrum_methods(self, output, input, show=False, save_path=None):
        accumulated_signal = output[0]
        channel_list = output[1]
        band = output[3]
        win_sec = output[5]
        dB = output[9]

        sns.set_theme(style="white", font_scale=1.2)
        local_signal = accumulated_signal.data
        frequency = accumulated_signal.rate
        if not channel_list:
            channel_list = list(range(local_signal.shape[1]))

        for channel in channel_list:
            freqs, psd = signal.periodogram(local_signal[:, channel], frequency)
            freqs_welch, psd_welch = signal.welch(local_signal[:, channel], frequency, nperseg=win_sec * frequency)
            psd_mt, freqs_mt = self.multitaper_psd(local_signal[:, channel], frequency)
            sharey = False

            # Optional: convert power to decibels (dB = 10 * log10(power))
            if dB:
                psd = 10 * np.log10(psd)
                psd_welch = 10 * np.log10(psd_welch)
                psd_mt = 10 * np.log10(psd_mt)
                sharey = True

            # Start plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=sharey)
            sc = 'slategrey'
            ax1.stem(freqs, psd, linefmt=sc, basefmt=" ", markerfmt=" ")
            ax2.stem(freqs_welch, psd_welch, linefmt=sc, basefmt=" ", markerfmt=" ")
            ax3.stem(freqs_mt, psd_mt, linefmt=sc, basefmt=" ", markerfmt=" ")
            lc, lw = 'k', 2
            ax1.plot(freqs, psd, lw=lw, color=lc)
            ax2.plot(freqs_welch, psd_welch, lw=lw, color=lc)
            ax3.plot(freqs_mt, psd_mt, lw=lw, color=lc)
            ax1.set_xlabel('Frequency (Hz)')
            if not dB:
                ax1.set_ylabel('Power spectral density (V^2/Hz)')
            else:
                ax1.set_ylabel('Decibels (dB / Hz)')
            ax1.set_title('Periodogram')
            ax2.set_title('Welch')
            ax3.set_title('Multitaper')
            if band is not None:
                ax1.set_xlim(band)
            ax1.set_ylim(ymin=0)
            ax2.set_ylim(ymin=0)
            ax3.set_ylim(ymin=0)
            sns.despine()
            if show:
                plt.show()
            if save_path is not None:
                fig_save_path = os.path.join(save_path, "Analysis figures")
                os.makedirs(fig_save_path, exist_ok=True)
                plot_path = os.path.join(fig_save_path, f"Comparison figure {channel}.png")
                plt.savefig(plot_path, dpi=300)
            plt.close()

    def plot_spectrogram(self, output, input, show=False, save_path=None):
        accumulated_signal = output[0]
        channel_list = output[1]
        time = output[4]
        frequency_limit = output[6]
        nperseg = output[7]
        noverlap = output[8]

        sns.set_theme(style="white", font_scale=1.2)
        local_signal = accumulated_signal.data
        frequency = accumulated_signal.rate
        if not channel_list:
            channel_list = list(range(local_signal.shape[1]))

        start_idx, end_idx = int(time[0] * frequency), int(time[1] * frequency)

        for channel in channel_list:
            # Compute the spectrogram
            data = local_signal[start_idx:end_idx, channel]
            frequencies, times, Sxx = signal.spectrogram(data, fs=frequency,
                                                         nperseg=nperseg, noverlap=noverlap)

            times += time[0]
            # Plot the spectrogram
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
            plt.colorbar(label='Power spectral density (dB/Hz)')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'Spectrogram for Channel {channel}')
            if frequency_limit:
                plt.ylim(frequency_limit)
            sns.despine()
            if show:
                plt.show()
            if save_path is not None:
                fig_save_path = os.path.join(save_path, "Analysis figures")
                plot_path = os.path.join(fig_save_path, f'Spectrogram_Channel_{channel}.png')
                plt.savefig(plot_path, dpi=300)
            plt.close()

    def multitaper_psd(self, x, sfreq, fmin=0.0, fmax=np.inf, bandwidth=None, adaptive=True, low_bias=True):

        n_times = x.shape[-1]
        dshape = x.shape[:-1]
        x = x.reshape(-1, n_times)

        half_nbw = bandwidth * n_times / (2.0 * sfreq) if bandwidth else 4.0
        n_tapers_max = int(2 * half_nbw)
        dpss_windows, eigvals = dpss(n_times, half_nbw, n_tapers_max, return_ratios=True)

        if low_bias:
            idx = eigvals > 0.9
            if not idx.any():
                idx = [np.argmax(eigvals)]
            dpss_windows, eigvals = dpss_windows[idx], eigvals[idx]

        freqs = rfftfreq(n_times, 1.0 / sfreq)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[freq_mask]
        n_freqs = len(freqs)

        n_tapers = len(dpss_windows)
        psd = np.zeros((x.shape[0], n_freqs))

        for i, sig in enumerate(x):
            x_mt = rfft(sig[np.newaxis, :] * dpss_windows, n=n_times)
            x_mt = x_mt[:, freq_mask]
            if adaptive and n_tapers > 1:
                psd_iter = np.mean(np.abs(x_mt[:2, :]) ** 2, axis=0)
                var = np.var(sig)
                tol = 1e-10
                for _ in range(150):
                    weights = psd_iter / (eigvals[:, np.newaxis] * psd_iter + (1 - eigvals[:, np.newaxis]) * var)
                    weights *= np.sqrt(eigvals)[:, np.newaxis]
                    psd_iter_new = np.sum(weights ** 2 * np.abs(x_mt) ** 2, axis=0) / np.sum(weights ** 2, axis=0)
                    if np.max(np.abs(psd_iter_new - psd_iter)) < tol:
                        break
                    psd_iter = psd_iter_new
                psd[i] = psd_iter
            else:
                psd[i] = np.sum((np.sqrt(eigvals)[:, np.newaxis] ** 2) * np.abs(x_mt) ** 2, axis=0) / n_tapers

        psd /= sfreq
        psd = psd.reshape(dshape + (n_freqs,))
        return psd, freqs
