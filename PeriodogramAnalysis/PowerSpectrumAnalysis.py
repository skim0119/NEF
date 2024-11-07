import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from dataclasses import dataclass, field
from collections import defaultdict

from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.typing import SignalType
from MultitaperPowerSpectrum import multitaper_psd

@dataclass
class SpectrumAnalysis(OperatorMixin):
    """
    A class to perform Spectrum Analysis using multiple methods including Welch's, Periodogram, and multitaper PSD.

    Attributes:
    -----------
    exclude_channel_list : list
        Channels to be excluded from analysis.
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
    exclude_channel_list: list = field(default_factory=list)
    band_display: list = field(default_factory=lambda: [0, 100])
    window_length_for_welch: float = 4
    frequency_limit: list = field(default_factory=lambda: [0.5, 100])
    nperseg: int = 2048
    noverlap: int = 1536

    tag = "Spectrum_Analysis"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, signal: SignalType):
        """
        Perform spectrum analysis on the given signal using multiple methods.

        Parameters:
        -----------
        signal : SignalType
            Input signal to be analyzed.

        Returns:
        --------
        psd_dict:
            PSD dictionary
        spec_dict
            spectrum dictionary
        """
        psd_dict = defaultdict(dict)
        spec_dict = defaultdict(dict)
        for chunk_index, signal_piece in enumerate(signal):
            # Plot spectrum using different methods
            psd_dict[chunk_index] = self.computing_multi_spectrum(signal_piece)
            # Plot spectrogram
            spec_dict[chunk_index] = self.computing_spectrum(signal_piece)

        return psd_dict, spec_dict

    def computing_multi_spectrum(self, signal):
        win = self.window_length_for_welch * signal.rate
        psd_dict = defaultdict(dict)

        for channel in range(signal.number_of_channels):
            if channel in self.exclude_channel_list:
                continue
            freqs_per, psd_per = sig.periodogram(signal.data[:, channel], signal.rate)
            freqs_welch, psd_welch = sig.welch(signal.data[:, channel], fs=signal.rate, nperseg=win)
            psd_mt, freqs_mt = multitaper_psd(signal.data[:, channel], signal.rate)

            psd_per = 10 * np.log10(psd_per)
            psd_welch = 10 * np.log10(psd_welch)
            psd_mt = 10 * np.log10(psd_mt)

            psd_dict[channel] = {
                'freqs_per': freqs_per,
                'freqs_welch': freqs_welch,
                'freqs_mt': freqs_mt,
                'psd_per': psd_per,
                'psd_welch': psd_welch,
                'psd_mt': psd_mt
            }
        return psd_dict

    def computing_spectrum(self, signal):
        spec_dict = defaultdict(dict)
        for channel in range(signal.number_of_channels):
            if channel in self.exclude_channel_list:
                continue
            frequencies, times, Sxx = sig.spectrogram(signal.data[:, channel], fs=signal.rate, nperseg=self.nperseg, noverlap=self.noverlap)
            spec_dict[channel] = {
                'frequencies': frequencies,
                'times': times,
                'Sxx': Sxx
            }
        return spec_dict

    def plot_spectrum_methods(self, output, input, show=False, save_path=None):
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
        psd_dict = output[0]

        for chunk in psd_dict.keys():
            for channel in psd_dict[chunk].keys():

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
                sc = 'slategrey'
                ax1.stem(psd_dict[chunk][channel]['freqs_per'], psd_dict[chunk][channel]['psd_per'], linefmt=sc, basefmt=" ", markerfmt=" ")
                ax2.stem(psd_dict[chunk][channel]['freqs_welch'], psd_dict[chunk][channel]['psd_welch'], linefmt=sc, basefmt=" ", markerfmt=" ")
                ax3.stem(psd_dict[chunk][channel]['freqs_mt'], psd_dict[chunk][channel]['psd_mt'], linefmt=sc, basefmt=" ", markerfmt=" ")
                lc, lw = 'k', 2
                ax1.plot(psd_dict[chunk][channel]['freqs_per'], psd_dict[chunk][channel]['psd_per'], lw=lw, color=lc)
                ax2.plot(psd_dict[chunk][channel]['freqs_welch'], psd_dict[chunk][channel]['psd_welch'], lw=lw, color=lc)
                ax3.plot(psd_dict[chunk][channel]['freqs_mt'], psd_dict[chunk][channel]['psd_mt'], lw=lw, color=lc)
                ax1.set_xlabel('Frequency (Hz)')
                ax1.set_ylabel('Decibels (dB / Hz)')
                ax1.set_title('Periodogram')
                ax2.set_title('Welch')
                ax3.set_title('Multitaper')
                ax1.set_xlim(self.band_display)

                ax1.set_ylim(ymin=0)
                ax2.set_ylim(ymin=0)
                ax3.set_ylim(ymin=0)
                if show:
                    plt.show()
                if save_path is not None:
                    plot_path = os.path.join(save_path, f"Chunk{chunk}_Comparison_figure_channel:{channel}.png")
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
        spec_dict = output[1]

        for chunk in spec_dict.keys():
            for channel in spec_dict[chunk].keys():

                spectrogram_data = spec_dict[chunk][channel]
                frequencies = spectrogram_data['frequencies']
                times = spectrogram_data['times']
                Sxx = spectrogram_data['Sxx']

                zero_freq_indices = np.where(frequencies == 0)[0]
                zero_freq_idx = zero_freq_indices[0]
                Sxx -= Sxx[zero_freq_idx, :]

                Sxx[zero_freq_idx, :] = 1e-10
                Sxx = np.maximum(Sxx, 1e-10)
                Sxx_log = 10 * np.log10(Sxx)

                plt.figure(figsize=(10, 6))
                plt.yscale('log')
                plt.pcolormesh(times, frequencies, Sxx_log, shading='gouraud')
                plt.colorbar(label='Power spectral density (dB/Hz)')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title(f'Spectrogram for Channel {channel}')
                plt.ylim(self.frequency_limit)

                if show:
                    plt.show()
                if save_path is not None:
                    plot_path = os.path.join(save_path, f'Chunk{chunk}_Spectrogram_Channel_{channel}.png')
                    plt.savefig(plot_path, dpi=300)
                plt.close()

