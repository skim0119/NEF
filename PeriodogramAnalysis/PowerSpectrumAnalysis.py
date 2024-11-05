import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import dpss
from dataclasses import dataclass, field
from typing import Optional

from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.typing import SignalType

@dataclass
class SpectrumAnalysis(OperatorMixin):
    """
    A class to perform Spectrum Analysis using multiple methods including Welch's, Periodogram, and multitaper PSD.

    Attributes:
    -----------
    exclude_channels : Optional[list]
        List of channels to be excluded from analysis.
    band_display : list
        Frequency band range to display in plots.
    win_sec : float
        Window length in seconds for Welch's method.
    frequency_limit : list
        Frequency range limit for analysis.
    nperseg : int
        Number of points per segment for spectrogram computation.
    noverlap : int
        Number of points to overlap between segments for spectrogram.
    convert_db : bool
        If True, convert power spectral density values to decibels (dB).
    """
    exclude_channels: Optional[list] = field(default_factory=list)
    band_display: list = field(default_factory=lambda: [0, 100])
    win_sec: float = 4
    frequency_limit: list = field(default_factory=lambda: [0, 100])
    nperseg: int = 2048
    noverlap: int = 1024
    convert_db: bool = True

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
        tuple
            PSD dictionary, excluded channels, number of chunks, band display range, dB conversion flag, spectrogram dictionary, frequency limit.
        """
        for idx, signal_piece in enumerate(signal):
            if idx >= 1:
                break
            self.num_channels = signal_piece.number_of_channels
            num_of_chunks = idx + 1

            # Plot spectrum using different methods
            psd_dict = self.computing_multi_spectrum(signal_piece, self.exclude_channels, num_of_chunks, self.win_sec, self.convert_db)
            # Plot spectrogram
            spec_dict = self.computing_spectrum(signal_piece, self.exclude_channels, num_of_chunks, self.nperseg, self.noverlap)

        return psd_dict, self.exclude_channels, num_of_chunks, self.band_display, self.convert_db, spec_dict, self.frequency_limit

    def computing_multi_spectrum(self, signal, exclude_channel_list, num_of_chunks, win_sec, dB):
        win = win_sec * signal.rate
        psd_dict = {}

        for chunk in range(num_of_chunks):
            if chunk not in psd_dict:
                psd_dict[chunk] = {}
            for channel in range(self.num_channels):
                if channel not in exclude_channel_list:
                    freqs_per, psd_per = sig.periodogram(signal.data[:, channel], signal.rate)
                    freqs_welch, psd_welch = sig.welch(signal.data[:, channel], fs=signal.rate, nperseg=win)
                    psd_mt, freqs_mt = multitaper_psd(signal.data[:, channel], signal.rate)
                    sharey = False

                    if dB:
                        psd_per = 10 * np.log10(psd_per)
                        psd_welch = 10 * np.log10(psd_welch)
                        psd_mt = 10 * np.log10(psd_mt)
                        sharey = True

                    psd_dict[chunk][channel] = {
                        'freqs_per': freqs_per,
                        'freqs_welch': freqs_welch,
                        'freqs_mt': freqs_mt,
                        'psd_per': psd_per,
                        'psd_welch': psd_welch,
                        'psd_mt': psd_mt,
                        'sharey': sharey
                    }
        return psd_dict

    def computing_spectrum(self, signal, exclude_channel_list, num_of_chunks, nperseg, noverlap):
        spec_dict = {}
        for chunk in range(num_of_chunks):
            if chunk not in spec_dict:
                spec_dict[chunk] = {}
            for channel in range(self.num_channels):
                if channel not in exclude_channel_list:
                    frequencies, times, Sxx = sig.spectrogram(signal.data[:, channel], fs=signal.rate, nperseg=nperseg, noverlap=noverlap)
                    spec_dict[chunk][channel] = {
                        'frequencies': frequencies,
                        'times': times,
                        'Sxx': Sxx
                    }
        return spec_dict

    def plot_spectrum_methods(self, output, input, show=False, save_path=None):
        psd_dict = output[0]
        exclude_channel_list = output[1]
        num_of_chunks = output[2]
        band = output[3]
        dB = output[4]

        for chunk in range(num_of_chunks):
            for channel in range(self.num_channels):
                if channel not in exclude_channel_list:

                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=psd_dict[chunk][channel]['sharey'])
                    sc = 'slategrey'
                    ax1.stem(psd_dict[chunk][channel]['freqs_per'], psd_dict[chunk][channel]['psd_per'], linefmt=sc, basefmt=" ", markerfmt=" ")
                    ax2.stem(psd_dict[chunk][channel]['freqs_welch'], psd_dict[chunk][channel]['psd_welch'], linefmt=sc, basefmt=" ", markerfmt=" ")
                    ax3.stem(psd_dict[chunk][channel]['freqs_mt'], psd_dict[chunk][channel]['psd_mt'], linefmt=sc, basefmt=" ", markerfmt=" ")
                    lc, lw = 'k', 2
                    ax1.plot(psd_dict[chunk][channel]['freqs_per'], psd_dict[chunk][channel]['psd_per'], lw=lw, color=lc)
                    ax2.plot(psd_dict[chunk][channel]['freqs_welch'], psd_dict[chunk][channel]['psd_welch'], lw=lw, color=lc)
                    ax3.plot(psd_dict[chunk][channel]['freqs_mt'], psd_dict[chunk][channel]['psd_mt'], lw=lw, color=lc)
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
                    if show:
                        plt.show()
                    if save_path is not None:
                        fig_save_path = os.path.join(save_path, "Analysis_figures")
                        os.makedirs(fig_save_path, exist_ok=True)
                        plot_path = os.path.join(fig_save_path, f"Comparison_figure_{channel}.png")
                        plt.savefig(plot_path, dpi=300)
                    plt.close()

    def plot_spectrogram(self, output, input, show=False, save_path=None):
        exclude_channel_list = output[1]
        num_of_chunks = output[2]
        spec_dict = output[5]
        frequency_limit = output[6]
        for chunk in range(num_of_chunks):
            for channel in range(self.num_channels):
                if channel not in exclude_channel_list:
                    spectrogram_data = spec_dict[chunk][channel]
                    frequencies = spectrogram_data['frequencies']
                    times = spectrogram_data['times']
                    Sxx = spectrogram_data['Sxx']

                    plt.figure(figsize=(10, 6))
                    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
                    plt.colorbar(label='Power spectral density (dB/Hz)')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Frequency (Hz)')
                    plt.title(f'Spectrogram for Channel {channel}')
                    if frequency_limit:
                        plt.ylim(frequency_limit)
                    if show:
                        plt.show()
                    if save_path is not None:
                        fig_save_path = os.path.join(save_path, "Analysis_figures")
                        os.makedirs(fig_save_path, exist_ok=True)
                        plot_path = os.path.join(fig_save_path, f'Spectrogram_Channel_{channel}.png')
                        plt.savefig(plot_path, dpi=300)
                    plt.close()

def multitaper_psd(x, sfreq, fmin=0.0, fmax=np.inf, bandwidth=None, adaptive=True, low_bias=True):
    """
    Compute multitaper power spectral density (PSD) of a given signal.

    Parameters:
    -----------
    x : array-like
        Input signal to compute PSD.
    sfreq : float
        Sampling frequency of the input signal.
    fmin : float
        Minimum frequency to consider in the PSD.
    fmax : float
        Maximum frequency to consider in the PSD.
    bandwidth : float, optional
        Bandwidth of the DPSS tapers. Default is None.
    adaptive : bool
        If True, use adaptive weighting to combine taper PSDs.
    low_bias : bool
        If True, only use tapers with eigenvalues greater than 0.9.

    Returns:
    --------
    tuple
        PSD values and corresponding frequency values.
    """
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
