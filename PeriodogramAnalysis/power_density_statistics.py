from dataclasses import dataclass

import scipy
import numpy as np

from miv.core.operator_generator.operator import GeneratorOperatorMixin
from miv.core.operator_generator.wrapper import cache_generator_call
from miv.core.datatype import Signal


@dataclass
class SpectrumAnalysisBase(GeneratorOperatorMixin):
    """
    A base class for performing spectral analysis on signal data.

    Attributes
    ----------
    window_size_for_welch : int
        The length of the window for Welch's method, defined as a multiple of the signal's sampling rate. Default is 4.
    band_display : Tuple[float, float]
        The frequency band range to display on the plot. Default is (0, 100) Hz.
    tag : str
        A string representing the tag used as the title of the plot. Default is "Base PSD Analysis".
    """

    window_size_for_welch: int = 4
    tag: str = "Base PSD spectrum analysis"

    def __post_init__(self) -> None:
        super().__init__()
        if (
            not isinstance(self.window_size_for_welch, int)
            or self.window_size_for_welch <= 0
        ):
            raise ValueError("window_size_for_welch must be a positive integer.")
        self.chunk: int = 0

    @cache_generator_call
    def __call__(self, signal: Signal) -> tuple:
        """
        Compute the Power Spectral Density (PSD) for each chunk of the given signal.

        Parameters
        ----------
        signal : SignalType
            The input signal to be analyzed.

        Returns
        -------
        dict[int, dict[str, Any]]
            A dictionary where the keys are channel indices, and each value is another dictionary containing
            the PSD data for the specified channels. The inner dictionary includes keys such as:
            - "freqs": A NumPy array of frequency values in Hz.
            - "psd": A NumPy array of PSD values.
        """
        if signal.data.shape[0] <= self.window_size_for_welch:
            raise ValueError("window_size_for_welch must be larger than data size")

        self.rate = signal.rate
        self.num_channel = signal.number_of_channels

        freqs, psd = self.compute_psd(signal.data)

        return freqs, psd

    def compute_psd(self, data: np.ndarray) -> tuple:
        """
        compute_psd(signal: Signal) -> Dict[int, Dict[str, Any]]:
        Abstract method to be overridden in subclasses to compute the PSD for a given signal.
        """
        raise NotImplementedError(
            "The compute_psd method is not implemented in the base class. "
            "This base class is not intended for standalone use. Please use a subclass "
            "such as SpectrumAnalysisWelch, SpectrumAnalysisPeriodogram, or SpectrumAnalysisMultitaper."
        )


@dataclass
class SpectrumAnalysisWelch(SpectrumAnalysisBase):
    """
    A class that performs spectral analysis using the Welch method.
    """

    tag: str = "Welch PSD spectrum analysis"

    def compute_psd(self, data: np.ndarray) -> tuple:
        win = self.window_size_for_welch * self.rate
        psd_list: list = []
        for ch in range(self.num_channel):
            signal_no_bias = data[:, ch] - np.mean(data[:, ch])
            freqs, psd_channel = scipy.signal.welch(
                signal_no_bias, fs=self.rate, nperseg=win, nfft=4 * win
            )
            psd_list.append(psd_channel)

        return freqs, np.array(psd_list).T


@dataclass
class SpectrumAnalysisPeriodogram(SpectrumAnalysisBase):
    """
    A class that performs spectral analysis using the Periodogram method.
    """

    tag: str = "Periodogram PSD spectrum analysis"

    def compute_psd(self, data: np.ndarray) -> tuple:
        psd_list: list = []

        for ch in range(self.num_channel):
            signal_no_bias = data[:, ch] - np.mean(data[:, ch])
            freqs, psd_channel = scipy.signal.periodogram(signal_no_bias, self.rate)
            psd_list.append(psd_channel)

        return freqs, np.array(psd_list).T
