import numpy as np
from scipy import signal as sig
from collections import defaultdict
from typing import Dict, Any

from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.core.datatype import Signal
from miv.typing import SignalType

class SpectrumAnalysisWelch(OperatorMixin):
    window_length_for_welch: int = 4

    tag: str = "Welch PSD"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(
        self, signal: SignalType
    ) -> Dict[int, Dict[int, Dict[str, Any]]]:

        psd_dict: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for chunk_idx, signal_piece in enumerate(signal):
            # Compute psd_dict and power_dict for welch_periodogram plotting
            psd_dict[chunk_idx] = self.spectrum_analysis_welch(signal_piece)

        return psd_dict

    def spectrum_analysis_welch(self, signal: Signal) -> Dict[int, Dict[str, Any]]:
        """
        Compute Welch's periodogram for the signal.

        Parameters:
        -----------
        signal : SignalType
            Input signal to be analyzed.

        Returns:
        --------
        psd_welch_dict
            Dictionary containing PSD values for all channels in this chunk.
        """
        win = self.window_length_for_welch * signal.rate
        psd_welch_dict: Dict[int, Dict[str, Any]] = defaultdict(dict)

        for channel in range(signal.number_of_channels):
            signal_no_bias = signal.data[:, channel] - np.mean(signal.data[:, channel])
            freqs, psd = sig.welch(
                signal_no_bias, fs=signal.rate, nperseg=win, nfft=4 * win
            )
            psd_welch_dict[channel] = {"freqs": freqs, "psd": psd}
        return psd_welch_dict
