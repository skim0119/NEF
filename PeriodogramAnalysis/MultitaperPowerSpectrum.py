import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import dpss


def multitaper_psd(
    x, sfreq, fmin=0.0, fmax=np.inf, bandwidth=None, adaptive=True, low_bias=True
):
    """
    Compute multitaper power spectral density (PSD) of a given signal.
    Parts of this code is inspired from MNE: https://mne.tools/stable/index.html

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
                weights = psd_iter / (
                    eigvals[:, np.newaxis] * psd_iter
                    + (1 - eigvals[:, np.newaxis]) * var
                )
                weights *= np.sqrt(eigvals)[:, np.newaxis]
                psd_iter_new = np.sum(
                    weights**2 * np.abs(x_mt) ** 2, axis=0
                ) / np.sum(weights**2, axis=0)
                if np.max(np.abs(psd_iter_new - psd_iter)) < tol:
                    break
                psd_iter = psd_iter_new
            psd[i] = psd_iter
        else:
            psd[i] = (
                np.sum(
                    (np.sqrt(eigvals)[:, np.newaxis] ** 2) * np.abs(x_mt) ** 2, axis=0
                )
                / n_tapers
            )

    psd /= sfreq
    psd = psd.reshape(dshape + (n_freqs,))
    return psd, freqs
