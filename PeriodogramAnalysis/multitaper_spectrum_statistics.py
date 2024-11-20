import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import dpss
from scipy.integrate import trapezoid


def multitaper_psd(x, sfreq, fmin=0.0, fmax=np.inf, bandwidth=None, low_bias=True):
    """
    Compute multitaper power spectral density (PSD) of a given signal.
    This code is inspired from MNE: https://mne.tools/stable/index.html

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

    psd = np.zeros((x.shape[0], n_freqs))

    n_chunk = max(x.shape[0] // 10, 1)
    offsets = np.concatenate((np.arange(0, x.shape[0], n_chunk), [x.shape[0]]))
    for start, stop in zip(offsets[:-1], offsets[1:]):
        x_mt = _mt_spectra(x[start:stop], dpss_windows, sfreq, remove_dc=True)[0]

        out = [
            _psd_from_mt_adaptive(x, eigvals, freq_mask, max_iter=150)
            for x in np.array_split(x_mt, stop - start)
        ]
        psd[start:stop] = np.concatenate(out)

    psd /= sfreq
    psd = psd.reshape(dshape + (n_freqs,))
    return psd, freqs


def _mt_spectra(x, dpss, sfreq, n_fft=None, remove_dc=True):
    """
    This code is copied from MNE: https://mne.tools/stable/index.html
    """
    if n_fft is None:
        n_fft = x.shape[-1]

    # remove mean (do not use in-place subtraction as it may modify input x)
    if remove_dc:
        x = x - np.mean(x, axis=-1, keepdims=True)

    # only keep positive frequencies
    freqs = rfftfreq(n_fft, 1.0 / sfreq)

    # The following is equivalent to this, but uses less memory:
    # x_mt = fftpack.fft(x[:, np.newaxis, :] * dpss, n=n_fft)
    n_tapers = dpss.shape[0] if dpss.ndim > 1 else 1
    x_mt = np.zeros(x.shape[:-1] + (n_tapers, len(freqs)), dtype=np.complex128)
    for idx, sig in enumerate(x):
        x_mt[idx] = rfft(sig[..., np.newaxis, :] * dpss, n=n_fft)
    # Adjust DC and maybe Nyquist, depending on one-sided transform
    x_mt[..., 0] /= np.sqrt(2.0)
    if n_fft % 2 == 0:
        x_mt[..., -1] /= np.sqrt(2.0)
    return x_mt, freqs


def _psd_from_mt_adaptive(x_mt, eigvals, freq_mask, max_iter=250, return_weights=False):
    """
    This code is copied from MNE: https://mne.tools/stable/index.html
    """
    n_signals, n_tapers, n_freqs = x_mt.shape

    if len(eigvals) != n_tapers:
        raise ValueError("Need one eigenvalue for each taper")

    if n_tapers < 3:
        raise ValueError("Not enough tapers to compute adaptive weights.")

    rt_eig = np.sqrt(eigvals)

    # estimate the variance from an estimate with fixed weights
    psd_est = _psd_from_mt(x_mt, rt_eig[np.newaxis, :, np.newaxis])
    x_var = trapezoid(psd_est, dx=np.pi / n_freqs) / (2 * np.pi)
    del psd_est

    # allocate space for output
    psd = np.empty((n_signals, np.sum(freq_mask)))

    # only keep the frequencies of interest
    x_mt = x_mt[:, :, freq_mask]

    if return_weights:
        weights = np.empty((n_signals, n_tapers, psd.shape[1]))

    for i, (xk, var) in enumerate(zip(x_mt, x_var)):
        psd_iter = _psd_from_mt(xk[:2, :], rt_eig[:2, np.newaxis])

        err = np.zeros_like(xk)
        for n in range(max_iter):
            d_k = psd_iter / (
                eigvals[:, np.newaxis] * psd_iter + (1 - eigvals[:, np.newaxis]) * var
            )
            d_k *= rt_eig[:, np.newaxis]
            err -= d_k
            if np.max(np.mean(err**2, axis=0)) < 1e-10:
                break

            # update the iterative estimate with this d_k
            psd_iter = _psd_from_mt(xk, d_k)
            err = d_k

        psd[i, :] = psd_iter

        if return_weights:
            weights[i, :, :] = d_k

    if return_weights:
        return psd, weights
    else:
        return psd


def _psd_from_mt(x_mt, weights):
    """
    This code is copied from MNE: https://mne.tools/stable/index.html
    """
    psd = weights * x_mt
    psd *= psd.conj()
    psd = psd.real.sum(axis=-2)
    psd *= 2 / (weights * weights.conj()).real.sum(axis=-2)
    return psd
