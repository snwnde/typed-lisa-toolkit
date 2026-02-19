"""Shared helper builders for representation backend tests."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import numpy as np

from typed_lisa_toolkit.containers.representations import FrequencySeries, STFT, TimeSeries


def _randn_array(xp, shape):
    return xp.asarray(np.random.randn(*shape))


def build_canonical_representations(
    xp,
    *,
    n_batches,
    n_channels,
    n_harmonics,
    n_features,
    len_time,
    len_freq,
    tf_grid_order,
):
    freqs = xp.linspace(0, 1, len_freq)
    times = xp.linspace(0, 10, len_time)

    entries_fs = _randn_array(
        xp,
        (n_batches, n_channels, n_harmonics, n_features, len_freq),
    )
    fs = FrequencySeries(grid=(freqs,), entries=entries_fs)

    entries_ts = _randn_array(
        xp,
        (n_batches, n_channels, n_harmonics, n_features, len_time),
    )
    ts = TimeSeries(grid=(times,), entries=entries_ts)

    if tf_grid_order == "freq_time":
        entries_tf = _randn_array(
            xp,
            (n_batches, n_channels, n_harmonics, n_features, len_freq, len_time),
        )
        tf = STFT(grid=(freqs, times), entries=entries_tf)
    elif tf_grid_order == "time_freq":
        entries_tf = _randn_array(
            xp,
            (n_batches, n_channels, n_harmonics, n_features, len_time, len_freq),
        )
        tf = STFT(grid=(times, freqs), entries=entries_tf)
    else:
        raise ValueError(f"Unknown tf_grid_order: {tf_grid_order}")

    return {
        "freqs": freqs,
        "times": times,
        "entries_fs": entries_fs,
        "entries_ts": entries_ts,
        "entries_tf": entries_tf,
        "fs": fs,
        "ts": ts,
        "tf": tf,
    }
