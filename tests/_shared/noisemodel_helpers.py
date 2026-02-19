"""Shared test helpers for noise model backend tests."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import numpy as np

from typed_lisa_toolkit.containers.data import FSData, WDMData
from typed_lisa_toolkit.containers.representations import FrequencySeries, WDM


def build_fd_pair(xp):
    frequencies = xp.asarray([1.0, 2.0, 4.0], dtype=xp.float64)

    left_x = xp.asarray([1.0 + 1.0j, 2.0 - 1.0j, -1.0 + 0.5j], dtype=xp.complex128)
    left_y = xp.asarray([0.5 - 0.25j, -1.0j, 2.0 + 0.0j], dtype=xp.complex128)
    right_x = xp.asarray([2.0 - 1.0j, -1.0 + 2.0j, 0.5 + 0.25j], dtype=xp.complex128)
    right_y = xp.asarray([1.0 + 0.0j, 0.25 + 1.0j, -0.5 + 2.0j], dtype=xp.complex128)

    left = FSData.from_dict(
        {
            "X": FrequencySeries((frequencies,), left_x[None, None, None, None, :]),
            "Y": FrequencySeries((frequencies,), left_y[None, None, None, None, :]),
        }
    )
    right = FSData.from_dict(
        {
            "X": FrequencySeries((frequencies,), right_x[None, None, None, None, :]),
            "Y": FrequencySeries((frequencies,), right_y[None, None, None, None, :]),
        }
    )

    return {
        "frequencies": frequencies,
        "left": left,
        "right": right,
        "left_x": left_x,
        "left_y": left_y,
        "right_x": right_x,
        "right_y": right_y,
    }


def build_wdm_pair(xp):
    times = xp.asarray([0.0, 1.0, 2.0], dtype=xp.float64)
    frequencies = xp.asarray([0.25, 0.5], dtype=xp.float64)

    left_x = xp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=xp.float64)
    left_y = xp.asarray([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=xp.float64)
    right_x = xp.asarray([[2.0, 0.0, 1.0], [1.0, 0.0, 2.0]], dtype=xp.float64)
    right_y = xp.asarray([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=xp.float64)

    left = WDMData.from_dict(
        {
            "X": WDM.make(
                times=times,
                frequencies=frequencies,
                entries=left_x[None, None, None, None, :, :],
            ),
            "Y": WDM.make(
                times=times,
                frequencies=frequencies,
                entries=left_y[None, None, None, None, :, :],
            ),
        }
    )
    right = WDMData.from_dict(
        {
            "X": WDM.make(
                times=times,
                frequencies=frequencies,
                entries=right_x[None, None, None, None, :, :],
            ),
            "Y": WDM.make(
                times=times,
                frequencies=frequencies,
                entries=right_y[None, None, None, None, :, :],
            ),
        }
    )

    return {
        "times": times,
        "frequencies": frequencies,
        "left": left,
        "right": right,
        "left_x": left_x,
        "left_y": left_y,
        "right_x": right_x,
        "right_y": right_y,
    }


def build_fd_pair_batched_2x2(xp):
    base = build_fd_pair(xp)
    frequencies = base["frequencies"]

    left_x = xp.stack(
        [
            base["left_x"],
            0.75 * base["left_x"] + (0.1 - 0.2j),
        ],
        axis=0,
    )
    left_y = xp.stack(
        [
            base["left_y"],
            1.25 * base["left_y"] + (-0.05 + 0.1j),
        ],
        axis=0,
    )
    right_x = xp.stack(
        [
            base["right_x"],
            -0.5 * base["right_x"] + (0.2 + 0.05j),
        ],
        axis=0,
    )
    right_y = xp.stack(
        [
            base["right_y"],
            0.6 * base["right_y"] + (-0.1 + 0.15j),
        ],
        axis=0,
    )

    left = FSData.from_dict(
        {
            "X": FrequencySeries((frequencies,), left_x[:, None, None, None, :]),
            "Y": FrequencySeries((frequencies,), left_y[:, None, None, None, :]),
        }
    )
    right = FSData.from_dict(
        {
            "X": FrequencySeries((frequencies,), right_x[:, None, None, None, :]),
            "Y": FrequencySeries((frequencies,), right_y[:, None, None, None, :]),
        }
    )

    return {
        "frequencies": frequencies,
        "left": left,
        "right": right,
        "left_x": left_x,
        "left_y": left_y,
        "right_x": right_x,
        "right_y": right_y,
    }


def build_wdm_pair_batched_2x2(xp):
    base = build_wdm_pair(xp)
    times = base["times"]
    frequencies = base["frequencies"]

    left_x = xp.stack(
        [
            base["left_x"],
            0.8 * base["left_x"] + 0.3,
        ],
        axis=0,
    )
    left_y = xp.stack(
        [
            base["left_y"],
            1.1 * base["left_y"] - 0.2,
        ],
        axis=0,
    )
    right_x = xp.stack(
        [
            base["right_x"],
            -0.4 * base["right_x"] + 0.5,
        ],
        axis=0,
    )
    right_y = xp.stack(
        [
            base["right_y"],
            0.7 * base["right_y"] + 0.25,
        ],
        axis=0,
    )

    left = WDMData.from_dict(
        {
            "X": WDM.make(
                times=times,
                frequencies=frequencies,
                entries=left_x[:, None, None, None, :, :],
            ),
            "Y": WDM.make(
                times=times,
                frequencies=frequencies,
                entries=left_y[:, None, None, None, :, :],
            ),
        }
    )
    right = WDMData.from_dict(
        {
            "X": WDM.make(
                times=times,
                frequencies=frequencies,
                entries=right_x[:, None, None, None, :, :],
            ),
            "Y": WDM.make(
                times=times,
                frequencies=frequencies,
                entries=right_y[:, None, None, None, :, :],
            ),
        }
    )

    return {
        "times": times,
        "frequencies": frequencies,
        "left": left,
        "right": right,
        "left_x": left_x,
        "left_y": left_y,
        "right_x": right_x,
        "right_y": right_y,
    }


def diagonal_kernel_2ch(xp):
    kernel = xp.asarray(np.zeros((3, 2, 2), dtype=np.float64))
    values_x = xp.asarray([2.0, 4.0, 8.0], dtype=xp.float64)
    values_y = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64)
    if hasattr(kernel, "at"):
        kernel = kernel.at[:, 0, 0].set(values_x)
        kernel = kernel.at[:, 1, 1].set(values_y)
        return kernel
    kernel[:, 0, 0] = values_x
    kernel[:, 1, 1] = values_y
    return kernel


def dense_kernel_2ch(xp):
    return xp.asarray(
        [
            [[2.0, 0.1], [0.1, 1.0]],
            [[4.0, 0.2], [0.2, 0.5]],
            [[8.0, -0.3], [-0.3, 0.25]],
        ],
        dtype=xp.float64,
    )


def dense_esdm_2ch(xp):
    return xp.asarray(
        [
            [
                [[2.0, 0.2], [0.2, 1.1]],
                [[1.8, -0.1], [-0.1, 1.3]],
                [[2.2, 0.05], [0.05, 0.9]],
            ],
            [
                [[1.5, 0.15], [0.15, 1.4]],
                [[1.7, -0.08], [-0.08, 1.0]],
                [[1.9, 0.12], [0.12, 1.2]],
            ],
        ],
        dtype=xp.float64,
    )
