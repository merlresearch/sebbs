# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from sebbs.change_detection import change_detection


def test_vs_discrete_filter():
    N = 500
    fil_len = 1.0
    y = np.random.RandomState(0).rand(N, 2)
    timestamps = np.linspace(0.0, 10.0, N + 1)
    changes = change_detection(y, timestamps, fil_len)

    step_filter = np.ones(N // 10)
    step_filter[: N // 20] = -1
    for i, (segment_bounds, *_) in enumerate(changes):
        onsets = segment_bounds[1:-2:2]
        offsets = segment_bounds[2:-1:2]
        deltas_ref = np.correlate(np.concatenate((np.zeros(N // 20), y[:, i], np.zeros(N // 20))), step_filter)
        deltas_ref = np.concatenate(([-np.inf], deltas_ref, [-np.inf]))
        onsets_ref = timestamps[
            np.argwhere((deltas_ref[1:-2] > deltas_ref[:-3]) * (deltas_ref[1:-2] > deltas_ref[2:-1])).flatten()
        ]
        offsets_ref = timestamps[
            np.argwhere((deltas_ref[2:-1] < deltas_ref[1:-2]) * (deltas_ref[2:-1] < deltas_ref[3:])).flatten() + 1
        ]
        if len(onsets_ref) > len(offsets_ref):
            offsets_ref = np.concatenate((offsets_ref, [timestamps[-1]]))
        assert (np.abs(onsets - onsets_ref) < 1e-6).all()
        assert (np.abs(offsets - offsets_ref) < 1e-6).all()
