# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# distutils: language = c++
#cython: language_level=3

import numpy as np

cimport cython
cimport numpy as np
from libcpp.vector cimport vector
from numpy.math cimport INFINITY


def change_detection(sed_scores, timestamps_in_sec, filter_length_in_sec, time_decimals=6):
    """change point detection in the piecewise constant SED posterior scores by
    employing a time continuous step filter.

    Args:
        sed_scores (2d np.ndarray): Predicted SED posterior scores of shape NxC,
            with N being the number of frames and C the number of sound classes.
        timestamps_in_sec (1d np.ndarray): The N+1 frame boundary times in sec
        filter_length_in_sec: filter length in sec T_fil. The step filter equals
            -2/T_fil \in [0,T_fil/2] (first half of the filter) and
            +2/T_fil \in [T_fil/2,T_fil] (second half of the filter). Hence, at
            each time point the filter predicts change/delta values as the
            difference between the average score in the next T_fil/2 seconds and
            the previous T_fil/2 seconds.
        time_decimals: the decimal precision to be used for timestamps.

    Returns (list of tuple): for each sound class:
        seg_bounds (1d np.ndarray): segment boundaries/change points.
            seg_bounds[0] equals audio onset 0., seg_bounds[1:-2:2] are event
            onset candidates, seg_bounds[2:-1:2] are event offset candidates,
            and seg_bounds[-1] equals audio length.
        mean_scores (1d np.ndarray): the average score within each segment.
        min_scores (1d np.ndarray): the minimum score within each segment.
        max_scores (1d np.ndarray): the maximum score within each segment.

    """
    assert filter_length_in_sec is not None, "You must provide filter_length_in_sec"

    if np.isscalar(filter_length_in_sec):
        filter_length_in_sec = np.array(sed_scores.shape[1] * [filter_length_in_sec], dtype=np.float64)

    sed_scores = np.asanyarray(sed_scores, dtype=np.float64)
    assert sed_scores.ndim == 2, sed_scores.shape
    timestamps_in_sec = np.asanyarray(timestamps_in_sec, dtype=np.float64)
    assert timestamps_in_sec.ndim == 1, timestamps_in_sec.shape
    assert (timestamps_in_sec[1:] > timestamps_in_sec[:-1]).all(), np.min(timestamps_in_sec[1:] - timestamps_in_sec[:-1])
    filter_length_in_sec = np.asanyarray(filter_length_in_sec, dtype=np.float64)
    cdef int num_timesteps = timestamps_in_sec.shape[0] + 2
    # breakpoint times, where the filter center, onset or offset moves into a new frame:
    cdef double [:,:] overlap_breakpoint_timestamps = np.array([
        np.sort(
            np.round(np.concatenate((
                timestamps_in_sec,
                timestamps_in_sec + fil_len / 2,
                timestamps_in_sec - fil_len / 2,
            )), decimals=time_decimals+2)
        )
        for fil_len in filter_length_in_sec
    ])
    # filter output is piecewise linear between these breakpoints and local min/max do lie on breakpoints.

    cdef double [:] filter_length = filter_length_in_sec

    cdef double [:,:] scores = np.concatenate((
        np.full_like(sed_scores[:1], 0),
        sed_scores,
        np.full_like(sed_scores[:1], 0),
    ))
    cdef double [:] timestamps = np.concatenate(([-np.inf], timestamps_in_sec, [np.inf]))
    cdef int num_classes = sed_scores.shape[1]
    cdef double eps = 10**(-time_decimals-1)

    cdef:
        int i, k, first_time_seg_idx, center_time_seg_idx, last_time_seg_idx, num_segments
        double t, t_diff, fil_len_k, center_time, left_part_sum, right_part_sum

    cdef vector[double] onsets = []
    cdef vector[double] offsets = []
    cdef vector[double] cum_scores = []
    cdef vector[double] min_scores = []
    cdef vector[double] max_scores = []
    cdef vector[double] deltas = []
    cdef vector[double] deltas_rel = []
    cdef vector[int] class_pos = [0]
    cdef double [:] scores_k
    cdef double [:] overlap_breakpoint_timestamps_k
    cdef int num_onsets = 0
    cdef int num_offsets = 0
    cdef int num_deltas = 0

    for k in range(num_classes):
        scores_k = scores[:, k]
        fil_len_k = filter_length[k]
        overlap_breakpoint_timestamps_k = overlap_breakpoint_timestamps[k]
        first_time_seg_idx = 0
        center_time_seg_idx = 1
        center_time = 0.
        left_part_sum = 0.
        last_time_seg_idx = 1
        right_part_sum = 0.
        while timestamps[last_time_seg_idx+1] <= fil_len_k / 2:
            right_part_sum += scores_k[last_time_seg_idx] * (
                timestamps[last_time_seg_idx+1] - timestamps[last_time_seg_idx]
            )
            last_time_seg_idx += 1
        right_part_sum += scores_k[last_time_seg_idx] * (
            fil_len_k / 2 - timestamps[last_time_seg_idx]
        )
        deltas.clear()
        deltas_rel.clear()
        deltas.push_back(2*(right_part_sum - left_part_sum)/fil_len_k)
        deltas_rel.push_back((right_part_sum+1e-6)/(left_part_sum+1e-6))
        num_deltas = 1
        cum_scores.push_back(0.)
        min_scores.push_back(INFINITY)
        max_scores.push_back(0.)
        for i in range(len(overlap_breakpoint_timestamps_k)):
            t = overlap_breakpoint_timestamps_k[i]  # current breakpoint
            # get time difference to previous breakpoint, i.e., the time the filter
            # moved since last breakpoint
            t_diff = t - overlap_breakpoint_timestamps_k[i-1]
            if (t < eps) or (t_diff < eps) or (t > timestamps[num_timesteps-2]):
                continue
            left_part_sum -= t_diff * scores_k[first_time_seg_idx]
            left_part_sum += t_diff * scores_k[center_time_seg_idx]
            right_part_sum -= t_diff * scores_k[center_time_seg_idx]
            right_part_sum += t_diff * scores_k[last_time_seg_idx]
            center_time += t_diff
            deltas.push_back(2*(right_part_sum - left_part_sum)/fil_len_k)
            deltas_rel.push_back((right_part_sum+1e-3)/(left_part_sum+1e-3))
            num_deltas += 1
            if (num_offsets == num_onsets) and (deltas[num_deltas-1] <= deltas[num_deltas-2]) and ((num_deltas == 2) or (deltas[num_deltas-2] > deltas[num_deltas-3])):
                # previous delta was a local maximum and therefore a new onset candidate
                onsets.push_back(overlap_breakpoint_timestamps_k[i-1])
                num_onsets += 1
                cum_scores.push_back(0.)
                min_scores.push_back(INFINITY)
                max_scores.push_back(0.)
            elif (num_onsets > num_offsets) and (deltas[num_deltas-1] > deltas[num_deltas-2]) and (deltas[num_deltas-2] <= deltas[num_deltas-3]):
                # previous delta was a local minimum and therefore a new offset candidate
                offsets.push_back(overlap_breakpoint_timestamps_k[i-1])
                num_offsets += 1
                cum_scores.push_back(0.)
                min_scores.push_back(INFINITY)
                max_scores.push_back(0.)
            num_segments = 1 + num_onsets + num_offsets + k
            cum_scores[num_segments-1] += t_diff * scores_k[center_time_seg_idx]
            min_scores[num_segments-1] = min(min_scores[num_segments-1], scores_k[center_time_seg_idx])
            max_scores[num_segments-1] = max(max_scores[num_segments-1], scores_k[center_time_seg_idx])
            if (timestamps[center_time_seg_idx+1] - center_time) < eps:
                center_time_seg_idx += 1
            if (timestamps[first_time_seg_idx+1] - (center_time - fil_len_k / 2)) < eps:
                first_time_seg_idx += 1
            if (timestamps[last_time_seg_idx+1] - (center_time + fil_len_k / 2)) < eps:
                last_time_seg_idx += 1
        if num_onsets > num_offsets:
            offsets.push_back(timestamps[num_timesteps-2])
            num_offsets += 1
            cum_scores.push_back(0.)
            min_scores.push_back(0.)
            max_scores.push_back(0.)
        class_pos.push_back(num_onsets)

    assert num_onsets == num_offsets, (num_onsets, num_offsets, onsets, offsets)
    onsets_arr = np.array(onsets)
    offsets_arr = np.array(offsets)
    class_pos_arr = np.array(class_pos)
    cum_scores_arr = np.array(cum_scores)
    min_scores_arr = np.array(min_scores)
    max_scores_arr = np.array(max_scores)
    out = []
    for c in range(int(num_classes)):
        onsets_c = onsets_arr[class_pos_arr[c]:class_pos_arr[c+1]]
        offsets_c = offsets_arr[class_pos_arr[c]:class_pos_arr[c+1]]
        seg_bounds_c = np.concatenate(([0], np.stack((onsets_c, offsets_c), axis=1).flatten(), [timestamps_in_sec[-1]]))
        mean_scores_c = cum_scores_arr[2*class_pos_arr[c]+c:2*class_pos_arr[c+1]+c+1]
        seg_lens_c = np.maximum(seg_bounds_c[1:] - seg_bounds_c[:-1], eps)
        mean_scores_c /= seg_lens_c
        min_scores_c = min_scores_arr[2*class_pos_arr[c]+c:2*class_pos_arr[c+1]+c+1]
        min_scores_c[min_scores_c == np.inf] = 0.
        min_scores_c[0] = min_scores_c[-1] = 0.
        max_scores_c = max_scores_arr[2*class_pos_arr[c]+c:2*class_pos_arr[c+1]+c+1]
        out.append((seg_bounds_c, mean_scores_c, min_scores_c, max_scores_c))
    return out
