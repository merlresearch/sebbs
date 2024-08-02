# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from sed_scores_eval.base_modules.scores import create_score_dataframe
from sed_scores_eval.utils.array_ops import get_first_index_where


def sed_scores_from_sebbs(sebbs, sound_classes, audio_duration=None, fill_value=-np.inf):
    """Save Sebbs in sed_scores_eval format with constant class score over the extent of a SEBB and -inf else.

    Args:
        sebbs (dict | list): (dict of) list of SEBBs (onset, offset, class, confidence) as provided by predict method.
        sound_classes: list of all sound classes.
        audio_duration: optional (dict of) audio duration(s).
        fill_value

    Returns:

    """
    if isinstance(sebbs, dict):
        return {
            key: sed_scores_from_sebbs(
                preds_for_key,
                sound_classes,
                audio_duration[key] if audio_duration is not None else None,
                fill_value=fill_value,
            )
            for key, preds_for_key in sebbs.items()
        }
    if len(sebbs) == 0:
        return create_score_dataframe(
            np.full((1, len(sound_classes)), fill_value),
            [0.0, 0.001 if audio_duration is None else audio_duration],
            sound_classes,
        )
    change_points = set()
    for bb in sebbs:
        change_points.update([bb[0], bb[1]])
    change_points = sorted(change_points)
    if len(change_points) == 0 or change_points[0] != 0.0:
        change_points = [0.0] + change_points
    if audio_duration is not None and change_points[-1] != audio_duration:
        change_points = change_points + [audio_duration]
    change_points = np.unique(change_points)
    score_arr = np.full((len(change_points) - 1, len(sound_classes)), fill_value)
    for cand_onset, cand_offset, class_name, mean_score in sebbs:
        cand_onset_idx = get_first_index_where(change_points, "geq", cand_onset)
        cand_offset_idx = get_first_index_where(change_points, "geq", cand_offset)
        class_idx = sound_classes.index(class_name)
        score_arr[cand_onset_idx:cand_offset_idx, class_idx] = mean_score
    return create_score_dataframe(score_arr, change_points, sound_classes)


def sed_scores_from_detections(detections, sound_classes, audio_duration=None, fill_value=-np.inf):
    return sed_scores_from_sebbs(
        {audio_id: [(*event, 1.0) for event in events] for audio_id, events in detections.items()},
        sound_classes=sound_classes,
        audio_duration=audio_duration,
        fill_value=fill_value,
    )
