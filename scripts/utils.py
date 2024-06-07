# Copyright (C) 2024 Romain Serizel
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict, defaultdict
from math import ceil

import numpy as np
from sed_scores_eval.base_modules.scores import create_score_dataframe, validate_score_dataframe

classes_labels_desed = OrderedDict(
    {
        "Alarm_bell_ringing": 0,
        "Blender": 1,
        "Cat": 2,
        "Dishes": 3,
        "Dog": 4,
        "Electric_shaver_toothbrush": 5,
        "Frying": 6,
        "Running_water": 7,
        "Speech": 8,
        "Vacuum_cleaner": 9,
    }
)


classes_labels_maestro_real = OrderedDict(
    {
        "cutlery and dishes": 0,
        "furniture dragging": 1,
        "people talking": 2,
        "children voices": 3,
        "coffee machine": 4,
        "footsteps": 5,
        "large_vehicle": 6,
        "car": 7,
        "brakes_squeaking": 8,
        "cash register beeping": 9,
        "announcement": 10,
        "shopping cart": 11,
        "metro leaving": 12,
        "metro approaching": 13,
        "door opens/closes": 14,
        "wind_blowing": 15,
        "birds_singing": 16,
    }
)

classes_labels_maestro_synth = OrderedDict(
    {
        "car_horn": 0,
        "children_voices": 1,
        "engine_idling": 2,
        "siren": 3,
        "street_music": 4,
        "dog_bark": 5,
    }
)

classes_labels_maestro_real_eval = {
    "birds_singing",
    "car",
    "people talking",
    "footsteps",
    "children voices",
    "wind_blowing",
    "brakes_squeaking",
    "large_vehicle",
    "cutlery and dishes",
    "metro approaching",
    "metro leaving",
}


def merge_overlapping_events(ground_truth_events):
    for clip_id, events in ground_truth_events.items():
        per_class_events = defaultdict(list)
        for event in events:
            per_class_events[event[2]].append(event)
        ground_truth_events[clip_id] = []
        for event_class, events in per_class_events.items():
            events = sorted(events)
            merged_events = []
            current_offset = -1e6
            for event in events:
                if event[0] > current_offset:
                    merged_events.append(list(event))
                else:
                    merged_events[-1][1] = max(current_offset, event[1])
                current_offset = merged_events[-1][1]
            ground_truth_events[clip_id].extend(merged_events)
    return ground_truth_events


def get_segment_scores_and_overlap_add(frame_scores, audio_durations, event_classes, segment_length=1.0):
    segment_scores_file = {}
    summand_count = {}
    keys = ["onset", "offset"] + event_classes
    for clip_id in frame_scores:
        file_id, clip_onset_time, clip_offset_time = clip_id.rsplit("-", maxsplit=2)
        clip_onset_time = float(clip_onset_time) / 100
        clip_offset_time = float(clip_offset_time) / 100
        if file_id not in segment_scores_file:
            segment_scores_file[file_id] = np.zeros(
                (ceil(audio_durations[file_id] / segment_length), len(event_classes))
            )
            summand_count[file_id] = np.zeros_like(segment_scores_file[file_id])
        segment_scores_clip = get_segment_scores(
            frame_scores[clip_id][keys],
            clip_length=(clip_offset_time - clip_onset_time),
            segment_length=1.0,
        )[event_classes].to_numpy()
        seg_idx = int(clip_onset_time // segment_length)
        segment_scores_file[file_id][seg_idx : seg_idx + len(segment_scores_clip)] += segment_scores_clip
        summand_count[file_id][seg_idx : seg_idx + len(segment_scores_clip)] += 1
    return {
        file_id: create_score_dataframe(
            segment_scores_file[file_id] / np.maximum(summand_count[file_id], 1),
            np.minimum(
                np.arange(0.0, audio_durations[file_id] + segment_length, segment_length),
                audio_durations[file_id],
            ),
            event_classes,
        )
        for file_id in segment_scores_file
    }


def get_segment_scores(scores_df, clip_length, segment_length=1.0):
    frame_timestamps, event_classes = validate_score_dataframe(scores_df)
    scores_arr = scores_df[event_classes].to_numpy()
    segment_scores = []
    segment_timestamps = []
    seg_onset_idx = 0
    seg_offset_idx = 0
    for seg_onset in np.arange(0.0, clip_length, segment_length):
        seg_offset = seg_onset + segment_length
        while frame_timestamps[seg_onset_idx + 1] <= seg_onset:
            seg_onset_idx += 1
        while seg_offset_idx < len(scores_arr) and frame_timestamps[seg_offset_idx] < seg_offset:
            seg_offset_idx += 1
        seg_weights = np.minimum(frame_timestamps[seg_onset_idx + 1 : seg_offset_idx + 1], seg_offset) - np.maximum(
            frame_timestamps[seg_onset_idx:seg_offset_idx], seg_onset
        )
        segment_scores.append(
            (seg_weights[:, None] * scores_arr[seg_onset_idx:seg_offset_idx]).sum(0) / seg_weights.sum()
        )
        segment_timestamps.append(seg_onset)
    segment_timestamps.append(clip_length)
    return create_score_dataframe(np.array(segment_scores), np.array(segment_timestamps), event_classes)
