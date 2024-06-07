# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path

from sed_scores_eval import intersection_based, io, segment_based
from utils import (
    classes_labels_desed,
    classes_labels_maestro_real,
    classes_labels_maestro_real_eval,
    get_segment_scores,
    get_segment_scores_and_overlap_add,
    merge_overlapping_events,
)

from sebbs import csebbs
from sebbs.utils import sed_scores_from_sebbs

desed_classes = list(classes_labels_desed.keys())
maestro_classes = list(classes_labels_maestro_real.keys())
maestro_classes_eval = [
    class_label for class_label in maestro_classes if class_label in classes_labels_maestro_real_eval
]


def tune_desed(
    scores_path,
    ground_truth_path,
    durations_path,
    step_filter_lengths=(
        0.384,
        0.512,
        0.64,
    ),
    merge_thresholds_rel=(
        1.5,
        2.0,
        3.0,
    ),
    merge_thresholds_abs=(
        0.15,
        0.2,
        0.3,
    ),
    classwise=False,
):
    desed_ground_truth = io.read_ground_truth_events(ground_truth_path)
    desed_audio_durations = io.read_audio_durations(durations_path)
    desed_ground_truth = {clip_id: gt for clip_id, gt in desed_ground_truth.items() if len(gt) > 0}
    desed_audio_durations = {clip_id: desed_audio_durations[clip_id] for clip_id in desed_ground_truth.keys()}
    desed_scores = io.read_sed_scores(scores_path)
    keys = ["onset", "offset"] + desed_classes
    desed_scores = {clip_id: desed_scores[clip_id][keys] for clip_id in desed_ground_truth.keys()}
    csebbs_predictor, best_psds_values = csebbs.tune(
        scores=desed_scores,
        ground_truth=desed_ground_truth,
        audio_durations=desed_audio_durations,
        step_filter_lengths=step_filter_lengths,
        merge_thresholds_rel=merge_thresholds_rel,
        merge_thresholds_abs=merge_thresholds_abs,
        selection_fn=csebbs.select_best_psds,
        classwise=classwise,
    )
    return csebbs_predictor


def write_sed_scores(scores_path, csebbs_predictor, output_scores_dir):
    scores_all_classes = io.read_sed_scores(scores_path)
    keys = ["onset", "offset"] + desed_classes
    scores_desed_classes = {clip_id: scores_all_classes[clip_id][keys] for clip_id in scores_all_classes.keys()}
    csebbs_desed_classes = csebbs_predictor.predict(scores_desed_classes)
    keys = ["onset", "offset"] + maestro_classes
    scores_maestro_classes = {clip_id: scores_all_classes[clip_id][keys] for clip_id in scores_all_classes.keys()}
    segment_scores_maestro_classes = {
        clip_id: get_segment_scores(
            clip_scores,
            clip_length=list(clip_scores["offset"])[-1],
            segment_length=1.0,
        )
        for clip_id, clip_scores in scores_maestro_classes.items()
    }
    sebbs_all = {
        clip_id: sorted(
            csebbs_desed_classes[clip_id]
            + [
                (float(onset), float(offset), class_name, float(seg_score))
                for onset, offset, scores_vec in zip(
                    segment_scores_maestro_classes[clip_id]["onset"],
                    segment_scores_maestro_classes[clip_id]["offset"],
                    segment_scores_maestro_classes[clip_id][maestro_classes].to_numpy(),
                )
                for class_name, seg_score in zip(maestro_classes, scores_vec)
            ]
        )
        for clip_id in scores_all_classes
    }
    sed_scores = sed_scores_from_sebbs(sebbs_all, sound_classes=desed_classes + maestro_classes, fill_value=0.0)
    io.write_sed_scores(sed_scores, output_scores_dir)


def eval_desed(scores_path, ground_truth_path, durations_path):
    scores = io.read_sed_scores(scores_path)
    ground_truth = io.read_ground_truth_events(ground_truth_path)
    audio_durations = io.read_audio_durations(durations_path)
    audio_durations = {clip_id: audio_durations[clip_id] for clip_id in ground_truth}
    keys = ["onset", "offset"] + desed_classes
    scores = {clip_id: scores[clip_id][keys] for clip_id in ground_truth}
    psds, single_class_psds, *_ = intersection_based.psds(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        alpha_ct=0.0,
        alpha_st=1.0,
    )
    print("PSDS1:", psds)


def eval_maestro(scores_path, ground_truth_path, durations_path):
    scores = io.read_sed_scores(scores_path)
    ground_truth = io.read_ground_truth_events(ground_truth_path)
    ground_truth = merge_overlapping_events(ground_truth)
    audio_durations = io.read_audio_durations(durations_path)
    keys = ["onset", "offset"] + maestro_classes_eval
    scores = {clip_id: scores[clip_id][keys] for clip_id in scores if clip_id.split("-")[0] in ground_truth}

    sed_scores = get_segment_scores_and_overlap_add(
        scores,
        audio_durations=audio_durations,
        event_classes=maestro_classes_eval,
    )
    mpauc = segment_based.auroc(
        scores=sed_scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        segment_length=1.0,
        max_fpr=0.1,
    )[0]["mean"]
    print("mPAUC:", mpauc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev_scores",
        required=True,
        type=str,
        help='Path to scores root directory (with "postprocessed" and "unprocessed" subdirectories in it) '
        "to be used for hyper-param tuning.",
    )
    parser.add_argument(
        "--dev_ground_truth_desed",
        required=True,
        type=str,
        help="Path to DESED ground truth tsv-file to be used for hyper-param tuning.",
    )
    parser.add_argument(
        "--dev_durations_desed",
        required=True,
        type=str,
        help="Path to DESED audio durations tsv-file to be used for hyper-param tuning.",
    )
    parser.add_argument(
        "--test_scores",
        required=True,
        type=str,
        help='Path to test scores root directory (with "postprocessed" and "unprocessed" subdirectories in it) '
        "to be used for performance report.",
    )
    parser.add_argument(
        "--output_scores_dir", required=True, type=str, help="Path directory where to to save output test scores."
    )
    parser.add_argument(
        "--test_ground_truth_desed",
        required=False,
        type=str,
        help="Path to DESED test ground truth tsv-file to be used for performance report.",
        default=None,
    )
    parser.add_argument(
        "--test_durations_desed",
        required=False,
        type=str,
        help="Path to DESED test audio durations tsv-file to be used for performance report.",
        default=None,
    )
    parser.add_argument(
        "--test_ground_truth_maestro",
        required=False,
        type=str,
        help="Path to Maestro test ground truth tsv-file (for long files) to be used for performance report.",
        default=None,
    )
    parser.add_argument(
        "--test_durations_maestro",
        required=False,
        type=str,
        help="Path to Maestro test audio durations tsv-file (for long files) to be used for performance report.",
        default=None,
    )

    args = parser.parse_args()
    dev_scores_dir = Path(args.dev_scores)
    dev_ground_truth_desed = args.dev_ground_truth_desed
    dev_durations_desed = args.dev_durations_desed
    test_scores_dir = Path(args.test_scores)
    test_ground_truth_desed = args.test_ground_truth_desed
    test_durations_desed = args.test_durations_desed
    test_ground_truth_maestro = args.test_ground_truth_maestro
    test_durations_maestro = args.test_durations_maestro
    output_scores_dir = args.output_scores_dir

    if test_ground_truth_desed or test_durations_desed or test_ground_truth_maestro or test_durations_maestro:
        print("\nperformance without csebbs:")
        for name in ["postprocessed", "unprocessed"]:
            print(f"{name}:")
            if test_ground_truth_desed or test_durations_desed:
                assert test_ground_truth_desed and test_durations_desed
                eval_desed(
                    scores_path=test_scores_dir / name,
                    ground_truth_path=test_ground_truth_desed,
                    durations_path=test_durations_desed,
                )
            if test_ground_truth_maestro or test_durations_maestro:
                assert test_ground_truth_maestro and test_durations_maestro
                eval_maestro(
                    scores_path=test_scores_dir / name,
                    ground_truth_path=test_ground_truth_maestro,
                    durations_path=test_durations_maestro,
                )

    print("\ntune csebbs for DESED classes ...")
    csebbs_predictor_desed = tune_desed(
        scores_path=dev_scores_dir / "unprocessed",
        ground_truth_path=dev_ground_truth_desed,
        durations_path=dev_durations_desed,
    )
    print("write output scores ...")
    write_sed_scores(
        scores_path=test_scores_dir / "unprocessed",
        csebbs_predictor=csebbs_predictor_desed,
        output_scores_dir=output_scores_dir,
    )

    if test_ground_truth_desed or test_durations_desed or test_ground_truth_maestro or test_durations_maestro:
        print("\nperformance with csebbs:")
        if test_ground_truth_desed or test_durations_desed:
            assert test_ground_truth_desed and test_durations_desed
            eval_desed(
                scores_path=output_scores_dir,
                ground_truth_path=test_ground_truth_desed,
                durations_path=test_durations_desed,
            )
        if test_ground_truth_maestro or test_durations_maestro:
            assert test_ground_truth_maestro and test_durations_maestro
            eval_maestro(
                scores_path=output_scores_dir,
                ground_truth_path=test_ground_truth_maestro,
                durations_path=test_durations_maestro,
            )
