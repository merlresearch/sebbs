# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from sed_scores_eval import collar_based, intersection_based, io

from sebbs import median_filter, package_dir


def test_median_filter_oracle_tuning_psds():
    # load predictions, ground_truth, and audio_durations
    scores = io.read_sed_scores(package_dir / "tests" / "dcase2023task4a" / "baseline2_run1_scores")
    ground_truth = io.read_ground_truth_events(
        package_dir / "tests" / "dcase2023task4a" / "metadata" / "ground_truth.tsv"
    )
    ground_truth = {audio_id: ground_truth[audio_id] for audio_id in ground_truth if audio_id in scores}
    scores = {audio_id: scores[audio_id] for audio_id in ground_truth}
    audio_durations = io.read_audio_durations(
        package_dir / "tests" / "dcase2023task4a" / "metadata" / "audio_durations.tsv"
    )
    audio_durations = {audio_id: audio_durations[audio_id] for audio_id in scores}

    # tune hyper-parameters on subset of DESED's public eval
    predictor, best_psds_values = median_filter.tune(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        filter_lengths=np.linspace(0.0, 2.0, 11),
        selection_fn=median_filter.select_best_psds,
    )
    # run inference on subset of DESED's public eval
    median_filtered_scores = predictor.predict(scores)
    # evaluate PSDS performance
    psds, single_class_psds, *_ = intersection_based.psds(
        scores=median_filtered_scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        alpha_ct=0.0,
        alpha_st=1.0,
        unit_of_time="hour",
        max_efpr=100.0,
    )
    for sound_class in single_class_psds:
        assert abs(single_class_psds[sound_class] - best_psds_values[sound_class]) < 1e-6, (
            sound_class,
            single_class_psds[sound_class],
            best_psds_values[sound_class],
        )


def test_median_filter_oracle_tuning_psds_and_cbf():
    scores = io.read_sed_scores(package_dir / "tests" / "dcase2023task4a" / "baseline2_run1_scores")
    ground_truth = io.read_ground_truth_events(
        package_dir / "tests" / "dcase2023task4a" / "metadata" / "ground_truth.tsv"
    )
    ground_truth = {audio_id: ground_truth[audio_id] for audio_id in ground_truth if audio_id in scores}
    scores = {audio_id: scores[audio_id] for audio_id in ground_truth}
    audio_durations = io.read_audio_durations(
        package_dir / "tests" / "dcase2023task4a" / "metadata" / "audio_durations.tsv"
    )
    audio_durations = {audio_id: audio_durations[audio_id] for audio_id in scores}

    # tune hyper-parameters on subset of DESED's public eval
    predictors = median_filter.tune(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        filter_lengths=np.linspace(0.0, 2.0, 11),
        selection_fn=median_filter.select_best_psds_and_cbf,
    )

    # psds predictor
    median_filter_psds, best_psds_values = predictors["psds"]
    # run inference on subset of DESED's public eval
    sed_scores = median_filter_psds.predict(scores)
    # evaluate PSDS performance
    psds, single_class_psds, *_ = intersection_based.psds(
        scores=sed_scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        alpha_ct=0.0,
        alpha_st=1.0,
        unit_of_time="hour",
        max_efpr=100.0,
    )
    for sound_class in single_class_psds:
        assert abs(single_class_psds[sound_class] - best_psds_values[sound_class]) < 1e-6, (
            sound_class,
            single_class_psds[sound_class],
            best_psds_values[sound_class],
        )

    # cbf predictor
    median_filter_cbf, best_cbf_values = predictors["cbf"]

    # the following two blocks should give the same results:
    # 1) soft score inference and thresholding in collar_based.fscore
    sed_scores = median_filter_cbf.predict(scores)
    f, *_ = collar_based.fscore(
        scores=sed_scores,
        ground_truth=ground_truth,
        threshold=median_filter_cbf.detection_threshold,
        onset_collar=0.2,
        offset_collar=0.2,
        offset_collar_rate=0.2,
    )
    for sound_class in median_filter_cbf.sound_classes:
        assert abs(f[sound_class] - best_cbf_values[sound_class]) < 1e-6, (
            sound_class,
            f[sound_class],
            best_cbf_values[sound_class],
        )

    # 2) binary score inference (detection) and simple 0.5 threshold in collar_based.fscore
    sed_scores = median_filter_cbf.detect(scores, return_sed_scores=True)
    f, *_ = collar_based.fscore(
        scores=sed_scores,
        ground_truth=ground_truth,
        threshold=0.5,  # can be any number between 0 and 1 here as the sed_scores obtained from detection are binary
        onset_collar=0.2,
        offset_collar=0.2,
        offset_collar_rate=0.2,
    )
    for sound_class in median_filter_cbf.sound_classes:
        assert abs(f[sound_class] - best_cbf_values[sound_class]) < 1e-6, (
            sound_class,
            f[sound_class],
            best_cbf_values[sound_class],
        )


def test_median_filter_cross_validation_psds():
    # load predictions, ground_truth, and audio_durations
    scores = io.read_sed_scores(package_dir / "tests" / "dcase2023task4a" / "baseline2_run1_scores")
    ground_truth = io.read_ground_truth_events(
        package_dir / "tests" / "dcase2023task4a" / "metadata" / "ground_truth.tsv"
    )
    ground_truth = {audio_id: ground_truth[audio_id] for audio_id in ground_truth if audio_id in scores}
    scores = {audio_id: scores[audio_id] for audio_id in ground_truth}
    audio_durations = io.read_audio_durations(
        package_dir / "tests" / "dcase2023task4a" / "metadata" / "audio_durations.tsv"
    )
    audio_durations = {audio_id: audio_durations[audio_id] for audio_id in scores}

    # 5-fold cross validation split
    n_folds = 5
    all_keys = scores.keys()
    shuffled_keys = sorted(all_keys)
    np.random.RandomState(0).shuffle(shuffled_keys)
    split_idx = np.linspace(0, len(shuffled_keys), n_folds + 1).astype(int).tolist()
    folds = [set(shuffled_keys[split_idx[i] : split_idx[i + 1]]) for i in range(n_folds)]

    # cross validation
    csebbs_predictors, sed_scores, detections = median_filter.cross_validation(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        folds=folds,
        filter_lengths=np.linspace(0.0, 2.0, 11),
        selection_fn=median_filter.select_best_psds,
        return_sed_scores=True,
    )

    # evaluate PSDS performance
    psds, single_class_psds, *_ = intersection_based.psds(
        scores=sed_scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        alpha_ct=0.0,
        alpha_st=1.0,
        unit_of_time="hour",
        max_efpr=100.0,
    )
    # comment assert as evaluating reduced set (20 files) only
    # assert abs(psds - 0.5839223860184434) < 1e-6, psds  # result from paper


def test_median_filter_cross_validation_psds_cbf():
    # load predictions, ground_truth, and audio_durations
    scores = io.read_sed_scores(package_dir / "tests" / "dcase2023task4a" / "baseline2_run1_scores")
    ground_truth = io.read_ground_truth_events(
        package_dir / "tests" / "dcase2023task4a" / "metadata" / "ground_truth.tsv"
    )
    ground_truth = {audio_id: ground_truth[audio_id] for audio_id in ground_truth if audio_id in scores}
    scores = {audio_id: scores[audio_id] for audio_id in ground_truth}
    audio_durations = io.read_audio_durations(
        package_dir / "tests" / "dcase2023task4a" / "metadata" / "audio_durations.tsv"
    )
    audio_durations = {audio_id: audio_durations[audio_id] for audio_id in scores}

    # 5-fold cross validation split
    n_folds = 5
    all_keys = scores.keys()
    shuffled_keys = sorted(all_keys)
    np.random.RandomState(0).shuffle(shuffled_keys)
    split_idx = np.linspace(0, len(shuffled_keys), n_folds + 1).astype(int).tolist()
    folds = [set(shuffled_keys[split_idx[i] : split_idx[i + 1]]) for i in range(n_folds)]

    # cross validation
    median_filters, sed_scores, detections = median_filter.cross_validation(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        folds=folds,
        filter_lengths=np.linspace(0.0, 2.0, 11),
        selection_fn=median_filter.select_best_psds_and_cbf,
        return_sed_scores=True,
    )

    # evaluate PSDS performance
    psds, single_class_psds, *_ = intersection_based.psds(
        scores=sed_scores["psds"],
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        alpha_ct=0.0,
        alpha_st=1.0,
        unit_of_time="hour",
        max_efpr=100.0,
    )
    # comment assert as evaluating reduced set (20 files) only
    # assert abs(psds - 0.5839223860184434) < 1e-6, psds  # result from paper

    # evaluate collar-based F1-score performance
    f, *_ = collar_based.fscore(
        scores=detections["cbf"],
        ground_truth=ground_truth,
        threshold=0.5,  # can be any number between 0 and 1 here as the sed_scores obtained from detection are binary
        onset_collar=0.2,
        offset_collar=0.2,
        offset_collar_rate=0.2,
    )
    # comment assert as evaluating reduced set (20 files) only
    # assert abs(f['macro_average'] - 0.6504733960881937) < 1e-6, f['macro_average']  # result from paper
