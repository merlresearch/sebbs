# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import itertools
from pathlib import Path
from typing import Callable, Iterable, Tuple, Union

import numpy as np
from sed_scores_eval import collar_based, intersection_based, io
from sed_scores_eval.base_modules.scores import validate_score_dataframe

from sebbs.change_detection import change_detection
from sebbs.utils import sed_scores_from_detections, sed_scores_from_sebbs


class CSEBBsPredictor:
    """change-point based predictor of Sound Event Bounding Boxes (cSEBBs).

    [1] J.Ebbers, F.Germain, G.Wichern and J.Le Roux, "Sound Event Bounding Boxes",
        Accepted for publication at Interspeech, 2024

    """

    def __init__(
        self,
        step_filter_length: Union[float, dict] = 0.5,
        merge_threshold_abs: Union[float, dict] = 1.0,
        merge_threshold_rel: Union[float, dict] = 2.0,
        detection_threshold: Union[float, dict, None] = None,
        sound_classes: Union[list, None] = None,
    ):
        """

        Args:
            step_filter_length (float|dict): (class-wise) step filter length(s)
                for change point detection.
            merge_threshold_abs (float|dict): (class-wise) absolute threshold(s) for segment merging.
                If the absolute difference between a min value in a gap segment
                and the max value in the neighbouring event candidates is
                smaller than this threshold, the three segments (two events with
                a gap in between) may be merged (if rel_threshold is fulfilled also)
            merge_threshold_rel (float|dict): (class-wise) relative threshold(s) for segment merging.
                If the relative difference between a min value in a gap segment
                and the max value in the neighbouring event candidates (max value/min_value)
                is smaller than this threshold, the three segments (two events with
                a gap in between) may be merged (if abs_threshold is fulfilled also)
            detection_threshold (float|dict|None): (class-wise) detection/decision thresholds.
            sound_classes (list of string|None): sound classes
        """
        self.step_filter_length = step_filter_length
        self.merge_threshold_abs = merge_threshold_abs
        self.merge_threshold_rel = merge_threshold_rel
        self.detection_threshold = detection_threshold
        self.sound_classes = sound_classes

    def copy(self):
        """get copy of the class instance (CSEBBsPredictor)"""
        return CSEBBsPredictor(
            {**self.step_filter_length} if isinstance(self.step_filter_length, dict) else self.step_filter_length,
            {**self.merge_threshold_abs} if isinstance(self.merge_threshold_abs, dict) else self.merge_threshold_abs,
            {**self.merge_threshold_rel} if isinstance(self.merge_threshold_rel, dict) else self.merge_threshold_rel,
            sound_classes=self.sound_classes,
        )

    def predict(
        self,
        scores: Union[str, dict],
        audio_ids: Union[Iterable, None] = None,
        return_sed_scores: bool = False,
    ) -> dict:
        """run post-processing on scores to predict sebbs

        Args:
            scores (str | dict of pandas.DataFrame): (path to directory with)
                SED posterior score data frames as used in sed_scores_eval
                (https://github.com/fgnt/sed_scores_eval).
            audio_ids (Iterable|None): audio ids that prediction should be restricted to.
            return_sed_scores (bool): whether to return outputs as SED scores
                data frame format for direct use with sed_scores_eval-based evaluation.

        Returns (dict of list|pd.DataFrame):
            a list of SEBBs (onset, offset, class, confidence) (which may have
            been converted to a SED scores data frame) for each audio_id

        """
        if isinstance(scores, (str, Path)):
            scores = io.read_sed_scores(scores)
        if audio_ids is None:
            audio_ids = scores.keys()
        scores = {audio_id: scores[audio_id] for audio_id in audio_ids}
        _, self.sound_classes = validate_score_dataframe(scores[list(audio_ids)[0]], event_classes=self.sound_classes)
        change_detection = self._run_change_detection(scores)
        sebbs = self._get_sebbs_from_change_detection(change_detection)
        if return_sed_scores:
            return sed_scores_from_sebbs(sebbs, self.sound_classes)
        return sebbs

    def detect(
        self,
        scores: Union[str, dict],
        detection_threshold: Union[float, dict, None] = None,
        audio_ids: Union[Iterable, None] = None,
        return_sed_scores: bool = False,
    ) -> dict:
        """predict SEBBs and derive detections/decisions by employing detection_threshold

        Args:
            scores (str | dict of pandas.DataFrame): (path to directory with)
                SED posterior score data frames as used in sed_scores_eval
                (https://github.com/fgnt/sed_scores_eval).
            detection_threshold (float|dict|None): (class-wise) detection/decision thresholds.
                If None, self.detection_threshold is used.
            audio_ids (iterable|None): audio ids that prediction should be restricted to.
            return_sed_scores (bool): whether to return outputs as SED scores
                data frame format for direct use with sed_scores_eval-based evaluation.
                In that case, sed_scores are binary due to detection thresholding.

        Returns (dict of list):
            a list of detected sound events (onset, offset, class) (which may have
            been converted to a SED scores data frame) for each audio_id

        """
        if detection_threshold is None:
            detection_threshold = self.detection_threshold
        assert detection_threshold is not None, "A detection threshold has to be provided to run detection."

        sebbs = self.predict(scores, audio_ids)

        return self.detection_thresholding(
            sebbs,
            detection_threshold,
            return_sed_scores=return_sed_scores,
        )

    def detection_thresholding(
        self,
        sebbs: dict,
        detection_threshold: Union[float, dict, None] = None,
        return_sed_scores: bool = False,
    ) -> dict:
        """

        Args:
            sebbs (dict of list): a list of SEBBs (onset, offset, class, confidence) for each audio_id
            detection_threshold (float|dict|None): (class-wise) detection/decision thresholds.
                If None, self.detection_threshold is used.
            return_sed_scores (bool): whether to return outputs as SED scores
                data frame format for direct use with sed_scores_eval-based evaluation.
                In that case, sed_scores are binary due to detection thresholding.

        Returns (dict of list):
            a list of detected sound events (onset, offset, class) (which may have
            been converted to a SED scores data frame) for each audio_id

        """
        if detection_threshold is None:
            detection_threshold = self.detection_threshold
        assert detection_threshold is not None, "A detection threshold has to be provided to run detection."
        if not isinstance(detection_threshold, dict):
            assert np.isscalar(detection_threshold), detection_threshold
            detection_threshold = {sound_class: detection_threshold for sound_class in self.sound_classes}
        detections = {
            audio_id: [sebb[:3] for sebb in sebbs_i if sebb[3] > detection_threshold[sebb[2]]]
            for audio_id, sebbs_i in sebbs.items()
        }
        if return_sed_scores:
            detections = sed_scores_from_detections(
                detections,
                sound_classes=self.sound_classes,
            )
        return detections

    def _run_change_detection(self, scores):
        """detect candidate segment boundaries/change points"""
        if isinstance(self.step_filter_length, dict):
            step_filter_length = np.array([self.step_filter_length[sound_class] for sound_class in self.sound_classes])
        else:
            step_filter_length = self.step_filter_length
        change_detection_out = {}
        for key, scores_df in scores.items():
            timestamps, self.sound_classes = validate_score_dataframe(scores_df, event_classes=self.sound_classes)
            scores_arr = scores_df[self.sound_classes].to_numpy()
            change_detection_out[key] = change_detection(scores_arr, timestamps, step_filter_length)
        for key, det in change_detection_out.items():
            change_detection_out[key] = {
                sound_class: change_detection_out[key][c] for c, sound_class in enumerate(self.sound_classes)
            }
        return change_detection_out

    def _get_sebbs_from_change_detection(self, change_detection):
        """perform merging of segments and infer SEBBs"""
        sebbs = {}
        for audio_id in change_detection.keys():
            onsets = []
            offsets = []
            confidences = []
            class_labels = []
            for k, sound_class in enumerate(self.sound_classes):
                seg_bounds, mean_scores_k, min_scores_k, max_scores_k = change_detection[audio_id][sound_class]
                abs_thres = (
                    self.merge_threshold_abs[sound_class]
                    if isinstance(self.merge_threshold_abs, dict)
                    else self.merge_threshold_abs
                )
                rel_thres = (
                    self.merge_threshold_rel[sound_class]
                    if isinstance(self.merge_threshold_rel, dict)
                    else self.merge_threshold_rel
                )
                onsets_c, offsets_c, confidences_c = _merge_segments(
                    seg_bounds,
                    mean_scores_k,
                    min_scores_k,
                    max_scores_k,
                    threshold_abs=abs_thres,
                    threshold_rel=rel_thres,
                )
                onsets.extend(onsets_c)
                offsets.extend(offsets_c)
                class_labels.extend(len(onsets_c) * [sound_class])
                confidences.extend(confidences_c)
            sebbs[audio_id] = [
                (onset, offset, class_label, confidence)
                for onset, offset, class_label, confidence in zip(
                    onsets,
                    offsets,
                    class_labels,
                    confidences,
                )
            ]
        return sebbs


def _merge_segments(
    seg_bounds,
    mean_scores,
    min_scores,
    max_scores,
    threshold_abs=0.5,
    threshold_rel=2.0,
):
    """perform merging of segments/candidate events when scores differ less than an absolute and/or relative threshold.

    Args:
        seg_bounds: candidate segment boundaries/change points.
            seg_bounds[0] equals audio onset 0., seg_bounds[1:-2:2] are event
            onset candidates, seg_bounds[2:-1:2] are event offset candidates,
            and seg_bounds[-1] equals audio length.
        mean_scores (1d np.ndarray): candidate segments' mean scores
        min_scores (1d np.ndarray): candidate segments' min scores
        max_scores (1d np.ndarray): candidate segments' max scores
        threshold_abs (float): the absolute threshold for segment merging.
            If the absolute difference between a min value in a gap segment
            and the max value in the neighbouring event candidates is
            smaller than this threshold, the three segments (two events with
            a gap in between) may be merged (if threshold_rel is fulfilled also).
        threshold_rel (float): the relative threshold for segment merging.
            If the relative difference between a min value in a gap segment
            and the max value in the neighbouring event candidates (max value/min_value)
            is smaller than this threshold, the three segments (two events with
            a gap in between) may be merged (if threshold_abs is fulfilled also).

    Returns:

    """
    if len(seg_bounds) >= 6:
        event_scores = max_scores[1:-1:2] + 1e-3
        gap_scores = min_scores[2:-2:2] + 1e-3
        gap_idx = np.argwhere(
            (
                ((event_scores[:-1] / gap_scores) < threshold_rel)
                * ((event_scores[1:] / gap_scores) < threshold_rel)
                * ((event_scores[:-1] - gap_scores) < threshold_abs)
                * ((event_scores[1:] - gap_scores) < threshold_abs)
            )
            < 0.5
        ).flatten()

        seg_idx = np.stack((2 + 2 * gap_idx, 2 + 2 * gap_idx + 1), axis=1).flatten()
        seg_idx = np.concatenate(([0, 1], seg_idx, [len(seg_bounds) - 2, len(seg_bounds) - 1])).astype(int)
        seg_bounds, mean_scores, min_scores, max_scores = _merge_scores(
            seg_bounds,
            mean_scores,
            min_scores,
            max_scores,
            seg_idx,
        )

    onsets = seg_bounds[1:-2:2]
    offsets = seg_bounds[2:-1:2]
    mean_scores = mean_scores[1:-1:2]
    return onsets, offsets, mean_scores


def _merge_scores(seg_bounds, mean_scores, min_scores, max_scores, seg_idx):
    """compute new mean,min and max score for segments after merging.

    Args:
        seg_bounds (1d np.ndarray): old segment bounds
        mean_scores (1d np.ndarray): old segments' mean scores
        min_scores (1d np.ndarray): old segments' min scores
        max_scores (1d np.ndarray): old segments' max scores
        seg_idx: indices of remaining segment boundaries

    Returns:
        seg_bounds (1d np.ndarray): new segment bounds
        mean_scores (1d np.ndarray): new segments' mean scores
        min_scores (1d np.ndarray): new segments' min scores
        max_scores (1d np.ndarray): new segments' max scores

    """
    seglens = seg_bounds[1:] - seg_bounds[:-1]
    mean_scores_new = []
    min_scores_new = []
    max_scores_new = []
    for j in range(0, len(seg_idx) - 1):
        if seglens[seg_idx[j] : seg_idx[j + 1]].sum() == 0.0:
            mean_scores_new.append(0.0)
            min_scores_new.append(0.0)
            max_scores_new.append(0.0)
        else:
            mean_scores_new.append(
                (seglens[seg_idx[j] : seg_idx[j + 1]] * mean_scores[seg_idx[j] : seg_idx[j + 1]]).sum()
                / (seglens[seg_idx[j] : seg_idx[j + 1]].sum() + 1e-6)
            )
            min_scores_new.append(min(min_scores[seg_idx[j] : seg_idx[j + 1]]))
            max_scores_new.append(max(max_scores[seg_idx[j] : seg_idx[j + 1]]))
    seg_bounds = seg_bounds[seg_idx]
    mean_scores = np.array(mean_scores_new)
    min_scores = np.array(min_scores_new)
    max_scores = np.array(max_scores_new)
    return seg_bounds, mean_scores, min_scores, max_scores


def tune(
    scores: Union[str, dict],
    ground_truth: Union[str, dict],
    audio_durations: Union[str, dict],
    *,
    step_filter_lengths: Iterable = (0.32, 0.48, 0.64),
    merge_thresholds_abs: Iterable = (0.15, 0.2, 0.3),
    merge_thresholds_rel: Iterable = (1.5, 2.0, 3.0),
    selection_fn: Callable,
    folds: Union[Iterable, None] = None,
    either_abs_or_rel_threshold: bool = True,
    **selection_kwargs,
) -> Union[Tuple[CSEBBsPredictor, dict], list]:
    """perform grid search over hyper-parameters

    Args:
        scores (str | dict of pandas.DataFrame): (path to directory with)
            SED posterior score data frames as used in sed_scores_eval
            (https://github.com/fgnt/sed_scores_eval).
        ground_truth (str | dict of list): path to/dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio as used in
            sed_scores_eval.
        audio_durations (str | dict): (path to) durations of each audio file.
        step_filter_lengths (iterable): candidates for step_filter_length.
        merge_thresholds_abs (iterable): candidates for merge_threshold_abs.
        merge_thresholds_rel (iterable): candidates for merge_threshold_abs.
        selection_fn (callable): function that takes list of
            (CSEBBsPredictor, Sebb predictions) tuples together with
            ground truth, audio durations, the audio ids to be considered and
            further optional arguments and selects best CSEBBsPredictor parameters.
        folds (iterable|None): optional list of audio_id sets for which best parameters
            should be selected separately in which case a list of
            CSEBBsPredictors is returned. If None, best parameters are selected
            based on the whole set and a single CSEBBsPredictor is returned.
        either_abs_or_rel_threshold (bool): if True, either merge_thresholds_rel
            or merge_thresholds_abs is used. Whereas if False, they may be used
            jointly (which, however, increases the number of parameter
            combinations that need to be evaluated during tuning).
        **selection_kwargs: any key-word arguments that should be forwarded to selection_fn.

    Returns: (list of) return(s) from selection_fn

    """
    if isinstance(scores, (str, Path)):
        scores = io.read_sed_scores(scores)
    if isinstance(ground_truth, (str, Path)):
        ground_truth = io.read_ground_truth_events(ground_truth)
    if isinstance(audio_durations, (str, Path)):
        audio_durations = io.read_audio_durations(audio_durations)
    csebbs = []
    for step_filt_len in step_filter_lengths:
        csebbs_predictor = CSEBBsPredictor(
            step_filter_length=step_filt_len,
        )
        change_detection = csebbs_predictor._run_change_detection(scores)
        if either_abs_or_rel_threshold:
            it = [(thres_abs, np.inf) for thres_abs in merge_thresholds_abs] + [
                (np.inf, thres_rel) for thres_rel in merge_thresholds_rel
            ]
        else:
            it = itertools.product(merge_thresholds_abs, merge_thresholds_rel)
        for abs_thres, rel_thres in it:
            pred = csebbs_predictor.copy()
            pred.merge_threshold_abs = abs_thres
            pred.merge_threshold_rel = rel_thres
            csebbs.append((pred, pred._get_sebbs_from_change_detection(change_detection)))
    if folds is None:
        return selection_fn(csebbs, ground_truth, audio_durations, **selection_kwargs)
    return [selection_fn(csebbs, ground_truth, audio_durations, audio_ids=fold, **selection_kwargs) for fold in folds]


def cross_validation(
    scores: Union[str, dict],
    ground_truth: Union[str, dict],
    audio_durations: Union[str, dict],
    folds: Iterable,
    *,
    step_filter_lengths: Iterable = (0.32, 0.48, 0.64),
    merge_thresholds_abs: Iterable = (0.15, 0.2, 0.3),
    merge_thresholds_rel: Iterable = (1.5, 2.0, 3.0),
    either_abs_or_rel_threshold: bool = True,
    selection_fn: Callable,
    return_sed_scores: bool = True,
    **selection_kwargs,
) -> Tuple[list, dict, dict]:
    """perform leave-one-out cross validation. I.e., tune hyper-parameters on
    all folds but one and write outputs for the left out fold.

    Args:
        scores (str | dict of pandas.DataFrame): (path to directory with)
            SED posterior score data frames as used in sed_scores_eval
            (https://github.com/fgnt/sed_scores_eval).
        ground_truth (str | dict of list): path to/dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio as used in
            sed_scores_eval.
        audio_durations (str | dict): (path to) durations of each audio file.
        folds (iterable): optional list of audio_id sets for which best parameters
            should be selected separately in which case a list of
            CSEBBsPredictors is returned. If None, best parameters are selected
            based on the whole set and a single CSEBBsPredictor is returned.
        step_filter_lengths (iterable): candidates for step_filter_length.
        merge_thresholds_abs (iterable): candidates for merge_threshold_abs.
        merge_thresholds_rel (iterable): candidates for merge_threshold_abs.
        selection_fn (callable): function that takes list of
            (CSEBBsPredictor, Sebb predictions) tuples together with
            ground truth, audio durations, the audio ids to be considered and
            further optional arguments and selects best CSEBBsPredictor parameters.
        either_abs_or_rel_threshold (bool): if True, either merge_thresholds_rel
            or merge_thresholds_abs is used. Whereas if False, they may be used
            jointly (which, however, increases the number of parameter
            combinations that need to be evaluated during tuning).
        return_sed_scores (bool): whether to return outputs in sed_scores_eval
            format for direct use with sed_scores_eval-based evaluation.
        **selection_kwargs: any key-word arguments that should be forwarded to selection_fn.

    Returns:
        csebbs_predictors (list): list of return(s) from selection_fn
        sebbs (dict): predicted SEBBs for the whole data set,
            where predictions for different folds used different tuned hyper-parameters
        detections (dict|None): Detected events for the whole data set,
            if detection thresholds are selected by selection_fn else None.

    """
    if isinstance(scores, (str, Path)):
        scores = io.read_sed_scores(scores)
    if isinstance(ground_truth, (str, Path)):
        ground_truth = io.read_ground_truth_events(ground_truth)
    if isinstance(audio_durations, (str, Path)):
        audio_durations = io.read_audio_durations(audio_durations)
    test_ids = [set(fold) for fold in folds]
    all_ids = set.union(*test_ids)
    val_ids = [(all_ids - fold) for fold in test_ids]

    csebbs_predictors = tune(
        scores,
        ground_truth,
        audio_durations,
        folds=val_ids,
        step_filter_lengths=step_filter_lengths,
        merge_thresholds_abs=merge_thresholds_abs,
        merge_thresholds_rel=merge_thresholds_rel,
        either_abs_or_rel_threshold=either_abs_or_rel_threshold,
        selection_fn=selection_fn,
        **selection_kwargs,
    )
    sebbs = {}
    detections = {}
    for i, predictors in enumerate(csebbs_predictors):
        if not isinstance(predictors, dict):
            predictors = {"": predictors}
        for key, predictor in predictors.items():
            if isinstance(predictor, tuple):
                predictor = predictor[0]
            assert isinstance(
                predictor, CSEBBsPredictor
            ), "selection_fn doesn't seem to have returned a CSEBBsPredictor (as first return element)."
            if key not in sebbs:
                sebbs[key] = {}
                detections[key] = {}
            sebbs_i = predictor.predict(scores, audio_ids=test_ids[i])
            if predictor.detection_threshold is not None:
                detections_i = predictor.detection_thresholding(
                    sebbs_i,
                    return_sed_scores=return_sed_scores,
                )
                detections[key].update(detections_i)
            if return_sed_scores:
                sebbs_i = sed_scores_from_sebbs(sebbs_i, predictor.sound_classes)
            sebbs[key].update(sebbs_i)
    if len(sebbs) == 1 and "" in sebbs:
        sebbs = sebbs[""]
        detections = detections[""]
    return csebbs_predictors, sebbs, detections


def select_best_psds(
    csebbs: dict,
    ground_truth: dict,
    audio_durations: dict,
    audio_ids: Union[Iterable, None] = None,
    dtc_threshold: float = 0.7,
    gtc_threshold: float = 0.7,
    cttc_threshold: Union[float, None] = None,
    alpha_ct: float = 0.0,
    unit_of_time: str = "hour",
    max_efpr: float = 100.0,
    classwise: bool = True,
) -> Tuple[CSEBBsPredictor, dict]:
    """select parameters which give highest PSDS

    Args:
        csebbs (list): list of (CSEBBsPredictor, SEBB predictions) from grid search performed in tune function above
        ground_truth (dict of list): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio as used in
            sed_scores_eval.
        audio_durations: The duration of each audio file.
        audio_ids (iterable|None): audio ids that parameter selection should be restricted to.
        dtc_threshold (float): detection tolerance criterion threshold of PSDS
        gtc_threshold (float): ground truth intersection criterion threshold of PSDS
        cttc_threshold (float|None): cross trigger tolerance criterion threshold of PSDS
        alpha_ct (float): cross trigger penalization weight of PSDS.
        unit_of_time (str): time unit for FPR computation in PSDS
        max_efpr (float): maximum false positives per time unit to be considered in PSD-ROC.
        classwise (bool): whether to use different parameters for each sound_class or not.

    Returns:
        csebbs_predictor: CSEBBsPredictor with best hyper-parameters w.r.t. PSDS
        best_values: class-wise PSDS values with best hyper-parameters

    """
    if audio_ids is not None:
        csebbs = [
            (predictor, {audio_id: csebbs_i[audio_id] for audio_id in audio_ids}) for predictor, csebbs_i in csebbs
        ]
        ground_truth = {audio_id: ground_truth[audio_id] for audio_id in audio_ids}
        audio_durations = {audio_id: audio_durations[audio_id] for audio_id in audio_ids}
    # convert sebbs to sed_scores_eval score format for PSDS evaluation.
    csebbs_scores = [
        (
            predictor,
            sed_scores_from_sebbs(csebbs_i, sound_classes=predictor.sound_classes, audio_duration=audio_durations),
        )
        for predictor, csebbs_i in csebbs
    ]
    best_step_filter_length = {}
    best_merge_threshold_abs = {}
    best_merge_threshold_rel = {}
    best_values = {}
    for predictor, csebbs_scores_i in csebbs_scores:
        single_class_psds = intersection_based.psds(
            scores=csebbs_scores_i,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            dtc_threshold=dtc_threshold,
            gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold,
            alpha_ct=alpha_ct,
            unit_of_time=unit_of_time,
            max_efpr=max_efpr,
        )[1]
        mean = np.mean(list(single_class_psds.values()))
        for sound_class in single_class_psds:
            value = single_class_psds[sound_class] if classwise else mean
            if (sound_class not in best_values) or (value > best_values[sound_class]):
                # for this sound class the performance achieved with current params is better than with previous params
                best_values[sound_class] = value
                best_step_filter_length[sound_class] = predictor.step_filter_length
                best_merge_threshold_abs[sound_class] = predictor.merge_threshold_abs
                best_merge_threshold_rel[sound_class] = predictor.merge_threshold_rel

    csebbs_predictor = CSEBBsPredictor(
        step_filter_length=best_step_filter_length,
        merge_threshold_rel=best_merge_threshold_rel,
        merge_threshold_abs=best_merge_threshold_abs,
    )
    return csebbs_predictor, best_values


def select_best_cbf(
    csebbs: dict,
    ground_truth: dict,
    audio_durations: dict,
    audio_ids: Union[Iterable, None] = None,
    onset_collar: float = 0.2,
    offset_collar: float = 0.2,
    offset_collar_rate: float = 0.2,
    classwise: bool = True,
) -> Tuple[CSEBBsPredictor, dict]:
    """select parameters which give highest collar-based F1-score

    Args:
        csebbs (list): list of (CSEBBsPredictor, SEBB predictions) from grid search performed in tune function above
        ground_truth (dict of list): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio as used in
            sed_scores_eval.
        audio_durations: The duration of each audio file.
        audio_ids (iterable|None): audio ids that parameter selection should be restricted to.
        onset_collar (float): onset collar in seconds to be used in collar-based evaluation.
        offset_collar (float): min offset collar in seconds to be used in collar-based evaluation.
        offset_collar_rate (float): min offset collar as a rate of the ground truth event length
            to be used in collar-based evaluation. Actual offset collar given as
            offset_collar_for_gt_event = max(
                offset_collar, offset_collar_rate*length_of_gt_event_in_seconds
            )
        classwise (bool): whether to use different parameters for each sound_class or not.

    Returns:
        csebbs_predictor: CSEBBsPredictor with best hyper-parameters w.r.t. collar-based F1-score
        best_values: class-wise collar-based F1-score values with best hyper-params

    """
    if audio_ids is not None:
        csebbs = [
            (predictor, {audio_id: csebbs_i[audio_id] for audio_id in audio_ids}) for predictor, csebbs_i in csebbs
        ]
        ground_truth = {audio_id: ground_truth[audio_id] for audio_id in audio_ids}
        audio_durations = {audio_id: audio_durations[audio_id] for audio_id in audio_ids}
    # convert sebbs to sed_scores_eval score format for F1-score evaluation.
    csebbs_scores = [
        (
            predictor,
            sed_scores_from_sebbs(csebbs_i, sound_classes=predictor.sound_classes, audio_duration=audio_durations),
        )
        for predictor, csebbs_i in csebbs
    ]
    best_step_filter_length = {}
    best_merge_threshold_abs = {}
    best_merge_threshold_rel = {}
    best_detection_threshold = {}
    best_values = {}
    for predictor, csebbs_scores_i in csebbs_scores:
        f, _, _, thresholds_cbf, _ = collar_based.best_fscore(
            scores=csebbs_scores_i,
            ground_truth=ground_truth,
            onset_collar=onset_collar,
            offset_collar=offset_collar,
            offset_collar_rate=offset_collar_rate,
        )
        for sound_class in thresholds_cbf:
            value = f[sound_class] if classwise else f["macro_average"]
            if (sound_class not in best_values) or (value > best_values[sound_class]):
                # for this sound class the performance achieved with current params is better than with previous params
                best_values[sound_class] = value
                best_step_filter_length[sound_class] = predictor.step_filter_length
                best_merge_threshold_abs[sound_class] = predictor.merge_threshold_abs
                best_merge_threshold_rel[sound_class] = predictor.merge_threshold_rel
                best_detection_threshold[sound_class] = thresholds_cbf[sound_class]

    csebbs_predictor = CSEBBsPredictor(
        step_filter_length=best_step_filter_length,
        merge_threshold_rel=best_merge_threshold_rel,
        merge_threshold_abs=best_merge_threshold_abs,
        detection_threshold=best_detection_threshold,
    )
    return csebbs_predictor, best_values


def select_best_psds_and_cbf(
    csebbs: list,
    ground_truth: dict,
    audio_durations: dict,
    audio_ids: Union[Iterable, None] = None,
    dtc_threshold: float = 0.7,
    gtc_threshold: float = 0.7,
    cttc_threshold: Union[float, None] = None,
    alpha_ct: float = 0.0,
    unit_of_time: str = "hour",
    max_efpr: float = 100.0,
    onset_collar: float = 0.2,
    offset_collar: float = 0.2,
    offset_collar_rate: float = 0.2,
    classwise: bool = True,
) -> dict:
    """Select both, best PSDS and collar-based parameters

    Args:
        csebbs (list): list of (CSEBBsPredictor, SEBB predictions) from grid search performed in tune function above
        ground_truth (dict of list): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio as used in
            sed_scores_eval.
        audio_durations: The duration of each audio file.
        audio_ids (iterable|None): audio ids that parameter selection should be restricted to.
        dtc_threshold (float): detection tolerance criterion threshold of PSDS
        gtc_threshold (float): ground truth intersection criterion threshold of PSDS
        cttc_threshold (float|None): cross trigger tolerance criterion threshold of PSDS
        alpha_ct (float): cross trigger penalization weight of PSDS.
        unit_of_time (str): time unit for FPR computation in PSDS
        max_efpr (float): maximum false positives per time unit to be considered in PSD-ROC.
        onset_collar (float): onset collar in seconds to be used in collar-based evaluation.
        offset_collar (float): min offset collar in seconds to be used in collar-based evaluation.
        offset_collar_rate (float): min offset collar as a rate of the ground truth event length
            to be used in collar-based evaluation. Actual offset collar given as
            offset_collar_for_gt_event = max(
                offset_collar, offset_collar_rate*length_of_gt_event_in_seconds
            )
        classwise (bool): whether to use different parameters for each sound_class or not.

    Returns: dict of (tuned CSEBBsPredictors, class-wise performance dict) tuples
        for both PSDS and collar-based F1-score metrics

    """
    return {
        "psds": select_best_psds(
            csebbs,
            ground_truth,
            audio_durations,
            audio_ids=audio_ids,
            dtc_threshold=dtc_threshold,
            gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold,
            alpha_ct=alpha_ct,
            unit_of_time=unit_of_time,
            max_efpr=max_efpr,
            classwise=classwise,
        ),
        "cbf": select_best_cbf(
            csebbs,
            ground_truth,
            audio_durations,
            audio_ids=audio_ids,
            onset_collar=onset_collar,
            offset_collar=offset_collar,
            offset_collar_rate=offset_collar_rate,
            classwise=classwise,
        ),
    }
