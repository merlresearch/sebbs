# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
from typing import Callable, Iterable, Tuple, Union

import numpy as np
from sed_scores_eval import collar_based, intersection_based, io
from sed_scores_eval.base_modules.detection import scores_to_event_list
from sed_scores_eval.base_modules.postprocessing import medfilt
from sed_scores_eval.base_modules.scores import validate_score_dataframe

from sebbs.utils import sed_scores_from_detections


class MedianFilter:
    def __init__(
        self,
        filter_length: Union[float, dict],
        detection_threshold: Union[float, dict, None] = None,
        sound_classes: Union[list, None] = None,
    ):
        """wrapper around sed_scores_eval's medfilt

        Args:
            filter_length (float|dict): (class-wise) median filter length(s).
            detection_threshold (float|dict|None): (class-wise) detection/decision thresholds.
            sound_classes (list of string|None): sound classes
        """
        self.filter_length = filter_length
        self.detection_threshold = detection_threshold
        self.sound_classes = sound_classes

    def copy(self):
        """get copy of the class instance (CSEBBsPredictor)"""
        return MedianFilter(
            {**self.filter_length} if isinstance(self.filter_length, dict) else self.filter_length,
            sound_classes=self.sound_classes,
        )

    def predict(
        self,
        scores: Union[str, dict],
        audio_ids: Union[Iterable, None] = None,
    ) -> dict:
        """run median filtering

        Args:
            scores (dict of pandas.DataFrame): SED posterior score data frames
                as used in sed_scores_eval (https://github.com/fgnt/sed_scores_eval).
            audio_ids (iterable): audio ids that prediction should be restricted to.

        Returns (dict of pd.DataFrame):
            median filtered SED posterior score data frames for each audio_id

        """
        if isinstance(scores, (str, Path)):
            scores = io.read_sed_scores(scores)
        if audio_ids is None:
            audio_ids = scores.keys()
        scores = {audio_id: scores[audio_id] for audio_id in audio_ids}
        _, self.sound_classes = validate_score_dataframe(scores[list(audio_ids)[0]], event_classes=self.sound_classes)

        if isinstance(self.filter_length, dict):
            filter_len = np.array([self.filter_length[sound_class] for sound_class in self.sound_classes])
        else:
            assert np.isscalar(self.filter_length), self.filter_length
            filter_len = self.filter_length
        return medfilt(scores, filter_len)

    def detect(
        self,
        scores: Union[str, dict],
        detection_threshold: Union[float, dict, None] = None,
        audio_ids: Union[Iterable, None] = None,
        return_sed_scores: bool = False,
    ) -> dict:
        """perform median filtering and derive detections/decisions by employing detection_threshold

        Args:
            scores (str | dict of pandas.DataFrame): (path to directory with)
                SED posterior score data frames as used in sed_scores_eval
                (https://github.com/fgnt/sed_scores_eval).
            detection_threshold (float|dict|None): (class-wise) detection/decision thresholds.
                If None, self.detection_threshold is used.
            audio_ids (iterable): audio ids that prediction should be restricted to.
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

        median_filtered_scores = self.predict(scores, audio_ids)

        return self.detection_thresholding(
            median_filtered_scores,
            detection_threshold,
            return_sed_scores=return_sed_scores,
        )

    def detection_thresholding(
        self,
        median_filtered_scores: dict,
        detection_threshold: Union[float, dict, None] = None,
        return_sed_scores: bool = False,
    ) -> dict:
        """

        Args:
            median_filtered_scores (dict of pd.DataFrame): median filtered SED posterior scores for each audio_id
            detection_threshold (float|dict|None): (class-wise) detection/decision thresholds.
                If None, self.detection_threshold is used.:
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
        detections = scores_to_event_list(median_filtered_scores, detection_threshold, self.sound_classes)
        if return_sed_scores:
            detections = sed_scores_from_detections(
                detections,
                sound_classes=self.sound_classes,
            )
        return detections


def tune(
    scores: Union[str, dict],
    ground_truth: Union[str, dict],
    audio_durations: Union[str, dict],
    *,
    filter_lengths: Iterable = np.linspace(0.0, 2.0, 11),
    selection_fn: Callable,
    folds: Union[Iterable, None] = None,
    **selection_kwargs,
) -> Union[Tuple[MedianFilter, dict], list]:
    """perform grid search over hyper-parameters

    Args:
        scores (str | dict of pandas.DataFrame): (path to directory with)
            SED posterior score data frames as used in sed_scores_eval
            (https://github.com/fgnt/sed_scores_eval).
        ground_truth (str | dict of list): path to/dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio as used in
            sed_scores_eval.
        audio_durations (str | dict): (path to) durations of each audio file.
        filter_lengths (iterable): candidates for median filter lengths.
        selection_fn (callable): function that takes list of
            (CSEBBsPredictor, Sebb predictions) tuples together with
            ground truth, audio durations, the audio ids to be considered and
            further optional arguments and selects best CSEBBsPredictor parameters.
        folds (list|None): optional list of audio_id sets for which best parameters
            should be selected separately in which case a list of
            CSEBBsPredictors is returned. If None, best parameters are selected
            based on the whole set and a single CSEBBsPredictor is returned.
        **selection_kwargs: any key-word arguments that should be forwarded to selection_fn.

    Returns: (list of) return(s) from selection_fn

    """
    if isinstance(scores, (str, Path)):
        scores = io.read_sed_scores(scores)
    if isinstance(ground_truth, (str, Path)):
        ground_truth = io.read_ground_truth_events(ground_truth)
    if isinstance(audio_durations, (str, Path)):
        audio_durations = io.read_audio_durations(audio_durations)
    median_filtered_scores = []
    for filt_len in filter_lengths:
        median_filtered_scores.append((filt_len, medfilt(scores, filt_len)))
    if folds is None:
        return selection_fn(median_filtered_scores, ground_truth, audio_durations, **selection_kwargs)
    return [
        selection_fn(median_filtered_scores, ground_truth, audio_durations, audio_ids=fold, **selection_kwargs)
        for fold in folds
    ]


def cross_validation(
    scores: Union[str, dict],
    ground_truth: Union[str, dict],
    audio_durations: Union[str, dict],
    folds: Iterable,
    *,
    filter_lengths: Iterable = np.linspace(0.0, 2.0, 11),
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
        filter_lengths (iterable): candidates for median filter lengths.
        selection_fn (callable): function that takes list of
            (CSEBBsPredictor, Sebb predictions) tuples together with
            ground truth, audio durations, the audio ids to be considered and
            further optional arguments and selects best CSEBBsPredictor parameters.
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

    median_filters = tune(
        scores,
        ground_truth,
        audio_durations,
        folds=val_ids,
        filter_lengths=filter_lengths,
        selection_fn=selection_fn,
        **selection_kwargs,
    )
    median_filtered_scores = {}
    detections = {}
    for i, predictors in enumerate(median_filters):
        if not isinstance(predictors, dict):
            predictors = {"": predictors}
        for key, predictor in predictors.items():
            if isinstance(predictor, tuple):
                predictor = predictor[0]
            assert isinstance(predictor, MedianFilter), "selection_fn doesn't seem to have returned a CSEBBsPredictor."
            median_filtered_scores_i = predictor.predict(
                scores,
                audio_ids=test_ids[i],
            )
            if key not in median_filtered_scores:
                median_filtered_scores[key] = {}
                detections[key] = {}
            median_filtered_scores[key].update(median_filtered_scores_i)
            if predictor.detection_threshold is not None:
                detections_i = predictor.detection_thresholding(
                    median_filtered_scores_i,
                    return_sed_scores=return_sed_scores,
                )
                detections[key].update(detections_i)
    if len(median_filtered_scores) == 1 and "" in median_filtered_scores:
        median_filtered_scores = median_filtered_scores[""]
        detections = detections[""]
    return median_filters, median_filtered_scores, detections


def select_best_psds(
    median_filtered_scores: list,
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
) -> Tuple[MedianFilter, dict]:
    """select parameters which give highest PSDS

    Args:
        median_filtered_scores (list): list of (filter length, filtered scores) provided from tune function above
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
        median_filter: MedianFilter with best hyper-parameters w.r.t. PSDS
        best_values: class-wise PSDS values with best hyper-parameters

    """
    if audio_ids is not None:
        median_filtered_scores = [
            (filter_len, {audio_id: scores_i[audio_id] for audio_id in audio_ids})
            for filter_len, scores_i in median_filtered_scores
        ]
        ground_truth = {audio_id: ground_truth[audio_id] for audio_id in audio_ids}
        audio_durations = {audio_id: audio_durations[audio_id] for audio_id in audio_ids}

    best_filter_length = {}
    best_values = {}
    for filter_len, scores_i in median_filtered_scores:
        single_class_psds = intersection_based.psds(
            scores=scores_i,
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
                best_filter_length[sound_class] = filter_len

    median_filter = MedianFilter(
        filter_length=best_filter_length,
    )
    return median_filter, best_values


def select_best_cbf(
    median_filtered_scores: list,
    ground_truth: dict,
    audio_durations: dict,
    audio_ids: Union[Iterable, None] = None,
    onset_collar: float = 0.2,
    offset_collar: float = 0.2,
    offset_collar_rate: float = 0.2,
    classwise: bool = True,
) -> Tuple[MedianFilter, dict]:
    """select parameters which give highest collar-based F1-score

    Args:
        median_filtered_scores (list): list of (filter length, filtered scores) provided from tune function above
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
        median_filter: MedianFilter with best hyper-parameters w.r.t. collar-based F1-score
        best_values: class-wise collar-based F1-score values with best hyper-params

    """
    if audio_ids is not None:
        median_filtered_scores = [
            (filter_len, {audio_id: scores_i[audio_id] for audio_id in audio_ids})
            for filter_len, scores_i in median_filtered_scores
        ]
        ground_truth = {audio_id: ground_truth[audio_id] for audio_id in audio_ids}
        audio_durations = {audio_id: audio_durations[audio_id] for audio_id in audio_ids}

    best_filter_length = {}
    best_detection_threshold = {}
    best_values = {}
    for filter_len, scores_i in median_filtered_scores:
        f, _, _, thresholds_cbf, _ = collar_based.best_fscore(
            scores=scores_i,
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
                best_filter_length[sound_class] = filter_len
                best_detection_threshold[sound_class] = thresholds_cbf[sound_class]

    median_filter = MedianFilter(
        filter_length=best_filter_length,
        detection_threshold=best_detection_threshold,
    )
    return median_filter, best_values


def select_best_psds_and_cbf(
    median_filtered_scores: list,
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
        median_filtered_scores (list): list of (filter length, filtered scores) provided from tune function above
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

    Returns: dict of (tuned MedianFilter, class-wise performance dict) tuples
        for both PSDS and collar-based F1-score metrics

    """
    return {
        "psds": select_best_psds(
            median_filtered_scores,
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
            median_filtered_scores,
            ground_truth,
            audio_durations,
            audio_ids=audio_ids,
            onset_collar=onset_collar,
            offset_collar=offset_collar,
            offset_collar_rate=offset_collar_rate,
            classwise=classwise,
        ),
    }
