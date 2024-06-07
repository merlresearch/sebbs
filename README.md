<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Sound Event Bounding Boxes (SEBBs)

A package for the prediction of sound event bounding boxes (SEBBs) as introduced in
> **Sound Event Bounding Boxes**
J. Ebbers, F. Germain, G. Wichern and J. Le Roux,
Accepted for publication at Interspeech 2024
https://arxiv.org/abs/2406.04212

SEBBs are one-dimensional bounding boxes defined by event onset time, event
offset time, sound class and a confidence.
They represent sound event candidates with a scalar confidence score assigned to it.
We call it (1d) bounding boxes to highlight the similarity to the (2d) bounding boxes
typically used for object detection in computer vision.

The final sound event detection can then be derived by a (class-wise)
event-level thresholding of the SEBBs' confidences.
Here, all SEBBs/candidates, whose confidence exceed the threshold, are accepted
as a detection, while discarding the rest.
The threshold can be used to control a system's sensitivity.

E.g., when a high sensitivity/recall (few missed hits) is required, a low
detection threshold can be used to detect events even when the system's
confidence is low.
When a high precision (few false alarms) is desired instead, a higher threshold
may be used to only detect events with high confidence.

With SEBBs the sensitivity of a system can be controlled without an impact on
the detection of an events' on- and offset times, which the previous frame-level
thresholding approach was suffering from.

## Table of contents

- [Installation](#installation)
- [Usage](#usage)
  - [Hyper-parameter tuning](#hyper-parameter-tuning)
  - [Inference](#inference)
  - [Paper results reproduction](#paper-results-reproduction)
- [DCASE 2024 Task 4](#dcase-2024-task-4)
- [Contributing](#contributing)
- [Copyright and license](#copyright-and-license)


## Installation

Install package directly
```bash
$ pip install git+https://github.com/merlresearch/sebbs.git
```
or clone and install (editable)
```bash
$ git clone https://github.com/merlresearch/sebbs.git
$ cd sebbs
$ pip install --editable .
```

## Usage

Currently, the package supports change-detection-based SEBBs (cSEBBs), which
are inferred by a post-processing of frame-level posterior scores.

### Hyper-parameter tuning
To perform hyper-parameter tuning of `step_filter_length`, `merge_threshold_abs`
and `merge_threshold_rel` simply call `sebbs.csebbs.tune`, which runs tuning and
returns a `sebbs.csebbs.CSEBBsPredictor` instance.
To tune hyper-parameters w.r.t. PSDS1 run
```python
from sebbs import csebbs

csebbs_predictor_psds, best_psds_values = csebbs.tune(
    scores="/path/to/validation/scores",
    ground_truth="/path/to/validation/ground_truth.tsv",
    audio_durations="/path/to/validation/audio_durations.tsv",
    step_filter_lengths=(.32, .48, .64),
    merge_thresholds_abs=(.15, .2, .3),
    merge_thresholds_rel=(1.5, 2., 3.),
    selection_fn=csebbs.select_best_psds,
    dtc_threshold=.7, gtc_threshold=.7,
    cttc_threshold=None, alpha_ct=0.,
)
```
Scores, ground_truth and audio_durations must be of the same format as used by
[sed_scores_eval](https://github.com/fgnt/sed_scores_eval) and can be either
paths or dictionaries with the (loaded) data.
An example of the required file format can be found in `tests/dcase2023task4a`.
To alternatively tune hyper-parameters w.r.t. collar-based F1-score run
```python
csebbs_predictor_cbf, best_cbf_values = csebbs.tune(
    scores="/path/to/validation/scores",
    ground_truth="/path/to/validation/ground_truth.tsv",
    audio_durations="/path/to/validation/audio_durations.tsv",
    step_filter_lengths=(.32, .48, .64),
    merge_thresholds_abs=(.15, .2, .3),
    merge_thresholds_rel=(1.5, 2., 3.),
    selection_fn=csebbs.select_best_cbf,
    onset_collar=.2, offset_collar=.2, offset_collar_rate=.2,
)
```
To tune hyper-parameters for both PSDS and collar-based F1-score in the same grid-search run
```python
csebbs_predictors = csebbs.tune(
    scores="/path/to/validation/scores",
    ground_truth="/path/to/validation/ground_truth.tsv",
    audio_durations="/path/to/validation/audio_durations.tsv",
    step_filter_lengths=(.32, .48, .64),
    merge_thresholds_abs=(.15, .2, .3),
    merge_thresholds_rel=(1.5, 2., 3.),
    selection_fn=csebbs.select_best_psds_and_cbf,
    dtc_threshold=.7, gtc_threshold=.7,
    cttc_threshold=None, alpha_ct=0.,
    onset_collar=.2, offset_collar=.2, offset_collar_rate=.2,
)
csebbs_predictor_psds, best_psds_values = csebbs_predictors['psds']
csebbs_predictor_cbf, best_cbf_values = csebbs_predictors['cbf']
```
You can also implement a custom selection_fn to define the criterion for
hyper-parameter selection yourself.

We also provide median filter baseline code following the same API, which can,
e.g., be tuned as:
```python
import numpy as np
from sebbs import median_filter

median_filter_psds, best_psds_values = median_filter.tune(
    scores="/path/to/validation/scores",
    ground_truth="/path/to/validation/ground_truth.tsv",
    audio_durations="/path/to/validation/audio_durations.tsv",
    filter_lengths=np.linspace(0.,2.,11),
    selection_fn=median_filter.select_best_psds,
    dtc_threshold=.7, gtc_threshold=.7,
    cttc_threshold=None, alpha_ct=0.,
)
```

### Inference

To run inference with a given instance `csebbs_predictor` call:
```python
csebbs_predictions = csebbs_predictor.predict("/path/to/test/scores")
```
To make the method directly return sebbs as sed scores data frames for direct
use with [sed_scores_eval](https://github.com/fgnt/sed_scores_eval) call
```python
csebbs_sed_scores = csebbs_predictor.predict("/path/to/test/scores", return_sed_scores=True)
```
Then, test performance in terms of, e.g., PSDS1 can be obtained as:
```python
psds, single_class_psds, *_ = sed_scores_eval.intersection_based.psds(
    scores=csebbs_sed_scores,
    ground_truth="/path/to/test/ground_truth.tsv",
    audio_durations="/path/to/test/audio_durations.tsv",
    dtc_threshold=.7, gtc_threshold=.7,
    alpha_ct=.0, alpha_st=1., unit_of_time='hour', max_efpr=100.,
)
```

### Paper results reproduction
To reproduce a system's cross-validation performance from our paper run:
```python
import numpy as np
from sebbs import csebbs, median_filter, package_dir
from sed_scores_eval import io, intersection_based, collar_based

# load predictions, ground_truth, and audio_durations
scores = io.read_sed_scores("/path/to/dcase2023task4a/system/scores")
ground_truth = io.read_ground_truth_events(
    f"{package_dir}/tests/dcase2023task4a/metadata/ground_truth.tsv")
scores = {audio_id: scores[audio_id] for audio_id in ground_truth}
audio_durations = io.read_audio_durations(
    f"{package_dir}/tests/dcase2023task4a/metadata/audio_durations.tsv")
audio_durations = {audio_id: audio_durations[audio_id] for audio_id in ground_truth}

# 5-fold cross validation split
n_folds = 5
all_keys = scores.keys()
shuffled_keys = sorted(all_keys)
np.random.RandomState(0).shuffle(shuffled_keys)
split_idx = np.linspace(0, len(shuffled_keys), n_folds + 1).astype(int).tolist()
folds = [set(shuffled_keys[split_idx[i]:split_idx[i + 1]]) for i in range(n_folds)]

# median_filter cross validation
median_filters, mf_scores, mf_detections = median_filter.cross_validation(
    scores=scores, ground_truth=ground_truth,
    audio_durations=audio_durations, folds=folds,
    filter_lengths=np.linspace(0., 2., 11),
    selection_fn=median_filter.select_best_psds_and_cbf,
    return_sed_scores=True,
)
# evaluate PSDS performance
psds_mf, single_class_psds_mf, *_ = intersection_based.psds(
    scores=mf_scores['psds'], ground_truth=ground_truth,
    audio_durations=audio_durations,
    dtc_threshold=.7, gtc_threshold=.7,
    alpha_ct=.0, alpha_st=1., unit_of_time='hour', max_efpr=100.,
)
# evaluate collar-based F1-score performance
f_mf, *_ = collar_based.fscore(
    scores=mf_detections['cbf'], ground_truth=ground_truth,
    threshold=.5,  # can be any number between 0 and 1 here as the sed_scores obtained from detection are binary
    onset_collar=.2, offset_collar=.2, offset_collar_rate=.2,
)

# csebbs cross validation
csebbs_predictors, csebbs_scores, csebbs_detections = csebbs.cross_validation(
    scores=scores, ground_truth=ground_truth,
    audio_durations=audio_durations, folds=folds,
    step_filter_lengths=(.32, .48, .64),
    merge_thresholds_abs=(.15, .2, .3),
    merge_thresholds_rel=(1.5, 2., 3.),
    selection_fn=csebbs.select_best_psds_and_cbf,
    return_sed_scores=True,
)
# evaluate PSDS performance
psds_csebbs, single_class_psds_csebbs, *_ = intersection_based.psds(
    scores=csebbs_scores['psds'], ground_truth=ground_truth, audio_durations=audio_durations,
    dtc_threshold=.7, gtc_threshold=.7,
    alpha_ct=.0, alpha_st=1., unit_of_time='hour', max_efpr=100.,
)
# evaluate collar-based F1-score performance
f_csebbs, *_ = collar_based.fscore(
    scores=csebbs_detections['cbf'], ground_truth=ground_truth,
    threshold=.5,  # can be any number between 0 and 1 here as the sed_scores obtained from detection are binary
    onset_collar=.2, offset_collar=.2, offset_collar_rate=.2,
)
```

## DCASE 2024 Task 4
To facilitate the usage of cSEBBs for DCASE 2024 Task 4, we provide an example
script [dcase2024.py](./scripts/dcase2024.py) showing how to tune
hyper parameters and infer output scores for
[DCASE 2024 Task 4 Baseline](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2024_task4_baseline).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## Copyright and license

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as listed below:
```
Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```

Files in `tests/dcase2023task4a/baseline2_run1_scores` were copied from [Submissions DCASE 2023 Task4a](https://zenodo.org/records/8248775):

```
Copyright (C) 2023, Janek Ebbers, Romain Serizel, Francesca Ronchini, Florian Angulo, David Perera, Slim Essid

SPDX-License-Identifier: CC-BY-4.0
```

Files in `tests/dcase2023task4a/metadata` were adapted from [Evaluation set DCASE 2021 task 4](https://zenodo.org/records/5524373):

```
Copyright (C) 2021, Francesca Ronchini, Nicolas Turpault, Romain Serizel, Scott Wisdom, Hakan Erdogan, John Hershey, Justin Salamon, Prem Seetharaman, Eduardo Fonseca, Samuele Cornell, Daniel P. W. Ellis

SPDX-License-Identifier: CC-BY-4.0
```

Content of `scripts/utils.py` was copied from [DCASE 2024 Task 4 Baseline](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2024_task4_baseline):

```
Copyright (C) 2024, Romain Serizel

SPDX-License-Identifier: MIT
```
