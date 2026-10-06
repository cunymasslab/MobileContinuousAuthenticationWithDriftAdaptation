# Drift Detection Analysis — Design Sketch
# Date: April 15, 2026

## Question
Does the raw sensor data itself change between sessions 1-3 and session 4,
independent of any model? If yes, we can confirm that the EER degradation
observed in Experiment 2 is caused by a real input distribution shift,
not by model limitations.

## Data
For each of the 50 (user, train_condition, stream_condition) combinations
from Experiment 2, I already have:
- Training windows from sessions 1-3 (the 2/3 training split)
- Session 4 windows from the stream condition

Each window is 100 timesteps x 12 features (accelerometer, gyroscope,
magnetometer, each with x/y/z plus magnitude).

To make KS and the classifier work on per-window summaries rather than
raw timestep data, I will compute the mean of each feature across the
100 timesteps within each window. This produces one 12-dimensional
vector per window. The mean captures the central tendency of each
sensor channel within the window, which is the level at which
distribution shift is most interpretable.

## Algorithm 1: Kolmogorov-Smirnov Test (per feature)

For each of the 12 features independently:
- Collect all window-mean values for that feature from sessions 1-3
- Collect all window-mean values for that feature from session 4
- Run scipy.stats.ks_2samp to get the KS statistic and p-value

Output per (user, condition pair): 12 KS statistics, 12 p-values.

Summary metrics:
- n_significant: how many of the 12 features have p < 0.05
- mean_ks_stat: average KS statistic across all 12 features
- max_ks_stat: largest KS statistic across the 12 features

Interpretation:
- n_significant near 12 = most sensor channels shifted
- n_significant near 0 = input distribution looks stable
- mean_ks_stat tells you the magnitude of the shift, not just
  whether it is statistically detectable

## Algorithm 2: Classifier-Based Detector

For each (user, condition pair):
- Take all window-mean vectors from sessions 1-3, label them 0
- Take all window-mean vectors from session 4, label them 1
- Combine into one dataset, shuffle
- Train a logistic regression with 5-fold cross-validation
- Record the mean CV accuracy

Output per (user, condition pair): one accuracy number.

Interpretation:
- Accuracy near 0.50 = classifier cannot tell the two apart,
  distributions are similar
- Accuracy near 1.00 = classifier easily distinguishes them,
  distributions are very different
- Accuracy above 0.70 = meaningful distributional difference

Why logistic regression: simple, fast, interpretable, does not
overfit easily on small datasets. If LR already gets high accuracy,
a more complex model would too, so LR is a conservative choice.

## Output Table

One CSV file: results_drift_detection.csv

Columns:
- target_user
- train_condition
- stream_condition
- n_significant_ks (out of 12)
- mean_ks_stat
- max_ks_stat
- classifier_accuracy
- drift_gap (copied from Experiment 2 results, for correlation)

## Validation Plan

Run on 6 users first:
- 3 high-drift users from Experiment 2: 525584, 717868, 796581
- 3 low-drift users from Experiment 2: 171538, 657486, 663153

Expected outcome: high-drift users should show high n_significant_ks,
high mean_ks_stat, and high classifier_accuracy. Low-drift users
should show lower values on all three. If this pattern holds,
scale to all 50 users.

If the pattern does NOT hold (e.g., high-drift users show low KS
statistics), stop and investigate before scaling. It would mean
the model performance gap is not explained by input distribution
shift, which would complicate the paper's drift narrative.

## What This Does NOT Cover

- Temporal ordering of drift (when does the shift happen within
  session 4). That would require a streaming detector like ADWIN.
- Multivariate interaction effects beyond what logistic regression
  captures. Could add a random forest later if needed.
- Causal explanation of why the distribution shifts. The KS test
  and classifier tell you THAT it shifted, not WHY.