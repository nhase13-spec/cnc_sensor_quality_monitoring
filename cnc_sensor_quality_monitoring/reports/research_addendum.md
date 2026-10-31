# Research addendum (Project 2, CNC sensor monitoring): what changed and why

Scope: reviewer-requested controlled experiments (correlation filtering, training-fold-only PCA, physically selected
sensor groups, grouped importance). All baseline experiments and result files are unchanged; everything new is written
to new files (`results/tables/extended_*`, `results/figures/extended_*`, `results/metrics/extended_*`).
These generated files are not committed; reproduce them with `python run_pipeline.py --stage extended --reuse-features`
(not part of `--stage all`).

**Status of every number below: exploratory and hypothesis-generating.** 18 experiments (14 for visual inspection;
only 4 negatives), one row per experiment, leave-one-experiment-out (LOEO). 14 feature variants x 3 models x 3 tasks
were evaluated, so some good-looking cells are expected by chance. No variant was selected from outer-fold results.

## Validity checks run first
* The reference variant (original 864 features) reproduces the stored balanced accuracies exactly (max |diff| = 0).
* Same splitter and rows for every variant (identical outer folds, checked for all tasks). Imputer, scaler,
  correlation filter and PCA are fitted on the 17 (or 13) training experiments only (`tests/test_extended.py`; 30 tests pass,
  including a pure-noise canary where a deliberately leaky selector is exposed and the in-fold pipeline shows no skill).
* **Reproducibility finding.** The original random forest uses `n_jobs=-1`. Parallel averaging changes the floating-point
  summation order (~1e-16), and a predicted probability of exactly 0.5 (one such experiment in the tool-condition task) can then flip its
  label between runs, so a multithreaded forest's balanced accuracy can move by one experiment. The extended runner fits forests single-threaded
  (deterministic, bit-identical on repeat; `tests/test_extended.py`) and still reproduces every stored balanced accuracy exactly. The original
  `build_model` is unchanged.
* New variants use the 712-feature base (no `n_samples`, no duration-scaling energies, no program-progress channels).
  Never used as features: experiment id, targets, `feedrate`, `clamp_pressure`.
* PCA k = 1, 2, 3, 5, 8 capped at `n_train // 2` (8 for 18 experiments, 6 for the 14-experiment task); k = 8 is **not run**
  for visual inspection (`extended_skipped_variants.csv`). Each fold has effective rank 16, so the cap, not rank, binds.
* **Caveat that limits one task.** The 712-feature base is *not* fully length-free: spindle position features
  (`S1_ActualPosition`, `S1_CommandPosition`) track experiment length (Spearman 0.86-0.89 with `n_samples`) and alone reach
  balanced accuracy 0.89 (LR) / 0.96 (RF) for `machining_completed`. Removing them does not remove the effect (LR still 1.00,
  RF 0.88), so **`machining_completed` results, including "phys_spindle = 1.00", are not sensor-physics evidence.** For
  `tool_condition` and `visual_inspection` the position-only features show no skill (BA 0.35-0.65, every CI includes 0.5), so those tasks
  are not affected in this way (`extended_spindle_position_diagnostic.csv`). The base also contains 13 `M1_CURRENT_FEEDRATE` features
  (a setting, not a sensor response).

## A. Correlation filtering (heatmaps `extended_correlation_heatmap_*.png`)
* Of 709 non-constant features, median |r| = 0.27, 5% of pairs have |r| >= 0.8, 2.6% >= 0.9, 1.6% >= 0.95. With 18 experiments, sizeable |r| also
  arises by chance, so this redundancy is partly sampling noise. The rank limit is 17.
* An in-fold filter at |r| >= 0.95 keeps ~360 of 712 features; at 0.80 keeps ~167.
* Tool condition balanced accuracy (LR / RF / XGB), 95% bootstrap CI over experiments in the full table:
  base 712: 0.55 / 0.53 / 0.40; filter 0.95: 0.61 / 0.38 / 0.51; filter 0.80: 0.50 / 0.44 / 0.46. Chance = 0.50 (all intervals include it).
  The changes go both ways and no paired interval excludes 0 (`dBA_vs_base712*` columns).

## B-C. PCA after training-fold standardisation (`extended_pca_explained_variance.png`, `extended_performance_vs_pca_components.png`)
* Descriptive (all 18): 9 PCs for 90% and 11 for 95% of variance. Per fold, 16 non-zero components.
* Tool condition BA (LR / RF / XGB): k=1 0.34 / 0.34 / 0.16; k=2 0.56 / 0.44 / 0.26; k=3 0.56 / 0.50 / 0.45; k=5 0.44 / 0.39 / 0.47;
  k=8 0.42 / 0.28 / 0.42. No monotone trend, most cells <= chance. Mean change vs the 712-feature base across PCA cells: -0.09 (tool condition),
  +0.06 (visual inspection, led by k=1: RF 0.90 [0.75, 1.00] on 14 experiments with 4 negatives), -0.01 (machining completed).
* PCA made predictions **less** stable across fold models (jackknife SD of probability 0.03-0.08 vs 0.002-0.03 without PCA; many experiments
  flip label for k <= 3, e.g. LR k=1 flips 33%).

## D. Physically selected sensor groups (`extended_variant_comparison_balanced_accuracy.png`)
Tool condition BA (LR / RF / XGB): spindle 0.55 / 0.38 / 0.55; axis current 0.28 / 0.15 / 0.23; axis motion 0.49 / 0.70 / 0.41;
combined sensors 0.50 / 0.57 / 0.47; a-priori 18 motor-load features 0.39 / 0.54 / 0.45.
* Axis-current features are significantly **worse** than base for LR and RF (paired dBA -0.28 [-0.50, -0.08] and -0.38 [-0.61, -0.16]); a
  below-chance result under LOEO is a sign of no transferable signal (anti-learning on tiny sets), not of an inverse relation.
* The best-looking cell (RF on axis motion, 0.70 [0.49, 0.90]) is the maximum of 42 tool-condition cells; its paired difference to base is
  +0.17 [-0.12, +0.47] and its ROC AUC interval is [0.34, 0.97]. It was not selected and is **not** reported as a result.
* Across all 114 paired comparisons with the 712-feature base only 3 intervals exclude 0 (about 6 would be expected by chance), and all 3 are negative.

## E-F. Metrics and stability
Balanced accuracy, macro F1, pooled confusion matrices (`extended_confusion_matrices_<task>.png`, table `extended_confusion_matrices.csv`) and bootstrap
CIs are in `extended_comparison.csv`. Pooled confusion matrices for tool condition show the models mostly predicting "worn" (e.g. XGBoost 0 of 8
unworn correct). Stable is not correct: unfiltered LR / RF / XGB have near-zero label flips between fold models yet balanced accuracy near 0.5.
Top-20 feature overlap between fold models is 0.54 / 0.33 / 0.29 (LR / RF / XGB) for 712 features and 0.63 / 0.30 / 0.28 after the 0.95 filter - the
filter did not make the selected features more consistent for tree models. (The 18-feature set has overlap 1.0 trivially because the top 20 is the whole set.)

## D2. Grouped importance (`extended_grouped_importance_tool_condition.png`)
Out-of-fold permutation (Brier) and grouped SHAP (signed group sums; additivity gap 6e-16) for the tool-condition random forest. Largest groups: Z1
subsystem (+0.037 Brier), position signals (+0.026), location statistics (+0.030); several groups are ~0 or negative. The out-of-fold Brier of the
model is 0.2465 (about 0.25 = no information), so these values describe what a non-generalising model leans on, not demonstrated predictive signal. Larger
groups score higher partly because they contain more columns; correlated columns share credit. Position features may encode setup geometry rather than wear.

## What this changes in the original conclusions
Nothing is overturned: with 14-18 experiments, no sensor-feature variant yields a balanced accuracy that is reliably above chance for tool
condition, and none of the dimensionality-reduction or filtering steps reliably improved on the 712-feature base. Additional information: redundancy
is moderate and partly a small-sample effect; PCA reduced stability; the `machining_completed` task remains contaminated by length-like spindle position
features even after the earlier robustness filters. Independent new experiments are required before any claim of generalisation.
