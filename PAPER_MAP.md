# What the paper reports, and where each number comes from

Each number in the manuscript is traced here to the result file that holds it.
Entries marked **verified** were checked value by value against that file.

## Results

| Section | Analysis reported | Result files |
|---|---|---|
| 3.1.1 Prompt effects | MAE_o / MAE_b per prompt, prompt contrasts, the *Detailed* construct mismatch | `Q1_metrics_by_prompt`, `Q1_contrasts_prompts` **verified** |
| 3.1.2 Model accuracy | MAE_o / MAE_b per model, with and without *Detailed*; zero rates | `Q1_metrics_by_model`, `Q1_metrics_by_model_no_detail`, `Q1_contrasts_models_no_detail`, `Q8_zero_rates` **verified** |
| 3.1.3 Configurations | 24-configuration ranking, top-set, P(best) | `Q1_metrics_by_combo`, `Q1_topset_bootstrap` **verified** |
| 3.1.4 Cover bins | per-bin MAE and signed bias, Jonckheere-Terpstra trend | `Q3_perbin_error`, `Q3_trend_tests`, `Q1_metrics_perbin` **verified** |
| 3.2 Scene and geometry | covariate model, block tests, obliquity structure, rooted-level error | `Q3_covariate_models`, `Q3_covariate_block_tests`, `Q3_obliquity_structure`, `Q3_rooted_level_error` **verified** |
| 3.3 Preprocessing | crop and rectification effects, per axis and per bin | `Q4_variant_by_axis`, `Q4_variant_contrasts`, `Q4_variant_nesting`, `Q4_obliquity_variant` **verified** |
| 3.4 Confidence | Spearman vs error, ROC and operating points, pairwise scene association, threshold sweep | `Q5_rank_association`, `Q5_rank_association_by_axis`, `Q5_roc_by_axis`, `Q5_roc_operating_points`, `Q5_feature_association`, `Q5_feature_association_by_axis`, `Q5_auc_by_model`, `Q5_confidence_filtering` **verified** |
| 3.5 Index baseline | per-bin MAE and bias, Pearson r, comparison against the top-set | `Q9_baseline_diagnostic`, `Q0_baseline_index_summary` **verified**. `Q9_contrasts_vs_baseline` is *not* a source: no value it holds appears anywhere in the manuscript |
| 3.6 Ensembles | 66 configurations, 42 ensembles, held-out selection | `Q2_configurations`, `Q2_selection_stability` **verified** |
| 3.7.1 Determinism | 8 of 100 images differing, 0.65 floor | `Q1_determinism_floor` **verified** |
| 3.7.2 Reasoning text | share of responses carrying prose, per model and prompt | **no result file**; see Known gaps |
| 3.7.3 API pathway | exact agreement, median difference, Kendall tau_b, rank reversals | `Q7_pathway_agreement`, `Q7_pathway_by_configuration`, `Q7_ranking_agreement`, `Q7_rank_reversals` **verified** |
| 3.8 Computational demand | MAE_b for the Gemma pair; weights and latency | `Q1_metrics_by_model` for the MAE values; hardware figures have **no result file** |

## Discussion

Section 4 restates Results values and introduces no new analysis, with one
exception: 4.6.3 Image Acquisition relies on the device-by-campaign structure in
`Q6_device_campaign_crosstab` and `Q6_identifiability_demonstration`.

## Files with no reported number

The remaining files in `results/` are not cited anywhere in the manuscript. That
includes all `*_assumption_checks` and all `*_clustered_companion` files, the Q2
cross-validation detail, and the Q3/Q4 interaction and per-configuration files.

They are shipped because each answers a methodological question a reader may
reasonably ask. The `*_assumption_checks` files are the record that each test's
assumptions were checked. The `*_clustered_companion` files measure what the
image-level independence assumption costs, by refitting with the campaigns
treated as clusters. The paper reports the image-level intervals; the clustered
refits are shipped beside them so that the cost of that assumption is on the
record rather than asserted to be small.

## Library

Every notebook runs on one shared library, `notebooks/_common.py`, so that no
two questions can drift apart on a data loader, a bin definition, a bootstrap or
a correction that has to be identical across all of them.

## Known gaps

- **Reasoning text (3.7.2 and Table 5)** is computed by no notebook and held in
  no result file. Those numbers appear only in the manuscript. They are
  recoverable from the response logs and would need a notebook of their own if
  that section stays.
- **Hardware figures in 3.8** (weight footprints, latency) are deployment
  metadata, not analysis output.
