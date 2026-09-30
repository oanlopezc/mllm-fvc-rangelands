# What the paper reports, and where each number comes from

Each number in the Results section is traced here to the result file that holds it.

## Results

| Section | Analysis reported | Result files |
|---|---|---|
| 3.1.1 Prompt effects | MAE_o / MAE_b per prompt, prompt contrasts | `Q1_metrics_by_prompt`, `Q1_contrasts_prompts` |
| 3.1.2 Model accuracy | MAE_o / MAE_b per model, with and without *Detailed*; zero rates | `Q1_metrics_by_model`, `Q1_metrics_by_model_no_detail`, `Q1_contrasts_models_no_detail`, `Q8_zero_rates` |
| 3.1.3 Configurations | 24-configuration ranking, top-set, P(best) | `Q1_metrics_by_combo`, `Q1_topset_bootstrap` |
| 3.1.4 Cover bins | per-bin MAE and signed bias, Jonckheere-Terpstra trend | `Q3_perbin_error`, `Q3_trend_tests`, `Q1_metrics_perbin` |
| 3.2 Scene and geometry | covariate model, block tests, obliquity structure, rooted-level error | `Q3_covariate_models`, `Q3_covariate_block_tests`, `Q3_obliquity_structure`, `Q3_rooted_level_error` |
| 3.3 Preprocessing | crop and rectification effects, per axis and per bin | `Q4_variant_by_axis`, `Q4_variant_contrasts`, `Q4_variant_nesting`, `Q4_obliquity_variant` |
| 3.4 Confidence | Spearman vs error, ROC and operating points, pairwise scene association, threshold sweep | `Q5_rank_association`, `Q5_rank_association_by_axis`, `Q5_roc_by_axis`, `Q5_roc_operating_points`, `Q5_feature_association`, `Q5_feature_association_by_axis`, `Q5_auc_by_model`, `Q5_confidence_filtering` |
| 3.5 Index baseline | per-bin MAE and bias, Pearson r, comparison against the top-set | `Q9_baseline_diagnostic`, `Q0_baseline_index_summary` |
| 3.6 Ensembles | 66 configurations, 42 ensembles, held-out selection | `Q2_configurations`, `Q2_selection_stability` |
| 3.7.1 Determinism | 8 of 100 images differing, 0.65 floor | `Q1_determinism_floor` |
| 3.7.3 API pathway | exact agreement, median difference, Kendall tau_b, rank reversals | `Q7_pathway_agreement`, `Q7_pathway_by_configuration`, `Q7_ranking_agreement`, `Q7_rank_reversals` |
