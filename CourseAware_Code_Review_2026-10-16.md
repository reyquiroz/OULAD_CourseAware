# CourseAware: Review of the Week 14–15 Code and Results

Xuan Guo, October 16, 2026

Rey,

I reviewed the Week 14–15 report, the updated results matrix, and the current code in LabRepo after pulling the latest version. I read the source code, tests, LS6 job scripts, result CSVs, and the graph artifact metadata. I also reran some of the LightGBM comparisons myself, using your Week 8 graph artifacts and your `random_student_split` function. I did not rerun the GNN models or the test suite, so my comments on the GNN are based on reading the code.

Overall, the data handling is careful, and I did not find label leakage. However, I do not think we can yet conclude that GraphSAGE outperforms LightGBM. Below I explain why, and why I think the results point to a clearer direction for the paper.

## 1. The comparison is not yet matched

### 1.1 The models do not receive the same inputs

`run_matched_comparison.py` gives LightGBM and the MLP the six behavioral summaries, `num_of_prev_attempts`, `studied_credits`, and `age_band`. GraphSAGE receives more than that:

* the student node carries gender, region, highest education, IMD band, and disability;
* the course-presentation and VLE-resource nodes carry one-hot `code_module` and `code_presentation`, so the model knows which course and which presentation each enrollment belongs to.

When I gave LightGBM the same inputs, most of the gap disappeared. All results below are for Week 8, LightGBM with 200 trees, your five seeds (42, 123, 7, 17, 99), and the 22 LCPO folds.

| Inputs given to LightGBM | Random AUROC | Random AUPRC | LCPO AUROC |
| --- | --- | --- | --- |
| A. Current matched set (six summaries, previous attempts, credits, age) | 0.857 ± 0.004 | 0.891 | 0.828 ± 0.077 |
| B. A + demographics | 0.862 ± 0.002 | 0.894 | 0.833 ± 0.078 |
| C. A + course identity | 0.873 ± 0.003 | 0.906 | 0.838 ± 0.078 |
| D. A + demographics + course identity | 0.876 ± 0.002 | 0.908 | 0.846 ± 0.069 |
| E. D + the four temporal features | 0.882 ± 0.002 | 0.913 | 0.855 ± 0.062 |
| Best GraphSAGE (temporal variant), for reference | 0.888 ± 0.002 | 0.916 | not run |
| Base GraphSAGE, for reference | 0.838 | 0.881 | 0.788 ± 0.065 |

Row A reproduces your matched LightGBM result (0.857), so the rebuilt features agree with your pipeline. In my LCPO runs, I trained on all 21 other presentations without holding out a validation set. That differs slightly from your setup but does not affect AUROC or AUPRC. The script I used, `lgbm_diag.py`, is attached; it takes the path to LabRepo as its only argument.

### 1.2 Students who had already withdrawn are still included

Nothing in the pipeline filters on `studentRegistration.date_unregistration`. By the Week 8 cutoff (day 56), 6,071 of the 32,593 enrollments (18.6%) had already unregistered, and 99.9% of them are labeled Withdrawn. Of these, 3,097 unregistered before the course started. Their outcome is already known at prediction time, so they inflate every model's score. When I removed them, matched LightGBM dropped from 0.857 to 0.789, and row E dropped from 0.882 to 0.821.

Please exclude enrollments with `date_unregistration` on or before the cutoff, separately for each prediction week, and report how many are removed. All results will need to be rerun after this change.

### 1.3 The GraphSAGE variants are probably not one change at a time

In `gnn_model.py`, `GraphDataLoader.load()` (around line 552) adds enrollment nodes and their four edge types whenever `week08_nodes_enrollment.parquet` exists on disk. `EnrollmentGNN` always registers these edge types, so they take part in message passing even when the condition is not `enrollment_node`. The current Week 8 metadata shows that the artifacts include enrollment nodes. As a result, the six-summaries, edge-aware, and temporal variants likely all ran with enrollment nodes. I cannot tell from the repository which runs came before or after the artifacts were rebuilt. However, four different variants falling between 0.883 and 0.888 is consistent with this.

Please make enrollment nodes an explicit option in the loader, rebuild the artifacts, and rerun the variants as a cumulative sequence, one change at a time.

### 1.4 LCPO results for different conditions would overwrite each other

In `run_gnn_experiment.py`, `lcpo_results.csv` and `lcpo_summary.csv` are deduplicated on `week`, `fold_idx`, and `model_seed` only. `condition` is not part of the key, and the first write happens before the `condition` column is added. Running LCPO for any variant would therefore overwrite the base model's rows. The current file contains only the base model (Weeks 2–6 labeled `full`, Week 8 unlabeled), so none of the variants has been evaluated under LCPO yet.

Please add `condition` to the key and to every record before writing.

### 1.5 The report used the historical LightGBM, not the matched one

The complete matched results already exist in `results/results/matched/`: five seeds and four weeks for the random split, plus all 22 LCPO folds, for both LightGBM and the MLP. Matched LightGBM reaches 0.857 (random) and 0.826 (LCPO) at Week 8, whereas the report uses the historical values of 0.863 and 0.835. The single matched value in the matrix (0.856, no standard deviation) comes from an older one-seed file in `results/matched/`.

Compared fold by fold, matched LightGBM outperforms base GraphSAGE under LCPO by 0.038 AUROC at Week 8 (18 of 22 folds; Wilcoxon signed-rank p = 0.003), and by about 0.03 at Weeks 4 and 6. At Week 2, the difference (0.006) is not significant.

### 1.6 The Zenodo GraphSAGE has almost no usable input

In `run_zenodo_pipeline.py`, the student features and enrollment attributes are constants (1.0). The interaction edge attributes are built but never used, because `use_interacted_with_attrs` defaults to False. Since SAGEConv uses mean aggregation, the model cannot see how much a student interacted. It only sees the mix of resource types the student touched. The near-random result therefore reflects the model setup rather than differences between institutions. Also, these experiments train and test within Zenodo, so they do not test transfer from OULAD.

LightGBM results for Zenodo already exist in `results/zenodo/comparison_results.csv` (about 0.65 random and 0.61 LCPO, using only three VLE summaries). Please add them to the matrix.

## 2. What this means for the research

I think these findings are useful for the paper. GraphSAGE's advantage on the random split appears to come largely from knowing which course each enrollment belongs to. At-risk rates differ considerably across courses, so a course ID works as a lookup of each course's base rate. Under the random-student split, every test student is in a course the model has already seen, so this lookup helps. In my LightGBM runs, course identity gave the largest single gain (+0.016 AUROC).

For a course the model has never seen, however, the course ID carries no learned information. This is consistent with your results, where GraphSAGE wins on the random split and loses under LCPO. Note that LCPO holds out one presentation, not a whole module, so other presentations of the same module remain in training and the ID is still partly informative. A leave-module-out evaluation would remove that.

This is exactly why our contribution needs to be course-design-aware. To generalize to new courses, the model should describe a course by its design rather than by its name, for example:

* the number, types, and weights of its assessments;
* its assessment schedule;
* its length;
* its mix of learning resources.

Course-conditioned prediction should then interpret student behavior relative to that design. For example, 50 clicks by Week 4 may be low in a resource-heavy course but normal in a lighter one.

I would like you to test this directly by comparing three ways of representing the course under three evaluation settings:

| Course information given to the model | Random-student split | LCPO (new presentation) | Leave-module-out (new course) |
| --- | --- | --- | --- |
| None | baseline | baseline | baseline |
| Course ID only | expected to help | expected to help a little | expected not to help |
| Course design features | expected to help | expected to help | expected to help |

Please run this for both LightGBM and GraphSAGE, on still-enrolled students only. If course design features help on new courses where the course ID does not, that would be the central result of the paper. At this point it is a hypothesis supported by the diagnostic above, not yet a finding.

## 3. Smaller issues

* In `graph_pipeline.py`, enrollments with no VLE activity receive `recency = 0` through `fillna(0.0)`, which reads as "active on the cutoff day." Please set it to the window length plus one, or add an explicit no-activity indicator.
* The "− temporal features" ablation zeroes `first_day`, `last_day`, and `active_days` on the interaction edges, plus the assessment due date. That is not the reverse of the "+ temporal features" variant, which adds four different features. All ablations were also run on the base model (0.837) rather than the best model. Please label these clearly.
* Each GraphSAGE variant has 10 rows (5 seeds × weighted and unweighted loss), and the reported mean ± SD pools both. Please report one loss setting, or both separately.
* The base GraphSAGE (`full`) has only 4 seeds; seed 42 is missing at all weeks.
* `results/` and `results/results/` contain different versions of the same files. For example, the copy of `random_student_results.csv` in `results/results/graph/` includes a temporal run with AUROC 0.824 that stopped at epoch 4. The CSVs can serve as the source of truth only if there is one results folder.
* LCPO training rarely stops early: the average best epoch is about 184 out of 200, so these models may be undertrained.
* Enrollment-node features are standardized using all enrollments, including test enrollments. The effect is small because no labels are involved, but it should follow the training-only rule used elsewhere.

## 4. What appears to be correct

* Labels: at-risk is Fail or Withdrawn, assigned per enrollment, and the label is never used as a feature.
* Time cutoffs: clicks are counted up to the cutoff day, and an assessment is counted only if both its due date and its submission date fall on or before the cutoff.
* The six behavioral features are computed per enrollment (student × module × presentation) in both the tabular and graph pipelines.
* The random-student split keeps each student entirely in train, validation, or test, and all models use the same function and seeds.
* All models use the same 22 LCPO folds. During GNN training, held-out assessment, resource, and course edges are removed, and held-out enrollments receive no loss.
* The MLP scaler and the GNN student, enrollment-edge, and interaction statistics are fit on training rows only, with the exception noted above.
* Only training enrollments and their students' edges are present in the GNN training graph.
* Thresholds are selected by maximizing F1 on the validation set, with the same sweep for all models.

## 5. Next steps

Please first fix the evaluation:

1. Exclude enrollments that unregistered by each cutoff, and report the number removed per week.
2. Give LightGBM and the MLP the same inputs as GraphSAGE (demographics, course identity, and temporal features), and also run a version without course identity.
3. Make enrollment nodes an explicit loader option, rebuild the artifacts, and rerun the variants one change at a time.
4. Add `condition` to the LCPO keys, then run LCPO for every variant at all four weeks.
5. Run the MLP with the six behavioral summaries only.
6. Fix `recency` for inactive enrollments.
7. Keep a single results folder and fill the matrix from it, using matched LightGBM rather than historical LightGBM in all tables.
8. Report LCPO comparisons as paired, fold-level differences with a significance test, since the variation across folds is larger than the differences between models.

Then, please test the research hypothesis:

9. Add a leave-module-out evaluation alongside LCPO.
10. Run the course-information experiment (none, course ID, course design features) for LightGBM and GraphSAGE under all three evaluation settings.
11. For Zenodo, add the existing LightGBM results to the matrix, give the GNN the interaction edge attributes or use sum aggregation, and document the class balance and graph statistics I asked for earlier.

Please introduce these changes one at a time and keep the intermediate results, so that we can tell which change produces each improvement.
