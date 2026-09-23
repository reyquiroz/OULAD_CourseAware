# OULAD GraphSAGE — Interactive HTML Presentation Plan

## Top-Level Overview

Build a single, fully self-contained **reveal.js** slide deck (`presentation.html`) at the
repo root. The deck is navigated with arrow keys in any modern browser, works entirely
offline, and contains no external file dependencies at open time.

**Three confirmed design decisions (post-review):**
1. Split-strategy comparison (random vs. LCPO) is an **interactive Plotly chart**, not a static PNG.
2. All 35 static PNGs are **base64-embedded** inline in the HTML so the file is fully portable.
3. A dedicated interactive slide shows the **Zenodo cross-dataset comparison** (OULAD vs. KU Leuven).

### Deck structure at a glance

| # | Slide Title | Type |
|---|---|---|
| 1 | Title + metadata | Narrative |
| 2 | Project Motivation & Research Question | Narrative |
| 3 | Dataset & Prediction Task | Narrative |
| 4 | Graph Schema | Static PNG (base64) |
| 5 | Pipeline Overview | Narrative |
| 6 | Leakage Prevention — Dual Temporal Guard | Narrative |
| 7 | Split Strategies: Random vs LCPO | **Interactive Plotly** |
| 8 | GNN vs LightGBM — Random-Student Split | **Interactive Plotly** |
| 9 | GNN vs LightGBM — LCPO (22 folds) | **Interactive Plotly** |
| 10 | LCPO Per-Fold Variance (box plot) | **Interactive Plotly** |
| 11 | Ablation Study (7 conditions) | **Interactive Plotly** |
| 12 | Week-by-Week Baseline Performance | **Interactive Plotly** |
| 13 | Cross-Dataset: OULAD vs Zenodo (KU Leuven) | **Interactive Plotly** |
| 14 | Key Findings & Takeaways | Narrative |
| 15 | Open Problems & Next Steps | Narrative |
| A1–A35 | Appendix — All Static Figures | Static PNGs (base64) |

---

## Sub-Task 1 — Scaffold the reveal.js deck shell

**Status:** `[ ] pending`

**Intent**
Create `presentation.html` with the reveal.js and Plotly.js CDN includes, global CSS,
and one `<section>` placeholder per slide. No chart data or content yet.

**Expected Outcomes**
- `presentation.html` exists at repo root and opens in a browser
- Arrow-key navigation works across all placeholder slides
- Plotly.js CDN script tag present
- Slide numbers, progress bar, and hash-based bookmarks active

**Todo List**
1. Create `presentation.html` with reveal.js 3.9.0 CDN links (JS + default theme CSS)
2. Add Plotly.js CDN: `https://cdn.plot.ly/plotly-2.27.0.min.js`
3. Define CSS overrides matching the existing project palette:
   - Primary blue `#3b82d4`, purple `#7c5cd8`, danger red `#c0392b`, success green `#2ecc71`
   - Chart `<div>` containers must have explicit `height: 460px`
   - Caption/label font: 13px, color `#57606a`
4. Add all 15 main section `<section>` placeholders plus an appendix `<section>` wrapper
5. Set `Reveal.initialize({ slideNumber: 'c/t', progress: true, hash: true })`

**Relevant Context**
- Colour tokens sourced from `graphsage-at-risk-prediction-progress-report.html` lines 7–110
- reveal.js 3.9.0 CDN base: `https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.9.0/`
- Plotly chart divs need `id="chart-N"` where N matches the slide index

---

## Sub-Task 2 — Narrative slides (1–6, 14–15)

**Status:** `[ ] pending`

**Intent**
Populate all prose slides with actual text drawn from existing reports and docs.

**Expected Outcomes**
- Slide 1 — Title: project name, "Pipeline v2_corrected", dataset names, date
- Slide 2 — Motivation: OULAD dataset context, 32,593 enrollments, at-risk detection goal
- Slide 3 — Dataset: enrollment-centric prediction unit, label convention (1 = Fail/Withdrawn), prediction windows (weeks 2, 4, 6, 8 → days 14, 28, 42, 56)
- Slide 4 — Graph Schema: `viz_net_schema.png` base64-embedded, centred, full-slide
- Slide 5 — Pipeline: five bullet stages (data prep → graph construction → split → training → evaluation)
- Slide 6 — Leakage: Strategy B dual-guard displayed as code block (`due_date ≤ window` AND `date_submitted ≤ window`)
- Slide 14 — Findings: GNN AUROC 0.838 (random), 0.788 (LCPO); ablation highlights `with_iw_attrs` best at 0.846
- Slide 15 — Open Problems: Zenodo GNN near-random LCPO, seed variance, memory on large graphs

**Todo List**
1. Write slides 1–3 text from `AGENTS.md` context and `graphsage-at-risk-prediction-progress-report.html` metadata block
2. Embed `results/graph/artifacts/viz_net_schema.png` as base64 on slide 4
3. Write slide 5 (pipeline bullets) and slide 6 (leakage code block)
4. Write slides 14–15 from `comparison_summary.md` and Section C of the progress report

**Relevant Context**
- Label convention: `1 = at-risk`, `0 = success` — do not invert
- Dual-guard rule: from `src/oulad_data.py` `filter_window()` per `AGENTS.md`
- Section C problems: C7 = Zenodo GNN near-random LCPO (key open problem to highlight)

---

## Sub-Task 3 — Split-strategy comparison chart (Slide 7)

**Status:** `[ ] pending`

**Intent**
Interactive grouped bar chart comparing GNN performance across random-student split vs.
LCPO (mean AUROC, AUPRC, F1) side-by-side, making split type the grouping dimension.

**Expected Outcomes**
- Slide 7: two grouped bars per metric (3 metrics), groups = Random / LCPO, y-axis = score
- Error bars showing std dev for each split type
- Hover tooltip: "GNN — Random Split: AUROC 0.838 ± 0.004"
- Clear visual that LCPO drops AUROC ~0.05 vs random, and variance increases

**Data to hard-code (from `comparison_summary.md` + computed from `lcpo_results.csv`):**

GNN (unweighted, the cleaner of the two):
- Random: AUROC 0.838±0.004, AUPRC 0.880±0.004, F1 0.761±0.005
- LCPO: AUROC 0.788±0.065, AUPRC 0.810±0.117, F1 0.678±0.099

**Todo List**
1. Emit one Plotly `bar` trace per split type (Random, LCPO), `x` = metrics, `y` = means, `error_y` = stds
2. Set `barmode: 'group'`, y-axis range [0.5, 1.0] to accommodate the wider LCPO error bars
3. Add annotation noting the larger std for LCPO as "cross-course generalisation challenge"
4. Lazy-init via `slidechanged` event (slide index 6, 0-based)

**Relevant Context**
- Values already available from `comparison_summary.md` (no additional CSV read needed)
- LCPO std is ~10–15× larger than random split std — the key visual takeaway

---

## Sub-Task 4 — GNN vs LightGBM comparison charts (Slides 8 & 9)

**Status:** `[ ] pending`

**Intent**
Two grouped bar charts — random-student split (slide 8) and LCPO (slide 9) — comparing
three model variants on AUROC, AUPRC, F1 with error bars.

**Expected Outcomes**
- Slide 8: 3 model bars × 3 metrics; GNN clearly beats LightGBM on random split (AUROC +0.073)
- Slide 9: 2 model bars × 3 metrics; GNN and LightGBM closer on LCPO (AUROC +0.013)
- Both charts share the same y-axis range [0.60, 1.0]
- Hover shows mean ± std

**Data to hard-code (from `comparison_summary.md`):**

Random-student:
- GNN (weighted):   AUROC 0.837±0.003, AUPRC 0.880±0.003, F1 0.757±0.008
- GNN (unweighted): AUROC 0.838±0.004, AUPRC 0.880±0.004, F1 0.761±0.005
- LightGBM:         AUROC 0.765±0.005, AUPRC 0.807±0.005, F1 0.723±0.007

LCPO:
- GNN:       AUROC 0.788±0.065, AUPRC 0.810±0.117, F1 0.678±0.099
- LightGBM:  AUROC 0.775±0.056, AUPRC 0.791±0.118, F1 0.686±0.093

**Todo List**
1. Write slide 8 Plotly `data` array (3 traces, `barmode: 'group'`)
2. Write slide 9 Plotly `data` array (2 traces)
3. Set consistent layout: title, axis labels, legend, y-range [0.60, 1.0]
4. Lazy-init via `slidechanged` at slide indices 7 and 8

---

## Sub-Task 5 — LCPO per-fold variance chart (Slide 10)

**Status:** `[ ] pending`

**Intent**
Box plot showing GNN AUROC distribution (5 seeds) for each of the 22 LCPO folds,
sorted by median AUROC, to expose which course-presentations are hard to generalise to.

**Expected Outcomes**
- 22 box traces, one per `(held_out_module, held_out_presentation)` fold
- x-axis labels: short fold name (e.g. "AAA/2013J"), rotated 45°
- Folds sorted ascending by median AUROC
- Hover: fold name, median, individual seed values
- y-axis: [0.4, 1.0]

**Data computation (from `results/graph/lcpo_results.csv`, fully read above):**

The 22 folds and their 5 seed AUROC values must be extracted from the 110-row CSV.
Agent reads the full file during this sub-task and produces the JS data literal.

Fold → seed AUROCs (to be computed by agent during implementation):
- AAA/2013J: [0.6525, 0.7266, 0.7748, 0.7265, 0.7412] (from rows 2–6 already read)
- AAA/2014J: [0.7580, 0.7609, 0.7531, 0.7513, 0.7482] (from rows 7–10 and row 11)
- ... (remaining 20 folds read during agent pass)

**Todo List**
1. Read full `results/graph/lcpo_results.csv` (110 rows) during agent implementation pass
2. Group rows by `(held_out_module, held_out_presentation)` → list of 5 AUROCs
3. Compute median per fold, sort ascending
4. Emit 22 Plotly `box` traces as inline JS JSON, sorted by median
5. Set `showlegend: false`, tick labels at 45°, y-range [0.4, 1.0]

**Relevant Context**
- Source: `results/graph/lcpo_results.csv` (columns: `held_out_module`, `held_out_presentation`, `model_seed`, `auroc`)
- Seeds: 42, 123, 7, 17, 99 — always 5 per fold

---

## Sub-Task 6 — Ablation study chart (Slide 11)

**Status:** `[ ] pending`

**Intent**
Horizontal bar chart of 7 ablation conditions sorted descending by mean AUROC with
error bars (std over 5 seeds) and a vertical reference line at the `full` model mean.

**Expected Outcomes**
- 7 horizontal bars sorted descending by mean AUROC
- `with_iw_attrs` appears at top (best); `no_assessment` at bottom
- Error bars via `error_x`
- Reference line at `full` model mean (0.8372) with label "Full model baseline"
- Human-readable condition labels

**Pre-computed values (from `ablation_results.csv` fully read above):**

| Condition | Mean AUROC | Std |
|---|---|---|
| with_iw_attrs | 0.8462 | 0.0031 |
| full | 0.8372 | 0.0010 |
| no_edge_attrs | 0.8347 | 0.0028 |
| no_temporal | 0.8299 | 0.0049 |
| no_course_features | 0.8297 | 0.0013 |
| no_vle | 0.8235 | 0.0044 |
| no_assessment | 0.8089 | 0.0087 |

Human-readable labels:
- `full` → "Full model"
- `with_iw_attrs` → "With IW edge attrs"
- `no_edge_attrs` → "No edge attributes"
- `no_temporal` → "No temporal features"
- `no_course_features` → "No course features"
- `no_vle` → "No VLE edges"
- `no_assessment` → "No assessment edges"

**Todo List**
1. Hard-code the 7 × (mean, std) values from the table above as JS JSON
2. Emit a single `bar` trace, `orientation: 'h'`, sorted descending by AUROC
3. Add a `shape` line at x = 0.8372 (full model) across the y-axis
4. Set x-axis range [0.78, 0.86] to zoom in on differences
5. Lazy-init at slide index 10

---

## Sub-Task 7 — Week-by-week baseline chart (Slide 12)

**Status:** `[ ] pending`

**Intent**
Multi-line chart showing how tabular model AUROC improves as the prediction window
grows from Week 2 to Week 8.

**Expected Outcomes**
- 5 lines (one per model), x-axis = [Week 2, Week 4, Week 6, Week 8]
- y-axis range [0.70, 0.90]
- Hover: model name + exact AUROC
- Horizontal dashed reference line at GNN Week 8 AUROC (0.838) labelled "GNN (week 8)"

**Pre-computed values (from `output_results/summary_auroc_by_week_model.csv`, fully read above):**

| Model | Wk2 | Wk4 | Wk6 | Wk8 |
|---|---|---|---|---|
| Gradient Boosting | 0.754 | 0.808 | 0.828 | 0.859 |
| LightGBM | 0.748 | 0.804 | 0.827 | 0.860 |
| Logistic Regression | 0.738 | 0.791 | 0.812 | 0.837 |
| Random Forest | 0.741 | 0.799 | 0.823 | 0.854 |
| XGBoost | 0.732 | 0.787 | 0.810 | 0.840 |

**Todo List**
1. Hard-code the 5×4 value matrix as inline JS JSON
2. Emit one `scatter` trace per model (`mode: 'lines+markers'`)
3. Add horizontal `shape` line at y = 0.838 (GNN) with annotation
4. Set `xaxis.tickvals: [0,1,2,3]`, `ticktext: ['Week 2','Week 4','Week 6','Week 8']`
5. Lazy-init at slide index 11

---

## Sub-Task 8 — Cross-dataset Zenodo comparison chart (Slide 13)

**Status:** `[ ] pending`

**Intent**
Side-by-side grouped bar chart comparing GNN and LightGBM on both OULAD and the
external Zenodo (KU Leuven) dataset, for random-student split only. This exposes the
stark generalisation gap: GNN AUROC drops from 0.838 (OULAD) to ~0.553 (Zenodo).

**Expected Outcomes**
- 4 bar groups (GNN-OULAD, LightGBM-OULAD, GNN-Zenodo, LightGBM-Zenodo) × 3 metrics
- OR alternatively: metric on x-axis, dataset as grouping (whichever is clearer — agent chooses)
- Error bars (std over 5 seeds)
- Annotations calling out the GNN drop and LightGBM relative stability
- y-axis: [0.4, 1.0] to accommodate Zenodo near-random scores

**Pre-computed values:**

OULAD (from `comparison_summary.md`):
- GNN (unweighted): AUROC 0.838±0.004, AUPRC 0.880±0.004, F1 0.761±0.005
- LightGBM: AUROC 0.765±0.005, AUPRC 0.807±0.005, F1 0.723±0.007

Zenodo (computed from `results/zenodo/random_student_results.csv` and `results/zenodo/comparison_results.csv`):
- GNN: seeds [0.564, 0.597, 0.518, 0.524, 0.567] → mean ≈ 0.554, std ≈ 0.032
  (from `results/zenodo/random_student_results.csv`, fully read above)
- LightGBM: seeds [0.651, 0.621, 0.638, 0.673, 0.648] → mean ≈ 0.646, std ≈ 0.019
  (from `results/zenodo/comparison_results.csv` rows 2–6, random_student rows)

Note: Zenodo AUPRC and F1 values must be computed by the agent during implementation
from the full `results/zenodo/random_student_results.csv` and `results/zenodo/comparison_results.csv`.

**Todo List**
1. Compute Zenodo LightGBM AUPRC and F1 means/stds from `results/zenodo/comparison_results.csv` rows 2–6
2. Compute Zenodo GNN AUPRC and F1 means/stds from `results/zenodo/random_student_results.csv`
3. Emit a 4-trace grouped bar chart with `barmode: 'group'`, x = metrics, y = means, error_y = stds
4. Add annotation box: "GNN AUROC: OULAD 0.838 → Zenodo 0.554 (−0.284)" to highlight the gap
5. Lazy-init at slide index 12

**Relevant Context**
- `results/zenodo/random_student_results.csv`: 5 rows (seeds 42, 123, 7, 17, 99), GNN Zenodo results
- `results/zenodo/comparison_results.csv`: rows 2–6 = LightGBM random-student, rows 7+ = LCPO folds
- The Zenodo open problem is Section C7 in the progress report

---

## Sub-Task 9 — Base64-embed all 35 static PNGs

**Status:** `[ ] pending`

**Intent**
Convert all 35 PNGs to base64 and embed them as `<img src="data:image/png;base64,...">` so
the HTML file is fully self-contained and portable (no broken images when moved).

**Expected Outcomes**
- All 35 images load correctly when `presentation.html` is opened from any directory
- Slide 4 shows `viz_net_schema.png` full-slide
- Appendix section has 35 image slides grouped into 4 sub-sections (A–D)
- Each appendix slide has a human-readable filename caption

**PNG inventory by group:**

Group A — Graph Artifacts (12 files in `results/graph/artifacts/`):
`viz_atrisk_by_course`, `viz_class_balance`, `viz_degree_clicks`,
`viz_enrollments_per_student`, `viz_graph_scale`, `viz_lcpo_splits`,
`viz_net_course_vle`, `viz_net_schema`, `viz_net_score_vs_clicks`,
`viz_net_student_course`, `viz_split_quality`, `viz_vle_activity_over_time`

Group B — Report Figures (7 files in `results/graph/figures/`):
`fig_ablation`, `fig_course_diagnostics`, `fig_course_variation`,
`fig_cross_dataset`, `fig_module_summary`, `fig_random_vs_lcpo`, `fig_week_performance`

Group C — Baseline & Importance (4 files):
`results/baseline/baseline_results_plot.png`
`results/feature_importance/category_importance.png`
`results/feature_importance/feature_importance_comparison.png`
`results/consolidated_comparison.png`

Group D — Weekly Output Plots (12 files in `output_results/`):
`week_2_class_distribution`, `week_2_sparsity_plot`, `week_2_variance_plot`,
`week_4_class_distribution`, `week_4_sparsity_plot`, `week_4_variance_plot`,
`week_6_class_distribution`, `week_6_sparsity_plot`, `week_6_variance_plot`,
`week_8_class_distribution`, `week_8_sparsity_plot`, `week_8_variance_plot`

**Implementation approach:**
The agent uses a short Python one-liner to generate base64 strings for each PNG:
```python
import base64, pathlib
b64 = base64.b64encode(pathlib.Path("results/graph/artifacts/viz_net_schema.png").read_bytes()).decode()
```
These strings are inlined as `data:image/png;base64,<b64>` in `<img src>` attributes.

**Todo List**
1. For slide 4: read `viz_net_schema.png`, base64-encode, embed in slide 4 `<img>`
2. For each of the 35 PNGs: base64-encode and build the appendix `<section>` slide
3. Group appendix slides under four `<section>` wrappers (vertical navigation: A/B/C/D)
4. Each slide: `<img>` centred, `max-height: 520px`, `max-width: 90%` + `<p class="caption">` label

**Relevant Context**
- Python base64 encoding must be run during agent pass (requires `execute_command` in agent mode)
- Total embedded image data may make the HTML file 10–30 MB — acceptable for an internal deck

---

## Sub-Task 10 — Lazy chart initialisation and final polish

**Status:** `[ ] pending`

**Intent**
Wire all 7 Plotly charts to initialise only when their slide is first navigated to,
preventing race conditions, and add final HTML polish.

**Expected Outcomes**
- No chart renders before its slide is active
- Each chart fills its container div at `height: 460px`
- Slide counter shows "8 / 50" style numbering
- Hash bookmarks work (slide URL changes on navigation)
- File opens without console errors in Chrome and Safari

**Todo List**
1. Build a `chartInit` map: `{slideIndex: () => Plotly.newPlot(...)}` for all 7 charts
2. In `Reveal.on('slidechanged', e => { ... })` call the matching init function once (guard with `initialized` set)
3. Also trigger slide 0 init in `Reveal.on('ready', ...)` if chart 0 is on slide 0
4. Validate that all `<div id="chart-N">` containers exist in the DOM before Plotly calls
5. Final read-through pass: check all `<section>` tags are closed, no duplicate IDs

**Relevant Context**
- reveal.js event: `Reveal.on('slidechanged', event => event.indexh)` gives horizontal index
- Plotly requires its container div to be visible and non-zero-height at render time
- CDN Plotly version: 2.27.0 (matches the version pinned in Sub-Task 1)
