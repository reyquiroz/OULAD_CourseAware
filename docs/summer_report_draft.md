# Preliminary Title

GraphSAGE for Early Identification of At-Risk Students in OULAD: A Comparison with LightGBM Across Random and Leave-Course-Presentation-Out Evaluation

# Abstract

This report studies whether a graph representation improves early prediction of at-risk students in the Open University Learning Analytics Dataset (OULAD). We model OULAD as a heterogeneous graph linking students, course presentations, assessments, and VLE resources, and evaluate a GraphSAGE-based enrollment classifier against a tabular LightGBM baseline. We focus on three research questions: whether graph structure improves predictive performance over tabular features, how predictive performance changes across weeks 2, 4, 6, and 8 of presentation, and which feature groups contribute most to graph model performance. Under random student splits, the weighted GraphSAGE model reaches an AUROC of 0.881 ± 0.002 at week 8, compared with 0.842 ± 0.005 for LightGBM. Under the more challenging leave-course-presentation-out (LCPO) setting at week 8, GraphSAGE achieves 0.844 ± 0.063 AUROC across 22 held-out course presentations, while LightGBM reaches 0.824 ± 0.075. Performance improves steadily from week 2 to week 8 for both models, but the graph model retains a consistent margin. Ablation results show that removing assessment information and edge attributes causes the largest degradations, indicating that relational behavioral signals drive much of the GraphSAGE advantage. Course-level analysis further shows that gains are uneven, with the largest improvements appearing in several AAA, FFF, and GGG presentations, while a small number of EEE and CCC presentations favor LightGBM. Overall, the findings suggest that graph-based representations can provide practical gains for early risk prediction, especially when generalizing across course presentations.

## 1. Introduction

Early identification of students at risk of failure or withdrawal is a central problem in learning analytics. Institutions seek models that can surface actionable signals early enough for intervention while remaining robust across different courses and presentation runs. OULAD is a widely used benchmark for this problem because it combines demographics, prior history, assessment behavior, and virtual learning environment (VLE) activity across multiple modules and presentations.

This study evaluates whether representing OULAD as a heterogeneous graph improves early risk prediction relative to a strong tabular baseline. Rather than collapsing all interactions into flat aggregate features alone, the graph formulation preserves links among students, course presentations, assessments, and VLE resources. This allows message passing to combine student context with structured behavioral information.

The report addresses three research questions:

1. Does a graph representation provide an advantage over tabular LightGBM for at-risk prediction?
2. How does performance change across prediction windows at weeks 2, 4, 6, and 8?
3. Which feature groups matter most to GraphSAGE performance?

## 2. Related Work

OULAD has been used extensively for predictive modeling of student outcomes, commonly with tabular machine learning methods built on aggregated clickstream and assessment features. Tree-based methods such as gradient boosting remain strong baselines because they handle heterogeneous feature scales, nonlinear interactions, and missingness well.

Graph neural networks offer a different perspective by explicitly modeling relationships between entities. GraphSAGE is particularly relevant because it learns neighborhood aggregation functions that scale to larger graphs and can integrate node and edge-derived signals. In learning analytics, graph-based models have been explored for student-resource interaction modeling, social learning representations, and relational prediction, but the gains over strong tabular baselines remain context dependent.

This project contributes a direct GraphSAGE versus LightGBM comparison on OULAD under both standard random splits and a harder leave-course-presentation-out setting that better tests generalization across course contexts.

## 3. Data and Preprocessing

The dataset is the Open University Learning Analytics Dataset (OULAD), which contains student enrollments across multiple modules and presentations together with demographics, prior attempts, assessments, and VLE interactions. The prediction target is a binary at-risk label derived from final outcomes.

For each prediction week, only information available up to that time window is used. Week cutoffs are evaluated at weeks 2, 4, 6, and 8. The graph construction is enrollment-centric and heterogeneous, with node types for students, course presentations, assessments, and VLE resources. Edge types connect students to course presentations, assessments, and resources, while edge attributes summarize enrollment, submission, and interaction behavior.

Preprocessing includes normalization of numeric features, alignment of graph artifacts to canonical enrollment order, and split generation for both random-student evaluation and leave-course-presentation-out evaluation. The LCPO protocol holds out one complete course presentation at a time, yielding 22 folds at week 8.

## 4. Methods

The primary model is a GraphSAGE-based heterogeneous graph neural network operating over the OULAD graph. Student, course-presentation, assessment, and VLE-resource node features are first projected into a shared hidden space. Message passing is performed with GraphSAGE layers, and enrollment, submission, and interaction edge attributes are injected through learned projections aggregated back to student representations before the final prediction head.

The baseline is a LightGBM classifier trained on aligned tabular features constructed from the same underlying OULAD records. This baseline uses the same split definitions as the graph model to ensure direct comparability.

For random-student evaluation, multiple seeds are used and both weighted and unweighted GraphSAGE runs are recorded. The main comparisons in this report use the weighted GraphSAGE results. For LCPO evaluation, each fold trains on all non-held-out course presentations and tests on the held-out presentation. Early stopping is based on validation AUROC, and decision thresholds are tuned on validation data for random-split experiments.

Ablation experiments remove specific feature groups while preserving graph structure by zeroing selected node or edge features. The evaluated conditions are full, no_assessment, no_vle, no_temporal, no_course_features, and no_edge_attrs.

## 5. Results

### Main comparison at week 8

| split | model | AUROC | AUPRC | F1 | PRECISION | RECALL | BALANCED_ACC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | GNN (weighted) | 0.881 ± 0.002 | 0.909 ± 0.002 | 0.801 ± 0.005 | 0.797 ± 0.052 | 0.809 ± 0.047 | 0.786 ± 0.021 |
| Random | LightGBM | 0.842 ± 0.005 | 0.868 ± 0.005 | 0.776 ± 0.007 | 0.787 ± 0.007 | 0.766 ± 0.007 | 0.767 ± 0.006 |
| LCPO | GNN | 0.844 ± 0.063 | 0.850 ± 0.106 | 0.738 ± 0.087 | 0.786 ± 0.144 | 0.726 ± 0.121 | 0.753 ± 0.069 |
| LCPO | LightGBM | 0.824 ± 0.075 | 0.823 ± 0.129 | 0.737 ± 0.095 | 0.744 ± 0.191 | 0.787 ± 0.111 | 0.738 ± 0.104 |

The graph model outperforms LightGBM on AUROC in both settings, with a margin of about 0.039 under random-student splitting and about 0.020 under LCPO. This supports the claim that relational structure adds useful predictive signal beyond a tabular aggregation baseline.

### Early-prediction performance across weeks

| week | GNN weighted | GNN unweighted | LightGBM |
| --- | --- | --- | --- |
| 2 | 0.792 ± 0.002 | 0.791 ± 0.001 | 0.732 ± 0.003 |
| 4 | 0.846 ± 0.002 | 0.845 ± 0.002 | 0.791 ± 0.005 |
| 6 | 0.859 ± 0.004 | 0.860 ± 0.003 | 0.812 ± 0.003 |
| 8 | 0.881 ± 0.002 | 0.883 ± 0.001 | 0.842 ± 0.005 |

Performance rises steadily as more course activity becomes available. The graph model leads at every week, with the largest absolute gap at week 4 and a sustained advantage through week 8.

### Ablation study

| condition | auroc | auprc | f1 |
| --- | --- | --- | --- |
| no_course_features | 0.881179558893192 | 0.9081665386827982 | 0.8006751572809575 |
| full | 0.879042489031354 | 0.9069085285228708 | 0.7992346187812776 |
| no_temporal | 0.8723391858003139 | 0.9012691356020668 | 0.7941303409581355 |
| no_vle | 0.8705277890024472 | 0.9007561195733472 | 0.7930197268588771 |
| no_assessment | 0.835629739900446 | 0.8718988137249544 | 0.7578532187632061 |
| no_edge_attrs | 0.8325803030509409 | 0.8745028677061035 | 0.75443114679594 |

Relative to the full model, removing assessment features reduces AUROC by about 0.043, and removing edge attributes reduces AUROC by about 0.046. By contrast, removing course features does not hurt performance in this single-seed run, suggesting that behavioral and relational information dominate static course-level descriptors.

### Course-level variation

| held_out_module | held_out_presentation | gnn_auroc | lgbm_auroc | auroc_delta | gnn_f1 | lgbm_f1 | f1_delta | n_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GGG | 2013J | 0.7048001126126127 | 0.6408502252252253 | 0.0639498873873873 | 0.5993945509586277 | 0.5460122699386503 | 0.0533822810199774 | 952 |
| AAA | 2013J | 0.8334361082562523 | 0.7730900993490921 | 0.0603460089071601 | 0.5644171779141104 | 0.5910652920962199 | -0.0266481141821095 | 383 |
| GGG | 2014J | 0.7121473932949343 | 0.6606778910057598 | 0.0514695022891744 | 0.6034912718204489 | 0.5826170009551098 | 0.0208742708653391 | 749 |
| AAA | 2014J | 0.8020539243365331 | 0.7541643139469225 | 0.0478896103896105 | 0.5968586387434555 | 0.577922077922078 | 0.0189365608213775 | 365 |
| BBB | 2014J | 0.8233659234892786 | 0.7788815028021443 | 0.0444844206871343 | 0.7299593646102697 | 0.717100633356791 | 0.0128587312534786 | 2292 |
| CCC | 2014J | 0.8107377868719045 | 0.8666864862530684 | -0.0559486993811639 | 0.7244936950706916 | 0.8132183908045977 | -0.088724695733906 | 2498 |
| EEE | 2014J | 0.8197287199480182 | 0.8612372946254526 | -0.0415085746774344 | 0.7006960556844548 | 0.7634615384615384 | -0.0627654827770836 | 1188 |
| EEE | 2014B | 0.8315171766035792 | 0.8453980998927761 | -0.0138809232891969 | 0.7384615384615385 | 0.7585139318885449 | -0.0200523934270063 | 694 |
| DDD | 2014J | 0.8571671212620768 | 0.869671617260638 | -0.0125044959985611 | 0.7334905660377359 | 0.7868131868131868 | -0.0533226207754509 | 1803 |
| BBB | 2013J | 0.8727812119659214 | 0.8703558388315931 | 0.0024253731343283 | 0.7507629704984741 | 0.7867924528301887 | -0.0360294823317146 | 2237 |

The largest GNN gains appear in both GGG presentations and both AAA presentations, while the clearest LightGBM wins are CCC 2014J and EEE 2014J. The pattern suggests that graph structure may be especially helpful for presentations where sparse or difficult-to-summarize relational behavior matters, while some presentations remain well captured by flat aggregated features.

## 6. Discussion

Across both evaluation settings, GraphSAGE consistently exceeds LightGBM on AUROC. The random-split advantage is larger than the LCPO advantage, but the LCPO result is arguably more important because it measures robustness when generalizing to unseen course presentations. The fact that the graph model still leads under LCPO suggests that relational inductive bias is not only memorizing familiar presentation-specific patterns.

The week-by-week trend indicates that predictive performance improves smoothly as more evidence accumulates. Even at week 2, the graph model already outperforms the tabular baseline, implying that early relational signals are informative. By week 8, both models are strong, but the graph model retains a clear margin.

The ablation findings show that assessment information and edge attributes are the most important components in the current architecture. In contrast, removing course features has negligible impact in the observed run, suggesting that student behavior and relational context dominate static presentation descriptors. This also helps explain why GraphSAGE gains are largest in some presentations where interaction structure may be especially informative.

Course-level variation shows that the graph advantage is not uniform. Strong wins on GGG and AAA coexist with losses on CCC 2014J and EEE 2014J. This heterogeneity likely reflects differences in class balance, assessment design, cohort size, and the density or usefulness of interaction signals. These results argue against treating a single average score as the whole story; deployment-oriented conclusions should consider where graph models help most and where simpler tabular models remain competitive.

## 7. Conclusion and Future Work

This study finds that a heterogeneous GraphSAGE model provides a consistent improvement over a matched LightGBM baseline for early at-risk prediction on OULAD. The advantage appears under both random-student and leave-course-presentation-out evaluation and persists across weeks 2, 4, 6, and 8. Ablation results indicate that assessment features and edge attributes are central to that gain.

Future work should extend the ablation study to multiple seeds, investigate calibration and threshold stability under LCPO, and analyze course-level correlates of graph-model gains more systematically. Additional directions include inductive evaluation on newly added students or presentations, richer temporal architectures, and intervention-oriented analyses linking predictions to actionable support strategies.

## References

- Hamilton, W. L., Ying, Z., and Leskovec, J. (2017). Inductive representation learning on large graphs.
- Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree.
- Kuzilek, J., Hlosta, M., and Zdrahal, Z. (2017). Open University Learning Analytics Dataset.
- Representative prior OULAD risk-prediction studies using clickstream and assessment aggregation.
- Representative graph-based learning analytics studies on student-resource interaction modeling.
