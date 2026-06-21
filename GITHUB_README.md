# OULAD At-Risk Student Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Predicting at-risk students using the Open University Learning Analytics Dataset (OULAD)**

This repository contains a comprehensive machine learning pipeline for early identification of at-risk students in online learning environments. The project implements multiple evaluation strategies, feature importance analysis, and threshold optimization for real-world deployment.

## 🎯 Key Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Cross-Course Evaluation**: Leave-Course-Presentation-Out (LCPO) validation
- **Temporal Prediction**: Early warning at weeks 2, 4, 6, and 8
- **Feature Analysis**: Comprehensive feature importance and category analysis
- **Threshold Optimization**: Deployment-ready thresholds for different scenarios
- **Leakage Prevention**: Rigorous temporal filtering to prevent data leakage
- **Reproducible Results**: All experiments fully documented and reproducible

## 📊 Results Summary

| Model | Baseline AUROC | LCPO AUROC | Performance Drop |
|-------|---------------|------------|------------------|
| **LightGBM** | **0.835±0.005** | **0.804±0.087** | **3.7%** |
| XGBoost | 0.831±0.006 | 0.799±0.089 | 3.9% |
| Random Forest | 0.826±0.006 | 0.793±0.091 | 4.0% |
| Logistic Regression | 0.803±0.007 | 0.772±0.095 | 3.9% |

**Best Model**: LightGBM achieves excellent performance with realistic cross-course generalization.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum
- OULAD dataset (download instructions below)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/OULAD.git
cd OULAD

# Create virtual environment
python -m venv oulad_env
source oulad_env/bin/activate  # On Windows: oulad_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download OULAD Dataset

1. Visit: https://analyse.kmi.open.ac.uk/open_dataset
2. Download all 7 CSV files
3. Place them in `data/raw/` directory:
   - `assessments.csv`
   - `courses.csv`
   - `studentAssessment.csv`
   - `studentInfo.csv`
   - `studentRegistration.csv`
   - `studentVle.csv`
   - `vle.csv`

### Run Baseline Evaluation

```bash
# Run complete baseline evaluation (4 models × 4 weeks)
python src/baseline_evaluation.py

# Results saved to: results/baseline/
```

### Run LCPO Evaluation

```bash
# Run leave-course-presentation-out evaluation
python src/lcpo_evaluation.py

# Results saved to: results/lcpo/
```

### View Results

```bash
# Start Jupyter Lab
jupyter lab notebooks/OULAD_Complete_Results.ipynb
```

## 📁 Repository Structure

```
OULAD/
├── src/                          # Source code
│   ├── config.py                 # Configuration and constants
│   ├── baseline_evaluation.py    # Baseline evaluation script
│   ├── lcpo_evaluation.py       # LCPO evaluation script
│   ├── feature_importance_analysis.py
│   ├── threshold_optimization.py
│   └── gnn_model.py             # GNN implementation (framework)
│
├── data/                         # Data directory
│   └── raw/                      # OULAD dataset (not in repo)
│
├── results/                      # Evaluation results
│   ├── baseline/                 # Baseline results
│   ├── lcpo/                     # LCPO results
│   ├── feature_importance/       # Feature analysis
│   └── threshold_optimization/   # Threshold analysis
│
├── docs/                         # Documentation
│   ├── COMPLETE_IMPLEMENTATION_PLAN.md
│   ├── EXECUTION_GUIDE.md
│   ├── LEAKAGE_PREVENTION.md
│   ├── EVALUATION_SPLITS.md
│   ├── GRAPH_SCHEMA.md
│   └── CROSS_COURSE_EVALUATION_REPORT.md
│
├── notebooks/                    # Jupyter notebooks
│   ├── OULAD_Complete_Results.ipynb
│   └── OULAD_Consolidated_Analysis.ipynb
│
├── README.md                     # Detailed project documentation
├── QUICK_START.md               # Quick start guide
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── CONTRIBUTING.md              # Contribution guidelines
```

## 🎓 Label Convention

**CRITICAL**: This project uses the following label convention:

- **1 = at-risk** (Fail/Withdrawn) - students requiring intervention
- **0 = success** (Pass/Distinction) - students on track

All metrics (precision, recall, F1, AUPRC) refer to identifying **at-risk students**.

## 📈 Evaluation Strategies

### 1. Random Split (Baseline)
- 5-fold cross-validation
- Random student/student-course split
- Optimistic performance estimate
- **AUROC**: 0.835±0.005

### 2. Leave-Course-Presentation-Out (LCPO)
- Each course-presentation held out once
- Tests cross-course generalization
- Realistic performance estimate
- **AUROC**: 0.804±0.087

### 3. Future-Presentation Split
- Train on past presentations
- Test on future presentations
- Most realistic deployment scenario
- **AUROC**: 0.789±0.095

## 🔍 Feature Groups

| Feature Group | AUROC | Description |
|--------------|-------|-------------|
| **Combined** | **0.804** | All features (best) |
| Assessment | 0.768 | Assessment scores and submissions |
| VLE Activity | 0.742 | Virtual learning environment interactions |
| Demographics | 0.651 | Student background information |

**Key Finding**: Combined features always outperform individual groups.

## ⚙️ Deployment Thresholds

| Scenario | Threshold | Precision | Recall | F1 | Use Case |
|----------|-----------|-----------|--------|----|----|
| **Max F1** | 0.45 | 0.72 | 0.68 | 0.70 | Balanced approach |
| **High Precision** | 0.60 | 0.82 | 0.52 | 0.64 | Minimize false alarms |
| **High Recall** | 0.30 | 0.58 | 0.85 | 0.69 | Catch all at-risk |
| **Resource-Constrained** | 0.55 | 0.78 | 0.58 | 0.67 | Limit to 20% |
| **Cost-Sensitive** | 0.40 | 0.68 | 0.75 | 0.71 | Optimize for costs |

## 🛡️ Leakage Prevention

This project implements rigorous temporal filtering to prevent data leakage:

- **No future information**: Only data available before prediction window is used
- **Temporal filtering**: VLE activity and assessments filtered by date
- **Removed features**: Final grades, post-prediction assessments
- **Documented**: See `docs/LEAKAGE_PREVENTION.md` for details

## 📊 Course-Level Variation

Performance varies significantly across courses:

- **Best courses**: DDD, CCC, FFF (AUROC 0.85-0.88)
- **Hardest courses**: GGG courses (AUROC 0.60)
- **Variation**: 0.27 AUROC spread

**Insight**: Some courses are much harder to predict, suggesting course-specific factors.

## 🔬 Research Applications

This codebase supports research in:

1. **Educational Data Mining**: Student success prediction
2. **Learning Analytics**: Early warning systems
3. **Machine Learning**: Cross-domain generalization
4. **Graph Neural Networks**: Heterogeneous graph learning (Phase 3)

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{oulad_prediction_2026,
  title = {OULAD At-Risk Student Prediction},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/OULAD}
}
```

And the original OULAD dataset:

```bibtex
@article{kuzilek2017open,
  title={Open university learning analytics dataset},
  author={Kuzilek, Jakub and Hlosta, Martin and Zdrahal, Zdenek},
  journal={Scientific data},
  volume={4},
  number={1},
  pages={1--8},
  year={2017},
  publisher={Nature Publishing Group}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit pull requests
- ⭐ Star the repository

## 📖 Documentation

- **[README.md](README.md)**: Detailed project documentation
- **[QUICK_START.md](QUICK_START.md)**: Quick start guide with copy-paste commands
- **[EXECUTION_GUIDE.md](docs/EXECUTION_GUIDE.md)**: Step-by-step execution guide
- **[LEAKAGE_PREVENTION.md](docs/LEAKAGE_PREVENTION.md)**: Temporal leakage prevention
- **[EVALUATION_SPLITS.md](docs/EVALUATION_SPLITS.md)**: Evaluation split definitions
- **[GRAPH_SCHEMA.md](docs/GRAPH_SCHEMA.md)**: Graph neural network schema

## 🗺️ Roadmap

### Phase 1: Baseline Pipeline ✅
- [x] Baseline evaluation
- [x] LCPO evaluation
- [x] Feature importance analysis
- [x] Threshold optimization

### Phase 2: Cross-Course Analysis ✅
- [x] Cross-course evaluation report
- [x] Course-level variation analysis
- [x] Graph schema design

### Phase 3: GNN Implementation 🔄
- [ ] Graph construction pipeline
- [ ] Heterogeneous attention network
- [ ] GNN evaluation
- [ ] Comparison with baseline

### Phase 4: Deployment 📋
- [ ] Model serving API
- [ ] Real-time prediction pipeline
- [ ] Monitoring dashboard
- [ ] User guide for educators

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: your.email@example.com

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

The OULAD dataset is licensed under CC BY 4.0 - see dataset website for details.

## 🙏 Acknowledgments

- Open University for providing the OULAD dataset
- Contributors to scikit-learn, XGBoost, LightGBM
- Educational data mining community

---

**⭐ If you find this project useful, please star the repository!**

Made with ❤️ for educational data mining