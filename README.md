# SkyFlow: Machine Learning Foundations for Air Traffic Congestion & Delay Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Studio-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning project directly aligned with **Air Traffic Congestion and Flight Delay Prediction**, meticulously structured to cover **Course Outcomes CO1 through CO6** and **Syllabus Units I through VI**.

---

## 🎯 Syllabus & Course Outcomes (CO1 - CO6) Coverage

- **CO1 (Unit I - Model Selection & Evaluation)**: Bias-Variance tradeoff curves, 5-Fold Cross-Validation, GridSearchCV & RandomizedSearchCV hyperparameter optimization.
- **CO2 (Unit II - Advanced Regression)**: Simple Linear Regression (closed-form & matrix derivation $(X^T X)^{-1} X^T y$), Multiple Linear Regression, OLS diagnostics (VIF for multicollinearity, Q-Q plots for normality, residuals vs fitted for homoscedasticity), Polynomial Regression, Ridge ($L_2$) and Lasso ($L_1$) regularization paths.
- **CO3 (Unit III - Support Vector Machines - I)**: 2D & 3D Hyperplane visualization, Maximal Margin vs Soft Margin Classifier, Slack Variables ($\xi_i$), and Cost of Misclassification ($C$).
- **CO4 (Unit IV - Support Vector Machines - II)**: The Kernel Trick, Linear vs Polynomial vs Radial Basis Function (RBF) Kernels, Support Vector Regression (SVR).
- **CO5 (Unit V - Tree Models - I)**: Decision Tree Classification & Regression, Impurity measures (Gini Index vs Entropy scratch derivation), Information Gain, Cost-Complexity Pruning (`ccp_alpha`), Feature Importance (MDI).
- **CO6 (Unit VI - Tree Models - II)**: Ensemble techniques, Bootstrap Aggregation (Bagging), Random Forests, Out-of-Bag (OOB) error estimation and mathematical proof, Random Forest Classifier and Regressor.

---

## 🚀 Quick Start Guide

### 1. Launch the Interactive ML Studio Dashboard
Double-click `run_ml_studio.bat` or run:
```bash
streamlit run app_ml_studio.py --server.port 8502
```
Open `http://localhost:8502` in your browser.

### 2. Execute the Master Benchmark Pipeline
To re-run all 6 units, evaluate all models, and generate high-resolution plots:
```bash
python src/master_pipeline.py
```

### 3. Run Individual Unit Modules
```bash
python src/unit1_model_selection.py    # Unit I (CO1)
python src/unit2_advanced_regression.py # Unit II (CO2)
python src/unit3_4_svm.py               # Unit III & IV (CO3 & CO4)
python src/unit5_decision_trees.py      # Unit V (CO5)
python src/unit6_random_forest.py       # Unit VI (CO6)
```

### 4. Explore the Interactive Jupyter Notebook
Open `Air_Traffic_Machine_Learning_Project_Notebook.ipynb` in VS Code or JupyterLab for step-by-step interactive derivations and plots.

---

## 📊 Summary of Master Benchmark Results

### Continuous Departure Delay Forecasting (Regression)
| Model Architecture | MAE (min) | RMSE (min) | R² Score |
| :--- | :---: | :---: | :---: |
| **Simple Linear Regression** | 31.03 | 39.38 | 0.0012 |
| **Polynomial Regression (Deg 2)** | 11.30 | 16.78 | 0.8187 |
| **Multiple Linear Regression** | 2.75 | 3.85 | 0.9904 |
| **Ridge Regression ($L_2$)** | 2.74 | 3.85 | 0.9904 |
| **Lasso Regression ($L_1$)** | 2.63 | 3.99 | 0.9898 |
| **Decision Tree Regressor** | 1.18 | 2.00 | 0.9974 |
| **Support Vector Regressor (SVR)** | 0.59 | 1.10 | 0.9992 |
| **Random Forest Regressor** | **0.35** | **0.87** | **0.9995** |

### Congestion Level Classification (Low / Med / High)
| Model Architecture | Accuracy (%) | Macro F1 (%) | Precision (%) | Recall (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 90.83% | 90.46% | 90.45% | 90.47% |
| **Linear SVC** | 91.20% | 90.91% | 90.90% | 91.12% |
| **Kernel SVM (RBF Kernel)** | 96.27% | 96.10% | 96.09% | 96.10% |
| **Random Forest Classifier** | 99.80% | 99.79% | 99.80% | 99.78% |
| **Decision Tree Classifier** | **99.86%** | **99.85%** | **99.85%** | **99.84%** |

---

## 📁 Project Directory Structure

```
AirTraffic_ML_Project/
├── data/
│   └── hourly_airport_traffic.csv     # 17,280 operational records across 8 major hubs
├── models/
│   ├── regression_models.pkl          # Saved binaries of all trained regressors
│   ├── classification_models.pkl      # Saved binaries of all trained classifiers
│   ├── scaler.pkl                     # Standard scaler object
│   └── master_metrics.json            # Consolidated JSON of all evaluation metrics
├── plots/                             # Auto-generated high-resolution academic figures
│   ├── unit1_bias_variance_tradeoff.png
│   ├── unit2_slr_fit.png
│   ├── unit2_mlr_residual_diagnostics.png
│   ├── unit2_regularization_paths.png
│   ├── unit3_svm_2d_hyperplane_margins.png
│   ├── unit3_svm_3d_hyperplane.png
│   ├── unit5_cost_complexity_pruning.png
│   ├── unit5_decision_tree_feature_importance.png
│   ├── unit6_rf_oob_vs_test_error.png
│   ├── unit6_rf_feature_importance.png
│   └── master_model_benchmark_comparison.png
├── src/
│   ├── data_loader.py                 # Data loading and scaling pipeline
│   ├── unit1_model_selection.py       # CO1 Implementation
│   ├── unit2_advanced_regression.py   # CO2 Implementation
│   ├── unit3_4_svm.py                 # CO3 & CO4 Implementation
│   ├── unit5_decision_trees.py        # CO5 Implementation
│   ├── unit6_random_forest.py         # CO6 Implementation
│   └── master_pipeline.py             # Complete benchmark execution script
├── app_ml_studio.py                   # Interactive Streamlit Web Laboratory
├── run_ml_studio.bat                  # One-click desktop launcher for the dashboard
├── Air_Traffic_Machine_Learning_Project_Notebook.ipynb # Complete annotated Jupyter Notebook
├── PROJECT_REPORT_CO1_CO6.md          # Comprehensive academic report for submission
└── README.md
```
