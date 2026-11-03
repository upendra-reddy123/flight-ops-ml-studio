# Machine Learning Foundations for Air Traffic Congestion & Delay Forecasting
## Comprehensive Academic Project Report & Course Outcome (CO1 - CO6) Attainment Document

---

### Project Title
**Predictive Airport Congestion and Flight Delay Modeling Using Machine Learning: A Multi-Paradigm Comparative Study**

### Domain & Relevance
This project is directly tied to the **Real-Time Air Traffic Congestion Forecasting System**. While the deep learning system uses recurrent temporal architectures (LSTM/GRU) for sliding sequential forecasting, this project establishes the foundational, rigorous **Machine Learning** framework across all core course units:
1. Predicting continuous flight departure delays (Regression).
2. Predicting airport congestion severity levels (Classification).

---

## 1. Course Outcomes (CO) & Syllabus Mapping Matrix

| Course Outcome | Unit / Module | Core Theory & Topics Covered | Aviation Domain Implementation |
| :--- | :--- | :--- | :--- |
| **CO1** | **Unit I / Module 1**: Model Selection & Evaluation | Simplicity, Complexity, Bias-Variance Tradeoff, K-Fold CV, Stratified K-Fold, GridSearchCV, Regularization hyperparameters | 5-Fold CV on airport flight delays; empirical Bias-Variance curve across tree depths; hyperparameter grid search for traffic congestion classification. |
| **CO2** | **Unit II / Module 2**: Advanced Regression | SLR closed-form derivation, Matrix formulation $\hat{\beta} = (X^T X)^{-1} X^T y$, MLR, OLS assumptions (Linearity, Homoscedasticity, Normality, VIF for Multicollinearity), Polynomial Regression, Ridge ($L_2$) & Lasso ($L_1$) regularization paths | Predicting departure delay as a function of runway load and taxi-out duration; VIF detection of flight count collinearity; Lasso automated feature selection. |
| **CO3** | **Unit III / Module 3**: Support Vector Machines - I | Hyperplane in 2D & 3D, Maximal Margin, Soft Margin, Slack Variables ($\xi_i \ge 0$), Cost of misclassification ($C$) | Geometric 2D and 3D hyperplane visualizations separating congested vs uncongested airport states; analysis of slack variable violations under varying $C$. |
| **CO4** | **Unit IV / Module 3**: Support Vector Machines - II | Kernel Methods, Feature Mapping $\phi(x)$, The Kernel Trick, Linear vs Polynomial vs RBF (Gaussian) Kernels, Support Vector Regression (SVR) | Non-linear mapping of airport weather and runway saturation; benchmark of Linear, Poly, and RBF kernels; SVR continuous delay regression. |
| **CO5** | **Unit V / Module 4**: Tree Models - I | Decision Trees, Splitting criteria, Impurity Measures (Gini Index vs Entropy), Information Gain, Tree Truncation, Cost-Complexity Pruning (ccp_alpha), Feature Importance (MDI) | Decision tree classification of congestion; scratch mathematical comparison of Gini vs Entropy; pruning overfitted trees; MDI feature ranking. |
| **CO6** | **Unit VI / Module 4**: Tree Models - II | Ensemble Methods, Bagging, Random Forests, Feature Subsampling, Out-of-Bag (OOB) error comprehension and mathematical proof, RF Regression & Classification | Random Forest ensemble classifier and regressor; tracking OOB misclassification error as $n_{\text{estimators}}$ scales; MDI importance with tree variance error bars. |

---

## 2. Unit-by-Unit Mathematical Formulations & Experimental Results

### Unit I (CO1): Model Selection and Evaluation

#### 1. The Bias-Variance Tradeoff
The generalization error of an estimator $\hat{f}(x)$ on target $y = f(x) + \epsilon$ (where $\mathbb{E}[\epsilon]=0, \text{Var}(\epsilon)=\sigma^2$) decomposes into:
$$\text{MSE}(x) = \mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}$$

- **Underfitting (High Bias)**: At `max_depth = 1`, Train MSE = 277.55, Test MSE = 294.28.
- **Optimal Tradeoff**: At `max_depth = 18`, Train MSE = 0.01, Test MSE = 1.17.
- **Overfitting (High Variance)**: At `max_depth = 20`, Train MSE = 0.00, Test MSE = 1.23.

#### 2. K-Fold Cross-Validation
- Evaluated Ridge Regression over 5 disjoint folds:
  - Fold 1: MAE = 2.72 min | $R^2 = 0.9906$
  - Fold 2: MAE = 2.70 min | $R^2 = 0.9910$
  - Fold 3: MAE = 2.77 min | $R^2 = 0.9902$
  - Fold 4: MAE = 2.74 min | $R^2 = 0.9904$
  - Fold 5: MAE = 2.69 min | $R^2 = 0.9913$
  - **Mean Across Folds**: $\text{MAE} = 2.72 \pm 0.03$ min, $R^2 = 0.9907 \pm 0.0004$. Demonstrates exceptional model stability across temporal partitions.

---

### Unit II (CO2): Advanced Regression Techniques

#### 1. Estimating Coefficients in SLR
- **Scalar Formulas**:
  $$\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$
- **Matrix Normal Equation**:
  $$\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T \mathbf{y}$$
- **Empirical Output**:
  - Closed-form scalar: $\hat{\beta}_0 = 33.1086, \hat{\beta}_1 = -0.0579$
  - Matrix solution: $\hat{\beta}_0 = 33.1086, \hat{\beta}_1 = -0.0579$
  - Scikit-Learn OLS: $\hat{\beta}_0 = 33.1086, \hat{\beta}_1 = -0.0579$ (100% exact numerical agreement).

#### 2. Multicollinearity Diagnostics via Variance Inflation Factor (VIF)
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
- `arriving_flights`, `departing_flights`, and `total_flights` exhibited $\text{VIF} \to \infty$ due to the exact identity $\text{Total} = \text{Arr} + \text{Dep}$.
- `avg_arrival_delay` ($\text{VIF} = 28.98$) and `taxi_out_avg` ($\text{VIF} = 24.59$) displayed high collinearity with systemic delay propagation.

#### 3. Regularization: Ridge ($L_2$) vs Lasso ($L_1$)
- **Ridge**: $\min_\beta \|y - X\beta\|_2^2 + \lambda \sum \beta_j^2$ $\to$ Smooth shrinkage, all features retained.
- **Lasso**: $\min_\beta \|y - X\beta\|_2^2 + \lambda \sum |\beta_j|$ $\to$ Performs sparse feature selection.
  - At $\alpha = 0.5$, Lasso **retained the 4 key operational drivers**: `['avg_arrival_delay', 'weather_delay_factor', 'taxi_out_avg', 'runway_utilization']`.
  - Zeroed out all redundant/collinear flight counters and calendar features.

---

### Unit III & IV (CO3 & CO4): Support Vector Machines & The Kernel Trick

#### 1. Hyperplanes, Margins & Slack Variables ($\xi_i$)
- Hyperplane: $\mathbf{w}^T \mathbf{x} + b = 0$, with margin corridor width $M = \frac{2}{\|\mathbf{w}\|}$.
- Soft Margin Objective:
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
- **Cost Parameter ($C$) Impact**:
  - Small $C = 0.05$: Wider margin ($M = 2.45$), tolerates higher slack penalty $\sum \xi_i$ (softer boundary).
  - High $C = 5.0$: Narrower margin ($M = 0.82$), heavily penalizes classification violations.

#### 2. The Kernel Trick & Comparison
Instead of explicitly mapping $\phi(\mathbf{x}) \in \mathbb{R}^D$ into higher/infinite dimensional spaces, the inner product is computed via kernel functions $K(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$:

| Kernel Type | Mathematical Formula | Test Accuracy | Support Vectors |
| :--- | :--- | :---: | :---: |
| **Linear Kernel** | $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$ | 92.01% | 3,449 |
| **Polynomial Kernel ($d=3$)** | $K(\mathbf{x}, \mathbf{x}') = (\gamma \mathbf{x}^T \mathbf{x}' + r)^d$ | 92.59% | 4,342 |
| **RBF (Gaussian) Kernel** | $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$ | **96.27%** | **2,782** |

The RBF Kernel achieved the highest accuracy (+4.26% over Linear) with the fewest support vectors, demonstrating the power of non-linear kernel transformations in aviation delay manifolds.

---

### Unit V (CO5): Decision Trees, Impurity & Pruning

#### 1. Gini Impurity vs Shannon Entropy
- **Gini Index**: $G = 1 - \sum_{k=1}^C p_k^2$
- **Shannon Entropy**: $H = -\sum_{k=1}^C p_k \log_2(p_k)$
- **Evaluation on Airport State Node ($N=100$: 50 Low, 30 Medium, 20 High)**:
  - Parent Gini Impurity = $1 - (0.5^2 + 0.3^2 + 0.2^2) = 0.6200$
  - Parent Shannon Entropy = $1.4855$ bits
  - Splitting on `runway_utilization > 75%` reduced weighted Gini impurity to $0.4895$, yielding a **Gini Information Gain of 0.1305**.

#### 2. Cost-Complexity Pruning (CCP)
- Cost function: $R_\alpha(T) = R(T) + \alpha |T|$.
- Pruning increased test set accuracy from 99.83% to 99.86% while cutting terminal leaves in half (from 30 to 15), eliminating spurious leaf branches.

---

### Unit VI (CO6): Ensemble Methods & Random Forests

#### 1. Out-of-Bag (OOB) Error Comprehension & Mathematical Proof
For a dataset of size $N$, drawing $N$ samples with replacement yields a bootstrap sample. The probability of any single instance NOT being chosen in a single draw is $1 - \frac{1}{N}$.
For $N$ independent draws:
$$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.3679 \approx 36.8\%$$
Therefore, **each decision tree in the Random Forest is trained on ~63.2% of the dataset, leaving ~36.8% as Out-of-Bag (OOB) samples**.
Evaluating each instance using only the subset of trees that did not contain it provides an **unbiased internal validation estimate**:

| Trees ($n_{\text{estimators}}$) | OOB Accuracy (%) | Test Set Accuracy (%) | Generalization Correlation |
| :---: | :---: | :---: | :---: |
| 10 | 98.81% | 99.68% | High initial variance |
| 25 | 99.68% | 99.77% | Rapid stabilization |
| 50 | 99.71% | 99.80% | Stable |
| 100 | 99.70% | 99.80% | Optimal convergence |
| 200 | 99.73% | 99.80% | Flat asymptotic plateau |

---

## 3. Comprehensive Master Benchmark Comparison

### A. Regression Benchmark (Target: `avg_departure_delay`)

| Model Architecture | Algorithm Paradigm | MAE (min) | RMSE (min) | $R^2$ Score | Rank |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Simple Linear Regression** | Single Feature Baseline | 31.03 | 39.38 | 0.0012 | 8 |
| **Polynomial Regression (Deg 2)** | Non-linear OLS | 11.30 | 16.78 | 0.8187 | 7 |
| **Multiple Linear Regression** | Parametric Linear | 2.75 | 3.85 | 0.9904 | 6 |
| **Ridge Regression ($L_2$)** | Regularized Linear | 2.74 | 3.85 | 0.9904 | 5 |
| **Lasso Regression ($L_1$)** | Sparse Regularized | 2.63 | 3.99 | 0.9898 | 4 |
| **Decision Tree Regressor** | Non-parametric Tree | 1.18 | 2.00 | 0.9974 | 3 |
| **Support Vector Regressor (SVR)** | Kernel Machine ($\epsilon$-tube) | 0.59 | 1.10 | 0.9992 | 2 |
| **Random Forest Regressor** | Bagged Ensemble | **0.35** | **0.87** | **0.9995** | **1 (Best)** |

### B. Classification Benchmark (Target: `congestion_level`)

| Model Architecture | Algorithm Paradigm | Accuracy (%) | Macro F1 (%) | Precision (%) | Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear Probability Model | 90.83% | 90.46% | 90.45% | 90.47% |
| **Linear SVC** | Maximal Margin Classifier | 91.20% | 90.91% | 90.90% | 91.12% |
| **Kernel SVM (RBF Kernel)** | Non-Linear Kernel Machine | 96.27% | 96.10% | 96.09% | 96.10% |
| **Random Forest Classifier** | Bagged Ensemble Trees | 99.80% | 99.79% | 99.80% | 99.78% |
| **Decision Tree Classifier** | Pruned CART Tree | **99.86%** | **99.85%** | **99.85%** | **99.84%** |

---

## 4. Software Deliverables & Interactive Tools

1. **Modular Python Scripts (`src/`)**:
   - [`src/data_loader.py`](file:///c:/Users/aupen/OneDrive/Desktop/AirTraffic_ML_Project/src/data_loader.py)
   - [`src/unit1_model_selection.py`](file:///c:/Users/aupen/OneDrive/Desktop/AirTraffic_ML_Project/src/unit1_model_selection.py) (CO1)
   - [`src/unit2_advanced_regression.py`](file:///c:/Users/aupen/OneDrive/Desktop/AirTraffic_ML_Project/src/unit2_advanced_regression.py) (CO2)
   - [`src/unit3_4_svm.py`](file:///c:/Users/aupen/OneDrive/Desktop/AirTraffic_ML_Project/src/unit3_4_svm.py) (CO3 & CO4)
   - [`src/unit5_decision_trees.py`](file:///c:/Users/aupen/OneDrive/Desktop/AirTraffic_ML_Project/src/unit5_decision_trees.py) (CO5)
   - [`src/unit6_random_forest.py`](file:///c:/Users/aupen/OneDrive/Desktop/AirTraffic_ML_Project/src/unit6_random_forest.py) (CO6)
   - [`src/master_pipeline.py`](file:///c:/Users/aupen/OneDrive/Desktop/AirTraffic_ML_Project/src/master_pipeline.py)
2. **Interactive Streamlit Web Dashboard**:
   - [`app_ml_studio.py`](file:///c:/Users/aupen/OneDrive/Desktop/AirTraffic_ML_Project/app_ml_studio.py)
   - Launch with 1-click script: [`run_ml_studio.bat`](file:///c:/Users/aupen/OneDrive/Desktop/AirTraffic_ML_Project/run_ml_studio.bat)
3. **Comprehensive Jupyter Notebook**:
   - [`Air_Traffic_Machine_Learning_Project_Notebook.ipynb`](file:///c:/Users/aupen/OneDrive/Desktop/AirTraffic_ML_Project/Air_Traffic_Machine_Learning_Project_Notebook.ipynb)
4. **Exported Visualizations (`plots/`)**:
   - `unit1_bias_variance_tradeoff.png`
   - `unit2_slr_fit.png`
   - `unit2_mlr_residual_diagnostics.png`
   - `unit2_regularization_paths.png`
   - `unit3_svm_2d_hyperplane_margins.png`
   - `unit3_svm_3d_hyperplane.png`
   - `unit5_cost_complexity_pruning.png`
   - `unit5_decision_tree_feature_importance.png`
   - `unit6_rf_oob_vs_test_error.png`
   - `unit6_rf_feature_importance.png`
   - `master_model_benchmark_comparison.png`

---

## 5. Conclusion
This project establishes that:
1. **Regularization and Feature Selection (CO1, CO2)**: In airport operational environments with multicollinear flight counts, Lasso ($L_1$) effectively eliminates redundant dimensions while maintaining near-perfect predictive accuracy ($R^2 > 0.989$).
2. **Kernel Transformations (CO3, CO4)**: Airport congestion boundaries are inherently non-linear; the RBF kernel elevates classification accuracy from 92.01% (Linear SVM) to 96.27% by operating in transformed Hilbert space without computational blowup.
3. **Tree & Ensemble Dominance (CO5, CO6)**: Tree ensembles (Random Forests) provide optimal continuous flight delay forecasting (MAE = 0.35 min, $R^2 = 0.9995$), with Out-of-Bag (OOB) error serving as an exceptional proxy for held-out generalization error.
4. **Integration with Real-Time Deep Learning System**: While this Machine Learning project handles cross-sectional delay forecasting, feature attribution, and linear/margin-based separation, it synergizes seamlessly with the LSTM/GRU recurrent network that forecasts temporal trajectories 60 minutes into the future.
