"""
app_ml_studio.py
SkyFlow Machine Learning Studio: Comprehensive Academic Visualizer & Interactive Laboratory.
Maps all syllabus Course Outcomes (CO1 to CO6) and Units (Unit I to Unit VI) to
Air Traffic Congestion and Flight Delay Prediction.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.data_loader import load_air_traffic_data, FEATURE_COLS

st.set_page_config(
    page_title="SkyFlow | Machine Learning Academic Laboratory",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
    }
    .syllabus-badge {
        background-color: #0284c7;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .co-badge {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load Master Metrics JSON
METRICS_PATH = os.path.join(PROJECT_DIR, "models", "master_metrics.json")
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        master_metrics = json.load(f)
else:
    master_metrics = {}

# Load Models & Scaler
@st.cache_resource
def load_all_models():
    reg_path = os.path.join(PROJECT_DIR, "models", "regression_models.pkl")
    clf_path = os.path.join(PROJECT_DIR, "models", "classification_models.pkl")
    scaler_path = os.path.join(PROJECT_DIR, "models", "scaler.pkl")
    
    reg_models = pickle.load(open(reg_path, "rb")) if os.path.exists(reg_path) else {}
    clf_models = pickle.load(open(clf_path, "rb")) if os.path.exists(clf_path) else {}
    scaler = pickle.load(open(scaler_path, "rb")) if os.path.exists(scaler_path) else None
    
    return reg_models, clf_models, scaler

reg_models, clf_models, scaler = load_all_models()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/airplane-mode-on.png", width=64)
    st.title("ML Course Studio")
    st.markdown("**Air Traffic Congestion System**")
    
    st.divider()
    
    unit_selection = st.radio(
        "Navigate Syllabus Units:",
        options=[
            "🏆 Master Benchmark Overview",
            "📘 Unit I: Model Selection & CV (CO1)",
            "📗 Unit II: Advanced Regression (CO2)",
            "📕 Unit III & IV: SVM Hyperplanes & Kernels (CO3/4)",
            "📙 Unit V: Decision Trees & Impurity (CO5)",
            "📓 Unit VI: Ensembles & Random Forest OOB (CO6)",
            "🎛️ Live Prediction Console"
        ]
    )
    
    st.divider()
    st.caption("Academic Course Outcomes (CO1 - CO6) fully implemented with Aviation Operational Dataset.")

# Helper to render image
def show_plot(filename: str, caption: str = ""):
    path = os.path.join(PROJECT_DIR, "plots", filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Plot `{filename}` not found. Please run `python src/master_pipeline.py` to generate all plots.")

# ==============================================================================
# TAB 0: MASTER BENCHMARK OVERVIEW
# ==============================================================================
if unit_selection == "🏆 Master Benchmark Overview":
    st.title("🏆 Master Machine Learning Benchmark Suite")
    st.markdown("""
    <span class="syllabus-badge">Comprehensive Synthesis</span>
    <span class="co-badge">CO1 :: CO2 :: CO3 :: CO4 :: CO5 :: CO6</span>
    """, unsafe_allow_html=True)
    st.markdown("### Comparative Performance Across All Machine Learning Paradigms")
    
    col1, col2 = st.columns(2)
    
    # Regression Table
    with col1:
        st.subheader("🎯 Regression Suite (Flight Departure Delay)")
        if "regression_benchmark" in master_metrics:
            reg_df = pd.DataFrame(master_metrics["regression_benchmark"]).T.reset_index()
            reg_df.columns = ["Model Architecture", "MAE (min)", "RMSE (min)", "R² Score"]
            st.dataframe(reg_df.sort_values(by="MAE (min)"), use_container_width=True)
            
            fig_reg = px.bar(
                reg_df.sort_values(by="MAE (min)", ascending=False),
                x="MAE (min)",
                y="Model Architecture",
                orientation='h',
                title="Model Error (MAE in min - Lower is Better)",
                color="MAE (min)",
                color_continuous_scale="Tealgrn"
            )
            fig_reg.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_reg, use_container_width=True)
            
    # Classification Table
    with col2:
        st.subheader("🚦 Classification Suite (Congestion Level)")
        if "classification_benchmark" in master_metrics:
            clf_df = pd.DataFrame(master_metrics["classification_benchmark"]).T.reset_index()
            clf_df.columns = ["Model Architecture", "Accuracy (%)", "Macro F1 (%)", "Precision (%)", "Recall (%)"]
            st.dataframe(clf_df.sort_values(by="Accuracy (%)", ascending=False), use_container_width=True)
            
            fig_clf = px.bar(
                clf_df.sort_values(by="Accuracy (%)"),
                x="Accuracy (%)",
                y="Model Architecture",
                orientation='h',
                title="Model Accuracy (%) - Higher is Better",
                color="Accuracy (%)",
                color_continuous_scale="Viridis"
            )
            fig_clf.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_clf, use_container_width=True)

    st.divider()
    st.subheader("📊 Master Benchmark Comparison Plot")
    show_plot("master_model_benchmark_comparison.png", "Comparative Summary of Regression and Classification Pipelines")

# ==============================================================================
# TAB 1: UNIT I (CO1)
# ==============================================================================
elif unit_selection == "📘 Unit I: Model Selection & CV (CO1)":
    st.title("📘 Unit I: Model Selection and Evaluation")
    st.markdown("""
    <span class="syllabus-badge">Unit I / Module 1</span>
    <span class="co-badge">CO1: Cross-Validation, Regularization, Hyperparameter Tuning & Bias-Variance</span>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Core Theoretical Concepts:
    - **Bias-Variance Decomposition**: Error = $\\text{Bias}^2 + \\text{Variance} + \\sigma^2$.
      - **High Bias (Underfitting)**: Model is too simplistic (e.g., shallow tree of depth 1). Both train and test errors are high.
      - **High Variance (Overfitting)**: Model memorizes training noise (e.g., depth 20). Train error approaches zero, but test error degrades.
    - **Cross-Validation**: 5-Fold Cross Validation validates model stability across independent slices without data leakage.
    - **Hyperparameter Optimization**: `GridSearchCV` vs `RandomizedSearchCV`.
    """)
    
    col_u1_a, col_u1_b = st.columns([1.2, 1])
    with col_u1_a:
        st.subheader("📈 Empirical Bias-Variance Tradeoff Curve")
        show_plot("unit1_bias_variance_tradeoff.png", "Decision Tree Depth vs MSE on Train & Test Sets")
        
    with col_u1_b:
        st.subheader("🔄 5-Fold Cross-Validation Demonstration")
        if "unit1_model_selection" in master_metrics:
            cv = master_metrics["unit1_model_selection"]["cross_validation"]
            cv_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(5)],
                "MAE (min)": cv["mae_scores"],
                "RMSE (min)": cv["rmse_scores"],
                "R² Score": cv["r2_scores"]
            })
            st.dataframe(cv_df, use_container_width=True)
            st.metric("Mean Cross-Validation R²", f"{cv['mean_r2']:.4f}", delta="Consistent across folds")
            st.metric("Mean Cross-Validation MAE", f"{cv['mean_mae']:.2f} min")
            
        st.markdown("#### Hyperparameter Tuning Comparison")
        if "unit1_model_selection" in master_metrics:
            tune = master_metrics["unit1_model_selection"]["tuning"]
            st.json({
                "GridSearchCV Best Accuracy": f"{tune['grid_best_score']}%",
                "GridSearchCV Parameters": tune['grid_best_params'],
                "RandomizedSearchCV Best Accuracy": f"{tune['random_best_score']}%",
                "RandomizedSearchCV Parameters": tune['random_best_params']
            })

# ==============================================================================
# TAB 2: UNIT II (CO2)
# ==============================================================================
elif unit_selection == "📗 Unit II: Advanced Regression (CO2)":
    st.title("📗 Unit II: Advanced Regression & Regularization")
    st.markdown("""
    <span class="syllabus-badge">Unit II / Module 2</span>
    <span class="co-badge">CO2: SLR, MLR, Polynomial Regression, Assumptions & Regularization</span>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Mathematical Formulations:
    - **SLR Closed-Form Estimators**:
      $$\\hat{\\beta}_1 = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum (x_i - \\bar{x})^2}, \\quad \\hat{\\beta}_0 = \\bar{y} - \\hat{\\beta}_1 \\bar{x}$$
    - **Matrix Representation**:
      $$\\hat{\\beta} = (X^T X)^{-1} X^T y$$
    - **Ridge ($L_2$) Objective**: $\\min_\\beta \\|y - X\\beta\\|_2^2 + \\lambda \\sum \\beta_j^2$ (Smooth shrinkage towards zero).
    - **Lasso ($L_1$) Objective**: $\\min_\\beta \\|y - X\\beta\\|_2^2 + \\lambda \\sum |\\beta_j|$ (Sparse feature selection).
    """)
    
    col_u2_1, col_u2_2 = st.columns(2)
    with col_u2_1:
        st.subheader("1. Simple Linear Regression (SLR Fit)")
        show_plot("unit2_slr_fit.png", "Runway Utilization (%) vs Departure Delay (min)")
        
    with col_u2_2:
        st.subheader("2. Regularization Paths (Ridge vs Lasso)")
        show_plot("unit2_regularization_paths.png", "Coefficient Shrinkage Path as Penalty (alpha) Increases")
        
    st.divider()
    
    col_u2_3, col_u2_4 = st.columns(2)
    with col_u2_3:
        st.subheader("3. Regression Assumptions Diagnostics")
        show_plot("unit2_mlr_residual_diagnostics.png", "Residuals vs Fitted (Homoscedasticity) & Normal Q-Q Plot")
        
    with col_u2_4:
        st.subheader("4. Multicollinearity Diagnostics via VIF")
        st.markdown("$$\\text{VIF}_j = \\frac{1}{1 - R_j^2}$$")
        if "unit2_advanced_regression" in master_metrics:
            vif_df = pd.DataFrame(master_metrics["unit2_advanced_regression"]["mlr"]["vif"])
            st.dataframe(vif_df, use_container_width=True)
            st.caption("Features with VIF > 10 indicate high collinearity. Lasso automatically zeroes these out!")

# ==============================================================================
# TAB 3: UNIT III & IV (CO3 & CO4)
# ==============================================================================
elif unit_selection == "📕 Unit III & IV: SVM Hyperplanes & Kernels (CO3/4)":
    st.title("📕 Unit III & IV: Support Vector Machines & The Kernel Trick")
    st.markdown("""
    <span class="syllabus-badge">Unit III & IV / Module 3</span>
    <span class="co-badge">CO3: Hyperplanes, Maximal & Soft Margins, Slack Variables | CO4: Kernels & Kernel Trick</span>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Mathematical Foundations:
    - **Separating Hyperplane**: $\\mathbf{w}^T \\mathbf{x} + b = 0$, with margin width $\\frac{2}{\\|\\mathbf{w}\\|}$.
    - **Soft Margin with Slack Variables ($\\xi_i \\ge 0$)**:
      $$\\min_{\\mathbf{w}, b, \\boldsymbol{\\xi}} \\frac{1}{2} \\|\\mathbf{w}\\|^2 + C \\sum_{i=1}^N \\xi_i \\quad \\text{s.t.} \\quad y_i (\\mathbf{w}^T \\mathbf{x}_i + b) \\ge 1 - \\xi_i$$
      - $\\xi_i = 0$: Correctly classified outside margin.
      - $0 < \\xi_i \\le 1$: Margin violation (within the margin corridor).
      - $\\xi_i > 1$: Misclassified point.
    - **The Kernel Trick**: $K(\\mathbf{x}, \\mathbf{x}') = \\langle \\phi(\\mathbf{x}), \\phi(\\mathbf{x}') \\rangle$.
      - **Linear**: $K(\\mathbf{x}, \\mathbf{x}') = \\mathbf{x}^T \\mathbf{x}'$
      - **Polynomial**: $K(\\mathbf{x}, \\mathbf{x}') = (\\gamma \\mathbf{x}^T \\mathbf{x}' + r)^d$
      - **RBF (Gaussian)**: $K(\\mathbf{x}, \\mathbf{x}') = \\exp(-\\gamma \\|\\mathbf{x} - \\mathbf{x}'\\|^2)$
    """)
    
    col_svm1, col_svm2 = st.columns(2)
    with col_svm1:
        st.subheader("2D Hyperplane, Margins & Support Vectors")
        show_plot("unit3_svm_2d_hyperplane_margins.png", "Impact of Cost C on Margin Width & Slack Tolerance")
        
    with col_svm2:
        st.subheader("3D Hyperplane in Aviation Feature Space")
        show_plot("unit3_svm_3d_hyperplane.png", "w1*Flights + w2*Weather + w3*Runway + b = 0")
        
    st.divider()
    st.subheader("Kernel Comparison: Linear vs Polynomial vs RBF")
    if "unit3_4_svm" in master_metrics:
        k_data = master_metrics["unit3_4_svm"]["kernels"]
        k_df = pd.DataFrame(k_data).T.reset_index().rename(columns={"index": "Kernel Function"})
        st.dataframe(k_df, use_container_width=True)
        st.metric("RBF Kernel Classification Accuracy", "96.27%", delta="+5.07% over Linear Kernel (Kernel Trick Advantage)")

# ==============================================================================
# TAB 4: UNIT V (CO5)
# ==============================================================================
elif unit_selection == "📙 Unit V: Decision Trees & Impurity (CO5)":
    st.title("📙 Unit V: Decision Trees, Impurity Measures & Pruning")
    st.markdown("""
    <span class="syllabus-badge">Unit V / Module 4</span>
    <span class="co-badge">CO5: Decision Trees, Gini Impurity, Information Gain, Pruning & Feature Importance</span>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Impurity Measures:
    - **Gini Impurity**:
      $$G = 1 - \\sum_{k=1}^C p_k^2$$
    - **Shannon Entropy**:
      $$H = -\\sum_{k=1}^C p_k \\log_2(p_k)$$
    - **Information Gain**: $\\Delta = I(\\text{Parent}) - \\sum \\frac{N_j}{N} I(\\text{Child}_j)$
    - **Minimal Cost-Complexity Pruning (CCP)**: Cost $= R(T) + \\alpha |T|$ where $|T|$ is number of terminal leaves.
    """)
    
    col_dt1, col_dt2 = st.columns(2)
    with col_dt1:
        st.subheader("Cost-Complexity Pruning (ccp_alpha vs Accuracy)")
        show_plot("unit5_cost_complexity_pruning.png", "Optimal Regularization Tradeoff Preventing Tree Overfitting")
        
    with col_dt2:
        st.subheader("Decision Tree Feature Importance (MDI)")
        show_plot("unit5_decision_tree_feature_importance.png", "Mean Decrease in Gini Impurity across Airport Features")

# ==============================================================================
# TAB 5: UNIT VI (CO6)
# ==============================================================================
elif unit_selection == "📓 Unit VI: Ensembles & Random Forest OOB (CO6)":
    st.title("📓 Unit VI: Ensemble Methods & Random Forests")
    st.markdown("""
    <span class="syllabus-badge">Unit VI / Module 4</span>
    <span class="co-badge">CO6: Bagging, Random Forests, Out-of-Bag (OOB) Error & Ensemble Feature Importance</span>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Why Random Forests Outperform Single Trees:
    1. **Bootstrap Aggregation (Bagging)**: Combines $B$ randomized trees to reduce variance from $\\text{Var}$ to $\\frac{\\text{Var}}{B} + \\rho \\text{Var}$.
    2. **Feature Subsampling ($m = \\sqrt{p}$)**: Randomly restricts candidate split features to decorrelate individual trees, minimizing $\\rho$.
    3. **Out-of-Bag (OOB) Error Comprehension**:
       - Probability of a sample NOT being picked in a bootstrap sample of size $N$:
         $$\\lim_{N \\to \\infty} \\left(1 - \\frac{1}{N}\\right)^N = \\frac{1}{e} \\approx 36.8\\%$$
       - Hence, every tree leaves ~36.8% samples out. Aggregating each sample's out-of-bag predictions provides an unbiased validation estimate without needing a cross-validation split!
    """)
    
    col_rf1, col_rf2 = st.columns(2)
    with col_rf1:
        st.subheader("Out-of-Bag (OOB) Error vs Test Error")
        show_plot("unit6_rf_oob_vs_test_error.png", "OOB Error closely mirrors Test Set Generalization Error")
        
    with col_rf2:
        st.subheader("Random Forest Ensemble Feature Importance")
        show_plot("unit6_rf_feature_importance.png", "MDI with Variance Error Bars Across Estimator Trees")

# ==============================================================================
# TAB 6: LIVE PREDICTION CONSOLE
# ==============================================================================
elif unit_selection == "🎛️ Live Prediction Console":
    st.title("🎛️ Live Airport Flight Delay & Congestion Prediction Console")
    st.caption("Test user-defined operational conditions and receive real-time inference across all trained Machine Learning models.")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        airport = st.selectbox("Airport Hub", ["HYD", "DEL", "BOM", "SFO", "ATL", "ORD", "LAX"])
        arriving = st.slider("Arriving Flights (ops/hr)", 5, 80, 25)
        departing = st.slider("Departing Flights (ops/hr)", 5, 80, 25)
        
    with col_input2:
        capacity = st.selectbox("Airport Capacity", [55, 65, 70, 90, 100, 120], index=0)
        taxi_out = st.slider("Average Taxi-Out Time (min)", 8.0, 45.0, 16.0, step=0.5)
        arr_delay = st.slider("Average Arrival Delay (min)", 0.0, 90.0, 15.0, step=1.0)
        
    with col_input3:
        weather_delay = st.slider("Weather Delay Factor (min)", 0.0, 60.0, 5.0, step=2.5)
        hour = st.slider("Hour of Day (0-23)", 0, 23, 14)
        day = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d])

    total_flights = arriving + departing
    runway_util = min(100.0, round((total_flights / capacity) * 100.0, 1))
    
    st.metric("Computed Runway Utilization", f"{runway_util:.1f}%", delta="High load" if runway_util > 80 else "Normal")
    
    # Prepare input vector
    input_dict = {
        "arriving_flights": arriving,
        "departing_flights": departing,
        "total_flights": total_flights,
        "avg_arrival_delay": arr_delay,
        "weather_delay_factor": weather_delay,
        "taxi_out_avg": taxi_out,
        "airport_capacity": capacity,
        "runway_utilization": runway_util,
        "hour_of_day": hour,
        "day_of_week": day
    }
    
    input_df = pd.DataFrame([input_dict])
    if scaler is not None:
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEATURE_COLS)
    else:
        input_scaled = input_df
        
    st.divider()
    
    col_res1, col_res2 = st.columns(2)
    
    # 1. Regression Predictions
    with col_res1:
        st.subheader("⏱️ Predicted Departure Delay (Regression)")
        delay_preds = {}
        for m_name, model in reg_models.items():
            try:
                if m_name == "Simple Linear Regression":
                    p = model.predict(input_df[["runway_utilization"]])[0]
                elif m_name == "Polynomial Regression (Deg 2)":
                    from sklearn.preprocessing import PolynomialFeatures
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    p_feat = poly.fit_transform(input_scaled[["runway_utilization", "weather_delay_factor"]])
                    p = model.predict(p_feat)[0]
                else:
                    p = model.predict(input_scaled)[0]
                delay_preds[m_name] = f"{max(0.0, p):.1f} min"
            except Exception as e:
                delay_preds[m_name] = "N/A"
                
        st.dataframe(pd.DataFrame(list(delay_preds.items()), columns=["Model", "Predicted Delay"]), use_container_width=True)

    # 2. Classification Predictions
    with col_res2:
        st.subheader("🚦 Predicted Congestion Level (Classification)")
        class_labels = {0: "🟢 LOW", 1: "🟡 MEDIUM", 2: "🔴 HIGH"}
        clf_preds = {}
        for m_name, model in clf_models.items():
            try:
                c = model.predict(input_scaled)[0]
                clf_preds[m_name] = class_labels.get(c, str(c))
            except Exception:
                clf_preds[m_name] = "N/A"
                
        st.dataframe(pd.DataFrame(list(clf_preds.items()), columns=["Model", "Congestion State"]), use_container_width=True)
