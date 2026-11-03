"""
master_pipeline.py
Master Execution Pipeline for Air Traffic Machine Learning Project.
Executes and evaluates models across all 6 Units (CO1 to CO6):
- Regression Models: SLR, MLR, Poly, Ridge, Lasso, SVR, Decision Tree, Random Forest
- Classification Models: Logistic Regression, Linear SVC, Kernel SVM (RBF), Decision Tree, Random Forest
Saves all model binaries, generates comparison plots, and dumps master_metrics.json.
"""

import os
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_cache")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

from src.data_loader import get_regression_data, get_classification_data, load_air_traffic_data
from src.unit1_model_selection import run_unit1_module
from src.unit2_advanced_regression import run_unit2_module
from src.unit3_4_svm import run_unit3_4_module
from src.unit5_decision_trees import run_unit5_module
from src.unit6_random_forest import run_unit6_module

MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def train_and_evaluate_all():
    print("=" * 80)
    print(" SKYFLOW: MACHINE LEARNING COURSE PROJECT - MASTER BENCHMARK PIPELINE")
    print("=" * 80)
    
    # 1. Run Unit Demonstrations
    print("\n>>> Executing Unit I: Model Selection & Cross-Validation (CO1)...")
    u1_res = run_unit1_module()
    
    print("\n>>> Executing Unit II: Advanced Regression Techniques (CO2)...")
    u2_res = run_unit2_module()
    
    print("\n>>> Executing Unit III & IV: Support Vector Machines & Kernels (CO3 & CO4)...")
    u34_res = run_unit3_4_module()
    
    print("\n>>> Executing Unit V: Decision Trees & Impurity (CO5)...")
    u5_res = run_unit5_module()
    
    print("\n>>> Executing Unit VI: Ensembles & Random Forests (CO6)...")
    u6_res = run_unit6_module()
    
    # 2. Master Comparison Setup
    reg_data = get_regression_data()
    clf_data = get_classification_data()
    
    X_reg_tr, y_reg_tr = reg_data["X_train"], reg_data["y_train"]
    X_reg_te, y_reg_te = reg_data["X_test"], reg_data["y_test"]
    
    X_clf_tr, y_clf_tr = clf_data["X_train"], clf_data["y_train"]
    X_clf_te, y_clf_te = clf_data["X_test"], clf_data["y_test"]
    
    # Save Feature Scaler
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(reg_data["scaler"], f)

    # -------------------------------------------------------------
    # REGRESSION BENCHMARK SUITE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" RUNNING SYSTEMATIC REGRESSION BENCHMARK (TARGET: avg_departure_delay)")
    print("=" * 80)
    
    reg_models = {
        "Simple Linear Regression": LinearRegression(),
        "Multiple Linear Regression": LinearRegression(),
        "Polynomial Regression (Deg 2)": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=10.0),
        "Lasso Regression (L1)": Lasso(alpha=0.5),
        "Support Vector Regressor (SVR)": SVR(kernel="rbf", C=10.0, epsilon=0.2),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=1)
    }
    
    poly_trans = PolynomialFeatures(degree=2, include_bias=False)
    X_reg_tr_poly = poly_trans.fit_transform(X_reg_tr[["runway_utilization", "weather_delay_factor"]])
    X_reg_te_poly = poly_trans.transform(X_reg_te[["runway_utilization", "weather_delay_factor"]])
    
    reg_results = {}
    saved_reg_models = {}
    
    for name, model in reg_models.items():
        if name == "Simple Linear Regression":
            model.fit(X_reg_tr[["runway_utilization"]], y_reg_tr)
            preds = model.predict(X_reg_te[["runway_utilization"]])
        elif name == "Polynomial Regression (Deg 2)":
            model.fit(X_reg_tr_poly, y_reg_tr)
            preds = model.predict(X_reg_te_poly)
        else:
            model.fit(X_reg_tr, y_reg_tr)
            preds = model.predict(X_reg_te)
            
        mae = mean_absolute_error(y_reg_te, preds)
        rmse = np.sqrt(mean_squared_error(y_reg_te, preds))
        r2 = r2_score(y_reg_te, preds)
        
        reg_results[name] = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        }
        saved_reg_models[name] = model
        print(f"  {name:<30} | MAE: {mae:<5.2f} min | RMSE: {rmse:<5.2f} min | R2: {r2:.4f}")
        
    with open(os.path.join(MODELS_DIR, "regression_models.pkl"), "wb") as f:
        pickle.dump(saved_reg_models, f)

    # -------------------------------------------------------------
    # CLASSIFICATION BENCHMARK SUITE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" RUNNING SYSTEMATIC CLASSIFICATION BENCHMARK (TARGET: congestion_level)")
    print("=" * 80)
    
    clf_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear Support Vector Classifier": LinearSVC(max_iter=3000, random_state=42),
        "Kernel SVM (RBF Kernel)": SVC(kernel="rbf", C=5.0, probability=True, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=12, oob_score=True, random_state=42, n_jobs=1)
    }
    
    clf_results = {}
    saved_clf_models = {}
    
    for name, model in clf_models.items():
        model.fit(X_clf_tr, y_clf_tr)
        preds = model.predict(X_clf_te)
        
        acc = accuracy_score(y_clf_te, preds)
        f1 = f1_score(y_clf_te, preds, average="macro", zero_division=0)
        prec = precision_score(y_clf_te, preds, average="macro", zero_division=0)
        rec = recall_score(y_clf_te, preds, average="macro", zero_division=0)
        
        clf_results[name] = {
            "Accuracy": round(acc * 100, 2),
            "Macro_F1": round(f1 * 100, 2),
            "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2)
        }
        saved_clf_models[name] = model
        print(f"  {name:<35} | Acc: {acc*100:<5.2f}% | F1: {f1*100:<5.2f}% | Prec: {prec*100:<5.2f}% | Rec: {rec*100:<5.2f}%")
        
    with open(os.path.join(MODELS_DIR, "classification_models.pkl"), "wb") as f:
        pickle.dump(saved_clf_models, f)

    # 3. Export Unified Master Metrics
    master_summary = {
        "regression_benchmark": reg_results,
        "classification_benchmark": clf_results,
        "unit1_model_selection": u1_res,
        "unit2_advanced_regression": u2_res,
        "unit3_4_svm": u34_res,
        "unit5_decision_trees": u5_res,
        "unit6_random_forest": u6_res
    }
    
    metrics_json_path = os.path.join(MODELS_DIR, "master_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(master_summary, f, indent=2)
    print(f"\nSaved master metrics to: {metrics_json_path}")

    # 4. Generate Master Comparison Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Regression MAE Bar Chart
    reg_names = list(reg_results.keys())
    reg_maes = [reg_results[k]["MAE"] for k in reg_names]
    axes[0].barh(reg_names[::-1], reg_maes[::-1], color="#38bdf8")
    axes[0].set_title("Regression Benchmark: Mean Absolute Error (Lower is Better)", fontsize=12)
    axes[0].set_xlabel("MAE (Minutes of Delay)")
    axes[0].grid(True, alpha=0.3, axis="x")
    
    # Classification Accuracy Bar Chart
    clf_names = list(clf_results.keys())
    clf_accs = [clf_results[k]["Accuracy"] for k in clf_names]
    axes[1].barh(clf_names[::-1], clf_accs[::-1], color="#22c55e")
    axes[1].set_title("Classification Benchmark: Accuracy (%) (Higher is Better)", fontsize=12)
    axes[1].set_xlabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    chart_path = os.path.join(PLOTS_DIR, "master_model_benchmark_comparison.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved master benchmark plot: {chart_path}")
    
    print("\n========================================================")
    print(" ALL 6 UNITS AND MASTER BENCHMARKS EXECUTED SUCCESSFULLY!")
    print("========================================================")
    return master_summary

if __name__ == "__main__":
    train_and_evaluate_all()
