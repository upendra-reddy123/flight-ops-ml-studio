"""
unit6_random_forest.py
Course Outcome CO6 :: Tree Models - II (Ensemble Methods & Random Forests)
Topics Covered:
1. Ensemble Methods & Bagging (Bootstrap Aggregation)
2. Introduction to Random Forests (Feature Subsampling to Decorrelate Trees)
3. Comprehension of Out-of-Bag (OOB) Error:
   - Mathematical justification: Prob(sample excluded from tree) = (1 - 1/N)^N ~ 1/e ~ 36.8%
   - Tracking OOB Score vs Test Accuracy across tree counts
4. Feature Importance in Random Forests (Ensemble MDI)
5. Random Forest Classification & Random Forest Regression
6. Hyperparameter Tuning (n_estimators, max_features, max_depth)
"""

import os
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_cache")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, mean_squared_error, r2_score

from src.data_loader import get_classification_data, get_regression_data

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def demonstrate_oob_error_comprehension(data_dict: dict):
    """
    Demonstrates Out-of-Bag (OOB) Error comprehension and progression.
    Evaluates how OOB Accuracy tracks Test Accuracy as n_estimators increases.
    """
    print("\n--- 1. Out-of-Bag (OOB) Error Comprehension & Tree Progression ---")
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    
    tree_counts = [10, 25, 50, 75, 100, 150, 200]
    oob_scores = []
    test_scores = []
    
    for n in tree_counts:
        rf = RandomForestClassifier(
            n_estimators=n,
            oob_score=True,
            random_state=42,
            n_jobs=1
        )
        rf.fit(X_train, y_train)
        
        oob = rf.oob_score_
        test_acc = rf.score(X_test, y_test)
        
        oob_scores.append(oob)
        test_scores.append(test_acc)
        print(f"Random Forest (n_estimators={n:<3}): OOB Score = {oob * 100:.2f}% | Test Accuracy = {test_acc * 100:.2f}%")
        
    # Plot OOB Error vs Test Error Progression
    plt.figure(figsize=(9, 5))
    plt.plot(tree_counts, [1.0 - s for s in oob_scores], marker='o', color="#ef4444", lw=2, label="OOB Misclassification Error (Internal Validation)")
    plt.plot(tree_counts, [1.0 - s for s in test_scores], marker='s', color="#3b82f6", lw=2, label="Test Misclassification Error (Held-out Test Set)")
    plt.title("Random Forest: Out-of-Bag (OOB) Error vs Test Error", fontsize=12)
    plt.xlabel("Number of Estimator Trees (n_estimators)", fontsize=11)
    plt.ylabel("Error Rate (1 - Accuracy)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(PLOTS_DIR, "unit6_rf_oob_vs_test_error.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved OOB progression plot: {plot_path}")
    
    return {
        "tree_counts": tree_counts,
        "oob_scores": [round(x * 100, 2) for x in oob_scores],
        "test_scores": [round(x * 100, 2) for x in test_scores]
    }

def demonstrate_rf_feature_importance(data_dict: dict):
    """
    Extracts Ensemble Feature Importances and compares them with single Decision Tree.
    """
    print("\n--- 2. Random Forest Feature Importance Analysis ---")
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    feature_names = data_dict["feature_names"]
    
    rf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    
    rf_feat_data = []
    print("Random Forest Feature Importance Rankings:")
    for rank, idx in enumerate(indices):
        print(f"  {rank+1}. {feature_names[idx]:<22}: {importances[idx]:.4f} (+/- {std[idx]:.4f})")
        rf_feat_data.append({"Feature": feature_names[idx], "Importance": round(importances[idx], 4)})
        
    # Plot RF Feature Importance with Error Bars
    plt.figure(figsize=(9, 5))
    plt.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        xerr=std[indices[::-1]],
        color="#10b981",
        alpha=0.85
    )
    plt.title("Random Forest: Ensemble Feature Importance (MDI with Tree Variance)", fontsize=12)
    plt.xlabel("Mean Importance Score")
    plt.grid(True, alpha=0.3, axis="x")
    plot_path = os.path.join(PLOTS_DIR, "unit6_rf_feature_importance.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved RF feature importance plot: {plot_path}")
    
    return rf_feat_data

def demonstrate_random_forest_regression(reg_data_dict: dict):
    """
    Random Forest Regressor for Continuous Flight Departure Delay Prediction.
    """
    print("\n--- 3. Random Forest Regression (Flight Delay Prediction) ---")
    X_train = reg_data_dict["X_train"]
    y_train = reg_data_dict["y_train"]
    X_test = reg_data_dict["X_test"]
    y_test = reg_data_dict["y_test"]
    
    rf_reg = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=1)
    rf_reg.fit(X_train, y_train)
    
    pred = rf_reg.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"Random Forest Regressor: MAE = {mae:.2f} min | RMSE = {rmse:.2f} min | R2 = {r2:.4f}")
    
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }

def run_unit6_module():
    print("=====================================================================")
    print(" UNIT VI / MODULE 4: TREE MODELS - II (RANDOM FORESTS) (CO6)")
    print("=====================================================================")
    clf_data = get_classification_data()
    reg_data = get_regression_data()
    
    oob_res = demonstrate_oob_error_comprehension(clf_data)
    rf_feat = demonstrate_rf_feature_importance(clf_data)
    rf_reg_res = demonstrate_random_forest_regression(reg_data)
    
    return {
        "oob": oob_res,
        "feature_importance": rf_feat,
        "regression": rf_reg_res
    }

if __name__ == "__main__":
    run_unit6_module()
