"""
unit5_decision_trees.py
Course Outcome CO5 :: Tree Models - I (Decision Trees)
Topics Covered:
1. Decision Tree Classification & Regression
2. Splitting & Impurity Measures:
   - Gini Index: G = 1 - sum(p_i^2)
   - Cross-Entropy: H = -sum(p_i * log2(p_i))
   - Manual computation vs Scikit-Learn
3. Tree Truncation & Hyperparameters:
   - max_depth, min_samples_split, min_samples_leaf, ccp_alpha
4. Feature Importance Calculation (Mean Decrease in Impurity - MDI)
5. Overfitting Disadvantages & Cost-Complexity Pruning
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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, mean_squared_error, r2_score

from src.data_loader import get_classification_data, get_regression_data

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def demonstrate_impurity_measures():
    """
    Compares Gini Index vs Entropy mathematically from scratch on candidate traffic splits.
    """
    print("\n--- 1. Impurity Measures: Gini Index vs Entropy Calculation ---")
    
    # Example Node: 100 observations of congestion states: [50 Low, 30 Medium, 20 High]
    counts = np.array([50, 30, 20])
    p = counts / counts.sum()
    
    # Gini Index = 1 - sum(p_i^2)
    gini = 1.0 - np.sum(p**2)
    
    # Entropy = - sum(p_i * log2(p_i))
    entropy = - np.sum([prob * np.log2(prob) for prob in p if prob > 0])
    
    print(f"Parent Node Distribution: {counts.tolist()} (Total: {counts.sum()})")
    print(f"  Class Probabilities: p = {np.round(p, 2).tolist()}")
    print(f"  Gini Impurity = 1 - (0.5^2 + 0.3^2 + 0.2^2) = {gini:.4f} (Max for 3 classes = 1 - 1/3 = 0.667)")
    print(f"  Shannon Entropy = - (0.5*log2(0.5) + ...)     = {entropy:.4f} bits (Max for 3 classes = log2(3) = 1.585)")
    
    # Candidate Split A (Splitting on Runway Utilization > 75%):
    # Left Child: [5 Low, 10 Medium, 19 High] -> 34 flights
    # Right Child: [45 Low, 20 Medium, 1 High] -> 66 flights
    left = np.array([5, 10, 19])
    right = np.array([45, 20, 1])
    
    p_left = left / left.sum()
    p_right = right / right.sum()
    
    gini_left = 1.0 - np.sum(p_left**2)
    gini_right = 1.0 - np.sum(p_right**2)
    
    weighted_gini_split = (left.sum() / 100) * gini_left + (right.sum() / 100) * gini_right
    gini_gain = gini - weighted_gini_split
    
    print(f"\nCandidate Split on Runway Utilization > 75%:")
    print(f"  Left Child Gini: {gini_left:.4f} | Right Child Gini: {gini_right:.4f}")
    print(f"  Weighted Split Gini: {weighted_gini_split:.4f}")
    print(f"  Gini Gain (Information Gain in Impurity): {gini_gain:.4f}")
    
    return {
        "parent_gini": round(gini, 4),
        "parent_entropy": round(entropy, 4),
        "split_gini": round(weighted_gini_split, 4),
        "gini_gain": round(gini_gain, 4)
    }

def demonstrate_tree_pruning_and_hyperparameters(data_dict: dict):
    """
    Demonstrates Decision Tree Classifier with varying hyperparameters:
    - Unconstrained Tree (overfitted)
    - Pruned Tree (max_depth=5, min_samples_leaf=10)
    - Cost-Complexity Pruning (ccp_alpha)
    """
    print("\n--- 2. Tree Hyperparameter Tuning & Pruning ---")
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    
    # 1. Unconstrained Deep Tree
    dt_deep = DecisionTreeClassifier(random_state=42)
    dt_deep.fit(X_train, y_train)
    acc_deep_train = dt_deep.score(X_train, y_train)
    acc_deep_test = dt_deep.score(X_test, y_test)
    print(f"Unconstrained Tree: Depth = {dt_deep.get_depth()}, Leaves = {dt_deep.get_n_leaves()}")
    print(f"  Train Accuracy: {acc_deep_train * 100:.2f}% | Test Accuracy: {acc_deep_test * 100:.2f}% (High Overfitting Gap!)")
    
    # 2. Pruned Tree with Hyperparameter Truncation
    dt_pruned = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    dt_pruned.fit(X_train, y_train)
    acc_pruned_train = dt_pruned.score(X_train, y_train)
    acc_pruned_test = dt_pruned.score(X_test, y_test)
    print(f"Pruned Tree (max_depth=5, min_samples_leaf=10): Leaves = {dt_pruned.get_n_leaves()}")
    print(f"  Train Accuracy: {acc_pruned_train * 100:.2f}% | Test Accuracy: {acc_pruned_test * 100:.2f}% (Balanced Generalization)")
    
    # 3. Minimal Cost-Complexity Pruning Path
    path = dt_deep.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[::15] # sample points along path
    clfs = []
    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
        
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    
    # Plot Cost-Complexity Pruning curve
    plt.figure(figsize=(8, 5))
    plt.plot(ccp_alphas, train_scores, marker='o', label="Train Accuracy", drawstyle="steps-post", color="#2563eb")
    plt.plot(ccp_alphas, test_scores, marker='s', label="Test Accuracy", drawstyle="steps-post", color="#dc2626")
    plt.title("Cost-Complexity Pruning: ccp_alpha vs Accuracy", fontsize=12)
    plt.xlabel("Effective Alpha (Regularization / Penalty parameter)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(PLOTS_DIR, "unit5_cost_complexity_pruning.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved CCP plot: {plot_path}")
    
    return dt_pruned, {
        "deep_train_acc": round(acc_deep_train * 100, 2),
        "deep_test_acc": round(acc_deep_test * 100, 2),
        "pruned_train_acc": round(acc_pruned_train * 100, 2),
        "pruned_test_acc": round(acc_pruned_test * 100, 2)
    }

def demonstrate_feature_importance_and_tree_plot(model, feature_names):
    """
    Extracts and plots Mean Decrease in Impurity (MDI) feature importances.
    """
    print("\n--- 3. Feature Importance in Decision Trees ---")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    feat_imp_data = []
    print("Decision Tree Feature Importance Rankings:")
    for rank, idx in enumerate(indices):
        print(f"  {rank+1}. {feature_names[idx]:<22}: {importances[idx]:.4f}")
        feat_imp_data.append({"Feature": feature_names[idx], "Importance": round(importances[idx], 4)})
        
    # Plot Feature Importances
    plt.figure(figsize=(9, 5))
    plt.barh([feature_names[i] for i in indices[::-1]], importances[indices[::-1]], color="#0284c7")
    plt.title("Decision Tree: Feature Importance (Mean Decrease in Impurity)", fontsize=12)
    plt.xlabel("Gini Importance Score")
    plt.grid(True, alpha=0.3, axis="x")
    plot_path = os.path.join(PLOTS_DIR, "unit5_decision_tree_feature_importance.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot: {plot_path}")
    
    return feat_imp_data

def demonstrate_decision_tree_regression(reg_data_dict: dict):
    """
    Decision Tree Regressor for Flight Delay Prediction.
    """
    print("\n--- 4. Decision Tree Regression (Flight Departure Delays) ---")
    X_train = reg_data_dict["X_train"]
    y_train = reg_data_dict["y_train"]
    X_test = reg_data_dict["X_test"]
    y_test = reg_data_dict["y_test"]
    
    dt_reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=42)
    dt_reg.fit(X_train, y_train)
    
    pred = dt_reg.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"Decision Tree Regressor (Depth=6): MAE = {mae:.2f} min | RMSE = {rmse:.2f} min | R2 = {r2:.4f}")
    
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }

def run_unit5_module():
    print("=====================================================================")
    print(" UNIT V / MODULE 4: TREE MODELS - I (DECISION TREES) (CO5)")
    print("=====================================================================")
    clf_data = get_classification_data()
    reg_data = get_regression_data()
    
    impurity_res = demonstrate_impurity_measures()
    best_tree, pruning_res = demonstrate_tree_pruning_and_hyperparameters(clf_data)
    feat_imp = demonstrate_feature_importance_and_tree_plot(best_tree, clf_data["feature_names"])
    reg_res = demonstrate_decision_tree_regression(reg_data)
    
    return {
        "impurity": impurity_res,
        "pruning": pruning_res,
        "feature_importance": feat_imp,
        "regression": reg_res
    }

if __name__ == "__main__":
    run_unit5_module()
