"""
unit1_model_selection.py
Course Outcome CO1 :: Model Selection and Evaluation
Topics Covered:
1. Simplicity, Complexity, and Overfitting
2. Bias-Variance Tradeoff (with Empirical Learning Curves)
3. Model Evaluation Metrics (Regression & Classification)
4. Cross-Validation (K-Fold & Stratified K-Fold)
5. Hyperparameter Tuning via GridSearchCV and RandomizedSearchCV
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
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, cross_validate,
    GridSearchCV, RandomizedSearchCV, learning_curve
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from src.data_loader import get_regression_data, get_classification_data

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def demonstrate_bias_variance_tradeoff(data_dict: dict):
    """
    Analyzes the Bias-Variance Tradeoff using Decision Trees with varying max_depth.
    Underfitting (High Bias) -> Sweet Spot -> Overfitting (High Variance).
    """
    print("\n--- 1. Bias-Variance Tradeoff Analysis (Tree Depth vs Errors) ---")
    X_train, X_test = data_dict["X_train"], data_dict["X_test"]
    y_train, y_test = data_dict["y_train"], data_dict["y_test"]
    
    depths = range(1, 22)
    train_errors = []
    test_errors = []
    
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        
        pred_train = dt.predict(X_train)
        pred_test = dt.predict(X_test)
        
        train_errors.append(mean_squared_error(y_train, pred_train))
        test_errors.append(mean_squared_error(y_test, pred_test))
        
    best_depth = depths[np.argmin(test_errors)]
    print(f"Optimal Depth (minimizing Test MSE): depth = {best_depth}")
    print(f"High Bias regime (depth=1): Train MSE={train_errors[0]:.2f}, Test MSE={test_errors[0]:.2f}")
    print(f"Optimal regime (depth={best_depth}): Train MSE={train_errors[best_depth-1]:.2f}, Test MSE={test_errors[best_depth-1]:.2f}")
    print(f"High Variance regime (depth=20): Train MSE={train_errors[19]:.2f}, Test MSE={test_errors[19]:.2f}")

    # Plot Bias-Variance Curve
    plt.figure(figsize=(9, 5))
    plt.plot(depths, train_errors, label="Train MSE (Model Error on Training)", color="#2563eb", lw=2, marker='o')
    plt.plot(depths, test_errors, label="Validation/Test MSE (Generalization Error)", color="#dc2626", lw=2, marker='s')
    plt.axvline(best_depth, color="#16a34a", linestyle="--", label=f"Optimal Tradeoff (depth={best_depth})")
    plt.title("Bias-Variance Tradeoff: Model Complexity vs Error (Air Traffic Delay Prediction)", fontsize=13)
    plt.xlabel("Model Complexity (Decision Tree Max Depth)", fontsize=11)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(PLOTS_DIR, "unit1_bias_variance_tradeoff.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")
    
    return {
        "depths": list(depths),
        "train_errors": [round(x, 2) for x in train_errors],
        "test_errors": [round(x, 2) for x in test_errors],
        "best_depth": best_depth
    }

def demonstrate_cross_validation(data_dict: dict):
    """
    Demonstrates K-Fold Cross Validation vs Single Train-Test Split.
    Validates model consistency across folds.
    """
    print("\n--- 2. K-Fold Cross Validation Demonstration ---")
    X = pd.concat([data_dict["X_train"], data_dict["X_test"]])
    y = pd.concat([data_dict["y_train"], data_dict["y_test"]])
    
    # 5-Fold Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    model = Ridge(alpha=1.0)
    
    scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
    cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring, return_train_score=True)
    
    mae_scores = -cv_results['test_neg_mean_absolute_error']
    rmse_scores = -cv_results['test_neg_root_mean_squared_error']
    r2_scores = cv_results['test_r2']
    
    print("5-Fold CV Results on Full Dataset:")
    for fold in range(5):
        print(f"  Fold {fold+1}: MAE = {mae_scores[fold]:.2f} min | RMSE = {rmse_scores[fold]:.2f} min | R2 = {r2_scores[fold]:.4f}")
        
    print(f"Mean MAE across folds: {mae_scores.mean():.2f} (+/- {mae_scores.std():.2f})")
    print(f"Mean RMSE across folds: {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")
    print(f"Mean R2 across folds:   {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
    
    return {
        "mae_scores": [round(x, 2) for x in mae_scores],
        "rmse_scores": [round(x, 2) for x in rmse_scores],
        "r2_scores": [round(x, 4) for x in r2_scores],
        "mean_mae": round(mae_scores.mean(), 2),
        "mean_rmse": round(rmse_scores.mean(), 2),
        "mean_r2": round(r2_scores.mean(), 4)
    }

def demonstrate_hyperparameter_tuning(clf_data_dict: dict):
    """
    Demonstrates GridSearchCV and RandomizedSearchCV for Hyperparameter Tuning.
    """
    print("\n--- 3. Hyperparameter Tuning (GridSearchCV vs RandomizedSearchCV) ---")
    X_train = clf_data_dict["X_train"]
    y_train = clf_data_dict["y_train"]
    
    # Stratified K-Fold for Classification
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Model: Decision Tree Classifier
    dt_clf = DecisionTreeClassifier(random_state=42)
    
    param_grid = {
        'max_depth': [3, 5, 8, 12, None],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    
    print(f"Running GridSearchCV over {5 * 3 * 3 * 2 = } parameter combinations (3-fold CV)...")
    grid_search = GridSearchCV(
        estimator=dt_clf,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"GridSearchCV Best Params: {grid_search.best_params_}")
    print(f"GridSearchCV Best Cross-Validation Accuracy: {grid_search.best_score_ * 100:.2f}%")
    
    # Randomized Search comparison
    random_search = RandomizedSearchCV(
        estimator=dt_clf,
        param_distributions=param_grid,
        n_iter=15,
        cv=skf,
        scoring='accuracy',
        random_state=42,
        n_jobs=1
    )
    random_search.fit(X_train, y_train)
    print(f"RandomizedSearchCV Best Params (15 iterations): {random_search.best_params_}")
    print(f"RandomizedSearchCV Best Cross-Validation Accuracy: {random_search.best_score_ * 100:.2f}%")
    
    return {
        "grid_best_params": grid_search.best_params_,
        "grid_best_score": round(grid_search.best_score_ * 100, 2),
        "random_best_params": random_search.best_params_,
        "random_best_score": round(random_search.best_score_ * 100, 2)
    }

def run_unit1_module():
    print("=====================================================================")
    print(" UNIT I / MODULE 1: MODEL SELECTION AND EVALUATION (CO1)")
    print("=====================================================================")
    reg_data = get_regression_data()
    clf_data = get_classification_data()
    
    bv_results = demonstrate_bias_variance_tradeoff(reg_data)
    cv_results = demonstrate_cross_validation(reg_data)
    tune_results = demonstrate_hyperparameter_tuning(clf_data)
    
    return {
        "bias_variance": bv_results,
        "cross_validation": cv_results,
        "tuning": tune_results
    }

if __name__ == "__main__":
    run_unit1_module()
