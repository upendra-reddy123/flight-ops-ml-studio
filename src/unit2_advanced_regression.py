"""
unit2_advanced_regression.py
Course Outcome CO2 :: Advanced Regression Techniques
Topics Covered:
1. Simple Linear Regression (SLR): Manual Closed-Form vs Matrix Formula
2. Multiple Linear Regression (MLR): Closed-Form (X^T X)^(-1) X^T y vs Scikit-Learn
3. Linear Regression Assumptions & Diagnostics:
   - Residuals vs Fitted Values (Linearity & Homoscedasticity)
   - Normality of Residuals (Q-Q Plot & Histogram)
   - Multicollinearity Diagnostics via Variance Inflation Factor (VIF)
4. Non-Linearity & Polynomial Regression (Degree 2 & 3)
5. Regularization:
   - Ridge Regression (L2 Penalty) with Coefficient Shrinkage Path
   - Lasso Regression (L1 Penalty) with Feature Sparsity / Selection
   - Geometrical Representation Interpretation (L1 Diamond vs L2 Circle)
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
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import get_regression_data, load_air_traffic_data

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def demonstrate_slr_and_matrix_derivation(df: pd.DataFrame):
    """
    Demonstrates Simple Linear Regression (SLR):
    Predicting Departure Delay using Runway Utilization.
    Computes coefficients manually using:
      beta_1 = sum((x - x_bar)(y - y_bar)) / sum((x - x_bar)^2)
      beta_0 = y_bar - beta_1 * x_bar
    And verifies with Matrix form: beta = (X^T X)^(-1) X^T y
    """
    print("\n--- 1. Simple Linear Regression (SLR) & Matrix Derivation ---")
    x = df["runway_utilization"].values
    y = df["avg_departure_delay"].values
    
    # Scalar Formulas
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    numerator = np.sum((x - x_bar) * (y - y_bar))
    denominator = np.sum((x - x_bar)**2)
    beta_1_manual = numerator / denominator
    beta_0_manual = y_bar - beta_1_manual * x_bar
    
    # Matrix Representation: X = [1, x], beta = (X^T X)^(-1) X^T y
    X_design = np.column_stack([np.ones_like(x), x])
    beta_matrix = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ y)
    
    # Scikit-learn
    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y)
    
    print(f"SLR Target: Departure Delay (min) vs Feature: Runway Utilization (%)")
    print(f"Manual Closed-Form: Intercept (beta_0) = {beta_0_manual:.4f}, Slope (beta_1) = {beta_1_manual:.4f}")
    print(f"Matrix Formulation: Intercept (beta_0) = {beta_matrix[0]:.4f}, Slope (beta_1) = {beta_matrix[1]:.4f}")
    print(f"Scikit-Learn Model: Intercept (beta_0) = {lr.intercept_:.4f}, Slope (beta_1) = {lr.coef_[0]:.4f}")
    
    # Plot SLR line
    plt.figure(figsize=(8, 5))
    sample_idx = np.random.choice(len(x), size=500, replace=False)
    plt.scatter(x[sample_idx], y[sample_idx], alpha=0.3, color="#3b82f6", label="Airport Hour Observation")
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = beta_0_manual + beta_1_manual * x_line
    plt.plot(x_line, y_line, color="#ef4444", lw=2.5, label=f"SLR Fit: y = {beta_0_manual:.1f} + {beta_1_manual:.2f}x")
    plt.title("Simple Linear Regression: Runway Utilization vs Departure Delay", fontsize=12)
    plt.xlabel("Runway Utilization (%)")
    plt.ylabel("Average Departure Delay (minutes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(PLOTS_DIR, "unit2_slr_fit.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return {
        "beta_0": round(float(beta_0_manual), 4),
        "beta_1": round(float(beta_1_manual), 4)
    }

def demonstrate_mlr_and_assumptions(data_dict: dict):
    """
    Multiple Linear Regression (MLR) and complete diagnostic checking of regression assumptions:
    1. Linearity & Homoscedasticity (Residuals vs Fitted)
    2. Normality of Residuals (Q-Q Plot)
    3. Multicollinearity using Variance Inflation Factor (VIF)
    """
    print("\n--- 2. Multiple Linear Regression (MLR) & Assumption Diagnostics ---")
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    
    y_pred_train = mlr.predict(X_train)
    y_pred_test = mlr.predict(X_test)
    residuals = y_test - y_pred_test
    
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    print(f"MLR Test Performance: MAE = {mae:.2f} min | RMSE = {rmse:.2f} min | R2 = {r2:.4f}")
    
    # Assumption 1: Multicollinearity via VIF
    print("\nVariance Inflation Factor (VIF) Analysis for Multicollinearity:")
    vif_data = []
    feature_cols = list(X_train.columns)
    for i, col in enumerate(feature_cols):
        # Regress feature col on all other features
        other_cols = [c for c in feature_cols if c != col]
        r2_j = LinearRegression().fit(X_train[other_cols], X_train[col]).score(X_train[other_cols], X_train[col])
        vif = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else float('inf')
        vif_data.append({"Feature": col, "VIF": round(vif, 2), "Status": "High Collinearity (>10)" if vif > 10 else "Acceptable (<10)"})
        
    vif_df = pd.DataFrame(vif_data)
    print(vif_df.to_string(index=False))
    
    # Assumption 2 & 3: Residual Diagnostics Plots (Residuals vs Fitted & Q-Q Plot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Fitted
    axes[0].scatter(y_pred_test[:500], residuals[:500], alpha=0.4, color="#6366f1")
    axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_title("Assumption Test: Residuals vs Fitted (Homoscedasticity & Linearity)", fontsize=11)
    axes[0].set_xlabel("Fitted Values (Predicted Delay)")
    axes[0].set_ylabel("Residuals (Actual - Predicted)")
    axes[0].grid(True, alpha=0.3)
    
    # Normal Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Assumption Test: Normal Q-Q Plot of Residuals", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plot_path = os.path.join(PLOTS_DIR, "unit2_mlr_residual_diagnostics.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved diagnostic plot: {plot_path}")
    
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "vif": vif_data
    }

def demonstrate_polynomial_regression(data_dict: dict):
    """
    Polynomial Regression to capture non-linear relationships.
    Compares Linear vs Degree-2 Polynomial vs Degree-3 Polynomial.
    """
    print("\n--- 3. Polynomial Regression (Non-Linearity Mapping) ---")
    # Using top predictive feature: runway_utilization & weather_delay_factor
    X_train_sub = data_dict["X_train"][["runway_utilization", "weather_delay_factor"]]
    X_test_sub = data_dict["X_test"][["runway_utilization", "weather_delay_factor"]]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]
    
    poly_results = {}
    degrees = [1, 2, 3]
    
    for deg in degrees:
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_sub)
        X_test_poly = poly.transform(X_test_sub)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        pred = model.predict(X_test_poly)
        
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        poly_results[f"Degree {deg}"] = {"R2": round(r2, 4), "RMSE": round(rmse, 2), "Num_Features": X_train_poly.shape[1]}
        print(f"Polynomial Degree {deg}: Features = {X_train_poly.shape[1]:<2} | Test RMSE = {rmse:.2f} min | R2 = {r2:.4f}")
        
    return poly_results

def demonstrate_regularization_ridge_lasso(data_dict: dict):
    """
    Regularization: Ridge (L2 penalty) and Lasso (L1 penalty).
    Plots the Regularization Paths showing coefficient shrinkage and feature selection.
    """
    print("\n--- 4. Regularization: Ridge (L2) vs Lasso (L1) & Coefficient Paths ---")
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    feature_names = data_dict["feature_names"]
    
    alphas = np.logspace(-2, 3, 50)
    ridge_coefs = []
    lasso_coefs = []
    
    for a in alphas:
        r = Ridge(alpha=a).fit(X_train, y_train)
        l = Lasso(alpha=a, max_iter=2000).fit(X_train, y_train)
        ridge_coefs.append(r.coef_)
        lasso_coefs.append(l.coef_)
        
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)
    
    # Plot Regularization Paths
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, col in enumerate(feature_names):
        axes[0].plot(alphas, ridge_coefs[:, i], label=col)
        axes[1].plot(alphas, lasso_coefs[:, i], label=col)
        
    axes[0].set_xscale("log")
    axes[0].set_title("Ridge (L2) Regularization Path: Smooth Shrinkage to Zero", fontsize=11)
    axes[0].set_xlabel("Regularization Strength (alpha)")
    axes[0].set_ylabel("Coefficients")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xscale("log")
    axes[1].set_title("Lasso (L1) Regularization Path: Sparsity & Feature Selection", fontsize=11)
    axes[1].set_xlabel("Regularization Strength (alpha)")
    axes[1].set_ylabel("Coefficients")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plot_path = os.path.join(PLOTS_DIR, "unit2_regularization_paths.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved regularization path plot: {plot_path}")
    
    # Train optimal Ridge & Lasso models for final comparison
    best_ridge = Ridge(alpha=10.0).fit(X_train, y_train)
    best_lasso = Lasso(alpha=0.5).fit(X_train, y_train)
    
    pred_ridge = best_ridge.predict(X_test)
    pred_lasso = best_lasso.predict(X_test)
    
    # Zeroed out features by Lasso
    zero_features = [feature_names[i] for i, c in enumerate(best_lasso.coef_) if abs(c) < 1e-4]
    retained_features = [feature_names[i] for i, c in enumerate(best_lasso.coef_) if abs(c) >= 1e-4]
    print(f"Lasso (alpha=0.5) Retained Features: {retained_features}")
    print(f"Lasso (alpha=0.5) Zeroed Out Features: {zero_features}")
    
    return {
        "ridge_r2": round(r2_score(y_test, pred_ridge), 4),
        "ridge_rmse": round(np.sqrt(mean_squared_error(y_test, pred_ridge)), 2),
        "lasso_r2": round(r2_score(y_test, pred_lasso), 4),
        "lasso_rmse": round(np.sqrt(mean_squared_error(y_test, pred_lasso)), 2),
        "lasso_retained": retained_features,
        "lasso_zeroed": zero_features
    }

def run_unit2_module():
    print("=====================================================================")
    print(" UNIT II / MODULE 2: ADVANCED REGRESSION (CO2)")
    print("=====================================================================")
    df = load_air_traffic_data()
    reg_data = get_regression_data(df)
    
    slr_results = demonstrate_slr_and_matrix_derivation(df)
    mlr_results = demonstrate_mlr_and_assumptions(reg_data)
    poly_results = demonstrate_polynomial_regression(reg_data)
    reg_results = demonstrate_regularization_ridge_lasso(reg_data)
    
    return {
        "slr": slr_results,
        "mlr": mlr_results,
        "poly": poly_results,
        "regularization": reg_results
    }

if __name__ == "__main__":
    run_unit2_module()
