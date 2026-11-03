"""
unit3_4_svm.py
Course Outcomes CO3 & CO4 :: Support Vector Machines (SVM) - I & II
Topics Covered:
1. Concept of a Hyperplane in 2D & 3D
2. Maximal Margin Classifier vs Soft Margin Classifier
3. Notion of Slack Variables (xi_i) and Cost of Misclassification (C)
4. Introduction to Kernels & The Kernel Trick:
   - Linear Kernel
   - Polynomial Kernel
   - Radial Basis Function (RBF / Gaussian) Kernel
5. Support Vector Classification (SVC) for Airport Congestion (Low vs High/Medium)
6. Support Vector Regression (SVR) for Continuous Delay Prediction
7. Visualizing 2D Decision Boundaries, Margins, and Support Vectors
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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, mean_squared_error, r2_score

from src.data_loader import get_classification_data, get_regression_data

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def demonstrate_hyperplane_2d_and_slack_variables(data_dict: dict):
    """
    Demonstrates 2D Hyperplane, Maximal Margin, Soft Margin, Support Vectors,
    and the impact of Cost parameter C (handling slack variables).
    Uses 2 primary intuitive features: runway_utilization vs weather_delay_factor.
    """
    print("\n--- 1. Concept of Hyperplane in 2D, Margins & Slack Variables (C Tradeoff) ---")
    
    # Binary classification for clear geometric visualization: 0 (Low) vs 1 (Congested: Med/High)
    X = data_dict["X_train_raw"][["runway_utilization", "weather_delay_factor"]].values
    y = (data_dict["y_train"].values > 0).astype(int) # 0=Uncongested, 1=Congested
    
    # Take a representative subsample of 300 points for crisp visualization
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=300, replace=False)
    X_sub = X[sample_indices]
    y_sub = y[sample_indices]
    
    # Train Linear SVM with two different values of C:
    # Low C (e.g. 0.05): Large margins, allows more slack (soft margin)
    # High C (e.g. 10.0): Strict margin, high penalty on slack variables (harder margin)
    c_values = [0.05, 5.0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, C_val in enumerate(c_values):
        clf = SVC(kernel="linear", C=C_val)
        clf.fit(X_sub, y_sub)
        
        # Decision boundary: w0*x + w1*y + b = 0
        w = clf.coef_[0]
        b = clf.intercept_[0]
        
        # Plot data points
        axes[idx].scatter(X_sub[y_sub == 0, 0], X_sub[y_sub == 0, 1], color="#22c55e", label="Low Congestion (Class 0)", alpha=0.7, s=40)
        axes[idx].scatter(X_sub[y_sub == 1, 0], X_sub[y_sub == 1, 1], color="#ef4444", label="High/Med Congestion (Class 1)", alpha=0.7, s=40)
        
        # Highlight Support Vectors
        sv = clf.support_vectors_
        axes[idx].scatter(sv[:, 0], sv[:, 1], s=120, facecolors='none', edgecolors='black', lw=1.5, label=f"Support Vectors (N={len(sv)})")
        
        # Draw Hyperplane and Margin Boundaries: w0*x + w1*y + b = 0, +1, -1
        x_min, x_max = X_sub[:, 0].min() - 5, X_sub[:, 0].max() + 5
        x_points = np.linspace(x_min, x_max, 100)
        y_decision = -(w[0] * x_points + b) / w[1]
        y_margin_up = -(w[0] * x_points + b - 1) / w[1]
        y_margin_down = -(w[0] * x_points + b + 1) / w[1]
        
        axes[idx].plot(x_points, y_decision, 'k-', lw=2, label="Separating Hyperplane (w^T x + b = 0)")
        axes[idx].plot(x_points, y_margin_up, 'k--', lw=1.2, label="Margin Boundary (+1)")
        axes[idx].plot(x_points, y_margin_down, 'k--', lw=1.2, label="Margin Boundary (-1)")
        
        # Margin width: 2 / ||w||
        margin_width = 2.0 / np.linalg.norm(w)
        
        axes[idx].set_title(f"Linear SVM (C = {C_val}): Margin Width = {margin_width:.2f}\nSlack Penalty = {'Low (Soft Margin)' if C_val < 1 else 'High (Harder Margin)'}", fontsize=11)
        axes[idx].set_xlabel("Runway Utilization (%)")
        axes[idx].set_ylabel("Weather Delay Factor (min)")
        axes[idx].set_ylim(-5, 65)
        axes[idx].legend(loc="upper left", fontsize=8)
        axes[idx].grid(True, alpha=0.3)
        
    plot_path = os.path.join(PLOTS_DIR, "unit3_svm_2d_hyperplane_margins.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 2D Hyperplane plot: {plot_path}")

def demonstrate_hyperplane_3d(data_dict: dict):
    """
    Demonstrates 3D Hyperplane separating 3 aviation features:
    Total Flights, Weather Delay Factor, Runway Utilization.
    """
    print("\n--- 2. Concept of Hyperplane in 3D Space ---")
    features_3d = ["total_flights", "weather_delay_factor", "runway_utilization"]
    X = data_dict["X_train_raw"][features_3d].values
    y = (data_dict["y_train"].values > 0).astype(int)
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=250, replace=False)
    X_sub = X[sample_indices]
    y_sub = y[sample_indices]
    
    clf = SVC(kernel="linear", C=1.0)
    clf.fit(X_sub, y_sub)
    
    w = clf.coef_[0]
    b = clf.intercept_[0]
    
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter 3D points
    ax.scatter(X_sub[y_sub == 0, 0], X_sub[y_sub == 0, 1], X_sub[y_sub == 0, 2], color="#22c55e", label="Low Congestion", alpha=0.6, s=30)
    ax.scatter(X_sub[y_sub == 1, 0], X_sub[y_sub == 1, 1], X_sub[y_sub == 1, 2], color="#ef4444", label="High/Med Congestion", alpha=0.6, s=30)
    
    # 3D Plane Meshgrid: w0*x + w1*y + w2*z + b = 0 => z = -(w0*x + w1*y + b) / w2
    xx, yy = np.meshgrid(
        np.linspace(X_sub[:, 0].min(), X_sub[:, 0].max(), 15),
        np.linspace(X_sub[:, 1].min(), X_sub[:, 1].max(), 15)
    )
    zz = -(w[0] * xx + w[1] * yy + b) / w[2]
    
    ax.plot_surface(xx, yy, zz, alpha=0.3, color="#38bdf8")
    ax.set_title("3D Separating Hyperplane in Aviation Feature Space\nw_1*x_1 + w_2*x_2 + w_3*x_3 + b = 0", fontsize=11)
    ax.set_xlabel("Total Flights")
    ax.set_ylabel("Weather Delay (min)")
    ax.set_zlabel("Runway Utilization (%)")
    ax.legend()
    
    plot_path = os.path.join(PLOTS_DIR, "unit3_svm_3d_hyperplane.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 3D Hyperplane plot: {plot_path}")

def demonstrate_kernel_methods_and_trick(data_dict: dict):
    """
    Demonstrates Kernel Methods (CO4):
    Compares Linear Kernel vs Polynomial Kernel vs RBF (Gaussian) Kernel.
    Shows how the Kernel Trick maps non-linear data into a linearly separable space.
    """
    print("\n--- 3. Kernel Trick & Comparison: Linear vs Polynomial vs RBF Kernels ---")
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    
    kernels = ["linear", "poly", "rbf"]
    kernel_results = {}
    
    for k in kernels:
        svm = SVC(kernel=k, C=1.0, gamma="scale", random_state=42)
        svm.fit(X_train, y_train)
        
        pred_test = svm.predict(X_test)
        acc = accuracy_score(y_test, pred_test)
        num_sv = len(svm.support_vectors_)
        
        kernel_results[k] = {
            "Accuracy": round(acc * 100, 2),
            "Support_Vectors": num_sv
        }
        print(f"SVM Kernel: {k.upper():<8} | Test Accuracy: {acc * 100:.2f}% | Number of Support Vectors: {num_sv}")
        
    return kernel_results

def demonstrate_support_vector_regression(reg_data_dict: dict):
    """
    Support Vector Regression (SVR) for Flight Delay Prediction.
    Evaluates Linear SVR vs RBF Kernel SVR with epsilon margin tube.
    """
    print("\n--- 4. Support Vector Regression (SVR) for Flight Delay Prediction ---")
    X_train = reg_data_dict["X_train"]
    y_train = reg_data_dict["y_train"]
    X_test = reg_data_dict["X_test"]
    y_test = reg_data_dict["y_test"]
    
    svr_rbf = SVR(kernel="rbf", C=10.0, epsilon=0.2)
    svr_rbf.fit(X_train, y_train)
    
    pred_test = svr_rbf.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    r2 = r2_score(y_test, pred_test)
    
    print(f"SVR (RBF Kernel, C=10, eps=0.2): MAE = {mae:.2f} min | RMSE = {rmse:.2f} min | R2 = {r2:.4f}")
    
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }

def run_unit3_4_module():
    print("=====================================================================")
    print(" UNIT III & IV / MODULE 3: SUPPORT VECTOR MACHINES - I & II (CO3 & CO4)")
    print("=====================================================================")
    clf_data = get_classification_data()
    reg_data = get_regression_data()
    
    demonstrate_hyperplane_2d_and_slack_variables(clf_data)
    demonstrate_hyperplane_3d(clf_data)
    kernel_results = demonstrate_kernel_methods_and_trick(clf_data)
    svr_results = demonstrate_support_vector_regression(reg_data)
    
    return {
        "kernels": kernel_results,
        "svr": svr_results
    }

if __name__ == "__main__":
    run_unit3_4_module()
