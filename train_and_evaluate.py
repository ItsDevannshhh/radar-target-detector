import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Set visual style for report plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150


# ==========================================
# STEP 1: LOAD DATASET (LOCAL + GITHUB + FALLBACK)
# ==========================================
def load_ionosphere_data():
    cols = [f"a{i:02d}" for i in range(1, 35)] + ["target"]
    local_path = "ionosphere.csv"
    github_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv"
    
    if os.path.exists(local_path):
        print(f"[1/6] Loading UCI Ionosphere dataset from local file '{local_path}'...")
        df = pd.read_csv(local_path, header=None, names=cols)
    else:
        try:
            print("[1/6] Fetching UCI Ionosphere dataset from GitHub...")
            df = pd.read_csv(github_url, header=None, names=cols)
        except Exception as e:
            print(f"GitHub fetch failed ({e}). Falling back to Scikit-Learn OpenML...")
            data = fetch_openml(name="ionosphere", version=1, as_frame=True)
            df = data.frame
            df.columns = cols

    print(f"Dataset successfully loaded! Shape: {df.shape}")
    print(f"Class distribution:\n{df['target'].value_counts()}\n")
    return df


# ==========================================
# STEP 2: PHYSICS FEATURE ENGINEERING & CLEANING
# ==========================================
def preprocess_and_engineer(df):
    print("[2/6] Performing Radar Signal Preprocessing & Feature Engineering...")
    data = df.copy()

    # Separate Real (odd columns: a01, a03..a33) and Imaginary (even columns: a02, a04..a34)
    real_cols = [f"a{i:02d}" for i in range(1, 35, 2)]
    imag_cols = [f"a{i:02d}" for i in range(2, 35, 2)]

    R = data[real_cols].values
    I = data[imag_cols].values

    # Calculate complex envelope amplitude across all 17 radar pulse lags: sqrt(Re^2 + Im^2)
    amplitudes = np.sqrt(R**2 + I**2)

    # Engineer 3 domain-specific radar features
    data["mean_pulse_amplitude"] = np.mean(amplitudes, axis=1)
    data["std_pulse_amplitude"] = np.std(amplitudes, axis=1)
    # Smoothness: mean absolute lag-to-lag jump (Good returns decay smoothly; Bad noise jumps erratically)
    data["lag_smoothness"] = np.mean(np.abs(np.diff(R, axis=1)), axis=1)

    # Verify and drop Column 2 (a02) -> Imaginary part of Lag 0 is mathematically 0.0 for all rows
    a02_variance = data["a02"].var()
    print(f"Variance of column 'a02' (Lag 0 Imaginary): {a02_variance:.6f} -> Dropping 'a02'!")
    data.drop(columns=["a02"], inplace=True)

    # Encode Target: 'g' (Good / Valid Target) -> 1, 'b' (Bad / Ghost Noise) -> 0
    data["target"] = data["target"].map({"g": 1, "b": 0, 1: 1, 0: 0})

    X = data.drop(columns=["target"])
    y = data["target"]
    print(f"Final Feature Matrix Shape: {X.shape} (33 raw features + 3 engineered radar features)\n")
    return X, y, df


# ==========================================
# STEP 3: GENERATE EDA PLOTS FOR REPORT
# ==========================================
def generate_eda_plots(df, X, y):
    print("[3/6] Generating Exploratory Data Analysis (EDA) plots...")
    real_cols = [f"a{i:02d}" for i in range(1, 35, 2)]
    
    # Plot 1: Coherent Good Signal vs Erratic Bad Signal across 17 Pulse Lags
    good_sample = df[df["target"] == "g"].iloc[0][real_cols].values.astype(float)
    bad_sample = df[df["target"] == "b"].iloc[1][real_cols].values.astype(float)
    lags = np.arange(0, 17)

    plt.figure(figsize=(9, 4.5))
    plt.plot(lags, good_sample, marker="o", linewidth=2.2, label="Good Return (Valid Ionospheric Target)", color="#1f77b4")
    plt.plot(lags, bad_sample, marker="x", linestyle="--", linewidth=2.0, label="Bad Return (Ghost / Pass-Through Noise)", color="#d62728")
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.title("Radar Autocorrelation Waveform Across 17 Pulse Lags (Real Component)", fontsize=12, fontweight="bold")
    plt.xlabel("Pulse Lag Number (k = 0 to 16)")
    plt.ylabel("Normalized Autocorrelation [-1, +1]")
    plt.xticks(lags)
    plt.legend()
    plt.tight_layout()
    plt.savefig("1_radar_waveform_comparison.png")
    plt.close()

    # Plot 2: 2D PCA Cluster Projection
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7.5, 5.5))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y, cmap="coolwarm_r", alpha=0.8, edgecolors="k", s=55
    )
    plt.title("2D PCA Projection: Tightly Clustered 'Good' Targets vs. Scattered 'Bad' Noise", fontsize=11, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    handles, _ = scatter.legend_elements()
    plt.legend(handles, ["Bad (Ghost/Noise = 0)", "Good (Valid Target = 1)"], title="Radar Return Class")
    plt.tight_layout()
    plt.savefig("2_pca_clusters.png")
    plt.close()
    print("Saved '1_radar_waveform_comparison.png' and '2_pca_clusters.png'.\n")


# ==========================================
# STEP 4: TRAIN & BENCHMARK 4 CLASSIFIERS
# ==========================================
def benchmark_models(X_train, X_test, y_train, y_test):
    print("[4/6] Training and Benchmarking 4 ML Classifiers...")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=1000))
        ]),
        "k-Nearest Neighbors (k=3)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=3))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        "SVM (RBF Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "Neural Network (MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500, random_state=42))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    trained_models = {}

    plt.figure(figsize=(8, 6))

    for name, pipeline in models.items():
        # 5-Fold Cross Validation on Training Set
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        
        # Fit and Predict on Hold-out Test Set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "Model": name,
            "CV Accuracy (Mean)": f"{cv_scores.mean()*100:.2f}%",
            "Test Accuracy": f"{acc*100:.2f}%",
            "Precision": f"{prec*100:.2f}%",
            "Recall": f"{rec*100:.2f}%",
            "F1-Score": f"{f1*100:.2f}%",
            "ROC-AUC": f"{auc:.4f}",
            "Ghost Alarms (FP)": fp,
            "Missed Targets (FN)": fn
        })
        trained_models[name] = pipeline

        # Add to ROC Curve Plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")

    # Finalize ROC Curve Plot
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    plt.title("ROC Curve Comparison — Radar Signal Target Validation", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (Ghost Alarm Rate)")
    plt.ylabel("True Positive Rate (Target Detection Recall)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("3_roc_curves.png")
    plt.close()

    results_df = pd.DataFrame(results)
    print("\n================ MODEL BENCHMARK TABLE ================")
    print(results_df.to_string(index=False))
    print("=======================================================\n")
    return trained_models, results_df


# ==========================================
# STEP 5: HYPERPARAMETER TUNING (RBF-SVM)
# ==========================================
def tune_best_svm(X_train, X_test, y_train, y_test):
    print("[5/6] Running GridSearchCV Hyperparameter Tuning on RBF-SVM...")
    
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))
    ])

    param_grid = {
        "clf__C": [0.5, 1, 2, 5, 10, 20],
        "clf__gamma": ["scale", "auto", 0.01, 0.03, 0.05, 0.1]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(svm_pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print(f"Best SVM Parameters: {grid.best_params_}")
    print(f"Tuned SVM Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("\nDetailed Classification Report (Tuned RBF-SVM):")
    print(classification_report(y_test, y_pred, target_names=["Bad (Noise/Ghost)", "Good (Valid Target)"]))

    # Plot Confusion Matrix for Tuned SVM
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Predicted: Bad (Noise)", "Predicted: Good (Target)"],
        yticklabels=["Actual: Bad (Noise)", "Actual: Good (Target)"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Confusion Matrix — Tuned RBF-SVM Radar Validator", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("4_confusion_matrix_svm.png")
    plt.close()
    print("Saved '3_roc_curves.png' and '4_confusion_matrix_svm.png'.\n")
    
    return best_model


# ==========================================
# STEP 6: EXPORT ARTIFACTS FOR STREAMLIT APP
# ==========================================
if __name__ == "__main__":
    raw_df = load_ionosphere_data()
    X, y, _ = preprocess_and_engineer(raw_df)
    generate_eda_plots(raw_df, X, y)

    # 80/20 Stratified Split (280 Train, 71 Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    trained_models, benchmark_df = benchmark_models(X_train, X_test, y_train, y_test)
    best_svm = tune_best_svm(X_train, X_test, y_train, y_test)

    print("[6/6] Saving trained model and test radar samples for live demo...")
    joblib.dump(best_svm, "ionosphere_svm_pipeline.pkl")
    joblib.dump(best_svm, "model.pkl")
    joblib.dump(trained_models["Neural Network (MLP)"], "neural_network_pipeline.pkl")
    benchmark_df.to_csv("benchmark_results.csv", index=False)
    print("Done! Exported model.pkl, ionosphere_svm_pipeline.pkl, and neural_network_pipeline.pkl.")
    print("Run `streamlit run app.py` to launch the Interactive Radar Console.")
