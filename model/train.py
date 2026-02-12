import pandas as pd
import pickle
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.preprocessing import clean_text
from model.pipelines import create_pipeline
from model.evaluate import run_experiments, evaluate_model, perform_error_analysis, plot_confusion_matrix, plot_learning_curve

# Paths
DATA_PATH = BASE_DIR / "data" / "tickets.csv"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "model"
os.makedirs(REPORTS_DIR, exist_ok=True)

def run_grid_search(X_train, y_train, model_type="svm"):
    """
    Formal GridSearchCV to optimize hyperparameters for research documentation.
    """
    pipe = create_pipeline(model_type)
    
    # Define param grid based on model
    if model_type == "svm":
        param_grid = {'clf__estimator__C': [0.1, 1, 10]}
    else:
        param_grid = {'clf__C': [0.1, 1, 10]}
    
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_params_, grid.best_score_

def main():
    print(f"Loading research dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["description"].fillna("").apply(clean_text)
    
    X = df["clean_text"]
    y_cat = df["category"]
    y_pri = df["priority"]
    
    # 80/20 Stratified Split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_pri, test_size=0.2, random_state=42, stratify=y_pri)
    
    # 1. Formal GridSearch Table
    print("\n--- Formal Hyperparameter Optimization (GridSearch) ---")
    results_gs = []
    for m in ["svm", "logreg"]:
        best_p, best_s = run_grid_search(X_train_p, y_train_p, m)
        results_gs.append({"Model": m.upper(), "Best Params": str(best_p), "Best CV Accuracy": round(best_s, 4)})
    
    df_gs = pd.DataFrame(results_gs)
    print(df_gs.to_string(index=False))

    # 2. Ensemble Modeling
    print("\n--- Ensemble (Soft Voting: SVM + LogReg) ---")
    ensemble_pipe = create_pipeline("ensemble")
    ensemble_eval = evaluate_model(ensemble_pipe, X_train_p, X_test_p, y_train_p, y_test_p, "Ensemble Priority")
    print(f"Ensemble Test Accuracy: {ensemble_eval['acc']:.4f}")

    # 3. Learning Curves
    print("\n--- Generating Learning Curves ---")
    best_svm = create_pipeline("svm")
    plot_learning_curve(best_svm, X_train_p, y_train_p, REPORTS_DIR / "lc_svm.png", "SVM Learning Curve")
    plot_learning_curve(ensemble_pipe, X_train_p, y_train_p, REPORTS_DIR / "lc_ensemble.png", "Ensemble Learning Curve")

    # 4. Feature Engineering & Model Experiments
    print("\nStarting Priority Prediction Experiments (Phase 4)...")
    pri_results = run_experiments(X_train_p, X_test_p, y_train_p, y_test_p, "Priority")
    
    # Add ensemble results to experiment table
    pri_results = pd.concat([pri_results, pd.DataFrame([{
        "Experiment": "Uni+Bi, Stop", "Model": "ENSEMBLE", "Accuracy": ensemble_eval["acc"], "CV_Mean": ensemble_eval["cv"]
    }])], ignore_index=True)
    
    pri_results.to_csv(REPORTS_DIR / "priority_experiments.csv", index=False)
    print("\nTop 5 Priority Experiment Results:")
    print(pri_results.sort_values("Accuracy", ascending=False).head(5))
    
    # 5. Final Model Selection
    print("\nTraining final research-grade models...")
    # Best Category (SVM is usually best here)
    best_cat_pipe = create_pipeline("svm")
    # Best Priority (Trying the Ensemble)
    best_pri_pipe = ensemble_pipe # Already fit in evaluate_model
    
    cat_eval = evaluate_model(best_cat_pipe, X_train_c, X_test_c, y_train_c, y_test_c, "Final Category")
    
    # 6. Error Analysis
    perform_error_analysis(best_pri_pipe, X_test_p, y_test_p)
    
    # 7. Visualizations
    plot_confusion_matrix(y_test_c, cat_eval["y_pred"], sorted(y_cat.unique()), REPORTS_DIR / "cat_cm.png", "Category CM")
    plot_confusion_matrix(y_test_p, ensemble_eval["y_pred"], ["Low", "Medium", "High", "Critical"], REPORTS_DIR / "pri_cm.png", "Priority CM")
    
    # 8. Save
    with open(MODEL_DIR / "category_pipeline.pkl", "wb") as f: pickle.dump(best_cat_pipe, f)
    with open(MODEL_DIR / "priority_pipeline.pkl", "wb") as f: pickle.dump(best_pri_pipe, f)
    print("\nResearch models and experiment reports saved.")

if __name__ == "__main__":
    main()
