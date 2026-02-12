import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

def evaluate_model(pipeline, X_train, X_test, y_train, y_test, title="Evaluation"):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    mean_cv = np.mean(cv_scores)
    
    return {
        "acc": acc,
        "cv": mean_cv,
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "y_pred": y_pred
    }

def run_experiments(X_train, X_test, y_train, y_test, task_name="Task"):
    from .pipelines import create_pipeline
    
    configs = [
        {"name": "Uni, NoStop", "ngram": (1,1), "stop": None},
        {"name": "Uni, Stop", "ngram": (1,1), "stop": 'english'},
        {"name": "Uni+Bi, NoStop", "ngram": (1,2), "stop": None},
        {"name": "Uni+Bi, Stop", "ngram": (1,2), "stop": 'english'},
    ]
    
    results = []
    for config in configs:
        for model in ["nb", "logreg", "svm"]:
            pipe = create_pipeline(
                model_type=model, 
                ngram_range=config["ngram"], 
                stop_words=config["stop"]
            )
            eval_res = evaluate_model(pipe, X_train, X_test, y_train, y_test)
            results.append({
                "Experiment": config["name"],
                "Model": model.upper(),
                "Accuracy": eval_res["acc"],
                "CV_Mean": eval_res["cv"]
            })
            
    return pd.DataFrame(results)

def perform_error_analysis(pipeline, X_test, y_test, n_samples=5):
    """
    Identifies misclassified samples for qualitative analysis.
    """
    y_pred = pipeline.predict(X_test)
    X_test_series = pd.Series(X_test) if not isinstance(X_test, pd.Series) else X_test
    
    df_error = pd.DataFrame({
        'Text': X_test_series.values,
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    
    errors = df_error[df_error['Actual'] != df_error['Predicted']]
    
    print(f"\n--- Error Analysis (Found {len(errors)} misclassifications) ---")
    if not errors.empty:
        print(errors.head(n_samples).to_string(index=False))
    
    return errors

from sklearn.model_selection import learning_curve

def plot_learning_curve(pipeline, X, y, filename, title="Learning Curve"):
    """
    Plots learning curves for the given pipeline and data.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    
    plt.xlabel("Training examples")
    plt.ylabel("Score (Accuracy)")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Learning curve saved to {filename}")

def plot_confusion_matrix(y_true, y_pred, labels, filename, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")
