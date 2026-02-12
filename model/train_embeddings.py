import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Paths
DATA_PATH = BASE_DIR / "data" / "tickets.csv"
REPORTS_DIR = BASE_DIR / "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def main():
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 2. Encode Labels
    priority_order = ["Low", "Medium", "High", "Critical"]
    
    # 3. Generate Embeddings
    print("Loading MiniLM model and generating sentence embeddings (CPU)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # We use the raw descriptions for semantic embeddings
    embeddings = model.encode(df['description'].tolist(), show_progress_bar=True)
    
    X = embeddings
    y = df['priority']
    
    # 4. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Train Linear SVM (with probability=True as requested for research depth)
    print("Training Linear SVM on embeddings...")
    clf = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # 6. Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=priority_order, zero_division=0)
    
    print("\n=== MiniLM + SVM Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Classification Report:")
    print(report)
    
    # 7. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=priority_order)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=priority_order, yticklabels=priority_order)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('MiniLM + SVM Confusion Matrix')
    plt.savefig(REPORTS_DIR / "embedding_svm_cm.png")
    plt.close()
    print(f"Confusion matrix saved to {REPORTS_DIR / 'embedding_svm_cm.png'}")
    
    # 8. Comparison Summary
    # Baseline data from previous experiments (hardcoded for the summary as requested)
    comparison_data = [
        {"Model": "NB (TF-IDF)", "Accuracy": 0.5644},
        {"Model": "LogReg (TF-IDF)", "Accuracy": 0.6733},
        {"Model": "SVM (TF-IDF)", "Accuracy": 0.6733},
        {"Model": "DistilBERT", "Accuracy": 0.6337},
        {"Model": "MiniLM + SVM", "Accuracy": round(acc, 4)}
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n=== Model Comparison Summary ===")
    print(comparison_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
