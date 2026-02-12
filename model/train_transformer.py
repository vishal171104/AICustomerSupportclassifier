import os
import sys
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "tickets.csv"
REPORTS_DIR = BASE_DIR / "reports"
CLASSICAL_RESULTS_PATH = REPORTS_DIR / "priority_experiments.csv"

# Model constants
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3

def load_and_prep_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Priority mapping
    priority_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    df['label'] = df['priority'].map(priority_map)
    
    # Stratified 80/20 split
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['priority']
    )
    
    return train_df, test_df, list(priority_map.keys())

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1": f1}

def main():
    train_df, test_df, target_names = load_and_prep_data()
    
    print(f"Initializing {MODEL_NAME} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples["description"], truncation=True, padding=False, max_length=MAX_LENGTH)

    # Convert to HuggingFace Datasets
    train_ds = Dataset.from_pandas(train_df[['description', 'label']])
    test_ds = Dataset.from_pandas(test_df[['description', 'label']])
    
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(target_names)
    )

    # CPU training args
    training_args = TrainingArguments(
        output_dir="./results_transformer",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="no",
        logging_dir="./logs",
        logging_steps=10,
        push_to_hub=False,
        no_cuda=True, # Force CPU
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning DistilBERT on CPU (this may take a few minutes)...")
    trainer.train()

    # Evaluation
    print("\nEvaluating fine-tuned model...")
    eval_results = trainer.evaluate()
    
    preds_output = trainer.predict(tokenized_test)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = test_df['label'].values
    
    # Results
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    print("\n=== DistilBERT Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Classification Report:")
    print(report)

    # Comparison Table
    if CLASSICAL_RESULTS_PATH.exists():
        classical_df = pd.read_csv(CLASSICAL_RESULTS_PATH)
        # Get best results for each model from experiments
        # For simplicity, we'll take the first occurrence of each model or highest accuracy
        latest_results = classical_df.loc[classical_df.groupby('Model')['Accuracy'].idxmax()]
        
        comparison_data = []
        for _, row in latest_results.iterrows():
            comparison_data.append({"Model": row["Model"], "Accuracy": round(row["Accuracy"], 4)})
        
        comparison_data.append({"Model": "DistilBERT", "Accuracy": round(acc, 4)})
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n=== Classical ML vs Transformer Comparison ===")
        print(comparison_df.to_string(index=False))
    else:
        print("\nClassical results not found. Run model/train.py first.")

if __name__ == "__main__":
    main()
