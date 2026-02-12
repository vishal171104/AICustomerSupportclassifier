import sys
import os
from pathlib import Path
import pickle

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.preprocessing import clean_text

MODEL_PATH = BASE_DIR / "model" / "ticket_model.pkl"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Load model globally (simulation of single load)
try:
    models = load_model()
    category_pipeline = models["category_pipeline"]
    priority_pipeline = models["priority_pipeline"]
except FileNotFoundError:
    print("Model not found. Please run train.py first.")
    category_pipeline = None
    priority_pipeline = None

def predict_ticket(text: str):
    if category_pipeline is None or priority_pipeline is None:
        return "Error: Model not loaded", "Error: Model not loaded"
        
    text = clean_text(text)
    category = category_pipeline.predict([text])[0]
    priority = priority_pipeline.predict([text])[0]
    return category, priority

if __name__ == "__main__":
    sample = "payment failed and amount deducted"
    print(f"Input: {sample}")
    cat, pri = predict_ticket(sample)
    print(f"Predicted Category: {cat}")
    print(f"Predicted Priority: {pri}")
