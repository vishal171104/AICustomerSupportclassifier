# Automated Customer Support Ticket Classification

## Project Overview
This project assumes an office internship context. It automates the classification and prioritization of customer support tickets using Machine Learning (Logistic Regression + TF-IDF) and provides a REST API via FastAPI.

## Project Structure
- `api/`: Contains the FastAPI application (`main.py`)
- `data/`: Contains the dataset (`tickets.csv`)
- `model/`: Contains training scripts (`train.py`), prediction logic (`predict.py`), and the trained model (`ticket_model.pkl`)
- `utils/`: Utility functions (preprocessing)
- `VIT_Major_Project_Report.md`: Full project report
- `Review_2_PPT_Content.md`: Content for the presentation slides

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   Run the training pipeline to generate `model/ticket_model.pkl`.
   ```bash
   python3 model/train.py
   ```

3. **Run the API Server**
   Start the FastAPI server.
   ```bash
   uvicorn api.main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`.

4. **Access Documentation**
   - **Swagger UI**: Visit `http://127.0.0.1:8000/docs` to test endpoints interactively.
   - **ReDoc**: Visit `http://127.0.0.1:8000/redoc` for alternative documentation.

## Usage
**Predict Functionality**:
Send a POST request to `/predict` with a JSON body:
```json
{
  "description": "My payment failed and I was charged twice."
}
```

**Response**:
```json
{
  "category": "Billing",
  "priority": "High"
}
```

## Constraints & Notes
- This system uses **Logistic Regression** for efficiency and interpretability.
- Deep Learning (BERT, etc.) is intentionally avoided to meet Review-2 constraints.
- No production deployment is claimed.
