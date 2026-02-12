from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier

def create_pipeline(model_type="logreg", class_weight="balanced", ngram_range=(1, 2), max_features=5000, stop_words='english'):
    """
    Creates an sklearn Pipeline with customizable TF-IDF and Model.
    """
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words=stop_words,
        min_df=2
    )
    
    if model_type == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)
    elif model_type == "nb":
        clf = MultinomialNB()
    elif model_type == "svm":
        # Wrap LinearSVC in CalibratedClassifierCV to support predict_proba()
        svc = LinearSVC(class_weight=class_weight, random_state=42, dual=False)
        clf = CalibratedClassifierCV(svc)
    elif model_type == "ensemble":
        # Combine LogReg and Calibrated SVM via Soft Voting
        svm = CalibratedClassifierCV(LinearSVC(class_weight=class_weight, random_state=42, dual=False))
        lr = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)
        clf = VotingClassifier(
            estimators=[('svm', svm), ('lr', lr)],
            voting='soft'
        )
    elif model_type == "baseline":
        clf = DummyClassifier(strategy="stratified", random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return Pipeline([
        ("tfidf", tfidf),
        ("clf", clf)
    ])
