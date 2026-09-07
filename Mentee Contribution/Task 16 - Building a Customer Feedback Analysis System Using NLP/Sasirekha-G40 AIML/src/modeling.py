from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .preprocessing import clean_text

RANDOM_STATE = 42

def vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)

def make_classifier():
    return Pipeline([
        ("tfidf", vectorizer()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])

def train_classifier(texts, labels):
    model = make_classifier()
    model.fit([clean_text(x) for x in texts], labels)
    return model

def split_data(df):
    return train_test_split(df, test_size=0.20, random_state=RANDOM_STATE, stratify=df["sentiment"])

def save_object(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_object(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run: python train.py")
    return joblib.load(path)

def evaluate(model, x_test, y_test):
    pred = model.predict(x_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="weighted", zero_division=0), classification_report(y_test, pred, zero_division=0)
