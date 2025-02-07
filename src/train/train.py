import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys, os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from data_loader import load_dataset
from preprocessing import preprocess_text
from feature_engineering import vectorize_tfidf

MODEL_PATH = "outputs/models/best_model.pkl"
VECTORIZER_PATH = "outputs/models/tfidf_vectorizer.pkl"
METRICS_PATH = "outputs/metrics.txt"

def main():
    print("Loading dataset...")
    data = load_dataset("train.csv")
    
    print("Preprocessing text...")
    data["processed_review"] = data["review"].apply(preprocess_text)
    
    print("Vectorizing text...")
    X, vectorizer = vectorize_tfidf(data["processed_review"])
    y = data["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")
    
    with open(METRICS_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\\n")
        f.write(f"Precision: {precision:.4f}\\n")
        f.write(f"Recall: {recall:.4f}\\n")
        f.write(f"F1-Score: {f1:.4f}\\n")
    print(f"Metrics saved to {METRICS_PATH}")
    
if __name__ == "__main__":
    main()
