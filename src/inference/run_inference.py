import argparse
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report

import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from data_loader import load_dataset
from preprocessing import preprocess_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to the input test dataset file")
    args = parser.parse_args()

    input_file = args.input_file

    output_dir = "outputs/predictions/"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.getenv("MODEL_PATH")
    vectorizer_path = os.getenv("VECTORIZER_PATH")

    print(f"Loading test dataset from {input_file}...")
    data = load_dataset(input_file)

    print("Preprocessing text...")
    data["processed_review"] = data["review"].apply(preprocess_text)

    print(f"Loading model from {model_path} and vectorizer from {vectorizer_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Ensure it exists.")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}. Ensure it exists.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    print("Vectorizing text...")
    X_test = vectorizer.transform(data["processed_review"])

    print("Running inference...")
    predictions = model.predict(X_test)
    data["predicted_sentiment"] = predictions
    data["predicted_sentiment"] = data["predicted_sentiment"].apply(lambda x: "positive" if x == 1 else "negative")

    output_file = os.path.join(output_dir, "predictions.csv")
    data[["review", "predicted_sentiment"]].to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    if "sentiment" in data.columns:
        print("Evaluating predictions...")
        y_true = data["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
        y_pred = predictions
        report = classification_report(y_true, y_pred, target_names=["negative", "positive"], output_dict=True)
        
        metrics_file = os.path.join(output_dir, "metrics.txt")
        with open(metrics_file, "w") as f:
            for label, metrics in report.items():
                if isinstance(metrics, dict):  # Metrics for each class
                    f.write(f"{label}:\n")
                    for metric_name, value in metrics.items():
                        f.write(f"  {metric_name}: {value:.4f}\n")
                else:  # Overall metrics
                    f.write(f"{label}: {metrics:.4f}\n")
        print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()
