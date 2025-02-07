# Binary Sentiment Classification
# DS PART

## **1. Conclusions from EDA**
- The dataset contains 50,000 movie reviews evenly distributed between **positive** and **negative** sentiments.
- Reviews vary significantly in length, with most containing fewer than 200 words, as shown in the text length distribution.
- Word clouds reveal that frequently occurring words in positive reviews include "great," "character," and "film," while negative reviews highlight words like "bad," "waste," and "boring."
- HTML tags like `<br>` were present in the reviews and were removed during preprocessing.

---

## **2. Description of Feature Engineering**
- **Preprocessing Steps**:
  - **HTML tag removal**: Cleaned reviews of `<br>` tags and other HTML artifacts.
  - **Tokenization**: Split text into individual words.
  - **Stop-word removal**: Filtered out common stopwords like "the," "and," and "is."
  - **Lemmatization**: Converted words to their base form (e.g., "running" → "run").
- **Vectorization Techniques**:
  - **TF-IDF Vectorization**: Assigned weights to words based on importance and rarity.
  - **Count Vectorization**: Represented reviews as sparse matrices based on raw word frequencies.

---

## **3. Reasonings on Model Selection**
- Three models were trained and evaluated:
  - **Logistic Regression**: Achieved the highest accuracy (88.91%) and balanced precision, recall, and F1-score.
  - **Naïve Bayes**: Performed the lowest with an accuracy of 85.66% due to its feature independence assumption.
  - **Support Vector Machine (SVM)**: Performed slightly below Logistic Regression with an accuracy of 88.42%.
- **Conclusion**: Logistic Regression was selected as the best model due to its slightly superior performance and simplicity.

---

## **4. Overall Performance Evaluation**
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.8891   | 0.8799    | 0.9012 | 0.8905   |
| Naïve Bayes          | 0.8566   | 0.8523    | 0.8628 | 0.8575   |
| SVM                  | 0.8842   | 0.8769    | 0.8940 | 0.8854   |

- **Logistic Regression** meets the target accuracy of **≥85%** and demonstrates robust performance across all metrics.

---

## **5. Potential Business Applications and Value for Business**
- **Customer Feedback Analysis**:
  - Businesses can analyze customer reviews to identify trends in satisfaction or dissatisfaction.
- **Brand Reputation Monitoring**:
  - Sentiment analysis helps monitor how products or services are perceived in the market.
- **Product Reviews on E-Commerce**:
  - Automatically classify reviews as positive or negative to help customers make informed purchasing decisions.
- **Marketing Campaigns**:
  - Analyze public sentiment toward campaigns or new product launches to adjust strategies effectively.

---

## **6. Future Improvements**
- Experiment with advanced deep learning models like **LSTM** or **Transformer-based models (e.g., BERT)**.
- Perform hyperparameter tuning to further improve model performance.
- Test the model's generalizability on other datasets or domains.

## MLE PART

### 1. Overview
The Machine Learning Engineering (MLE) part involves containerizing the training and inference processes to ensure reproducibility and ease of deployment. Docker is used to build images and run containers for both phases.

---

### 2. How to Run the Project

#### **Training**
1. Build the Docker image for training:
   ```bash
   docker build -f src/train/Dockerfile -t sentiment-train .
   ```
2. Run the container to train the model:
   ```bash
   docker run -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data sentiment-train
   ```
3. After execution, the trained model (`best_model.pkl`) and vectorizer (`tfidf_vectorizer.pkl`) will be saved in the `outputs/models/` directory.

#### **Inference**
1. Build the Docker image for inference:
   ```bash
   docker build -f src/inference/Dockerfile --build-arg model_name=best_model.pkl --build-arg vectorizer_name=tfidf_vectorizer.pkl -t sentiment-inference .
   ```
2. Run the container to perform inference on the test dataset:
   ```bash
   docker run -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data sentiment-inference --input_file /app/data/raw/test.csv
   ```
3. After execution, the following outputs will be saved in the `outputs/predictions/` directory:
   - **`predictions.csv`**: Contains the reviews and their predicted sentiments.
   - **`metrics.txt`**: Contains the performance metrics of the model on the test dataset.

---

### 3. Outputs
The following outputs are generated after running the training and inference containers:
- **Models (`outputs/models/`)**:
  - `best_model.pkl`: The trained Logistic Regression model.
  - `tfidf_vectorizer.pkl`: The vectorizer used for text preprocessing.
- **Predictions (`outputs/predictions/`)**:
  - `predictions.csv`: Predicted sentiments for the test dataset.
  - `metrics.txt`: Evaluation metrics including precision, recall, and F1-score.

---

### 4. Reproducibility
The outputs (serialized models, predictions, and metrics) are saved to the mounted `outputs/` directory, ensuring reproducibility across different runs and environments.

---

### 5. Quickstart Guide
#### Train and Inference Locally
- **Training**:
  ```bash
  python src/train/train.py
  ```
- **Inference**:
  ```bash
  python src/inference/run_inference.py --input_file data/raw/test.csv
  ```

#### Docker-Based Workflow
- **Training**:
  ```bash
  docker build -f src/train/Dockerfile -t sentiment-train .
  docker run -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data sentiment-train
  ```
- **Inference**:
  ```bash
  docker build -f src/inference/Dockerfile --build-arg model_name=best_model.pkl --build-arg vectorizer_name=tfidf_vectorizer.pkl -t sentiment-inference .
  docker run -v ${PWD}/outputs:/app/outputs -v ${PWD}/data:/app/data sentiment-inference --input_file /app/data/raw/test.csv
  ```

---

### 6. Final Metrics
The performance metrics for the test dataset are as follows:

```plaintext
negative:
  precision: 0.8982
  recall: 0.8806
  f1-score: 0.8893
  support: 5000.0000
positive:
  precision: 0.8829
  recall: 0.9002
  f1-score: 0.8915
  support: 5000.0000
accuracy: 0.8904
macro avg:
  precision: 0.8906
  recall: 0.8904
  f1-score: 0.8904
  support: 10000.0000
weighted avg:
  precision: 0.8906
  recall: 0.8904
  f1-score: 0.8904
  support: 10000.0000
```

---

### 7. Notes
- Ensure Docker is installed and running on your system before executing the commands.
- Adjust paths (`${PWD}` or `/app/...`) as necessary for your environment.

---
