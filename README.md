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
