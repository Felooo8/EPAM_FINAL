from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def vectorize_tfidf(corpus, max_features=5000):
    """
    Vectorize text data using TF-IDF.
    Args:
        corpus (list of str): List of processed text data.
        max_features (int): Maximum number of features to include in the TF-IDF matrix.
    Returns:
        (sparse matrix, TfidfVectorizer): TF-IDF matrix and the trained vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def vectorize_count(corpus, max_features=5000):
    """
    Vectorize text data using Count Vectorization.
    Args:
        corpus (list of str): List of processed text data.
        max_features (int): Maximum number of features to include in the Count matrix.
    Returns:
        (sparse matrix, CountVectorizer): Count matrix and the trained vectorizer.
    """
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def vectorize_tfidf_for_inference(corpus, vectorizer_path="outputs/models/tfidf_vectorizer.pkl"):
    """
    Vectorize text for inference using a saved TF-IDF vectorizer.
    Args:
        corpus (list of str): List of processed text data.
        vectorizer_path (str): Path to the saved vectorizer.
    Returns:
        sparse matrix: TF-IDF matrix for the input corpus.
    """
    import joblib
    if not vectorizer_path:
        raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer.transform(corpus), vectorizer
