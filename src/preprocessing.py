import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import nltk

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

def remove_html_tags(text):
    """
    Remove HTML tags from the text.
    """
    return BeautifulSoup(text, "html.parser").get_text()

def tokenize_text(text):
    """
    Tokenize text into words.
    """
    return text.split()

def remove_stopwords(tokens):
    """
    Remove stopwords from the list of tokens.
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

def apply_lemmatization(tokens):
    """
    Lemmatize tokens using WordNet Lemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text):
    """
    Full preprocessing pipeline for a single text.
    """
    # Step 1: Remove HTML tags
    text = remove_html_tags(text)
    
    # Step 2: Tokenize
    tokens = tokenize_text(text)
    
    # Step 3: Remove stopwords
    tokens = remove_stopwords(tokens)
    
    # Step 4: Lemmatize
    tokens = apply_lemmatization(tokens)
    
    # Join tokens back into a single string
    return ' '.join(tokens)
