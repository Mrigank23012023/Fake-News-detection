import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re
import string

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Load dataset from CSV
def load_data(filepath):
    """
    Load data from CSV file
    Expected columns: 'text' (news article) and 'label' (0=fake, 1=real)
    """
    df = pd.read_csv(filepath)
    return df

# Train with Logistic Regression
def train_model(data_path):
    """
    Train the fake news detection model with Logistic Regression
    
    Parameters:
    - data_path: Path to CSV file with columns 'text' and 'label'
    """
    
    print("Loading data...")
    df = load_data(data_path)
    
    print(f"Dataset size: {len(df)} articles")
    print(f"Fake news: {sum(df['label'] == 0)}, Real news: {sum(df['label'] == 1)}")
    
    # Preprocess text
    print("\nPreprocessing text...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.strip() != '']
    
    # Split data
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create TF-IDF vectorizer
    print("\nCreating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate model
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and vectorizer
    print("\nSaving model and vectorizer...")
    joblib.dump(model, 'news_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    print("\n✅ Model and vectorizer saved successfully!")
    print("   - news_model.pkl")
    print("   - vectorizer.pkl")
    
    return model, vectorizer

# Train with Random Forest
def train_random_forest(data_path):
    """Train using Random Forest classifier"""
    
    print("Loading data...")
    df = load_data(data_path)
    
    print(f"Dataset size: {len(df)} articles")
    print(f"Fake news: {sum(df['label'] == 0)}, Real news: {sum(df['label'] == 1)}")
    
    # Preprocess
    print("\nPreprocessing text...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    df = df[df['cleaned_text'].str.strip() != '']
    
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Vectorize
    print("\nCreating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Random Forest
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save
    print("\nSaving model and vectorizer...")
    joblib.dump(model, 'news_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    print("\n✅ Random Forest model saved!")
    print("   - news_model.pkl")
    print("   - vectorizer.pkl")
    
    return model, vectorizer

if __name__ == "__main__":
    # Provide your CSV file path here
    DATA_PATH = 'news_data.csv'  # Change this to your actual file path
    
    # Option 1: Train with Logistic Regression (faster)
    model, vectorizer = train_model(DATA_PATH)
    
    # Option 2: Train with Random Forest (more powerful)
    # model, vectorizer = train_random_forest(DATA_PATH)
    
    print("\n🎉 Training complete!")