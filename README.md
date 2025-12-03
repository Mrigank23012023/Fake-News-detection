#📰 Fake News Detection System

AI-powered model to classify news articles as Real or Fake

This project uses Natural Language Processing (NLP) and Machine Learning to detect fake news from textual content. It processes raw news data, cleans the text, extracts features, and classifies news with high accuracy using advanced ML models.

🚀 Features
✅ 1. Fake vs Real News Classification

Trained on a labeled Fake News dataset

Uses ML/NLP pipeline for accurate predictions

✅ 2. Powerful Text Processing

Text cleaning (lowercase, punctuation removal, stopwords removal)

Lemmatization

TF-IDF vectorization

✅ 3. Machine Learning Models

Experimented with multiple models:

Logistic Regression

Random Forest

SVM

Naive Bayes

XGBoost (optional)

Best-performing model saved using Pickle.

✅ 4. Interactive UI (optional)

If you used Streamlit / Flask:

Paste news → Get prediction

Displays probability score

🧠 Workflow
Raw Text → Preprocessing → TF-IDF Vectorizer → ML Model → Prediction (Real/Fake)

🏗️ Tech Stack

Languages: Python
NLP: NLTK, Scikit-learn
ML Models: LR, NB, RF, SVM, XGBoost
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
UI (if added): Streamlit


📊 Model Performance

(Add your actual metrics later)

Example format:

Accuracy: 96%

Precision: 95%

Recall: 94%

F1 Score: 94%

🔧 How to Run

1️⃣ Clone the Repo
git clone https://github.com/mrigankmathur/FakeNewsDetection.git
cd FakeNewsDetection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run App (if UI added)
streamlit run app.py

📌 Future Enhancements

Use LSTMs / Transformers for higher accuracy

Include multimodal analysis (text + image)

Deploy as a full web app

Add explainability (LIME/SHAP)

👤 Author

Mrigank Mathur
AI/ML Developer | NLP Enthusiast
