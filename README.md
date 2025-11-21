#  Email Spam Detection using Machine Learning

This project implements a Machine Learning model that classifies email or SMS messages as **Spam** or **Ham (Not Spam)**. It uses an effective text-processing pipeline along with **TF-IDF vectorization** and a **Multinomial Naive Bayes** classifier to detect unwanted or malicious messages with high accuracy.

The system performs thorough text preprocessing including cleaning, tokenization, punctuation removal, stopword elimination, and vectorization. After processing, messages are transformed into numerical feature vectors suitable for machine learning. The model is trained, tested, evaluated, and saved for later deployment.

---

## Features

- Spam vs Ham classification  
- Complete text preprocessing pipeline  
- Feature engineering (word count, character count, sentence count)  
- TF-IDF vectorization  
- Multinomial Naive Bayes classifier  
- Visualizations for EDA (histograms, confusion matrix)  
- Saved model & vectorizer for deployment  
- Jupyter Notebook with step-by-step workflow

---

## Dataset

The dataset contains two columns:

- **v1** → Label (`spam` or `ham`)  
- **v2** → Message text

Additional `Unnamed` columns are removed during preprocessing.

---

## Installation
pip install -r requirements.txt

---
## # 🔹 Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 🔹 Load Dataset from Drive
import pandas as pd
data_path = '/content/drive/MyDrive/spam_dataset/spam.csv'
df = pd.read_csv(data_path, encoding='latin-1')

# 🔹 Keep only first two columns and rename
df = df.iloc[:, :2]
df.columns = ['label', 'message']

# 🔹 Basic cleaning
df['label'] = df['label'].astype(str).str.strip().str.lower()
df['message'] = df['message'].fillna('').astype(str)

# 🔹 Download NLTK tokenizers
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# 🔹 Add simple text features
from nltk import word_tokenize, sent_tokenize

df['num_characters'] = df['message'].apply(len)
df['num_words'] = df['message'].apply(lambda x: len(word_tokenize(x)))
df['num_sentences'] = df['message'].apply(lambda x: len(sent_tokenize(x)))

# 🔹 Clean text
import re
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if t not in stops and len(t) > 1]
    return ' '.join(tokens)

df['clean_msg'] = df['message'].apply(clean_text)

# 🔹 TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_msg']).toarray()
y = (df['label'] == 'spam').astype(int)

# 🔹 Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 Train Model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 🔹 Predictions & Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔹 Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 🔹 Save Model & Vectorizer
import joblib
joblib.dump(clf, '/content/drive/MyDrive/spam_dataset/spam_classifier_nb.joblib')
joblib.dump(tfidf, '/content/drive/MyDrive/spam_dataset/tfidf_vectorizer.joblib')

print("\nModel and vectorizer saved successfully!")
