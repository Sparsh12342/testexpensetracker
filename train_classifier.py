# train_classifier.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load and clean data
df = pd.read_csv("descriptions.csv")
df = df.dropna(subset=["Description", "Category"])
X = df["Description"]
y = df["Category"]

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=1000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "transaction_classifier.joblib")
print("\n✅ Model saved to transaction_classifier.joblib")
