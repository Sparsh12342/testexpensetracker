import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
from collections import Counter
from sklearn.metrics import silhouette_score
import sys

# ✅ Accept filename from command-line
FILENAME = sys.argv[1] if len(sys.argv) > 1 else "stmt.csv"
MIN_SUBCLUSTERS = 2
MAX_SUBCLUSTERS = 5

bert_model_dir = "./bert_expense_classifier"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_dir)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_dir)
bert_model.eval()

label_encoder = joblib.load("label_encoder.joblib")

def classify_with_bert(descriptions):
    inputs = bert_tokenizer(list(descriptions), padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    categories = label_encoder.inverse_transform(preds)
    return categories

def choose_optimal_clusters(features, min_k=MIN_SUBCLUSTERS, max_k=MAX_SUBCLUSTERS):
    best_k = min_k
    best_score = -1
    for k in range(min_k, min(max_k + 1, features.shape[0])):
        if k <= 1:
            continue
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def cluster_subgroups_within_category(category_df, category_name):
    descriptions = category_df["Description"].tolist()
    if len(descriptions) < MIN_SUBCLUSTERS:
        return pd.Series(["Only_Cluster"] * len(descriptions)), None

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    optimal_k = min(MAX_SUBCLUSTERS, len(descriptions))  # optionally: choose_optimal_clusters(tfidf_matrix)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    return pd.Series([f"{category_name}_Sub{label}" for label in cluster_labels]), kmeans

def summarize_by_category(df):
    print("\n📊 Category Summary:")
    summary = []
    for category, group in df.groupby("BERT_Category"):
        count = group.shape[0]
        total = group["Amount"].sum()
        withdrawals = group[group["Amount"] < 0]["Amount"].sum()
        deposits = group[group["Amount"] >= 0]["Amount"].sum()

        print(f"🗂️ {category}")
        print(f"   Transactions: {count}")
        print(f"   Total Amount: ${total:.2f}")
        print(f"   Withdrawals: ${withdrawals:.2f}")
        print(f"   Deposits: ${deposits:.2f}")

        summary.append({
            "Category": category,
            "Transaction Count": count,
            "Total Amount": total,
            "Withdrawals": withdrawals,
            "Deposits": deposits
        })

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values("Total Amount", ascending=False)
    summary_df.to_csv("stmt_bert_category_summary.csv", index=False)
    print("\n✅ Saved category summary to stmt_bert_category_summary.csv")

def classify_and_subcluster():
    df = pd.read_csv(FILENAME, on_bad_lines='skip')
    df["Description"] = df["Description"].fillna("")

    if 'Amount' not in df.columns:
        raise ValueError("CSV must contain an 'Amount' numeric column.")

    df["Amount"] = pd.to_numeric(df["Amount"], errors='coerce').fillna(0)

    print("🔍 Running BERT classification for main categories...")
    df["BERT_Category"] = classify_with_bert(df["Description"])

    print("\n✅ Predicted Main Categories:")
    for cat in sorted(df["BERT_Category"].unique()):
        print(f" - {cat}")

    print("\n🔍 Running subclustering within each BERT category...")
    all_subcluster_labels = []
    for category in df["BERT_Category"].unique():
        category_df = df[df["BERT_Category"] == category]
        subcluster_labels, _ = cluster_subgroups_within_category(category_df, category)
        all_subcluster_labels.extend(subcluster_labels)

    df["Subcluster_Label"] = all_subcluster_labels

    summarize_by_category(df)
    df.to_csv("stmt_clustered_labeled.csv", index=False)

    print("\n✅ Saved transactions with BERT categories + subclusters to stmt_clustered_labeled.csv")

if __name__ == "__main__":
    classify_and_subcluster()
